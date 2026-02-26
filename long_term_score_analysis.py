#!/usr/bin/env python3
"""
Long-Term Score Analysis
========================
Find optimal factor weights for 1-year and 2-year returns.
The current Compass Score is optimized for 3-month returns.
This analysis explores whether different factors predict longer-term performance.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Horizons to test
HORIZONS = {
    '3-Month': 63,   # Current Compass Score horizon (baseline)
    '1-Year': 252,   # Long-term
    '2-Year': 504,   # Very long-term
}

# In-sample / Out-of-sample split
IS_END = '2020-01-01'
SAMPLE_SIZE = 2000  # Sample for speed

# Factors to test (same as Compass Score plus additional candidates)
FACTOR_COLS = [
    # Current Compass Score factors
    'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d', 'asset_growth',
    # Additional candidates for long-term
    'revenue_growth', 'pb_ratio', 'momentum_12m',
]

print("=" * 70, flush=True)
print("LONG-TERM SCORE ANALYSIS", flush=True)
print("Finding optimal factor weights for 1-year and 2-year returns", flush=True)
print("=" * 70, flush=True)
print(f"Started: {datetime.now()}", flush=True)
sys.stdout.flush()

# Load data
print("\nLoading data...", flush=True)
conn = sqlite3.connect(BACKTEST_DB)

# Get prices
prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 1 AND date >= '2015-01-01'
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])
print(f"  {len(prices):,} price records", flush=True)

# Get fundamentals
fund = pd.read_sql_query("""
    SELECT
        i.symbol, i.date,
        i.net_income, b.total_assets, c.operating_cash_flow,
        c.free_cash_flow, i.gross_profit, i.revenue,
        b.total_equity, m.market_cap
    FROM historical_income_statements i
    JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
    JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    WHERE i.date >= '2015-01-01'
""", conn)
fund['date'] = pd.to_datetime(fund['date'])
print(f"  {len(fund):,} fundamental records", flush=True)
conn.close()

# Compute fundamental factors
print("\nComputing factors...", flush=True)
fund['roa'] = fund['net_income'] / fund['total_assets']
fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
fund['pb_ratio'] = fund['market_cap'] / fund['total_equity']

# Asset growth and revenue growth
fund = fund.sort_values(['symbol', 'date'])
fund['assets_lag'] = fund.groupby('symbol')['total_assets'].shift(4)
fund['asset_growth'] = (fund['total_assets'] / fund['assets_lag']) - 1
fund['revenue_lag'] = fund.groupby('symbol')['revenue'].shift(4)
fund['revenue_growth'] = (fund['revenue'] / fund['revenue_lag']) - 1

# Clean infinities
for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'revenue_growth', 'pb_ratio']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

fund = fund.dropna(subset=['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth'])
print(f"  {len(fund):,} records with fundamental factors", flush=True)

# Sample symbols
symbols = fund['symbol'].unique()
np.random.seed(42)
if len(symbols) > SAMPLE_SIZE:
    symbols = np.random.choice(symbols, size=SAMPLE_SIZE, replace=False)
print(f"  Sampled {len(symbols)} symbols", flush=True)

# Filter to sampled symbols
fund = fund[fund['symbol'].isin(symbols)]
prices = prices[prices['symbol'].isin(symbols)]

# Compute volatility and momentum
print("\nComputing volatility and momentum...", flush=True)
sys.stdout.flush()

vol_mom_results = []
for i, symbol in enumerate(symbols):
    if i % 500 == 0:
        print(f"  {i}/{len(symbols)}...", flush=True)
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    if len(sym_prices) < 280:
        continue

    sym_prices['return'] = sym_prices['close'].pct_change()
    sym_prices['vol_60d'] = sym_prices['return'].rolling(60, min_periods=40).std() * np.sqrt(252)

    # 12-month momentum
    sym_prices['momentum_12m'] = sym_prices['close'].pct_change(252)

    for _, row in sym_prices[sym_prices['vol_60d'].notna() & sym_prices['momentum_12m'].notna()].iterrows():
        vol_mom_results.append({
            'symbol': symbol,
            'date': row['date'],
            'vol_60d': row['vol_60d'],
            'momentum_12m': row['momentum_12m'],
        })

vol_mom_df = pd.DataFrame(vol_mom_results)
print(f"  {len(vol_mom_df):,} volatility/momentum records", flush=True)

# Compute forward returns for all horizons
print("\nComputing forward returns...", flush=True)
sys.stdout.flush()

max_horizon = max(HORIZONS.values())
fwd_results = []

for i, symbol in enumerate(symbols):
    if i % 500 == 0:
        print(f"  {i}/{len(symbols)}...", flush=True)
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    n = len(sym_prices)

    if n < max_horizon + 20:
        continue

    close = sym_prices['close'].values
    dates = sym_prices['date'].values

    # Sample every 21 days (monthly)
    for j in range(0, n - max_horizon, 21):
        entry = close[j]
        if entry <= 0:
            continue

        row = {
            'symbol': symbol,
            'date': pd.Timestamp(dates[j]),
        }

        for horizon_name, days in HORIZONS.items():
            if j + days < n:
                exit_price = close[j + days]
                fwd_ret = ((exit_price / entry) - 1) * 100
                fwd_ret = np.clip(fwd_ret, -100, 500)
                row[f'fwd_{horizon_name}'] = fwd_ret
            else:
                row[f'fwd_{horizon_name}'] = np.nan

        fwd_results.append(row)

fwd_df = pd.DataFrame(fwd_results)
print(f"  {len(fwd_df):,} forward return records", flush=True)

# Merge data
print("\nMerging data...", flush=True)
sys.stdout.flush()

# Create monthly periods for merging
fund['month'] = fund['date'].dt.to_period('M')
vol_mom_df['month'] = vol_mom_df['date'].dt.to_period('M')
fwd_df['month'] = fwd_df['date'].dt.to_period('M')

# Get latest fundamental per symbol-month
fund_monthly = fund.sort_values('date').groupby(['symbol', 'month']).last().reset_index()

# Merge
df = vol_mom_df.merge(
    fund_monthly[['symbol', 'month', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                   'asset_growth', 'revenue_growth', 'pb_ratio']],
    on=['symbol', 'month'], how='inner'
)
df = df.merge(fwd_df[['symbol', 'month'] + [f'fwd_{h}' for h in HORIZONS.keys()]],
              on=['symbol', 'month'], how='inner')

print(f"  {len(df):,} merged records", flush=True)

# Drop rows with missing forward returns
df = df.dropna(subset=[f'fwd_{h}' for h in HORIZONS.keys()])
print(f"  {len(df):,} records with all forward returns", flush=True)

# Split IS/OOS
is_df = df[df['date'] < IS_END].copy()
oos_df = df[df['date'] >= IS_END].copy()
print(f"  IS: {len(is_df):,}, OOS: {len(oos_df):,}", flush=True)

if len(is_df) < 1000 or len(oos_df) < 500:
    print("ERROR: Not enough data. Exiting.")
    sys.exit(1)

# Available factors
available_factors = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d',
                     'asset_growth', 'revenue_growth', 'momentum_12m']

# Check for pb_ratio
if 'pb_ratio' in df.columns and df['pb_ratio'].notna().sum() > 1000:
    available_factors.append('pb_ratio')

print(f"\n  Available factors: {available_factors}", flush=True)

# Compute z-scores using IS statistics only
print("\nComputing z-scores (using IS stats)...", flush=True)

z_stats = {}
for col in available_factors:
    # Winsorize using IS bounds
    is_lower = is_df[col].quantile(0.01)
    is_upper = is_df[col].quantile(0.99)

    is_df[col] = is_df[col].clip(is_lower, is_upper)
    oos_df[col] = oos_df[col].clip(is_lower, is_upper)

    # Z-score using IS mean/std
    is_mean = is_df[col].mean()
    is_std = is_df[col].std()

    if is_std > 0:
        is_df[f'{col}_z'] = ((is_df[col] - is_mean) / is_std).clip(-3, 3)
        oos_df[f'{col}_z'] = ((oos_df[col] - is_mean) / is_std).clip(-3, 3)
        z_stats[col] = {'mean': is_mean, 'std': is_std}
    else:
        is_df[f'{col}_z'] = 0
        oos_df[f'{col}_z'] = 0

# ==========================================
# UNIVARIATE ANALYSIS FOR EACH HORIZON
# ==========================================

print("\n" + "=" * 70)
print("UNIVARIATE FACTOR ANALYSIS BY HORIZON")
print("=" * 70)

for horizon_name in HORIZONS.keys():
    return_col = f'fwd_{horizon_name}'

    print(f"\n--- {horizon_name} Horizon ---")
    print(f"{'Factor':<18} {'IS Spread':>12} {'OOS Spread':>12} {'Direction':>12}")
    print("-" * 60)

    for factor in available_factors:
        z_col = f'{factor}_z'

        # IS quintiles
        try:
            is_df[f'{factor}_q'] = pd.qcut(is_df[z_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        is_q5 = is_df[is_df[f'{factor}_q'] == 'Q5'][return_col].mean()
        is_q1 = is_df[is_df[f'{factor}_q'] == 'Q1'][return_col].mean()
        is_spread = is_q5 - is_q1

        # OOS quintiles using IS bounds
        try:
            is_bounds = is_df[z_col].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
            oos_df[f'{factor}_q'] = pd.cut(oos_df[z_col], bins=is_bounds, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], include_lowest=True)
        except:
            continue

        oos_q5 = oos_df[oos_df[f'{factor}_q'] == 'Q5'][return_col].mean()
        oos_q1 = oos_df[oos_df[f'{factor}_q'] == 'Q1'][return_col].mean()
        oos_spread = oos_q5 - oos_q1

        direction = "+" if oos_spread > 0 else "-"
        print(f"{factor:<18} {is_spread:>+11.2f}% {oos_spread:>+11.2f}% {direction:>12}")

# ==========================================
# MULTIVARIATE REGRESSION FOR EACH HORIZON
# ==========================================

print("\n" + "=" * 70)
print("MULTIVARIATE REGRESSION BY HORIZON")
print("=" * 70)

from sklearn.linear_model import LinearRegression

optimal_weights = {}

for horizon_name in HORIZONS.keys():
    return_col = f'fwd_{horizon_name}'

    print(f"\n--- {horizon_name} Horizon ---")

    # Prepare data
    z_cols = [f'{f}_z' for f in available_factors]

    is_clean = is_df.dropna(subset=z_cols + [return_col])

    X = is_clean[z_cols].values
    y = is_clean[return_col].values

    # Fit regression
    reg = LinearRegression()
    reg.fit(X, y)

    # Get coefficients
    coefs = dict(zip(available_factors, reg.coef_))

    print(f"{'Factor':<18} {'Coefficient':>12} {'|Coef|':>10}")
    print("-" * 45)

    for factor in sorted(coefs.keys(), key=lambda x: abs(coefs[x]), reverse=True):
        print(f"{factor:<18} {coefs[factor]:>+11.2f} {abs(coefs[factor]):>10.2f}")

    print(f"\nIntercept: {reg.intercept_:.2f}")

    # Compute optimal weights (normalized absolute coefficients)
    abs_sum = sum(abs(c) for c in coefs.values())
    weights = {f: coefs[f] / abs_sum for f in coefs.keys()}
    optimal_weights[horizon_name] = weights

    print(f"\nOptimal Weights (normalized):")
    for factor in sorted(weights.keys(), key=lambda x: abs(weights[x]), reverse=True):
        direction = "+" if weights[factor] > 0 else "-"
        print(f"  {factor}: {direction}{abs(weights[factor])*100:.0f}%")

# ==========================================
# COMPARE OPTIMAL WEIGHTS
# ==========================================

print("\n" + "=" * 70)
print("OPTIMAL WEIGHT COMPARISON")
print("=" * 70)

# Current Compass Score weights
compass_weights = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': -0.15,
    'asset_growth': -0.15,
}

print(f"\n{'Factor':<18} {'Compass (3mo)':>14} {'1-Year Opt':>12} {'2-Year Opt':>12}")
print("-" * 60)

all_factors = set(available_factors)
for factor in sorted(all_factors):
    compass = compass_weights.get(factor, 0) * 100
    y1 = optimal_weights.get('1-Year', {}).get(factor, 0) * 100
    y2 = optimal_weights.get('2-Year', {}).get(factor, 0) * 100
    print(f"{factor:<18} {compass:>+13.0f}% {y1:>+11.0f}% {y2:>+11.0f}%")

# ==========================================
# TEST LONG-TERM SCORE ON OOS
# ==========================================

print("\n" + "=" * 70)
print("LONG-TERM SCORE PERFORMANCE (OOS)")
print("=" * 70)

for horizon_name in ['1-Year', '2-Year']:
    return_col = f'fwd_{horizon_name}'
    weights = optimal_weights[horizon_name]

    print(f"\n--- {horizon_name} Optimized Score ---")

    # Compute score
    oos_df[f'lt_score_{horizon_name}'] = sum(
        oos_df[f'{f}_z'] * weights[f] for f in weights.keys() if f'{f}_z' in oos_df.columns
    )

    # Quintiles
    try:
        oos_df[f'lt_quintile_{horizon_name}'] = pd.qcut(
            oos_df[f'lt_score_{horizon_name}'], 5,
            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop'
        )
    except:
        print("  Could not compute quintiles")
        continue

    print(f"{'Quintile':<10} {'Avg Return':>12} {'Median':>10} {'Count':>10}")
    print("-" * 50)

    quintile_returns = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_df = oos_df[oos_df[f'lt_quintile_{horizon_name}'] == q]
        if len(q_df) > 0:
            avg_ret = q_df[return_col].mean()
            med_ret = q_df[return_col].median()
            count = len(q_df)
            print(f"{q:<10} {avg_ret:>+11.2f}% {med_ret:>+9.2f}% {count:>10,}")
            quintile_returns.append(avg_ret)

    if len(quintile_returns) == 5:
        spread = quintile_returns[4] - quintile_returns[0]
        print("-" * 50)
        print(f"{'Q5-Q1 Spread':<10} {spread:>+11.2f}%")

        # Check monotonicity
        is_monotonic = all(quintile_returns[i] <= quintile_returns[i+1] for i in range(4))
        print(f"{'Monotonic':<10} {'YES' if is_monotonic else 'NO':>12}")

# ==========================================
# COMPARE TO COMPASS SCORE
# ==========================================

print("\n" + "=" * 70)
print("COMPASS SCORE vs LONG-TERM SCORE COMPARISON (OOS)")
print("=" * 70)

# Compute Compass Score
oos_df['compass_score'] = sum(
    oos_df[f'{f}_z'] * w for f, w in compass_weights.items() if f'{f}_z' in oos_df.columns
)

try:
    oos_df['compass_q'] = pd.qcut(oos_df['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
except:
    pass

print(f"\n{'Horizon':<12} {'Compass Q5-Q1':>15} {'LT-Optimized Q5-Q1':>20}")
print("-" * 55)

for horizon_name in HORIZONS.keys():
    return_col = f'fwd_{horizon_name}'

    # Compass Score spread
    compass_q5 = oos_df[oos_df['compass_q'] == 'Q5'][return_col].mean()
    compass_q1 = oos_df[oos_df['compass_q'] == 'Q1'][return_col].mean()
    compass_spread = compass_q5 - compass_q1

    # LT-optimized spread (use 1-Year weights for comparison)
    if horizon_name in ['1-Year', '2-Year']:
        lt_q5 = oos_df[oos_df[f'lt_quintile_{horizon_name}'] == 'Q5'][return_col].mean()
        lt_q1 = oos_df[oos_df[f'lt_quintile_{horizon_name}'] == 'Q1'][return_col].mean()
        lt_spread = lt_q5 - lt_q1
    else:
        lt_spread = compass_spread  # Use same for 3-month

    print(f"{horizon_name:<12} {compass_spread:>+14.2f}% {lt_spread:>+19.2f}%")

# ==========================================
# SUMMARY TABLE FOR PAPER
# ==========================================

print("\n" + "=" * 70)
print("TABLE FOR RESEARCH PAPER")
print("=" * 70)

print("\n**Optimal Factor Weights by Horizon**\n")
print("| Factor | Compass (3mo) | 1-Year | 2-Year |")
print("|--------|---------------|--------|--------|")

for factor in sorted(all_factors):
    c = compass_weights.get(factor, 0) * 100
    y1 = optimal_weights.get('1-Year', {}).get(factor, 0) * 100
    y2 = optimal_weights.get('2-Year', {}).get(factor, 0) * 100

    if factor == 'roa':
        name = 'ROA'
    elif factor == 'ocf_assets':
        name = 'OCF/Assets'
    elif factor == 'fcf_assets':
        name = 'FCF/Assets'
    elif factor == 'gp_assets':
        name = 'GP/Assets'
    elif factor == 'vol_60d':
        name = 'Volatility'
    elif factor == 'asset_growth':
        name = 'Asset Growth'
    elif factor == 'revenue_growth':
        name = 'Revenue Growth'
    elif factor == 'momentum_12m':
        name = 'Momentum'
    elif factor == 'pb_ratio':
        name = 'P/B Ratio'
    else:
        name = factor

    print(f"| {name:<13} | {c:>+11.0f}% | {y1:>+5.0f}% | {y2:>+5.0f}% |")

print(f"\nCompleted: {datetime.now()}")

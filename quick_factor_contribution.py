#!/usr/bin/env python3
"""
Quick Factor Contribution Analysis
===================================
Fast version using sampled data.
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

# Updated V4 Factor Weights (sum to 100%)
V4_WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,  # Updated from 0.10
    'vol_60d': -0.15,
    'asset_growth': -0.15,
}

FORWARD_DAYS = 63
SAMPLE_SIZE = 800  # Smaller sample for speed

print("=" * 70, flush=True)
print("V4 FACTOR CONTRIBUTION ANALYSIS (Quick)", flush=True)
print("Updated weights: GP/Assets = 20%", flush=True)
print("=" * 70, flush=True)
print(f"Started: {datetime.now()}", flush=True)
sys.stdout.flush()

# Load data
print("\nLoading data...", flush=True)
conn = sqlite3.connect(BACKTEST_DB)

# Get prices for recent period only (2019+)
prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 1 AND date >= '2019-01-01'
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])
print(f"  {len(prices):,} price records", flush=True)

# Get fundamentals
fund = pd.read_sql_query("""
    SELECT
        i.symbol, i.date,
        i.net_income, b.total_assets, c.operating_cash_flow,
        c.free_cash_flow, i.gross_profit
    FROM historical_income_statements i
    JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
    JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    WHERE i.date >= '2019-01-01'
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

# Asset growth
fund = fund.sort_values(['symbol', 'date'])
fund['assets_lag'] = fund.groupby('symbol')['total_assets'].shift(4)
fund['asset_growth'] = (fund['total_assets'] / fund['assets_lag']) - 1

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

fund = fund.dropna(subset=['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth'])
print(f"  {len(fund):,} records with factors", flush=True)

# Sample symbols
symbols = fund['symbol'].unique()
np.random.seed(42)
if len(symbols) > SAMPLE_SIZE:
    symbols = np.random.choice(symbols, size=SAMPLE_SIZE, replace=False)
print(f"  Sampled {len(symbols)} symbols", flush=True)

# Filter to sampled symbols
fund = fund[fund['symbol'].isin(symbols)]
prices = prices[prices['symbol'].isin(symbols)]

# Compute volatility
print("\nComputing volatility...", flush=True)
sys.stdout.flush()

vol_results = []
for i, symbol in enumerate(symbols):
    if i % 200 == 0:
        print(f"  {i}/{len(symbols)}...", flush=True)
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    if len(sym_prices) < 80:
        continue

    sym_prices['return'] = sym_prices['close'].pct_change()
    sym_prices['vol_60d'] = sym_prices['return'].rolling(60, min_periods=40).std() * np.sqrt(252)

    for _, row in sym_prices[sym_prices['vol_60d'].notna()].iterrows():
        vol_results.append({
            'symbol': symbol,
            'date': row['date'],
            'vol_60d': row['vol_60d'],
        })

vol_df = pd.DataFrame(vol_results)
print(f"  {len(vol_df):,} volatility records", flush=True)

# Compute forward returns
print("\nComputing forward returns...", flush=True)
sys.stdout.flush()

fwd_results = []
for i, symbol in enumerate(symbols):
    if i % 200 == 0:
        print(f"  {i}/{len(symbols)}...", flush=True)
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    n = len(sym_prices)

    if n < FORWARD_DAYS + 20:
        continue

    close = sym_prices['close'].values
    dates = sym_prices['date'].values

    # Sample every 21 days (monthly)
    for j in range(0, n - FORWARD_DAYS, 21):
        entry = close[j]
        exit_price = close[j + FORWARD_DAYS]

        if entry > 0:
            fwd_ret = ((exit_price / entry) - 1) * 100
            fwd_ret = np.clip(fwd_ret, -100, 200)

            fwd_results.append({
                'symbol': symbol,
                'date': pd.Timestamp(dates[j]),
                'fwd_return': fwd_ret
            })

fwd_df = pd.DataFrame(fwd_results)
print(f"  {len(fwd_df):,} forward return records", flush=True)

# Merge data
print("\nMerging data...", flush=True)
sys.stdout.flush()

# Create monthly periods for merging
fund['month'] = fund['date'].dt.to_period('M')
vol_df['month'] = vol_df['date'].dt.to_period('M')
fwd_df['month'] = fwd_df['date'].dt.to_period('M')

# Get latest fundamental per symbol-month
fund_monthly = fund.sort_values('date').groupby(['symbol', 'month']).last().reset_index()

# Merge
df = vol_df.merge(fund_monthly[['symbol', 'month', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']],
                  on=['symbol', 'month'], how='inner')
df = df.merge(fwd_df[['symbol', 'month', 'fwd_return']], on=['symbol', 'month'], how='inner')

print(f"  {len(df):,} merged records", flush=True)

# Filter to 2020+ (OOS)
df = df[df['date'] >= '2020-01-01']
print(f"  {len(df):,} OOS records (2020+)", flush=True)

if len(df) < 1000:
    print("ERROR: Not enough data. Exiting.")
    sys.exit(1)

# Compute z-scores
print("\nComputing z-scores...", flush=True)
factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d', 'asset_growth']

for col in factor_cols:
    # Winsorize
    lower = df[col].quantile(0.01)
    upper = df[col].quantile(0.99)
    df[col] = df[col].clip(lower, upper)

    # Z-score
    mean = df[col].mean()
    std = df[col].std()
    if std > 0:
        df[f'{col}_z'] = ((df[col] - mean) / std).clip(-3, 3)
    else:
        df[f'{col}_z'] = 0

# Compute V4 score
df['v4_score'] = (
    df['roa_z'] * V4_WEIGHTS['roa'] +
    df['ocf_assets_z'] * V4_WEIGHTS['ocf_assets'] +
    df['fcf_assets_z'] * V4_WEIGHTS['fcf_assets'] +
    df['gp_assets_z'] * V4_WEIGHTS['gp_assets'] +
    df['vol_60d_z'] * V4_WEIGHTS['vol_60d'] +
    df['asset_growth_z'] * V4_WEIGHTS['asset_growth']
)

# Assign quintiles
df['quintile'] = pd.qcut(df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Overall performance
print("\n" + "=" * 70)
print("V4 QUINTILE PERFORMANCE (OOS 2020+)")
print("=" * 70)

print(f"\n{'Quintile':<10} {'Avg Return':>12} {'Median':>10} {'Count':>10}")
print("-" * 50)

for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    q_df = df[df['quintile'] == q]
    avg_ret = q_df['fwd_return'].mean()
    med_ret = q_df['fwd_return'].median()
    count = len(q_df)
    print(f"{q:<10} {avg_ret:>+11.2f}% {med_ret:>+9.2f}% {count:>10,}")

q5_ret = df[df['quintile'] == 'Q5']['fwd_return'].mean()
q1_ret = df[df['quintile'] == 'Q1']['fwd_return'].mean()
spread = q5_ret - q1_ret

print("-" * 50)
print(f"{'Q5-Q1 Spread':<10} {spread:>+11.2f}%")

# ==========================================
# FACTOR CONTRIBUTION ANALYSIS
# ==========================================
print("\n" + "=" * 70)
print("FACTOR CONTRIBUTION ANALYSIS")
print("=" * 70)

factor_spreads = []

for factor in factor_cols:
    z_col = f'{factor}_z'
    weight = V4_WEIGHTS[factor]

    # Create quintiles for this single factor
    try:
        df[f'{factor}_quintile'] = pd.qcut(df[z_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except:
        continue

    # Compute spread
    q5_ret_f = df[df[f'{factor}_quintile'] == 'Q5']['fwd_return'].mean()
    q1_ret_f = df[df[f'{factor}_quintile'] == 'Q1']['fwd_return'].mean()

    # For negative factors, Q1 is "good"
    if weight < 0:
        single_spread = q1_ret_f - q5_ret_f
    else:
        single_spread = q5_ret_f - q1_ret_f

    weighted_contrib = single_spread * abs(weight)

    factor_spreads.append({
        'factor': factor,
        'weight': abs(weight) * 100,
        'direction': '+' if weight > 0 else '-',
        'single_spread': single_spread,
        'contribution': weighted_contrib,
    })

spread_df = pd.DataFrame(factor_spreads)
spread_df = spread_df.sort_values('contribution', ascending=False)

# Compute percentage
total = spread_df['contribution'].sum()
spread_df['pct'] = spread_df['contribution'] / total * 100

print(f"\n{'Factor':<18} {'Weight':>8} {'Dir':>5} {'Single Spread':>15} {'Contribution':>14} {'% Total':>10}")
print("-" * 75)

for _, row in spread_df.iterrows():
    print(f"{row['factor']:<18} {row['weight']:>7.0f}% {row['direction']:>5} "
          f"{row['single_spread']:>+14.2f}% {row['contribution']:>+13.2f}% {row['pct']:>9.1f}%")

print("-" * 75)
print(f"{'TOTAL':<18} {'100':>7}% {'':<5} {'':<15} {total:>+13.2f}%")

# Summary for paper
print("\n" + "=" * 70)
print("TABLE FOR RESEARCH PAPER (Section 4.7)")
print("=" * 70)

print("\n| Factor | Contribution | % of Total |")
print("|--------|--------------|------------|")

for _, row in spread_df.iterrows():
    f = row['factor']
    if f == 'roa':
        name = 'ROA'
    elif f == 'ocf_assets':
        name = 'OCF/Assets'
    elif f == 'fcf_assets':
        name = 'FCF/Assets'
    elif f == 'gp_assets':
        name = 'GP/Assets'
    elif f == 'vol_60d':
        name = '-Volatility'
    elif f == 'asset_growth':
        name = '-Asset Growth'
    else:
        name = f

    print(f"| {name:<14} | {row['contribution']:>+10.2f}% | {row['pct']:>8.0f}% |")

print(f"\nCompleted: {datetime.now()}")

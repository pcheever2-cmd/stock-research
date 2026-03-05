#!/usr/bin/env python3
"""
Additional Institutional Tests:
1. Sector-Neutral Backtest (rank within sectors)
2. Rolling 3-Year Sharpe Analysis
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import requests
import time

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Load API key
try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
except:
    FMP_API_KEY = ''

print("=" * 70)
print("SECTOR-NEUTRAL BACKTEST & ROLLING 3-YEAR SHARPE")
print("=" * 70)
sys.stdout.flush()

# ============================================================================
# PART 1: FETCH SECTOR DATA
# ============================================================================

print("\n[1/4] Fetching sector classifications...")
sys.stdout.flush()

# Use hardcoded sector classifications for major large-cap stocks
# This covers the most important stocks for sector-neutral analysis
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology',
    'AVGO': 'Technology', 'TXN': 'Technology', 'QCOM': 'Technology', 'IBM': 'Technology',
    'NOW': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology',
    'LRCX': 'Technology', 'KLAC': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology',
    'MRVL': 'Technology', 'ADI': 'Technology', 'NXPI': 'Technology', 'MCHP': 'Technology',
    'FTNT': 'Technology', 'PANW': 'Technology', 'CRWD': 'Technology', 'ZS': 'Technology',
    'NET': 'Technology', 'DDOG': 'Technology', 'SNOW': 'Technology', 'MDB': 'Technology',
    'PLTR': 'Technology', 'PATH': 'Technology', 'DELL': 'Technology', 'HPE': 'Technology',
    'HPQ': 'Technology', 'WDC': 'Technology', 'STX': 'Technology', 'SMCI': 'Technology',
    'ARM': 'Technology', 'TSM': 'Technology', 'ASML': 'Technology', 'SAP': 'Technology',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'SCHW': 'Financials', 'BLK': 'Financials', 'AXP': 'Financials', 'COF': 'Financials',
    'CME': 'Financials', 'ICE': 'Financials', 'MCO': 'Financials', 'SPGI': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials', 'SQ': 'Financials',
    'BRK-A': 'Financials', 'BRK-B': 'Financials', 'MET': 'Financials', 'AIG': 'Financials',
    'PRU': 'Financials', 'TRV': 'Financials', 'ALL': 'Financials', 'CB': 'Financials',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'ISRG': 'Healthcare', 'MDT': 'Healthcare',
    'SYK': 'Healthcare', 'BSX': 'Healthcare', 'ZBH': 'Healthcare', 'EW': 'Healthcare',
    'CVS': 'Healthcare', 'CI': 'Healthcare', 'HUM': 'Healthcare', 'ELV': 'Healthcare',
    'MRNA': 'Healthcare', 'BIIB': 'Healthcare', 'ILMN': 'Healthcare', 'A': 'Healthcare',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'ROST': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
    'HLT': 'Consumer Discretionary', 'BKNG': 'Consumer Discretionary', 'EXPE': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'LVS': 'Consumer Discretionary',
    'WYNN': 'Consumer Discretionary', 'MGM': 'Consumer Discretionary', 'RCL': 'Consumer Discretionary',
    'CCL': 'Consumer Discretionary', 'NCLH': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'COST': 'Consumer Staples', 'WMT': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'MDLZ': 'Consumer Staples', 'CL': 'Consumer Staples',
    'KMB': 'Consumer Staples', 'GIS': 'Consumer Staples', 'K': 'Consumer Staples',
    'HSY': 'Consumer Staples', 'KHC': 'Consumer Staples', 'STZ': 'Consumer Staples',
    'ADM': 'Consumer Staples', 'EL': 'Consumer Staples', 'MNST': 'Consumer Staples',

    # Industrials
    'CAT': 'Industrials', 'DE': 'Industrials', 'GE': 'Industrials', 'HON': 'Industrials',
    'UNP': 'Industrials', 'UPS': 'Industrials', 'RTX': 'Industrials', 'LMT': 'Industrials',
    'BA': 'Industrials', 'NOC': 'Industrials', 'GD': 'Industrials', 'MMM': 'Industrials',
    'EMR': 'Industrials', 'ITW': 'Industrials', 'ETN': 'Industrials', 'PH': 'Industrials',
    'ROK': 'Industrials', 'FDX': 'Industrials', 'CSX': 'Industrials', 'NSC': 'Industrials',
    'WM': 'Industrials', 'RSG': 'Industrials', 'CTAS': 'Industrials', 'PCAR': 'Industrials',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'PXD': 'Energy', 'MPC': 'Energy', 'VLO': 'Energy',
    'PSX': 'Energy', 'OXY': 'Energy', 'HAL': 'Energy', 'BKR': 'Energy',
    'KMI': 'Energy', 'WMB': 'Energy', 'OKE': 'Energy', 'DVN': 'Energy',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'SRE': 'Utilities', 'EXC': 'Utilities', 'XEL': 'Utilities',
    'ED': 'Utilities', 'WEC': 'Utilities', 'ES': 'Utilities', 'AWK': 'Utilities',

    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate',
    'AVB': 'Real Estate', 'EQR': 'Real Estate', 'DLR': 'Real Estate', 'ARE': 'Real Estate',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials',
    'DD': 'Materials', 'DOW': 'Materials', 'NEM': 'Materials', 'FCX': 'Materials',
    'NUE': 'Materials', 'VMC': 'Materials', 'MLM': 'Materials', 'PPG': 'Materials',

    # Communication Services
    'DIS': 'Communication Services', 'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    'T': 'Communication Services', 'VZ': 'Communication Services', 'TMUS': 'Communication Services',
    'CHTR': 'Communication Services', 'EA': 'Communication Services', 'ATVI': 'Communication Services',
    'TTWO': 'Communication Services', 'WBD': 'Communication Services', 'PARA': 'Communication Services',
}

sectors_df = pd.DataFrame([
    {'symbol': sym, 'sector': sect} for sym, sect in SECTOR_MAP.items()
])
print(f"  {len(sectors_df):,} sector mappings (hardcoded large-caps)")
sys.stdout.flush()

# ============================================================================
# PART 2: LOAD DATA
# ============================================================================

print("\n[2/4] Loading backtest data...")
sys.stdout.flush()

conn = sqlite3.connect(BACKTEST_DB)

prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 1
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])
print(f"  {len(prices):,} price records")

fund = pd.read_sql_query("""
    SELECT i.symbol, i.date, i.gross_profit, i.net_income,
           b.total_assets, c.operating_cash_flow, c.free_cash_flow,
           m.market_cap
    FROM historical_income_statements i
    JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
    JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
""", conn)
fund['date'] = pd.to_datetime(fund['date'])
print(f"  {len(fund):,} fundamental records")
conn.close()
sys.stdout.flush()

# Compute factors
fund['roa'] = fund['net_income'] / fund['total_assets']
fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
fund = fund.sort_values(['symbol', 'date'])
fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

# Merge sectors
fund = fund.merge(sectors_df[['symbol', 'sector']], on='symbol', how='left')
fund['sector'] = fund['sector'].fillna('Unknown')

# ============================================================================
# PART 3: BUILD MONTHLY OBSERVATIONS (LARGE-CAP ONLY)
# ============================================================================

print("\n[3/4] Building monthly observations (Large-Cap >$10B)...")
sys.stdout.flush()

results = []
all_symbols = prices['symbol'].unique()

for i, symbol in enumerate(all_symbols):
    if i % 1000 == 0 and i > 0:
        print(f"  {i}/{len(all_symbols)}...")
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    n = len(sym_prices)

    if n < 300:
        continue

    close = sym_prices['close'].values

    for j in range(252, n - 21, 21):
        date = sym_prices['date'].iloc[j]

        # Forward 1-month return
        fwd_1m = ((close[j + 21] / close[j]) - 1) * 100

        # Volatility
        rets = np.diff(close[j-60:j+1]) / close[j-60:j]
        vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

        results.append({
            'symbol': symbol,
            'date': date,
            'fwd_1m': fwd_1m,
            'vol_60d': vol_60d,
        })

df = pd.DataFrame(results)
print(f"  {len(df):,} observations")
sys.stdout.flush()

# Merge fundamentals
fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'market_cap', 'sector']
fund_clean = fund[fund_cols].dropna(subset=['roa'])

all_merged = []
for symbol in df['symbol'].unique():
    pf = df[df['symbol'] == symbol].sort_values('date')
    f = fund_clean[fund_clean['symbol'] == symbol].sort_values('date')

    if len(f) == 0:
        continue

    merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                           on='date', direction='backward',
                           tolerance=pd.Timedelta('365 days'))
    all_merged.append(merged)

df = pd.concat(all_merged, ignore_index=True)
df = df.dropna(subset=['fwd_1m', 'roa', 'market_cap', 'vol_60d', 'sector'])

# Filter to large-cap
df = df[df['market_cap'] > 10e9]
print(f"  {len(df):,} large-cap observations with sectors")
sys.stdout.flush()

# Split IS/OOS
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
is_df = df[df['date'] <= IS_END]
oos_df = df[df['date'] >= OOS_START]
print(f"  IS: {len(is_df):,}  |  OOS: {len(oos_df):,}")
sys.stdout.flush()

# ============================================================================
# PART 4: SECTOR-NEUTRAL BACKTEST
# ============================================================================

print("\n" + "=" * 70)
print("SECTOR-NEUTRAL BACKTEST (Large-Cap OOS)")
print("=" * 70)
sys.stdout.flush()

# Compute z-scores using IS stats
is_stats = {}
for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
    is_stats[col] = {'mean': is_df[col].mean(), 'std': is_df[col].std()}

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
    oos_df[f'{col}_z'] = (oos_df[col] - is_stats[col]['mean']) / is_stats[col]['std']

# Compute global Compass Score
oos_df['compass_score'] = (
    oos_df['roa_z'] * 0.20 +
    oos_df['ocf_assets_z'] * 0.15 +
    oos_df['fcf_assets_z'] * 0.15 +
    oos_df['gp_assets_z'] * 0.20 +
    (-oos_df['vol_60d_z']) * 0.15 +
    (-oos_df['asset_growth_z']) * 0.15
)

# Global quintiles (standard approach)
oos_df['quintile_global'] = pd.qcut(oos_df['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Sector-neutral: rank within each sector-month
def sector_neutral_quintile(group):
    if len(group) >= 5:
        return pd.qcut(group['compass_score'].rank(method='first'), 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    else:
        return pd.Series(['Q3'] * len(group), index=group.index)

oos_df['quintile_sector_neutral'] = oos_df.groupby([oos_df['date'].dt.to_period('M'), 'sector']).apply(
    sector_neutral_quintile
).reset_index(level=[0, 1], drop=True)

# Compare results
print("\nGLOBAL QUINTILES (Standard):")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    ret = oos_df[oos_df['quintile_global'] == q]['fwd_1m'].mean()
    print(f"  {q}: {ret:+.2f}%")
global_spread = oos_df[oos_df['quintile_global'] == 'Q5']['fwd_1m'].mean() - oos_df[oos_df['quintile_global'] == 'Q1']['fwd_1m'].mean()
print(f"  Q5-Q1 Spread: {global_spread:+.2f}%")
sys.stdout.flush()

print("\nSECTOR-NEUTRAL QUINTILES:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    ret = oos_df[oos_df['quintile_sector_neutral'] == q]['fwd_1m'].mean()
    print(f"  {q}: {ret:+.2f}%")
sn_spread = oos_df[oos_df['quintile_sector_neutral'] == 'Q5']['fwd_1m'].mean() - oos_df[oos_df['quintile_sector_neutral'] == 'Q1']['fwd_1m'].mean()
print(f"  Q5-Q1 Spread: {sn_spread:+.2f}%")
sys.stdout.flush()

print(f"\nSector-Neutral vs Global: {sn_spread - global_spread:+.2f}% difference")

# Check sector distribution in Q1 and Q5 (global)
print("\nSector Distribution in Global Quintiles:")
print("\nQ1 (Low Quality) - Top 5 Sectors:")
q1_sectors = oos_df[oos_df['quintile_global'] == 'Q1']['sector'].value_counts(normalize=True).head(5) * 100
for sector, pct in q1_sectors.items():
    print(f"  {sector}: {pct:.1f}%")

print("\nQ5 (High Quality) - Top 5 Sectors:")
q5_sectors = oos_df[oos_df['quintile_global'] == 'Q5']['sector'].value_counts(normalize=True).head(5) * 100
for sector, pct in q5_sectors.items():
    print(f"  {sector}: {pct:.1f}%")
sys.stdout.flush()

# ============================================================================
# PART 5: ROLLING 3-YEAR SHARPE
# ============================================================================

print("\n" + "=" * 70)
print("ROLLING 3-YEAR SHARPE ANALYSIS (36-Month Window)")
print("=" * 70)
sys.stdout.flush()

# Compute monthly Q5-Q1 returns
monthly_rets = oos_df.groupby([oos_df['date'].dt.to_period('M'), 'quintile_global'])['fwd_1m'].mean().unstack()
monthly_rets['spread'] = monthly_rets['Q5'] - monthly_rets['Q1']
monthly_rets = monthly_rets.reset_index()
monthly_rets['date'] = monthly_rets['date'].dt.to_timestamp()

# Rolling 36-month Sharpe
window = 36
rolling_sharpe = []

spread_series = monthly_rets['spread'].values
dates = monthly_rets['date'].values

for i in range(window, len(spread_series)):
    window_data = spread_series[i-window:i]
    mean_ret = np.mean(window_data)
    std_ret = np.std(window_data)
    if std_ret > 0:
        sharpe = (mean_ret / std_ret) * np.sqrt(12)  # Annualized
    else:
        sharpe = 0
    rolling_sharpe.append({
        'date': dates[i],
        'sharpe_36m': sharpe,
        'mean_ret_36m': mean_ret,
        'std_ret_36m': std_ret
    })

rolling_df = pd.DataFrame(rolling_sharpe)

print("\nRolling 36-Month Sharpe Ratio:")
print("-" * 50)
print(f"{'End Date':<15} {'Sharpe':<10} {'Mean Ret':<12} {'Volatility':<10}")
print("-" * 50)
for _, row in rolling_df.iterrows():
    print(f"{row['date'].strftime('%Y-%m'):<15} {row['sharpe_36m']:+.2f}      {row['mean_ret_36m']:+.2f}%       {row['std_ret_36m']:.2f}%")
sys.stdout.flush()

print("\n" + "-" * 50)
print("ROLLING SHARPE SUMMARY:")
print(f"  Min Sharpe: {rolling_df['sharpe_36m'].min():+.2f}")
print(f"  Max Sharpe: {rolling_df['sharpe_36m'].max():+.2f}")
print(f"  Mean Sharpe: {rolling_df['sharpe_36m'].mean():+.2f}")
print(f"  Current (latest): {rolling_df['sharpe_36m'].iloc[-1]:+.2f}")
sys.stdout.flush()

# Check if Sharpe ever went negative
negative_periods = rolling_df[rolling_df['sharpe_36m'] < 0]
if len(negative_periods) > 0:
    print(f"\n⚠️ Periods with NEGATIVE 36-month Sharpe: {len(negative_periods)}")
    for _, row in negative_periods.iterrows():
        print(f"  {row['date'].strftime('%Y-%m')}: {row['sharpe_36m']:+.2f}")
else:
    print("\n✓ Sharpe remained POSITIVE across all 36-month windows")
sys.stdout.flush()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY FOR RESEARCH PAPER")
print("=" * 70)
sys.stdout.flush()

print(f"""
## Sector-Neutral Analysis (Large-Cap OOS 2020-2026)

| Approach | Q5-Q1 Spread | Notes |
|----------|--------------|-------|
| Global Quintiles | {global_spread:+.2f}% | Standard methodology |
| Sector-Neutral | {sn_spread:+.2f}% | Rank within sectors |

**Finding**: Sector-neutral spread is {sn_spread - global_spread:+.2f}% vs global.
{"The signal remains strong even after neutralizing sector tilts." if sn_spread > 0 else "Sector tilts contribute meaningfully to the spread."}

## Rolling 3-Year Sharpe

| Metric | Value |
|--------|-------|
| Min 36-mo Sharpe | {rolling_df['sharpe_36m'].min():+.2f} |
| Max 36-mo Sharpe | {rolling_df['sharpe_36m'].max():+.2f} |
| Mean 36-mo Sharpe | {rolling_df['sharpe_36m'].mean():+.2f} |
| Current 36-mo Sharpe | {rolling_df['sharpe_36m'].iloc[-1]:+.2f} |
| Negative Periods | {len(negative_periods)} |

**Finding**: {"Rolling Sharpe remained positive throughout, indicating consistent risk-adjusted performance." if len(negative_periods) == 0 else f"Rolling Sharpe went negative in {len(negative_periods)} periods, indicating regime sensitivity."}
""")

print("=" * 70)
print("COMPLETED")
print("=" * 70)

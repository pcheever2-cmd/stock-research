#!/usr/bin/env python3
"""
Investigate why Compass Score underperformed in 2025-2026.
Hypothesis: AI/tech rally benefited low-quality growth stocks.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

print("=" * 70)
print("INVESTIGATING 2025-2026 COMPASS SCORE UNDERPERFORMANCE")
print("=" * 70)
sys.stdout.flush()

# Load data
print("\nLoading data...")
sys.stdout.flush()
conn = sqlite3.connect(BACKTEST_DB)

prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 1 AND date >= '2024-01-01'
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])
print(f"  {len(prices):,} price records")
sys.stdout.flush()

fund = pd.read_sql_query("""
    SELECT i.symbol, i.date, i.gross_profit, i.net_income,
           b.total_assets, c.operating_cash_flow, c.free_cash_flow,
           m.market_cap
    FROM historical_income_statements i
    JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
    JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    WHERE i.date >= '2023-01-01'
""", conn)
fund['date'] = pd.to_datetime(fund['date'])
print(f"  {len(fund):,} fundamental records")
sys.stdout.flush()

# Sector data not available in DB - use symbol patterns for rough classification
def classify_sector(symbol):
    """Rough sector classification based on known tickers."""
    tech_symbols = {'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'ADBE',
                    'ORCL', 'CSCO', 'AVGO', 'TXN', 'QCOM', 'IBM', 'NOW', 'INTU', 'AMAT', 'MU',
                    'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'ADI', 'NXPI', 'MCHP', 'ON', 'FTNT',
                    'PANW', 'CRWD', 'ZS', 'NET', 'DDOG', 'SNOW', 'MDB', 'PLTR', 'PATH', 'DOCN',
                    'HPQ', 'HPE', 'DELL', 'WDC', 'STX', 'SMCI', 'ARM', 'TSM', 'ASML', 'SHOP'}
    ai_symbols = {'NVDA', 'AMD', 'SMCI', 'PLTR', 'ARM', 'MRVL', 'AVGO', 'TSM', 'ASML', 'MU',
                  'META', 'GOOGL', 'MSFT', 'AMZN', 'CRM', 'SNOW', 'DDOG', 'PATH', 'NOW', 'AI'}
    if symbol in tech_symbols:
        return 'Technology'
    elif symbol in ai_symbols:
        return 'AI-Related'
    else:
        return 'Other'

conn.close()

# Compute factors
print("\nComputing factors...")
sys.stdout.flush()
fund['roa'] = fund['net_income'] / fund['total_assets']
fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
fund = fund.sort_values(['symbol', 'date'])
fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

# Build monthly observations for 2024-2026
print("\nBuilding monthly observations...")
sys.stdout.flush()

results = []
all_symbols = prices['symbol'].unique()

for i, symbol in enumerate(all_symbols):
    if i % 500 == 0 and i > 0:
        print(f"  {i}/{len(all_symbols)}...")
        sys.stdout.flush()

    sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
    n = len(sym_prices)

    if n < 60:
        continue

    close = sym_prices['close'].values

    for j in range(60, n - 21, 21):
        date = sym_prices['date'].iloc[j]

        # Only 2025-2026
        if date.year < 2025:
            continue

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
print(f"  {len(df):,} observations in 2025-2026")
sys.stdout.flush()

# Merge fundamentals
fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'market_cap']
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
df = df.dropna(subset=['fwd_1m', 'roa', 'market_cap', 'vol_60d'])

# Classify sectors using known symbols
df['sector'] = df['symbol'].apply(classify_sector)

# Filter to large-cap only (>$10B)
df = df[df['market_cap'] > 10e9]
print(f"  {len(df):,} large-cap observations with fundamentals")
sys.stdout.flush()

# Compute Compass Score
print("\nComputing Compass Score...")
sys.stdout.flush()

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
    df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()

df['compass_score'] = (
    df['roa_z'] * 0.20 +
    df['ocf_assets_z'] * 0.15 +
    df['fcf_assets_z'] * 0.15 +
    df['gp_assets_z'] * 0.20 +
    (-df['vol_60d_z']) * 0.15 +
    (-df['asset_growth_z']) * 0.15
)

# Assign quintiles
df['quintile'] = pd.qcut(df['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

print("\n" + "=" * 70)
print("ANALYSIS 1: SECTOR COMPOSITION BY QUINTILE")
print("=" * 70)
sys.stdout.flush()

# Sector breakdown by quintile
for q in ['Q1', 'Q5']:
    print(f"\n{q} Sector Distribution:")
    q_df = df[df['quintile'] == q]
    sector_counts = q_df['sector'].value_counts(normalize=True) * 100
    for sector, pct in sector_counts.head(10).items():
        print(f"  {sector}: {pct:.1f}%")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 2: MONTHLY RETURNS BY QUINTILE (2025-2026)")
print("=" * 70)
sys.stdout.flush()

# Monthly returns by quintile
monthly_rets = df.groupby([df['date'].dt.to_period('M'), 'quintile'])['fwd_1m'].mean().unstack()
print("\nMonthly Q5-Q1 Spread:")
monthly_rets['spread'] = monthly_rets['Q5'] - monthly_rets['Q1']
for period, row in monthly_rets.iterrows():
    print(f"  {period}: Q5={row['Q5']:+.2f}%, Q1={row['Q1']:+.2f}%, Spread={row['spread']:+.2f}%")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 3: SECTOR RETURNS (Q1 vs Q5)")
print("=" * 70)
sys.stdout.flush()

# Which sectors drove Q1 outperformance?
sector_q1_ret = df[df['quintile'] == 'Q1'].groupby('sector')['fwd_1m'].mean()
sector_q5_ret = df[df['quintile'] == 'Q5'].groupby('sector')['fwd_1m'].mean()
sector_count = df.groupby('sector').size()

sector_analysis = pd.DataFrame({
    'Q1_ret': sector_q1_ret,
    'Q5_ret': sector_q5_ret,
    'count': sector_count
})
sector_analysis['spread'] = sector_analysis['Q5_ret'] - sector_analysis['Q1_ret']
sector_analysis = sector_analysis.sort_values('spread')

print("\nSector Q5-Q1 Spreads (sorted worst to best):")
for sector, row in sector_analysis.iterrows():
    if row['count'] > 50:
        print(f"  {sector}: Spread={row['spread']:+.2f}% (Q1={row['Q1_ret']:+.2f}%, Q5={row['Q5_ret']:+.2f}%, n={int(row['count'])})")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 4: TOP Q1 PERFORMERS (Low Quality, High Returns)")
print("=" * 70)
sys.stdout.flush()

# Find the Q1 stocks that drove the weakness
q1_best = df[df['quintile'] == 'Q1'].groupby('symbol').agg({
    'fwd_1m': 'mean',
    'sector': 'first',
    'market_cap': 'mean',
    'roa': 'mean',
    'asset_growth': 'mean',
    'vol_60d': 'mean'
}).sort_values('fwd_1m', ascending=False)

print("\nTop 20 Q1 (Low Quality) Performers:")
print(f"{'Symbol':<8} {'Sector':<20} {'Avg Return':<12} {'ROA':<10} {'AssetGr':<10} {'Vol':<10}")
print("-" * 80)
for symbol, row in q1_best.head(20).iterrows():
    print(f"{symbol:<8} {row['sector'][:20]:<20} {row['fwd_1m']:+.2f}%       {row['roa']*100:+.1f}%      {row['asset_growth']*100:+.1f}%      {row['vol_60d']:.1f}%")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 5: Q5 UNDERPERFORMERS (High Quality, Low Returns)")
print("=" * 70)
sys.stdout.flush()

q5_worst = df[df['quintile'] == 'Q5'].groupby('symbol').agg({
    'fwd_1m': 'mean',
    'sector': 'first',
    'market_cap': 'mean',
    'roa': 'mean'
}).sort_values('fwd_1m')

print("\nBottom 20 Q5 (High Quality) Performers:")
print(f"{'Symbol':<8} {'Sector':<20} {'Avg Return':<12} {'ROA':<10}")
print("-" * 60)
for symbol, row in q5_worst.head(20).iterrows():
    print(f"{symbol:<8} {row['sector'][:20]:<20} {row['fwd_1m']:+.2f}%       {row['roa']*100:+.1f}%")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 6: KNOWN TECH/AI STOCKS DEEP DIVE")
print("=" * 70)
sys.stdout.flush()

tech_df = df[df['sector'] == 'Technology']
if len(tech_df) > 0:
    print(f"\nKnown Tech stocks: {len(tech_df):,} observations")

    tech_by_q = tech_df.groupby('quintile')['fwd_1m'].agg(['mean', 'count'])
    print("\nTech Returns by Quintile:")
    for q, row in tech_by_q.iterrows():
        print(f"  {q}: {row['mean']:+.2f}% (n={int(row['count'])})")

    # What characterizes Q1 tech vs Q5 tech?
    q1_tech = tech_df[tech_df['quintile'] == 'Q1']
    q5_tech = tech_df[tech_df['quintile'] == 'Q5']

    if len(q1_tech) > 0:
        print("\nQ1 Tech (Low Quality) characteristics:")
        print(f"  Avg ROA: {q1_tech['roa'].mean()*100:.1f}%")
        print(f"  Avg Asset Growth: {q1_tech['asset_growth'].mean()*100:.1f}%")
        print(f"  Avg Volatility: {q1_tech['vol_60d'].mean():.1f}%")
        print(f"  Avg Return: {q1_tech['fwd_1m'].mean():+.2f}%")
        print(f"  Stocks: {q1_tech['symbol'].unique()[:10].tolist()}")

    if len(q5_tech) > 0:
        print("\nQ5 Tech (High Quality) characteristics:")
        print(f"  Avg ROA: {q5_tech['roa'].mean()*100:.1f}%")
        print(f"  Avg Asset Growth: {q5_tech['asset_growth'].mean()*100:.1f}%")
        print(f"  Avg Volatility: {q5_tech['vol_60d'].mean():.1f}%")
        print(f"  Avg Return: {q5_tech['fwd_1m'].mean():+.2f}%")
        print(f"  Stocks: {q5_tech['symbol'].unique()[:10].tolist()}")
sys.stdout.flush()

print("\n" + "=" * 70)
print("ANALYSIS 7: ASSET GROWTH FACTOR (AI CAPEX BOOM?)")
print("=" * 70)
sys.stdout.flush()

# High asset growth stocks (AI capex?)
high_ag = df[df['asset_growth'] > df['asset_growth'].quantile(0.8)]
low_ag = df[df['asset_growth'] < df['asset_growth'].quantile(0.2)]

print(f"\nHigh Asset Growth (top 20%): Avg return = {high_ag['fwd_1m'].mean():+.2f}%")
print(f"Low Asset Growth (bottom 20%): Avg return = {low_ag['fwd_1m'].mean():+.2f}%")
print(f"Spread (Low - High): {low_ag['fwd_1m'].mean() - high_ag['fwd_1m'].mean():+.2f}%")
sys.stdout.flush()

# Check if asset growth hurting the score
print("\nAsset Growth by Quintile:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    q_ag = df[df['quintile'] == q]['asset_growth'].mean() * 100
    print(f"  {q}: Avg Asset Growth = {q_ag:+.1f}%")
sys.stdout.flush()

print("\n" + "=" * 70)
print("SUMMARY: WHY 2025-2026 UNDERPERFORMED")
print("=" * 70)
sys.stdout.flush()

avg_spread = monthly_rets['spread'].mean()
print(f"\nAverage monthly Q5-Q1 spread in 2025-2026: {avg_spread:+.2f}%")

# Key findings
print("\nKEY FINDINGS:")
print("-" * 50)

# Check tech concentration
tech_q1_pct = (df[df['quintile'] == 'Q1']['sector'] == 'Technology').mean() * 100
tech_q5_pct = (df[df['quintile'] == 'Q5']['sector'] == 'Technology').mean() * 100
print(f"1. Tech concentration: Q1={tech_q1_pct:.1f}%, Q5={tech_q5_pct:.1f}%")

# Asset growth reversal
ag_spread = low_ag['fwd_1m'].mean() - high_ag['fwd_1m'].mean()
print(f"2. Asset Growth factor: {ag_spread:+.2f}% (negative = high growth winning)")

# Volatility
high_vol = df[df['vol_60d'] > df['vol_60d'].quantile(0.8)]['fwd_1m'].mean()
low_vol = df[df['vol_60d'] < df['vol_60d'].quantile(0.2)]['fwd_1m'].mean()
vol_spread = low_vol - high_vol
print(f"3. Volatility factor: {vol_spread:+.2f}% (negative = high vol winning)")

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

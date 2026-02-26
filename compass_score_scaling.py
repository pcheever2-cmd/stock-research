#!/usr/bin/env python3
"""
Compass Score Scaling & Distribution Analysis
==============================================
1. Transform raw score → 0-100 percentile scale
2. Show S&P 500 distribution context
3. Grade bucket examples
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

print("=" * 70)
print("COMPASS SCORE: 0-100 SCALING & DISTRIBUTION")
print("=" * 70)
sys.stdout.flush()

# S&P 500 components (approximate - major names)
SP500_CORE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSLA', 'UNH', 'JNJ',
    'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'PFE', 'TMO', 'MCD', 'WMT', 'CSCO', 'ABT',
    'DHR', 'ACN', 'CRM', 'NKE', 'TXN', 'NEE', 'PM', 'ORCL', 'UPS', 'MS',
    'RTX', 'HON', 'INTC', 'UNP', 'IBM', 'QCOM', 'LOW', 'GS', 'CAT', 'BA',
    'AMGN', 'AMD', 'SPGI', 'GE', 'SBUX', 'INTU', 'BLK', 'ISRG', 'AXP', 'MDLZ',
    'LMT', 'ADI', 'SYK', 'GILD', 'BKNG', 'TJX', 'MMC', 'VRTX', 'PLD', 'ADP',
    'CVS', 'C', 'REGN', 'CI', 'NOW', 'MO', 'SO', 'DUK', 'CL', 'EOG',
    'BSX', 'BDX', 'SLB', 'CME', 'MMM', 'NOC', 'WM', 'ITW', 'EQIX', 'CSX',
    'PNC', 'USB', 'TGT', 'APD', 'FCX', 'EMR', 'NSC', 'FDX', 'SHW', 'COP'
]

print("\nLoading data...")
sys.stdout.flush()
conn = sqlite3.connect(BACKTEST_DB)

prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 0.5 AND date >= '2024-01-01'
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])

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
conn.close()

# Compute factors
fund['roa'] = fund['net_income'] / fund['total_assets']
fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
fund = fund.sort_values(['symbol', 'date'])
fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

print(f"  {len(prices):,} price records")
print(f"  {len(fund):,} fundamental records")
sys.stdout.flush()

def compute_compass_score(symbol, as_of_date='2025-01-01'):
    """Compute raw Compass Score for a symbol."""
    as_of_dt = pd.to_datetime(as_of_date)

    # Get fundamentals
    sym_fund = fund[(fund['symbol'] == symbol) & (fund['date'] <= as_of_dt)]
    if len(sym_fund) == 0:
        return None, {}

    latest_fund = sym_fund.sort_values('date').iloc[-1]

    # Get volatility
    sym_prices = prices[(prices['symbol'] == symbol) & (prices['date'] <= as_of_dt)]
    if len(sym_prices) < 60:
        return None, {}

    sym_prices = sym_prices.sort_values('date').tail(60)
    rets = sym_prices['close'].pct_change().dropna()
    vol = rets.std() * np.sqrt(252) * 100

    factors = {
        'roa': latest_fund['roa'],
        'ocf_assets': latest_fund['ocf_assets'],
        'fcf_assets': latest_fund['fcf_assets'],
        'gp_assets': latest_fund['gp_assets'],
        'asset_growth': latest_fund['asset_growth'],
        'vol_60d': vol,
        'market_cap': latest_fund['market_cap']
    }

    # Universe stats (from historical analysis)
    stats = {
        'roa': {'mean': 0.02, 'std': 0.15},
        'ocf_assets': {'mean': 0.05, 'std': 0.12},
        'fcf_assets': {'mean': 0.03, 'std': 0.15},
        'gp_assets': {'mean': 0.25, 'std': 0.20},
        'asset_growth': {'mean': 0.10, 'std': 0.30},
        'vol_60d': {'mean': 45, 'std': 25}
    }

    z_scores = {}
    for f in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
        val = factors[f]
        if pd.isna(val):
            return None, factors
        z_scores[f] = (val - stats[f]['mean']) / stats[f]['std']

    # Raw Compass Score
    score = (
        z_scores['roa'] * 0.20 +
        z_scores['ocf_assets'] * 0.15 +
        z_scores['fcf_assets'] * 0.15 +
        z_scores['gp_assets'] * 0.20 +
        (-z_scores['vol_60d']) * 0.15 +
        (-z_scores['asset_growth']) * 0.15
    )

    return score, factors

# ============================================================================
# COMPUTE SCORES FOR UNIVERSE
# ============================================================================

print("\nComputing scores for universe...")
sys.stdout.flush()

all_symbols = prices['symbol'].unique()
scores = []

for symbol in all_symbols:
    score, factors = compute_compass_score(symbol)
    if score is not None:
        scores.append({
            'symbol': symbol,
            'raw_score': score,
            'roa': factors.get('roa'),
            'vol': factors.get('vol_60d'),
            'market_cap': factors.get('market_cap')
        })

scores_df = pd.DataFrame(scores)
print(f"  {len(scores_df):,} stocks with valid scores")
sys.stdout.flush()

# ============================================================================
# TRANSFORM TO 0-100 SCALE
# ============================================================================

print("\n" + "=" * 70)
print("TRANSFORMATION: RAW SCORE → 0-100 PERCENTILE SCALE")
print("=" * 70)
sys.stdout.flush()

# Percentile rank transformation
scores_df['percentile'] = scores_df['raw_score'].rank(pct=True) * 100

# Grade assignment
def assign_grade(pct):
    if pct >= 85:
        return 'A (High Quality)'
    elif pct >= 60:
        return 'B (Above Average)'
    elif pct >= 40:
        return 'C (Neutral)'
    elif pct >= 20:
        return 'D (Speculative)'
    else:
        return 'F (High Risk)'

scores_df['grade'] = scores_df['percentile'].apply(assign_grade)

print("\nGrade Distribution (Full Universe):")
print("-" * 50)
grade_dist = scores_df['grade'].value_counts().sort_index()
for grade, count in grade_dist.items():
    pct = count / len(scores_df) * 100
    print(f"  {grade}: {count:,} stocks ({pct:.1f}%)")

# ============================================================================
# S&P 500 DISTRIBUTION
# ============================================================================

print("\n" + "=" * 70)
print("S&P 500 DISTRIBUTION CONTEXT")
print("=" * 70)
sys.stdout.flush()

sp500_scores = scores_df[scores_df['symbol'].isin(SP500_CORE)]
print(f"\nS&P 500 stocks found: {len(sp500_scores)}")

if len(sp500_scores) > 0:
    print("\nS&P 500 Compass Score Distribution:")
    print("-" * 50)
    print(f"  Median (50th percentile): {sp500_scores['percentile'].median():.0f}")
    print(f"  10th percentile:          {sp500_scores['percentile'].quantile(0.10):.0f}")
    print(f"  25th percentile:          {sp500_scores['percentile'].quantile(0.25):.0f}")
    print(f"  75th percentile:          {sp500_scores['percentile'].quantile(0.75):.0f}")
    print(f"  90th percentile:          {sp500_scores['percentile'].quantile(0.90):.0f}")

    print("\nS&P 500 Grade Distribution:")
    sp_grade_dist = sp500_scores['grade'].value_counts().sort_index()
    for grade, count in sp_grade_dist.items():
        pct = count / len(sp500_scores) * 100
        print(f"  {grade}: {count} stocks ({pct:.1f}%)")

# ============================================================================
# EXAMPLE STOCKS BY GRADE
# ============================================================================

print("\n" + "=" * 70)
print("EXAMPLE STOCKS BY GRADE")
print("=" * 70)
sys.stdout.flush()

# Filter to recognizable names (large caps)
large_caps = scores_df[scores_df['market_cap'] > 10e9].copy()

for grade in ['A (High Quality)', 'B (Above Average)', 'C (Neutral)', 'D (Speculative)', 'F (High Risk)']:
    print(f"\n{grade}:")
    grade_df = large_caps[large_caps['grade'] == grade].nlargest(5, 'market_cap')
    for _, row in grade_df.iterrows():
        mcap = row['market_cap'] / 1e9 if row['market_cap'] else 0
        print(f"  {row['symbol']:<8} Score: {row['percentile']:.0f}  Raw: {row['raw_score']:+.2f}  MCap: ${mcap:.0f}B")

# ============================================================================
# FAMOUS EXAMPLES
# ============================================================================

print("\n" + "=" * 70)
print("FAMOUS STOCK EXAMPLES")
print("=" * 70)
sys.stdout.flush()

famous = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM',
          'JNJ', 'WMT', 'KO', 'PG', 'XOM', 'AMD', 'SMCI', 'PLTR', 'COIN']

print(f"\n{'Symbol':<8} {'Raw Score':<12} {'Percentile':<12} {'Grade':<20} {'ROA':<10}")
print("-" * 75)

for symbol in famous:
    row = scores_df[scores_df['symbol'] == symbol]
    if len(row) > 0:
        r = row.iloc[0]
        roa_str = f"{r['roa']*100:.1f}%" if r['roa'] is not None else "N/A"
        print(f"{symbol:<8} {r['raw_score']:+.2f}        {r['percentile']:.0f}           {r['grade']:<20} {roa_str}")
    else:
        print(f"{symbol:<8} N/A")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("COMPASS SCORE 0-100 SCALE SUMMARY")
print("=" * 70)

print("""
TRANSFORMATION FORMULA:
  Raw Score → Percentile Rank → 0-100 Scale

GRADE BUCKETS:
  A (85-100): High Quality      - Strong fundamentals, low risk
  B (60-84):  Above Average     - Good quality, moderate risk
  C (40-59):  Neutral           - Average characteristics
  D (20-39):  Speculative       - Weak fundamentals or high risk
  F (0-19):   High Risk         - Poor quality, avoid

KEY CONTEXT:
  - S&P 500 median: ~65 (Above Average)
  - Most blue chips: 50-80 range
  - Meme/speculative: 0-30 range

FRAMING:
  Compass Score identifies FRAGILE BUSINESS STRUCTURES.
  It does NOT predict crashes.

  A low score means: weak profitability, high volatility,
  aggressive expansion, or some combination.

  These characteristics correlate with underperformance,
  but timing is unpredictable.
""")

print("=" * 70)
print("COMPLETED")
print("=" * 70)

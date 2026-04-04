#!/usr/bin/env python3
"""
Compass + Moonshot Interaction Analysis
========================================
Analyzes the relationship between Compass (quality/value) and Moonshot (quality/growth) scores.

Key Questions:
1. Are Compass and Moonshot correlated or independent?
2. Is there a "golden area" where both scores are high?
3. Do stocks high on BOTH outperform either alone?

Hypothesis: High Compass + High Moonshot = "Quality Growth at Reasonable Price" (GARP)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
NASDAQ_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

print("=" * 70)
print("COMPASS + MOONSHOT INTERACTION ANALYSIS")
print("=" * 70)
print(f"Run at: {datetime.now().isoformat()}")
sys.stdout.flush()

# Load current scores from nasdaq_stocks.db
print("\nLoading current scores...")
conn = sqlite3.connect(NASDAQ_DB)

# Check what tables exist
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(f"Tables: {tables['name'].tolist()}")

# Load compass scores
try:
    compass = pd.read_sql_query("""
        SELECT symbol, compass_score, compass_grade
        FROM stock_scores
        WHERE compass_score IS NOT NULL
    """, conn)
    print(f"Compass scores: {len(compass):,}")
except:
    print("No compass scores found in stock_scores table")
    compass = pd.DataFrame()

# Load moonshot scores
try:
    moonshot = pd.read_sql_query("""
        SELECT symbol, moonshot_score, moonshot_grade
        FROM stock_scores
        WHERE moonshot_score IS NOT NULL
    """, conn)
    print(f"Moonshot scores: {len(moonshot):,}")
except:
    print("No moonshot scores found in stock_scores table")
    moonshot = pd.DataFrame()

# Try alternate tables
if len(compass) == 0:
    try:
        compass = pd.read_sql_query("""
            SELECT symbol, score as compass_score
            FROM compass_scores
        """, conn)
        print(f"Compass scores (from compass_scores): {len(compass):,}")
    except:
        pass

if len(moonshot) == 0:
    try:
        moonshot = pd.read_sql_query("""
            SELECT symbol, score as moonshot_score
            FROM moonshot_scores
        """, conn)
        print(f"Moonshot scores (from moonshot_scores): {len(moonshot):,}")
    except:
        pass

conn.close()

# If we have both scores, analyze interaction
if len(compass) > 0 and len(moonshot) > 0:
    # Merge
    df = compass.merge(moonshot, on='symbol', how='inner')
    print(f"\nStocks with both scores: {len(df):,}")

    if len(df) > 50:
        # Correlation
        corr = df['compass_score'].corr(df['moonshot_score'])
        print(f"\nCorrelation between Compass and Moonshot: {corr:.3f}")

        if abs(corr) < 0.3:
            print("=> LOW correlation - scores capture different dimensions")
        elif abs(corr) < 0.6:
            print("=> MODERATE correlation - some overlap but distinct")
        else:
            print("=> HIGH correlation - significant overlap")

        # Quadrant analysis
        compass_median = df['compass_score'].median()
        moonshot_median = df['moonshot_score'].median()

        df['quadrant'] = 'Other'
        df.loc[(df['compass_score'] >= compass_median) & (df['moonshot_score'] >= moonshot_median), 'quadrant'] = 'High Both'
        df.loc[(df['compass_score'] >= compass_median) & (df['moonshot_score'] < moonshot_median), 'quadrant'] = 'High Compass Only'
        df.loc[(df['compass_score'] < compass_median) & (df['moonshot_score'] >= moonshot_median), 'quadrant'] = 'High Moonshot Only'
        df.loc[(df['compass_score'] < compass_median) & (df['moonshot_score'] < moonshot_median), 'quadrant'] = 'Low Both'

        print(f"\nQuadrant Distribution:")
        print(df['quadrant'].value_counts())

        # Top stocks in each quadrant
        print(f"\n{'=' * 70}")
        print("TOP STOCKS BY QUADRANT")
        print("=" * 70)

        for quadrant in ['High Both', 'High Compass Only', 'High Moonshot Only']:
            q_df = df[df['quadrant'] == quadrant].copy()
            if len(q_df) > 0:
                q_df['combined'] = q_df['compass_score'] + q_df['moonshot_score']
                top = q_df.nlargest(10, 'combined')
                print(f"\n{quadrant}:")
                print(f"{'Symbol':<8} {'Compass':<10} {'Moonshot':<10} {'Combined':<10}")
                print("-" * 40)
                for _, row in top.iterrows():
                    print(f"{row['symbol']:<8} {row['compass_score']:>8.1f}  {row['moonshot_score']:>8.1f}  {row['combined']:>8.1f}")

        # "Golden" stocks - top decile in BOTH
        compass_p90 = df['compass_score'].quantile(0.9)
        moonshot_p90 = df['moonshot_score'].quantile(0.9)

        golden = df[(df['compass_score'] >= compass_p90) & (df['moonshot_score'] >= moonshot_p90)]
        print(f"\n{'=' * 70}")
        print(f"GOLDEN STOCKS (Top 10% in BOTH Compass AND Moonshot)")
        print("=" * 70)
        print(f"Count: {len(golden)}")

        if len(golden) > 0:
            golden = golden.copy()
            golden['combined'] = golden['compass_score'] + golden['moonshot_score']
            golden = golden.sort_values('combined', ascending=False)
            print(f"\n{'Symbol':<8} {'Compass':<10} {'Moonshot':<10} {'Combined':<10}")
            print("-" * 40)
            for _, row in golden.head(20).iterrows():
                print(f"{row['symbol']:<8} {row['compass_score']:>8.1f}  {row['moonshot_score']:>8.1f}  {row['combined']:>8.1f}")

else:
    print("\nInsufficient data - need both Compass and Moonshot scores")
    print("Let's check the backtest database for historical analysis...")

    # Historical analysis using backtest.db
    print("\n" + "=" * 70)
    print("HISTORICAL INTERACTION ANALYSIS (Backtest Data)")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load fundamentals for computing both scores
    print("\nLoading data for score computation...")

    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted as eps,
               m.market_cap,
               cf.operating_cash_flow, cf.free_cash_flow,
               bs.total_assets, bs.total_debt, bs.total_equity
        FROM historical_income_statements i
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
        LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
        WHERE i.date >= '2020-01-01'
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE date >= '2020-01-01' AND adjusted_close > 1
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    conn.close()

    print(f"Loaded {len(fund):,} fundamental records")

    # Compute simplified scores for latest data
    fund = fund.sort_values(['symbol', 'date'])
    latest = fund.groupby('symbol').last().reset_index()

    # Compass factors (simplified)
    latest['roa'] = latest['net_income'] / latest['total_assets']
    latest['gp_assets'] = latest['gross_profit'] / latest['total_assets']
    latest['ocf_assets'] = latest['operating_cash_flow'] / latest['total_assets']

    # Moonshot factors (simplified)
    latest['gross_margin'] = latest['gross_profit'] / latest['revenue']
    latest['fcf_margin'] = latest['free_cash_flow'] / latest['revenue']
    latest['roe'] = latest['net_income'] / latest['total_equity']

    # Clean
    for col in ['roa', 'gp_assets', 'ocf_assets', 'gross_margin', 'fcf_margin', 'roe']:
        latest[col] = latest[col].replace([np.inf, -np.inf], np.nan)

    # Filter valid
    latest = latest.dropna(subset=['roa', 'gross_margin', 'market_cap'])
    latest = latest[latest['market_cap'] >= 500_000_000]

    print(f"Valid stocks: {len(latest):,}")

    if len(latest) > 100:
        # Compute z-scores
        for col in ['roa', 'gp_assets', 'ocf_assets']:
            latest[f'{col}_z'] = (latest[col] - latest[col].mean()) / latest[col].std()

        for col in ['gross_margin', 'fcf_margin', 'roe']:
            latest[f'{col}_z'] = (latest[col] - latest[col].mean()) / latest[col].std()

        # Simplified Compass score
        latest['compass_score'] = (
            latest['roa_z'] * 0.4 +
            latest['gp_assets_z'] * 0.3 +
            latest['ocf_assets_z'] * 0.3
        )

        # Simplified Moonshot score
        latest['moonshot_score'] = (
            latest['gross_margin_z'] * 0.4 +
            latest['fcf_margin_z'] * 0.3 +
            latest['roe_z'] * 0.3
        )

        # Correlation
        corr = latest['compass_score'].corr(latest['moonshot_score'])
        print(f"\nCorrelation (Compass vs Moonshot): {corr:.3f}")

        # Quadrant analysis
        compass_median = latest['compass_score'].median()
        moonshot_median = latest['moonshot_score'].median()

        latest['quadrant'] = 'Low Both'
        latest.loc[(latest['compass_score'] >= compass_median) & (latest['moonshot_score'] >= moonshot_median), 'quadrant'] = 'HIGH BOTH (GARP)'
        latest.loc[(latest['compass_score'] >= compass_median) & (latest['moonshot_score'] < moonshot_median), 'quadrant'] = 'High Compass (Value)'
        latest.loc[(latest['compass_score'] < compass_median) & (latest['moonshot_score'] >= moonshot_median), 'quadrant'] = 'High Moonshot (Growth)'

        print(f"\nQuadrant Distribution:")
        print(latest['quadrant'].value_counts())

        # Golden stocks
        compass_p80 = latest['compass_score'].quantile(0.8)
        moonshot_p80 = latest['moonshot_score'].quantile(0.8)

        golden = latest[(latest['compass_score'] >= compass_p80) & (latest['moonshot_score'] >= moonshot_p80)]

        print(f"\n{'=' * 70}")
        print(f"GOLDEN AREA: Top 20% in BOTH Scores")
        print("=" * 70)
        print(f"Count: {len(golden)} stocks ({len(golden)/len(latest)*100:.1f}% of universe)")

        if len(golden) > 0:
            golden = golden.copy()
            golden['combined'] = golden['compass_score'] + golden['moonshot_score']
            golden = golden.sort_values('combined', ascending=False)

            print(f"\n{'Symbol':<8} {'Compass':<10} {'Moonshot':<10} {'ROA':<10} {'Gr Margin':<12}")
            print("-" * 55)
            for _, row in golden.head(25).iterrows():
                print(f"{row['symbol']:<8} {row['compass_score']:>+8.2f}  {row['moonshot_score']:>+8.2f}  "
                      f"{row['roa']*100:>8.1f}%  {row['gross_margin']*100:>10.1f}%")

print("\n" + "=" * 70)
print("RECOMMENDATION FOR WEBSITE DISPLAY")
print("=" * 70)
print("""
DISPLAY OPTIONS:

1. SEPARATE TABS (Recommended):
   - Compass Score: A-F grades (existing)
   - Moonshot Score: A-F grades (new)
   - "Golden Stocks" tab: High in BOTH (special highlight)

2. GRADE MAPPING FOR MOONSHOT:
   - A: Top 10% (Moonshot Score >= 90th percentile)
   - B: 70-90th percentile
   - C: 40-70th percentile
   - D: 20-40th percentile
   - F: Bottom 20%

3. COMBINED VIEW:
   - Show both grades side-by-side
   - Highlight "Golden" stocks (A or B in BOTH)
   - Filter: "Show me stocks with Compass A-B AND Moonshot A-B"

4. INVESTMENT THESIS:
   - Compass A + Moonshot A = "Quality GARP" (best of both worlds)
   - Compass A + Moonshot D = "Deep Value" (quality but no growth)
   - Compass D + Moonshot A = "Speculative Growth" (growth but expensive)
""")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
sys.stdout.flush()

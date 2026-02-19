#!/usr/bin/env python3
"""
Sector Valuation Analysis
=========================
Analyzes EV/EBITDA distributions by sector to understand why absolute
valuation thresholds may not work and calculate sector-specific percentiles.

Key insight: "Cheap" for Tech (EV/EBITDA < 15) is very different from
"Cheap" for Utilities (EV/EBITDA < 8).
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import DATABASE_NAME, BACKTEST_DB


def analyze_current_valuations():
    """Analyze EV/EBITDA by sector in current database."""
    print("=" * 70)
    print("SECTOR VALUATION ANALYSIS - Current Data")
    print("=" * 70)

    conn = sqlite3.connect(DATABASE_NAME)

    # Get EV/EBITDA by sector
    df = pd.read_sql_query("""
        SELECT symbol, sector, ev_ebitda, value_score_v2, upside_percent
        FROM stock_consensus
        WHERE ev_ebitda IS NOT NULL
          AND ev_ebitda > 0
          AND ev_ebitda < 100
          AND sector IS NOT NULL
          AND sector != 'N/A'
    """, conn)
    conn.close()

    if df.empty:
        print("No data available")
        return

    print(f"\nTotal stocks with valid EV/EBITDA: {len(df)}")

    # Calculate sector statistics
    sector_stats = df.groupby('sector').agg({
        'ev_ebitda': ['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), 'count'],
        'symbol': 'count'
    })
    sector_stats.columns = ['median', 'mean', 'p25', 'p75', 'count', 'total']
    sector_stats = sector_stats.sort_values('median')

    print(f"\n{'Sector':<30} {'Median':>8} {'Mean':>8} {'P25':>8} {'P75':>8} {'Count':>8}")
    print("-" * 75)

    for sector, row in sector_stats.iterrows():
        print(f"{str(sector)[:30]:<30} {row['median']:>8.1f} {row['mean']:>8.1f} "
              f"{row['p25']:>8.1f} {row['p75']:>8.1f} {int(row['count']):>8}")

    # Show the problem with absolute thresholds
    print("\n" + "=" * 70)
    print("PROBLEM WITH ABSOLUTE THRESHOLDS")
    print("=" * 70)

    # Current scoring: EV/EBITDA < 12 = "cheap" (max valuation points)
    threshold = 12

    print(f"\nUsing absolute threshold: EV/EBITDA < {threshold} = 'cheap'\n")
    print(f"{'Sector':<30} {'% Cheap':>10} {'Median':>10} {'Interpretation':<25}")
    print("-" * 80)

    for sector, row in sector_stats.iterrows():
        sector_data = df[df['sector'] == sector]['ev_ebitda']
        pct_cheap = (sector_data < threshold).mean() * 100
        interpretation = "Too many cheap" if pct_cheap > 60 else "Too few cheap" if pct_cheap < 20 else "OK"
        print(f"{str(sector)[:30]:<30} {pct_cheap:>9.1f}% {row['median']:>10.1f} {interpretation:<25}")

    # Calculate sector-specific thresholds
    print("\n" + "=" * 70)
    print("RECOMMENDED SECTOR-SPECIFIC THRESHOLDS")
    print("=" * 70)
    print("(Based on sector percentiles - 'cheap' = bottom 25% of sector)\n")

    thresholds = {}
    print(f"{'Sector':<30} {'Cheap (<P25)':>12} {'Fair (P25-P75)':>15} {'Expensive (>P75)':>18}")
    print("-" * 80)

    for sector, row in sector_stats.iterrows():
        thresholds[sector] = {
            'cheap': row['p25'],
            'fair_low': row['p25'],
            'fair_high': row['p75'],
            'expensive': row['p75']
        }
        print(f"{str(sector)[:30]:<30} {row['p25']:>12.1f} {row['p25']:>7.1f} - {row['p75']:<6.1f} {row['p75']:>18.1f}")

    return thresholds


def analyze_historical_valuations():
    """Analyze EV/EBITDA by sector in historical backtest data."""
    print("\n" + "=" * 70)
    print("SECTOR VALUATION ANALYSIS - Historical Data")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Get sector mapping from main database
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute("SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    # Get historical EV/EBITDA data
    df = pd.read_sql_query("""
        SELECT symbol, date, ev_to_ebitda
        FROM historical_key_metrics
        WHERE ev_to_ebitda IS NOT NULL
          AND ev_to_ebitda > 0
          AND ev_to_ebitda < 100
    """, conn)
    conn.close()

    if df.empty:
        print("No historical data available")
        return

    df['sector'] = df['symbol'].map(sector_map)
    df = df.dropna(subset=['sector'])
    df = df[df['sector'] != 'N/A']

    print(f"\nTotal historical observations: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Calculate sector statistics over time
    sector_stats = df.groupby('sector').agg({
        'ev_to_ebitda': ['median', 'mean', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)],
        'symbol': 'nunique'
    })
    sector_stats.columns = ['median', 'mean', 'p25', 'p75', 'unique_symbols']
    sector_stats = sector_stats.sort_values('median')

    print(f"\n{'Sector':<30} {'Median':>8} {'Mean':>8} {'P25':>8} {'P75':>8} {'Symbols':>10}")
    print("-" * 75)

    for sector, row in sector_stats.iterrows():
        print(f"{str(sector)[:30]:<30} {row['median']:>8.1f} {row['mean']:>8.1f} "
              f"{row['p25']:>8.1f} {row['p75']:>8.1f} {int(row['unique_symbols']):>10}")

    return sector_stats


def generate_threshold_config():
    """Generate sector threshold configuration for scoring."""
    conn = sqlite3.connect(DATABASE_NAME)

    df = pd.read_sql_query("""
        SELECT symbol, sector, ev_ebitda
        FROM stock_consensus
        WHERE ev_ebitda IS NOT NULL
          AND ev_ebitda > 0
          AND ev_ebitda < 100
          AND sector IS NOT NULL
          AND sector != 'N/A'
    """, conn)
    conn.close()

    thresholds = {}
    for sector in df['sector'].unique():
        sector_data = df[df['sector'] == sector]['ev_ebitda']
        if len(sector_data) >= 10:
            thresholds[sector] = {
                'p25': round(sector_data.quantile(0.25), 1),
                'p50': round(sector_data.median(), 1),
                'p75': round(sector_data.quantile(0.75), 1),
            }

    print("\n" + "=" * 70)
    print("SECTOR THRESHOLD CONFIGURATION")
    print("=" * 70)
    print("\nCopy this into score_long_term_OPTIMIZED.py:\n")

    print("SECTOR_VALUATION_THRESHOLDS = {")
    for sector, vals in sorted(thresholds.items()):
        print(f"    '{sector}': {{'p25': {vals['p25']}, 'p50': {vals['p50']}, 'p75': {vals['p75']}}},")
    print("}")

    return thresholds


if __name__ == '__main__':
    current_thresholds = analyze_current_valuations()
    analyze_historical_valuations()
    generate_threshold_config()

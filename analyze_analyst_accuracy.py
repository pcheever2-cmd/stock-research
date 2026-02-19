#!/usr/bin/env python3
"""
Analyst Accuracy Analysis
=========================
Tracks historical accuracy of analyst firms by sector.
Uses this data to weight analyst recommendations in scoring.

Key metrics:
- Hit Rate: % of upgrades that led to positive returns
- Average Return: Mean return following analyst action
- Sector Specialization: Which sectors each analyst is best at

Usage:
    python analyze_analyst_accuracy.py              # Full analysis
    python analyze_analyst_accuracy.py --top 20     # Top 20 analysts only
    python analyze_analyst_accuracy.py --export     # Export accuracy to DB
"""

import sqlite3
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME


def load_grades_with_returns(conn: sqlite3.Connection, forward_days: int = 63) -> pd.DataFrame:
    """
    Load analyst grades and join with forward returns.

    Args:
        conn: Database connection
        forward_days: Days to measure forward return (default 63 = ~3 months)
    """
    print(f"Loading analyst grades with {forward_days}-day forward returns...")

    # Load grades
    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, previous_grade, new_grade, action
        FROM historical_grades
        WHERE action IN ('upgrade', 'downgrade')
        ORDER BY symbol, date
    """, conn)

    print(f"  Loaded {len(grades):,} upgrade/downgrade actions")

    # Load prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close
        FROM historical_prices
        ORDER BY symbol, date
    """, conn)

    print(f"  Loaded {len(prices):,} price records")

    # Get sector mapping
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute("SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    # Calculate forward returns for each grade
    results = []

    for symbol in grades['symbol'].unique():
        symbol_grades = grades[grades['symbol'] == symbol]
        symbol_prices = prices[prices['symbol'] == symbol].sort_values('date')

        if len(symbol_prices) < forward_days + 10:
            continue

        price_dict = dict(zip(symbol_prices['date'], symbol_prices['adjusted_close']))
        dates_list = symbol_prices['date'].tolist()

        for _, grade in symbol_grades.iterrows():
            grade_date = grade['date']

            # Find the price on grade date
            if grade_date not in price_dict:
                # Find closest date after
                close_dates = [d for d in dates_list if d >= grade_date]
                if not close_dates:
                    continue
                grade_date = close_dates[0]

            entry_price = price_dict.get(grade_date)
            if not entry_price or entry_price <= 0:
                continue

            # Find forward date
            try:
                idx = dates_list.index(grade_date)
                if idx + forward_days >= len(dates_list):
                    continue
                forward_date = dates_list[idx + forward_days]
                exit_price = price_dict.get(forward_date)
            except (ValueError, IndexError):
                continue

            if not exit_price or exit_price <= 0:
                continue

            forward_return = ((exit_price / entry_price) - 1) * 100

            results.append({
                'symbol': symbol,
                'date': grade['date'],
                'grading_company': grade['grading_company'],
                'action': grade['action'],
                'new_grade': grade['new_grade'],
                'sector': sector_map.get(symbol, 'Unknown'),
                'forward_return': forward_return,
            })

    df = pd.DataFrame(results)
    print(f"  Matched {len(df):,} grades with forward returns")
    return df


def analyze_analyst_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze accuracy of each analyst firm."""

    # For upgrades: positive return = correct
    # For downgrades: negative return = correct
    df['correct'] = np.where(
        df['action'] == 'upgrade',
        df['forward_return'] > 0,
        df['forward_return'] < 0
    )

    # Aggregate by analyst
    analyst_stats = df.groupby('grading_company').agg({
        'correct': ['sum', 'count', 'mean'],
        'forward_return': ['mean', 'std'],
        'symbol': 'nunique'
    })
    analyst_stats.columns = ['correct_calls', 'total_calls', 'hit_rate',
                              'avg_return', 'return_std', 'unique_symbols']
    analyst_stats = analyst_stats.reset_index()

    # Filter to analysts with enough data
    analyst_stats = analyst_stats[analyst_stats['total_calls'] >= 50]
    analyst_stats = analyst_stats.sort_values('hit_rate', ascending=False)

    return analyst_stats


def analyze_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze analyst accuracy broken down by sector."""

    df['correct'] = np.where(
        df['action'] == 'upgrade',
        df['forward_return'] > 0,
        df['forward_return'] < 0
    )

    # Aggregate by analyst + sector
    sector_stats = df.groupby(['grading_company', 'sector']).agg({
        'correct': ['sum', 'count', 'mean'],
        'forward_return': 'mean',
    })
    sector_stats.columns = ['correct_calls', 'total_calls', 'hit_rate', 'avg_return']
    sector_stats = sector_stats.reset_index()

    # Filter to meaningful sample sizes
    sector_stats = sector_stats[sector_stats['total_calls'] >= 20]

    return sector_stats


def find_sector_specialists(sector_stats: pd.DataFrame) -> dict:
    """Find the best analyst for each sector."""
    specialists = {}

    for sector in sector_stats['sector'].unique():
        if sector == 'Unknown' or pd.isna(sector):
            continue

        sector_data = sector_stats[sector_stats['sector'] == sector]
        if len(sector_data) == 0:
            continue

        # Best by hit rate (with minimum calls)
        best = sector_data.nlargest(1, 'hit_rate').iloc[0]
        specialists[sector] = {
            'best_analyst': best['grading_company'],
            'hit_rate': best['hit_rate'],
            'total_calls': best['total_calls'],
            'avg_return': best['avg_return'],
        }

    return specialists


def export_accuracy_to_db(analyst_stats: pd.DataFrame, sector_stats: pd.DataFrame):
    """Export accuracy scores to database for use in scoring."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Create analyst accuracy table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_accuracy (
            grading_company TEXT PRIMARY KEY,
            hit_rate REAL,
            avg_return REAL,
            total_calls INTEGER,
            unique_symbols INTEGER,
            updated_at TEXT
        )
    """)

    # Create sector accuracy table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyst_sector_accuracy (
            grading_company TEXT,
            sector TEXT,
            hit_rate REAL,
            avg_return REAL,
            total_calls INTEGER,
            updated_at TEXT,
            PRIMARY KEY (grading_company, sector)
        )
    """)

    now = datetime.now().isoformat()

    # Insert analyst accuracy
    for _, row in analyst_stats.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO analyst_accuracy
            (grading_company, hit_rate, avg_return, total_calls, unique_symbols, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row['grading_company'], row['hit_rate'], row['avg_return'],
              int(row['total_calls']), int(row['unique_symbols']), now))

    # Insert sector accuracy
    for _, row in sector_stats.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO analyst_sector_accuracy
            (grading_company, sector, hit_rate, avg_return, total_calls, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (row['grading_company'], row['sector'], row['hit_rate'],
              row['avg_return'], int(row['total_calls']), now))

    conn.commit()
    conn.close()

    print(f"\nExported {len(analyst_stats)} analyst accuracy records")
    print(f"Exported {len(sector_stats)} sector accuracy records")


def run_analysis(top_n: int = None, export: bool = False):
    """Run the full analyst accuracy analysis."""
    print("=" * 70)
    print("ANALYST ACCURACY ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load data with forward returns
    df = load_grades_with_returns(conn, forward_days=63)

    if df.empty:
        print("No data available for analysis")
        conn.close()
        return

    print(f"\nTotal analyzed: {len(df):,} analyst actions")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique analysts: {df['grading_company'].nunique()}")
    print(f"Unique sectors: {df['sector'].nunique()}")

    # Overall accuracy
    upgrades = df[df['action'] == 'upgrade']
    downgrades = df[df['action'] == 'downgrade']

    upgrade_hit_rate = (upgrades['forward_return'] > 0).mean()
    downgrade_hit_rate = (downgrades['forward_return'] < 0).mean()

    print("\n" + "=" * 70)
    print("OVERALL ACCURACY (3-Month Forward Returns)")
    print("=" * 70)
    print(f"Upgrade hit rate: {upgrade_hit_rate:.1%} ({len(upgrades):,} upgrades)")
    print(f"Downgrade hit rate: {downgrade_hit_rate:.1%} ({len(downgrades):,} downgrades)")
    print(f"Avg return after upgrade: {upgrades['forward_return'].mean():+.2f}%")
    print(f"Avg return after downgrade: {downgrades['forward_return'].mean():+.2f}%")

    # Analyst accuracy
    print("\n" + "=" * 70)
    print("TOP ANALYSTS BY ACCURACY")
    print("=" * 70)

    analyst_stats = analyze_analyst_accuracy(df)

    if top_n:
        analyst_stats = analyst_stats.head(top_n)

    print(f"\n{'Analyst':<25} {'Hit Rate':>10} {'Avg Ret':>10} {'Calls':>8} {'Symbols':>8}")
    print("-" * 65)

    for _, row in analyst_stats.head(20).iterrows():
        print(f"{row['grading_company'][:25]:<25} {row['hit_rate']:>9.1%} "
              f"{row['avg_return']:>+9.2f}% {int(row['total_calls']):>8} {int(row['unique_symbols']):>8}")

    # Worst analysts
    print("\n--- WORST ANALYSTS BY ACCURACY ---")
    for _, row in analyst_stats.tail(10).iterrows():
        print(f"{row['grading_company'][:25]:<25} {row['hit_rate']:>9.1%} "
              f"{row['avg_return']:>+9.2f}% {int(row['total_calls']):>8}")

    # Sector analysis
    print("\n" + "=" * 70)
    print("SECTOR SPECIALISTS")
    print("=" * 70)

    sector_stats = analyze_by_sector(df)
    specialists = find_sector_specialists(sector_stats)

    print(f"\n{'Sector':<25} {'Best Analyst':<25} {'Hit Rate':>10} {'Calls':>8} {'Avg Ret':>10}")
    print("-" * 85)

    for sector, info in sorted(specialists.items()):
        print(f"{sector[:25]:<25} {info['best_analyst'][:25]:<25} "
              f"{info['hit_rate']:>9.1%} {int(info['total_calls']):>8} {info['avg_return']:>+9.2f}%")

    # Show detailed sector breakdown for top analysts
    print("\n" + "=" * 70)
    print("TOP ANALYSTS - SECTOR BREAKDOWN")
    print("=" * 70)

    top_analysts = analyst_stats.head(5)['grading_company'].tolist()

    for analyst in top_analysts:
        analyst_sectors = sector_stats[sector_stats['grading_company'] == analyst]
        if len(analyst_sectors) == 0:
            continue

        print(f"\n--- {analyst} ---")
        analyst_sectors = analyst_sectors.sort_values('hit_rate', ascending=False)

        for _, row in analyst_sectors.head(5).iterrows():
            print(f"  {row['sector'][:20]:<20} Hit: {row['hit_rate']:>6.1%}  "
                  f"Ret: {row['avg_return']:>+6.2f}%  ({int(row['total_calls'])} calls)")

    # Export if requested
    if export:
        export_accuracy_to_db(analyst_stats, sector_stats)

    conn.close()

    # Summary recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    avg_hit_rate = analyst_stats['hit_rate'].mean()
    best_analyst = analyst_stats.iloc[0]
    worst_analyst = analyst_stats.iloc[-1]

    print(f"""
Key Findings:
- Average analyst hit rate: {avg_hit_rate:.1%}
- Best analyst: {best_analyst['grading_company']} ({best_analyst['hit_rate']:.1%} hit rate)
- Worst analyst: {worst_analyst['grading_company']} ({worst_analyst['hit_rate']:.1%} hit rate)

Scoring Recommendations:
1. Weight upgrades from high-accuracy analysts more heavily
2. Use sector-specific accuracy when available
3. Consider contrarian signal for consistently wrong analysts (<45% hit rate)

To integrate into scoring:
- Run with --export to save accuracy data to database
- Update score_long_term_OPTIMIZED.py to query analyst_accuracy table
- Boost/penalize scores based on analyst accuracy
""")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze analyst accuracy')
    parser.add_argument('--top', type=int, help='Show only top N analysts')
    parser.add_argument('--export', action='store_true', help='Export accuracy to database')
    args = parser.parse_args()

    run_analysis(top_n=args.top, export=args.export)

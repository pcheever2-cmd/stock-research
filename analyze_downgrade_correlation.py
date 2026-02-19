#!/usr/bin/env python3
"""
Downgrade Correlation Analysis
==============================
Checks whether downgrades from top 3 analysts per sector correlate
with negative forward returns (i.e., are they right about sells too?)

Key questions:
1. Do top 3 analyst downgrades predict negative returns?
2. How does downgrade hit rate compare to upgrade hit rate?
3. Should we weight downgrades differently in V3?
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME
from analyst_accuracy_scorer import SECTOR_TOP3_ANALYSTS

def analyze_downgrade_accuracy():
    """Analyze whether top 3 analyst downgrades predict negative returns."""

    print("=" * 70)
    print("DOWNGRADE CORRELATION ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Get sector mapping
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute("SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    # Load grades with forward returns (63-day / 3 months)
    print("\nLoading downgrade data with 63-day forward returns...")

    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE action = 'downgrade'
        ORDER BY symbol, date
    """, conn)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close
        FROM historical_prices
        ORDER BY symbol, date
    """, conn)

    conn.close()

    grades['sector'] = grades['symbol'].map(sector_map)
    grades = grades.dropna(subset=['sector'])

    print(f"  Total downgrades: {len(grades):,}")

    # Build set of top 3 analysts per sector for fast lookup
    top3_set = {}
    for sector, analysts in SECTOR_TOP3_ANALYSTS.items():
        top3_set[sector] = {name for name, _ in analysts}

    # Mark if downgrade is from top 3 analyst
    def is_top3(row):
        sector = row['sector']
        if sector in top3_set:
            return row['grading_company'] in top3_set[sector]
        return False

    grades['is_top3'] = grades.apply(is_top3, axis=1)

    top3_downgrades = grades[grades['is_top3']]
    other_downgrades = grades[~grades['is_top3']]

    print(f"  Top 3 analyst downgrades: {len(top3_downgrades):,}")
    print(f"  Other analyst downgrades: {len(other_downgrades):,}")

    # Calculate forward returns for downgrades
    forward_days = 63
    results_top3 = []
    results_other = []

    def calculate_returns(grades_subset, results_list):
        for symbol in grades_subset['symbol'].unique():
            symbol_grades = grades_subset[grades_subset['symbol'] == symbol]
            symbol_prices = prices[prices['symbol'] == symbol].sort_values('date')

            if len(symbol_prices) < forward_days + 10:
                continue

            price_dict = dict(zip(symbol_prices['date'], symbol_prices['adjusted_close']))
            dates_list = symbol_prices['date'].tolist()

            for _, grade in symbol_grades.iterrows():
                grade_date = grade['date']

                if grade_date not in price_dict:
                    close_dates = [d for d in dates_list if d >= grade_date]
                    if not close_dates:
                        continue
                    grade_date = close_dates[0]

                entry_price = price_dict.get(grade_date)
                if not entry_price or entry_price <= 0:
                    continue

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

                results_list.append({
                    'symbol': symbol,
                    'date': grade['date'],
                    'grading_company': grade['grading_company'],
                    'sector': grade['sector'],
                    'forward_return': forward_return,
                    'correct': forward_return < 0,  # Downgrade correct if price dropped
                })

    print("\nCalculating forward returns for top 3 downgrades...")
    calculate_returns(top3_downgrades, results_top3)

    print("Calculating forward returns for other downgrades...")
    calculate_returns(other_downgrades, results_other)

    df_top3 = pd.DataFrame(results_top3)
    df_other = pd.DataFrame(results_other)

    # Overall downgrade accuracy
    print("\n" + "=" * 70)
    print("DOWNGRADE ACCURACY COMPARISON")
    print("=" * 70)

    if len(df_top3) > 0:
        top3_hit_rate = df_top3['correct'].mean()
        top3_avg_return = df_top3['forward_return'].mean()
        print(f"\n🌟 TOP 3 ANALYSTS (per sector):")
        print(f"   Downgrade Hit Rate: {top3_hit_rate:.1%} ({len(df_top3):,} downgrades)")
        print(f"   Avg Forward Return: {top3_avg_return:+.2f}%")
        print(f"   (Negative return = correct downgrade)")

    if len(df_other) > 0:
        other_hit_rate = df_other['correct'].mean()
        other_avg_return = df_other['forward_return'].mean()
        print(f"\n📊 ALL OTHER ANALYSTS:")
        print(f"   Downgrade Hit Rate: {other_hit_rate:.1%} ({len(df_other):,} downgrades)")
        print(f"   Avg Forward Return: {other_avg_return:+.2f}%")

    # Sector breakdown for top 3
    print("\n" + "=" * 70)
    print("TOP 3 DOWNGRADE ACCURACY BY SECTOR")
    print("=" * 70)

    if len(df_top3) > 0:
        sector_stats = df_top3.groupby('sector').agg({
            'correct': ['mean', 'count'],
            'forward_return': 'mean'
        })
        sector_stats.columns = ['hit_rate', 'count', 'avg_return']
        sector_stats = sector_stats.sort_values('hit_rate', ascending=False)

        print(f"\n{'Sector':<25} {'Hit Rate':>10} {'Avg Return':>12} {'Count':>8}")
        print("-" * 60)

        for sector, row in sector_stats.iterrows():
            print(f"{str(sector)[:25]:<25} {row['hit_rate']:>9.1%} "
                  f"{row['avg_return']:>+11.2f}% {int(row['count']):>8}")

    # Individual top 3 analyst breakdown
    print("\n" + "=" * 70)
    print("TOP 3 ANALYST DOWNGRADE ACCURACY (Individual)")
    print("=" * 70)

    if len(df_top3) > 0:
        analyst_stats = df_top3.groupby('grading_company').agg({
            'correct': ['mean', 'count'],
            'forward_return': 'mean'
        })
        analyst_stats.columns = ['hit_rate', 'count', 'avg_return']
        analyst_stats = analyst_stats.sort_values('hit_rate', ascending=False)

        print(f"\n{'Analyst':<30} {'Hit Rate':>10} {'Avg Return':>12} {'Count':>8}")
        print("-" * 65)

        for analyst, row in analyst_stats.iterrows():
            print(f"{str(analyst)[:30]:<30} {row['hit_rate']:>9.1%} "
                  f"{row['avg_return']:>+11.2f}% {int(row['count']):>8}")

    # Compare upgrade vs downgrade accuracy for top 3
    print("\n" + "=" * 70)
    print("UPGRADE vs DOWNGRADE COMPARISON")
    print("=" * 70)

    # Load upgrade data for comparison
    conn = sqlite3.connect(BACKTEST_DB)
    upgrades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE action = 'upgrade'
        ORDER BY symbol, date
    """, conn)
    conn.close()

    upgrades['sector'] = upgrades['symbol'].map(sector_map)
    upgrades = upgrades.dropna(subset=['sector'])
    upgrades['is_top3'] = upgrades.apply(is_top3, axis=1)

    top3_upgrades = upgrades[upgrades['is_top3']]
    results_upgrade = []

    print("\nCalculating forward returns for top 3 upgrades...")
    for symbol in top3_upgrades['symbol'].unique():
        symbol_grades = top3_upgrades[top3_upgrades['symbol'] == symbol]
        symbol_prices = prices[prices['symbol'] == symbol].sort_values('date')

        if len(symbol_prices) < forward_days + 10:
            continue

        price_dict = dict(zip(symbol_prices['date'], symbol_prices['adjusted_close']))
        dates_list = symbol_prices['date'].tolist()

        for _, grade in symbol_grades.iterrows():
            grade_date = grade['date']

            if grade_date not in price_dict:
                close_dates = [d for d in dates_list if d >= grade_date]
                if not close_dates:
                    continue
                grade_date = close_dates[0]

            entry_price = price_dict.get(grade_date)
            if not entry_price or entry_price <= 0:
                continue

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

            results_upgrade.append({
                'forward_return': forward_return,
                'correct': forward_return > 0,  # Upgrade correct if price rose
            })

    df_upgrade = pd.DataFrame(results_upgrade)

    if len(df_upgrade) > 0 and len(df_top3) > 0:
        upgrade_hit = df_upgrade['correct'].mean()
        upgrade_ret = df_upgrade['forward_return'].mean()
        downgrade_hit = df_top3['correct'].mean()
        downgrade_ret = df_top3['forward_return'].mean()

        print(f"\n{'Action':<15} {'Hit Rate':>12} {'Avg Return':>12} {'Count':>10}")
        print("-" * 55)
        print(f"{'Upgrades':<15} {upgrade_hit:>11.1%} {upgrade_ret:>+11.2f}% {len(df_upgrade):>10,}")
        print(f"{'Downgrades':<15} {downgrade_hit:>11.1%} {downgrade_ret:>+11.2f}% {len(df_top3):>10,}")

        # Summary
        print("\n" + "=" * 70)
        print("RECOMMENDATIONS FOR V3 SCORING")
        print("=" * 70)

        if downgrade_hit >= 0.55:
            print(f"""
✅ TOP 3 DOWNGRADES ARE PREDICTIVE: {downgrade_hit:.1%} hit rate

Recommendations:
1. Include downgrade signal in V3 scoring
2. Weight downgrades from top 3 analysts as negative signal
3. Downgrade from top 3 → reduce score by similar magnitude as upgrade boost
""")
        elif downgrade_hit >= 0.45:
            print(f"""
⚠️  TOP 3 DOWNGRADES ARE MARGINAL: {downgrade_hit:.1%} hit rate (near 50%)

Recommendations:
1. Still include downgrade signal but with reduced weight
2. Only use strong downgrade signal (multiple top 3 downgrades)
3. Consider ignoring single downgrades
""")
        else:
            print(f"""
❌ TOP 3 DOWNGRADES NOT PREDICTIVE: {downgrade_hit:.1%} hit rate

Recommendations:
1. Do NOT use downgrade signal in V3
2. Only use upgrade signal from top 3 analysts
3. May want to use downgrade as contrarian signal if < 40%
""")


if __name__ == '__main__':
    analyze_downgrade_accuracy()

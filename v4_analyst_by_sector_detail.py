#!/usr/bin/env python3
"""
Analyst Signal Performance by Sector
=====================================
Analyze whether analyst signal decay is uniform across sectors or driven by specific ones.
Also test if sector-level sentiment affects individual signal effectiveness.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

FORWARD_DAYS = 63


def load_data():
    """Load all required data."""
    conn = sqlite3.connect(BACKTEST_DB)

    print("Loading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    print("Loading analyst grades...")
    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE action IN ('upgrade', 'downgrade')
        ORDER BY date
    """, conn)
    grades['date'] = pd.to_datetime(grades['date'])

    conn.close()

    # Load sectors
    print("Loading sectors...")
    conn_main = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query("""
        SELECT DISTINCT symbol, sector
        FROM stock_consensus
        WHERE sector IS NOT NULL AND sector != ''
    """, conn_main)
    conn_main.close()

    return prices, grades, sectors


def compute_grade_outcomes(grades, prices, sector_map):
    """Compute forward returns for all grades."""
    print("\nComputing forward returns for all grades...")

    grades_with_sector = grades.merge(sector_map, on='symbol', how='left')
    grades_with_sector = grades_with_sector.dropna(subset=['sector'])

    prices_by_symbol = {sym: grp.sort_values('date')[['date', 'close']].values.tolist()
                        for sym, grp in prices.groupby('symbol')}

    results = []
    total = len(grades_with_sector)

    for i, (_, grade) in enumerate(grades_with_sector.iterrows()):
        if i % 20000 == 0:
            print(f"  {i:,}/{total:,}...")

        symbol = grade['symbol']
        grade_date = grade['date']

        if symbol not in prices_by_symbol:
            continue

        price_list = prices_by_symbol[symbol]

        entry_idx = None
        for idx, (d, p) in enumerate(price_list):
            if d >= grade_date:
                entry_idx = idx
                break

        if entry_idx is None:
            continue

        exit_idx = entry_idx + FORWARD_DAYS
        if exit_idx >= len(price_list):
            continue

        entry_price = price_list[entry_idx][1]
        exit_price = price_list[exit_idx][1]

        if entry_price <= 0:
            continue

        fwd_return = ((exit_price / entry_price) - 1) * 100

        results.append({
            'date': grade_date,
            'year': grade_date.year,
            'grading_company': grade['grading_company'],
            'sector': grade['sector'],
            'symbol': symbol,
            'action': grade['action'],
            'fwd_return': fwd_return,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("ANALYST SIGNAL PERFORMANCE BY SECTOR")
    print("Are certain sectors driving the signal decay?")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    prices, grades, sectors = load_data()
    sector_map = sectors[['symbol', 'sector']].drop_duplicates()

    grade_outcomes = compute_grade_outcomes(grades, prices, sector_map)
    print(f"\nTotal grade outcomes: {len(grade_outcomes):,}")

    # Filter to recent periods for analysis
    test_periods = [
        ('2018-2019', 2018, 2019),
        ('2020-2021', 2020, 2021),
        ('2022-2023', 2022, 2023),
        ('2024-2025', 2024, 2025),
    ]

    # Get unique sectors
    all_sectors = sorted(grade_outcomes['sector'].unique())
    print(f"\nSectors: {all_sectors}")

    # =========================================
    # 1. SECTOR-LEVEL ANALYST SPREAD BY PERIOD
    # =========================================
    print("\n" + "=" * 70)
    print("1. UPGRADE/DOWNGRADE SPREAD BY SECTOR AND PERIOD")
    print("=" * 70)

    sector_period_results = []

    for period_name, year_start, year_end in test_periods:
        period_data = grade_outcomes[
            (grade_outcomes['year'] >= year_start) &
            (grade_outcomes['year'] <= year_end)
        ]

        print(f"\n--- {period_name} ({len(period_data):,} grades) ---")
        print(f"{'Sector':<25} {'Upgrades':>10} {'Avg Ret':>10} {'Downgrades':>10} {'Avg Ret':>10} {'Spread':>10}")
        print("-" * 80)

        for sector in all_sectors:
            sector_data = period_data[period_data['sector'] == sector]

            upgrades = sector_data[sector_data['action'] == 'upgrade']
            downgrades = sector_data[sector_data['action'] == 'downgrade']

            if len(upgrades) < 20 or len(downgrades) < 20:
                continue

            up_ret = upgrades['fwd_return'].mean()
            down_ret = downgrades['fwd_return'].mean()
            spread = up_ret - down_ret

            print(f"{sector:<25} {len(upgrades):>10,} {up_ret:>+9.2f}% {len(downgrades):>10,} {down_ret:>+9.2f}% {spread:>+9.2f}%")

            sector_period_results.append({
                'period': period_name,
                'sector': sector,
                'n_upgrades': len(upgrades),
                'n_downgrades': len(downgrades),
                'upgrade_ret': up_ret,
                'downgrade_ret': down_ret,
                'spread': spread,
            })

    # Convert to DataFrame
    sp_df = pd.DataFrame(sector_period_results)

    # =========================================
    # 2. WHICH SECTORS STILL HAVE SIGNAL?
    # =========================================
    print("\n" + "=" * 70)
    print("2. SECTORS WITH CONSISTENT VS DEGRADED SIGNAL")
    print("=" * 70)

    # Calculate average spread by sector across periods
    sector_avg = sp_df.groupby('sector').agg({
        'spread': ['mean', 'std', 'count'],
        'n_upgrades': 'sum',
        'n_downgrades': 'sum',
    })
    sector_avg.columns = ['avg_spread', 'spread_std', 'n_periods', 'total_upgrades', 'total_downgrades']
    sector_avg = sector_avg.reset_index()
    sector_avg = sector_avg.sort_values('avg_spread', ascending=False)

    print(f"\n{'Sector':<25} {'Avg Spread':>12} {'Std Dev':>10} {'Periods':>10} {'Signal':>12}")
    print("-" * 75)

    for _, row in sector_avg.iterrows():
        signal_status = "✓ STRONG" if row['avg_spread'] > 2.0 else ("~ Weak" if row['avg_spread'] > 0 else "✗ REVERSED")
        print(f"{row['sector']:<25} {row['avg_spread']:>+11.2f}% {row['spread_std']:>9.2f}% {int(row['n_periods']):>10} {signal_status:>12}")

    # =========================================
    # 3. SECTOR SENTIMENT VS SIGNAL EFFECTIVENESS
    # =========================================
    print("\n" + "=" * 70)
    print("3. SECTOR SENTIMENT VS SIGNAL EFFECTIVENESS")
    print("Does bearish sector sentiment make even upgrades fail?")
    print("=" * 70)

    # For each sector-period, compute:
    # - Upgrade ratio (upgrades / total grades) as sentiment proxy
    # - Whether upgrades still work

    sentiment_results = []

    for period_name, year_start, year_end in test_periods:
        period_data = grade_outcomes[
            (grade_outcomes['year'] >= year_start) &
            (grade_outcomes['year'] <= year_end)
        ]

        for sector in all_sectors:
            sector_data = period_data[period_data['sector'] == sector]

            if len(sector_data) < 50:
                continue

            n_upgrades = (sector_data['action'] == 'upgrade').sum()
            n_downgrades = (sector_data['action'] == 'downgrade').sum()
            total = n_upgrades + n_downgrades

            upgrade_ratio = n_upgrades / total if total > 0 else 0.5

            upgrades = sector_data[sector_data['action'] == 'upgrade']
            downgrades = sector_data[sector_data['action'] == 'downgrade']

            if len(upgrades) >= 10 and len(downgrades) >= 10:
                up_ret = upgrades['fwd_return'].mean()
                down_ret = downgrades['fwd_return'].mean()
                spread = up_ret - down_ret

                sentiment_results.append({
                    'period': period_name,
                    'sector': sector,
                    'upgrade_ratio': upgrade_ratio,
                    'upgrade_ret': up_ret,
                    'downgrade_ret': down_ret,
                    'spread': spread,
                    'sentiment': 'Bullish' if upgrade_ratio > 0.55 else ('Bearish' if upgrade_ratio < 0.45 else 'Neutral'),
                })

    sent_df = pd.DataFrame(sentiment_results)

    # Group by sentiment
    print("\nSignal effectiveness by sector sentiment:")
    print(f"\n{'Sentiment':<12} {'Obs':>8} {'Avg Up Ret':>12} {'Avg Down Ret':>14} {'Avg Spread':>12} {'Signal Works?':>14}")
    print("-" * 75)

    for sentiment in ['Bullish', 'Neutral', 'Bearish']:
        subset = sent_df[sent_df['sentiment'] == sentiment]
        if len(subset) > 0:
            avg_up = subset['upgrade_ret'].mean()
            avg_down = subset['downgrade_ret'].mean()
            avg_spread = subset['spread'].mean()
            works = "YES" if avg_spread > 1.0 else ("Marginal" if avg_spread > 0 else "NO")
            print(f"{sentiment:<12} {len(subset):>8} {avg_up:>+11.2f}% {avg_down:>+13.2f}% {avg_spread:>+11.2f}% {works:>14}")

    # =========================================
    # 4. RECENT PERIOD DEEP DIVE (2024-2025)
    # =========================================
    print("\n" + "=" * 70)
    print("4. RECENT PERIOD DEEP DIVE (2024-2025)")
    print("Which sectors have working vs broken signals?")
    print("=" * 70)

    recent = grade_outcomes[grade_outcomes['year'] >= 2024]
    print(f"\nRecent grades: {len(recent):,}")

    print(f"\n{'Sector':<25} {'Upgrades':>10} {'Up Ret':>10} {'Downgrades':>10} {'Down Ret':>10} {'Spread':>10} {'Status':>12}")
    print("-" * 95)

    recent_results = []
    for sector in all_sectors:
        sector_data = recent[recent['sector'] == sector]

        upgrades = sector_data[sector_data['action'] == 'upgrade']
        downgrades = sector_data[sector_data['action'] == 'downgrade']

        if len(upgrades) < 15 or len(downgrades) < 15:
            continue

        up_ret = upgrades['fwd_return'].mean()
        down_ret = downgrades['fwd_return'].mean()
        spread = up_ret - down_ret

        status = "✓ WORKS" if spread > 1.5 else ("~ Weak" if spread > 0 else "✗ BROKEN")

        print(f"{sector:<25} {len(upgrades):>10,} {up_ret:>+9.2f}% {len(downgrades):>10,} {down_ret:>+9.2f}% {spread:>+9.2f}% {status:>12}")

        recent_results.append({
            'sector': sector,
            'spread': spread,
            'status': status,
        })

    # =========================================
    # 5. TOP ANALYSTS BY SECTOR (Recent)
    # =========================================
    print("\n" + "=" * 70)
    print("5. TOP ANALYSTS BY SECTOR (2022-2025)")
    print("Which analysts still have edge in each sector?")
    print("=" * 70)

    recent_2yr = grade_outcomes[grade_outcomes['year'] >= 2022]

    for sector in ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical']:
        sector_data = recent_2yr[recent_2yr['sector'] == sector]

        if len(sector_data) < 100:
            continue

        # Calculate hit rate by analyst
        analyst_stats = []
        for analyst in sector_data['grading_company'].unique():
            analyst_grades = sector_data[sector_data['grading_company'] == analyst]

            if len(analyst_grades) < 10:
                continue

            # Hit rate: upgrades that went up, downgrades that went down
            upgrades = analyst_grades[analyst_grades['action'] == 'upgrade']
            downgrades = analyst_grades[analyst_grades['action'] == 'downgrade']

            up_hits = (upgrades['fwd_return'] > 0).sum() if len(upgrades) > 0 else 0
            down_hits = (downgrades['fwd_return'] < 0).sum() if len(downgrades) > 0 else 0

            total_hits = up_hits + down_hits
            total_calls = len(analyst_grades)
            hit_rate = total_hits / total_calls if total_calls > 0 else 0

            analyst_stats.append({
                'analyst': analyst,
                'hit_rate': hit_rate,
                'n_calls': total_calls,
                'avg_return': analyst_grades['fwd_return'].mean(),
            })

        if not analyst_stats:
            continue

        stats_df = pd.DataFrame(analyst_stats)
        stats_df = stats_df[stats_df['n_calls'] >= 15]  # Minimum sample
        stats_df = stats_df.sort_values('hit_rate', ascending=False)

        print(f"\n--- {sector} ---")
        print(f"{'Analyst':<30} {'Hit Rate':>10} {'N Calls':>10} {'Avg Ret':>10}")
        print("-" * 65)

        for _, row in stats_df.head(5).iterrows():
            print(f"{row['analyst']:<30} {row['hit_rate']:>9.1%} {int(row['n_calls']):>10} {row['avg_return']:>+9.2f}%")

        if len(stats_df) > 5:
            print("...")
            for _, row in stats_df.tail(3).iterrows():
                print(f"{row['analyst']:<30} {row['hit_rate']:>9.1%} {int(row['n_calls']):>10} {row['avg_return']:>+9.2f}%")

    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY FINDINGS")
    print("=" * 70)

    # Count sectors by signal status
    if recent_results:
        working = sum(1 for r in recent_results if '✓' in r['status'])
        weak = sum(1 for r in recent_results if '~' in r['status'])
        broken = sum(1 for r in recent_results if '✗' in r['status'])

        print(f"""
1. SECTOR-LEVEL SIGNAL STATUS (2024-2025):
   - Working (spread > 1.5%): {working} sectors
   - Weak (0 < spread < 1.5%): {weak} sectors
   - Broken (spread < 0): {broken} sectors

2. SENTIMENT EFFECT:
""")

        for sentiment in ['Bullish', 'Neutral', 'Bearish']:
            subset = sent_df[sent_df['sentiment'] == sentiment]
            if len(subset) > 0:
                avg_spread = subset['spread'].mean()
                print(f"   - {sentiment} sectors: {avg_spread:+.2f}% avg spread")

        # Best and worst sectors
        working_sectors = [r['sector'] for r in recent_results if '✓' in r['status']]
        broken_sectors = [r['sector'] for r in recent_results if '✗' in r['status']]

        print(f"""
3. ACTIONABLE INSIGHTS:
   - SECTORS WHERE ANALYSTS STILL ADD VALUE: {', '.join(working_sectors) if working_sectors else 'None'}
   - SECTORS TO IGNORE ANALYST SIGNALS: {', '.join(broken_sectors) if broken_sectors else 'None'}

4. RECOMMENDATION:
   - Consider using analyst signals ONLY in sectors where they historically work
   - Ignore or use contrarian signals in "broken" sectors
   - Monitor sector sentiment - bearish sectors may need different treatment
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

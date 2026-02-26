#!/usr/bin/env python3
"""
V4 + Rolling Top Analyst by Sector
==================================
Continuously re-evaluates which analysts are performing best.
Top analysts are re-computed monthly based on trailing 2-year performance.
This captures analyst skill decay/improvement over time.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Settings
ROLLING_WINDOW_YEARS = 2  # Years of trailing data for analyst ranking
FORWARD_DAYS = 63  # 3-month forward return for accuracy calc
MIN_CALLS_PER_SECTOR = 8  # Min calls to rank an analyst in a sector
TOP_N_PER_SECTOR = 5  # Top N analysts per sector
RECOMPUTE_FREQ_DAYS = 30  # Recompute rankings monthly


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
    print(f"  {len(prices):,} price records")

    print("Loading fundamentals...")
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        WHERE m.market_cap IS NOT NULL AND m.market_cap > 0
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    print("Loading analyst grades...")
    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE action IN ('upgrade', 'downgrade')
        ORDER BY date
    """, conn)
    grades['date'] = pd.to_datetime(grades['date'])
    print(f"  {len(grades):,} analyst grades")

    conn.close()

    # Load sectors
    print("Loading sectors...")
    try:
        conn_main = sqlite3.connect(MAIN_DB)
        sectors = pd.read_sql_query("""
            SELECT DISTINCT symbol, sector
            FROM stock_consensus
            WHERE sector IS NOT NULL AND sector != ''
        """, conn_main)
        conn_main.close()
    except:
        sectors = pd.DataFrame(columns=['symbol', 'sector'])
    print(f"  {len(sectors):,} symbol-sector mappings")

    return prices, fund, grades, sectors


def precompute_forward_returns(grades, prices, sector_map):
    """
    Precompute forward returns for all grades to speed up rolling calculations.
    """
    print("\nPrecomputing forward returns for all grades...")

    # Add sector to grades
    grades_with_sector = grades.merge(sector_map, on='symbol', how='left')
    grades_with_sector = grades_with_sector.dropna(subset=['sector'])

    # Create price lookup dict for speed
    prices_by_symbol = {sym: grp.sort_values('date')[['date', 'close']].values.tolist()
                        for sym, grp in prices.groupby('symbol')}

    results = []
    total = len(grades_with_sector)

    for i, (_, grade) in enumerate(grades_with_sector.iterrows()):
        if i % 10000 == 0:
            print(f"  {i:,}/{total:,} grades processed...")

        symbol = grade['symbol']
        grade_date = grade['date']

        if symbol not in prices_by_symbol:
            continue

        price_list = prices_by_symbol[symbol]

        # Find entry price
        entry_idx = None
        for idx, (d, p) in enumerate(price_list):
            if d >= grade_date:
                entry_idx = idx
                break

        if entry_idx is None:
            continue

        # Find exit price (63 days forward)
        exit_idx = entry_idx + FORWARD_DAYS
        if exit_idx >= len(price_list):
            continue

        entry_price = price_list[entry_idx][1]
        exit_price = price_list[exit_idx][1]

        if entry_price <= 0:
            continue

        fwd_return = ((exit_price / entry_price) - 1) * 100

        is_upgrade = grade['action'] == 'upgrade'
        correct = (fwd_return > 0) if is_upgrade else (fwd_return < 0)

        results.append({
            'date': grade_date,
            'grading_company': grade['grading_company'],
            'sector': grade['sector'],
            'symbol': symbol,
            'action': grade['action'],
            'fwd_return': fwd_return,
            'correct': correct,
        })

    print(f"  Computed {len(results):,} grade outcomes")
    return pd.DataFrame(results)


def compute_rolling_top_analysts(grade_outcomes, as_of_date, training_years=2):
    """
    Compute top analysts per sector using grades from the trailing window.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    train_end = as_of_dt - timedelta(days=FORWARD_DAYS)  # Gap for forward return
    train_start = as_of_dt - timedelta(days=training_years * 365)

    # Filter to training window
    mask = (grade_outcomes['date'] >= train_start) & (grade_outcomes['date'] <= train_end)
    window_grades = grade_outcomes[mask]

    if window_grades.empty:
        return {}

    # Aggregate by analyst + sector
    stats = window_grades.groupby(['grading_company', 'sector']).agg({
        'correct': ['sum', 'count', 'mean'],
    })
    stats.columns = ['correct_calls', 'n_calls', 'hit_rate']
    stats = stats.reset_index()

    # Filter to analysts with enough calls
    stats = stats[stats['n_calls'] >= MIN_CALLS_PER_SECTOR]

    # Get top N per sector
    sector_top = {}
    for sector in stats['sector'].unique():
        sector_stats = stats[stats['sector'] == sector].nlargest(TOP_N_PER_SECTOR, 'hit_rate')
        sector_top[sector] = dict(zip(sector_stats['grading_company'], sector_stats['hit_rate']))

    return sector_top


def main():
    print("=" * 70)
    print("V4 + ROLLING TOP ANALYST BY SECTOR")
    print("Analyst rankings re-computed monthly based on trailing 2yr performance")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    prices, fund, grades, sectors = load_data()
    sector_map = sectors[['symbol', 'sector']].drop_duplicates()

    # Precompute grade outcomes (forward returns)
    grade_outcomes = precompute_forward_returns(grades, prices, sector_map)

    # Compute fundamental factors
    print("\n" + "=" * 70)
    print("COMPUTING FACTORS")
    print("=" * 70)

    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
        lower = fund[col].quantile(0.01)
        upper = fund[col].quantile(0.99)
        fund[col] = fund[col].clip(lower, upper)

    # Sample symbols
    print("\nSampling symbols with grades and sectors...")
    np.random.seed(42)
    symbols_with_grades = set(grades['symbol'].unique())
    symbols_with_sectors = set(sector_map['symbol'].unique())
    valid_symbols = list(symbols_with_grades & symbols_with_sectors)
    sample_symbols = np.random.choice(valid_symbols, size=min(1200, len(valid_symbols)), replace=False)
    print(f"  Using {len(sample_symbols)} symbols")

    prices_subset = prices[prices['symbol'].isin(sample_symbols)]
    grades_subset = grades[grades['symbol'].isin(sample_symbols)]

    # Generate observations
    print("\nGenerating observations...")
    observations = []

    for i, symbol in enumerate(sample_symbols):
        if i % 300 == 0:
            print(f"  {i}/{len(sample_symbols)} symbols...")

        sym_prices = prices_subset[prices_subset['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values

        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            if date.year < 2017:
                continue

            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            observations.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'vol_60d': vol_60d,
            })

    df = pd.DataFrame(observations)
    print(f"  Generated {len(df):,} observations")

    # Add sector
    df = df.merge(sector_map, on='symbol', how='left')
    df = df.dropna(subset=['sector'])

    # Merge fundamentals
    print("\nMerging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']
    fund_subset = fund[fund_cols].dropna(subset=['roa'])

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund_subset[fund_subset['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)
    df = df.dropna(subset=['fwd_3m', 'roa', 'sector'])
    print(f"  Clean observations: {len(df):,}")

    # Get unique months for rolling computation
    df['year_month'] = df['date'].dt.to_period('M')
    unique_months = sorted(df['year_month'].unique())
    print(f"\n  Unique months to process: {len(unique_months)}")

    # Cache for rolling top analysts (recompute monthly)
    analyst_cache = {}

    # Compute rolling analyst signal for each observation
    print("\n" + "=" * 70)
    print("COMPUTING ROLLING ANALYST SIGNALS")
    print("(Re-ranking analysts each month based on trailing 2yr performance)")
    print("=" * 70)

    signals = []
    has_signal = []
    last_recompute = None

    for i, (_, row) in enumerate(df.iterrows()):
        if i % 5000 == 0:
            print(f"  {i:,}/{len(df):,} observations...")

        obs_date = row['date']
        obs_month = obs_date.to_period('M')
        symbol = row['symbol']
        sector = row['sector']

        # Recompute analyst rankings monthly
        if obs_month not in analyst_cache:
            sector_top = compute_rolling_top_analysts(
                grade_outcomes, obs_date, training_years=ROLLING_WINDOW_YEARS
            )
            analyst_cache[obs_month] = sector_top

        sector_top = analyst_cache[obs_month]
        top_analysts = sector_top.get(sector, {})

        # Get recent grades for this symbol from top analysts
        lookback = obs_date - timedelta(days=90)
        recent_grades = grades_subset[
            (grades_subset['symbol'] == symbol) &
            (grades_subset['date'] >= lookback) &
            (grades_subset['date'] <= obs_date)
        ]

        signal = 0
        has_top_signal = False

        for _, grade in recent_grades.iterrows():
            analyst = grade['grading_company']
            action = grade['action']

            if analyst in top_analysts:
                hit_rate = top_analysts[analyst]
                # Weight by accuracy above baseline (0.53)
                weight = (hit_rate - 0.53) * 10 + 1

                if action == 'upgrade':
                    signal += weight
                else:
                    signal -= weight

                has_top_signal = True

        signals.append(signal)
        has_signal.append(has_top_signal)

    df['rolling_analyst_signal'] = signals
    df['has_rolling_signal'] = has_signal

    # Analyze by period
    test_periods = [
        ('2017-2019', '2017-01-01', '2019-12-31'),
        ('2020', '2020-01-01', '2020-12-31'),
        ('2021-2022', '2021-01-01', '2022-12-31'),
        ('2023-2024', '2023-01-01', '2024-12-31'),
        ('2025+', '2025-01-01', '2026-12-31'),
    ]

    print("\n" + "=" * 70)
    print("RESULTS BY PERIOD")
    print("=" * 70)

    all_results = []

    for period_name, start_date, end_date in test_periods:
        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        if len(period_df) < 500:
            print(f"\n{period_name}: Skipping (only {len(period_df)} obs)")
            continue

        print(f"\n{'='*60}")
        print(f"PERIOD: {period_name}")
        print(f"{'='*60}")

        # V4 z-scores
        for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
            mean = period_df[col].mean()
            std = period_df[col].std()
            period_df[f'{col}_z'] = ((period_df[col] - mean) / std).clip(-3, 3).fillna(0)

        period_df['v4_score'] = (
            period_df['roa_z'] * 0.20 +
            period_df['ocf_assets_z'] * 0.15 +
            period_df['fcf_assets_z'] * 0.15 +
            period_df['gp_assets_z'] * 0.20 +  # Novy-Marx
            (-period_df['vol_60d_z']) * 0.15 +
            (-period_df['asset_growth_z']) * 0.15
        )

        n_with_signal = period_df['has_rolling_signal'].sum()
        pct_with_signal = n_with_signal / len(period_df) * 100

        print(f"  Observations: {len(period_df):,}")
        print(f"  With rolling top-analyst signal: {n_with_signal:,} ({pct_with_signal:.1f}%)")

        # Filter to observations with signals
        with_signal = period_df[period_df['has_rolling_signal']].copy()

        if len(with_signal) < 100:
            print(f"  Not enough signals for analysis")
            continue

        # Analyst signal correlation
        analyst_corr = with_signal['rolling_analyst_signal'].corr(with_signal['fwd_3m'])
        print(f"  Rolling analyst signal correlation: {analyst_corr:+.4f}")

        # Upgrade vs Downgrade
        upgraded = with_signal[with_signal['rolling_analyst_signal'] > 0]
        downgraded = with_signal[with_signal['rolling_analyst_signal'] < 0]

        print(f"\n  Rolling Top-Analyst Signals:")
        print(f"    Upgraded:   {len(upgraded):>5,}, Avg Return: {upgraded['fwd_3m'].mean():+.2f}%")
        print(f"    Downgraded: {len(downgraded):>5,}, Avg Return: {downgraded['fwd_3m'].mean():+.2f}%")

        if len(upgraded) > 30 and len(downgraded) > 30:
            analyst_spread = upgraded['fwd_3m'].mean() - downgraded['fwd_3m'].mean()
            print(f"    Spread:     {analyst_spread:+.2f}%")
        else:
            analyst_spread = np.nan

        # V4 baseline
        try:
            period_df['v4_quintile'] = pd.qcut(period_df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        v4_q5 = period_df[period_df['v4_quintile'] == 'Q5']['fwd_3m'].mean()
        v4_q1 = period_df[period_df['v4_quintile'] == 'Q1']['fwd_3m'].mean()
        v4_spread = v4_q5 - v4_q1

        print(f"\n  V4 Only (all obs): Spread = {v4_spread:+.2f}%")

        # Hybrid on signals subset
        if len(with_signal) >= 200:
            try:
                with_signal['v4_quintile'] = pd.qcut(with_signal['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except:
                continue

            v4_subset_q5 = with_signal[with_signal['v4_quintile'] == 'Q5']['fwd_3m'].mean()
            v4_subset_q1 = with_signal[with_signal['v4_quintile'] == 'Q1']['fwd_3m'].mean()
            v4_subset_spread = v4_subset_q5 - v4_subset_q1

            # Normalize analyst signal
            if with_signal['rolling_analyst_signal'].std() > 0:
                with_signal['analyst_z'] = (
                    (with_signal['rolling_analyst_signal'] - with_signal['rolling_analyst_signal'].mean()) /
                    with_signal['rolling_analyst_signal'].std()
                ).clip(-3, 3).fillna(0)
            else:
                with_signal['analyst_z'] = 0

            print(f"\n  Hybrid test on signal subset ({len(with_signal):,} obs):")
            print(f"    V4 Only: Spread = {v4_subset_spread:+.2f}%")

            best_spread = v4_subset_spread
            best_weight = 0

            for weight in [0.05, 0.10, 0.15, 0.20]:
                with_signal['hybrid'] = with_signal['v4_score'] + with_signal['analyst_z'] * weight

                try:
                    with_signal['hybrid_quintile'] = pd.qcut(with_signal['hybrid'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
                except:
                    continue

                hyb_q5 = with_signal[with_signal['hybrid_quintile'] == 'Q5']['fwd_3m'].mean()
                hyb_q1 = with_signal[with_signal['hybrid_quintile'] == 'Q1']['fwd_3m'].mean()
                hyb_spread = hyb_q5 - hyb_q1
                change = hyb_spread - v4_subset_spread

                print(f"    Hybrid ({weight*100:.0f}%): Spread = {hyb_spread:+.2f}%, Change = {change:+.2f}%")

                if hyb_spread > best_spread:
                    best_spread = hyb_spread
                    best_weight = weight

            all_results.append({
                'period': period_name,
                'n_obs': len(period_df),
                'n_with_signal': n_with_signal,
                'pct_with_signal': pct_with_signal,
                'analyst_corr': analyst_corr,
                'analyst_spread': analyst_spread,
                'v4_spread': v4_subset_spread,
                'best_hybrid_spread': best_spread,
                'improvement': best_spread - v4_subset_spread,
            })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: ROLLING TOP-ANALYST PERFORMANCE")
    print("(Analyst rankings updated monthly based on trailing 2yr accuracy)")
    print("=" * 70)

    print(f"\n{'Period':<12} {'Obs':>8} {'% Signal':>10} {'Analyst r':>12} {'Up/Down Spr':>12} {'V4 Spread':>12} {'Hybrid':>10} {'Improve':>10}")
    print("-" * 100)

    for r in all_results:
        analyst_spr = f"{r['analyst_spread']:+.2f}%" if not np.isnan(r['analyst_spread']) else "n/a"
        print(f"{r['period']:<12} {r['n_obs']:>8,} {r['pct_with_signal']:>9.1f}% {r['analyst_corr']:>+11.4f} "
              f"{analyst_spr:>12} {r['v4_spread']:>+11.2f}% {r['best_hybrid_spread']:>+9.2f}% {r['improvement']:>+9.2f}%")

    if all_results:
        avg_analyst_corr = np.mean([r['analyst_corr'] for r in all_results])
        avg_analyst_spread = np.nanmean([r['analyst_spread'] for r in all_results])
        avg_improvement = np.mean([r['improvement'] for r in all_results])
        periods_improved = sum(1 for r in all_results if r['improvement'] > 0)

        print(f"\n*** ROLLING vs STATIC COMPARISON ***")
        print(f"  Average analyst correlation:   {avg_analyst_corr:+.4f}")
        print(f"  Average upgrade/downgrade spread: {avg_analyst_spread:+.2f}%")
        print(f"  Average hybrid improvement:    {avg_improvement:+.2f}%")
        print(f"  Periods improved: {periods_improved}/{len(all_results)}")

        # Check analyst turnover
        print(f"\n*** ANALYST RANKING STABILITY ***")
        print(f"  Unique months with rankings: {len(analyst_cache)}")

        # Sample a few months to show turnover
        sample_months = list(analyst_cache.keys())[:3] + list(analyst_cache.keys())[-3:]
        print(f"\n  Sample top analysts for Technology sector:")
        for month in sample_months:
            if 'Technology' in analyst_cache[month]:
                top = list(analyst_cache[month]['Technology'].items())[:3]
                top_str = ', '.join([f"{a}:{r:.0%}" for a, r in top])
                print(f"    {month}: {top_str}")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

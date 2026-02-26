#!/usr/bin/env python3
"""
V4 + Top Analyst by Sector
==========================
Test using TOP analysts within each sector/industry.
Sector specialists should have better predictive power than overall accuracy.
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

# Settings
ROLLING_WINDOW_YEARS = 2
FORWARD_DAYS = 63
MIN_CALLS_PER_SECTOR = 8  # Minimum calls in a sector to be ranked
TOP_N_PER_SECTOR = 5  # Top 5 analysts per sector


def load_data():
    """Load all required data."""
    conn = sqlite3.connect(BACKTEST_DB)

    print("Loading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
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
        ORDER BY grading_company, symbol, date
    """, conn)
    grades['date'] = pd.to_datetime(grades['date'])
    print(f"  {len(grades):,} analyst grades")

    conn.close()

    # Load sectors from main DB
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


def compute_rolling_sector_top_analysts(grades_df, prices_df, sector_map, as_of_date, training_years=2):
    """
    Compute top N analysts per sector using ONLY data before as_of_date.
    Returns dict: sector -> {analyst: hit_rate}
    """
    as_of_dt = pd.to_datetime(as_of_date)
    train_end = as_of_dt - timedelta(days=FORWARD_DAYS)
    train_start = as_of_dt - timedelta(days=training_years * 365)

    # Filter grades to training window
    train_grades = grades_df[
        (grades_df['date'] >= train_start) &
        (grades_df['date'] <= train_end)
    ].copy()

    if train_grades.empty:
        return {}

    # Add sector
    train_grades = train_grades.merge(sector_map, on='symbol', how='left')
    train_grades = train_grades.dropna(subset=['sector'])

    if train_grades.empty:
        return {}

    # Get prices for forward return calculation
    price_window = prices_df[
        (prices_df['date'] >= train_start) &
        (prices_df['date'] <= as_of_dt)
    ].copy()

    # Calculate accuracy for each analyst-sector combo
    results = []
    symbols = train_grades['symbol'].unique()

    for symbol in symbols:
        sym_grades = train_grades[train_grades['symbol'] == symbol]
        sym_prices = price_window[price_window['symbol'] == symbol].sort_values('date')

        if len(sym_prices) < FORWARD_DAYS + 10:
            continue

        price_list = sym_prices[['date', 'close']].values.tolist()

        for _, grade in sym_grades.iterrows():
            grade_date = grade['date']

            # Find entry index
            entry_idx = None
            for i, (d, p) in enumerate(price_list):
                if d >= grade_date:
                    entry_idx = i
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

            is_upgrade = grade['action'] == 'upgrade'
            correct = (fwd_return > 0) if is_upgrade else (fwd_return < 0)

            results.append({
                'grading_company': grade['grading_company'],
                'sector': grade['sector'],
                'correct': correct,
            })

    if not results:
        return {}

    df = pd.DataFrame(results)

    # Aggregate by analyst + sector
    stats = df.groupby(['grading_company', 'sector']).agg({
        'correct': ['sum', 'count', 'mean'],
    })
    stats.columns = ['correct_calls', 'n_calls', 'hit_rate']
    stats = stats.reset_index()

    # Filter to analysts with enough calls in sector
    stats = stats[stats['n_calls'] >= MIN_CALLS_PER_SECTOR]

    # Get top N per sector
    sector_top = {}
    for sector in stats['sector'].unique():
        sector_stats = stats[stats['sector'] == sector].nlargest(TOP_N_PER_SECTOR, 'hit_rate')
        sector_top[sector] = dict(zip(sector_stats['grading_company'], sector_stats['hit_rate']))

    return sector_top


def compute_sector_analyst_signal(symbol, sector, grades_df, sector_top_analysts, as_of_date, lookback_days=90):
    """
    Compute analyst signal using ONLY top sector analysts.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    cutoff = as_of_dt - timedelta(days=lookback_days)

    recent_grades = grades_df[
        (grades_df['symbol'] == symbol) &
        (grades_df['date'] >= cutoff) &
        (grades_df['date'] <= as_of_dt)
    ]

    if recent_grades.empty:
        return 0, 0, False

    top_analysts = sector_top_analysts.get(sector, {})

    if not top_analysts:
        return 0, 0, False

    score = 0
    n_top_signals = 0
    has_top_signal = False

    for _, grade in recent_grades.iterrows():
        analyst = grade['grading_company']
        action = grade['action']

        if analyst in top_analysts:
            hit_rate = top_analysts[analyst]
            # Weight by how much better than baseline (0.53)
            weight = (hit_rate - 0.53) * 10 + 1  # 0.63 hit rate -> weight = 2.0

            if action == 'upgrade':
                score += weight
            elif action == 'downgrade':
                score -= weight

            n_top_signals += 1
            has_top_signal = True

    return score, n_top_signals, has_top_signal


def main():
    print("=" * 70)
    print("V4 + TOP ANALYST BY SECTOR")
    print("Using top 5 analysts within each sector for signals")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    prices, fund, grades, sectors = load_data()
    sector_map = sectors[['symbol', 'sector']].drop_duplicates()

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

    # Winsorize
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
        lower = fund[col].quantile(0.01)
        upper = fund[col].quantile(0.99)
        fund[col] = fund[col].clip(lower, upper)

    # Sample symbols with both grades and sectors
    print("\nSampling symbols with grades and sectors...")
    np.random.seed(42)
    symbols_with_grades = set(grades['symbol'].unique())
    symbols_with_sectors = set(sector_map['symbol'].unique())
    valid_symbols = list(symbols_with_grades & symbols_with_sectors)
    sample_symbols = np.random.choice(valid_symbols, size=min(1500, len(valid_symbols)), replace=False)
    print(f"  Using {len(sample_symbols)} symbols")

    prices = prices[prices['symbol'].isin(sample_symbols)]
    grades = grades[grades['symbol'].isin(sample_symbols)]

    # Generate observations
    print("\nGenerating observations...")
    observations = []

    for i, symbol in enumerate(sample_symbols):
        if i % 300 == 0:
            print(f"  {i}/{len(sample_symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values

        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            if date.year < 2017:  # Need history for analyst accuracy
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

    # Show sector distribution
    print("\nSector distribution:")
    print(df['sector'].value_counts().head(10))

    # Test periods
    test_periods = [
        ('2017-2019', '2017-01-01', '2019-12-31'),
        ('2020', '2020-01-01', '2020-12-31'),
        ('2021-2022', '2021-01-01', '2022-12-31'),
        ('2023-2024', '2023-01-01', '2024-12-31'),
        ('2025+', '2025-01-01', '2026-12-31'),
    ]

    print("\n" + "=" * 70)
    print("COMPUTING V4 + SECTOR-TOP ANALYST SCORES")
    print("=" * 70)

    all_results = []

    for period_name, start_date, end_date in test_periods:
        print(f"\n{'='*60}")
        print(f"PERIOD: {period_name}")
        print(f"{'='*60}")

        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        if len(period_df) < 1000:
            print(f"  Skipping: only {len(period_df)} observations")
            continue

        # Compute sector-top analysts
        print("  Computing sector-top analysts...")
        sector_top = compute_rolling_sector_top_analysts(
            grades, prices, sector_map, start_date, training_years=ROLLING_WINDOW_YEARS
        )

        print(f"  Sectors with top analysts: {len(sector_top)}")
        for sector, analysts in list(sector_top.items())[:3]:
            print(f"    {sector}: {list(analysts.keys())[:3]}")

        # V4 z-scores
        for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
            mean = period_df[col].mean()
            std = period_df[col].std()
            period_df[f'{col}_z'] = ((period_df[col] - mean) / std).clip(-3, 3).fillna(0)

        period_df['v4_score'] = (
            period_df['roa_z'] * 0.20 +
            period_df['ocf_assets_z'] * 0.15 +
            period_df['fcf_assets_z'] * 0.15 +
            period_df['gp_assets_z'] * 0.10 +
            (-period_df['vol_60d_z']) * 0.15 +
            (-period_df['asset_growth_z']) * 0.15
        )

        # Compute sector analyst signals
        print("  Computing sector-top analyst signals...")
        signals = []
        n_signals = []
        has_signal = []

        for _, row in period_df.iterrows():
            sig, n, has = compute_sector_analyst_signal(
                row['symbol'], row['sector'], grades, sector_top,
                row['date'], lookback_days=90
            )
            signals.append(sig)
            n_signals.append(n)
            has_signal.append(has)

        period_df['sector_analyst_signal'] = signals
        period_df['n_sector_signals'] = n_signals
        period_df['has_sector_signal'] = has_signal

        # Stats
        n_with_signal = period_df['has_sector_signal'].sum()
        pct_with_signal = n_with_signal / len(period_df) * 100

        print(f"\n  Observations: {len(period_df):,}")
        print(f"  With sector-top analyst signal: {n_with_signal:,} ({pct_with_signal:.1f}%)")

        if n_with_signal < 100:
            print(f"  Not enough signals for analysis")
            continue

        # Correlation of sector analyst signal
        with_signal = period_df[period_df['has_sector_signal']].copy()
        analyst_corr = with_signal['sector_analyst_signal'].corr(with_signal['fwd_3m'])
        print(f"  Sector analyst signal correlation: {analyst_corr:+.4f}")

        # V4 baseline (on all data)
        try:
            period_df['v4_quintile'] = pd.qcut(period_df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            print(f"  Could not create quintiles")
            continue

        v4_q5 = period_df[period_df['v4_quintile'] == 'Q5']['fwd_3m'].mean()
        v4_q1 = period_df[period_df['v4_quintile'] == 'Q1']['fwd_3m'].mean()
        v4_spread = v4_q5 - v4_q1

        print(f"\n  V4 Only (all): Spread = {v4_spread:+.2f}%")

        # Split by analyst signal direction (only where signal exists)
        upgraded = with_signal[with_signal['sector_analyst_signal'] > 0]
        downgraded = with_signal[with_signal['sector_analyst_signal'] < 0]

        print(f"\n  Sector-Top Analyst Signals:")
        print(f"    Upgraded: {len(upgraded):,}, Avg Return: {upgraded['fwd_3m'].mean():+.2f}%")
        print(f"    Downgraded: {len(downgraded):,}, Avg Return: {downgraded['fwd_3m'].mean():+.2f}%")

        if len(upgraded) > 50 and len(downgraded) > 50:
            analyst_spread = upgraded['fwd_3m'].mean() - downgraded['fwd_3m'].mean()
            print(f"    Upgrade/Downgrade Spread: {analyst_spread:+.2f}%")

        # Test hybrid on stocks WITH signals
        print(f"\n  Hybrid test on stocks with sector signals ({len(with_signal):,} obs):")

        # V4 on this subset
        try:
            with_signal['v4_quintile'] = pd.qcut(with_signal['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        v4_subset_q5 = with_signal[with_signal['v4_quintile'] == 'Q5']['fwd_3m'].mean()
        v4_subset_q1 = with_signal[with_signal['v4_quintile'] == 'Q1']['fwd_3m'].mean()
        v4_subset_spread = v4_subset_q5 - v4_subset_q1

        # Normalize analyst signal
        if with_signal['sector_analyst_signal'].std() > 0:
            with_signal['analyst_z'] = (
                (with_signal['sector_analyst_signal'] - with_signal['sector_analyst_signal'].mean()) /
                with_signal['sector_analyst_signal'].std()
            ).clip(-3, 3).fillna(0)
        else:
            with_signal['analyst_z'] = 0

        # Test weights
        print(f"    V4 Only: Spread = {v4_subset_spread:+.2f}%")

        best_spread = v4_subset_spread
        best_weight = 0

        for weight in [0.05, 0.10, 0.15, 0.20, 0.25]:
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
            'v4_spread': v4_subset_spread,
            'best_hybrid_spread': best_spread,
            'best_weight': best_weight,
            'improvement': best_spread - v4_subset_spread,
            'analyst_spread': analyst_spread if len(upgraded) > 50 and len(downgraded) > 50 else np.nan,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SECTOR-TOP ANALYST PERFORMANCE")
    print("=" * 70)

    print(f"\n{'Period':<12} {'Obs':>8} {'% Signal':>10} {'Analyst r':>12} {'Analyst Spr':>12} {'V4 Spread':>12} {'Hybrid':>10} {'Improve':>10}")
    print("-" * 100)

    for r in all_results:
        analyst_spr = f"{r['analyst_spread']:+.2f}%" if not np.isnan(r['analyst_spread']) else "n/a"
        print(f"{r['period']:<12} {r['n_obs']:>8,} {r['pct_with_signal']:>9.1f}% {r['analyst_corr']:>+11.4f} "
              f"{analyst_spr:>12} {r['v4_spread']:>+11.2f}% {r['best_hybrid_spread']:>+9.2f}% {r['improvement']:>+9.2f}%")

    # Overall
    if all_results:
        avg_analyst_corr = np.mean([r['analyst_corr'] for r in all_results])
        avg_improvement = np.mean([r['improvement'] for r in all_results])
        avg_analyst_spread = np.nanmean([r['analyst_spread'] for r in all_results])
        periods_improved = sum(1 for r in all_results if r['improvement'] > 0)

        print(f"\n*** OVERALL FINDINGS ***")
        print(f"  Average sector-analyst correlation: {avg_analyst_corr:+.4f}")
        print(f"  Average upgrade/downgrade spread: {avg_analyst_spread:+.2f}%")
        print(f"  Average hybrid improvement: {avg_improvement:+.2f}%")
        print(f"  Periods improved: {periods_improved}/{len(all_results)}")

        if avg_analyst_spread > 2.0:
            print("\n  FINDING: Sector-top analysts provide MEANINGFUL predictive signal!")
            print("  Consider using for:")
            print("    - Tiebreaker when V4 scores are similar")
            print("    - Confidence boost for positions")
            print("    - Avoid stocks downgraded by sector-top analysts")
        elif avg_analyst_spread > 0:
            print("\n  FINDING: Sector-top analysts provide MARGINAL predictive value.")
        else:
            print("\n  FINDING: Sector-top analysts do not add predictive value.")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
V4 + Analyst Hybrid Testing V2
==============================
Broader analyst signal: ALL upgrades/downgrades weighted by analyst accuracy.
This version uses full analyst coverage, not just top-3 sector analysts.
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
MIN_CALLS_FOR_ACCURACY = 10


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


def compute_rolling_analyst_accuracy(grades_df, prices_df, as_of_date, training_years=2):
    """
    Compute analyst accuracy using ONLY data before as_of_date.
    Returns dict: analyst -> hit_rate
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

    # Get prices for forward return calculation
    price_window = prices_df[
        (prices_df['date'] >= train_start) &
        (prices_df['date'] <= as_of_dt)
    ].copy()

    # Calculate accuracy for each analyst
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
                'correct': correct,
            })

    if not results:
        return {}

    df = pd.DataFrame(results)

    # Aggregate by analyst
    stats = df.groupby('grading_company').agg({
        'correct': ['sum', 'count', 'mean'],
    })
    stats.columns = ['correct_calls', 'n_calls', 'hit_rate']
    stats = stats.reset_index()

    # Filter to analysts with enough calls
    stats = stats[stats['n_calls'] >= MIN_CALLS_FOR_ACCURACY]

    # Create dict
    accuracy = dict(zip(stats['grading_company'], stats['hit_rate']))

    return accuracy


def compute_analyst_signal(symbol, grades_df, analyst_accuracy, as_of_date, lookback_days=90):
    """
    Compute weighted analyst signal for a symbol.
    Upgrades weighted by accuracy, downgrades weighted negatively.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    cutoff = as_of_dt - timedelta(days=lookback_days)

    recent_grades = grades_df[
        (grades_df['symbol'] == symbol) &
        (grades_df['date'] >= cutoff) &
        (grades_df['date'] <= as_of_dt)
    ]

    if recent_grades.empty:
        return 0, 0

    baseline = 0.53  # Average analyst accuracy
    score = 0
    n_signals = 0

    for _, grade in recent_grades.iterrows():
        analyst = grade['grading_company']
        action = grade['action']

        # Get analyst accuracy (use baseline if unknown)
        accuracy = analyst_accuracy.get(analyst, baseline)

        # Convert accuracy to weight (above-average analysts get more weight)
        weight = accuracy / baseline  # 1.0 = average, 1.2 = 20% better than average

        if action == 'upgrade':
            score += weight
            n_signals += 1
        elif action == 'downgrade':
            score -= weight
            n_signals += 1

    return score, n_signals


def main():
    print("=" * 70)
    print("V4 + ANALYST HYBRID V2 (Broader Signal)")
    print("Using ALL analyst upgrades/downgrades weighted by accuracy")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    prices, fund, grades, sectors = load_data()

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

    # Sample symbols that have analyst grades
    print("\nSampling 1500 symbols with analyst coverage...")
    np.random.seed(42)
    symbols_with_grades = grades['symbol'].unique()
    sample_symbols = np.random.choice(symbols_with_grades, size=min(1500, len(symbols_with_grades)), replace=False)
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
        volume = sym_prices['volume'].values

        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            if date.year < 2015:  # Start from 2015 for better analyst data
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
    df = df.dropna(subset=['fwd_3m', 'roa'])
    print(f"  Clean observations: {len(df):,}")

    # Test periods
    test_periods = [
        ('2015-2017', '2015-01-01', '2017-12-31'),
        ('2018-2019', '2018-01-01', '2019-12-31'),
        ('2020', '2020-01-01', '2020-12-31'),
        ('2021-2022', '2021-01-01', '2022-12-31'),
        ('2023-2024', '2023-01-01', '2024-12-31'),
        ('2025+', '2025-01-01', '2026-12-31'),
    ]

    print("\n" + "=" * 70)
    print("COMPUTING V4 + ANALYST SCORES BY PERIOD")
    print("=" * 70)

    all_results = []

    for period_name, start_date, end_date in test_periods:
        print(f"\n--- {period_name} ---")

        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        if len(period_df) < 500:
            print(f"  Skipping: only {len(period_df)} observations")
            continue

        # Compute rolling analyst accuracy as of period start
        print("  Computing rolling analyst accuracy...")
        analyst_accuracy = compute_rolling_analyst_accuracy(
            grades, prices, start_date, training_years=ROLLING_WINDOW_YEARS
        )
        print(f"  Tracked {len(analyst_accuracy)} analysts with accuracy data")

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

        # Compute analyst signals
        print("  Computing analyst signals...")
        signals = []
        n_signals_list = []
        for _, row in period_df.iterrows():
            sig, n = compute_analyst_signal(
                row['symbol'], grades, analyst_accuracy,
                row['date'], lookback_days=90
            )
            signals.append(sig)
            n_signals_list.append(n)

        period_df['analyst_signal'] = signals
        period_df['n_analyst_signals'] = n_signals_list

        # Normalize analyst signal
        if period_df['analyst_signal'].std() > 0:
            period_df['analyst_z'] = (
                (period_df['analyst_signal'] - period_df['analyst_signal'].mean()) /
                period_df['analyst_signal'].std()
            ).clip(-3, 3).fillna(0)
        else:
            period_df['analyst_z'] = 0

        # Different hybrid weights to test
        weights_to_test = [0.05, 0.10, 0.15, 0.20]

        n_with_signal = (period_df['analyst_signal'] != 0).sum()
        pct_with_signal = n_with_signal / len(period_df) * 100
        analyst_corr = period_df['analyst_signal'].corr(period_df['fwd_3m'])

        print(f"  Observations: {len(period_df):,}")
        print(f"  With analyst signal: {n_with_signal:,} ({pct_with_signal:.1f}%)")
        print(f"  Analyst signal correlation: {analyst_corr:+.4f}")

        # V4 only baseline
        try:
            period_df['v4_quintile'] = pd.qcut(period_df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            print(f"  Could not create quintiles")
            continue

        v4_q5 = period_df[period_df['v4_quintile'] == 'Q5']['fwd_3m'].mean()
        v4_q1 = period_df[period_df['v4_quintile'] == 'Q1']['fwd_3m'].mean()
        v4_spread = v4_q5 - v4_q1
        v4_corr = period_df['v4_score'].corr(period_df['fwd_3m'])

        print(f"\n  V4 Only: Spread = {v4_spread:+.2f}%, Corr = {v4_corr:+.4f}")

        best_hybrid_spread = v4_spread
        best_weight = 0

        for weight in weights_to_test:
            period_df['hybrid'] = period_df['v4_score'] + period_df['analyst_z'] * weight

            try:
                period_df['hybrid_quintile'] = pd.qcut(period_df['hybrid'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            except:
                continue

            hyb_q5 = period_df[period_df['hybrid_quintile'] == 'Q5']['fwd_3m'].mean()
            hyb_q1 = period_df[period_df['hybrid_quintile'] == 'Q1']['fwd_3m'].mean()
            hyb_spread = hyb_q5 - hyb_q1
            improvement = hyb_spread - v4_spread

            print(f"  Hybrid ({weight*100:.0f}% analyst): Spread = {hyb_spread:+.2f}%, Change = {improvement:+.2f}%")

            if hyb_spread > best_hybrid_spread:
                best_hybrid_spread = hyb_spread
                best_weight = weight

        all_results.append({
            'period': period_name,
            'n_obs': len(period_df),
            'pct_with_signal': pct_with_signal,
            'analyst_corr': analyst_corr,
            'v4_spread': v4_spread,
            'best_hybrid_spread': best_hybrid_spread,
            'best_weight': best_weight,
            'improvement': best_hybrid_spread - v4_spread,
        })

    # Focus on stocks WITH analyst coverage
    print("\n" + "=" * 70)
    print("FOCUS: STOCKS WITH ANALYST COVERAGE")
    print("Testing on subset where analyst signal != 0")
    print("=" * 70)

    for period_name, start_date, end_date in test_periods:
        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        if len(period_df) < 500:
            continue

        # Get analyst accuracy and signals
        analyst_accuracy = compute_rolling_analyst_accuracy(
            grades, prices, start_date, training_years=ROLLING_WINDOW_YEARS
        )

        signals = []
        for _, row in period_df.iterrows():
            sig, _ = compute_analyst_signal(
                row['symbol'], grades, analyst_accuracy,
                row['date'], lookback_days=90
            )
            signals.append(sig)

        period_df['analyst_signal'] = signals

        # Filter to stocks with analyst coverage
        covered = period_df[period_df['analyst_signal'] != 0].copy()

        if len(covered) < 200:
            continue

        print(f"\n--- {period_name}: {len(covered):,} obs with analyst coverage ---")

        # V4 z-scores on covered subset
        for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
            mean = covered[col].mean()
            std = covered[col].std()
            covered[f'{col}_z'] = ((covered[col] - mean) / std).clip(-3, 3).fillna(0)

        covered['v4_score'] = (
            covered['roa_z'] * 0.20 +
            covered['ocf_assets_z'] * 0.15 +
            covered['fcf_assets_z'] * 0.15 +
            covered['gp_assets_z'] * 0.10 +
            (-covered['vol_60d_z']) * 0.15 +
            (-covered['asset_growth_z']) * 0.15
        )

        # Analyst signal z-score
        covered['analyst_z'] = (
            (covered['analyst_signal'] - covered['analyst_signal'].mean()) /
            covered['analyst_signal'].std()
        ).clip(-3, 3).fillna(0)

        # Split by analyst signal direction
        upgraded = covered[covered['analyst_signal'] > 0]
        downgraded = covered[covered['analyst_signal'] < 0]

        print(f"  Upgraded: {len(upgraded):,}, Avg Return: {upgraded['fwd_3m'].mean():+.2f}%")
        print(f"  Downgraded: {len(downgraded):,}, Avg Return: {downgraded['fwd_3m'].mean():+.2f}%")

        if len(upgraded) > 50 and len(downgraded) > 50:
            spread = upgraded['fwd_3m'].mean() - downgraded['fwd_3m'].mean()
            print(f"  Upgrade/Downgrade Spread: {spread:+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Period':<12} {'Obs':>8} {'% Signal':>10} {'Analyst r':>12} {'V4 Spread':>12} {'Best Hybrid':>12} {'Improve':>10}")
    print("-" * 90)

    for r in all_results:
        print(f"{r['period']:<12} {r['n_obs']:>8,} {r['pct_with_signal']:>9.1f}% {r['analyst_corr']:>+11.4f} "
              f"{r['v4_spread']:>+11.2f}% {r['best_hybrid_spread']:>+11.2f}% {r['improvement']:>+9.2f}%")

    # Overall assessment
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    periods_improved = sum(1 for r in all_results if r['improvement'] > 0)
    avg_analyst_corr = np.mean([r['analyst_corr'] for r in all_results])

    print(f"\n*** OVERALL ASSESSMENT ***")
    print(f"  Average analyst correlation: {avg_analyst_corr:+.4f}")
    print(f"  Average spread improvement: {avg_improvement:+.2f}%")
    print(f"  Periods improved: {periods_improved}/{len(all_results)}")

    if avg_analyst_corr > 0.02 and avg_improvement > 0.3:
        print("\n  RECOMMENDATION: Analyst signals add value. Consider 10-15% weight.")
    elif avg_analyst_corr > 0 or avg_improvement > 0:
        print("\n  RECOMMENDATION: Marginal value. Use for tiebreakers only.")
    else:
        print("\n  RECOMMENDATION: Analyst signals do not improve V4.")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

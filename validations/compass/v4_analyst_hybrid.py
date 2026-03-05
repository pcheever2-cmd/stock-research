#!/usr/bin/env python3
"""
V4 + Analyst Hybrid Testing
============================
Test whether adding top-3 analyst signals can improve V4 performance,
especially during periods like 2020 when quality factors underperformed.

Uses rolling analyst accuracy to avoid look-ahead bias.
Data available from 2011-12 onward.
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

# V4 factor weights (sum to 100%)
V4_WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,  # Novy-Marx gross profitability premium
    'vol_60d': -0.15,  # Lower volatility is better
    'asset_growth': -0.15,  # Lower asset growth is better
}

# Settings
MIN_ANALYST_CALLS = 10  # Minimum calls for analyst to be ranked
ROLLING_WINDOW_YEARS = 3  # Years of history for analyst ranking
FORWARD_DAYS = 63  # 3-month forward return


def load_data():
    """Load price and fundamental data."""
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

    print("Loading sectors from main database...")
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

    print("Loading analyst grades...")
    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE action IN ('upgrade', 'downgrade')
        ORDER BY symbol, date
    """, conn)
    grades['date'] = pd.to_datetime(grades['date'])
    print(f"  {len(grades):,} analyst grades")

    conn.close()

    return prices, fund, sectors, grades


def compute_rolling_top3_analysts(grades_df, prices_df, sector_map, as_of_date,
                                   training_years=3, forward_days=63):
    """
    Compute top 3 analysts per sector using ONLY data before as_of_date.
    No look-ahead bias.
    """
    as_of_dt = pd.to_datetime(as_of_date)
    train_end = as_of_dt - timedelta(days=forward_days)
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

    # Get prices for forward return calculation
    price_window = prices_df[
        (prices_df['date'] >= train_start) &
        (prices_df['date'] <= as_of_dt)
    ].copy()

    # Calculate forward returns for each grade
    results = []
    for symbol in train_grades['symbol'].unique():
        sym_grades = train_grades[train_grades['symbol'] == symbol]
        sym_prices = price_window[price_window['symbol'] == symbol].sort_values('date')

        if len(sym_prices) < forward_days + 10:
            continue

        price_list = sym_prices[['date', 'close']].values.tolist()
        date_to_idx = {row[0]: i for i, row in enumerate(price_list)}

        for _, grade in sym_grades.iterrows():
            grade_date = grade['date']

            # Find entry price
            entry_idx = None
            for i, (d, p) in enumerate(price_list):
                if d >= grade_date:
                    entry_idx = i
                    break

            if entry_idx is None:
                continue

            # Find exit price
            exit_idx = entry_idx + forward_days
            if exit_idx >= len(price_list):
                continue

            entry_price = price_list[entry_idx][1]
            exit_price = price_list[exit_idx][1]

            if entry_price <= 0:
                continue

            fwd_return = ((exit_price / entry_price) - 1) * 100

            # Check if correct
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

    # Filter to analysts with enough calls
    stats = stats[stats['n_calls'] >= MIN_ANALYST_CALLS]

    # Get top 3 per sector
    top3_by_sector = {}
    for sector in stats['sector'].unique():
        sector_stats = stats[stats['sector'] == sector].nlargest(3, 'hit_rate')
        top3_by_sector[sector] = set(sector_stats['grading_company'].tolist())

    return top3_by_sector


def compute_analyst_signal(symbol, sector, grades_df, top3_analysts, as_of_date, lookback_days=90):
    """
    Compute analyst signal score for a symbol as of a date.
    Uses rolling top 3 analysts (no look-ahead bias).
    """
    as_of_dt = pd.to_datetime(as_of_date)
    cutoff = as_of_dt - timedelta(days=lookback_days)

    recent_grades = grades_df[
        (grades_df['symbol'] == symbol) &
        (grades_df['date'] >= cutoff) &
        (grades_df['date'] <= as_of_dt)
    ]

    if recent_grades.empty:
        return 0

    sector_top3 = top3_analysts.get(sector, set())

    score = 0
    for _, grade in recent_grades.iterrows():
        analyst = grade['grading_company']
        action = grade['action']

        if analyst in sector_top3:
            if action == 'upgrade':
                score += 5
            elif action == 'downgrade':
                score -= 5

    return np.clip(score, -15, 15)


def main():
    print("=" * 70)
    print("V4 + ANALYST HYBRID ANALYSIS")
    print("Testing whether top-3 analyst signals improve V4 performance")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load data
    prices, fund, sectors, grades = load_data()

    # Create sector map
    sector_map = sectors.rename(columns={'symbol': 'symbol', 'sector': 'sector'})

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

    # Sample symbols for speed
    print("\nSampling 1500 symbols for analysis...")
    np.random.seed(42)
    symbols_with_grades = grades['symbol'].unique()
    symbols_with_sectors = set(sector_map['symbol'].unique())

    # Focus on symbols with both grades and sectors
    valid_symbols = [s for s in symbols_with_grades if s in symbols_with_sectors]
    sample_symbols = np.random.choice(valid_symbols, size=min(1500, len(valid_symbols)), replace=False)
    print(f"  Using {len(sample_symbols)} symbols with analyst grades and sector data")

    prices = prices[prices['symbol'].isin(sample_symbols)]
    grades = grades[grades['symbol'].isin(sample_symbols)]

    # Generate observations (2012 onward - need 1 year for analyst history)
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

        # Start from 2012 (need analyst history)
        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            if date.year < 2012:
                continue

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Volatility
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

    # Add sector
    df = df.merge(sector_map, on='symbol', how='left')
    df = df.dropna(subset=['sector'])
    print(f"  With sector: {len(df):,}")

    # Test periods
    test_periods = [
        ('2012-2014', '2012-01-01', '2014-12-31'),
        ('2015-2017', '2015-01-01', '2017-12-31'),
        ('2018-2019', '2018-01-01', '2019-12-31'),
        ('2020', '2020-01-01', '2020-12-31'),  # Problem year
        ('2021-2023', '2021-01-01', '2023-12-31'),
        ('2024+', '2024-01-01', '2026-12-31'),
    ]

    print("\n" + "=" * 70)
    print("COMPUTING V4 + ANALYST SCORES BY PERIOD")
    print("(Using rolling analyst accuracy - no look-ahead bias)")
    print("=" * 70)

    all_period_results = []

    for period_name, start_date, end_date in test_periods:
        print(f"\n--- {period_name} ---")

        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        if len(period_df) < 500:
            print(f"  Skipping: only {len(period_df)} observations")
            continue

        # Compute rolling top 3 analysts as of period start
        # (Conservative: use analysts known at start of period)
        top3_analysts = compute_rolling_top3_analysts(
            grades, prices, sector_map, start_date,
            training_years=3, forward_days=63
        )

        n_sectors_with_analysts = len(top3_analysts)
        print(f"  Rolling top-3 analysts available for {n_sectors_with_analysts} sectors")

        # Compute V4 z-scores within period
        for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
            mean = period_df[col].mean()
            std = period_df[col].std()
            period_df[f'{col}_z'] = ((period_df[col] - mean) / std).clip(-3, 3).fillna(0)

        # V4 score
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
        analyst_signals = []
        for _, row in period_df.iterrows():
            signal = compute_analyst_signal(
                row['symbol'], row['sector'], grades, top3_analysts,
                row['date'], lookback_days=90
            )
            analyst_signals.append(signal)

        period_df['analyst_signal'] = analyst_signals

        # Normalize analyst signal
        if period_df['analyst_signal'].std() > 0:
            period_df['analyst_z'] = (
                (period_df['analyst_signal'] - period_df['analyst_signal'].mean()) /
                period_df['analyst_signal'].std()
            ).clip(-3, 3).fillna(0)
        else:
            period_df['analyst_z'] = 0

        # Hybrid score: V4 + Analyst signal (10% weight)
        period_df['v4_hybrid'] = period_df['v4_score'] + period_df['analyst_z'] * 0.10

        # Analyze results
        v4_only = period_df.copy()
        v4_hybrid = period_df.copy()

        try:
            v4_only['quintile'] = pd.qcut(v4_only['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            v4_hybrid['quintile'] = pd.qcut(v4_hybrid['v4_hybrid'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            print(f"  Could not create quintiles for {period_name}")
            continue

        # V4 only results
        v4_q5 = v4_only[v4_only['quintile'] == 'Q5']['fwd_3m'].mean()
        v4_q1 = v4_only[v4_only['quintile'] == 'Q1']['fwd_3m'].mean()
        v4_spread = v4_q5 - v4_q1
        v4_corr = v4_only['v4_score'].corr(v4_only['fwd_3m'])

        # Hybrid results
        hyb_q5 = v4_hybrid[v4_hybrid['quintile'] == 'Q5']['fwd_3m'].mean()
        hyb_q1 = v4_hybrid[v4_hybrid['quintile'] == 'Q1']['fwd_3m'].mean()
        hyb_spread = hyb_q5 - hyb_q1
        hyb_corr = v4_hybrid['v4_hybrid'].corr(v4_hybrid['fwd_3m'])

        improvement = hyb_spread - v4_spread

        # Analyst signal alone
        analyst_corr = period_df['analyst_signal'].corr(period_df['fwd_3m'])
        n_with_signal = (period_df['analyst_signal'] != 0).sum()

        print(f"  Observations: {len(period_df):,}")
        print(f"  With analyst signal: {n_with_signal:,} ({n_with_signal/len(period_df)*100:.1f}%)")
        print(f"  Analyst signal correlation: {analyst_corr:+.4f}")
        print(f"\n  V4 Only:  Spread = {v4_spread:+.2f}%  Corr = {v4_corr:+.4f}")
        print(f"  Hybrid:   Spread = {hyb_spread:+.2f}%  Corr = {hyb_corr:+.4f}")
        print(f"  Change:   {improvement:+.2f}%")

        all_period_results.append({
            'period': period_name,
            'n_obs': len(period_df),
            'pct_with_signal': n_with_signal/len(period_df)*100,
            'analyst_corr': analyst_corr,
            'v4_spread': v4_spread,
            'hybrid_spread': hyb_spread,
            'improvement': improvement,
        })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: V4 vs V4+ANALYST HYBRID")
    print("=" * 70)

    print(f"\n{'Period':<12} {'Obs':>8} {'% Signal':>10} {'Analyst r':>12} {'V4 Spread':>12} {'Hybrid':>12} {'Change':>10}")
    print("-" * 85)

    for r in all_period_results:
        print(f"{r['period']:<12} {r['n_obs']:>8,} {r['pct_with_signal']:>9.1f}% {r['analyst_corr']:>+11.4f} "
              f"{r['v4_spread']:>+11.2f}% {r['hybrid_spread']:>+11.2f}% {r['improvement']:>+9.2f}%")

    # Key finding: 2020
    period_2020 = [r for r in all_period_results if r['period'] == '2020']
    if period_2020:
        r = period_2020[0]
        print(f"\n*** 2020 ANALYSIS (Problem Year) ***")
        print(f"  Analyst signal correlation: {r['analyst_corr']:+.4f}")
        print(f"  V4 spread: {r['v4_spread']:+.2f}%")
        print(f"  Hybrid spread: {r['hybrid_spread']:+.2f}%")
        print(f"  Improvement: {r['improvement']:+.2f}%")

        if r['improvement'] > 0:
            print(f"\n  FINDING: Analyst signals HELPED during 2020 momentum reversal!")
        else:
            print(f"\n  FINDING: Analyst signals did not help during 2020.")

    # Test analyst signal alone on Q1 (worst stocks)
    print("\n" + "=" * 70)
    print("ANALYST SIGNAL ON V4-Q1 (WORST STOCKS)")
    print("Can analyst signals help identify which 'bad' stocks will recover?")
    print("=" * 70)

    # Combine all periods
    combined_results = []

    for period_name, start_date, end_date in test_periods:
        period_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        if len(period_df) < 500:
            continue

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

        # Get rolling top 3
        top3_analysts = compute_rolling_top3_analysts(
            grades, prices, sector_map, start_date,
            training_years=3, forward_days=63
        )

        # Compute analyst signals
        analyst_signals = []
        for _, row in period_df.iterrows():
            signal = compute_analyst_signal(
                row['symbol'], row['sector'], grades, top3_analysts,
                row['date'], lookback_days=90
            )
            analyst_signals.append(signal)

        period_df['analyst_signal'] = analyst_signals

        try:
            period_df['v4_quintile'] = pd.qcut(period_df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        # Focus on Q1 (worst V4 scores)
        q1_df = period_df[period_df['v4_quintile'] == 'Q1'].copy()

        if len(q1_df) > 100:
            # Split Q1 by analyst signal
            q1_upgraded = q1_df[q1_df['analyst_signal'] > 0]
            q1_neutral = q1_df[q1_df['analyst_signal'] == 0]
            q1_downgraded = q1_df[q1_df['analyst_signal'] < 0]

            combined_results.append({
                'period': period_name,
                'q1_total': len(q1_df),
                'q1_avg_ret': q1_df['fwd_3m'].mean(),
                'upgraded_n': len(q1_upgraded),
                'upgraded_ret': q1_upgraded['fwd_3m'].mean() if len(q1_upgraded) > 10 else np.nan,
                'neutral_n': len(q1_neutral),
                'neutral_ret': q1_neutral['fwd_3m'].mean() if len(q1_neutral) > 10 else np.nan,
                'downgraded_n': len(q1_downgraded),
                'downgraded_ret': q1_downgraded['fwd_3m'].mean() if len(q1_downgraded) > 10 else np.nan,
            })

    print(f"\nQ1 stocks (worst V4) split by analyst signal:")
    print(f"{'Period':<12} {'Q1 Avg':>10} {'Upgraded':>12} {'Neutral':>12} {'Downgraded':>12}")
    print("-" * 60)

    for r in combined_results:
        upgraded = f"{r['upgraded_ret']:+.2f}% ({r['upgraded_n']})" if not np.isnan(r['upgraded_ret']) else f"n/a ({r['upgraded_n']})"
        neutral = f"{r['neutral_ret']:+.2f}% ({r['neutral_n']})" if not np.isnan(r['neutral_ret']) else f"n/a ({r['neutral_n']})"
        downgraded = f"{r['downgraded_ret']:+.2f}% ({r['downgraded_n']})" if not np.isnan(r['downgraded_ret']) else f"n/a ({r['downgraded_n']})"

        print(f"{r['period']:<12} {r['q1_avg_ret']:>+9.2f}% {upgraded:>12} {neutral:>12} {downgraded:>12}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    # Calculate overall stats
    total_improvement = sum(r['improvement'] for r in all_period_results)
    avg_improvement = total_improvement / len(all_period_results) if all_period_results else 0

    periods_improved = sum(1 for r in all_period_results if r['improvement'] > 0)

    print(f"""
1. Average spread improvement: {avg_improvement:+.2f}%
2. Periods where hybrid outperformed: {periods_improved}/{len(all_period_results)}

3. RECOMMENDATION:
""")

    if avg_improvement > 0.5:
        print("   Adding analyst signals provides MEANINGFUL improvement.")
        print("   Consider integrating as +10% weight when top-3 analyst signal present.")
    elif avg_improvement > 0:
        print("   Adding analyst signals provides MARGINAL improvement.")
        print("   May be worth using for tiebreakers or confirmation.")
    else:
        print("   Adding analyst signals does NOT improve overall performance.")
        print("   V4 fundamentals alone are sufficient.")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

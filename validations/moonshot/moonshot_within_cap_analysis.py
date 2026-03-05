#!/usr/bin/env python3
"""
Moonshot Quality-First Within-Cap Quintile Analysis
====================================================
Tests whether Moonshot works WITHIN each market cap segment.

Key Question: Does the score rank stocks well within their peer group,
or does it just favor certain cap segments?

Methodology:
1. Rank stocks within each cap segment (not across all caps)
2. Compute quintile returns within each segment
3. Check if Q5-Q1 spread is positive within each segment

This is different from the basic market cap analysis which ranks across
all stocks and then slices by cap. Here we rank WITHIN each cap segment.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000

# Market cap segments
CAP_SEGMENTS = {
    'Small ($500M-$2B)': (500_000_000, 2_000_000_000),
    'Mid ($2B-$10B)': (2_000_000_000, 10_000_000_000),
    'Large (>$10B)': (10_000_000_000, float('inf')),
}

# Quality-First v2.0 Weights
WEIGHTS = {
    'revenue_growth_3yr': 0.20,
    'eps_growth_3yr': 0.15,
    'gross_margin': 0.15,
    'margin_improvement': 0.10,
    'fcf_margin': 0.15,
    'roe': 0.10,
    'small_cap': 0.10,
    'momentum_12_1': 0.05
}


def passes_quality_filters(row):
    """Check if stock passes Quality-First v2.0 filters."""
    if pd.isna(row['gross_margin']) or row['gross_margin'] < 0.30 or row['gross_margin'] > 0.95:
        return False
    if pd.isna(row['revenue_ttm']) or row['revenue_ttm'] < 50_000_000:
        return False
    if pd.isna(row['revenue_growth']) or row['revenue_growth'] < 0.15 or row['revenue_growth'] > 3.0:
        return False
    if not pd.isna(row['net_income_ttm']) and not pd.isna(row['operating_cash_flow_ttm']):
        if row['net_income_ttm'] > 0:
            if row['operating_cash_flow_ttm'] / row['net_income_ttm'] < 0.7:
                return False
        else:
            if row['operating_cash_flow_ttm'] / row['revenue_ttm'] < -0.5:
                return False
    if not pd.isna(row['total_debt']) and not pd.isna(row['total_assets']) and row['total_assets'] > 0:
        if row['total_debt'] / row['total_assets'] > 2.0:
            return False
    if not pd.isna(row['operating_income_ttm']) and not pd.isna(row['net_income_ttm']):
        if abs(row['net_income_ttm']) > 0 and abs(row['operating_income_ttm']) > 0:
            oi_ni_ratio = abs(row['operating_income_ttm'] / row['net_income_ttm'])
            if oi_ni_ratio < 0.3 or oi_ni_ratio > 3.0:
                return False
    return True


def load_and_prepare_data():
    """Load and prepare data."""
    print("=" * 70)
    print("MOONSHOT WITHIN-CAP QUINTILE ANALYSIS")
    print("=" * 70)
    print(f"Run at: {datetime.now().isoformat()}")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Load fundamentals
    print("Loading fundamentals...")
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
        WHERE i.date >= '1990-01-01'
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()

    # Compute TTM metrics
    print("\nComputing metrics...")
    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    def compute_cagr_3yr(series):
        if len(series) < 13:
            return np.nan
        current = series.iloc[-1]
        three_years_ago = series.iloc[-13]
        if pd.isna(current) or pd.isna(three_years_ago) or three_years_ago <= 0:
            return np.nan
        return (current / three_years_ago) ** (1/3) - 1

    fund['revenue_growth_3yr'] = fund.groupby('symbol')['revenue_ttm'].transform(compute_cagr_3yr)
    fund['eps_growth_3yr'] = fund.groupby('symbol')['eps_ttm'].transform(compute_cagr_3yr)

    fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']
    fund['roe'] = np.where(
        fund['total_equity'].notna() & (fund['total_equity'] > 0),
        fund['net_income_ttm'] / fund['total_equity'],
        np.nan
    )

    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement',
                'revenue_growth_3yr', 'eps_growth_3yr', 'fcf_margin', 'roe']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Compute forward returns
    print("Computing forward returns...")
    prices = prices.sort_values(['symbol', 'date'])
    prices['fwd_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252) * 100

    # Merge
    fund['year_month'] = fund['date'].dt.to_period('M')
    prices['year_month'] = prices['date'].dt.to_period('M')
    month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

    df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_1yr']],
                    on=['symbol', 'year_month'], how='inner')
    df = df.dropna(subset=['fwd_1yr', 'market_cap'])
    df = df[df['market_cap'] >= MIN_MARKET_CAP].copy()

    # Apply quality filters
    print("Applying quality filters...")
    df['passes_quality'] = df.apply(passes_quality_filters, axis=1)
    df = df[df['passes_quality']].copy()
    print(f"  {len(df):,} stock-months passed filters")

    # Compute momentum
    prices_sorted = prices.sort_values(['symbol', 'date'])
    prices_sorted['price_12m'] = prices_sorted.groupby('symbol')['close'].shift(252)
    prices_sorted['price_1m'] = prices_sorted.groupby('symbol')['close'].shift(21)
    prices_sorted['momentum_12_1'] = (prices_sorted['price_1m'] - prices_sorted['price_12m']) / prices_sorted['price_12m']

    month_end_momentum = prices_sorted.groupby(['symbol', 'year_month']).last().reset_index()
    df = df.merge(month_end_momentum[['symbol', 'year_month', 'momentum_12_1']],
                  on=['symbol', 'year_month'], how='left')

    for factor in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
        df[factor] = df[factor].fillna(0)

    df['revenue_growth_3yr'] = df['revenue_growth_3yr'].clip(-0.3, 1.5)
    df['eps_growth_3yr'] = df['eps_growth_3yr'].clip(-0.5, 2.0)
    df['fcf_margin'] = df['fcf_margin'].clip(-0.5, 0.5)
    df['roe'] = df['roe'].clip(-0.5, 1.0)
    df['momentum_12_1'] = df['momentum_12_1'].clip(-0.5, 2.0)

    df['small_cap'] = -np.log10(df['market_cap'] / 1e9)
    df = df.dropna(subset=['revenue_growth_3yr', 'gross_margin'])

    # Add cap segment
    def get_cap_segment(cap):
        for name, (low, high) in CAP_SEGMENTS.items():
            if low <= cap < high:
                return name
        return None

    df['cap_segment'] = df['market_cap'].apply(get_cap_segment)

    # Split IS/OOS
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    # Compute z-score stats from IS
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }

    # Apply z-scores and compute score
    for period_df in [is_df, oos_df]:
        for factor in WEIGHTS.keys():
            period_df[f'{factor}_z'] = (
                (period_df[factor] - factor_stats[factor]['mean']) /
                factor_stats[factor]['std']
            )
        period_df['moonshot_score'] = sum(
            period_df[f'{factor}_z'] * weight
            for factor, weight in WEIGHTS.items()
        )

    return is_df, oos_df


def within_cap_analysis(df, period_name):
    """Compute within-cap quintile returns."""
    print(f"\n{'=' * 70}")
    print(f"WITHIN-CAP QUINTILE ANALYSIS ({period_name})")
    print("=" * 70)
    print("Ranking stocks WITHIN each cap segment (not across all)")

    results = {}

    for segment in CAP_SEGMENTS.keys():
        seg_df = df[df['cap_segment'] == segment].copy()

        if len(seg_df) < 100:
            print(f"\n{segment}: Insufficient data ({len(seg_df)} obs)")
            continue

        # Rank within segment (across all months)
        seg_df['within_cap_quintile'] = pd.qcut(seg_df['moonshot_score'], 5,
                                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                  duplicates='drop')

        print(f"\n{segment}:")
        print(f"{'Quintile':<10} {'Avg Return':<12} {'Count':<10}")
        print("-" * 35)

        q_returns = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = seg_df[seg_df['within_cap_quintile'] == q]
            if len(q_data) > 0:
                avg_ret = q_data['fwd_1yr'].mean()
                q_returns[q] = avg_ret
                print(f"{q:<10} {avg_ret:>+10.2f}%  {len(q_data):>8,}")

        if 'Q5' in q_returns and 'Q1' in q_returns:
            spread = q_returns['Q5'] - q_returns['Q1']
            print("-" * 35)
            print(f"{'Q5-Q1':<10} {spread:>+10.2f}%")

            # T-test
            q5_returns = seg_df[seg_df['within_cap_quintile'] == 'Q5']['fwd_1yr']
            q1_returns = seg_df[seg_df['within_cap_quintile'] == 'Q1']['fwd_1yr']
            from scipy import stats
            t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns)
            print(f"{'t-stat':<10} {t_stat:>+10.2f}")

            # Check monotonicity
            returns_list = [q_returns.get(q, np.nan) for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
            is_monotonic = all(returns_list[i] <= returns_list[i+1] for i in range(4) if not pd.isna(returns_list[i]) and not pd.isna(returns_list[i+1]))
            print(f"{'Monotonic':<10} {'Yes' if is_monotonic else 'No':>10}")

            results[segment] = {
                'spread': spread,
                't_stat': t_stat,
                'monotonic': is_monotonic,
                'obs': len(seg_df)
            }

    return results


def compare_cross_vs_within(df, period_name):
    """Compare cross-sectional ranking vs within-cap ranking."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: CROSS-SECTIONAL vs WITHIN-CAP ({period_name})")
    print("=" * 70)

    print("\nCross-Sectional (standard method):")
    print("- Rank all stocks together, then slice by cap")

    # Cross-sectional quintiles
    df['cross_quintile'] = pd.qcut(df['moonshot_score'], 5,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                    duplicates='drop')

    cross_results = {}
    for segment in CAP_SEGMENTS.keys():
        seg_df = df[df['cap_segment'] == segment]
        if len(seg_df) >= 100:
            q5_ret = seg_df[seg_df['cross_quintile'] == 'Q5']['fwd_1yr'].mean()
            q1_ret = seg_df[seg_df['cross_quintile'] == 'Q1']['fwd_1yr'].mean()
            cross_results[segment] = q5_ret - q1_ret

    print("\nWithin-Cap (this analysis):")
    print("- Rank stocks within each cap segment separately")

    within_results = {}
    for segment in CAP_SEGMENTS.keys():
        seg_df = df[df['cap_segment'] == segment].copy()
        if len(seg_df) >= 100:
            seg_df['within_quintile'] = pd.qcut(seg_df['moonshot_score'], 5,
                                                 labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                 duplicates='drop')
            q5_ret = seg_df[seg_df['within_quintile'] == 'Q5']['fwd_1yr'].mean()
            q1_ret = seg_df[seg_df['within_quintile'] == 'Q1']['fwd_1yr'].mean()
            within_results[segment] = q5_ret - q1_ret

    # Print comparison
    print(f"\n{'Segment':<25} {'Cross-Sect':<15} {'Within-Cap':<15} {'Diff':<10}")
    print("-" * 70)

    for segment in CAP_SEGMENTS.keys():
        cross = cross_results.get(segment, np.nan)
        within = within_results.get(segment, np.nan)
        diff = within - cross if not pd.isna(within) and not pd.isna(cross) else np.nan

        cross_str = f"{cross:+.2f}%" if not pd.isna(cross) else "N/A"
        within_str = f"{within:+.2f}%" if not pd.isna(within) else "N/A"
        diff_str = f"{diff:+.2f}%" if not pd.isna(diff) else "N/A"

        print(f"{segment:<25} {cross_str:<15} {within_str:<15} {diff_str:<10}")

    # Summary
    print("\nInterpretation:")
    if within_results:
        avg_within = np.mean([v for v in within_results.values() if not pd.isna(v)])
        positive_within = sum(1 for v in within_results.values() if v > 0)

        if positive_within == len(within_results):
            print("=> PASS: Moonshot works within ALL cap segments")
        elif positive_within > 0:
            print(f"=> PARTIAL: Moonshot works within {positive_within}/{len(within_results)} cap segments")
        else:
            print("=> FAIL: Moonshot doesn't work within cap segments")


def main():
    is_df, oos_df = load_and_prepare_data()

    # Within-cap analysis
    is_results = within_cap_analysis(is_df, "IN-SAMPLE (1995-2019)")
    oos_results = within_cap_analysis(oos_df, "OUT-OF-SAMPLE (2020-2026)")

    # Comparison
    compare_cross_vs_within(is_df, "IN-SAMPLE")
    compare_cross_vs_within(oos_df, "OUT-OF-SAMPLE")

    # Summary
    print("\n" + "=" * 70)
    print("WITHIN-CAP ANALYSIS SUMMARY")
    print("=" * 70)

    print("\nOUT-OF-SAMPLE Results:")
    if oos_results:
        for segment, res in oos_results.items():
            status = "PASS" if res['spread'] > 0 and res['t_stat'] > 1.65 else "CHECK"
            print(f"  {segment}: {res['spread']:+.2f}% spread (t={res['t_stat']:.2f}) [{status}]")

        all_positive = all(r['spread'] > 0 for r in oos_results.values())
        if all_positive:
            print("\n=> OVERALL: Moonshot score works within ALL cap segments")
        else:
            print("\n=> OVERALL: Moonshot score works in some but not all segments")

    print("\n" + "=" * 70)
    print("WITHIN-CAP ANALYSIS COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

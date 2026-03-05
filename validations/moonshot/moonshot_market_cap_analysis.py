#!/usr/bin/env python3
"""
Moonshot Quality-First Market Cap Segmentation Analysis
========================================================
Tests if Moonshot works differently across market cap segments.

SEGMENTS:
- Small-cap: $500M-$2B
- Mid-cap: $2B-$10B
- Large-cap: >$10B

NOTE: Micro-caps (<$500M) excluded due to insufficient validation.

METHODOLOGY:
- Quality-First v2.0 filters and weights
- Z-scores computed from full IS universe
- Performance measured separately for each segment
- In-sample: 1995-2019 | Out-of-sample: 2020-2026

EXPECTED OUTCOMES:
- All segments should show positive Q5-Q1 spread
- Small/mid may outperform large (less efficient, higher growth)
- Quality filters should work across all segments
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000  # $500M

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

# Market cap segments
SEGMENTS = {
    'Small-cap': (500_000_000, 2_000_000_000),
    'Mid-cap': (2_000_000_000, 10_000_000_000),
    'Large-cap': (10_000_000_000, np.inf)
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
    """Load and prepare data for analysis."""
    print("\n" + "=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    print("\nLoading data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

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

    conn.close()

    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")
    print(f"  {len(fund):,} fundamental records")

    # Compute metrics
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

    # 3-year CAGR
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

    # Compute forward 1-year returns (252 trading days)
    print("Computing forward 1-year returns...")
    prices = prices.sort_values(['symbol', 'date'])
    prices['fwd_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252) * 100

    # Merge monthly
    fund['year_month'] = fund['date'].dt.to_period('M')
    prices['year_month'] = prices['date'].dt.to_period('M')
    month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

    df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_1yr']],
                    on=['symbol', 'year_month'], how='inner')
    df = df.dropna(subset=['fwd_1yr', 'market_cap'])

    # Filter for ≥$500M
    df = df[df['market_cap'] >= MIN_MARKET_CAP].copy()
    print(f"  {len(df):,} observations after market cap filter")

    # Apply quality filters
    df['passes_quality'] = df.apply(passes_quality_filters, axis=1)
    df = df[df['passes_quality']].copy()
    print(f"  {len(df):,} observations after quality filters")

    # Compute momentum
    prices_sorted = prices.sort_values(['symbol', 'date'])
    prices_sorted['price_12m'] = prices_sorted.groupby('symbol')['close'].shift(252)
    prices_sorted['price_1m'] = prices_sorted.groupby('symbol')['close'].shift(21)
    prices_sorted['momentum_12_1'] = (prices_sorted['price_1m'] - prices_sorted['price_12m']) / prices_sorted['price_12m']

    month_end_momentum = prices_sorted.groupby(['symbol', 'year_month']).last().reset_index()
    df = df.merge(month_end_momentum[['symbol', 'year_month', 'momentum_12_1']],
                  on=['symbol', 'year_month'], how='left')

    # Handle missing factors
    for factor in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
        df[factor] = df[factor].fillna(0)

    # Cap extreme values
    df['revenue_growth_3yr'] = df['revenue_growth_3yr'].clip(-0.3, 1.5)
    df['eps_growth_3yr'] = df['eps_growth_3yr'].clip(-0.5, 2.0)
    df['fcf_margin'] = df['fcf_margin'].clip(-0.5, 0.5)
    df['roe'] = df['roe'].clip(-0.5, 1.0)
    df['momentum_12_1'] = df['momentum_12_1'].clip(-0.5, 2.0)

    # Compute small cap factor
    df['small_cap'] = -np.log10(df['market_cap'] / 1e9)

    # Remove critical missing values
    df = df.dropna(subset=['revenue_growth_3yr', 'gross_margin'])
    print(f"  {len(df):,} observations ready for scoring")

    return df


def analyze_by_market_cap(df):
    """Analyze performance by market cap segment."""
    print("\n" + "=" * 70)
    print("MARKET CAP SEGMENTATION ANALYSIS")
    print("=" * 70)

    # Split IS/OOS
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\n  In-sample: {len(is_df):,} observations")
    print(f"  Out-of-sample: {len(oos_df):,} observations")

    # Compute z-scores from IS data ONLY
    print("\nComputing z-scores from IS data...")
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }

    # Apply z-scores to both IS and OOS
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

    # Assign market cap segments
    for period_df in [is_df, oos_df]:
        period_df['cap_segment'] = 'Unknown'
        for seg_name, (min_cap, max_cap) in SEGMENTS.items():
            mask = (period_df['market_cap'] >= min_cap) & (period_df['market_cap'] < max_cap)
            period_df.loc[mask, 'cap_segment'] = seg_name

    # Analyze each segment
    print("\n" + "=" * 70)
    print("IN-SAMPLE PERFORMANCE BY SEGMENT (1995-2019)")
    print("=" * 70)

    analyze_segment_performance(is_df, "In-Sample")

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE PERFORMANCE BY SEGMENT (2020-2026)")
    print("=" * 70)

    analyze_segment_performance(oos_df, "Out-of-Sample")


def analyze_segment_performance(df, period_name):
    """Analyze performance for each market cap segment."""

    segments = df['cap_segment'].unique()
    segments = [s for s in ['Small-cap', 'Mid-cap', 'Large-cap'] if s in segments]

    print(f"\n{'Segment':<15} {'Observations':>15} {'Avg Market Cap':>18} {'Q5-Q1 Spread':>15}")
    print("-" * 70)

    results = []

    for segment in segments:
        seg_df = df[df['cap_segment'] == segment].copy()

        if len(seg_df) == 0:
            continue

        # Assign quintiles across ALL time periods for this segment (time-series ranking)
        if len(seg_df) >= 250:
            try:
                seg_df['quintile'] = pd.qcut(seg_df['moonshot_score'], 5,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                            duplicates='drop')
            except ValueError:
                continue
        else:
            continue

        seg_df = seg_df.dropna(subset=['quintile'])

        # Compute quintile returns
        quintile_returns = seg_df.groupby('quintile')['fwd_1yr'].mean()

        if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
            spread = quintile_returns['Q5'] - quintile_returns['Q1']
        else:
            spread = np.nan

        avg_cap = seg_df['market_cap'].median()

        print(f"{segment:<15} {len(seg_df):>15,} ${avg_cap/1e9:>14.1f}B {spread:>+14.2f}%")

        results.append({
            'segment': segment,
            'observations': len(seg_df),
            'avg_market_cap': avg_cap,
            'spread': spread,
            'quintile_returns': quintile_returns
        })

        # Print quintile breakdown
        print(f"\n  {segment} - Quintile Returns:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in quintile_returns:
                print(f"    {q}: {quintile_returns[q]:>+8.2f}%")

    # Check monotonicity
    print(f"\n  Monotonicity Check:")
    for result in results:
        qr = result['quintile_returns']
        is_monotonic = all(qr.loc[f'Q{i}'] < qr.loc[f'Q{i+1}'] for i in range(1, 5) if f'Q{i}' in qr and f'Q{i+1}' in qr)
        status = "✅ MONOTONIC" if is_monotonic else "⚠️ NON-MONOTONIC"
        print(f"    {result['segment']:<15} {status}")

    # Summary
    print(f"\n  Summary:")
    positive_segments = sum(1 for r in results if r['spread'] > 0)
    print(f"    Segments with positive spread: {positive_segments}/{len(results)}")

    if len(results) >= 2:
        best_segment = max(results, key=lambda x: x['spread'] if not pd.isna(x['spread']) else -999)
        print(f"    Best performing segment: {best_segment['segment']} ({best_segment['spread']:+.2f}%)")


def main():
    print("=" * 80)
    print("MOONSHOT QUALITY-FIRST MARKET CAP SEGMENTATION ANALYSIS")
    print("=" * 80)
    print(f"Run at: {datetime.now().isoformat()}")
    print()
    print("Methodology: Quality-First v2.0 (≥$500M market cap)")
    print("Segments: Small ($500M-$2B), Mid ($2B-$10B), Large (>$10B)")
    print("=" * 80)

    # Load and prepare data
    df = load_and_prepare_data()

    # Analyze by market cap
    analyze_by_market_cap(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n✅ SUCCESS CRITERIA:")
    print("   - All segments show positive OOS Q5-Q1 spread")
    print("   - Small/Mid-cap may outperform Large-cap (less efficient markets)")
    print("   - Monotonic quintile returns within each segment")
    print("   - Consistent performance across market conditions")


if __name__ == '__main__':
    main()

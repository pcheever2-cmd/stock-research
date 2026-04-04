#!/usr/bin/env python3
"""
Moonshot Quality-First Factor Contribution Analysis
====================================================
Analyzes how much each factor contributes to the Q5-Q1 spread.

FACTORS ANALYZED:
1. Revenue Growth (3yr CAGR) - 20% weight
2. EPS Growth (3yr CAGR) - 15% weight
3. Gross Margin - 15% weight
4. Margin Improvement - 10% weight
5. FCF Margin - 15% weight
6. ROE - 10% weight
7. Small Cap - 10% weight
8. Momentum 12-1 - 5% weight

METHODOLOGY:
- Compute "single-factor scores" (each factor alone)
- Measure Q5-Q1 spread for each single factor
- Compare to full Moonshot spread
- Identify which factors contribute most

EXPECTED OUTCOMES:
- Revenue/EPS growth should contribute heavily (growth focus)
- FCF margin and ROE should add quality signal
- Small cap should add some spread (small-cap premium)
- All factors should contribute positively (no deadweight)
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
    """Load and prepare data for analysis with IS stats for z-scores."""
    print("\n" + "=" * 70)
    print("LOADING AND PREPARING DATA (IS STATS FOR Z-SCORES)")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    print("\nLoading data...")
    # Load full history to compute IS stats, not just OOS period
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Load full history to compute IS stats
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

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")

    # Compute metrics
    print("Computing metrics...")
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

    # Filter for ≥$500M (keep both IS and OOS for now)
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
    print(f"  {len(df):,} observations ready for analysis")

    # Split IS/OOS and compute z-score stats from IS data
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\n  In-sample: {len(is_df):,} observations")
    print(f"  Out-of-sample: {len(oos_df):,} observations")

    # Compute z-score statistics from IS data ONLY (critical for no look-ahead bias)
    print("\nComputing z-score statistics from IS data...")
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }
        print(f"  {factor}: mean={factor_stats[factor]['mean']:.4f}, std={factor_stats[factor]['std']:.4f}")

    # Return OOS data and IS factor stats
    return oos_df, factor_stats


def analyze_single_factor_contribution(df, factor_name):
    """Analyze contribution of a single factor."""
    # Compute quintiles across ALL time periods based on this single factor (time-series ranking)
    df_factor = df.copy()

    if len(df_factor) >= 250:
        try:
            df_factor['quintile'] = pd.qcut(df_factor[factor_name], 5,
                                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                           duplicates='drop')
        except ValueError:
            return np.nan, np.nan
    else:
        return np.nan, np.nan

    df_factor = df_factor.dropna(subset=['quintile'])

    if len(df_factor) == 0:
        return np.nan, np.nan

    # Compute quintile returns
    quintile_returns = df_factor.groupby('quintile')['fwd_1yr'].mean()

    if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
        spread = quintile_returns['Q5'] - quintile_returns['Q1']
        q5_ret = quintile_returns['Q5']
    else:
        spread = np.nan
        q5_ret = np.nan

    return spread, q5_ret


def analyze_full_moonshot(df, factor_stats):
    """Analyze full Moonshot score using IS statistics for z-scores."""
    # Compute z-scores using IS statistics (no look-ahead bias)
    for factor in WEIGHTS.keys():
        mean = factor_stats[factor]['mean']
        std = factor_stats[factor]['std']
        if std > 0:
            df[f'{factor}_z'] = (df[factor] - mean) / std
        else:
            df[f'{factor}_z'] = 0

    # Compute full Moonshot score
    df['moonshot_score'] = sum(df[f'{factor}_z'] * weight for factor, weight in WEIGHTS.items())

    # Quintiles across ALL time periods (time-series ranking)
    if len(df) >= 250:
        try:
            df['quintile'] = pd.qcut(df['moonshot_score'], 5,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                    duplicates='drop')
        except ValueError:
            df['quintile'] = np.nan
    else:
        df['quintile'] = np.nan

    df = df.dropna(subset=['quintile'])

    # Compute quintile returns
    quintile_returns = df.groupby('quintile')['fwd_1yr'].mean()

    if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
        spread = quintile_returns['Q5'] - quintile_returns['Q1']
    else:
        spread = np.nan

    return spread


def main():
    print("=" * 80)
    print("MOONSHOT QUALITY-FIRST FACTOR CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"Run at: {datetime.now().isoformat()}")
    print()
    print("Methodology: Quality-First v2.0 (≥$500M market cap)")
    print("Period: Out-of-sample (2020-2026)")
    print("=" * 80)

    # Load OOS data with IS statistics
    df, factor_stats = load_and_prepare_data()

    # Analyze full Moonshot
    print("\n" + "=" * 70)
    print("FULL MOONSHOT SPREAD")
    print("=" * 70)
    full_spread = analyze_full_moonshot(df.copy(), factor_stats)
    print(f"\nFull Moonshot Q5-Q1 Spread (OOS): {full_spread:+.2f}%")

    # Analyze single factors
    print("\n" + "=" * 70)
    print("SINGLE FACTOR CONTRIBUTIONS")
    print("=" * 70)

    print(f"\n{'Factor':<25} {'Weight':>8} {'Q5-Q1 Spread':>15} {'Q5 Return':>15}")
    print("-" * 70)

    results = []
    for factor_name, weight in WEIGHTS.items():
        spread, q5_ret = analyze_single_factor_contribution(df, factor_name)
        results.append({
            'factor': factor_name,
            'weight': weight,
            'spread': spread,
            'q5_ret': q5_ret
        })

        spread_str = f"{spread:+.2f}%" if not pd.isna(spread) else "N/A"
        q5_str = f"{q5_ret:+.2f}%" if not pd.isna(q5_ret) else "N/A"
        print(f"{factor_name:<25} {weight*100:>7.0f}% {spread_str:>15} {q5_str:>15}")

    # Summary analysis
    print("\n" + "=" * 70)
    print("FACTOR ANALYSIS SUMMARY")
    print("=" * 70)

    positive_factors = sum(1 for r in results if not pd.isna(r['spread']) and r['spread'] > 0)
    total_factors = len(results)
    print(f"\nFactors with positive spread: {positive_factors}/{total_factors}")

    # Rank by contribution
    results_sorted = sorted([r for r in results if not pd.isna(r['spread'])],
                           key=lambda x: x['spread'], reverse=True)

    print(f"\nTop 3 contributing factors:")
    for i, r in enumerate(results_sorted[:3], 1):
        print(f"  {i}. {r['factor']}: {r['spread']:+.2f}% spread")

    print(f"\nWeakest factors:")
    for i, r in enumerate(results_sorted[-3:], 1):
        print(f"  {i}. {r['factor']}: {r['spread']:+.2f}% spread")

    # Check if any factors are deadweight
    deadweight = [r for r in results if not pd.isna(r['spread']) and r['spread'] < 0]
    if deadweight:
        print(f"\n⚠️ WARNING: {len(deadweight)} factors have negative spread:")
        for r in deadweight:
            print(f"  - {r['factor']}: {r['spread']:+.2f}%")
    else:
        print(f"\n✅ All factors contribute positively")

    # Weighted contribution estimate
    print(f"\n" + "=" * 70)
    print("WEIGHTED CONTRIBUTION ESTIMATE")
    print("=" * 70)
    print("\nNote: This is a rough approximation. Factors interact non-linearly.")

    weighted_sum = sum(r['spread'] * r['weight'] for r in results if not pd.isna(r['spread']))
    print(f"\nSum of weighted single-factor spreads: {weighted_sum:+.2f}%")
    print(f"Full Moonshot spread: {full_spread:+.2f}%")
    print(f"Ratio (interaction effect): {full_spread / weighted_sum:.2f}x")

    if full_spread > weighted_sum:
        print("\n✅ Factors have POSITIVE interaction (diversification benefit)")
    else:
        print("\n⚠️ Factors have NEGATIVE interaction (some redundancy)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n✅ SUCCESS CRITERIA:")
    print("   - All factors should contribute positively")
    print("   - Growth factors (revenue/EPS growth) should be top contributors")
    print("   - Quality factors (FCF margin, ROE) should add signal")
    print("   - No single factor should dominate (diversification)")


if __name__ == '__main__':
    main()

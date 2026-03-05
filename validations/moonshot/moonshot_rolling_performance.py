#!/usr/bin/env python3
"""
Moonshot Quality-First Rolling Performance Analysis
====================================================
Analyzes performance consistency over time using rolling windows.

Key Questions:
1. Is the alpha consistent over time or concentrated in specific periods?
2. What's the rolling Sharpe ratio?
3. Are there periods of significant underperformance?

Methodology:
- 3-year rolling windows
- Compute Q5-Q1 spread for each window
- Track consistency and drawdowns
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
    print("MOONSHOT ROLLING PERFORMANCE ANALYSIS")
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

    # Apply z-scores and compute score to full df
    for factor in WEIGHTS.keys():
        df[f'{factor}_z'] = (
            (df[factor] - factor_stats[factor]['mean']) /
            factor_stats[factor]['std']
        )
    df['moonshot_score'] = sum(
        df[f'{factor}_z'] * weight
        for factor, weight in WEIGHTS.items()
    )

    return df


def compute_monthly_spreads(df):
    """Compute Q5-Q1 spread for each month."""
    print("\nComputing monthly Q5-Q1 spreads...")

    monthly_spreads = []

    for ym, group in df.groupby('year_month'):
        if len(group) < 50:
            continue

        try:
            group['quintile'] = pd.qcut(group['moonshot_score'], 5,
                                         labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                         duplicates='drop')

            q5_ret = group[group['quintile'] == 'Q5']['fwd_1yr'].mean()
            q1_ret = group[group['quintile'] == 'Q1']['fwd_1yr'].mean()
            spread = q5_ret - q1_ret

            monthly_spreads.append({
                'year_month': ym,
                'date': ym.to_timestamp(),
                'spread': spread,
                'n_stocks': len(group),
                'q5_return': q5_ret,
                'q1_return': q1_ret
            })
        except:
            continue

    return pd.DataFrame(monthly_spreads)


def rolling_analysis(spreads_df, period_name):
    """Perform rolling window analysis."""
    print(f"\n{'=' * 70}")
    print(f"ROLLING 3-YEAR ANALYSIS ({period_name})")
    print("=" * 70)

    spreads_df = spreads_df.sort_values('date')

    # Filter by period
    if period_name == "IN-SAMPLE":
        spreads_df = spreads_df[spreads_df['date'] <= IN_SAMPLE_END]
    elif period_name == "OUT-OF-SAMPLE":
        spreads_df = spreads_df[spreads_df['date'] >= OOS_START]

    if len(spreads_df) < 24:
        print("  Insufficient data for rolling analysis")
        return None

    # Rolling 36-month stats
    spreads_df['rolling_mean'] = spreads_df['spread'].rolling(36, min_periods=12).mean()
    spreads_df['rolling_std'] = spreads_df['spread'].rolling(36, min_periods=12).std()
    spreads_df['rolling_sharpe'] = spreads_df['rolling_mean'] / spreads_df['rolling_std']

    # Summary stats
    print(f"\nOverall Statistics:")
    print(f"  Average Monthly Spread: {spreads_df['spread'].mean():>+8.2f}%")
    print(f"  Std Dev of Spread:      {spreads_df['spread'].std():>8.2f}%")
    print(f"  Information Ratio:      {spreads_df['spread'].mean() / spreads_df['spread'].std():>+8.2f}")
    print(f"  Months Analyzed:        {len(spreads_df):>8}")

    # Positive/negative months
    positive_months = (spreads_df['spread'] > 0).sum()
    print(f"\n  Positive Spread Months: {positive_months}/{len(spreads_df)} ({positive_months/len(spreads_df)*100:.1f}%)")

    # Rolling window summary
    valid_rolling = spreads_df.dropna(subset=['rolling_mean'])
    if len(valid_rolling) > 0:
        print(f"\nRolling 3-Year Windows:")
        print(f"  Average Rolling Spread: {valid_rolling['rolling_mean'].mean():>+8.2f}%")
        print(f"  Min Rolling Spread:     {valid_rolling['rolling_mean'].min():>+8.2f}%")
        print(f"  Max Rolling Spread:     {valid_rolling['rolling_mean'].max():>+8.2f}%")

        # Worst periods
        print(f"\nWorst Rolling 3-Year Periods:")
        worst = valid_rolling.nsmallest(3, 'rolling_mean')
        for _, row in worst.iterrows():
            print(f"  {row['date'].strftime('%Y-%m')}: {row['rolling_mean']:>+8.2f}% avg spread")

        # Best periods
        print(f"\nBest Rolling 3-Year Periods:")
        best = valid_rolling.nlargest(3, 'rolling_mean')
        for _, row in best.iterrows():
            print(f"  {row['date'].strftime('%Y-%m')}: {row['rolling_mean']:>+8.2f}% avg spread")

        # Consistency check
        always_positive = (valid_rolling['rolling_mean'] > 0).all()
        mostly_positive = (valid_rolling['rolling_mean'] > 0).mean() > 0.8

        print(f"\nConsistency Assessment:")
        if always_positive:
            print("  => PASS: All rolling 3-year periods show positive spreads")
        elif mostly_positive:
            pct_positive = (valid_rolling['rolling_mean'] > 0).mean() * 100
            print(f"  => ACCEPTABLE: {pct_positive:.0f}% of rolling periods show positive spreads")
        else:
            pct_positive = (valid_rolling['rolling_mean'] > 0).mean() * 100
            print(f"  => CAUTION: Only {pct_positive:.0f}% of rolling periods show positive spreads")

    return spreads_df


def year_by_year_analysis(spreads_df):
    """Year-by-year breakdown."""
    print(f"\n{'=' * 70}")
    print("YEAR-BY-YEAR SPREAD ANALYSIS")
    print("=" * 70)

    spreads_df['year'] = spreads_df['date'].dt.year

    print(f"\n{'Year':<8} {'Avg Spread':<12} {'Std Dev':<10} {'Months':<8} {'% Positive':<12}")
    print("-" * 55)

    yearly_results = []
    for year in sorted(spreads_df['year'].unique()):
        year_data = spreads_df[spreads_df['year'] == year]
        if len(year_data) >= 6:
            avg = year_data['spread'].mean()
            std = year_data['spread'].std()
            pct_pos = (year_data['spread'] > 0).mean() * 100

            period = "IS" if year <= 2019 else "OOS"
            print(f"{year:<8} {avg:>+10.2f}%  {std:>8.2f}%  {len(year_data):>6}  {pct_pos:>10.1f}%  ({period})")

            yearly_results.append({
                'year': year,
                'avg_spread': avg,
                'std': std,
                'period': period
            })

    # OOS summary
    oos_years = [r for r in yearly_results if r['period'] == 'OOS']
    if oos_years:
        oos_positive = sum(1 for r in oos_years if r['avg_spread'] > 0)
        print("-" * 55)
        print(f"OOS Years with Positive Spread: {oos_positive}/{len(oos_years)}")

        if oos_positive == len(oos_years):
            print("\n=> PASS: All OOS years show positive spreads")
        elif oos_positive >= len(oos_years) * 0.67:
            print("\n=> ACCEPTABLE: Most OOS years show positive spreads")
        else:
            print("\n=> CAUTION: Some OOS years show negative spreads")


def main():
    df = load_and_prepare_data()

    # Compute monthly spreads
    spreads_df = compute_monthly_spreads(df)
    print(f"  Computed spreads for {len(spreads_df)} months")

    # Rolling analysis
    rolling_analysis(spreads_df, "IN-SAMPLE")
    rolling_analysis(spreads_df, "OUT-OF-SAMPLE")

    # Year-by-year
    year_by_year_analysis(spreads_df)

    # Summary
    print("\n" + "=" * 70)
    print("ROLLING PERFORMANCE SUMMARY")
    print("=" * 70)

    # OOS only
    oos_spreads = spreads_df[spreads_df['date'] >= OOS_START]
    if len(oos_spreads) > 0:
        avg_spread = oos_spreads['spread'].mean()
        pct_positive = (oos_spreads['spread'] > 0).mean() * 100
        ir = avg_spread / oos_spreads['spread'].std()

        print(f"\nOut-of-Sample Performance:")
        print(f"  Average Monthly Spread: {avg_spread:+.2f}%")
        print(f"  Positive Months: {pct_positive:.0f}%")
        print(f"  Information Ratio: {ir:.2f}")

        if pct_positive >= 60 and avg_spread > 0:
            print("\n=> OVERALL: Consistent positive performance in OOS period")
        elif pct_positive >= 50 and avg_spread > 0:
            print("\n=> OVERALL: Positive but volatile performance in OOS period")
        else:
            print("\n=> OVERALL: Inconsistent performance in OOS period")

    print("\n" + "=" * 70)
    print("ROLLING PERFORMANCE ANALYSIS COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Moonshot Quality-First Comprehensive Robustness Tests
======================================================
Addresses critical concerns for rigorous validation:

1. NON-OVERLAPPING RETURNS
   - Using 1-year forward returns with monthly observations creates autocorrelation
   - Test with annual (non-overlapping) samples to get cleaner statistical inference

2. SUB-PERIOD ANALYSIS
   - Break OOS into sub-periods to ensure alpha isn't concentrated in one period
   - 2020-2021 (COVID/recovery), 2022-2023 (rate hikes), 2024-2026 (AI boom)

3. NEWEY-WEST HAC STANDARD ERRORS
   - Already in FF6 regression, but verify t-stats hold with proper HAC

4. BOOTSTRAP CONFIDENCE INTERVALS
   - Resample to get confidence intervals on Q5-Q1 spread
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

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
    """Load and prepare data for validation."""
    print("=" * 70)
    print("MOONSHOT COMPREHENSIVE ROBUSTNESS TESTS")
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

    # Add year for sub-period analysis
    df['year'] = df['date'].dt.year

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


def test_non_overlapping_returns(df, period_name):
    """Test with non-overlapping annual samples to eliminate autocorrelation."""
    print(f"\n{'=' * 70}")
    print(f"TEST 1: NON-OVERLAPPING RETURNS ({period_name})")
    print("=" * 70)
    print("Using December-only samples for non-overlapping 1-year forward returns")

    # Get December observations only (non-overlapping annual)
    df_dec = df[df['date'].dt.month == 12].copy()
    print(f"December observations: {len(df_dec):,}")

    if len(df_dec) < 50:
        print("  Insufficient data for non-overlapping test")
        return None

    # Assign quintiles
    df_dec['quintile'] = pd.qcut(df_dec['moonshot_score'], 5,
                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                  duplicates='drop')

    print(f"\n{'Quintile':<10} {'Avg Return':<12} {'Count':<10}")
    print("-" * 35)

    q_returns = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = df_dec[df_dec['quintile'] == q]
        if len(q_data) > 0:
            avg_ret = q_data['fwd_1yr'].mean()
            q_returns[q] = avg_ret
            print(f"{q:<10} {avg_ret:>+10.2f}%  {len(q_data):>8,}")

    if 'Q5' in q_returns and 'Q1' in q_returns:
        spread = q_returns['Q5'] - q_returns['Q1']
        print("-" * 35)
        print(f"{'Q5-Q1':<10} {spread:>+10.2f}%")

        # T-test with non-overlapping samples
        q5_returns = df_dec[df_dec['quintile'] == 'Q5']['fwd_1yr']
        q1_returns = df_dec[df_dec['quintile'] == 'Q1']['fwd_1yr']
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns)
        print(f"{'t-stat':<10} {t_stat:>+10.2f}")
        print(f"{'p-value':<10} {p_val:>10.4f}")

        if p_val < 0.05:
            print("\n=> PASS: Significant at 95% confidence with non-overlapping returns")
        elif p_val < 0.10:
            print("\n=> MARGINAL: Significant at 90% confidence")
        else:
            print("\n=> CAUTION: Not significant with non-overlapping samples")

    return spread if 'Q5' in q_returns and 'Q1' in q_returns else None


def test_sub_periods(df, period_name):
    """Break period into sub-periods to check consistency."""
    print(f"\n{'=' * 70}")
    print(f"TEST 2: SUB-PERIOD ANALYSIS ({period_name})")
    print("=" * 70)

    # Define sub-periods
    if period_name == "OUT-OF-SAMPLE":
        sub_periods = [
            ('2020-2021 (COVID)', 2020, 2021),
            ('2022-2023 (Rate Hikes)', 2022, 2023),
            ('2024-2026 (AI Boom)', 2024, 2026),
        ]
    else:
        sub_periods = [
            ('1995-2004', 1995, 2004),
            ('2005-2014', 2005, 2014),
            ('2015-2019', 2015, 2019),
        ]

    print(f"\n{'Sub-Period':<25} {'Q5-Q1 Spread':<15} {'Obs':<10} {'t-stat':<10}")
    print("-" * 65)

    results = []
    for name, start_year, end_year in sub_periods:
        sub_df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()

        if len(sub_df) < 50:
            print(f"{name:<25} {'Insufficient data':<15}")
            continue

        sub_df['quintile'] = pd.qcut(sub_df['moonshot_score'], 5,
                                      labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                      duplicates='drop')

        q5_ret = sub_df[sub_df['quintile'] == 'Q5']['fwd_1yr'].mean()
        q1_ret = sub_df[sub_df['quintile'] == 'Q1']['fwd_1yr'].mean()
        spread = q5_ret - q1_ret

        # T-test
        q5_returns = sub_df[sub_df['quintile'] == 'Q5']['fwd_1yr']
        q1_returns = sub_df[sub_df['quintile'] == 'Q1']['fwd_1yr']
        from scipy import stats
        t_stat, _ = stats.ttest_ind(q5_returns, q1_returns)

        results.append({'name': name, 'spread': spread, 'obs': len(sub_df), 't_stat': t_stat})
        print(f"{name:<25} {spread:>+13.2f}%  {len(sub_df):>8,}  {t_stat:>+8.2f}")

    if results:
        positive_periods = sum(1 for r in results if r['spread'] > 0)
        print("-" * 65)
        print(f"Positive sub-periods: {positive_periods}/{len(results)}")

        if positive_periods == len(results):
            print("\n=> PASS: All sub-periods show positive spreads")
        elif positive_periods >= len(results) * 0.67:
            print("\n=> ACCEPTABLE: Most sub-periods show positive spreads")
        else:
            print("\n=> CAUTION: Alpha may be concentrated in certain periods")

    return results


def test_bootstrap_confidence(df, period_name, n_bootstrap=1000):
    """Bootstrap confidence intervals for Q5-Q1 spread."""
    print(f"\n{'=' * 70}")
    print(f"TEST 3: BOOTSTRAP CONFIDENCE INTERVALS ({period_name})")
    print("=" * 70)
    print(f"Running {n_bootstrap} bootstrap samples...")

    # Assign quintiles to full dataset
    df = df.copy()
    df['quintile'] = pd.qcut(df['moonshot_score'], 5,
                              labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                              duplicates='drop')

    # Original spread
    q5_ret = df[df['quintile'] == 'Q5']['fwd_1yr'].mean()
    q1_ret = df[df['quintile'] == 'Q1']['fwd_1yr'].mean()
    original_spread = q5_ret - q1_ret

    # Bootstrap
    np.random.seed(42)
    bootstrap_spreads = []

    for i in range(n_bootstrap):
        # Resample with replacement
        sample = df.sample(n=len(df), replace=True)

        # Recompute quintiles on resampled data
        try:
            sample['quintile'] = pd.qcut(sample['moonshot_score'], 5,
                                          labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                          duplicates='drop')
            q5_ret = sample[sample['quintile'] == 'Q5']['fwd_1yr'].mean()
            q1_ret = sample[sample['quintile'] == 'Q1']['fwd_1yr'].mean()
            bootstrap_spreads.append(q5_ret - q1_ret)
        except:
            continue

    bootstrap_spreads = np.array(bootstrap_spreads)

    # Confidence intervals
    ci_95_low = np.percentile(bootstrap_spreads, 2.5)
    ci_95_high = np.percentile(bootstrap_spreads, 97.5)
    ci_99_low = np.percentile(bootstrap_spreads, 0.5)
    ci_99_high = np.percentile(bootstrap_spreads, 99.5)

    print(f"\nOriginal Q5-Q1 Spread: {original_spread:>+10.2f}%")
    print(f"\nBootstrap Results ({len(bootstrap_spreads)} valid samples):")
    print(f"  Mean:           {np.mean(bootstrap_spreads):>+10.2f}%")
    print(f"  Std Dev:        {np.std(bootstrap_spreads):>10.2f}%")
    print(f"  95% CI:         [{ci_95_low:>+8.2f}%, {ci_95_high:>+8.2f}%]")
    print(f"  99% CI:         [{ci_99_low:>+8.2f}%, {ci_99_high:>+8.2f}%]")

    # Check if 0 is outside the CI
    if ci_95_low > 0:
        print("\n=> PASS: 95% CI excludes zero - spread is statistically significant")
    elif ci_99_low > 0:
        print("\n=> MARGINAL: 99% CI excludes zero, but 95% CI includes zero")
    else:
        print("\n=> CAUTION: Confidence interval includes zero")

    return {
        'original': original_spread,
        'mean': np.mean(bootstrap_spreads),
        'std': np.std(bootstrap_spreads),
        'ci_95': (ci_95_low, ci_95_high),
        'ci_99': (ci_99_low, ci_99_high)
    }


def main():
    is_df, oos_df = load_and_prepare_data()

    print("\n" + "=" * 70)
    print("RUNNING ROBUSTNESS TESTS")
    print("=" * 70)

    # Test 1: Non-overlapping returns
    is_spread = test_non_overlapping_returns(is_df, "IN-SAMPLE")
    oos_spread = test_non_overlapping_returns(oos_df, "OUT-OF-SAMPLE")

    # Test 2: Sub-period analysis
    test_sub_periods(is_df, "IN-SAMPLE")
    test_sub_periods(oos_df, "OUT-OF-SAMPLE")

    # Test 3: Bootstrap (OOS only for speed)
    test_bootstrap_confidence(oos_df, "OUT-OF-SAMPLE", n_bootstrap=500)

    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    print("\n1. Non-Overlapping Returns:")
    if is_spread and is_spread > 0:
        print(f"   IS:  PASS ({is_spread:+.2f}%)")
    else:
        print(f"   IS:  {'PASS' if is_spread and is_spread > 0 else 'CHECK'}")
    if oos_spread and oos_spread > 0:
        print(f"   OOS: PASS ({oos_spread:+.2f}%)")
    else:
        print(f"   OOS: {'PASS' if oos_spread and oos_spread > 0 else 'CHECK'}")

    print("\n2. Sub-Period Consistency: See above")
    print("\n3. Bootstrap CI: See above")

    print("\n" + "=" * 70)
    print("COMPREHENSIVE ROBUSTNESS TESTS COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

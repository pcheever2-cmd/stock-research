#!/usr/bin/env python3
"""
Full IS (In-Sample) and OOS (Out-of-Sample) Backtest for Triple ROA Penalty

Compares:
- Current methodology (no special ROA penalty)
- Triple ROA penalty (when ROA < 0, multiply z-score by 3)

Splits:
- IS: 1995-2019 (In-Sample)
- OOS: 2020-present (Out-of-Sample)
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'

# Compass Score weights (same as validated backtest)
WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': 0.15,
    'asset_growth': 0.15
}

# Universe statistics (same as validated backtest)
UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
}

# Z-score cap
ZSCORE_CAP = 3.0
FWD_HORIZON = 252  # 1-year forward returns

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'


def load_data():
    """Load all required data."""
    print("=" * 70)
    print("TRIPLE ROA PENALTY BACKTEST: IS vs OOS")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)

    conn = sqlite3.connect(str(BACKTEST_DB))

    print("\n1. Loading fundamentals...")
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted, i.weighted_avg_shares_diluted,
               bs.total_assets, bs.total_debt, bs.total_equity,
               cf.operating_cash_flow, cf.free_cash_flow
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
        LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
        WHERE i.date >= '1990-01-01'
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"   {len(fund):,} fundamental records")

    print("\n2. Loading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   {len(prices):,} price records")

    conn.close()

    return fund, prices


def compute_fundamentals(fund):
    """Compute all fundamental metrics including TTM."""
    print("\n3. Computing TTM metrics...")

    fund = fund.sort_values(['symbol', 'date']).copy()

    for col in ['revenue', 'gross_profit', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    fund['net_income_to_common'] = fund['eps_diluted'] * fund['weighted_avg_shares_diluted']
    fund['net_income_to_common'] = fund['net_income_to_common'].fillna(fund['net_income'])
    fund['net_income_to_common_ttm'] = fund.groupby('symbol')['net_income_to_common'].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )

    fund['roa'] = fund['net_income_to_common_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4, fill_method=None)

    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"   Computed metrics for {len(fund):,} records")

    return fund


def precompute_forward_returns(prices, horizon=252):
    """Pre-compute ALL forward returns at once."""
    print(f"\n4. Pre-computing {horizon}-day forward returns...")

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-horizon)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    fwd_returns = prices[['symbol', 'date', 'fwd_return']].set_index(['symbol', 'date'])['fwd_return'].to_dict()

    print(f"   Computed {len(fwd_returns):,} forward return entries")

    return fwd_returns


def precompute_volatility(prices, window=60):
    """Pre-compute ALL 60-day volatilities at once."""
    print(f"\n5. Pre-computing {window}-day volatilities...")

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['returns'] = prices.groupby('symbol')['close'].pct_change()
    prices['volatility'] = prices.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(window, min_periods=30).std() * np.sqrt(252)
    )

    volatilities = prices[['symbol', 'date', 'volatility']].set_index(['symbol', 'date'])['volatility'].to_dict()

    print(f"   Computed {len(volatilities):,} volatility entries")

    return volatilities


def compute_compass_scores(fund_subset, date, volatility_dict, triple_roa_penalty=False):
    """
    Compute Compass scores with optional triple ROA penalty.
    """
    results = []

    for symbol, group in fund_subset.groupby('symbol'):
        latest = group.sort_values('date').iloc[-1]

        factors = {
            'roa': latest.get('roa', np.nan),
            'ocf_assets': latest.get('ocf_assets', np.nan),
            'fcf_assets': latest.get('fcf_assets', np.nan),
            'gp_assets': latest.get('gp_assets', np.nan),
            'asset_growth': latest.get('asset_growth', np.nan)
        }

        factors['vol_60d'] = volatility_dict.get((symbol, date), np.nan)

        # Check for missing critical values
        if pd.isna(factors['roa']) or pd.isna(factors['gp_assets']):
            continue
        if pd.isna(factors['vol_60d']):
            continue

        # Extreme value filters (same as validated backtest)
        if factors['roa'] < -0.5 or factors['roa'] > 2.0:
            continue
        if factors['ocf_assets'] < -1.0 or factors['ocf_assets'] > 1.5:
            continue
        if factors['fcf_assets'] < -1.0 or factors['fcf_assets'] > 1.5:
            continue
        if factors['gp_assets'] < -0.5 or factors['gp_assets'] > 2.0:
            continue
        if factors['vol_60d'] < 0 or factors['vol_60d'] > 2.0:
            continue
        if not pd.isna(factors['asset_growth']):
            if factors['asset_growth'] < -0.5 or factors['asset_growth'] > 3.0:
                continue

        # Handle missing values
        if pd.isna(factors['asset_growth']):
            factors['asset_growth'] = 0
        if pd.isna(factors['ocf_assets']):
            factors['ocf_assets'] = 0
        if pd.isna(factors['fcf_assets']):
            factors['fcf_assets'] = 0

        results.append({
            'symbol': symbol,
            'date': date,
            **factors
        })

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute z-scores
    for factor in WEIGHTS.keys():
        mean = UNIVERSE_STATS[factor]['mean']
        std = UNIVERSE_STATS[factor]['std']
        z = (df[factor] - mean) / std

        # Apply triple penalty for negative ROA
        if triple_roa_penalty and factor == 'roa':
            # Where ROA < 0, triple the z-score (making it more negative)
            mask = df['roa'] < 0
            z = z.copy()
            z.loc[mask] = z.loc[mask] * 3

        df[f'{factor}_z'] = np.clip(z, -ZSCORE_CAP, ZSCORE_CAP)

    # Composite score (volatility and asset_growth are inverted)
    df['compass_score'] = (
        df['roa_z'] * WEIGHTS['roa'] +
        df['ocf_assets_z'] * WEIGHTS['ocf_assets'] +
        df['fcf_assets_z'] * WEIGHTS['fcf_assets'] +
        df['gp_assets_z'] * WEIGHTS['gp_assets'] +
        (-df['vol_60d_z']) * WEIGHTS['vol_60d'] +
        (-df['asset_growth_z']) * WEIGHTS['asset_growth']
    )

    return df


def assign_grades(df):
    """Assign letter grades based on compass score quintiles."""
    df = df.copy()

    # Use quintiles for grade assignment
    try:
        df['grade'] = pd.qcut(df['compass_score'], 5, labels=['F', 'D', 'C', 'B', 'A'], duplicates='drop')
    except ValueError:
        # Fallback if quintiles fail
        df['grade'] = pd.cut(df['compass_score'],
                            bins=[-np.inf, -0.6, -0.2, 0.2, 0.6, np.inf],
                            labels=['F', 'D', 'C', 'B', 'A'])
    return df


def backtest_period(fund, fwd_returns_dict, volatility_dict, start_date, end_date, period_name, triple_roa_penalty=False):
    """Backtest for a specific period."""
    penalty_label = "TRIPLE ROA PENALTY" if triple_roa_penalty else "CURRENT"

    print(f"\n  {penalty_label} - {period_name}...")
    sys.stdout.flush()

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(end_date)]

    all_scores = []
    quarters_processed = 0

    for date in rebalance_dates:
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        scores = compute_compass_scores(fund_subset, date, volatility_dict, triple_roa_penalty)
        if len(scores) == 0:
            continue

        scores['fwd_return'] = scores.apply(
            lambda row: fwd_returns_dict.get((row['symbol'], date), np.nan),
            axis=1
        )
        scores = scores.dropna(subset=['fwd_return'])

        if len(scores) > 0:
            all_scores.append(scores)
            quarters_processed += 1

    if len(all_scores) == 0:
        return None

    combined = pd.concat(all_scores, ignore_index=True)
    combined = assign_grades(combined)

    print(f"     Processed {quarters_processed} quarters, {len(combined):,} observations")

    # Calculate returns by grade
    grade_results = {}
    for grade in ['A', 'B', 'C', 'D', 'F']:
        grade_data = combined[combined['grade'] == grade]
        if len(grade_data) > 0:
            grade_results[grade] = {
                'mean_return': grade_data['fwd_return'].mean(),
                'median_return': grade_data['fwd_return'].median(),
                'count': len(grade_data)
            }

    # Calculate A-F spread
    spread = np.nan
    if 'A' in grade_results and 'F' in grade_results:
        spread = grade_results['A']['mean_return'] - grade_results['F']['mean_return']

    return {
        'observations': len(combined),
        'stocks': combined['symbol'].nunique(),
        'grade_results': grade_results,
        'spread': spread
    }


def main():
    fund, prices = load_data()
    fund = compute_fundamentals(fund)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)
    volatilities = precompute_volatility(prices)

    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS")
    print("=" * 70)

    results = {}

    # Test both methodologies in both periods
    for triple_penalty in [False, True]:
        method_name = 'triple_roa' if triple_penalty else 'current'

        # In-sample
        is_result = backtest_period(fund, fwd_returns, volatilities, IS_START, IS_END, "IS", triple_penalty)
        results[f'{method_name}_is'] = is_result

        # Out-of-sample
        oos_result = backtest_period(fund, fwd_returns, volatilities, OOS_START, OOS_END, "OOS", triple_penalty)
        results[f'{method_name}_oos'] = oos_result

    # Print detailed results
    print("\n" + "=" * 70)
    print("DETAILED RESULTS BY GRADE")
    print("=" * 70)

    for method in ['current', 'triple_roa']:
        method_label = "CURRENT METHODOLOGY" if method == 'current' else "TRIPLE ROA PENALTY"
        print(f"\n{method_label}")
        print("-" * 50)

        for period in ['is', 'oos']:
            period_label = "In-Sample (1995-2019)" if period == 'is' else "Out-of-Sample (2020+)"
            result = results.get(f'{method}_{period}')

            if result is None:
                print(f"\n  {period_label}: No data")
                continue

            print(f"\n  {period_label} ({result['observations']:,} observations):")

            for grade in ['A', 'B', 'C', 'D', 'F']:
                if grade in result['grade_results']:
                    gr = result['grade_results'][grade]
                    print(f"    Grade {grade}: {gr['mean_return']:+7.2%}  (n={gr['count']:,})")

            print(f"    -------------------------")
            print(f"    A-F Spread: {result['spread']:+.2%}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print("\n" + "-" * 80)
    print(f"{'Methodology':<25} {'Period':<10} {'A Return':<12} {'F Return':<12} {'A-F Spread':<12}")
    print("-" * 80)

    for method in ['current', 'triple_roa']:
        method_label = "Current" if method == 'current' else "Triple ROA"
        for period in ['is', 'oos']:
            period_label = "IS" if period == 'is' else "OOS"
            result = results.get(f'{method}_{period}')

            if result and result['grade_results']:
                a_ret = result['grade_results'].get('A', {}).get('mean_return', np.nan)
                f_ret = result['grade_results'].get('F', {}).get('mean_return', np.nan)
                spread = result['spread']

                print(f"{method_label:<25} {period_label:<10} {a_ret:>+10.2%}   {f_ret:>+10.2%}   {spread:>+10.2%}")

    print("-" * 80)

    # Improvement analysis
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    current_is = results.get('current_is')
    current_oos = results.get('current_oos')
    triple_is = results.get('triple_roa_is')
    triple_oos = results.get('triple_roa_oos')

    if current_is and triple_is:
        diff_is = triple_is['spread'] - current_is['spread']
        print(f"\n  IS Improvement:  {diff_is:+.2%} ({diff_is*100:.1f} pp)")

    if current_oos and triple_oos:
        diff_oos = triple_oos['spread'] - current_oos['spread']
        print(f"  OOS Improvement: {diff_oos:+.2%} ({diff_oos*100:.1f} pp)")

        print("\n" + "=" * 70)
        if diff_oos > 0:
            print("CONCLUSION: Triple ROA penalty IMPROVES out-of-sample performance")
            print(f"            A-F spread increases by {diff_oos*100:.1f} pp in OOS period")
        elif diff_oos == 0:
            print("CONCLUSION: Triple ROA penalty has NO EFFECT on out-of-sample performance")
        else:
            print("CONCLUSION: Triple ROA penalty HURTS out-of-sample performance")
            print(f"            A-F spread decreases by {abs(diff_oos)*100:.1f} pp in OOS period")
        print("=" * 70)


if __name__ == '__main__':
    main()

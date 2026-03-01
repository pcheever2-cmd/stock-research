#!/usr/bin/env python3
"""
Moonshot Score - Phase 2 Rigorous Validation
=============================================
Following Compass Score methodology:

Phase 2 Tests:
1. Value-weighted quintile returns
2. Transaction cost analysis (20, 50, 100, 200 bps)
3. Fama-French 6-factor regression
4. Sector-neutral performance
5. Non-overlapping returns
6. K-fold cross-validation
7. Rolling 36-month Sharpe
8. Turnover analysis

Requires: Phase 1 results (optimal weights)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Period split
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

# Optimal weights from Phase 1 (will be passed as parameter)
# These should come from multivariate regression results
OPTIMAL_WEIGHTS = {
    'revenue_growth': 0.25,
    'eps_growth': 0.20,
    'gross_margin': 0.15,
    'margin_improvement': 0.10,
    'ebitda_growth': 0.10,
    'roe': 0.05,
    'momentum_12_1': 0.15
}


def load_scored_data(period_start, period_end):
    """Load pre-scored data from Phase 1 or recompute."""
    # This would ideally load cached results from Phase 1
    # For now, simplified version
    print(f"Loading data for {period_start} to {period_end}...")

    # Load from database and compute scores
    # (Simplified - in practice would reuse Phase 1 pipeline)

    return None  # Placeholder


def value_weighted_quintiles(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Compute value-weighted quintile returns."""
    print("\n" + "=" * 80)
    print("VALUE-WEIGHTED QUINTILE ANALYSIS")
    print("=" * 80)

    test_df = df[[score_col, return_col, 'market_cap']].dropna().copy()

    if len(test_df) < 100:
        print("  Insufficient data")
        return None

    # Assign quintiles
    try:
        test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        print("  Failed to create quintiles")
        return None

    # Value-weighted returns by quintile
    results = {}
    for q in [1, 2, 3, 4, 5]:
        q_df = test_df[test_df['quintile'] == q]
        if len(q_df) == 0:
            results[f'Q{q}'] = np.nan
            continue

        # Weight returns by market cap
        total_cap = q_df['market_cap'].sum()
        if total_cap > 0:
            weighted_ret = (q_df[return_col] * q_df['market_cap']).sum() / total_cap
            results[f'Q{q}'] = weighted_ret
        else:
            results[f'Q{q}'] = np.nan

    results['spread'] = results.get('Q5', 0) - results.get('Q1', 0)
    results['count'] = len(test_df)

    print(f"\n  Q1 (Low):  {results['Q1']:+.2%}")
    print(f"  Q5 (High): {results['Q5']:+.2%}")
    print(f"  Spread:    {results['spread']:+.2%}")
    print(f"  N:         {results['count']:,}")

    return results


def transaction_cost_analysis(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Analyze impact of transaction costs."""
    print("\n" + "=" * 80)
    print("TRANSACTION COST ANALYSIS")
    print("=" * 80)

    # Compute monthly turnover
    df_sorted = df.sort_values(['date', score_col], ascending=[True, False])

    # Top quintile stocks each month
    monthly_holdings = {}
    for date in df['date'].unique():
        month_df = df[df['date'] == date].sort_values(score_col, ascending=False)
        n_stocks = len(month_df) // 5  # Top 20%
        top_stocks = set(month_df.head(n_stocks)['symbol'].values)
        monthly_holdings[date] = top_stocks

    # Calculate turnover
    dates_sorted = sorted(monthly_holdings.keys())
    turnovers = []

    for i in range(1, len(dates_sorted)):
        prev_holdings = monthly_holdings[dates_sorted[i-1]]
        curr_holdings = monthly_holdings[dates_sorted[i]]

        if len(prev_holdings) > 0:
            changed = len(curr_holdings - prev_holdings)
            turnover = changed / len(prev_holdings)
            turnovers.append(turnover)

    avg_turnover = np.mean(turnovers) if turnovers else 0

    print(f"\nAverage Monthly Turnover: {avg_turnover:.1%}")

    # Compute gross spread
    test_df = df[[score_col, return_col]].dropna()
    test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1,2,3,4,5], duplicates='drop')
    q_rets = test_df.groupby('quintile', observed=False)[return_col].mean()

    # Annualized returns (assuming quarterly rebalance, 1-year forward horizon)
    gross_spread_annual = (q_rets.get(5, 0) - q_rets.get(1, 0)) * 4  # 4 quarters per year

    print(f"\nGross Spread (annualized): {gross_spread_annual:+.2%}")

    # Transaction cost scenarios
    print(f"\n{'Cost (bps/side)':<20} {'Annual Cost':<15} {'Net Spread'}")
    print("-" * 60)

    for bps in [20, 50, 100, 200]:
        annual_cost = avg_turnover * 12 * (bps / 10000) * 2  # Both sides
        net_spread = gross_spread_annual - annual_cost
        print(f"{bps:<20} {annual_cost:+.2%}          {net_spread:+.2%}")

    return {
        'turnover': avg_turnover,
        'gross_spread_annual': gross_spread_annual
    }


def fama_french_regression(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Regress Q5-Q1 returns against FF6 factors."""
    print("\n" + "=" * 80)
    print("FAMA-FRENCH 6-FACTOR REGRESSION")
    print("=" * 80)

    # This requires FF6 factor data
    # Placeholder for now
    print("\n  NOTE: Requires Fama-French factor data")
    print("  Would regress monthly Q5-Q1 spread against:")
    print("    - Mkt-RF (Market excess return)")
    print("    - SMB (Size)")
    print("    - HML (Value)")
    print("    - RMW (Profitability)")
    print("    - CMA (Investment)")
    print("    - MOM (Momentum)")
    print("\n  Output: Alpha, t-stat, R², factor loadings")

    return None


def sector_neutral_test(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Test if signal works within sectors."""
    print("\n" + "=" * 80)
    print("SECTOR-NEUTRAL ANALYSIS")
    print("=" * 80)

    # Requires sector data
    # Would assign quintiles within each sector
    print("\n  NOTE: Requires sector classification data")
    print("  Would compare:")
    print("    - Global quintiles (current method)")
    print("    - Sector-neutral quintiles (within-sector ranking)")

    return None


def non_overlapping_returns(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Test with non-overlapping 1-year windows."""
    print("\n" + "=" * 80)
    print("NON-OVERLAPPING RETURNS VALIDATION")
    print("=" * 80)

    # Filter to only include observations every 252 trading days
    # This eliminates autocorrelation from overlapping windows

    dates_sorted = sorted(df['date'].unique())
    non_overlapping_dates = []

    last_date = None
    for date in dates_sorted:
        if last_date is None:
            non_overlapping_dates.append(date)
            last_date = date
        else:
            # Check if 252+ days have passed
            if (date - last_date).days >= 365:  # Roughly 252 trading days
                non_overlapping_dates.append(date)
                last_date = date

    no_df = df[df['date'].isin(non_overlapping_dates)].copy()

    print(f"\n  Total observations: {len(df):,}")
    print(f"  Non-overlapping: {len(no_df):,}")
    print(f"  Independent periods: {len(non_overlapping_dates)}")

    # Run quintile analysis
    test_df = no_df[[score_col, return_col]].dropna()

    if len(test_df) < 100:
        print("  Insufficient data")
        return None

    test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1,2,3,4,5], duplicates='drop')
    q_rets = test_df.groupby('quintile', observed=False)[return_col].mean()

    spread = q_rets.get(5, 0) - q_rets.get(1, 0)

    print(f"\n  Q1 (Low):  {q_rets.get(1, np.nan):+.2%}")
    print(f"  Q5 (High): {q_rets.get(5, np.nan):+.2%}")
    print(f"  Spread:    {spread:+.2%}")

    return {'spread': spread, 'count': len(test_df)}


def k_fold_cross_validation(df, k=5):
    """K-fold CV on in-sample data."""
    print("\n" + "=" * 80)
    print(f"{k}-FOLD CROSS-VALIDATION")
    print("=" * 80)

    # Split data into k folds by time
    dates_sorted = sorted(df['date'].unique())
    n_dates = len(dates_sorted)
    fold_size = n_dates // k

    fold_results = []

    for i in range(k):
        # Validation fold
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k-1 else n_dates
        val_dates = dates_sorted[val_start:val_end]

        # Training folds
        train_dates = dates_sorted[:val_start] + dates_sorted[val_end:]

        print(f"\n  Fold {i+1}/{k}:")
        print(f"    Train: {len(train_dates)} periods")
        print(f"    Val:   {len(val_dates)} periods")

        # Would recompute scores on train, test on val
        # Placeholder for now
        fold_results.append({'fold': i+1, 'spread': np.nan})

    print("\n  NOTE: Full implementation requires recomputing scores per fold")

    return fold_results


def rolling_sharpe_analysis(df, score_col='moonshot_raw', return_col='fwd_return'):
    """Compute rolling 36-month Sharpe ratios."""
    print("\n" + "=" * 80)
    print("ROLLING 36-MONTH SHARPE ANALYSIS")
    print("=" * 80)

    # Compute monthly Q5-Q1 spread
    monthly_spreads = []

    for date in sorted(df['date'].unique()):
        month_df = df[df['date'] == date]

        test_df = month_df[[score_col, return_col]].dropna()
        if len(test_df) < 100:
            continue

        try:
            test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1,2,3,4,5], duplicates='drop')
            q_rets = test_df.groupby('quintile', observed=False)[return_col].mean()
            spread = q_rets.get(5, 0) - q_rets.get(1, 0)
            monthly_spreads.append({'date': date, 'spread': spread})
        except:
            continue

    spread_df = pd.DataFrame(monthly_spreads).set_index('date')

    # Rolling 36-month Sharpe
    spread_df['rolling_mean'] = spread_df['spread'].rolling(36).mean()
    spread_df['rolling_std'] = spread_df['spread'].rolling(36).std()
    spread_df['sharpe_36m'] = spread_df['rolling_mean'] / spread_df['rolling_std'] * np.sqrt(12)

    sharpe_36m = spread_df['sharpe_36m'].dropna()

    if len(sharpe_36m) > 0:
        print(f"\n  Min 36-mo Sharpe:  {sharpe_36m.min():+.2f}")
        print(f"  Max 36-mo Sharpe:  {sharpe_36m.max():+.2f}")
        print(f"  Mean 36-mo Sharpe: {sharpe_36m.mean():+.2f}")
        print(f"  Current Sharpe:    {sharpe_36m.iloc[-1]:+.2f}")
        print(f"  Negative periods:  {(sharpe_36m < 0).sum()}")

    return sharpe_36m


def main():
    print("=" * 80)
    print("MOONSHOT SCORE - PHASE 2 RIGOROUS VALIDATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("NOTICE")
    print("=" * 80)
    print("\nThis is a TEMPLATE for Phase 2 validation.")
    print("It requires:")
    print("  1. Scored dataset from Phase 1 (optimal weights)")
    print("  2. Fama-French factor data")
    print("  3. Sector classification data")
    print("\nEach test is implemented as a function that can be called")
    print("once the prerequisite data is available.")

    print("\n" + "=" * 80)
    print("TEST ROADMAP")
    print("=" * 80)

    tests = [
        "1. Value-Weighted Quintiles",
        "2. Transaction Cost Analysis (20, 50, 100, 200 bps)",
        "3. Fama-French 6-Factor Regression",
        "4. Sector-Neutral Performance",
        "5. Non-Overlapping Returns",
        "6. K-Fold Cross-Validation (5 folds)",
        "7. Rolling 36-Month Sharpe",
        "8. Turnover Analysis"
    ]

    for test in tests:
        print(f"  {test}")

    print("\n" + "=" * 80)
    print("IMPLEMENTATION NOTES")
    print("=" * 80)

    print("\nFollowing Compass Score methodology exactly:")
    print("  - Use IS (1995-2019) to determine weights")
    print("  - Apply to OOS (2020-2026) with no tuning")
    print("  - Z-scores computed from IS stats only")
    print("  - All tests run on OOS period")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print("\n1. Wait for Phase 1 to complete")
    print("2. Extract optimal weights from multivariate regression")
    print("3. Load scored dataset")
    print("4. Run each test function above")
    print("5. Compile comprehensive results report")

    print("\n" + "=" * 80)
    print("TEMPLATE READY")
    print("=" * 80)


if __name__ == '__main__':
    main()

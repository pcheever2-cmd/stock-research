#!/usr/bin/env python3
"""
Backtest Compass Score WITH Z-SCORE CAPPING (±3.0)
==================================================
Tests the impact of capping extreme z-scores to prevent outliers
from dominating the Compass Score calculation.

MODIFICATION FROM ORIGINAL:
- Z-scores are capped at ±3.0 before aggregation
- This prevents extreme GP/Assets ratios (e.g., 143% for insurance)
  from inflating scores unreasonably

Tests:
- Compass Score with z-score capping (baseline)
- Compass Score + Quality Filters + z-score capping

Time Periods:
- In-sample: 1995-2019
- Out-of-sample: 2020-2026

Goal: Validate that z-score capping doesn't degrade the signal
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Use main project's backtest.db (which has the historical data)
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # 1-year forward returns

# Compass Score factor weights (from research paper)
WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': 0.15,      # Negative weight (lower volatility is better)
    'asset_growth': 0.15  # Negative weight (slower growth is better)
}

# Universe statistics (from research paper)
# These are the fixed benchmarks all stocks are measured against
UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
}


def load_data():
    """Load all required data."""
    print("=" * 80)
    print("COMPASS SCORE QUALITY FILTER BACKTEST (OPTIMIZED)")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Load fundamentals with all needed fields
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

    # Load prices
    print("\n2. Loading prices...")
    sys.stdout.flush()

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   {len(prices):,} price records")

    conn.close()
    sys.stdout.flush()

    return fund, prices


def compute_fundamentals(fund):
    """Compute all fundamental metrics including TTM."""
    print("\n3. Computing TTM metrics...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM metrics
    for col in ['revenue', 'gross_profit', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Compute net income to common shareholders (like actual Compass Score)
    fund['net_income_to_common'] = fund['eps_diluted'] * fund['weighted_avg_shares_diluted']
    fund['net_income_to_common'] = fund['net_income_to_common'].fillna(fund['net_income'])
    fund['net_income_to_common_ttm'] = fund.groupby('symbol')['net_income_to_common'].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )

    # Compass Score factors
    fund['roa'] = fund['net_income_to_common_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']

    # Asset growth (YoY)
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4, fill_method=None)

    # Gross margin (for quality filters)
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'gross_margin']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"   Computed metrics for {len(fund):,} records")
    sys.stdout.flush()

    return fund


def precompute_forward_returns(prices, horizon=252):
    """Pre-compute ALL forward returns at once (major optimization)."""
    print(f"\n4. Pre-computing {horizon}-day forward returns...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-horizon)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    fwd_returns = prices[['symbol', 'date', 'fwd_return']].set_index(['symbol', 'date'])['fwd_return'].to_dict()

    print(f"   Computed {len(fwd_returns):,} forward return entries")
    sys.stdout.flush()

    return fwd_returns


def precompute_volatility(prices, window=60):
    """Pre-compute ALL 60-day volatilities at once (major optimization)."""
    print(f"\n5. Pre-computing {window}-day volatilities...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['returns'] = prices.groupby('symbol')['close'].pct_change()
    prices['volatility'] = prices.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(window, min_periods=30).std() * np.sqrt(252)
    )

    volatilities = prices[['symbol', 'date', 'volatility']].set_index(['symbol', 'date'])['volatility'].to_dict()

    print(f"   Computed {len(volatilities):,} volatility entries")
    sys.stdout.flush()

    return volatilities


def passes_quality_filters(row):
    """
    Apply quality filters to Compass Score.
    Returns: (passes: bool, exclusion_reason: str)
    """
    # Get TTM values
    oi_ni_ratio = np.nan
    operating_income_ttm = row.get('operating_income_ttm', np.nan)
    net_income_ttm = row.get('net_income_to_common_ttm', np.nan)

    if not pd.isna(operating_income_ttm) and not pd.isna(net_income_ttm):
        if abs(net_income_ttm) > 0:
            oi_ni_ratio = abs(operating_income_ttm) / abs(net_income_ttm)

    fcf_ni_ratio = np.nan
    fcf_ttm = row.get('free_cash_flow_ttm', np.nan)
    if not pd.isna(fcf_ttm) and not pd.isna(net_income_ttm) and net_income_ttm != 0:
        fcf_ni_ratio = fcf_ttm / net_income_ttm

    gross_margin = row.get('gross_margin', np.nan)
    revenue_ttm = row.get('revenue_ttm', np.nan)
    debt_to_assets = np.nan
    total_debt = row.get('total_debt', np.nan)
    total_assets = row.get('total_assets', np.nan)
    if not pd.isna(total_debt) and not pd.isna(total_assets) and total_assets > 0:
        debt_to_assets = total_debt / total_assets

    # Filter 1: Operating income / Net income ratio (0.3-3.0)
    if not pd.isna(oi_ni_ratio) and not pd.isna(net_income_ttm):
        if abs(net_income_ttm) > 1_000_000:
            if oi_ni_ratio < 0.3:
                return False, "OI/NI ratio too low (<0.3)"
            elif oi_ni_ratio > 3.0:
                return False, "OI/NI ratio too high (>3.0)"

    # Filter 2: Cash flow quality (FCF/NI > 0.4 for profitable companies)
    if not pd.isna(net_income_ttm) and net_income_ttm > 50_000_000:
        if not pd.isna(fcf_ni_ratio) and fcf_ni_ratio < 0.4:
            return False, "Low cash quality (FCF/NI < 0.4)"

    # Filter 3: Negative FCF despite significant positive net income
    if not pd.isna(net_income_ttm) and not pd.isna(fcf_ttm):
        if net_income_ttm > 50_000_000 and fcf_ttm < -10_000_000:
            return False, "Negative FCF despite positive NI"

    # Filter 4: Gross margin sanity check
    if not pd.isna(gross_margin):
        if gross_margin > 0.98:
            return False, "Suspicious gross margin (>98%)"
        elif gross_margin > 0.95 and (pd.isna(revenue_ttm) or revenue_ttm < 10_000_000_000):
            return False, "Suspicious gross margin (>95% for small company)"

    # Filter 5: Overleveraged (Debt > 2x Assets)
    if not pd.isna(debt_to_assets) and debt_to_assets > 2.0:
        return False, "Overleveraged (Debt/Assets > 2.0)"

    return True, "Passed all quality filters"


def compute_compass_scores(fund_subset, prices, date, volatility_dict, apply_filters=False):
    """
    Compute Compass scores for a specific date.

    Args:
        fund_subset: Fundamentals data
        prices: Price data (not used, kept for compatibility)
        date: Scoring date
        volatility_dict: Pre-computed volatilities
        apply_filters: If True, apply quality filters
    """
    results = []

    # Group by symbol and get latest fundamentals for each
    for symbol, group in fund_subset.groupby('symbol'):
        latest = group.sort_values('date').iloc[-1]

        # Apply quality filters if requested
        if apply_filters:
            passes, reason = passes_quality_filters(latest)
            if not passes:
                continue

        # Get factor values
        factors = {
            'roa': latest.get('roa', np.nan),
            'ocf_assets': latest.get('ocf_assets', np.nan),
            'fcf_assets': latest.get('fcf_assets', np.nan),
            'gp_assets': latest.get('gp_assets', np.nan),
            'asset_growth': latest.get('asset_growth', np.nan)
        }

        # Get pre-computed volatility
        factors['vol_60d'] = volatility_dict.get((symbol, date), np.nan)

        # Check for missing critical values
        if pd.isna(factors['roa']) or pd.isna(factors['gp_assets']):
            continue
        if pd.isna(factors['vol_60d']):
            continue

        # Original Compass Score has extreme value filters
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

        # Handle missing asset_growth
        if pd.isna(factors['asset_growth']):
            factors['asset_growth'] = 0

        results.append({
            'symbol': symbol,
            'date': date,
            **factors
        })

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute z-scores using FIXED universe statistics (from research paper)
    # This ensures absolute comparison across all time periods
    # Z-scores are CAPPED at ±3.0 to prevent extreme outliers from dominating
    for factor in WEIGHTS.keys():
        mean = UNIVERSE_STATS[factor]['mean']
        std = UNIVERSE_STATS[factor]['std']
        df[f'{factor}_z'] = np.clip((df[factor] - mean) / std, -3.0, 3.0)

    # Composite score (negative weights for vol_60d and asset_growth)
    df['compass_score'] = (
        df['roa_z'] * WEIGHTS['roa'] +
        df['ocf_assets_z'] * WEIGHTS['ocf_assets'] +
        df['fcf_assets_z'] * WEIGHTS['fcf_assets'] +
        df['gp_assets_z'] * WEIGHTS['gp_assets'] +
        (-df['vol_60d_z']) * WEIGHTS['vol_60d'] +
        (-df['asset_growth_z']) * WEIGHTS['asset_growth']
    )

    return df


def backtest_period(fund, prices, fwd_returns_dict, volatility_dict, start_date, end_date, period_name, apply_filters=False):
    """
    Backtest Compass Score for a specific period.

    Args:
        fund: Fundamentals data
        prices: Price data (not used, kept for compatibility)
        fwd_returns_dict: Pre-computed forward returns
        volatility_dict: Pre-computed volatilities
        start_date: Start date
        end_date: End date
        period_name: Name of period (for display)
        apply_filters: If True, apply quality filters
    """
    filter_label = "WITH Quality Filters" if apply_filters else "WITHOUT Quality Filters (Baseline)"

    print(f"\n{'=' * 80}")
    print(f"{period_name.upper()} BACKTEST - {filter_label}")
    print(f"({start_date} to {end_date})")
    print(f"{'=' * 80}")
    sys.stdout.flush()

    # Get quarterly rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(end_date)]

    all_scores = []

    for i, date in enumerate(rebalance_dates):
        if (i + 1) % 4 == 0:
            print(f"  Processing {date.strftime('%Y-%m')} ({i+1}/{len(rebalance_dates)})...")
            sys.stdout.flush()

        # Get fundamentals up to this date
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        # Compute scores
        scores = compute_compass_scores(fund_subset, prices, date, volatility_dict, apply_filters=apply_filters)
        if len(scores) == 0:
            continue

        # Get pre-computed forward returns
        scores['fwd_return'] = scores.apply(
            lambda row: fwd_returns_dict.get((row['symbol'], date), np.nan),
            axis=1
        )
        scores = scores.dropna(subset=['fwd_return'])

        all_scores.append(scores)

    if len(all_scores) == 0:
        print(f"  No valid scores for {period_name}")
        return None

    # Combine all scores
    combined = pd.concat(all_scores, ignore_index=True)
    print(f"\n  Total observations: {len(combined):,}")
    print(f"  Unique stocks scored: {combined['symbol'].nunique():,}")
    sys.stdout.flush()

    # Quintile analysis
    combined['quintile'] = pd.qcut(combined['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    print(f"\n  Quintile Performance:")
    print(f"  {'Quintile':<10} {'Avg Return':<12} {'Count':<10}")
    print(f"  {'-' * 35}")

    q_results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = combined[combined['quintile'] == q]
        if len(q_data) > 0:
            avg_ret = q_data['fwd_return'].mean()
            q_results[q] = avg_ret
            print(f"  {q:<10} {avg_ret:>+10.2%}  {len(q_data):>8,}")

    if 'Q5' in q_results and 'Q1' in q_results:
        spread = q_results['Q5'] - q_results['Q1']
        print(f"  {'-' * 35}")
        print(f"  {'Q5 - Q1':<10} {spread:>+10.2%}")
        print(f"  {'=' * 35}")

    sys.stdout.flush()

    return {
        'observations': len(combined),
        'stocks': combined['symbol'].nunique(),
        'q1_return': q_results.get('Q1', np.nan),
        'q5_return': q_results.get('Q5', np.nan),
        'spread': spread if 'Q5' in q_results and 'Q1' in q_results else np.nan
    }


def main():
    # Load data
    fund, prices = load_data()

    # Compute fundamentals
    fund = compute_fundamentals(fund)

    # PRE-COMPUTE EXPENSIVE OPERATIONS (MAJOR OPTIMIZATION)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)
    volatilities = precompute_volatility(prices)

    # ============================================================================
    # BASELINE: Original Compass Score (no quality filters)
    # ============================================================================

    print("\n" + "=" * 80)
    print("BASELINE: ORIGINAL COMPASS SCORE (Research Paper)")
    print("=" * 80)

    is_baseline = backtest_period(fund, prices, fwd_returns, volatilities, IS_START, IS_END, "IN-SAMPLE", apply_filters=False)
    oos_baseline = backtest_period(fund, prices, fwd_returns, volatilities, OOS_START, OOS_END, "OUT-OF-SAMPLE", apply_filters=False)

    # ============================================================================
    # IMPROVED: Compass Score WITH Quality Filters
    # ============================================================================

    print("\n" + "=" * 80)
    print("IMPROVED: COMPASS SCORE WITH QUALITY FILTERS")
    print("=" * 80)

    is_filtered = backtest_period(fund, prices, fwd_returns, volatilities, IS_START, IS_END, "IN-SAMPLE", apply_filters=True)
    oos_filtered = backtest_period(fund, prices, fwd_returns, volatilities, OOS_START, OOS_END, "OUT-OF-SAMPLE", apply_filters=True)

    # ============================================================================
    # COMPARISON SUMMARY
    # ============================================================================

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\nIN-SAMPLE (1995-2019):")
    print(f"{'Metric':<30} {'Baseline':<15} {'With Filters':<15} {'Change':<15}")
    print("-" * 75)

    if is_baseline and is_filtered:
        print(f"{'Observations':<30} {is_baseline['observations']:>14,} {is_filtered['observations']:>14,} "
              f"{is_filtered['observations'] - is_baseline['observations']:>+14,}")
        print(f"{'Unique Stocks':<30} {is_baseline['stocks']:>14,} {is_filtered['stocks']:>14,} "
              f"{is_filtered['stocks'] - is_baseline['stocks']:>+14,}")
        print(f"{'Q5-Q1 Spread':<30} {is_baseline['spread']:>+13.2%} {is_filtered['spread']:>+13.2%} "
              f"{is_filtered['spread'] - is_baseline['spread']:>+13.2%}")

    print("\nOUT-OF-SAMPLE (2020-2026):")
    print(f"{'Metric':<30} {'Baseline':<15} {'With Filters':<15} {'Change':<15}")
    print("-" * 75)

    if oos_baseline and oos_filtered:
        print(f"{'Observations':<30} {oos_baseline['observations']:>14,} {oos_filtered['observations']:>14,} "
              f"{oos_filtered['observations'] - oos_baseline['observations']:>+14,}")
        print(f"{'Unique Stocks':<30} {oos_baseline['stocks']:>14,} {oos_filtered['stocks']:>14,} "
              f"{oos_filtered['stocks'] - oos_baseline['stocks']:>+14,}")
        print(f"{'Q5-Q1 Spread':<30} {oos_baseline['spread']:>+13.2%} {oos_filtered['spread']:>+13.2%} "
              f"{oos_filtered['spread'] - oos_baseline['spread']:>+13.2%}")

    print("\n" + "=" * 80)

    # Verdict
    if oos_baseline and oos_filtered:
        if oos_filtered['spread'] > oos_baseline['spread']:
            improvement = oos_filtered['spread'] - oos_baseline['spread']
            print(f"✓ QUALITY FILTERS IMPROVE PERFORMANCE")
            print(f"  Out-of-sample spread improved by {improvement:+.2%}")
            print(f"  Baseline: {oos_baseline['spread']:+.2%} → With Filters: {oos_filtered['spread']:+.2%}")
            print(f"\n  RECOMMENDATION: SAFE TO IMPLEMENT quality filters in production")
        else:
            decline = oos_baseline['spread'] - oos_filtered['spread']
            print(f"✗ QUALITY FILTERS HURT PERFORMANCE")
            print(f"  Out-of-sample spread declined by {decline:-.2%}")
            print(f"  Baseline: {oos_baseline['spread']:+.2%} → With Filters: {oos_filtered['spread']:+.2%}")
            print(f"\n  RECOMMENDATION: DO NOT IMPLEMENT - filters need refinement")

    print("=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

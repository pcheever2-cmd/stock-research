#!/usr/bin/env python3
"""
Compass Score OOS Validation (Optimized)
=========================================
Tests whether quality filters improve Compass Score performance.

Compares:
- Baseline: Original Compass Score (no filters)
- Improved: Compass Score + Quality Filters

Period: 2020-2026 (Out-of-Sample only)
Rebalancing: Quarterly
Horizon: 1-year forward returns

Key Optimization: Pre-computes all forward returns once instead of
calculating them individually for each stock/quarter.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Test period (OOS only)
TEST_START = '2020-01-01'
TEST_END = '2026-12-31'
FWD_HORIZON = 252  # 1-year forward returns

# Compass Score weights
WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': 0.15,
    'asset_growth': 0.15
}


def load_data():
    """Load fundamentals and prices."""
    print("=" * 80)
    print("COMPASS SCORE OOS VALIDATION (2020-2026)")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted, i.weighted_avg_shares_diluted,
               bs.total_assets, bs.total_debt, bs.total_equity,
               cf.operating_cash_flow, cf.free_cash_flow
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
        LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
        WHERE i.date >= '2015-01-01'
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"   {len(fund):,} fundamental records")

    print("\n2. Loading prices...")
    sys.stdout.flush()

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5 AND date >= '2015-01-01'
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   {len(prices):,} price records")

    conn.close()

    return fund, prices


def compute_ttm_metrics(fund):
    """Compute TTM metrics."""
    print("\n3. Computing TTM metrics...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM aggregations
    for col in ['revenue', 'gross_profit', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Net income to common
    fund['net_income_to_common_ttm'] = (
        fund['eps_diluted'] * fund['weighted_avg_shares_diluted']
    ).fillna(fund['net_income'])
    fund['net_income_to_common_ttm'] = fund.groupby('symbol')['net_income_to_common_ttm'].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )

    # Compass factors
    fund['roa'] = fund['net_income_to_common_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Quality indicators
    fund['oi_ni_ratio'] = np.abs(fund['operating_income_ttm']) / np.abs(fund['net_income_to_common_ttm'])
    fund['fcf_ni_ratio'] = fund['free_cash_flow_ttm'] / fund['net_income_to_common_ttm']
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['debt_to_assets'] = fund['total_debt'] / fund['total_assets']

    print(f"   Computed metrics for {len(fund):,} records")
    sys.stdout.flush()

    return fund


def precompute_forward_returns(prices, horizon=252):
    """Pre-compute ALL forward returns at once (major optimization)."""
    print(f"\n4. Pre-computing {horizon}-day forward returns...")
    sys.stdout.flush()

    # Sort prices
    prices = prices.sort_values(['symbol', 'date']).copy()

    # Compute forward returns using shift
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-horizon)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    # Create lookup dictionary for fast access
    fwd_returns = prices[['symbol', 'date', 'fwd_return']].set_index(['symbol', 'date'])['fwd_return'].to_dict()

    print(f"   Computed {len(fwd_returns):,} forward return entries")
    sys.stdout.flush()

    return fwd_returns


def compute_volatility(prices, symbol, date, window=60):
    """Compute 60-day volatility."""
    sym_prices = prices[(prices['symbol'] == symbol) & (prices['date'] <= date)]
    sym_prices = sym_prices.sort_values('date').tail(window + 1)

    if len(sym_prices) < window:
        return np.nan

    returns = sym_prices['close'].pct_change().dropna()
    return returns.std() * np.sqrt(252)


def passes_quality_filters(row):
    """Check if stock passes quality filters."""
    # Filter 1: OI/NI ratio 0.3-3.0
    oi_ni = row.get('oi_ni_ratio', np.nan)
    ni = row.get('net_income_to_common_ttm', np.nan)
    if not pd.isna(oi_ni) and not pd.isna(ni) and abs(ni) > 1_000_000:
        if oi_ni < 0.3 or oi_ni > 3.0:
            return False

    # Filter 2: FCF/NI > 0.4 for profitable companies
    fcf_ni = row.get('fcf_ni_ratio', np.nan)
    if not pd.isna(ni) and ni > 50_000_000:
        if not pd.isna(fcf_ni) and fcf_ni < 0.4:
            return False

    # Filter 3: Negative FCF despite positive NI
    fcf = row.get('free_cash_flow_ttm', np.nan)
    if not pd.isna(ni) and not pd.isna(fcf):
        if ni > 50_000_000 and fcf < -10_000_000:
            return False

    # Filter 4: Gross margin sanity
    gm = row.get('gross_margin', np.nan)
    rev = row.get('revenue_ttm', np.nan)
    if not pd.isna(gm):
        if gm > 0.98:
            return False
        if gm > 0.95 and (pd.isna(rev) or rev < 10_000_000_000):
            return False

    # Filter 5: Overleveraged
    debt_assets = row.get('debt_to_assets', np.nan)
    if not pd.isna(debt_assets) and debt_assets > 2.0:
        return False

    return True


def compute_scores_for_date(fund_subset, prices, date, fwd_returns, apply_filters=False):
    """Compute Compass scores for a specific rebalance date."""
    results = []

    for symbol, group in fund_subset.groupby('symbol'):
        latest = group.sort_values('date').iloc[-1]

        # Apply quality filters if requested
        if apply_filters and not passes_quality_filters(latest):
            continue

        # Get factors
        factors = {
            'roa': latest.get('roa', np.nan),
            'ocf_assets': latest.get('ocf_assets', np.nan),
            'fcf_assets': latest.get('fcf_assets', np.nan),
            'gp_assets': latest.get('gp_assets', np.nan),
            'asset_growth': latest.get('asset_growth', np.nan)
        }

        # Compute volatility
        factors['vol_60d'] = compute_volatility(prices, symbol, date)

        # Check for missing critical values
        if pd.isna(factors['roa']) or pd.isna(factors['gp_assets']) or pd.isna(factors['vol_60d']):
            continue

        # Extreme value filters (original Compass)
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

        # Get forward return from pre-computed lookup
        fwd_ret = fwd_returns.get((symbol, date), np.nan)
        if pd.isna(fwd_ret):
            continue

        results.append({
            'symbol': symbol,
            'date': date,
            'fwd_return': fwd_ret,
            **factors
        })

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute z-scores
    for factor in WEIGHTS.keys():
        mean = df[factor].mean()
        std = df[factor].std()
        if std > 0:
            df[f'{factor}_z'] = (df[factor] - mean) / std
        else:
            df[f'{factor}_z'] = 0

    # Composite score
    df['compass_score'] = (
        df['roa_z'] * WEIGHTS['roa'] +
        df['ocf_assets_z'] * WEIGHTS['ocf_assets'] +
        df['fcf_assets_z'] * WEIGHTS['fcf_assets'] +
        df['gp_assets_z'] * WEIGHTS['gp_assets'] +
        (-df['vol_60d_z']) * WEIGHTS['vol_60d'] +
        (-df['asset_growth_z']) * WEIGHTS['asset_growth']
    )

    return df


def run_backtest(fund, prices, fwd_returns, apply_filters=False):
    """Run backtest for OOS period."""
    filter_label = "WITH Filters" if apply_filters else "Baseline (No Filters)"

    print(f"\n{'=' * 80}")
    print(f"BACKTEST: {filter_label}")
    print(f"Period: {TEST_START} to {TEST_END}")
    print(f"{'=' * 80}")
    sys.stdout.flush()

    # Quarterly rebalance dates
    rebalance_dates = pd.date_range(start=TEST_START, end=TEST_END, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(TEST_END)]

    all_scores = []

    for i, date in enumerate(rebalance_dates):
        print(f"  Processing {date.strftime('%Y-%m')} ({i+1}/{len(rebalance_dates)})...")
        sys.stdout.flush()

        # Get fundamentals up to this date
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        # Compute scores
        scores = compute_scores_for_date(fund_subset, prices, date, fwd_returns, apply_filters)
        if len(scores) > 0:
            all_scores.append(scores)

    if len(all_scores) == 0:
        print("  No valid scores!")
        return None

    # Combine
    combined = pd.concat(all_scores, ignore_index=True)

    print(f"\n  Total observations: {len(combined):,}")
    print(f"  Unique stocks: {combined['symbol'].nunique():,}")
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

    sys.stdout.flush()

    return {
        'observations': len(combined),
        'stocks': combined['symbol'].nunique(),
        'q1': q_results.get('Q1', np.nan),
        'q5': q_results.get('Q5', np.nan),
        'spread': spread if 'Q5' in q_results and 'Q1' in q_results else np.nan
    }


def main():
    # Load data
    fund, prices = load_data()

    # Compute metrics
    fund = compute_ttm_metrics(fund)

    # Pre-compute forward returns (major optimization!)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)

    # Run baseline
    print("\n" + "=" * 80)
    print("TEST 1: BASELINE (Original Compass Score)")
    print("=" * 80)
    baseline = run_backtest(fund, prices, fwd_returns, apply_filters=False)

    # Run with filters
    print("\n" + "=" * 80)
    print("TEST 2: WITH QUALITY FILTERS")
    print("=" * 80)
    filtered = run_backtest(fund, prices, fwd_returns, apply_filters=True)

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    if baseline and filtered:
        print(f"\nBaseline:")
        print(f"  Observations: {baseline['observations']:,}")
        print(f"  Unique stocks: {baseline['stocks']:,}")
        print(f"  Q5-Q1 spread: {baseline['spread']:+.2%}")

        print(f"\nWith Filters:")
        print(f"  Observations: {filtered['observations']:,}")
        print(f"  Unique stocks: {filtered['stocks']:,}")
        print(f"  Q5-Q1 spread: {filtered['spread']:+.2%}")

        improvement = filtered['spread'] - baseline['spread']
        reduction = (baseline['stocks'] - filtered['stocks']) / baseline['stocks']

        print(f"\nImpact:")
        print(f"  Spread change: {improvement:+.2%}")
        print(f"  Stock reduction: {reduction:.1%}")

        print(f"\n{'=' * 80}")
        if improvement > 0:
            print("✅ FILTERS IMPROVE PERFORMANCE")
        elif improvement > -0.02:
            print("⚖️  FILTERS NEUTRAL (within 2%)")
        else:
            print("❌ FILTERS HURT PERFORMANCE")
        print("=" * 80)

    print(f"\n✓ Validation complete")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

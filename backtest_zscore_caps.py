#!/usr/bin/env python3
"""
Backtest Different Z-Score Caps for Compass Score
==================================================
Tests the impact of different z-score cap levels (2.0 vs 2.5 vs 3.0)
WITHOUT quality filters to find the best configuration.

Goal: Determine optimal z-score cap level while maximizing coverage.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # 1-year forward returns

# Compass Score factor weights
WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': 0.15,
    'asset_growth': 0.15
}

# Universe statistics
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
    print("Z-SCORE CAP COMPARISON BACKTEST")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

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
    sys.stdout.flush()

    return fund


def precompute_forward_returns(prices, horizon=252):
    """Pre-compute ALL forward returns at once."""
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
    """Pre-compute ALL 60-day volatilities at once."""
    print(f"\n5. Pre-computing {window}-day volatilities...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['returns'] = prices.groupby('symbol')['close'].pct_change()
    prices['volatility'] = prices.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(window, min_periods=30).std() * np.sqrt(252)  # Same as validated backtest (no * 100)
    )

    volatilities = prices[['symbol', 'date', 'volatility']].set_index(['symbol', 'date'])['volatility'].to_dict()

    print(f"   Computed {len(volatilities):,} volatility entries")
    sys.stdout.flush()

    return volatilities


def compute_compass_scores(fund_subset, date, volatility_dict, zscore_cap, apply_extreme_filters=True):
    """
    Compute Compass scores with specified z-score cap.

    Args:
        fund_subset: Fundamentals data
        date: Scoring date
        volatility_dict: Pre-computed volatilities
        zscore_cap: Z-score cap level (e.g., 2.0, 2.5, 3.0)
        apply_extreme_filters: If True, apply extreme value exclusions
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

        # Extreme value filters - exact match to validated backtest
        if apply_extreme_filters:
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

        # Handle missing OCF/FCF
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

    # Compute z-scores with specified cap
    for factor in WEIGHTS.keys():
        mean = UNIVERSE_STATS[factor]['mean']
        std = UNIVERSE_STATS[factor]['std']
        df[f'{factor}_z'] = np.clip((df[factor] - mean) / std, -zscore_cap, zscore_cap)

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


def backtest_period(fund, fwd_returns_dict, volatility_dict, start_date, end_date, period_name, zscore_cap, apply_extreme_filters=True):
    """Backtest Compass Score for a specific period with given z-score cap."""
    filter_label = "WITH extreme filters" if apply_extreme_filters else "NO extreme filters"

    print(f"\n  Testing z-score cap ±{zscore_cap} ({filter_label})...")
    sys.stdout.flush()

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(end_date)]

    all_scores = []

    for i, date in enumerate(rebalance_dates):
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        scores = compute_compass_scores(fund_subset, date, volatility_dict, zscore_cap, apply_extreme_filters)
        if len(scores) == 0:
            continue

        scores['fwd_return'] = scores.apply(
            lambda row: fwd_returns_dict.get((row['symbol'], date), np.nan),
            axis=1
        )
        scores = scores.dropna(subset=['fwd_return'])

        all_scores.append(scores)

    if len(all_scores) == 0:
        return None

    combined = pd.concat(all_scores, ignore_index=True)

    combined['quintile'] = pd.qcut(combined['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    q_results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = combined[combined['quintile'] == q]
        if len(q_data) > 0:
            q_results[q] = q_data['fwd_return'].mean()

    spread = np.nan
    if 'Q5' in q_results and 'Q1' in q_results:
        spread = q_results['Q5'] - q_results['Q1']

    return {
        'observations': len(combined),
        'stocks': combined['symbol'].nunique(),
        'q1_return': q_results.get('Q1', np.nan),
        'q5_return': q_results.get('Q5', np.nan),
        'spread': spread
    }


def main():
    fund, prices = load_data()
    fund = compute_fundamentals(fund)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)
    volatilities = precompute_volatility(prices)

    # Test different z-score caps
    caps_to_test = [2.0, 2.5, 3.0]

    # Also test without extreme value filters (the user's request)
    filter_configs = [
        (True, "WITH extreme filters"),
        (False, "NO extreme filters (user request)")
    ]

    results = {}

    for apply_filters, filter_desc in filter_configs:
        print(f"\n{'=' * 80}")
        print(f"TESTING: {filter_desc}")
        print(f"{'=' * 80}")

        for cap in caps_to_test:
            key = f"cap_{cap}_{filter_desc}"

            # In-sample
            is_result = backtest_period(fund, fwd_returns, volatilities, IS_START, IS_END, "IS", cap, apply_filters)

            # Out-of-sample
            oos_result = backtest_period(fund, fwd_returns, volatilities, OOS_START, OOS_END, "OOS", cap, apply_filters)

            results[key] = {
                'cap': cap,
                'filters': apply_filters,
                'is': is_result,
                'oos': oos_result
            }

    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print("\n" + "-" * 100)
    print(f"{'Configuration':<45} {'IS Spread':<12} {'OOS Spread':<12} {'IS Stocks':<12} {'OOS Stocks':<12}")
    print("-" * 100)

    for key, data in results.items():
        cap = data['cap']
        filters = "w/filters" if data['filters'] else "no filters"
        config = f"Cap ±{cap} ({filters})"

        is_spread = data['is']['spread'] if data['is'] else np.nan
        oos_spread = data['oos']['spread'] if data['oos'] else np.nan
        is_stocks = data['is']['stocks'] if data['is'] else 0
        oos_stocks = data['oos']['stocks'] if data['oos'] else 0

        print(f"{config:<45} {is_spread:>+10.2%}   {oos_spread:>+10.2%}   {is_stocks:>10,}   {oos_stocks:>10,}")

    print("-" * 100)

    # Find best config
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    best_oos_spread = -999
    best_config = None

    for key, data in results.items():
        if data['oos'] and not pd.isna(data['oos']['spread']):
            if data['oos']['spread'] > best_oos_spread:
                best_oos_spread = data['oos']['spread']
                best_config = key

    if best_config:
        data = results[best_config]
        print(f"\nBest out-of-sample performance:")
        print(f"  Z-score cap: ±{data['cap']}")
        print(f"  Extreme filters: {'Yes' if data['filters'] else 'No'}")
        print(f"  OOS Q5-Q1 Spread: {data['oos']['spread']:+.2%}")
        print(f"  OOS Stocks scored: {data['oos']['stocks']:,}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

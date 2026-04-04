#!/usr/bin/env python3
"""
Compass Score V2 Validation
===========================
Tests the impact of two modifications to the Compass Score:
1. 3x penalty for negative ROA
2. V-shaped asset growth (penalize both shrinking and excessive growth)

Compares 4 configurations:
A. V1 Baseline (linear ROA, cap at 0 asset growth)
B. 3x ROA penalty only
C. V-shaped asset growth only
D. V2 (both modifications)

Period: 2020-2026 (Out-of-Sample)
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'

# Compass Score weights
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
    'vol_60d': {'mean': 0.45, 'std': 0.25}  # As decimal, not percentage
}

ZSCORE_CAP = 3.0
FWD_HORIZON = 63  # 3-month forward returns (matches V1 validation methodology)

# Period definitions
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

# V-shaped asset growth: ideal growth rate
IDEAL_GROWTH = 0.05  # 5% annual growth is ideal


def load_data():
    """Load all required data."""
    print("=" * 70)
    print("COMPASS SCORE V2 VALIDATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)

    conn = sqlite3.connect(str(BACKTEST_DB))

    # Load US-only symbols from nasdaq_stocks.db
    NASDAQ_DB = PROJECT_ROOT / 'nasdaq_stocks.db'
    print("\n1. Loading US exchange symbols...")
    nasdaq_conn = sqlite3.connect(str(NASDAQ_DB))
    us_symbols_df = pd.read_sql_query("""
        SELECT symbol FROM stock_consensus
        WHERE exchange IN ('NYSE', 'NASDAQ', 'AMEX')
           OR exchange IS NULL  -- Keep NULL as benefit of doubt
    """, nasdaq_conn)
    nasdaq_conn.close()
    us_symbols = set(us_symbols_df['symbol'].tolist())
    print(f"   {len(us_symbols):,} US exchange symbols")

    print("\n2. Loading fundamentals (US only)...")
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
    # Filter to US-only
    fund = fund[fund['symbol'].isin(us_symbols)]
    print(f"   {len(fund):,} fundamental records (US only)")

    print("\n3. Loading prices (US only)...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5 AND date >= '2015-01-01'
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    # Filter to US-only
    prices = prices[prices['symbol'].isin(us_symbols)]
    print(f"   {len(prices):,} price records (US only)")

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


def compute_compass_scores(fund_subset, date, volatility_dict,
                           triple_roa_penalty=False,
                           vshaped_asset_growth=False):
    """
    Compute Compass scores with optional V2 modifications.

    Args:
        triple_roa_penalty: If True, multiply ROA z-score by 3 when ROA < 0
        vshaped_asset_growth: If True, use V-shaped penalty centered on 5% growth
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

        # Extreme value filters
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

    # Compute z-scores for all factors except asset_growth (handled separately)
    for factor in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d']:
        mean = UNIVERSE_STATS[factor]['mean']
        std = UNIVERSE_STATS[factor]['std']
        z = (df[factor] - mean) / std

        # Apply 3x penalty for negative ROA
        if triple_roa_penalty and factor == 'roa':
            mask = df['roa'] < 0
            z = z.copy()
            z.loc[mask] = z.loc[mask] * 3

        df[f'{factor}_z'] = np.clip(z, -ZSCORE_CAP, ZSCORE_CAP)

    # Handle asset growth separately based on configuration
    if vshaped_asset_growth:
        # V-shaped: penalize deviation from ideal growth (5%)
        deviation = np.abs(df['asset_growth'] - IDEAL_GROWTH)
        # Higher deviation = worse (so we use negative z-score of deviation)
        mean_dev = deviation.mean()
        std_dev = deviation.std()
        if std_dev > 0:
            deviation_z = (deviation - mean_dev) / std_dev
        else:
            deviation_z = 0
        df['asset_growth_z'] = np.clip(deviation_z, -ZSCORE_CAP, ZSCORE_CAP)
        # Note: We'll use -asset_growth_z in the formula, so higher deviation = lower score
    else:
        # V1 behavior: cap at 0, then compute z-score
        asset_growth_capped = df['asset_growth'].clip(lower=0)
        mean = UNIVERSE_STATS['asset_growth']['mean']
        std = UNIVERSE_STATS['asset_growth']['std']
        z = (asset_growth_capped - mean) / std
        df['asset_growth_z'] = np.clip(z, -ZSCORE_CAP, ZSCORE_CAP)

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


def run_backtest(fund, fwd_returns_dict, volatility_dict, start_date, end_date,
                 triple_roa_penalty=False, vshaped_asset_growth=False):
    """Run backtest for OOS period with specified configuration."""

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(end_date)]

    all_scores = []

    for date in rebalance_dates:
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        scores = compute_compass_scores(
            fund_subset, date, volatility_dict,
            triple_roa_penalty=triple_roa_penalty,
            vshaped_asset_growth=vshaped_asset_growth
        )

        if len(scores) == 0:
            continue

        scores['fwd_return'] = scores.apply(
            lambda row: fwd_returns_dict.get((row['symbol'], date), np.nan),
            axis=1
        )
        scores = scores.dropna(subset=['fwd_return'])

        if len(scores) > 0:
            all_scores.append(scores)

    if len(all_scores) == 0:
        return None

    combined = pd.concat(all_scores, ignore_index=True)

    # Quintile analysis - Q1 = BEST (highest scores), Q5 = WORST (lowest scores)
    try:
        # Reverse labels so Q1 gets highest scores
        combined['quintile'] = pd.qcut(combined['compass_score'], 5,
                                       labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1'],
                                       duplicates='drop')
    except ValueError:
        return None

    q_results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = combined[combined['quintile'] == q]
        if len(q_data) > 0:
            q_results[q] = q_data['fwd_return'].mean()

    # Spread = Q1 (best) - Q5 (worst) -> should be positive if quality works
    spread = q_results.get('Q1', 0) - q_results.get('Q5', 0)
    correlation = combined['compass_score'].corr(combined['fwd_return'])

    # Check monotonicity: Q1 >= Q2 >= Q3 >= Q4 >= Q5
    qrets = [q_results.get(f'Q{i}', 0) for i in range(1, 6)]
    is_monotonic = all(qrets[i] >= qrets[i+1] for i in range(4))

    return {
        'observations': len(combined),
        'stocks': combined['symbol'].nunique(),
        'quintile_returns': q_results,
        'spread': spread,
        'correlation': correlation,
        'monotonic': is_monotonic,
        'data': combined
    }


def run_year_analysis(combined_data, config_name):
    """Analyze results by year. Q1 = best (highest scores), Q5 = worst (lowest scores)."""
    combined_data = combined_data.copy()
    combined_data['year'] = combined_data['date'].dt.year

    print(f"\n  {config_name} - Year-by-Year (Q1=Best, Q5=Worst):")
    print(f"  {'Year':<6} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Q5':>10} {'Q1-Q5':>10} {'Mono?':>8}")
    print(f"  {'-'*76}")

    for year in sorted(combined_data['year'].unique()):
        year_data = combined_data[combined_data['year'] == year].copy()

        try:
            # Reverse labels so Q1 = highest scores (best)
            year_data['quintile'] = pd.qcut(year_data['compass_score'], 5,
                                           labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1'],
                                           duplicates='drop')
        except ValueError:
            continue

        qrets = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = year_data[year_data['quintile'] == q]
            if len(q_data) > 0:
                qrets[q] = q_data['fwd_return'].mean()

        if 'Q1' in qrets and 'Q5' in qrets:
            spread = qrets['Q1'] - qrets['Q5']  # Best - Worst
            qlist = [qrets.get(f'Q{i}', 0) for i in range(1, 6)]
            is_mono = all(qlist[i] >= qlist[i+1] for i in range(4))  # Q1 >= Q2 >= ... >= Q5

            print(f"  {year:<6} {qrets.get('Q1', 0):>+9.1%} {qrets.get('Q2', 0):>+9.1%} "
                  f"{qrets.get('Q3', 0):>+9.1%} {qrets.get('Q4', 0):>+9.1%} "
                  f"{qrets.get('Q5', 0):>+9.1%} {spread:>+9.1%} {'Yes' if is_mono else 'No':>8}")


def main():
    fund, prices = load_data()
    fund = compute_fundamentals(fund)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)
    volatilities = precompute_volatility(prices)

    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS (OOS: 2020-2026)")
    print("=" * 70)

    configs = [
        ('A. V1 Baseline', False, False),
        ('B. +3x ROA penalty', True, False),
        ('C. +V-shape asset growth', False, True),
        ('D. V2 (both mods)', True, True),
    ]

    results = {}

    for name, triple_roa, vshaped in configs:
        print(f"\n  Testing {name}...")
        sys.stdout.flush()

        result = run_backtest(
            fund, fwd_returns, volatilities,
            OOS_START, OOS_END,
            triple_roa_penalty=triple_roa,
            vshaped_asset_growth=vshaped
        )

        if result:
            results[name] = result
            print(f"    Observations: {result['observations']:,}")
            print(f"    Q1-Q5 Spread: {result['spread']:+.2%}")
            print(f"    Correlation:  {result['correlation']:+.4f}")
            print(f"    Monotonic:    {'Yes' if result['monotonic'] else 'No'}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'N Obs':>10} {'Q1-Q5 Spread':>14} {'Correlation':>12} {'Monotonic':>10}")
    print("-" * 80)

    baseline_spread = results.get('A. V1 Baseline', {}).get('spread', 0)

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in results:
            r = results[name]
            diff = r['spread'] - baseline_spread if name != 'A. V1 Baseline' else 0
            diff_str = f"({diff:+.2%})" if name != 'A. V1 Baseline' else ""
            print(f"{name:<30} {r['observations']:>10,} {r['spread']:>+12.2%} {diff_str:>8} "
                  f"{r['correlation']:>+11.4f} {'Yes' if r['monotonic'] else 'No':>10}")

    # Quintile detail
    print("\n" + "=" * 70)
    print("QUINTILE RETURNS BY CONFIGURATION (Q1=Best, Q5=Worst)")
    print("=" * 70)

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in results:
            qrets = results[name]['quintile_returns']
            print(f"\n  {name}:")
            print(f"    Q1: {qrets.get('Q1', 0):>+8.2%}  Q2: {qrets.get('Q2', 0):>+8.2%}  "
                  f"Q3: {qrets.get('Q3', 0):>+8.2%}  Q4: {qrets.get('Q4', 0):>+8.2%}  "
                  f"Q5: {qrets.get('Q5', 0):>+8.2%}")

    # Year-by-year analysis
    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR BREAKDOWN")
    print("=" * 70)

    for name in ['A. V1 Baseline', 'D. V2 (both mods)']:
        if name in results:
            run_year_analysis(results[name]['data'], name)

    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    v1_spread = results.get('A. V1 Baseline', {}).get('spread', 0)
    v2_spread = results.get('D. V2 (both mods)', {}).get('spread', 0)

    diff = v2_spread - v1_spread

    if diff > 0.005:  # More than 0.5% improvement
        print(f"\n  V2 IMPROVES OOS performance by {diff:+.2%} ({diff*100:.1f} pp)")
        print("  Recommendation: Consider adopting V2 modifications")
    elif diff > -0.005:  # Within +/- 0.5%
        print(f"\n  V2 has SIMILAR performance to V1 ({diff:+.2%})")
        print("  V2 modifications have minimal impact on OOS results")
    else:
        print(f"\n  V2 HURTS OOS performance by {diff:+.2%} ({abs(diff)*100:.1f} pp)")
        print("  Recommendation: Keep V1 methodology")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

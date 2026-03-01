#!/usr/bin/env python3
"""
Compass Score Backtest with TTM Methodology
============================================
Tests the Compass Score quality factor model using TTM (Trailing Twelve Months)
data across historical periods.

Methodology:
- At each quarterly rebalance date, compute Compass Scores using only
  data available at that time (no look-ahead bias)
- Rank stocks into quintiles (Q1=top 20%, Q5=bottom 20%)
- Measure forward returns over 1M, 3M, 6M, 12M periods
- Compare quintile performance to validate the scoring methodology
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Universe statistics (from research paper)
UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
}


def load_all_data(conn):
    """Load all historical data needed for backtesting."""
    print("Loading historical data...")
    sys.stdout.flush()

    # Load all prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Load all fundamentals
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    sys.stdout.flush()
    return prices, fund


def compute_ttm_fundamentals(fund_df):
    """Compute TTM metrics for all stocks."""
    print("Computing TTM metrics for all stocks...")
    sys.stdout.flush()

    fund = fund_df.sort_values(['symbol', 'date']).copy()

    # Compute TTM for flow metrics (sum of last 4 quarters)
    for col in ['net_income', 'gross_profit', 'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Compute ratios using TTM for flows, latest for balance sheet
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']

    # Asset growth: YoY change
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"  Computed TTM for {fund['symbol'].nunique():,} stocks")
    sys.stdout.flush()
    return fund


def compute_volatility(prices_df, symbol, as_of_date, vol_lookup):
    """Look up pre-computed volatility for a symbol as of a given date."""
    if symbol not in vol_lookup:
        return np.nan

    sym_vol = vol_lookup[symbol]
    valid = sym_vol[sym_vol['date'] <= as_of_date]
    if len(valid) == 0:
        return np.nan

    return valid.iloc[-1]['vol_60d']


def precompute_all_volatilities(prices_df):
    """Pre-compute 60-day rolling volatility for all symbols."""
    print("Pre-computing volatilities for all stocks...")
    sys.stdout.flush()

    prices = prices_df.sort_values(['symbol', 'date']).copy()

    # Compute returns
    prices['ret'] = prices.groupby('symbol')['close'].pct_change()

    # Rolling 60-day volatility
    prices['vol_60d'] = prices.groupby('symbol')['ret'].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    )

    # Create lookup dict
    vol_lookup = {}
    for symbol in prices['symbol'].unique():
        sym_data = prices[prices['symbol'] == symbol][['date', 'vol_60d']].dropna()
        if len(sym_data) > 0:
            vol_lookup[symbol] = sym_data

    print(f"  Volatilities computed for {len(vol_lookup):,} stocks")
    sys.stdout.flush()
    return vol_lookup


def compute_compass_score(factors):
    """Compute raw Compass Score from factor values."""
    # Quality filters
    if abs(factors['roa']) > 0.5:
        return np.nan
    if abs(factors['ocf_assets']) > 0.75:
        return np.nan
    if abs(factors['fcf_assets']) > 0.75:
        return np.nan
    if abs(factors['gp_assets']) > 1.0:
        return np.nan
    if factors['asset_growth'] < -0.5:
        return np.nan

    # Check for missing values
    for val in factors.values():
        if pd.isna(val):
            return np.nan

    # Compute z-scores
    z_scores = {}
    for f, val in factors.items():
        z_scores[f] = (val - UNIVERSE_STATS[f]['mean']) / UNIVERSE_STATS[f]['std']

    # Raw Compass Score (weighted z-score combination)
    raw_score = (
        z_scores['roa'] * 0.20 +
        z_scores['ocf_assets'] * 0.15 +
        z_scores['fcf_assets'] * 0.15 +
        z_scores['gp_assets'] * 0.20 +
        (-z_scores['vol_60d']) * 0.15 +
        (-z_scores['asset_growth']) * 0.15
    )

    return raw_score


def get_rebalance_dates(prices_df, start_year=2020, end_year=2025):
    """Get quarterly rebalance dates (end of Jan, Apr, Jul, Oct)."""
    all_dates = prices_df['date'].sort_values().unique()
    all_dates = pd.to_datetime(all_dates)

    rebalance_dates = []
    for year in range(start_year, end_year + 1):
        for month in [1, 4, 7, 10]:  # Quarterly
            # Find the last trading day of the month
            month_end = pd.Timestamp(year=year, month=month, day=28)
            if month == 1:
                month_end = pd.Timestamp(year=year, month=1, day=31)
            elif month == 4:
                month_end = pd.Timestamp(year=year, month=4, day=30)
            elif month == 7:
                month_end = pd.Timestamp(year=year, month=7, day=31)
            elif month == 10:
                month_end = pd.Timestamp(year=year, month=10, day=31)

            # Find closest date in our data
            valid_dates = all_dates[all_dates <= month_end]
            if len(valid_dates) > 0:
                closest = valid_dates[-1]
                if closest not in rebalance_dates:
                    rebalance_dates.append(closest)

    return sorted(rebalance_dates)


def compute_forward_returns_lookup(symbol, from_date, fwd_returns_lookup):
    """Look up pre-computed forward returns for a symbol as of a given date."""
    if symbol not in fwd_returns_lookup:
        return np.nan

    sym_data = fwd_returns_lookup[symbol]
    # Find the closest date on or after from_date
    valid = sym_data[sym_data['date'] >= from_date]
    if len(valid) == 0:
        return np.nan

    return valid.iloc[0]['fwd_63d']


def precompute_forward_returns(prices_df):
    """Pre-compute 63-day forward returns for all symbols."""
    print("Pre-computing forward returns for all stocks...")
    sys.stdout.flush()

    prices = prices_df.sort_values(['symbol', 'date']).copy()

    # Compute forward returns (shift by -63 days)
    prices['fwd_price'] = prices.groupby('symbol')['close'].shift(-63)
    prices['fwd_63d'] = ((prices['fwd_price'] / prices['close']) - 1) * 100

    # Create lookup dict
    fwd_lookup = {}
    for symbol in prices['symbol'].unique():
        sym_data = prices[prices['symbol'] == symbol][['date', 'fwd_63d']].dropna()
        if len(sym_data) > 0:
            fwd_lookup[symbol] = sym_data

    print(f"  Forward returns computed for {len(fwd_lookup):,} stocks")
    sys.stdout.flush()
    return fwd_lookup


def run_backtest():
    """Run the Compass Score backtest."""
    print("=" * 60)
    print("COMPASS SCORE BACKTEST (TTM METHODOLOGY)")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)
    prices, fund = load_all_data(conn)
    fund = compute_ttm_fundamentals(fund)
    conn.close()

    # Pre-compute volatilities for all stocks
    vol_lookup = precompute_all_volatilities(prices)

    # Get rebalance dates
    rebalance_dates = get_rebalance_dates(prices, start_year=2020, end_year=2025)
    print(f"\nBacktest period: {len(rebalance_dates)} quarterly rebalances")
    print(f"  From: {rebalance_dates[0].strftime('%Y-%m-%d')}")
    print(f"  To: {rebalance_dates[-1].strftime('%Y-%m-%d')}")
    sys.stdout.flush()

    all_results = []

    for i, rebal_date in enumerate(rebalance_dates):
        print(f"\n[{i+1}/{len(rebalance_dates)}] Rebalance: {rebal_date.strftime('%Y-%m-%d')}")
        sys.stdout.flush()

        # Get fundamentals available as of this date
        # Use filing_date or just date <= rebal_date
        available_fund = fund[fund['date'] <= rebal_date]

        # Get latest fundamentals per symbol
        latest_fund = available_fund.sort_values('date').groupby('symbol').last().reset_index()

        # Get symbols with price data
        price_symbols = set(prices[prices['date'] <= rebal_date]['symbol'].unique())
        fund_symbols = set(latest_fund['symbol'].unique())
        eligible_symbols = price_symbols & fund_symbols

        print(f"  {len(eligible_symbols):,} eligible stocks")

        # Compute scores for all eligible stocks
        scores = []
        for symbol in eligible_symbols:
            sym_fund = latest_fund[latest_fund['symbol'] == symbol].iloc[0]

            # Look up pre-computed volatility
            vol = compute_volatility(prices, symbol, rebal_date, vol_lookup)
            if pd.isna(vol):
                continue

            factors = {
                'roa': sym_fund['roa'],
                'ocf_assets': sym_fund['ocf_assets'],
                'fcf_assets': sym_fund['fcf_assets'],
                'gp_assets': sym_fund['gp_assets'],
                'asset_growth': sym_fund['asset_growth'],
                'vol_60d': vol
            }

            raw_score = compute_compass_score(factors)
            if not pd.isna(raw_score):
                scores.append({
                    'symbol': symbol,
                    'rebal_date': rebal_date,
                    'raw_score': raw_score
                })

        if len(scores) < 50:
            print(f"  Skipping - only {len(scores)} valid scores")
            continue

        scores_df = pd.DataFrame(scores)

        # Assign quintiles (Q1 = top 20%, Q5 = bottom 20%)
        scores_df['quintile'] = pd.qcut(
            scores_df['raw_score'],
            q=5,
            labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']  # Q1 is best
        )

        print(f"  {len(scores_df):,} stocks scored, computing forward returns...")

        # Compute forward returns for each stock
        for idx, row in scores_df.iterrows():
            returns = compute_forward_returns(prices, row['symbol'], rebal_date)
            for k, v in returns.items():
                scores_df.loc[idx, k] = v

        all_results.append(scores_df)
        sys.stdout.flush()

    # Combine all results
    results = pd.concat(all_results, ignore_index=True)

    # Analyze by quintile
    print("\n" + "=" * 60)
    print("RESULTS BY QUINTILE")
    print("=" * 60)

    for period in ['ret_21d', 'ret_63d', 'ret_126d', 'ret_252d']:
        period_name = {
            'ret_21d': '1-Month',
            'ret_63d': '3-Month',
            'ret_126d': '6-Month',
            'ret_252d': '12-Month'
        }[period]

        print(f"\n{period_name} Returns:")
        quintile_stats = results.groupby('quintile')[period].agg(['mean', 'median', 'std', 'count'])
        quintile_stats = quintile_stats.sort_index(ascending=False)  # Q1 first

        print(f"{'Quintile':<10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Count':>10}")
        print("-" * 50)
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            if q in quintile_stats.index:
                row = quintile_stats.loc[q]
                print(f"{q:<10} {row['mean']:>10.2f}% {row['median']:>10.2f}% {row['std']:>10.2f}% {int(row['count']):>10}")

        # Spread (Q1 - Q5)
        if 'Q1' in quintile_stats.index and 'Q5' in quintile_stats.index:
            spread = quintile_stats.loc['Q1', 'mean'] - quintile_stats.loc['Q5', 'mean']
            print(f"\nQ1-Q5 Spread: {spread:+.2f}%")

    # Save results
    results.to_csv(PROJECT_ROOT / 'compass_backtest_results.csv', index=False)
    print(f"\nResults saved to compass_backtest_results.csv")

    print("\n" + "=" * 60)
    print("BACKTEST COMPLETE")
    print("=" * 60)
    sys.stdout.flush()

    return results


def run_comparison_backtest():
    """Run backtest comparing single-quarter vs TTM methodology."""
    print("=" * 60)
    print("METHODOLOGY COMPARISON: Single Quarter vs TTM")
    print("=" * 60)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)
    prices, fund_raw = load_all_data(conn)
    conn.close()

    # Pre-compute volatilities for all stocks
    vol_lookup = precompute_all_volatilities(prices)

    # Pre-compute forward returns for all stocks
    fwd_lookup = precompute_forward_returns(prices)

    # Get rebalance dates (full historical period for robust testing)
    rebalance_dates = get_rebalance_dates(prices, start_year=2015, end_year=2024)
    print(f"\nComparing across {len(rebalance_dates)} rebalances (2015-2024)")

    # Prepare both methodologies
    fund_sq = fund_raw.copy()  # Single quarter
    fund_sq['roa'] = fund_sq['net_income'] / fund_sq['total_assets']
    fund_sq['ocf_assets'] = fund_sq['operating_cash_flow'] / fund_sq['total_assets']
    fund_sq['fcf_assets'] = fund_sq['free_cash_flow'] / fund_sq['total_assets']
    fund_sq['gp_assets'] = fund_sq['gross_profit'] / fund_sq['total_assets']
    fund_sq = fund_sq.sort_values(['symbol', 'date'])
    fund_sq['asset_growth'] = fund_sq.groupby('symbol')['total_assets'].pct_change(4)

    fund_ttm = compute_ttm_fundamentals(fund_raw)

    results_sq = []
    results_ttm = []

    # Pre-index the prices DataFrame for faster lookups
    print("  Pre-indexing price data...")
    prices_indexed = prices.set_index(['symbol', 'date']).sort_index()
    sys.stdout.flush()

    for i, rebal_date in enumerate(rebalance_dates):
        print(f"  [{i+1}/{len(rebalance_dates)}] {rebal_date.strftime('%Y-%m-%d')}...", end=" ", flush=True)

        for fund, results_list, method in [(fund_sq, results_sq, 'SQ'), (fund_ttm, results_ttm, 'TTM')]:
            available_fund = fund[fund['date'] <= rebal_date]
            latest_fund = available_fund.sort_values('date').groupby('symbol').last()

            # Use index directly for faster lookup
            fund_symbols = set(latest_fund.index)
            price_symbols = set(vol_lookup.keys())
            eligible_symbols = fund_symbols & price_symbols

            scores = []
            for symbol in eligible_symbols:
                sym_fund = latest_fund.loc[symbol]
                vol = compute_volatility(prices, symbol, rebal_date, vol_lookup)
                if pd.isna(vol):
                    continue

                factors = {
                    'roa': sym_fund['roa'],
                    'ocf_assets': sym_fund['ocf_assets'],
                    'fcf_assets': sym_fund['fcf_assets'],
                    'gp_assets': sym_fund['gp_assets'],
                    'asset_growth': sym_fund['asset_growth'],
                    'vol_60d': vol
                }

                raw_score = compute_compass_score(factors)
                if not pd.isna(raw_score):
                    scores.append({'symbol': symbol, 'raw_score': raw_score})

            if len(scores) < 50:
                continue

            scores_df = pd.DataFrame(scores)
            scores_df['quintile'] = pd.qcut(scores_df['raw_score'], q=5,
                                            labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1'])

            # Vectorized forward returns lookup
            scores_df['ret_63d'] = scores_df['symbol'].apply(
                lambda s: compute_forward_returns_lookup(s, rebal_date, fwd_lookup)
            )

            scores_df['method'] = method
            scores_df['rebal_date'] = rebal_date
            results_list.append(scores_df)
            print(f"{method}({len(scores_df)})", end=" ", flush=True)

        print()  # Newline after each rebalance date

    print("Processing complete.")
    sys.stdout.flush()

    # Combine and compare
    all_sq = pd.concat(results_sq, ignore_index=True)
    all_ttm = pd.concat(results_ttm, ignore_index=True)

    print("\n" + "=" * 60)
    print("3-MONTH RETURNS COMPARISON")
    print("=" * 60)

    for method, df, name in [('SQ', all_sq, 'Single Quarter'), ('TTM', all_ttm, 'TTM (Trailing 12M)')]:
        print(f"\n{name}:")
        stats = df.groupby('quintile')['ret_63d'].mean()
        stats = stats.reindex(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        spread = stats['Q1'] - stats['Q5']
        print(f"  Q1 (top): {stats['Q1']:.2f}%")
        print(f"  Q5 (bot): {stats['Q5']:.2f}%")
        print(f"  Spread:   {spread:+.2f}%")

    # Save combined results
    all_results = pd.concat([all_sq, all_ttm], ignore_index=True)
    all_results.to_csv(PROJECT_ROOT / 'compass_comparison_results.csv', index=False)
    print(f"\nResults saved to compass_comparison_results.csv")

    # Print detailed quintile statistics
    print("\n" + "=" * 60)
    print("DETAILED QUINTILE RETURNS")
    print("=" * 60)
    for method in ['SQ', 'TTM']:
        print(f"\n{method}:")
        df = all_results[all_results['method'] == method]
        stats = df.groupby('quintile')['ret_63d'].agg(['mean', 'median', 'count'])
        stats = stats.reindex(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        print(stats)

    print("\n" + "=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true', help='Compare SQ vs TTM')
    args = parser.parse_args()

    if args.compare:
        run_comparison_backtest()
    else:
        run_backtest()

#!/usr/bin/env python3
"""
Backtest All Three Scoring Systems
===================================
Tests predictive power of:
1. Compass Score (Quality) - already validated, 6-month forward returns
2. Moonshot Score (Growth) - 1-year, 2-year forward returns
3. Momentum Score (Short-term) - 1-month, 3-month forward returns

Uses quintile analysis: do high-scoring stocks outperform low-scoring stocks?
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Backtest parameters
BACKTEST_START = '2015-01-01'
BACKTEST_END = '2023-12-31'  # Leave 2024+ for out-of-sample


def load_all_data():
    """Load all price and fundamental data for backtesting."""
    print("Loading all data...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Load all prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Load fundamentals
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.revenue, i.gross_profit, i.net_income, i.eps_diluted as eps,
               b.total_assets,
               c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    conn.close()

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")
    sys.stdout.flush()

    return prices, fund


def precompute_ttm_fundamentals(fund):
    """Pre-compute TTM metrics for all fundamentals."""
    print("Computing TTM metrics...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date'])

    # TTM for flow metrics
    for col in ['revenue', 'gross_profit', 'net_income', 'operating_cash_flow', 'free_cash_flow', 'eps']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Compute ratios
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # YoY growth
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'gross_margin',
                'asset_growth', 'revenue_growth', 'eps_growth', 'margin_improvement']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    return fund


def precompute_price_metrics(prices):
    """Pre-compute all price-based metrics."""
    print("Computing price metrics...")
    sys.stdout.flush()

    # Create lookup dictionary for fast access
    price_dict = {}
    vol_dict = {}

    for symbol in prices['symbol'].unique():
        sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
        if len(sym_prices) < 252:
            continue

        sym_prices = sym_prices.set_index('date')

        # 60-day volatility
        returns = sym_prices['close'].pct_change()
        vol = returns.rolling(60, min_periods=30).std() * np.sqrt(252) * 100

        # Store lookups
        price_dict[symbol] = sym_prices['close'].to_dict()
        vol_dict[symbol] = vol.to_dict()

    print(f"  {len(price_dict):,} symbols with price data")
    return price_dict, vol_dict


def precompute_forward_returns(prices, horizons=[21, 63, 126, 252, 504]):
    """Pre-compute forward returns for all horizons."""
    print(f"Computing forward returns for horizons: {horizons}...")
    sys.stdout.flush()

    returns_dict = {h: {} for h in horizons}

    for symbol in prices['symbol'].unique():
        sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
        sym_prices = sym_prices.set_index('date')

        for h in horizons:
            fwd_ret = sym_prices['close'].pct_change(h).shift(-h)
            returns_dict[h][symbol] = fwd_ret.to_dict()

    return returns_dict


def compute_compass_score(fund_row, vol, universe_stats):
    """Compute Compass Score for a single observation."""
    factors = {
        'roa': fund_row['roa'],
        'ocf_assets': fund_row['ocf_assets'],
        'fcf_assets': fund_row['fcf_assets'],
        'gp_assets': fund_row['gp_assets'],
        'asset_growth': fund_row['asset_growth'],
        'vol_60d': vol
    }

    # Check for missing values
    for val in factors.values():
        if pd.isna(val):
            return np.nan

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

    # Compute z-scores
    z_scores = {}
    for f, val in factors.items():
        z_scores[f] = (val - universe_stats[f]['mean']) / universe_stats[f]['std']

    # Raw score
    raw_score = (
        z_scores['roa'] * 0.20 +
        z_scores['ocf_assets'] * 0.15 +
        z_scores['fcf_assets'] * 0.15 +
        z_scores['gp_assets'] * 0.20 +
        (-z_scores['vol_60d']) * 0.15 +
        (-z_scores['asset_growth']) * 0.15
    )

    return raw_score


def compute_moonshot_score(fund_row, momentum_12_1):
    """Compute Moonshot Score for a single observation."""
    factors = {
        'revenue_growth': fund_row.get('revenue_growth', np.nan),
        'eps_growth': fund_row.get('eps_growth', np.nan),
        'gross_margin': fund_row.get('gross_margin', np.nan),
        'margin_improvement': fund_row.get('margin_improvement', np.nan),
        'market_cap': fund_row.get('market_cap', np.nan),
        'momentum_12_1': momentum_12_1
    }

    # Check critical values
    if pd.isna(factors['revenue_growth']) or pd.isna(factors['gross_margin']):
        return np.nan
    if pd.isna(factors['market_cap']) or factors['market_cap'] <= 0:
        return np.nan

    # Quality filters
    if factors['revenue_growth'] > 5.0 or factors['revenue_growth'] < -0.8:
        return np.nan
    if factors['gross_margin'] < 0 or factors['gross_margin'] > 1:
        return np.nan

    # Handle missing values
    for key in ['eps_growth', 'margin_improvement', 'momentum_12_1']:
        if pd.isna(factors[key]):
            factors[key] = 0

    # Cap extreme values
    factors['revenue_growth'] = np.clip(factors['revenue_growth'], -0.5, 2.0)
    factors['eps_growth'] = np.clip(factors['eps_growth'], -0.5, 3.0)
    factors['momentum_12_1'] = np.clip(factors['momentum_12_1'], -0.5, 2.0)

    # Small cap score
    factors['small_cap'] = -np.log10(factors['market_cap'] / 1e9)

    return factors  # Return dict for later z-scoring


def compute_momentum_score(prices_df, symbol, date, price_dict, vol_dict):
    """Compute Momentum Score factors for a single observation."""
    if symbol not in price_dict:
        return np.nan

    sym_prices = price_dict[symbol]

    # Get dates we need
    dates = sorted([d for d in sym_prices.keys() if d <= date])
    if len(dates) < 252:
        return np.nan

    latest_price = sym_prices.get(dates[-1], np.nan)
    price_12m = sym_prices.get(dates[-252] if len(dates) >= 252 else dates[0], np.nan)
    price_1m = sym_prices.get(dates[-21] if len(dates) >= 21 else dates[0], np.nan)
    price_3m = sym_prices.get(dates[-63] if len(dates) >= 63 else dates[0], np.nan)

    if any(pd.isna([latest_price, price_12m, price_1m, price_3m])):
        return np.nan
    if price_12m <= 0 or price_3m <= 0:
        return np.nan

    momentum_12_1 = (price_1m - price_12m) / price_12m
    momentum_3m = (latest_price - price_3m) / price_3m

    # 52-week high
    recent_prices = [sym_prices[d] for d in dates[-252:] if d in sym_prices]
    high_52w = max(recent_prices) if recent_prices else np.nan
    if high_52w and high_52w > 0:
        high_52w_proximity = latest_price / high_52w
    else:
        return np.nan

    # SMA50
    sma50_prices = [sym_prices[d] for d in dates[-50:] if d in sym_prices]
    sma50 = np.mean(sma50_prices) if sma50_prices else np.nan
    if sma50 and sma50 > 0:
        price_vs_sma50 = (latest_price - sma50) / sma50
    else:
        return np.nan

    return {
        'momentum_12_1': momentum_12_1,
        'momentum_3m': momentum_3m,
        'high_52w_proximity': high_52w_proximity,
        'price_vs_sma50': price_vs_sma50,
        'volume_surge': 0  # Skip for simplicity in backtest
    }


def run_quintile_backtest(scores, forward_returns, horizon_name):
    """Run quintile analysis on scores vs forward returns."""
    # Create DataFrame
    df = pd.DataFrame({
        'score': scores,
        'fwd_return': forward_returns
    }).dropna()

    if len(df) < 100:
        return None

    # Assign quintiles
    df['quintile'] = pd.qcut(df['score'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Compute mean return by quintile
    quintile_returns = df.groupby('quintile')['fwd_return'].mean()

    return {
        'Q1': quintile_returns.get(1, np.nan),
        'Q2': quintile_returns.get(2, np.nan),
        'Q3': quintile_returns.get(3, np.nan),
        'Q4': quintile_returns.get(4, np.nan),
        'Q5': quintile_returns.get(5, np.nan),
        'spread': quintile_returns.get(5, 0) - quintile_returns.get(1, 0),
        'count': len(df)
    }


def backtest_compass_score(fund, prices, price_dict, vol_dict, fwd_returns):
    """Backtest Compass Score with 6-month forward returns."""
    print("\n" + "=" * 60)
    print("BACKTESTING COMPASS SCORE (Quality)")
    print("=" * 60)

    # Universe stats
    universe_stats = {
        'roa': {'mean': 0.02, 'std': 0.15},
        'ocf_assets': {'mean': 0.05, 'std': 0.12},
        'fcf_assets': {'mean': 0.03, 'std': 0.15},
        'gp_assets': {'mean': 0.25, 'std': 0.20},
        'asset_growth': {'mean': 0.10, 'std': 0.30},
        'vol_60d': {'mean': 45, 'std': 25}
    }

    # Get quarterly rebalance dates
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='QE')

    all_scores = []
    all_returns = []

    for rebal_date in rebalance_dates:
        # Get fundamentals closest to rebalance date
        fund_at_date = fund[fund['date'] <= rebal_date].groupby('symbol').last().reset_index()

        for _, row in fund_at_date.iterrows():
            symbol = row['symbol']
            if symbol not in vol_dict:
                continue

            # Get volatility
            vol_dates = [d for d in vol_dict[symbol].keys() if d <= rebal_date]
            if not vol_dates:
                continue
            vol = vol_dict[symbol][max(vol_dates)]

            # Compute score
            score = compute_compass_score(row, vol, universe_stats)
            if pd.isna(score):
                continue

            # Get forward return (126 days = 6 months)
            if symbol not in fwd_returns[126]:
                continue
            ret_dates = [d for d in fwd_returns[126][symbol].keys() if d <= rebal_date]
            if not ret_dates:
                continue
            fwd_ret = fwd_returns[126][symbol][max(ret_dates)]

            if not pd.isna(fwd_ret):
                all_scores.append(score)
                all_returns.append(fwd_ret)

    print(f"  {len(all_scores):,} score-return pairs")

    result = run_quintile_backtest(all_scores, all_returns, '6-month')
    if result:
        print(f"\n  6-Month Forward Returns by Quintile:")
        print(f"    Q1 (Low Quality):  {result['Q1']:+.2%}")
        print(f"    Q2:                {result['Q2']:+.2%}")
        print(f"    Q3:                {result['Q3']:+.2%}")
        print(f"    Q4:                {result['Q4']:+.2%}")
        print(f"    Q5 (High Quality): {result['Q5']:+.2%}")
        print(f"    Spread (Q5-Q1):    {result['spread']:+.2%}")

    return result


def backtest_moonshot_score(fund, prices, price_dict, fwd_returns):
    """Backtest Moonshot Score with 1-year and 2-year forward returns."""
    print("\n" + "=" * 60)
    print("BACKTESTING MOONSHOT SCORE (Growth)")
    print("=" * 60)

    # Get quarterly rebalance dates
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='QE')

    # Collect all factor values first for z-scoring
    factor_data = []

    for rebal_date in rebalance_dates:
        fund_at_date = fund[fund['date'] <= rebal_date].groupby('symbol').last().reset_index()

        for _, row in fund_at_date.iterrows():
            symbol = row['symbol']
            if symbol not in price_dict:
                continue

            # Compute 12-1 momentum
            sym_prices = price_dict[symbol]
            dates = sorted([d for d in sym_prices.keys() if d <= rebal_date])
            if len(dates) < 252:
                continue

            price_12m = sym_prices.get(dates[-252], np.nan)
            price_1m = sym_prices.get(dates[-21] if len(dates) >= 21 else dates[0], np.nan)
            if pd.isna(price_12m) or price_12m <= 0:
                continue
            momentum_12_1 = (price_1m - price_12m) / price_12m

            factors = compute_moonshot_score(row, momentum_12_1)
            if isinstance(factors, dict):
                factors['symbol'] = symbol
                factors['date'] = rebal_date
                factor_data.append(factors)

    if not factor_data:
        print("  No valid data for Moonshot backtest")
        return None

    df = pd.DataFrame(factor_data)
    print(f"  {len(df):,} observations collected")

    # Compute z-scores
    weights = {
        'revenue_growth': 0.25,
        'eps_growth': 0.20,
        'gross_margin': 0.15,
        'margin_improvement': 0.10,
        'small_cap': 0.15,
        'momentum_12_1': 0.15
    }

    for factor in weights.keys():
        df[f'{factor}_z'] = (df[factor] - df[factor].mean()) / df[factor].std()

    df['raw_score'] = sum(df[f'{f}_z'] * w for f, w in weights.items())

    # Get forward returns
    results = {}
    for horizon, horizon_name in [(252, '1-year'), (504, '2-year')]:
        scores = []
        returns = []

        for _, row in df.iterrows():
            symbol = row['symbol']
            date = row['date']

            if symbol not in fwd_returns[horizon]:
                continue

            ret_dates = [d for d in fwd_returns[horizon][symbol].keys() if d <= date]
            if not ret_dates:
                continue

            fwd_ret = fwd_returns[horizon][symbol][max(ret_dates)]
            if not pd.isna(fwd_ret):
                scores.append(row['raw_score'])
                returns.append(fwd_ret)

        result = run_quintile_backtest(scores, returns, horizon_name)
        if result:
            print(f"\n  {horizon_name} Forward Returns by Quintile:")
            print(f"    Q1 (Low Growth):  {result['Q1']:+.2%}")
            print(f"    Q2:               {result['Q2']:+.2%}")
            print(f"    Q3:               {result['Q3']:+.2%}")
            print(f"    Q4:               {result['Q4']:+.2%}")
            print(f"    Q5 (High Growth): {result['Q5']:+.2%}")
            print(f"    Spread (Q5-Q1):   {result['spread']:+.2%}")
            results[horizon_name] = result

    return results


def backtest_momentum_score(prices, price_dict, fwd_returns):
    """Backtest Momentum Score with 1-month and 3-month forward returns."""
    print("\n" + "=" * 60)
    print("BACKTESTING MOMENTUM SCORE (Short-term)")
    print("=" * 60)

    # Monthly rebalance dates
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='ME')

    # Collect factor data
    factor_data = []

    for rebal_date in rebalance_dates:
        for symbol in price_dict.keys():
            factors = compute_momentum_score(None, symbol, rebal_date, price_dict, None)
            if isinstance(factors, dict):
                factors['symbol'] = symbol
                factors['date'] = rebal_date
                factor_data.append(factors)

    if not factor_data:
        print("  No valid data for Momentum backtest")
        return None

    df = pd.DataFrame(factor_data)

    # Apply filters
    df = df[
        (df['momentum_12_1'] > -0.8) &
        (df['momentum_12_1'] < 5.0) &
        (df['momentum_3m'] > -0.5) &
        (df['momentum_3m'] < 3.0)
    ].copy()

    print(f"  {len(df):,} observations after filters")

    # Compute z-scores
    weights = {
        'momentum_12_1': 0.35,
        'high_52w_proximity': 0.25,
        'momentum_3m': 0.25,
        'price_vs_sma50': 0.15
    }

    for factor in weights.keys():
        df[f'{factor}_z'] = (df[factor] - df[factor].mean()) / df[factor].std()

    df['raw_score'] = sum(df[f'{f}_z'] * w for f, w in weights.items())

    # Get forward returns
    results = {}
    for horizon, horizon_name in [(21, '1-month'), (63, '3-month')]:
        scores = []
        returns = []

        for _, row in df.iterrows():
            symbol = row['symbol']
            date = row['date']

            if symbol not in fwd_returns[horizon]:
                continue

            ret_dates = [d for d in fwd_returns[horizon][symbol].keys() if d <= date]
            if not ret_dates:
                continue

            fwd_ret = fwd_returns[horizon][symbol][max(ret_dates)]
            if not pd.isna(fwd_ret):
                scores.append(row['raw_score'])
                returns.append(fwd_ret)

        result = run_quintile_backtest(scores, returns, horizon_name)
        if result:
            print(f"\n  {horizon_name} Forward Returns by Quintile:")
            print(f"    Q1 (Low Momentum):  {result['Q1']:+.2%}")
            print(f"    Q2:                 {result['Q2']:+.2%}")
            print(f"    Q3:                 {result['Q3']:+.2%}")
            print(f"    Q4:                 {result['Q4']:+.2%}")
            print(f"    Q5 (High Momentum): {result['Q5']:+.2%}")
            print(f"    Spread (Q5-Q1):     {result['spread']:+.2%}")
            results[horizon_name] = result

    return results


def main():
    print("=" * 60)
    print("BACKTEST ALL SCORING SYSTEMS")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    # Load data
    prices, fund = load_all_data()

    # Pre-compute metrics
    fund = precompute_ttm_fundamentals(fund)
    price_dict, vol_dict = precompute_price_metrics(prices)
    fwd_returns = precompute_forward_returns(prices, [21, 63, 126, 252, 504])

    # Run backtests
    compass_result = backtest_compass_score(fund, prices, price_dict, vol_dict, fwd_returns)
    moonshot_result = backtest_moonshot_score(fund, prices, price_dict, fwd_returns)
    momentum_result = backtest_momentum_score(prices, price_dict, fwd_returns)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nCompass Score (Quality):")
    if compass_result:
        print(f"  6-month spread: {compass_result['spread']:+.2%}")
        print(f"  {'PREDICTIVE' if compass_result['spread'] > 0 else 'NOT PREDICTIVE'}")

    print("\nMoonshot Score (Growth):")
    if moonshot_result:
        for horizon, res in moonshot_result.items():
            print(f"  {horizon} spread: {res['spread']:+.2%}")

    print("\nMomentum Score (Short-term):")
    if momentum_result:
        for horizon, res in momentum_result.items():
            print(f"  {horizon} spread: {res['spread']:+.2%}")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

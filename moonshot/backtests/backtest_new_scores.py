#!/usr/bin/env python3
"""
Backtest Moonshot and Momentum Scores
======================================
Tests predictive power of:
1. Moonshot Score (Growth) - 1-year, 2-year forward returns
2. Momentum Score (Short-term) - 1-month, 3-month forward returns

Compass Score already validated - see RESEARCH_PAPER.md
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Backtest parameters
BACKTEST_START = '2015-01-01'
BACKTEST_END = '2023-12-31'


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
        SELECT i.symbol, i.date, i.revenue, i.gross_profit, i.eps_diluted as eps,
               m.market_cap
        FROM historical_income_statements i
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
    """Pre-compute TTM metrics for Moonshot score."""
    print("Computing TTM metrics...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date'])

    # TTM for flow metrics
    for col in ['revenue', 'gross_profit', 'eps']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Compute ratios
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # YoY growth
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4, fill_method=None)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4, fill_method=None)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Clean infinities
    for col in ['gross_margin', 'revenue_growth', 'eps_growth', 'margin_improvement']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    return fund


def precompute_price_lookups(prices):
    """Create fast price lookup dictionaries."""
    print("Creating price lookups...")
    sys.stdout.flush()

    price_dict = {}
    for symbol in prices['symbol'].unique():
        sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
        price_dict[symbol] = dict(zip(sym_prices['date'], sym_prices['close']))

    print(f"  {len(price_dict):,} symbols")
    return price_dict


def precompute_forward_returns(prices, horizons=[21, 63, 252, 504]):
    """Pre-compute forward returns for all horizons."""
    print(f"Computing forward returns for horizons: {horizons}...")
    sys.stdout.flush()

    returns_dict = {h: {} for h in horizons}

    for symbol in prices['symbol'].unique():
        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)

        for h in horizons:
            fwd_prices = sym_prices['close'].shift(-h)
            fwd_ret = (fwd_prices - sym_prices['close']) / sym_prices['close']
            returns_dict[h][symbol] = dict(zip(sym_prices['date'], fwd_ret))

    return returns_dict


def run_quintile_backtest(scores, forward_returns, horizon_name):
    """Run quintile analysis on scores vs forward returns."""
    df = pd.DataFrame({
        'score': scores,
        'fwd_return': forward_returns
    }).dropna()

    if len(df) < 100:
        return None

    try:
        df['quintile'] = pd.qcut(df['score'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except ValueError:
        return None

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


def backtest_moonshot_score(fund, price_dict, fwd_returns):
    """Backtest Moonshot Score with 1-year and 2-year forward returns."""
    print("\n" + "=" * 60)
    print("BACKTESTING MOONSHOT SCORE (Growth Predictor)")
    print("=" * 60)
    sys.stdout.flush()

    # Quarterly rebalance dates
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='QE')

    # Collect all factor values
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

            # Get factor values
            revenue_growth = row.get('revenue_growth', np.nan)
            eps_growth = row.get('eps_growth', np.nan)
            gross_margin = row.get('gross_margin', np.nan)
            margin_improvement = row.get('margin_improvement', np.nan)
            market_cap = row.get('market_cap', np.nan)

            # Skip if missing critical values
            if pd.isna(revenue_growth) or pd.isna(gross_margin):
                continue
            if pd.isna(market_cap) or market_cap <= 0:
                continue

            # Quality filters
            if revenue_growth > 5.0 or revenue_growth < -0.8:
                continue
            if gross_margin < 0 or gross_margin > 1:
                continue

            # Handle missing
            eps_growth = 0 if pd.isna(eps_growth) else eps_growth
            margin_improvement = 0 if pd.isna(margin_improvement) else margin_improvement

            # Cap extremes
            revenue_growth = np.clip(revenue_growth, -0.5, 2.0)
            eps_growth = np.clip(eps_growth, -0.5, 3.0)
            momentum_12_1 = np.clip(momentum_12_1, -0.5, 2.0)

            # Small cap score
            small_cap = -np.log10(market_cap / 1e9)

            factor_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth,
                'gross_margin': gross_margin,
                'margin_improvement': margin_improvement,
                'small_cap': small_cap,
                'momentum_12_1': momentum_12_1
            })

    if not factor_data:
        print("  No valid data for Moonshot backtest")
        return None

    df = pd.DataFrame(factor_data)
    print(f"  {len(df):,} observations collected")
    sys.stdout.flush()

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

    # Test forward returns
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
            print(f"    Observations:     {result['count']:,}")
            results[horizon_name] = result

    return results


def backtest_momentum_score(price_dict, fwd_returns):
    """Backtest Momentum Score with 1-month and 3-month forward returns."""
    print("\n" + "=" * 60)
    print("BACKTESTING MOMENTUM SCORE (Short-term Predictor)")
    print("=" * 60)
    sys.stdout.flush()

    # Monthly rebalance dates
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='ME')

    # Collect factor data
    factor_data = []

    for rebal_date in rebalance_dates:
        for symbol, sym_prices in price_dict.items():
            dates = sorted([d for d in sym_prices.keys() if d <= rebal_date])
            if len(dates) < 252:
                continue

            latest_price = sym_prices.get(dates[-1], np.nan)
            price_12m = sym_prices.get(dates[-252], np.nan)
            price_1m = sym_prices.get(dates[-21] if len(dates) >= 21 else dates[0], np.nan)
            price_3m = sym_prices.get(dates[-63] if len(dates) >= 63 else dates[0], np.nan)

            if any(pd.isna([latest_price, price_12m, price_1m, price_3m])):
                continue
            if price_12m <= 0 or price_3m <= 0:
                continue

            momentum_12_1 = (price_1m - price_12m) / price_12m
            momentum_3m = (latest_price - price_3m) / price_3m

            # 52-week high
            recent_prices = [sym_prices[d] for d in dates[-252:] if d in sym_prices]
            high_52w = max(recent_prices) if recent_prices else np.nan
            if not high_52w or high_52w <= 0:
                continue
            high_52w_proximity = latest_price / high_52w

            # SMA50
            sma50_prices = [sym_prices[d] for d in dates[-50:] if d in sym_prices]
            sma50 = np.mean(sma50_prices) if sma50_prices else np.nan
            if not sma50 or sma50 <= 0:
                continue
            price_vs_sma50 = (latest_price - sma50) / sma50

            # Quality filters
            if momentum_12_1 < -0.8 or momentum_12_1 > 5.0:
                continue
            if momentum_3m < -0.5 or momentum_3m > 3.0:
                continue

            factor_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'momentum_12_1': momentum_12_1,
                'momentum_3m': momentum_3m,
                'high_52w_proximity': high_52w_proximity,
                'price_vs_sma50': np.clip(price_vs_sma50, -0.5, 1.0)
            })

    if not factor_data:
        print("  No valid data for Momentum backtest")
        return None

    df = pd.DataFrame(factor_data)
    print(f"  {len(df):,} observations collected")
    sys.stdout.flush()

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

    # Test forward returns
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
            print(f"    Observations:       {result['count']:,}")
            results[horizon_name] = result

    return results


def main():
    print("=" * 60)
    print("BACKTEST MOONSHOT & MOMENTUM SCORES")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    # Load data
    prices, fund = load_all_data()

    # Pre-compute
    fund = precompute_ttm_fundamentals(fund)
    price_dict = precompute_price_lookups(prices)
    fwd_returns = precompute_forward_returns(prices, [21, 63, 252, 504])

    # Run backtests
    moonshot_result = backtest_moonshot_score(fund, price_dict, fwd_returns)
    momentum_result = backtest_momentum_score(price_dict, fwd_returns)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nMoonshot Score (Growth Predictor):")
    if moonshot_result:
        for horizon, res in moonshot_result.items():
            status = "PREDICTIVE" if res['spread'] > 0.02 else "WEAK" if res['spread'] > 0 else "NOT PREDICTIVE"
            print(f"  {horizon}: spread = {res['spread']:+.2%} -> {status}")

    print("\nMomentum Score (Short-term Predictor):")
    if momentum_result:
        for horizon, res in momentum_result.items():
            status = "PREDICTIVE" if res['spread'] > 0.01 else "WEAK" if res['spread'] > 0 else "NOT PREDICTIVE"
            print(f"  {horizon}: spread = {res['spread']:+.2%} -> {status}")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

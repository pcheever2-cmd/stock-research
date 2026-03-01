#!/usr/bin/env python3
"""
Backtest Short-Term Factors
============================
Testing various approaches to predict short-term gains:
1. Mean Reversion (Oversold) - buy beaten down stocks
2. Earnings Momentum - recent EPS beats
3. Volume Breakout - unusual volume with price move
4. Analyst Upgrades proxy - revenue estimate changes
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


def load_data():
    """Load price data."""
    print("Loading data...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    conn.close()

    print(f"  {len(prices):,} price records")
    sys.stdout.flush()

    return prices


def run_quintile_backtest(df, score_col, return_col, name):
    """Run quintile analysis."""
    test_df = df[[score_col, return_col]].dropna()

    if len(test_df) < 100:
        return None

    try:
        test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    except ValueError:
        return None

    quintile_returns = test_df.groupby('quintile', observed=False)[return_col].mean()

    spread = quintile_returns.get(5, 0) - quintile_returns.get(1, 0)

    print(f"\n  {name}:")
    print(f"    Q1 (Low):  {quintile_returns.get(1, np.nan):+.2%}")
    print(f"    Q5 (High): {quintile_returns.get(5, np.nan):+.2%}")
    print(f"    Spread:    {spread:+.2%}")
    print(f"    N:         {len(test_df):,}")

    return {
        'Q1': quintile_returns.get(1, np.nan),
        'Q5': quintile_returns.get(5, np.nan),
        'spread': spread,
        'count': len(test_df)
    }


def backtest_mean_reversion(prices):
    """Test mean reversion (oversold bounce)."""
    print("\n" + "=" * 60)
    print("TESTING MEAN REVERSION (OVERSOLD BOUNCE)")
    print("=" * 60)

    # Monthly rebalance
    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='ME')

    # Pre-compute factors by symbol
    price_dict = {}
    for symbol in prices['symbol'].unique():
        sym = prices[prices['symbol'] == symbol].sort_values('date')
        if len(sym) < 252:
            continue
        price_dict[symbol] = sym.set_index('date')['close'].to_dict()

    results_data = []

    for rebal_date in rebalance_dates:
        for symbol, sym_prices in price_dict.items():
            dates = sorted([d for d in sym_prices.keys() if d <= rebal_date])
            if len(dates) < 252:
                continue

            # Current price
            current = sym_prices.get(dates[-1], np.nan)
            # 1-month ago
            price_1m = sym_prices.get(dates[-21] if len(dates) >= 21 else dates[0], np.nan)
            # 3-month ago
            price_3m = sym_prices.get(dates[-63] if len(dates) >= 63 else dates[0], np.nan)
            # 52-week high
            high_52w = max([sym_prices[d] for d in dates[-252:]])
            # 52-week low
            low_52w = min([sym_prices[d] for d in dates[-252:]])

            if any(pd.isna([current, price_1m, price_3m, high_52w, low_52w])):
                continue
            if price_1m <= 0 or price_3m <= 0 or high_52w <= 0:
                continue

            # Factors
            ret_1m = (current - price_1m) / price_1m
            ret_3m = (current - price_3m) / price_3m
            dist_from_high = (current - high_52w) / high_52w  # Negative = oversold
            dist_from_low = (current - low_52w) / low_52w  # Positive = room to fall

            # Forward returns (need to look up)
            future_dates = sorted([d for d in sym_prices.keys() if d > rebal_date])
            if len(future_dates) < 63:
                continue

            fwd_1m = (sym_prices[future_dates[21]] - current) / current if len(future_dates) > 21 else np.nan
            fwd_3m = (sym_prices[future_dates[63]] - current) / current if len(future_dates) > 63 else np.nan

            results_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'ret_1m': ret_1m,
                'ret_3m': ret_3m,
                'dist_from_high': dist_from_high,
                'dist_from_low': dist_from_low,
                'oversold_score': -ret_1m - 0.5 * ret_3m,  # More negative recent returns = higher oversold
                'fwd_1m': fwd_1m,
                'fwd_3m': fwd_3m
            })

    df = pd.DataFrame(results_data)
    print(f"  {len(df):,} observations collected")

    # Test oversold score (higher = more oversold = expecting bounce)
    print("\n  Testing: OVERSOLD SCORE (negative recent returns)")
    run_quintile_backtest(df, 'oversold_score', 'fwd_1m', '1-Month Forward')
    run_quintile_backtest(df, 'oversold_score', 'fwd_3m', '3-Month Forward')

    # Test distance from 52-week high (more negative = more oversold)
    print("\n  Testing: DISTANCE FROM 52W HIGH")
    run_quintile_backtest(df, 'dist_from_high', 'fwd_1m', '1-Month Forward')
    run_quintile_backtest(df, 'dist_from_high', 'fwd_3m', '3-Month Forward')

    return df


def backtest_breakout_momentum(prices):
    """Test breakout momentum (stocks breaking to new highs)."""
    print("\n" + "=" * 60)
    print("TESTING BREAKOUT MOMENTUM (NEW HIGHS)")
    print("=" * 60)

    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='ME')

    price_dict = {}
    for symbol in prices['symbol'].unique():
        sym = prices[prices['symbol'] == symbol].sort_values('date')
        if len(sym) < 252:
            continue
        price_dict[symbol] = sym.set_index('date')['close'].to_dict()

    results_data = []

    for rebal_date in rebalance_dates:
        for symbol, sym_prices in price_dict.items():
            dates = sorted([d for d in sym_prices.keys() if d <= rebal_date])
            if len(dates) < 252:
                continue

            current = sym_prices.get(dates[-1], np.nan)
            high_52w = max([sym_prices[d] for d in dates[-252:]])

            if pd.isna(current) or high_52w <= 0:
                continue

            # Proximity to 52-week high (1.0 = at high)
            high_proximity = current / high_52w

            # Days since last 52-week high
            recent_highs = [d for d in dates[-252:] if sym_prices[d] >= high_52w * 0.98]
            days_since_high = (dates[-1] - recent_highs[-1]).days if recent_highs else 252

            # Forward returns
            future_dates = sorted([d for d in sym_prices.keys() if d > rebal_date])
            if len(future_dates) < 63:
                continue

            fwd_1m = (sym_prices[future_dates[21]] - current) / current if len(future_dates) > 21 else np.nan
            fwd_3m = (sym_prices[future_dates[63]] - current) / current if len(future_dates) > 63 else np.nan

            results_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'high_proximity': high_proximity,
                'days_since_high': days_since_high,
                'breakout_score': high_proximity - days_since_high / 252,  # Near high + recent = breakout
                'fwd_1m': fwd_1m,
                'fwd_3m': fwd_3m
            })

    df = pd.DataFrame(results_data)
    print(f"  {len(df):,} observations collected")

    # Test breakout score
    print("\n  Testing: BREAKOUT SCORE (near highs + recent)")
    run_quintile_backtest(df, 'breakout_score', 'fwd_1m', '1-Month Forward')
    run_quintile_backtest(df, 'breakout_score', 'fwd_3m', '3-Month Forward')

    return df


def backtest_classic_momentum(prices):
    """Test classic 12-1 momentum (well-documented anomaly)."""
    print("\n" + "=" * 60)
    print("TESTING CLASSIC 12-1 MOMENTUM")
    print("=" * 60)

    rebalance_dates = pd.date_range(BACKTEST_START, BACKTEST_END, freq='ME')

    price_dict = {}
    for symbol in prices['symbol'].unique():
        sym = prices[prices['symbol'] == symbol].sort_values('date')
        if len(sym) < 300:  # Need more history for 12-1
            continue
        price_dict[symbol] = sym.set_index('date')['close'].to_dict()

    results_data = []

    for rebal_date in rebalance_dates:
        for symbol, sym_prices in price_dict.items():
            dates = sorted([d for d in sym_prices.keys() if d <= rebal_date])
            if len(dates) < 280:
                continue

            # 12-1 momentum: return from 12 months ago to 1 month ago
            price_12m = sym_prices.get(dates[-252], np.nan)
            price_1m = sym_prices.get(dates[-21], np.nan)
            current = sym_prices.get(dates[-1], np.nan)

            if pd.isna(price_12m) or pd.isna(price_1m) or price_12m <= 0:
                continue

            mom_12_1 = (price_1m - price_12m) / price_12m

            # Filter extreme values
            if mom_12_1 < -0.8 or mom_12_1 > 5.0:
                continue

            # Forward returns at different horizons
            future_dates = sorted([d for d in sym_prices.keys() if d > rebal_date])
            if len(future_dates) < 126:
                continue

            fwd_1m = (sym_prices[future_dates[21]] - current) / current if len(future_dates) > 21 else np.nan
            fwd_3m = (sym_prices[future_dates[63]] - current) / current if len(future_dates) > 63 else np.nan
            fwd_6m = (sym_prices[future_dates[126]] - current) / current if len(future_dates) > 126 else np.nan

            results_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'mom_12_1': mom_12_1,
                'fwd_1m': fwd_1m,
                'fwd_3m': fwd_3m,
                'fwd_6m': fwd_6m
            })

    df = pd.DataFrame(results_data)
    print(f"  {len(df):,} observations collected")

    # Test at different forward horizons
    print("\n  Testing: 12-1 MOMENTUM at different forward horizons")
    run_quintile_backtest(df, 'mom_12_1', 'fwd_1m', '1-Month Forward')
    run_quintile_backtest(df, 'mom_12_1', 'fwd_3m', '3-Month Forward')
    run_quintile_backtest(df, 'mom_12_1', 'fwd_6m', '6-Month Forward')

    return df


def main():
    print("=" * 60)
    print("BACKTEST SHORT-TERM FACTORS")
    print(f"Period: {BACKTEST_START} to {BACKTEST_END}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    prices = load_data()

    # Test different approaches
    backtest_mean_reversion(prices)
    backtest_breakout_momentum(prices)
    backtest_classic_momentum(prices)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Key Findings:
- Oversold Score: Higher scores = more beaten down stocks
  If positive spread: Mean reversion works (buy the dip)

- Breakout Score: Higher scores = stocks at/near 52w highs
  If positive spread: Breakout momentum works

- Classic 12-1 Momentum: Well-documented factor
  Expected positive spread at 6+ month horizons
""")
    print("=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

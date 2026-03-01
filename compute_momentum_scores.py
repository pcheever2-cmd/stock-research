#!/usr/bin/env python3
"""
Compute Momentum Scores for Dashboard
======================================
Identifies stocks with high likelihood of short-term gains.

Factors (based on momentum research):
1. 12-1 Month Price Momentum - 30% weight (classic momentum factor)
2. 52-Week High Proximity - 20% weight (stocks near highs tend to break out)
3. 3-Month Momentum - 20% weight (recent strength)
4. Volume Surge - 15% weight (institutional buying signal)
5. Price vs SMA50 - 15% weight (trend strength)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
NASDAQ_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Factor weights
WEIGHTS = {
    'momentum_12_1': 0.30,
    'high_52w_proximity': 0.20,
    'momentum_3m': 0.20,
    'volume_surge': 0.15,
    'price_vs_sma50': 0.15
}


def assign_grade(percentile):
    """Assign letter grade based on percentile."""
    if percentile >= 85:
        return 'A'
    elif percentile >= 60:
        return 'B'
    elif percentile >= 40:
        return 'C'
    elif percentile >= 20:
        return 'D'
    else:
        return 'F'


def load_data():
    """Load price data from backtest.db."""
    print("Loading data from backtest.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Get prices for momentum calculations (last 14 months)
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 0.5
          AND date >= date('now', '-14 months')
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    conn.close()

    print(f"  {len(prices):,} price records")
    sys.stdout.flush()

    return prices


def compute_momentum_factors(prices_df, symbol):
    """Compute all momentum factors for a single symbol."""
    sym_prices = prices_df[prices_df['symbol'] == symbol].sort_values('date')

    if len(sym_prices) < 252:  # Need ~1 year of data
        return None

    latest_price = sym_prices.iloc[-1]['close']

    # 12-1 Month Momentum (skip most recent month to avoid reversal)
    price_12m = sym_prices.iloc[-252]['close']
    price_1m = sym_prices.iloc[-21]['close']
    if price_12m > 0:
        momentum_12_1 = (price_1m - price_12m) / price_12m
    else:
        return None

    # 3-Month Momentum
    price_3m = sym_prices.iloc[-63]['close']
    if price_3m > 0:
        momentum_3m = (latest_price - price_3m) / price_3m
    else:
        return None

    # 52-Week High Proximity (how close to 52-week high)
    high_52w = sym_prices.tail(252)['close'].max()
    if high_52w > 0:
        high_52w_proximity = latest_price / high_52w  # 1.0 = at high, 0.5 = 50% below
    else:
        return None

    # Volume Surge (recent volume vs 50-day average)
    if len(sym_prices) >= 60:
        recent_volume = sym_prices.tail(5)['volume'].mean()
        avg_volume = sym_prices.tail(60).head(55)['volume'].mean()  # 50-day avg, excluding last 5
        if avg_volume > 0:
            volume_surge = recent_volume / avg_volume - 1  # 0 = normal, 1 = 2x volume
        else:
            volume_surge = 0
    else:
        volume_surge = 0

    # Price vs SMA50 (trend strength)
    sma50 = sym_prices.tail(50)['close'].mean()
    if sma50 > 0:
        price_vs_sma50 = (latest_price - sma50) / sma50
    else:
        return None

    return {
        'momentum_12_1': momentum_12_1,
        'momentum_3m': momentum_3m,
        'high_52w_proximity': high_52w_proximity,
        'volume_surge': volume_surge,
        'price_vs_sma50': price_vs_sma50
    }


def compute_all_scores():
    """Compute Momentum Scores for all stocks."""
    prices = load_data()

    print("\nComputing momentum factors for all stocks...")
    sys.stdout.flush()

    all_symbols = prices['symbol'].unique()
    print(f"  {len(all_symbols):,} stocks with price data")
    sys.stdout.flush()

    # Compute factor values
    results = []
    for i, symbol in enumerate(all_symbols):
        factors = compute_momentum_factors(prices, symbol)
        if factors is not None:
            factors['symbol'] = symbol
            results.append(factors)

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1:,}/{len(all_symbols):,} processed...")
            sys.stdout.flush()

    df = pd.DataFrame(results)
    print(f"  {len(df):,} stocks with valid factor data")
    sys.stdout.flush()

    # Apply quality filters
    df = df[
        (df['momentum_12_1'] > -0.8) &  # Not in complete freefall
        (df['momentum_12_1'] < 5.0) &   # Not suspicious spikes
        (df['momentum_3m'] > -0.5) &
        (df['momentum_3m'] < 3.0)
    ].copy()
    print(f"  {len(df):,} stocks after quality filters")

    # Cap extreme values
    df['volume_surge'] = df['volume_surge'].clip(-0.5, 5.0)
    df['price_vs_sma50'] = df['price_vs_sma50'].clip(-0.5, 1.0)

    # Compute z-scores for each factor
    print("  Computing z-scores and composite score...")
    for factor in WEIGHTS.keys():
        mean = df[factor].mean()
        std = df[factor].std()
        df[f'{factor}_z'] = (df[factor] - mean) / std

    # Compute composite score
    df['raw_score'] = sum(
        df[f'{factor}_z'] * weight
        for factor, weight in WEIGHTS.items()
    )

    # Convert to percentile-based score (0-100)
    percentile = df['raw_score'].rank(pct=True)
    df['momentum_score'] = (percentile * 100).round(0).astype(int)

    # Spread out top tier based on raw score
    df.loc[df['raw_score'] >= 0.8, 'momentum_score'] = 96
    df.loc[df['raw_score'] >= 1.0, 'momentum_score'] = 97
    df.loc[df['raw_score'] >= 1.3, 'momentum_score'] = 98
    df.loc[df['raw_score'] >= 1.6, 'momentum_score'] = 99
    df.loc[df['raw_score'] >= 2.0, 'momentum_score'] = 100

    # Assign grades
    df['momentum_grade'] = df['momentum_score'].apply(assign_grade)

    return df


def update_nasdaq_db(scores_df):
    """Update nasdaq_stocks.db with Momentum Scores."""
    print("\nUpdating nasdaq_stocks.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(NASDAQ_DB)
    cursor = conn.cursor()

    # Add columns if they don't exist
    for col, dtype in [('momentum_score', 'INTEGER'),
                       ('momentum_grade', 'TEXT'),
                       ('momentum_updated_at', 'TEXT')]:
        try:
            cursor.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col} {dtype}")
            print(f"  Added {col} column")
        except sqlite3.OperationalError:
            pass

    # Clear existing scores
    cursor.execute("""
        UPDATE stock_consensus
        SET momentum_score = NULL, momentum_grade = NULL, momentum_updated_at = NULL
    """)
    cleared = cursor.rowcount
    print(f"  Cleared {cleared:,} existing scores")

    # Update scores
    updated = 0
    timestamp = datetime.now().isoformat()

    for _, row in scores_df.iterrows():
        cursor.execute("""
            UPDATE stock_consensus
            SET momentum_score = ?, momentum_grade = ?, momentum_updated_at = ?
            WHERE symbol = ?
        """, (row['momentum_score'], row['momentum_grade'], timestamp, row['symbol']))
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"  Updated {updated:,} stocks in stock_consensus")
    sys.stdout.flush()


def print_examples(scores_df):
    """Print example stocks for verification."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Example Stocks")
    print("=" * 60)

    famous = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'PLTR', 'COIN', 'SMCI', 'AMD']

    print(f"\n{'Symbol':<8} {'Score':<8} {'Grade':<6} {'Mom12-1':<10} {'Mom3M':<10} {'52wHigh':<10}")
    print("-" * 70)

    for symbol in famous:
        row = scores_df[scores_df['symbol'] == symbol]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{symbol:<8} {r['momentum_score']:<8} {r['momentum_grade']:<6} "
                  f"{r['momentum_12_1']:+.1%}    {r['momentum_3m']:+.1%}    {r['high_52w_proximity']:.1%}")
        else:
            print(f"{symbol:<8} N/A")

    # Show top 10 momentum stocks
    print("\n" + "=" * 60)
    print("TOP 10 MOMENTUM STOCKS")
    print("=" * 60)
    top10 = scores_df.nlargest(10, 'raw_score')
    for _, r in top10.iterrows():
        print(f"{r['symbol']:<8} {r['momentum_score']:<8} Mom12-1: {r['momentum_12_1']:+.1%}")

    sys.stdout.flush()


def main():
    print("=" * 60)
    print("MOMENTUM SCORE COMPUTATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 60)
    sys.stdout.flush()

    # Compute scores
    scores_df = compute_all_scores()

    # Update database
    update_nasdaq_db(scores_df)

    # Print examples for verification
    print_examples(scores_df)

    # Summary stats
    print("\n" + "=" * 60)
    print("GRADE DISTRIBUTION")
    print("=" * 60)
    grade_counts = scores_df['momentum_grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        pct = count / len(scores_df) * 100
        print(f"  {grade}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

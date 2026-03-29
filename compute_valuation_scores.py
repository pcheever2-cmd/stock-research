#!/usr/bin/env python3
"""
Compute Valuation Scores for Dashboard
======================================
Computes technical valuation scores based on Price vs SMA200.
This score supports/contradicts analyst price targets in the valuation section.

Research validated (12-month horizon, OOS 2020+):
- Price vs SMA200: +37pp Q1-Q5 spread
- Combined with quality: 86% hit rate

Score interpretation:
- High score (70+): Stock is technically undervalued (below SMA200)
- Mid score (40-70): Fair value range
- Low score (<40): Stock is technically overvalued (above SMA200)
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

# Universe statistics (from research)
VALUATION_STATS = {
    'price_vs_sma200': {'mean': 0, 'std': 20},  # % deviation from SMA200
    'price_vs_sma50': {'mean': 0, 'std': 10},   # % deviation from SMA50
    'position_52w': {'mean': 50, 'std': 25},    # 0-100 position in 52w range
}


def load_price_data():
    """Load price data for valuation calculations."""
    print("Loading price data from backtest.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Need ~300 days for SMA200 + some buffer
    prices = pd.read_sql_query("""
        SELECT symbol, date, high, low, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
          AND date >= date('now', '-400 days')
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    conn.close()

    print(f"  {len(prices):,} price records")
    sys.stdout.flush()

    return prices


def compute_valuation_score(symbol, prices_df):
    """Compute valuation score for a single symbol.

    Returns score 0-100 where:
    - Higher = more undervalued (below moving averages)
    - Lower = more overvalued (above moving averages)
    """
    sym_prices = prices_df[prices_df['symbol'] == symbol].sort_values('date')

    if len(sym_prices) < 200:  # Need 200 days for SMA200
        return None, None

    current_price = sym_prices['close'].iloc[-1]

    # SMA200
    sma200 = sym_prices['close'].tail(200).mean()
    price_vs_sma200 = ((current_price - sma200) / sma200) * 100

    # SMA50
    sma50 = sym_prices['close'].tail(50).mean()
    price_vs_sma50 = ((current_price - sma50) / sma50) * 100

    # 52-week position (0 = at 52w low, 100 = at 52w high)
    high_52w = sym_prices['high'].tail(252).max()
    low_52w = sym_prices['low'].tail(252).min()
    if high_52w > low_52w:
        position_52w = ((current_price - low_52w) / (high_52w - low_52w)) * 100
    else:
        position_52w = 50

    factors = {
        'price_vs_sma200': price_vs_sma200,
        'price_vs_sma50': price_vs_sma50,
        'position_52w': position_52w,
        'sma200': sma200,
        'sma50': sma50,
        'high_52w': high_52w,
        'low_52w': low_52w,
        'current_price': current_price
    }

    # Compute z-scores (capped at ±3)
    # Invert so that BELOW average = positive z-score = higher valuation score
    z_sma200 = np.clip(
        -(price_vs_sma200 - VALUATION_STATS['price_vs_sma200']['mean']) / VALUATION_STATS['price_vs_sma200']['std'],
        -3, 3
    )
    z_sma50 = np.clip(
        -(price_vs_sma50 - VALUATION_STATS['price_vs_sma50']['mean']) / VALUATION_STATS['price_vs_sma50']['std'],
        -3, 3
    )
    z_52w = np.clip(
        -(position_52w - VALUATION_STATS['position_52w']['mean']) / VALUATION_STATS['position_52w']['std'],
        -3, 3
    )

    # Weighted combination (SMA200 is the strongest signal from research)
    raw_score = (
        z_sma200 * 0.50 +  # Primary signal
        z_sma50 * 0.25 +   # Short-term confirmation
        z_52w * 0.25       # Range context
    )

    return raw_score, factors


def compute_all_valuation_scores():
    """Compute valuation scores for all stocks."""
    prices = load_price_data()

    print("\nComputing valuation scores...")
    sys.stdout.flush()

    symbols = prices['symbol'].unique()
    print(f"  {len(symbols):,} symbols to process")

    results = []
    for i, symbol in enumerate(symbols):
        raw_score, factors = compute_valuation_score(symbol, prices)
        if raw_score is not None:
            results.append({
                'symbol': symbol,
                'valuation_raw_score': raw_score,
                'price_vs_sma200': factors['price_vs_sma200'],
                'price_vs_sma50': factors['price_vs_sma50'],
                'position_52w': factors['position_52w'],
                'sma200': factors['sma200'],
                'sma50': factors['sma50'],
                'high_52w': factors['high_52w'],
                'low_52w': factors['low_52w'],
                'current_price': factors['current_price']
            })

        if (i + 1) % 2000 == 0:
            print(f"    {i + 1:,}/{len(symbols):,} processed...")
            sys.stdout.flush()

    df = pd.DataFrame(results)
    print(f"  {len(df):,} stocks with valid valuation scores")

    # Convert to 0-100 scale using percentile ranking
    df['valuation_score'] = (df['valuation_raw_score'].rank(pct=True) * 100).round(0).astype(int)

    # Assign valuation rating
    def get_rating(score):
        if score >= 70:
            return 'Undervalued'
        elif score >= 40:
            return 'Fair Value'
        else:
            return 'Overvalued'

    df['valuation_rating'] = df['valuation_score'].apply(get_rating)

    return df


def update_nasdaq_db(scores_df):
    """Update nasdaq_stocks.db with valuation scores."""
    print("\nUpdating nasdaq_stocks.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(NASDAQ_DB)
    cursor = conn.cursor()

    # Add columns if they don't exist
    new_columns = [
        ('valuation_score', 'INTEGER'),
        ('valuation_rating', 'TEXT'),
        ('valuation_updated_at', 'TEXT'),
        ('price_vs_sma200', 'REAL'),
        ('price_vs_sma50', 'REAL'),
        ('position_52w', 'REAL'),
        ('sma200', 'REAL'),
        ('sma50', 'REAL'),
        ('high_52w', 'REAL'),
        ('low_52w', 'REAL')
    ]

    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col_name} {col_type}")
            print(f"  Added {col_name} column")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Clear existing valuation scores
    cursor.execute("""
        UPDATE stock_consensus
        SET valuation_score = NULL, valuation_rating = NULL, valuation_updated_at = NULL
    """)

    # Update scores
    timestamp = datetime.now().isoformat()
    updated = 0

    for _, row in scores_df.iterrows():
        cursor.execute("""
            UPDATE stock_consensus
            SET valuation_score = ?,
                valuation_rating = ?,
                valuation_updated_at = ?,
                price_vs_sma200 = ?,
                price_vs_sma50 = ?,
                position_52w = ?,
                sma200 = ?,
                sma50 = ?,
                high_52w = ?,
                low_52w = ?
            WHERE symbol = ?
        """, (
            row['valuation_score'],
            row['valuation_rating'],
            timestamp,
            row['price_vs_sma200'],
            row['price_vs_sma50'],
            row['position_52w'],
            row['sma200'],
            row['sma50'],
            row['high_52w'],
            row['low_52w'],
            row['symbol']
        ))
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"  Updated {updated:,} stocks")
    sys.stdout.flush()


def print_examples(scores_df):
    """Print example stocks."""
    print("\n" + "=" * 70)
    print("EXAMPLE STOCKS")
    print("=" * 70)

    famous = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'JPM', 'SMCI', 'PLTR', 'COIN']

    print(f"\n{'Symbol':<8} {'Score':<8} {'Rating':<12} {'vs SMA200':<12} {'vs SMA50':<10} {'52W Pos'}")
    print("-" * 70)

    for symbol in famous:
        row = scores_df[scores_df['symbol'] == symbol]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{symbol:<8} {r['valuation_score']:<8} {r['valuation_rating']:<12} "
                  f"{r['price_vs_sma200']:>+8.1f}%   {r['price_vs_sma50']:>+6.1f}%   {r['position_52w']:>5.0f}%")
        else:
            print(f"{symbol:<8} N/A")

    # Show most undervalued and overvalued
    print("\n" + "-" * 70)
    print("TOP 10 UNDERVALUED (highest scores):")
    top = scores_df.nlargest(10, 'valuation_score')
    for _, r in top.iterrows():
        print(f"  {r['symbol']:<8} Score: {r['valuation_score']}  vs SMA200: {r['price_vs_sma200']:+.1f}%")

    print("\nTOP 10 OVERVALUED (lowest scores):")
    bottom = scores_df.nsmallest(10, 'valuation_score')
    for _, r in bottom.iterrows():
        print(f"  {r['symbol']:<8} Score: {r['valuation_score']}  vs SMA200: {r['price_vs_sma200']:+.1f}%")


def main():
    print("=" * 70)
    print("VALUATION SCORE COMPUTATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)
    sys.stdout.flush()

    # Compute scores
    scores_df = compute_all_valuation_scores()

    # Update database
    update_nasdaq_db(scores_df)

    # Print examples
    print_examples(scores_df)

    # Distribution
    print("\n" + "=" * 70)
    print("RATING DISTRIBUTION")
    print("=" * 70)
    dist = scores_df['valuation_rating'].value_counts()
    for rating, count in dist.items():
        pct = count / len(scores_df) * 100
        print(f"  {rating}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)


if __name__ == '__main__':
    main()

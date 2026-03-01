#!/usr/bin/env python3
"""
Compute Compass Scores for Dashboard
====================================
Runs Mon/Wed/Fri via GitHub Actions.
Computes the research-validated Compass Score for all stocks
and stores in nasdaq_stocks.db for dashboard display.
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

# Universe statistics (from research paper)
UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
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
    """Load price and fundamental data from backtest.db."""
    print("Loading data from backtest.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Get recent prices for volatility calculation (last 90 days should be enough)
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
          AND date >= date('now', '-90 days')
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Get fundamentals (last 2 years for asset growth calculation)
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        WHERE i.date >= date('now', '-2 years')
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    conn.close()

    # Sort by symbol and date for rolling calculations
    fund = fund.sort_values(['symbol', 'date'])

    # Compute TTM (Trailing Twelve Months) for flow metrics
    # Sum of last 4 quarters for income statement and cash flow items
    print("  Computing TTM metrics...")
    sys.stdout.flush()

    for col in ['net_income', 'gross_profit', 'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Compute fundamental ratios using TTM for flow metrics, latest for balance sheet
    # Note: total_assets is point-in-time (latest quarter), not summed
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']

    # Asset growth: compare latest assets to assets 4 quarters ago
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean up infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")
    sys.stdout.flush()

    return prices, fund


def compute_raw_score(symbol, prices_df, fund_df):
    """Compute raw Compass Score for a single symbol."""
    # Get latest fundamentals
    sym_fund = fund_df[fund_df['symbol'] == symbol]
    if len(sym_fund) == 0:
        return None, None

    latest_fund = sym_fund.sort_values('date').iloc[-1]

    # Get volatility (60-day)
    sym_prices = prices_df[prices_df['symbol'] == symbol].sort_values('date').tail(60)
    if len(sym_prices) < 30:  # Need at least 30 days
        return None, None

    rets = sym_prices['close'].pct_change().dropna()
    vol = rets.std() * np.sqrt(252) * 100

    # Get factor values
    factors = {
        'roa': latest_fund['roa'],
        'ocf_assets': latest_fund['ocf_assets'],
        'fcf_assets': latest_fund['fcf_assets'],
        'gp_assets': latest_fund['gp_assets'],
        'asset_growth': latest_fund['asset_growth'],
        'vol_60d': vol
    }

    # Check for missing values
    for f, val in factors.items():
        if pd.isna(val):
            return None, None

    # Quality filters - exclude stocks with suspicious/extreme values
    # These thresholds catch data quality issues
    if abs(factors['roa']) > 0.5:  # ROA > 50% is suspicious
        return None, None
    if abs(factors['ocf_assets']) > 0.75:  # OCF/Assets > 75% is suspicious
        return None, None
    if abs(factors['fcf_assets']) > 0.75:  # FCF/Assets > 75% is suspicious
        return None, None
    if abs(factors['gp_assets']) > 1.0:  # GP/Assets > 100% is suspicious
        return None, None
    if factors['asset_growth'] < -0.5:  # Asset shrinkage > 50% is suspicious
        return None, None

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

    return raw_score, factors


def compute_all_scores():
    """Compute Compass Scores for all stocks."""
    prices, fund = load_data()

    print("\nComputing raw scores for all stocks...")
    sys.stdout.flush()

    # Get all symbols that have both price and fundamental data
    price_symbols = set(prices['symbol'].unique())
    fund_symbols = set(fund['symbol'].unique())
    all_symbols = price_symbols & fund_symbols

    print(f"  {len(all_symbols):,} stocks with both price and fundamental data")
    sys.stdout.flush()

    # Compute raw scores
    results = []
    for i, symbol in enumerate(all_symbols):
        raw_score, factors = compute_raw_score(symbol, prices, fund)
        if raw_score is not None:
            results.append({
                'symbol': symbol,
                'raw_score': raw_score,
                'roa': factors['roa'],
                'vol_60d': factors['vol_60d']
            })

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1:,}/{len(all_symbols):,} processed...")
            sys.stdout.flush()

    df = pd.DataFrame(results)
    print(f"  {len(df):,} stocks with valid Compass Scores")
    sys.stdout.flush()

    # Percentile base (maintains ~20% per grade) with raw score tiers at top
    percentile = df['raw_score'].rank(pct=True)
    df['compass_score'] = (percentile * 100).round(0).astype(int)

    # Top tier: use raw score to spread out the best stocks
    # Tighter thresholds = fewer stocks per tier
    df.loc[df['raw_score'] >= 0.15, 'compass_score'] = 96
    df.loc[df['raw_score'] >= 0.20, 'compass_score'] = 97
    df.loc[df['raw_score'] >= 0.28, 'compass_score'] = 98
    df.loc[df['raw_score'] >= 0.40, 'compass_score'] = 99
    df.loc[df['raw_score'] >= 0.55, 'compass_score'] = 100

    # Assign grades
    df['compass_grade'] = df['compass_score'].apply(assign_grade)

    return df[['symbol', 'compass_score', 'compass_grade', 'raw_score']]


def update_nasdaq_db(scores_df):
    """Update nasdaq_stocks.db with Compass Scores."""
    print("\nUpdating nasdaq_stocks.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(NASDAQ_DB)
    cursor = conn.cursor()

    # Add columns if they don't exist
    try:
        cursor.execute("ALTER TABLE stock_consensus ADD COLUMN compass_score INTEGER")
        print("  Added compass_score column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE stock_consensus ADD COLUMN compass_grade TEXT")
        print("  Added compass_grade column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    try:
        cursor.execute("ALTER TABLE stock_consensus ADD COLUMN compass_updated_at TEXT")
        print("  Added compass_updated_at column")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Clear ALL existing compass scores first
    # This ensures filtered-out stocks don't retain old scores
    cursor.execute("""
        UPDATE stock_consensus
        SET compass_score = NULL, compass_grade = NULL, compass_updated_at = NULL
    """)
    cleared = cursor.rowcount
    print(f"  Cleared {cleared:,} existing scores")

    # Update scores
    updated = 0
    timestamp = datetime.now().isoformat()

    for _, row in scores_df.iterrows():
        cursor.execute("""
            UPDATE stock_consensus
            SET compass_score = ?, compass_grade = ?, compass_updated_at = ?
            WHERE symbol = ?
        """, (row['compass_score'], row['compass_grade'], timestamp, row['symbol']))
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"  Updated {updated:,} stocks in stock_consensus")
    sys.stdout.flush()


def print_examples(scores_df):
    """Print example stocks for verification."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Famous Stock Examples")
    print("=" * 60)

    famous = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'JPM', 'SMCI', 'PLTR', 'COIN', 'FLWS']

    print(f"\n{'Symbol':<8} {'Score':<8} {'Grade':<6} {'Raw':<10}")
    print("-" * 40)

    for symbol in famous:
        row = scores_df[scores_df['symbol'] == symbol]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{symbol:<8} {r['compass_score']:<8} {r['compass_grade']:<6} {r['raw_score']:+.2f}")
        else:
            print(f"{symbol:<8} N/A")

    print("\nExpected (from research paper):")
    print("  AAPL: ~99, NVDA: ~99, TSLA: ~29, SMCI: ~11")
    sys.stdout.flush()


def main():
    print("=" * 60)
    print("COMPASS SCORE COMPUTATION")
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
    grade_counts = scores_df['compass_grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        pct = count / len(scores_df) * 100
        print(f"  {grade}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

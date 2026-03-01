#!/usr/bin/env python3
"""
Compute Moonshot Scores for Dashboard
======================================
Identifies stocks with high likelihood of significant long-term gains.

Factors (based on academic research on long-term outperformance):
1. Revenue Growth (TTM YoY) - 25% weight
2. EPS Growth (TTM YoY) - 20% weight
3. Gross Margin Level - 15% weight (high margins = competitive advantage)
4. Gross Margin Improvement - 10% weight
5. Smaller Market Cap - 15% weight (more room to grow)
6. 12-1 Month Momentum - 15% weight (winners keep winning)
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
    'revenue_growth': 0.25,
    'eps_growth': 0.20,
    'gross_margin': 0.15,
    'margin_improvement': 0.10,
    'small_cap': 0.15,
    'momentum_12_1': 0.15
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

    # Get prices for momentum calculation (last 14 months)
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
          AND date >= date('now', '-14 months')
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Get fundamentals (last 2+ years for growth calculations)
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.revenue, i.gross_profit, i.eps_diluted as eps,
               m.market_cap
        FROM historical_income_statements i
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        WHERE i.date >= date('now', '-3 years')
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    conn.close()

    # Sort by symbol and date
    fund = fund.sort_values(['symbol', 'date'])

    # Compute TTM metrics
    print("  Computing TTM metrics...")
    sys.stdout.flush()

    for col in ['revenue', 'gross_profit', 'eps']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Compute gross margin
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # Compute YoY growth (compare to 4 quarters ago)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Clean up infinities
    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")
    sys.stdout.flush()

    return prices, fund


def compute_momentum_12_1(prices_df, symbol):
    """Compute 12-1 month momentum (skip most recent month)."""
    sym_prices = prices_df[prices_df['symbol'] == symbol].sort_values('date')

    if len(sym_prices) < 252:  # Need ~1 year of data
        return np.nan

    # Price 12 months ago
    price_12m = sym_prices.iloc[-252]['close']
    # Price 1 month ago (skip last ~21 trading days)
    price_1m = sym_prices.iloc[-21]['close']

    if price_12m <= 0:
        return np.nan

    return (price_1m - price_12m) / price_12m


def compute_raw_score(symbol, prices_df, fund_df):
    """Compute raw Moonshot Score for a single symbol."""
    # Get latest fundamentals
    sym_fund = fund_df[fund_df['symbol'] == symbol]
    if len(sym_fund) == 0:
        return None, None

    latest_fund = sym_fund.sort_values('date').iloc[-1]

    # Get factor values
    factors = {
        'revenue_growth': latest_fund.get('revenue_growth', np.nan),
        'eps_growth': latest_fund.get('eps_growth', np.nan),
        'gross_margin': latest_fund.get('gross_margin', np.nan),
        'margin_improvement': latest_fund.get('margin_improvement', np.nan),
        'market_cap': latest_fund.get('market_cap', np.nan)
    }

    # Compute momentum
    factors['momentum_12_1'] = compute_momentum_12_1(prices_df, symbol)

    # Check for critical missing values
    if pd.isna(factors['revenue_growth']) or pd.isna(factors['gross_margin']):
        return None, None
    if pd.isna(factors['market_cap']) or factors['market_cap'] <= 0:
        return None, None

    # Quality filters - exclude suspicious data
    if factors['revenue_growth'] > 5.0:  # 500% revenue growth is suspicious
        return None, None
    if factors['revenue_growth'] < -0.8:  # 80% revenue decline
        return None, None
    if factors['gross_margin'] < 0 or factors['gross_margin'] > 1:
        return None, None

    # Handle missing EPS growth (use 0 as neutral)
    if pd.isna(factors['eps_growth']):
        factors['eps_growth'] = 0
    if pd.isna(factors['margin_improvement']):
        factors['margin_improvement'] = 0
    if pd.isna(factors['momentum_12_1']):
        factors['momentum_12_1'] = 0

    # Cap extreme values
    factors['revenue_growth'] = np.clip(factors['revenue_growth'], -0.5, 2.0)
    factors['eps_growth'] = np.clip(factors['eps_growth'], -0.5, 3.0)
    factors['momentum_12_1'] = np.clip(factors['momentum_12_1'], -0.5, 2.0)

    # Compute small cap score (inverse of market cap, log-scaled)
    # Smaller companies score higher
    factors['small_cap'] = -np.log10(factors['market_cap'] / 1e9)  # log of billions

    return factors, factors


def compute_all_scores():
    """Compute Moonshot Scores for all stocks."""
    prices, fund = load_data()

    print("\nComputing raw scores for all stocks...")
    sys.stdout.flush()

    # Get all symbols that have both price and fundamental data
    price_symbols = set(prices['symbol'].unique())
    fund_symbols = set(fund['symbol'].unique())
    all_symbols = price_symbols & fund_symbols

    print(f"  {len(all_symbols):,} stocks with both price and fundamental data")
    sys.stdout.flush()

    # Compute raw factor values
    results = []
    for i, symbol in enumerate(all_symbols):
        factors, _ = compute_raw_score(symbol, prices, fund)
        if factors is not None:
            results.append({
                'symbol': symbol,
                'revenue_growth': factors['revenue_growth'],
                'eps_growth': factors['eps_growth'],
                'gross_margin': factors['gross_margin'],
                'margin_improvement': factors['margin_improvement'],
                'small_cap': factors['small_cap'],
                'momentum_12_1': factors['momentum_12_1']
            })

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1:,}/{len(all_symbols):,} processed...")
            sys.stdout.flush()

    df = pd.DataFrame(results)
    print(f"  {len(df):,} stocks with valid factor data")
    sys.stdout.flush()

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
    df['moonshot_score'] = (percentile * 100).round(0).astype(int)

    # Spread out top tier based on raw score
    df.loc[df['raw_score'] >= 0.8, 'moonshot_score'] = 96
    df.loc[df['raw_score'] >= 1.0, 'moonshot_score'] = 97
    df.loc[df['raw_score'] >= 1.3, 'moonshot_score'] = 98
    df.loc[df['raw_score'] >= 1.6, 'moonshot_score'] = 99
    df.loc[df['raw_score'] >= 2.0, 'moonshot_score'] = 100

    # Assign grades
    df['moonshot_grade'] = df['moonshot_score'].apply(assign_grade)

    return df


def update_nasdaq_db(scores_df):
    """Update nasdaq_stocks.db with Moonshot Scores."""
    print("\nUpdating nasdaq_stocks.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(NASDAQ_DB)
    cursor = conn.cursor()

    # Add columns if they don't exist
    for col, dtype in [('moonshot_score', 'INTEGER'),
                       ('moonshot_grade', 'TEXT'),
                       ('moonshot_updated_at', 'TEXT')]:
        try:
            cursor.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col} {dtype}")
            print(f"  Added {col} column")
        except sqlite3.OperationalError:
            pass

    # Clear existing scores
    cursor.execute("""
        UPDATE stock_consensus
        SET moonshot_score = NULL, moonshot_grade = NULL, moonshot_updated_at = NULL
    """)
    cleared = cursor.rowcount
    print(f"  Cleared {cleared:,} existing scores")

    # Update scores
    updated = 0
    timestamp = datetime.now().isoformat()

    for _, row in scores_df.iterrows():
        cursor.execute("""
            UPDATE stock_consensus
            SET moonshot_score = ?, moonshot_grade = ?, moonshot_updated_at = ?
            WHERE symbol = ?
        """, (row['moonshot_score'], row['moonshot_grade'], timestamp, row['symbol']))
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

    print(f"\n{'Symbol':<8} {'Score':<8} {'Grade':<6} {'RevGr':<10} {'EPSGr':<10} {'Mom12-1':<10}")
    print("-" * 60)

    for symbol in famous:
        row = scores_df[scores_df['symbol'] == symbol]
        if len(row) > 0:
            r = row.iloc[0]
            print(f"{symbol:<8} {r['moonshot_score']:<8} {r['moonshot_grade']:<6} "
                  f"{r['revenue_growth']:+.1%}    {r['eps_growth']:+.1%}    {r['momentum_12_1']:+.1%}")
        else:
            print(f"{symbol:<8} N/A")

    # Show top 10 moonshot stocks
    print("\n" + "=" * 60)
    print("TOP 10 MOONSHOT STOCKS")
    print("=" * 60)
    top10 = scores_df.nlargest(10, 'raw_score')
    for _, r in top10.iterrows():
        print(f"{r['symbol']:<8} {r['moonshot_score']:<8} RevGr: {r['revenue_growth']:+.1%}")

    sys.stdout.flush()


def main():
    print("=" * 60)
    print("MOONSHOT SCORE COMPUTATION")
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
    grade_counts = scores_df['moonshot_grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        pct = count / len(scores_df) * 100
        print(f"  {grade}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

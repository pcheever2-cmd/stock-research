#!/usr/bin/env python3
"""
Sentiment Backtest Analysis
===========================
Tests whether news sentiment scores are predictive of future stock returns.

Methodology:
1. Fetch current sentiment for stocks with analyst coverage
2. Group stocks into sentiment quintiles (1=most negative, 5=most positive)
3. Analyze forward returns over 1, 2, and 4 week periods
4. Calculate if high-sentiment stocks outperform low-sentiment stocks

Note: Since we can only get current news, this is a point-in-time analysis.
Run this script, then check back in 1-4 weeks to see actual forward returns.
"""

import sqlite3
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import from existing modules
from fetch_news_sentiment import calculate_news_sentiment, fetch_stock_news, analyze_sentiment

PROJECT_ROOT = Path(__file__).parent
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Rate limiting
MAX_WORKERS = 5


def get_stocks_for_analysis(limit: int = 500) -> pd.DataFrame:
    """Get stocks with analyst coverage and recent price data."""
    conn = sqlite3.connect(MAIN_DB)

    df = pd.read_sql_query("""
        SELECT
            symbol,
            company_name,
            sector,
            current_price,
            num_analysts,
            long_term_score,
            value_score_v2,
            forward_pe,
            peg_ratio
        FROM stock_consensus
        WHERE num_analysts >= 3
          AND current_price > 5
          AND current_price IS NOT NULL
        ORDER BY num_analysts DESC
        LIMIT ?
    """, conn, params=(limit,))

    conn.close()
    return df


def get_historical_returns(symbols: list, lookback_days: int = 30) -> dict:
    """Get historical returns for symbols from backtest database."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Get the most recent date in our price data
    max_date = conn.execute("SELECT MAX(date) FROM historical_prices").fetchone()[0]

    if not max_date:
        conn.close()
        return {}

    start_date = (datetime.strptime(max_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    returns = {}
    for symbol in symbols:
        prices = pd.read_sql_query("""
            SELECT date, close
            FROM historical_prices
            WHERE symbol = ? AND date >= ?
            ORDER BY date
        """, conn, params=(symbol, start_date))

        if len(prices) >= 2:
            start_price = prices.iloc[0]['close']
            end_price = prices.iloc[-1]['close']
            if start_price and start_price > 0:
                returns[symbol] = ((end_price - start_price) / start_price) * 100

    conn.close()
    return returns


def fetch_sentiment_batch(symbols: list) -> dict:
    """Fetch sentiment for multiple symbols."""
    print(f"Fetching sentiment for {len(symbols)} symbols...")

    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(calculate_news_sentiment, sym): sym for sym in symbols}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            symbol = futures[future]
            result = future.result()
            if result:
                results[symbol] = result

            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(symbols)} ({len(results)} with news)")

    return results


def run_backtest():
    """Run the sentiment backtest analysis."""
    print("=" * 70)
    print("SENTIMENT BACKTEST ANALYSIS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Get stocks for analysis
    print("\n1. Loading stocks with analyst coverage...")
    stocks = get_stocks_for_analysis(limit=500)
    print(f"   Found {len(stocks)} stocks")

    # Get historical returns (past 30 days)
    print("\n2. Fetching historical returns (past 30 days)...")
    hist_returns = get_historical_returns(stocks['symbol'].tolist(), lookback_days=30)
    print(f"   Got returns for {len(hist_returns)} stocks")

    # Fetch sentiment
    print("\n3. Fetching news sentiment...")
    sentiments = fetch_sentiment_batch(stocks['symbol'].tolist())
    print(f"   Got sentiment for {len(sentiments)} stocks")

    # Merge data
    stocks['sentiment_score'] = stocks['symbol'].map(lambda x: sentiments.get(x, {}).get('news_sentiment_score'))
    stocks['article_count'] = stocks['symbol'].map(lambda x: sentiments.get(x, {}).get('news_article_count'))
    stocks['positive_pct'] = stocks['symbol'].map(lambda x: sentiments.get(x, {}).get('news_positive_pct'))
    stocks['past_30d_return'] = stocks['symbol'].map(hist_returns)

    # Filter to stocks with sentiment data
    analysis_df = stocks.dropna(subset=['sentiment_score', 'past_30d_return'])
    print(f"\n4. Analyzing {len(analysis_df)} stocks with both sentiment and price data")

    if len(analysis_df) < 50:
        print("ERROR: Not enough data for meaningful analysis")
        return

    # Create sentiment quintiles
    analysis_df['sentiment_quintile'] = pd.qcut(
        analysis_df['sentiment_score'],
        q=5,
        labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
    )

    # Analyze returns by quintile
    print("\n" + "=" * 70)
    print("RESULTS: Past 30-Day Returns by Sentiment Quintile")
    print("=" * 70)
    print("(Higher sentiment = more positive news coverage)")
    print()

    quintile_stats = analysis_df.groupby('sentiment_quintile').agg({
        'past_30d_return': ['mean', 'median', 'std', 'count'],
        'sentiment_score': 'mean'
    }).round(2)

    print(f"{'Quintile':<12} {'Avg Sent':>10} {'Avg Ret%':>10} {'Med Ret%':>10} {'Std':>10} {'Count':>8}")
    print("-" * 62)

    for quintile in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
        if quintile in quintile_stats.index:
            row = quintile_stats.loc[quintile]
            avg_sent = row[('sentiment_score', 'mean')]
            avg_ret = row[('past_30d_return', 'mean')]
            med_ret = row[('past_30d_return', 'median')]
            std_ret = row[('past_30d_return', 'std')]
            count = int(row[('past_30d_return', 'count')])
            print(f"{quintile:<12} {avg_sent:>10.1f} {avg_ret:>+10.2f} {med_ret:>+10.2f} {std_ret:>10.2f} {count:>8}")

    # Calculate spread
    q5_ret = analysis_df[analysis_df['sentiment_quintile'] == 'Q5 (High)']['past_30d_return'].mean()
    q1_ret = analysis_df[analysis_df['sentiment_quintile'] == 'Q1 (Low)']['past_30d_return'].mean()
    spread = q5_ret - q1_ret

    print("-" * 62)
    print(f"{'Q5-Q1 Spread':<12} {'':>10} {spread:>+10.2f}%")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    corr_ret = analysis_df['sentiment_score'].corr(analysis_df['past_30d_return'])
    print(f"Sentiment vs Past 30d Return: {corr_ret:+.3f}")

    if 'value_score_v2' in analysis_df.columns:
        valid_v2 = analysis_df.dropna(subset=['value_score_v2'])
        if len(valid_v2) > 20:
            corr_v2 = valid_v2['sentiment_score'].corr(valid_v2['value_score_v2'])
            print(f"Sentiment vs V2 Score:        {corr_v2:+.3f}")

    if 'long_term_score' in analysis_df.columns:
        valid_analyst = analysis_df.dropna(subset=['long_term_score'])
        if len(valid_analyst) > 20:
            corr_analyst = valid_analyst['sentiment_score'].corr(valid_analyst['long_term_score'])
            print(f"Sentiment vs Long-Term Score: {corr_analyst:+.3f}")

    # Top and bottom stocks by sentiment
    print("\n" + "=" * 70)
    print("TOP 10 HIGHEST SENTIMENT STOCKS")
    print("=" * 70)
    top_sentiment = analysis_df.nlargest(10, 'sentiment_score')[
        ['symbol', 'sentiment_score', 'article_count', 'positive_pct', 'past_30d_return', 'value_score_v2']
    ]
    print(f"{'Symbol':<8} {'Sent':>6} {'Arts':>6} {'Pos%':>6} {'30d Ret':>10} {'V2':>6}")
    print("-" * 50)
    for _, row in top_sentiment.iterrows():
        v2 = f"{row['value_score_v2']:.0f}" if pd.notna(row['value_score_v2']) else "N/A"
        print(f"{row['symbol']:<8} {row['sentiment_score']:>6.0f} {row['article_count']:>6.0f} {row['positive_pct']:>5.1f}% {row['past_30d_return']:>+9.2f}% {v2:>6}")

    print("\n" + "=" * 70)
    print("TOP 10 LOWEST SENTIMENT STOCKS")
    print("=" * 70)
    bottom_sentiment = analysis_df.nsmallest(10, 'sentiment_score')[
        ['symbol', 'sentiment_score', 'article_count', 'positive_pct', 'past_30d_return', 'value_score_v2']
    ]
    print(f"{'Symbol':<8} {'Sent':>6} {'Arts':>6} {'Pos%':>6} {'30d Ret':>10} {'V2':>6}")
    print("-" * 50)
    for _, row in bottom_sentiment.iterrows():
        v2 = f"{row['value_score_v2']:.0f}" if pd.notna(row['value_score_v2']) else "N/A"
        print(f"{row['symbol']:<8} {row['sentiment_score']:>6.0f} {row['article_count']:>6.0f} {row['positive_pct']:>5.1f}% {row['past_30d_return']:>+9.2f}% {v2:>6}")

    # Sector analysis
    print("\n" + "=" * 70)
    print("SENTIMENT BY SECTOR")
    print("=" * 70)
    sector_stats = analysis_df.groupby('sector').agg({
        'sentiment_score': 'mean',
        'past_30d_return': 'mean',
        'symbol': 'count'
    }).round(2)
    sector_stats.columns = ['Avg Sentiment', 'Avg 30d Return', 'Count']
    sector_stats = sector_stats.sort_values('Avg Sentiment', ascending=False)

    print(f"{'Sector':<25} {'Sent':>8} {'30d Ret':>10} {'Count':>8}")
    print("-" * 55)
    for sector, row in sector_stats.iterrows():
        if pd.notna(sector) and row['Count'] >= 5:
            print(f"{str(sector)[:25]:<25} {row['Avg Sentiment']:>8.1f} {row['Avg 30d Return']:>+9.2f}% {int(row['Count']):>8}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if spread > 2:
        print("✅ POSITIVE SIGNAL: High-sentiment stocks outperformed low-sentiment stocks")
        print(f"   by {spread:.2f}% over the past 30 days.")
        print("   Sentiment may be a useful confirming indicator.")
    elif spread < -2:
        print("⚠️  CONTRARIAN SIGNAL: Low-sentiment stocks outperformed high-sentiment stocks")
        print(f"   by {abs(spread):.2f}% over the past 30 days.")
        print("   This suggests potential mean reversion - beaten-down stocks recovering.")
    else:
        print("➖ NEUTRAL: No significant relationship between sentiment and returns")
        print(f"   (spread of only {spread:.2f}%)")

    if abs(corr_ret) > 0.15:
        direction = "positive" if corr_ret > 0 else "negative"
        print(f"\n📊 Correlation of {corr_ret:+.3f} suggests a {direction} relationship")
        print("   between news sentiment and recent stock performance.")

    # Save results for tracking
    results_file = PROJECT_ROOT / 'data' / 'sentiment_backtest_results.csv'
    analysis_df.to_csv(results_file, index=False)
    print(f"\n📁 Full results saved to: {results_file}")
    print("   Track these stocks over the next 1-4 weeks to measure forward returns.")


if __name__ == '__main__':
    run_backtest()

#!/usr/bin/env python3
"""
News Sentiment Analysis
=======================
Fetches stock news from FMP and calculates sentiment scores using VADER.
Stores sentiment data in the database for use in stock scoring.

Sentiment Score Interpretation:
- compound >= 0.05: Positive
- compound <= -0.05: Negative
- Otherwise: Neutral

News Sentiment Score (0-100):
- Based on average compound sentiment of recent news
- 50 = neutral, >50 = positive, <50 = negative
"""

import sqlite3
import requests
import os
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Installing vaderSentiment...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'vaderSentiment'])
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from config import FMP_API_KEY
except ImportError:
    FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Rate limiting
MAX_WORKERS = 5
RATE_LIMIT_DELAY = 0.25  # 4 requests/second

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def fetch_stock_news(symbol: str, limit: int = 10) -> list:
    """Fetch recent news for a stock from FMP."""
    if not FMP_API_KEY:
        return []

    url = f"https://financialmodelingprep.com/stable/news/stock?symbols={symbol}&limit={limit}&apikey={FMP_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        pass

    return []


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text using VADER.

    Returns dict with:
    - neg: negative sentiment (0-1)
    - neu: neutral sentiment (0-1)
    - pos: positive sentiment (0-1)
    - compound: overall sentiment (-1 to 1)
    """
    return analyzer.polarity_scores(text)


def calculate_news_sentiment(symbol: str) -> dict:
    """Calculate aggregated news sentiment for a stock.

    Returns:
    - news_sentiment_score: 0-100 score (50 = neutral)
    - news_article_count: number of articles analyzed
    - news_positive_pct: % of positive articles
    - news_negative_pct: % of negative articles
    - avg_compound: raw average compound score
    """
    news = fetch_stock_news(symbol, limit=20)

    if not news:
        return None

    sentiments = []
    positive_count = 0
    negative_count = 0

    for article in news:
        # Combine title and text for better sentiment analysis
        title = article.get('title', '')
        text = article.get('text', '')
        combined = f"{title}. {text}"

        sentiment = analyze_sentiment(combined)
        compound = sentiment['compound']
        sentiments.append(compound)

        if compound >= 0.05:
            positive_count += 1
        elif compound <= -0.05:
            negative_count += 1

    if not sentiments:
        return None

    # Calculate metrics
    avg_compound = sum(sentiments) / len(sentiments)
    # Convert compound (-1 to 1) to score (0 to 100)
    # -1 -> 0, 0 -> 50, 1 -> 100
    news_sentiment_score = int((avg_compound + 1) * 50)
    news_sentiment_score = max(0, min(100, news_sentiment_score))

    return {
        'symbol': symbol,
        'news_sentiment_score': news_sentiment_score,
        'news_article_count': len(sentiments),
        'news_positive_pct': round(positive_count / len(sentiments) * 100, 1),
        'news_negative_pct': round(negative_count / len(sentiments) * 100, 1),
        'avg_compound': round(avg_compound, 3),
    }


def fetch_sentiment_batch(symbols: list) -> list:
    """Fetch sentiment for multiple symbols using parallel requests."""
    print(f"\nFetching news sentiment for {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(calculate_news_sentiment, sym): sym for sym in symbols}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                results.append(result)

            if completed % 100 == 0:
                print(f"  Progress: {completed}/{len(symbols)} ({len(results)} with news)")

    return results


def ensure_sentiment_columns():
    """Add sentiment columns to stock_consensus if they don't exist."""
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    columns = [
        ('news_sentiment_score', 'INTEGER'),
        ('news_article_count', 'INTEGER'),
        ('news_positive_pct', 'REAL'),
        ('news_negative_pct', 'REAL'),
        ('news_sentiment_updated', 'TEXT'),
    ]

    cur.execute("PRAGMA table_info(stock_consensus)")
    existing = [row[1] for row in cur.fetchall()]

    for col_name, col_type in columns:
        if col_name not in existing:
            print(f"Adding column: {col_name}")
            cur.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col_name} {col_type}")

    conn.commit()
    conn.close()


def update_sentiment_data(sentiments: list):
    """Update database with sentiment scores."""
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    updated = 0
    for s in sentiments:
        cur.execute("""
            UPDATE stock_consensus
            SET news_sentiment_score = ?,
                news_article_count = ?,
                news_positive_pct = ?,
                news_negative_pct = ?,
                news_sentiment_updated = ?
            WHERE symbol = ?
        """, (
            s['news_sentiment_score'],
            s['news_article_count'],
            s['news_positive_pct'],
            s['news_negative_pct'],
            now,
            s['symbol']
        ))
        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()
    return updated


def save_sentiment_history(sentiments: list):
    """Save sentiment to historical table for future backtesting."""
    import sqlite3
    backtest_db = str(PROJECT_ROOT / 'backtest.db')
    conn = sqlite3.connect(backtest_db)

    # Create table if not exists
    conn.execute('''
        CREATE TABLE IF NOT EXISTS historical_sentiment (
            symbol TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            sentiment_score INTEGER,
            article_count INTEGER,
            positive_pct REAL,
            negative_pct REAL,
            avg_compound REAL,
            collected_at TEXT,
            PRIMARY KEY (symbol, snapshot_date)
        )
    ''')

    today = datetime.now().strftime('%Y-%m-%d')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    saved = 0
    for s in sentiments:
        try:
            conn.execute('''
                INSERT OR REPLACE INTO historical_sentiment
                (symbol, snapshot_date, sentiment_score, article_count,
                 positive_pct, negative_pct, avg_compound, collected_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                s['symbol'], today, s['news_sentiment_score'],
                s['news_article_count'], s['news_positive_pct'],
                s['news_negative_pct'], s.get('avg_compound', 0), now
            ))
            saved += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return saved


def main():
    print("=" * 60)
    print("NEWS SENTIMENT ANALYSIS")
    print("=" * 60)
    print(f"Database: {DATABASE_NAME}")
    print(f"API Key: {'Set' if FMP_API_KEY else 'NOT SET'}")

    if not FMP_API_KEY:
        print("Error: FMP_API_KEY not set")
        return

    # Ensure columns exist
    ensure_sentiment_columns()

    # Get symbols to analyze (prioritize those with analyst coverage)
    conn = sqlite3.connect(DATABASE_NAME)
    symbols = [r[0] for r in conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE num_analysts >= 1
        ORDER BY num_analysts DESC
        LIMIT 1000
    """).fetchall()]
    conn.close()

    print(f"\nAnalyzing {len(symbols)} stocks with analyst coverage")

    # Fetch sentiment
    sentiments = fetch_sentiment_batch(symbols)
    print(f"\nGot sentiment data for {len(sentiments)} stocks")

    # Update database
    updated = update_sentiment_data(sentiments)
    print(f"Updated {updated} stocks with sentiment scores")

    # Save to historical table for future backtesting
    saved = save_sentiment_history(sentiments)
    print(f"Saved {saved} records to historical sentiment table")

    # Print summary
    if sentiments:
        scores = [s['news_sentiment_score'] for s in sentiments]
        avg_score = sum(scores) / len(scores)
        positive = sum(1 for s in scores if s > 55)
        negative = sum(1 for s in scores if s < 45)
        neutral = len(scores) - positive - negative

        print("\n" + "=" * 60)
        print("SENTIMENT SUMMARY")
        print("=" * 60)
        print(f"Average Sentiment Score: {avg_score:.1f}/100 (50 = neutral)")
        print(f"Positive (>55): {positive} stocks ({positive/len(scores)*100:.1f}%)")
        print(f"Neutral (45-55): {neutral} stocks ({neutral/len(scores)*100:.1f}%)")
        print(f"Negative (<45): {negative} stocks ({negative/len(scores)*100:.1f}%)")

        # Show most positive/negative
        sorted_by_sentiment = sorted(sentiments, key=lambda x: x['news_sentiment_score'], reverse=True)
        print("\nMost Positive Sentiment:")
        for s in sorted_by_sentiment[:5]:
            print(f"  {s['symbol']:<6} Score: {s['news_sentiment_score']} ({s['news_article_count']} articles)")

        print("\nMost Negative Sentiment:")
        for s in sorted_by_sentiment[-5:]:
            print(f"  {s['symbol']:<6} Score: {s['news_sentiment_score']} ({s['news_article_count']} articles)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Collect Historical Sentiment Data
==================================
Fetches historical news from FMP and calculates sentiment for past periods.
This enables proper forward-looking backtests where we measure if sentiment
at time T predicted returns from T to T+30/60/90 days.
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
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Rate limiting
MAX_WORKERS = 3
RATE_LIMIT_DELAY = 0.3

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


def setup_sentiment_table():
    """Create historical sentiment table if it doesn't exist."""
    conn = sqlite3.connect(BACKTEST_DB)
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
    conn.execute('CREATE INDEX IF NOT EXISTS idx_hs_date ON historical_sentiment(snapshot_date)')
    conn.commit()
    conn.close()
    print("Historical sentiment table ready")


def fetch_historical_news(symbol: str, from_date: str, to_date: str, limit: int = 20) -> list:
    """Fetch news for a symbol within a date range."""
    if not FMP_API_KEY:
        return []

    url = f"https://financialmodelingprep.com/stable/news/stock?symbols={symbol}&from={from_date}&to={to_date}&limit={limit}&apikey={FMP_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        time.sleep(RATE_LIMIT_DELAY)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        pass

    return []


def calculate_sentiment_for_period(symbol: str, from_date: str, to_date: str) -> dict:
    """Calculate sentiment for a symbol during a specific period."""
    news = fetch_historical_news(symbol, from_date, to_date, limit=20)

    if not news:
        return None

    sentiments = []
    positive_count = 0
    negative_count = 0

    for article in news:
        title = article.get('title', '')
        text = article.get('text', '')
        combined = f"{title}. {text}"

        scores = analyzer.polarity_scores(combined)
        compound = scores['compound']
        sentiments.append(compound)

        if compound >= 0.05:
            positive_count += 1
        elif compound <= -0.05:
            negative_count += 1

    if not sentiments:
        return None

    avg_compound = sum(sentiments) / len(sentiments)
    sentiment_score = int((avg_compound + 1) * 50)
    sentiment_score = max(0, min(100, sentiment_score))

    return {
        'symbol': symbol,
        'sentiment_score': sentiment_score,
        'article_count': len(sentiments),
        'positive_pct': round(positive_count / len(sentiments) * 100, 1),
        'negative_pct': round(negative_count / len(sentiments) * 100, 1),
        'avg_compound': round(avg_compound, 3),
    }


def collect_sentiment_for_date(symbols: list, snapshot_date: str) -> list:
    """Collect sentiment for all symbols on a specific date (using 30-day window before date)."""
    to_date = snapshot_date
    from_date = (datetime.strptime(snapshot_date, '%Y-%m-%d') - timedelta(days=30)).strftime('%Y-%m-%d')

    print(f"\nCollecting sentiment for {snapshot_date} (news from {from_date} to {to_date})...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(calculate_sentiment_for_period, sym, from_date, to_date): sym
            for sym in symbols
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                result['snapshot_date'] = snapshot_date
                results.append(result)

            if completed % 50 == 0:
                print(f"  Progress: {completed}/{len(symbols)} ({len(results)} with news)")

    return results


def save_sentiment_data(sentiments: list):
    """Save sentiment data to database."""
    conn = sqlite3.connect(BACKTEST_DB)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for s in sentiments:
        conn.execute('''
            INSERT OR REPLACE INTO historical_sentiment
            (symbol, snapshot_date, sentiment_score, article_count,
             positive_pct, negative_pct, avg_compound, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            s['symbol'], s['snapshot_date'], s['sentiment_score'],
            s['article_count'], s['positive_pct'], s['negative_pct'],
            s['avg_compound'], now
        ))

    conn.commit()
    conn.close()
    return len(sentiments)


def get_symbols_to_analyze(limit: int = 300) -> list:
    """Get top symbols by analyst coverage."""
    conn = sqlite3.connect(str(PROJECT_ROOT / 'nasdaq_stocks.db'))
    symbols = [r[0] for r in conn.execute('''
        SELECT symbol FROM stock_consensus
        WHERE num_analysts >= 3 AND current_price > 5
        ORDER BY num_analysts DESC
        LIMIT ?
    ''', (limit,)).fetchall()]
    conn.close()
    return symbols


def main():
    print("=" * 70)
    print("HISTORICAL SENTIMENT COLLECTION")
    print("=" * 70)

    if not FMP_API_KEY:
        print("Error: FMP_API_KEY not set")
        return

    setup_sentiment_table()

    # Get symbols to analyze
    symbols = get_symbols_to_analyze(limit=300)
    print(f"Will collect sentiment for {len(symbols)} stocks")

    # Generate monthly snapshot dates (1st of each month for past 5 years = 60 months)
    today = datetime.now()
    snapshot_dates = []
    for months_ago in range(1, 61):  # 5 years = 60 months
        # First of each month
        d = today.replace(day=1) - timedelta(days=months_ago * 30)
        d = d.replace(day=1)
        snapshot_dates.append(d.strftime('%Y-%m-%d'))

    snapshot_dates = sorted(snapshot_dates)
    print(f"Snapshot dates: {snapshot_dates}")

    # Check which dates we already have
    conn = sqlite3.connect(BACKTEST_DB)
    existing = set(r[0] for r in conn.execute(
        'SELECT DISTINCT snapshot_date FROM historical_sentiment'
    ).fetchall())
    conn.close()

    dates_to_collect = [d for d in snapshot_dates if d not in existing]
    print(f"Already have: {existing}")
    print(f"Need to collect: {dates_to_collect}")

    if not dates_to_collect:
        print("\nAll historical sentiment data already collected!")
        return

    # Collect sentiment for each missing date
    total_saved = 0
    for snapshot_date in dates_to_collect:
        sentiments = collect_sentiment_for_date(symbols, snapshot_date)
        saved = save_sentiment_data(sentiments)
        total_saved += saved
        print(f"  Saved {saved} records for {snapshot_date}")

    print(f"\n{'=' * 70}")
    print(f"COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total records saved: {total_saved}")

    # Show summary
    conn = sqlite3.connect(BACKTEST_DB)
    summary = conn.execute('''
        SELECT snapshot_date, COUNT(*) as cnt, AVG(sentiment_score) as avg_sent
        FROM historical_sentiment
        GROUP BY snapshot_date
        ORDER BY snapshot_date
    ''').fetchall()
    conn.close()

    print(f"\n{'Date':<12} {'Stocks':>8} {'Avg Sentiment':>15}")
    print("-" * 38)
    for date, cnt, avg in summary:
        print(f"{date:<12} {cnt:>8} {avg:>15.1f}")


if __name__ == '__main__':
    main()

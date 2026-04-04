#!/usr/bin/env python3
"""
Backfill Country Data
=====================
Fetches country data for all stocks in the database from the FMP API.
"""

import sqlite3
import requests
import os
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')


def get_symbols_missing_country():
    """Get symbols that are missing country data."""
    conn = sqlite3.connect(DATABASE_NAME)
    rows = conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE country IS NULL OR country = ''
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def fetch_country_data(symbols: list) -> dict:
    """Fetch country data from FMP API."""
    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY set")
        return {}

    country_data = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    country = item.get('country')
                    if country:
                        country_data[symbol] = country

            # Progress update every 100 symbols
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total} ({len(country_data)} found)")

            time.sleep(0.05)  # Rate limiting - 20 requests/second
        except Exception as e:
            if i % 100 == 0:
                print(f"Error fetching {symbol}: {e}")

    return country_data


def update_database(country_data: dict):
    """Update database with country data."""
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    updated = 0
    for symbol, country in country_data.items():
        cur.execute("""
            UPDATE stock_consensus
            SET country = ?
            WHERE symbol = ?
        """, (country, symbol))
        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()
    return updated


def main():
    print("=" * 60)
    print("BACKFILLING COUNTRY DATA")
    print("=" * 60)

    # Get symbols missing country data
    missing = get_symbols_missing_country()
    print(f"Found {len(missing)} stocks missing country data")

    if not missing:
        print("All stocks have country data!")
        return

    # Fetch country data
    print(f"Fetching country data from FMP API...")
    country_data = fetch_country_data(missing)
    print(f"Retrieved country data for {len(country_data)} stocks")

    # Update database
    updated = update_database(country_data)
    print(f"Updated {updated} stock records")

    # Check remaining missing
    still_missing = get_symbols_missing_country()
    print(f"Still missing: {len(still_missing)} stocks")

    # Show country distribution
    conn = sqlite3.connect(DATABASE_NAME)
    country_dist = conn.execute("""
        SELECT country, COUNT(*) as count
        FROM stock_consensus
        WHERE country IS NOT NULL
        GROUP BY country
        ORDER BY count DESC
        LIMIT 20
    """).fetchall()
    conn.close()

    print("\nCountry distribution (top 20):")
    for country, count in country_dist:
        print(f"  {country}: {count} stocks")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Backfill Country Data (v2 - with batch updates)
===================================
Fetches country data for all stocks and updates database in batches.
"""

import sqlite3
import requests
import os
import time
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

BATCH_SIZE = 100  # Update database every 100 stocks


def get_symbols_missing_country():
    """Get symbols that are missing country data."""
    conn = sqlite3.connect(DATABASE_NAME)
    rows = conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE country IS NULL OR country = ''
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def fetch_and_update_country_data(symbols: list):
    """Fetch country data and update database in batches."""
    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY set", flush=True)
        return

    total = len(symbols)
    batch = {}
    total_updated = 0

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
                        batch[symbol] = country

            # Update database every BATCH_SIZE stocks
            if len(batch) >= BATCH_SIZE or i == total - 1:
                if batch:
                    conn = sqlite3.connect(DATABASE_NAME)
                    cur = conn.cursor()
                    for sym, country in batch.items():
                        cur.execute("UPDATE stock_consensus SET country = ? WHERE symbol = ?", (country, sym))
                    conn.commit()
                    conn.close()
                    total_updated += len(batch)
                    print(f"Progress: {i + 1}/{total} processed, {total_updated} updated", flush=True)
                    batch = {}

            time.sleep(0.05)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {symbol}: {e}", flush=True)

    return total_updated


def main():
    print("=" * 60, flush=True)
    print("BACKFILLING COUNTRY DATA (v2)", flush=True)
    print("=" * 60, flush=True)

    # Get symbols missing country data
    missing = get_symbols_missing_country()
    print(f"Found {len(missing)} stocks missing country data", flush=True)

    if not missing:
        print("All stocks have country data!", flush=True)
        return

    # Fetch and update country data
    print(f"Fetching country data from FMP API...", flush=True)
    total_updated = fetch_and_update_country_data(missing)
    print(f"\nCompleted! Updated {total_updated} stock records", flush=True)

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

    print("\nCountry distribution (top 20):", flush=True)
    for country, count in country_dist:
        print(f"  {country}: {count} stocks", flush=True)


if __name__ == '__main__':
    main()

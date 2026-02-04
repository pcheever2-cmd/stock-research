#!/usr/bin/env python3
"""
Fill Missing Profiles
=====================
Fetches company profile data for stocks that are missing basic info
(company_name, sector, industry) from the FMP API.
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

BATCH_SIZE = 50  # FMP allows up to 50 symbols per request


def get_missing_profile_symbols():
    """Get symbols that are missing profile data."""
    conn = sqlite3.connect(DATABASE_NAME)
    rows = conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE company_name IS NULL OR company_name = ''
           OR sector IS NULL
    """).fetchall()
    conn.close()
    return [r[0] for r in rows]


def fetch_profiles(symbols: list) -> dict:
    """Fetch company profiles from FMP API (one at a time since batch doesn't work)."""
    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY set")
        return {}

    profiles = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    item = data[0]
                    profiles[symbol] = {
                        'company_name': item.get('companyName'),
                        'sector': item.get('sector'),
                        'industry': item.get('industry'),
                        'description': item.get('description', '')[:500] if item.get('description') else None,
                        'market_cap': item.get('mktCap') or item.get('marketCap'),
                        'exchange': item.get('exchange'),
                    }

            # Progress update every 100 symbols
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i + 1}/{total} ({len(profiles)} found)")

            time.sleep(0.05)  # Rate limiting - 20 requests/second
        except Exception as e:
            if i % 100 == 0:
                print(f"Error fetching {symbol}: {e}")

    return profiles


def update_database(profiles: dict):
    """Update database with fetched profile data."""
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    updated = 0
    for symbol, data in profiles.items():
        cur.execute("""
            UPDATE stock_consensus
            SET company_name = COALESCE(?, company_name),
                sector = COALESCE(?, sector),
                industry = COALESCE(?, industry),
                company_description = COALESCE(?, company_description)
            WHERE symbol = ?
        """, (
            data.get('company_name'),
            data.get('sector'),
            data.get('industry'),
            data.get('description'),
            symbol
        ))
        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()
    return updated


def main():
    print("=" * 60)
    print("FILLING MISSING STOCK PROFILES")
    print("=" * 60)

    # Get symbols missing profile data
    missing = get_missing_profile_symbols()
    print(f"Found {len(missing)} stocks missing profile data")

    if not missing:
        print("All stocks have profile data!")
        return

    # Fetch profiles in batches
    print(f"Fetching profiles from FMP API...")
    profiles = fetch_profiles(missing)
    print(f"Retrieved {len(profiles)} profiles")

    # Update database
    updated = update_database(profiles)
    print(f"Updated {updated} stock records")

    # Check remaining missing
    still_missing = get_missing_profile_symbols()
    print(f"Still missing: {len(still_missing)} stocks")

    if still_missing and len(still_missing) <= 20:
        print("Missing symbols:", ', '.join(still_missing[:20]))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
COVERAGE EXPANSION - Discover and add stocks from all US exchanges
Uses FMP stable API to find actively traded stocks on NYSE, NASDAQ, and AMEX.
"""

import asyncio
import aiohttp
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Set

from config import FMP_API_KEY, DATABASE_NAME

BASE_URL = 'https://financialmodelingprep.com'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ==================== STOCK DISCOVERY ====================

async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None) -> list:
    """Generic JSON fetcher with error handling"""
    if params is None:
        params = {}
    params['apikey'] = FMP_API_KEY

    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data if isinstance(data, list) else []
            else:
                text = await response.text()
                log.warning(f"API {response.status} for {url}: {text[:200]}")
                return []
    except Exception as e:
        log.error(f"Fetch error for {url}: {e}")
        return []


async def discover_from_screener(session: aiohttp.ClientSession) -> Set[str]:
    """Use FMP stable company-screener to get all actively traded US stocks"""
    log.info("Discovering stocks from company screener...")

    discovered = set()
    screener_url = f"{BASE_URL}/stable/company-screener"

    for exchange in ['NYSE', 'NASDAQ', 'AMEX']:
        results = await fetch_json(session, screener_url, {
            'exchange': exchange,
            'isActivelyTrading': 'true',
            'isEtf': 'false',
            'isFund': 'false',
            'limit': 10000,
        })
        symbols = [
            r['symbol'] for r in results
            if r.get('symbol')
            and r.get('exchangeShortName') in ('NYSE', 'NASDAQ', 'AMEX')
            and not r.get('symbol', '').endswith(('-UN', '-WT', '.WS'))
        ]
        discovered.update(symbols)
        log.info(f"  {exchange}: {len(symbols)} stocks")
        await asyncio.sleep(0.3)

    return discovered


async def discover_from_earnings(session: aiohttp.ClientSession) -> Set[str]:
    """Get stocks from upcoming earnings calendar (stable endpoint)"""
    log.info("Discovering stocks from earnings calendar...")

    discovered = set()
    today = datetime.now()

    for days_ahead in [0, 7, 14, 30]:
        from_date = (today + timedelta(days=max(0, days_ahead - 3))).strftime('%Y-%m-%d')
        to_date = (today + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
        results = await fetch_json(session, f"{BASE_URL}/stable/earnings-calendar", {
            'from': from_date,
            'to': to_date,
        })
        symbols = {
            r.get('symbol') for r in results
            if r.get('symbol')
            and '.' not in r.get('symbol', '')  # Exclude non-US (e.g. .L, .TO)
            and '-' not in r.get('symbol', '')   # Exclude units/warrants
        }
        discovered.update(symbols)
        await asyncio.sleep(0.2)

    log.info(f"  Found {len(discovered)} US stocks with upcoming earnings")
    return discovered


# ==================== DATABASE OPERATIONS ====================

def get_existing_symbols() -> Set[str]:
    """Get symbols already in database"""
    conn = sqlite3.connect(DATABASE_NAME)
    rows = conn.execute("SELECT DISTINCT symbol FROM stock_consensus").fetchall()
    conn.close()
    return {r[0] for r in rows}


def add_new_symbols(new_symbols: List[str]):
    """Add new symbols to database (minimal placeholder rows)"""
    if not new_symbols:
        log.info("No new symbols to add")
        return 0

    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()
    added = 0

    for symbol in sorted(new_symbols):
        try:
            cur.execute("""
                INSERT OR IGNORE INTO stock_consensus (symbol, last_updated)
                VALUES (?, NULL)
            """, (symbol,))
            if cur.rowcount > 0:
                added += 1
        except Exception as e:
            log.warning(f"Could not add {symbol}: {e}")

    conn.commit()
    conn.close()
    log.info(f"Added {added} new symbols to database")
    return added


# ==================== MAIN ====================

async def main():
    log.info("=" * 60)
    log.info("COVERAGE EXPANSION - All US Exchanges (NYSE + NASDAQ + AMEX)")
    log.info("=" * 60)

    existing = get_existing_symbols()
    log.info(f"Current database: {len(existing)} stocks\n")

    all_discovered = set()

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            discover_from_screener(session),
            discover_from_earnings(session),
        )
        for s in results:
            all_discovered.update(s)

    new_symbols = all_discovered - existing
    overlap = all_discovered & existing

    log.info(f"\nDiscovery Results:")
    log.info(f"  Total discovered: {len(all_discovered)}")
    log.info(f"  Already in database: {len(overlap)}")
    log.info(f"  NEW stocks to add: {len(new_symbols)}")

    if new_symbols:
        log.info(f"\nSample of new stocks: {sorted(list(new_symbols))[:30]}")
        response = input(f"\nAdd {len(new_symbols)} new stocks to database? (y/n): ")
        if response.lower() == 'y':
            added = add_new_symbols(list(new_symbols))
            log.info(f"\nCoverage expansion complete! Added {added} stocks.")
            log.info("Next steps:")
            log.info("  1. Run: python run_pipeline_OPTIMIZED.py  (fetch data + score)")
            log.info("  2. Push updated parquet to deploy")
        else:
            log.info("Cancelled - no stocks added")
    else:
        log.info("\nNo new stocks found - coverage is already comprehensive!")

    log.info("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

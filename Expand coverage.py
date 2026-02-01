#!/usr/bin/env python3
"""
COVERAGE EXPANSION - Discover and add more stocks to database
Uses FMP screener to find stocks with analyst coverage that aren't in your DB yet
"""

import asyncio
import aiohttp
import sqlite3
import logging
from typing import List, Set
import pandas as pd

from config import FMP_API_KEY, DATABASE_NAME
BASE_URL = 'https://financialmodelingprep.com'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ==================== STOCK DISCOVERY ====================

async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None) -> list:
    """Generic JSON fetcher"""
    if params is None:
        params = {}
    params['apikey'] = FMP_API_KEY
    
    try:
        async with session.get(url, params=params) as response:
            if response.status == 200:
                return await response.json()
            elif response.status == 403:
                # Get the response text to see the error message
                error_text = await response.text()
                log.warning(f"API error 403 for {url}: {error_text[:200]}")
                return []
            else:
                log.warning(f"API error {response.status} for {url}")
                return []
    except Exception as e:
        log.error(f"Fetch error: {e}")
        return []

async def discover_stocks_from_screener(session: aiohttp.ClientSession) -> Set[str]:
    """Use FMP stock screener to find stocks with analyst coverage"""
    log.info("Discovering stocks from screener...")
    
    discovered = set()
    
    # Strategy 1: Get stocks with price targets
    screener_url = f"{BASE_URL}/api/v3/stock-screener"
    
    # Multiple screener queries to maximize coverage
    queries = [
        # Large caps with coverage
        {
            'marketCapMoreThan': 10_000_000_000,
            'limit': 1000
        },
        # Mid caps with coverage
        {
            'marketCapMoreThan': 2_000_000_000,
            'marketCapLowerThan': 10_000_000_000,
            'limit': 1000
        },
        # Small caps with good volume
        {
            'marketCapMoreThan': 300_000_000,
            'marketCapLowerThan': 2_000_000_000,
            'volumeMoreThan': 100_000,
            'limit': 1000
        },
        # High growth stocks
        {
            'marketCapMoreThan': 100_000_000,
            'limit': 500
        }
    ]
    
    for query in queries:
        results = await fetch_json(session, screener_url, query)
        symbols = [r['symbol'] for r in results if r.get('symbol') and r.get('exchangeShortName') in ['NASDAQ', 'NYSE', 'AMEX']]
        discovered.update(symbols)
        log.info(f"  Found {len(symbols)} stocks (total unique: {len(discovered)})")
        await asyncio.sleep(0.5)  # Rate limiting
    
    return discovered

async def discover_from_indexes(session: aiohttp.ClientSession) -> Set[str]:
    """Get stocks from major indexes"""
    log.info("Discovering stocks from major indexes...")
    
    discovered = set()
    
    indexes = ['SPY', 'QQQ', 'DIA', 'IWM', 'MDY']  # S&P500, Nasdaq100, Dow30, Russell2000, MidCap
    
    for index in indexes:
        # Try v4 endpoint first (newer)
        url = f"{BASE_URL}/api/v4/etf-holder/{index}"
        holdings = await fetch_json(session, url)
        
        # If v4 fails, try v3
        if not holdings:
            url = f"{BASE_URL}/api/v3/etf-holder/{index}"
            holdings = await fetch_json(session, url)
        
        if holdings:
            symbols = [h.get('asset') or h.get('symbol') for h in holdings]
            symbols = [s for s in symbols if s]  # Filter out None values
            discovered.update(symbols)
            log.info(f"  {index}: {len(symbols)} holdings")
        else:
            log.warning(f"  {index}: No holdings data available")
        
        await asyncio.sleep(0.3)
    
    return discovered

async def discover_from_analyst_upgrades(session: aiohttp.ClientSession) -> Set[str]:
    """Get stocks from recent analyst upgrades/downgrades"""
    log.info("Discovering stocks from recent analyst activity...")
    
    # Try v4 endpoint first
    url = f"{BASE_URL}/api/v4/upgrades-downgrades"
    params = {'limit': 500}
    activity = await fetch_json(session, url, params)
    
    # If v4 fails, try v3
    if not activity:
        url = f"{BASE_URL}/api/v3/upgrades-downgrades"
        activity = await fetch_json(session, url, params)
    
    if activity:
        symbols = {a.get('symbol') for a in activity if a.get('symbol')}
        log.info(f"  Found {len(symbols)} stocks with recent analyst activity")
        return symbols
    else:
        log.warning("  Could not fetch analyst activity data")
        return set()

async def discover_from_earnings(session: aiohttp.ClientSession) -> Set[str]:
    """Get stocks with upcoming earnings (= likely have coverage)"""
    log.info("Discovering stocks from earnings calendar...")
    
    discovered = set()
    
    # Get earnings for next few days
    for days_ahead in [0, 1, 2, 3, 7, 14]:
        # Try v4 first
        url = f"{BASE_URL}/api/v4/earnings-calendar"
        params = {'from': days_ahead, 'to': days_ahead}
        
        earnings = await fetch_json(session, url, params)
        
        # If v4 fails, try v3
        if not earnings:
            url = f"{BASE_URL}/api/v3/earnings-calendar"
            earnings = await fetch_json(session, url, params)
        
        if earnings:
            symbols = {e.get('symbol') for e in earnings if e.get('symbol')}
            discovered.update(symbols)
        
        await asyncio.sleep(0.2)
    
    log.info(f"  Found {len(discovered)} stocks with upcoming earnings")
    return discovered

# ==================== DATABASE OPERATIONS ====================

def get_existing_symbols() -> Set[str]:
    """Get symbols already in database"""
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT DISTINCT symbol FROM stock_consensus", conn)
    conn.close()
    return set(df['symbol'].tolist())

def add_new_symbols(new_symbols: List[str]):
    """Add new symbols to database (minimal row for update scripts to fill)"""
    if not new_symbols:
        log.info("No new symbols to add")
        return
    
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()
    
    # Insert minimal placeholder rows
    for symbol in new_symbols:
        try:
            cur.execute("""
                INSERT OR IGNORE INTO stock_consensus (symbol, last_updated)
                VALUES (?, NULL)
            """, (symbol,))
        except Exception as e:
            log.warning(f"Could not add {symbol}: {e}")
    
    conn.commit()
    conn.close()
    log.info(f"✓ Added {len(new_symbols)} new symbols to database")

# ==================== MAIN ====================

async def main():
    log.info("="*60)
    log.info("🔍 COVERAGE EXPANSION - Discovering New Stocks")
    log.info("="*60)
    
    # Get current coverage
    existing = get_existing_symbols()
    log.info(f"Current database has {len(existing)} stocks\n")
    
    # Discover new stocks
    all_discovered = set()
    
    async with aiohttp.ClientSession() as session:
        # Run all discovery methods
        tasks = [
            discover_stocks_from_screener(session),
            discover_from_indexes(session),
            discover_from_analyst_upgrades(session),
            discover_from_earnings(session)
        ]
        
        results = await asyncio.gather(*tasks)
        
        for discovered_set in results:
            all_discovered.update(discovered_set)
    
    log.info(f"\n📊 Discovery Results:")
    log.info(f"  Total discovered: {len(all_discovered)}")
    log.info(f"  Already in database: {len(all_discovered & existing)}")
    
    # Find new symbols
    new_symbols = all_discovered - existing
    log.info(f"  NEW stocks to add: {len(new_symbols)}")
    
    if new_symbols:
        log.info(f"\nSample of new stocks: {list(new_symbols)[:20]}")
        
        # Add to database
        response = input(f"\nAdd {len(new_symbols)} new stocks to database? (y/n): ")
        if response.lower() == 'y':
            add_new_symbols(list(new_symbols))
            log.info("\n✅ Coverage expansion complete!")
            log.info("Next: Run update_analyst_OPTIMIZED.py to fetch data for new stocks")
        else:
            log.info("Cancelled - no stocks added")
    else:
        log.info("\n✅ No new stocks found - your coverage is already comprehensive!")
    
    log.info("\n" + "="*60)

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
DCF Data Collector
Fetches discounted cash flow valuations from FMP for all symbols.
Stores in backtest.db → dcf_valuations table.

Usage:
    python collect_dcf_data.py              # Collect all
    python collect_dcf_data.py --symbols AAPL,MSFT  # Specific symbols
"""

import asyncio
import aiohttp
import sqlite3
import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FMP_API_KEY, DATABASE_NAME, BACKTEST_DB,
    HISTORICAL_BATCH_SIZE, HISTORICAL_MAX_CONCURRENT,
    HISTORICAL_CALLS_PER_MINUTE, HISTORICAL_REQUEST_TIMEOUT,
    FMP_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

DCF_ENDPOINT = f'{FMP_BASE_URL}/stable/discounted-cash-flow'
MAX_RETRIES = 3


def setup_dcf_table():
    conn = sqlite3.connect(BACKTEST_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dcf_valuations (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            dcf_value REAL,
            stock_price REAL,
            dcf_upside_pct REAL,
            collected_at TEXT,
            PRIMARY KEY (symbol)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dcf_symbol ON dcf_valuations(symbol)")
    conn.commit()
    conn.close()


class RateLimiter:
    def __init__(self, calls_per_minute):
        self.min_interval = 60.0 / calls_per_minute
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()


async def fetch_dcf(session, symbol, rate_limiter, api_key, retry=0):
    await rate_limiter.acquire()
    try:
        url = f'{DCF_ENDPOINT}?symbol={symbol}&apikey={api_key}'
        async with session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0]
            elif resp.status == 429 and retry < MAX_RETRIES:
                await asyncio.sleep(2 ** retry)
                return await fetch_dcf(session, symbol, rate_limiter, api_key, retry + 1)
    except asyncio.TimeoutError:
        if retry < MAX_RETRIES:
            await asyncio.sleep(2 ** retry)
            return await fetch_dcf(session, symbol, rate_limiter, api_key, retry + 1)
    except Exception as e:
        if retry < MAX_RETRIES:
            return await fetch_dcf(session, symbol, rate_limiter, api_key, retry + 1)
    return None


async def collect_dcf_batch(session, symbols, rate_limiter, api_key, semaphore):
    async def bounded_fetch(sym):
        async with semaphore:
            return sym, await fetch_dcf(session, sym, rate_limiter, api_key)
    tasks = [bounded_fetch(s) for s in symbols]
    return await asyncio.gather(*tasks)


def save_dcf_batch(results):
    conn = sqlite3.connect(BACKTEST_DB)
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for symbol, data in results:
        if data is None:
            continue
        dcf_val = data.get('dcf')
        price = data.get('Stock Price') or data.get('stockPrice')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        if dcf_val is not None and price is not None and price > 0:
            upside = (dcf_val - price) / price * 100
        else:
            upside = None
        rows.append((symbol, date, dcf_val, price, upside, now))

    if rows:
        conn.executemany("""
            INSERT OR REPLACE INTO dcf_valuations
            (symbol, date, dcf_value, stock_price, dcf_upside_pct, collected_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
        conn.commit()
    conn.close()
    return len(rows)


def get_symbols(specific=None):
    if specific:
        return [s.strip().upper() for s in specific.split(',')]
    conn = sqlite3.connect(DATABASE_NAME)
    syms = [r[0] for r in conn.execute("SELECT symbol FROM stock_consensus ORDER BY symbol").fetchall()]
    conn.close()
    return syms


def get_completed():
    conn = sqlite3.connect(BACKTEST_DB)
    try:
        rows = conn.execute("SELECT symbol FROM dcf_valuations").fetchall()
        conn.close()
        return {r[0] for r in rows}
    except Exception:
        conn.close()
        return set()


async def main():
    parser = argparse.ArgumentParser(description='Collect DCF valuations from FMP')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--force', action='store_true', help='Re-collect even if already done')
    args = parser.parse_args()

    setup_dcf_table()

    symbols = get_symbols(args.symbols)
    if not args.force:
        completed = get_completed()
        symbols = [s for s in symbols if s not in completed]

    if not symbols:
        log.info("All symbols already have DCF data. Use --force to re-collect.")
        return

    log.info("=" * 60)
    log.info("DCF Data Collection")
    log.info("=" * 60)
    log.info(f"Symbols to collect: {len(symbols)}")
    log.info(f"Rate limit: {HISTORICAL_CALLS_PER_MINUTE}/min")

    rate_limiter = RateLimiter(HISTORICAL_CALLS_PER_MINUTE)
    semaphore = asyncio.Semaphore(HISTORICAL_MAX_CONCURRENT)
    total_saved = 0
    start = datetime.now()

    timeout = aiohttp.ClientTimeout(total=HISTORICAL_REQUEST_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i in range(0, len(symbols), HISTORICAL_BATCH_SIZE):
            batch = symbols[i:i + HISTORICAL_BATCH_SIZE]
            results = await collect_dcf_batch(session, batch, rate_limiter, FMP_API_KEY, semaphore)
            saved = save_dcf_batch(results)
            total_saved += saved
            done = min(i + HISTORICAL_BATCH_SIZE, len(symbols))
            log.info(f"  Progress: {done}/{len(symbols)} ({total_saved:,} saved)")

    elapsed = datetime.now() - start
    log.info(f"\nDCF collection complete! {total_saved:,} valuations saved in {elapsed}")

    # Quick summary
    conn = sqlite3.connect(BACKTEST_DB)
    total = conn.execute("SELECT COUNT(*) FROM dcf_valuations").fetchone()[0]
    avg_upside = conn.execute("SELECT AVG(dcf_upside_pct) FROM dcf_valuations WHERE dcf_upside_pct IS NOT NULL").fetchone()[0]
    undervalued = conn.execute("SELECT COUNT(*) FROM dcf_valuations WHERE dcf_upside_pct > 0").fetchone()[0]
    conn.close()
    log.info(f"Total DCF records: {total:,}")
    log.info(f"Average DCF upside: {avg_upside:+.1f}%")
    log.info(f"Undervalued (DCF > price): {undervalued:,}/{total:,}")


if __name__ == "__main__":
    asyncio.run(main())

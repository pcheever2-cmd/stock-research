#!/usr/bin/env python3
"""
Backfill recent historical prices in backtest.db.
Detects gaps per symbol and fetches only missing dates from FMP.

Usage:
    python scripts/backfill_recent_prices.py                # Auto-detect gaps, fill them
    python scripts/backfill_recent_prices.py --lookback 7   # Force fetch last 7 days
    python scripts/backfill_recent_prices.py --dry-run      # Show what would be fetched
    python scripts/backfill_recent_prices.py --symbols AAPL,MSFT  # Specific symbols only
"""
import asyncio
import argparse
import sqlite3
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    FMP_API_KEY, BACKTEST_DB, DATABASE_NAME,
    HISTORICAL_CALLS_PER_MINUTE, HISTORICAL_MAX_CONCURRENT,
    HISTORICAL_REQUEST_TIMEOUT, HISTORICAL_ENDPOINTS,
)
from collectors.collect_historical_data import AsyncFMPFetcher, save_prices_batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

MAX_LOOKBACK_DAYS = 400  # For symbols with no history at all


def get_symbol_gaps(db_path: str, symbols: list[str] = None) -> Dict[str, Optional[str]]:
    """Get the latest date per symbol in historical_prices.
    Returns dict of symbol -> last_date (YYYY-MM-DD string) or None if no data.
    """
    conn = sqlite3.connect(db_path)

    if symbols:
        placeholders = ','.join('?' for _ in symbols)
        query = f"SELECT symbol, MAX(date) FROM historical_prices WHERE symbol IN ({placeholders}) GROUP BY symbol"
        rows = conn.execute(query, symbols).fetchall()
    else:
        rows = conn.execute("SELECT symbol, MAX(date) FROM historical_prices GROUP BY symbol").fetchall()

    gaps = {row[0]: row[1] for row in rows}
    conn.close()
    return gaps


def get_all_symbols(db_path: str) -> list[str]:
    """Get all symbols from stock_consensus table."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT symbol FROM stock_consensus WHERE current_price > 0").fetchall()
    conn.close()
    return [r[0] for r in rows]


async def fetch_prices_for_symbol(fetcher: AsyncFMPFetcher, symbol: str, from_date: str, to_date: str) -> list:
    """Fetch historical prices for a single symbol from FMP."""
    data = await fetcher.fetch(
        HISTORICAL_ENDPOINTS['prices'],
        {'symbol': symbol, 'from': from_date, 'to': to_date}
    )
    if not data:
        return []
    # FMP returns dict with 'historical' key or a list directly
    if isinstance(data, dict):
        records = data.get('historical', [])
    elif isinstance(data, list):
        records = data
    else:
        return []
    # Filter valid records
    return [r for r in records if r.get('date') and r.get('close', 0) > 0]


async def backfill(args):
    """Main backfill logic."""
    if not FMP_API_KEY:
        log.error("FMP_API_KEY not set")
        sys.exit(1)

    today = datetime.now().strftime('%Y-%m-%d')

    # Get symbols to process
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = get_all_symbols(DATABASE_NAME)
    log.info(f"Processing {len(symbols)} symbols")

    # Get current gaps
    gaps = get_symbol_gaps(BACKTEST_DB, symbols)

    # Determine what to fetch per symbol
    to_fetch = []  # (symbol, from_date, to_date)
    for sym in symbols:
        last_date = gaps.get(sym)
        if args.lookback:
            from_date = (datetime.now() - timedelta(days=args.lookback)).strftime('%Y-%m-%d')
        elif last_date:
            from_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            # No data at all — fetch MAX_LOOKBACK_DAYS
            from_date = (datetime.now() - timedelta(days=MAX_LOOKBACK_DAYS)).strftime('%Y-%m-%d')

        if from_date <= today:
            to_fetch.append((sym, from_date, today))

    log.info(f"Need to fetch prices for {len(to_fetch)} symbols")

    if not to_fetch:
        log.info("All symbols are up to date!")
        return

    # Show sample of what we'll fetch
    sample = to_fetch[:5]
    for sym, fd, td in sample:
        log.info(f"  {sym}: {fd} → {td}")
    if len(to_fetch) > 5:
        log.info(f"  ... and {len(to_fetch) - 5} more")

    if args.dry_run:
        log.info("DRY RUN — no data will be fetched or inserted")
        return

    # Count rows before
    conn = sqlite3.connect(BACKTEST_DB)
    rows_before = conn.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0]
    conn.close()

    # Fetch in batches
    batch_size = 50
    total_inserted = 0
    errors = 0

    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        for i in range(0, len(to_fetch), batch_size):
            batch = to_fetch[i:i + batch_size]
            tasks = [fetch_prices_for_symbol(fetcher, sym, fd, td) for sym, fd, td in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            symbols_data = {}
            for (sym, _, _), result in zip(batch, results):
                if isinstance(result, Exception):
                    log.warning(f"Error fetching {sym}: {result}")
                    errors += 1
                elif result:
                    symbols_data[sym] = result

            if symbols_data:
                inserted = save_prices_batch(symbols_data)
                total_inserted += inserted

            progress = min(i + batch_size, len(to_fetch))
            log.info(f"Progress: {progress}/{len(to_fetch)} symbols ({total_inserted} rows inserted, {errors} errors)")

    # Count rows after
    conn = sqlite3.connect(BACKTEST_DB)
    rows_after = conn.execute("SELECT COUNT(*) FROM historical_prices").fetchone()[0]
    max_date = conn.execute("SELECT MAX(date) FROM historical_prices").fetchone()[0]
    conn.close()

    log.info(f"Done! Rows: {rows_before} → {rows_after} (+{rows_after - rows_before})")
    log.info(f"Latest date in historical_prices: {max_date}")
    log.info(f"API calls: {fetcher.total_calls}, Errors: {errors}")


def main():
    parser = argparse.ArgumentParser(description='Backfill recent historical prices')
    parser.add_argument('--lookback', type=int, default=None,
                        help='Force fetch last N days (default: auto-detect gap)')
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated symbols (default: all from stock_consensus)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be fetched without actually fetching')
    args = parser.parse_args()

    asyncio.run(backfill(args))


if __name__ == '__main__':
    main()

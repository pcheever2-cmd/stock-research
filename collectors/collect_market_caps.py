#!/usr/bin/env python3
"""
Re-pull clean daily market caps from FMP into backtest.db.

WHY: historical_key_metrics.market_cap is systematically corrupt — foreign ADRs
stored ~100-1000x too large (SKM $20.5T vs real $14.5B) plus int64 sentinels
(~9.2e18) across >1,100 symbols. FMP's market-cap endpoint returns the CORRECT
values, so the fix is a clean re-pull (the corruption was a storage/parsing bug
on our side, not FMP's data). This also gives delisted names point-in-time market
caps for the survivorship correction.

Writes a fresh `historical_market_caps` (symbol, date, market_cap) table. The
validation harness joins from this instead of the corrupt field.

Endpoint note: /stable/historical-market-capitalization returns only ~63 days by
default, so we window by date (from/to). Daily series; delisted names covered.

Usage:
    python collectors/collect_market_caps.py                 # full universe (live + delisted)
    python collectors/collect_market_caps.py --symbols AAPL,MSFT
    python collectors/collect_market_caps.py --status
"""
import asyncio, argparse, sqlite3, sys, logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FMP_API_KEY, FMP_BASE_URL, BACKTEST_DB, DATABASE_NAME
from collectors.collect_historical_data import AsyncFMPFetcher, ProgressTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

MCAP_URL = f'{FMP_BASE_URL}/stable/historical-market-capitalization'
WINDOWS = [('1995-01-01', '2002-12-31'), ('2003-01-01', '2010-12-31'),
           ('2011-01-01', '2018-12-31'), ('2019-01-01', '2026-12-31')]
DATA_TYPE = 'market_caps'
REAL_EXIT_REASONS = ('acquired', 'merged', 'went_private', 'bankrupt', 'distress_delist', 'liquidated')


def ensure_table(db):
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_market_caps (
            symbol TEXT NOT NULL, date TEXT NOT NULL, market_cap REAL,
            PRIMARY KEY (symbol, date)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hmc_symbol ON historical_market_caps(symbol)")
    # progress table (shared shape with collect_historical_data)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS collection_progress (
            symbol TEXT, data_type TEXT, status TEXT, rows_collected INTEGER,
            min_date TEXT, max_date TEXT, collected_at TEXT, error_message TEXT,
            PRIMARY KEY (symbol, data_type))""")
    conn.commit(); conn.close()


def get_universe(specific=None):
    if specific:
        return [s.strip().upper() for s in specific.split(',')]
    syms = set()
    try:
        c = sqlite3.connect(DATABASE_NAME)
        syms |= {r[0] for r in c.execute("SELECT DISTINCT symbol FROM stock_consensus").fetchall()}
        c.close()
    except Exception as e:
        log.warning(f"live universe: {e}")
    c = sqlite3.connect(BACKTEST_DB)
    ph = ','.join('?' * len(REAL_EXIT_REASONS))
    syms |= {r[0] for r in c.execute(
        f"SELECT symbol FROM delisted_companies WHERE reason IN ({ph})", REAL_EXIT_REASONS).fetchall()}
    c.close()
    return sorted(syms)


async def collect_for_symbol(fetcher, symbol, db_lock, db):
    rows = {}
    for frm, to in WINDOWS:
        data = await fetcher.fetch(MCAP_URL, {'symbol': symbol, 'from': frm, 'to': to})
        if isinstance(data, list):
            for r in data:
                mc = r.get('marketCap')
                if r.get('date') and isinstance(mc, (int, float)) and 0 < mc < 1e13:
                    rows[r['date']] = mc
    async with db_lock:
        conn = sqlite3.connect(db, timeout=60)
        conn.executemany(
            "INSERT OR REPLACE INTO historical_market_caps (symbol, date, market_cap) VALUES (?,?,?)",
            [(symbol, d, mc) for d, mc in rows.items()])
        conn.commit(); conn.close()
        ProgressTracker(db).mark_completed(symbol, DATA_TYPE, len(rows),
                                           min(rows) if rows else None, max(rows) if rows else None)
    return len(rows)


async def main_async(args):
    ensure_table(BACKTEST_DB)
    tracker = ProgressTracker(BACKTEST_DB)
    if args.status:
        print(tracker.get_summary().get(DATA_TYPE, 'none yet')); return
    universe = get_universe(args.symbols)
    done = tracker.get_completed(DATA_TYPE) if not args.symbols else set()
    todo = [s for s in universe if s not in done]
    print(f"Universe {len(universe)} | already done {len(done)} | to fetch {len(todo)}")
    db_lock = asyncio.Lock()
    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        sem = asyncio.Semaphore(40)
        async def worker(sym):
            async with sem:
                try:
                    n = await collect_for_symbol(fetcher, sym, db_lock, BACKTEST_DB)
                    return sym, n
                except Exception as e:
                    tracker.mark_failed(sym, DATA_TYPE, str(e)); return sym, -1
        completed = 0
        for i in range(0, len(todo), 200):
            batch = todo[i:i+200]
            for sym, n in await asyncio.gather(*[worker(s) for s in batch]):
                completed += 1
            print(f"  {completed}/{len(todo)} symbols ({fetcher.total_calls} calls)", flush=True)
    # report
    conn = sqlite3.connect(BACKTEST_DB)
    tot, nsym, mx = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(market_cap) FROM historical_market_caps").fetchone()
    conn.close()
    print(f"\n✓ historical_market_caps: {tot:,} rows, {nsym:,} symbols, max ${mx/1e9:.0f}B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=str)
    ap.add_argument('--status', action='store_true')
    asyncio.run(main_async(ap.parse_args()))


if __name__ == '__main__':
    main()

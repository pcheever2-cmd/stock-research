#!/usr/bin/env python3
"""
Forward-collection of analyst consensus estimates into an append-only LEDGER.

WHY: estimate *revisions* are among the most robust forward-return predictors, but FMP
serves only the CURRENT forward consensus (analyst-estimates) — there is no point-in-time
history to backfill (confirmed: analyst_estimates_snapshot holds a single 2026-02-02 pull).
The only way to get a revision time series is to snapshot the consensus on a schedule and
accumulate our own dated ledger. This seeds that ledger and is meant to run daily via CI.

Unlike `analyst_estimates_snapshot` (INSERT OR REPLACE on (symbol, fiscal_year) — overwrites,
no history), `estimate_ledger` keys on (symbol, fiscal_year, snapshot_date) so each run adds a
new dated layer. After N runs, `eps_avg` for a (symbol, fiscal_year) across snapshot_dates IS
the revision series. No historical value this pass — it's groundwork for future forward-tracking.

NOTE (CI wiring): add a daily invocation to the data pipeline. ~6k symbols × ~1 row each ≈ a
few thousand rows/day, tiny growth. Confirm the exact daily-pipeline hook before wiring.

Usage:
    python collectors/snapshot_estimates.py            # seed one dated layer (today)
    python collectors/snapshot_estimates.py --status
    python collectors/snapshot_estimates.py --symbols AAPL,MSFT
"""
import asyncio, argparse, sqlite3, sys, logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FMP_API_KEY, BACKTEST_DB
from collect_historical_data import AsyncFMPFetcher, _to_float
from collect_analyst_data import fetch_analyst_estimates, get_symbols

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def ensure_table(db):
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS estimate_ledger (
            symbol TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            snapshot_date TEXT NOT NULL,     -- the date we captured this consensus
            eps_avg REAL,
            revenue_avg REAL,
            ebitda_avg REAL,
            num_analysts_eps INTEGER,
            PRIMARY KEY (symbol, fiscal_year, snapshot_date)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_el_symbol ON estimate_ledger(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_el_snap ON estimate_ledger(snapshot_date)")
    conn.commit(); conn.close()


def _fiscal_year(r):
    fy = r.get('calendarYear') or r.get('calendar_year')
    if fy is None:
        d = r.get('date', '')
        fy = d[:4] if len(d) >= 4 else None
    try:
        return int(fy)
    except (TypeError, ValueError):
        return None


async def run(symbols, snapshot_date):
    ensure_table(BACKTEST_DB)
    rows = []
    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        sem = asyncio.Semaphore(40)
        async def worker(sym):
            async with sem:
                try:
                    data = await fetch_analyst_estimates(fetcher, sym)
                except Exception as e:
                    log.warning(f"{sym}: {e}"); return []
                out = []
                for r in (data or []):
                    fy = _fiscal_year(r)
                    if fy is None:
                        continue
                    out.append((sym, fy, snapshot_date,
                                _to_float(r.get('epsAvg') or r.get('eps_avg')),
                                _to_float(r.get('revenueAvg') or r.get('revenue_avg')),
                                _to_float(r.get('ebitdaAvg') or r.get('ebitda_avg')),
                                r.get('numAnalystsEps') or r.get('numberAnalystEstimatedEps')))
                return out
        for i in range(0, len(symbols), 200):
            batch = symbols[i:i+200]
            for res in await asyncio.gather(*[worker(s) for s in batch]):
                rows.extend(res)
            log.info(f"  {min(i+200, len(symbols))}/{len(symbols)} symbols | {len(rows):,} rows ({fetcher.total_calls} calls)")
    conn = sqlite3.connect(BACKTEST_DB, timeout=60)
    conn.executemany(
        """INSERT OR REPLACE INTO estimate_ledger
           (symbol, fiscal_year, snapshot_date, eps_avg, revenue_avg, ebitda_avg, num_analysts_eps)
           VALUES (?,?,?,?,?,?,?)""", rows)
    conn.commit(); conn.close()
    log.info(f"✓ estimate_ledger += {len(rows):,} rows for snapshot {snapshot_date}")
    report(BACKTEST_DB)


def report(db):
    conn = sqlite3.connect(db)
    tot, nsym, nsnap, smin, smax = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), COUNT(DISTINCT snapshot_date), MIN(snapshot_date), MAX(snapshot_date) FROM estimate_ledger").fetchone()
    print(f"\nestimate_ledger: {tot:,} rows | {nsym:,} symbols | {nsnap} snapshot date(s) {smin}..{smax}")
    if nsnap and nsnap < 2:
        print("  (only one snapshot so far — revision series accrues once this runs daily)")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbols', type=str)
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if not FMP_API_KEY:
        log.error("FMP_API_KEY not set"); sys.exit(1)
    if args.status:
        ensure_table(BACKTEST_DB); report(BACKTEST_DB); return
    snapshot_date = datetime.now(timezone.utc).date().isoformat()
    symbols = get_symbols(args.symbols)
    log.info(f"Snapshotting consensus for {len(symbols):,} symbols @ {snapshot_date}")
    asyncio.run(run(symbols, snapshot_date))


if __name__ == '__main__':
    main()

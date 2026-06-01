#!/usr/bin/env python3
"""
Backfill historical earnings surprises (announcement-dated) into backtest.db.

WHY: the big-winners pivot needs a backtestable CATALYST. Estimate *revisions* are
not backfillable on FMP (current-consensus snapshot only), but earnings SURPRISES
are — and they drive PEAD. We use /stable/earnings-calendar (not earnings-surprises-bulk)
for two reasons the data probe established:
  1. Its `date` is the ANNOUNCEMENT date (e.g. AAPL's Dec quarter reports Jan 29),
     so the surprise is look-ahead-safe natively — no period-end leakage.
  2. It carries REVENUE actual/estimated too, not just EPS (the bulk endpoint is EPS-only).

Endpoint cap: earnings-calendar returns at most ~4000 rows, date-DESCending, so a wide
window silently truncates to the most recent days. We therefore window WEEKLY and, if a
window still hits the cap, recursively split it down to single days.

Writes `earnings_surprises(symbol, date, eps_actual, eps_estimated, eps_surprise_pct,
revenue_actual, revenue_estimated, rev_surprise_pct, collected_at)`, PK (symbol, date).
`date` = announcement date. Filtered to the backtest universe (historical_prices symbols,
which already includes delisted names) so the table joins cleanly and stays survivorship-safe.

Usage:
    python collectors/collect_earnings_surprises.py                 # full backfill 2012->today
    python collectors/collect_earnings_surprises.py --from 2020-01-01 --to 2020-12-31
    python collectors/collect_earnings_surprises.py --status
"""
import argparse, json, sqlite3, sys, time, logging
from datetime import datetime, timezone, date, timedelta
from pathlib import Path
import urllib.request, urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FMP_API_KEY, FMP_BASE_URL, BACKTEST_DB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

CAL_URL = f'{FMP_BASE_URL}/stable/earnings-calendar'
ROW_CAP = 4000              # FMP hard cap per response; hitting it means truncation
START_DEFAULT = '2012-01-01'
PACE_SECONDS = 0.05         # be polite; ~20 req/s, far under the 3000/min limit


def ensure_table(db):
    conn = sqlite3.connect(db)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS earnings_surprises (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,              -- announcement date (look-ahead safe)
            eps_actual REAL,
            eps_estimated REAL,
            eps_surprise_pct REAL,
            revenue_actual REAL,
            revenue_estimated REAL,
            rev_surprise_pct REAL,
            collected_at TEXT,
            PRIMARY KEY (symbol, date)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_es_symbol ON earnings_surprises(symbol)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_es_date ON earnings_surprises(date)")
    conn.commit(); conn.close()


def load_universe(db):
    """Backtest universe = every symbol with price history (already includes delisted)."""
    conn = sqlite3.connect(db)
    syms = {r[0] for r in conn.execute("SELECT DISTINCT symbol FROM historical_prices").fetchall()}
    conn.close()
    return syms


def _surprise_pct(actual, est):
    if actual is None or est is None:
        return None
    try:
        if est == 0:
            return None
        return (actual - est) / abs(est)
    except (TypeError, ValueError):
        return None


def fetch_window(frm, to, retries=4):
    """GET earnings-calendar for [frm, to]; returns parsed list or raises after retries."""
    url = f'{CAL_URL}?from={frm}&to={to}&apikey={FMP_API_KEY}'
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                body = r.read().decode('utf-8', 'replace')
            data = json.loads(body)
            return data if isinstance(data, list) else []
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = 2 ** attempt
                log.warning(f"  HTTP {e.code} on {frm}..{to}, backoff {wait}s")
                time.sleep(wait); continue
            raise
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt); continue
            raise
    return []


def collect_window(frm, to, universe, rows_out):
    """Fetch [frm,to]; on cap-hit recursively split (keeps the global cap from truncating us)."""
    time.sleep(PACE_SECONDS)
    data = fetch_window(frm, to)
    if len(data) >= ROW_CAP and frm != to:
        # truncated -> split in half by days and recurse
        d0 = datetime.strptime(frm, '%Y-%m-%d').date()
        d1 = datetime.strptime(to, '%Y-%m-%d').date()
        mid = d0 + (d1 - d0) // 2
        collect_window(frm, mid.isoformat(), universe, rows_out)
        collect_window((mid + timedelta(days=1)).isoformat(), to, universe, rows_out)
        return
    if len(data) >= ROW_CAP:
        log.warning(f"  single day {frm} hit the {ROW_CAP}-row cap — possible global truncation")
    now = datetime.now(timezone.utc).isoformat()
    for r in data:
        sym = r.get('symbol')
        if sym not in universe:
            continue
        ea, ee = r.get('epsActual'), r.get('epsEstimated')
        if ea is None or ee is None:      # need both to define a surprise / beat
            continue
        ra, re = r.get('revenueActual'), r.get('revenueEstimated')
        rows_out[(sym, r['date'])] = (
            sym, r['date'], ea, ee, _surprise_pct(ea, ee),
            ra, re, _surprise_pct(ra, re), now)


def weekly_windows(start, end):
    d = start
    while d <= end:
        wend = min(d + timedelta(days=6), end)
        yield d.isoformat(), wend.isoformat()
        d = wend + timedelta(days=1)


def run(frm, to):
    ensure_table(BACKTEST_DB)
    universe = load_universe(BACKTEST_DB)
    log.info(f"Backtest universe: {len(universe):,} symbols | window {frm}..{to}")
    start = datetime.strptime(frm, '%Y-%m-%d').date()
    end = datetime.strptime(to, '%Y-%m-%d').date()
    windows = list(weekly_windows(start, end))
    rows = {}
    calls = 0
    for i, (w0, w1) in enumerate(windows):
        collect_window(w0, w1, universe, rows)
        calls += 1
        if (i + 1) % 50 == 0:
            log.info(f"  {i+1}/{len(windows)} weeks | {len(rows):,} US rows so far")
    conn = sqlite3.connect(BACKTEST_DB, timeout=60)
    conn.executemany(
        """INSERT OR REPLACE INTO earnings_surprises
           (symbol, date, eps_actual, eps_estimated, eps_surprise_pct,
            revenue_actual, revenue_estimated, rev_surprise_pct, collected_at)
           VALUES (?,?,?,?,?,?,?,?,?)""", list(rows.values()))
    conn.commit(); conn.close()
    log.info(f"✓ wrote {len(rows):,} rows over {len(windows)} weekly windows")
    report(BACKTEST_DB)


def report(db):
    conn = sqlite3.connect(db)
    tot, nsym, dmin, dmax = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(date), MAX(date) FROM earnings_surprises").fetchone()
    print(f"\nearnings_surprises: {tot:,} rows | {nsym:,} symbols | {dmin} → {dmax}")
    print("rows by year:")
    for y, n, s in conn.execute(
        "SELECT substr(date,1,4) y, COUNT(*), COUNT(DISTINCT symbol) FROM earnings_surprises GROUP BY y ORDER BY y"):
        print(f"  {y}: {n:>7,} rows  {s:>5,} symbols")
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='frm', default=START_DEFAULT)
    ap.add_argument('--to', dest='to', default=date.today().isoformat())
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if not FMP_API_KEY:
        log.error("FMP_API_KEY not set"); sys.exit(1)
    if args.status:
        ensure_table(BACKTEST_DB); report(BACKTEST_DB); return
    run(args.frm, args.to)


if __name__ == '__main__':
    main()

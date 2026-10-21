#!/usr/bin/env python3
"""
Grow the scored universe automatically: add new US common stocks (IPOs,
spin-offs, renamed tickers) to stock_consensus and collect their history.

Why: the daily scoring path only refreshes symbols ALREADY in stock_consensus
(update_analyst_OPTIMIZED: universe = tracked ∩ profile-bulk), and the only
script that added symbols ("scripts/Expand coverage.py") was manual. SpaceX
(SPCX, IPO 2026-06-12, ~$1.9T) went unscored for four months as a result.

Flow (daily, before run_pipeline):
  1. DISCOVER: FMP company-screener per US exchange — actively trading,
     not ETF/fund, price > $1, market cap >= --min-mcap.
  2. FILTER: drop tracked symbols; non-common securities (the IPO collector's
     noise rules + coupon/SPAC name rules here); second securities of
     companies we already cover (same normalized name as a tracked row whose
     ticker still trades); and duplicate listings of one company in the same
     batch (keep the largest).
  3. VERIFY per candidate via FMP profile: US exchange, sector + industry,
     description >= 50 chars, not a SPAC, market cap consistent with traded
     volume (FMP sometimes reports a nonsense cap), and NOT a reused ticker
     (backtest.db holds a dead company's history under the same symbol —
     merging would splice two companies' charts; skipped for manual handling).
  4. RENAMES: a candidate sharing a normalized name with a tracked row whose
     old ticker FMP confirms is no longer actively trading is a ticker change.
     The old row is RENAMED in place (keeps its score/fields until the next
     scoring run refreshes them, so the company never drops off the site) and
     the old ticker is recorded in retired_symbols so compute_compass_scores
     never re-inserts it from leftover backtest rows.
  5. ADD + COLLECT: insert seeded rows (name/sector/industry/description/
     market cap/exchange from the profile) and run collect_historical_data
     --force for new and renamed symbols (5y prices + 20 quarters).
     The next compass-scores run scores them once they meet its data bar.

Bounded: at most --max-add symbols per run (largest market cap first).

Usage:
    python collectors/expand_universe.py --dry-run
    python collectors/expand_universe.py [--min-mcap 300000000] [--max-add 150]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FMP_API_KEY, FMP_BASE_URL, DATABASE_NAME, BACKTEST_DB  # noqa: E402
from collect_ipos import (  # noqa: E402
    NOISE_NAME, NOISE_SYM, NOISE_SYM_BARE, SPAC_DESC, MIN_DESC_LEN, is_us,
)

US_EXCHANGES = ('NASDAQ', 'NYSE', 'AMEX')
MIN_SCREENER_ROWS = {'NASDAQ': 500, 'NYSE': 500}  # sanity floor: below this the call failed
DEFAULT_MIN_MCAP = 300_000_000
DEFAULT_MAX_ADD = 150  # ~5 FMP calls per add; the cap guards against a screener glitch flooding the universe
STALE_DAYS = 21
# Reused ticker: our stored history under this symbol ends this long before
# the current listing's IPO date — a different, dead company.
REUSED_TICKER_GAP_DAYS = 60

# Coupon-bearing securities (notes/preferreds) whose names FMP truncates before
# the "Notes"/"Subordinated" word the IPO noise filter keys on: "KKR & Co. Inc.
# 6.875% Subordina", "Saratoga Investment Corp 7.50%", "Corebridge Financial
# Inc. 6.375", "DTE Energy Company 2025 Series".
COUPON_NAME = re.compile(r'\d+(\.\d+)?\s*%|\b(19|20)\d{2} Series\b|\b\d{1,2}\.\d{2,3}\s*$')
# SPAC shells whose names run "acquisition" into another word ("KRAKacquisition").
SPAC_NAME = re.compile(r'acquisition\s*corp', re.I)
SPAC_DESC_EXTRA = re.compile(r'business combination', re.I)
# FMP sometimes multiplies a depositary-interest price by the parent's share
# count (Amaero CDI showed $216B on $2M/day traded). Real equities turn over
# far more than 0.01% of market cap per day; below that the cap is bad data.
MIN_DAILY_TURNOVER = 1e-4
# $25-par baby bonds / preferreds listed under 5-letter tickers with a generic
# company name (NSLRL "Neostellar Capital Corp." at $25.06, 52-wk 24.72–26.50).
# A real common stock essentially never holds a <12% 52-week range near $25.
PAR_PRICE_BAND = (20.0, 30.0)
PAR_MAX_RANGE_FRAC = 0.12


def looks_like_par_security(sym: str, p: dict) -> bool:
    price = p.get('price') or 0
    if len(sym) < 5 or not (PAR_PRICE_BAND[0] <= price <= PAR_PRICE_BAND[1]):
        return False
    try:
        lo, hi = (float(x) for x in str(p.get('range') or '').split('-', 1))
    except ValueError:
        return False
    return hi > 0 and (hi - lo) / price < PAR_MAX_RANGE_FRAC
NAME_SUFFIXES = re.compile(
    r'\b(the|inc|incorporated|corp|corporation|co|company|ltd|limited|plc|'
    r'holdings?|group|sa|s\.?p\.?a|nv|ag|se|lp|llc)\b')


def norm_name(name: str | None) -> str:
    n = (name or '').lower().replace('&', ' and ')
    n = re.sub(r'[^a-z0-9 ]+', ' ', n)
    n = NAME_SUFFIXES.sub(' ', n)
    return ' '.join(n.split())


def fetch(path: str, **params):
    params['apikey'] = FMP_API_KEY
    qs = '&'.join(f'{k}={v}' for k, v in params.items())
    req = urllib.request.Request(f'{FMP_BASE_URL}/stable/{path}?{qs}',
                                 headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def discover(min_mcap: float) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for ex in US_EXCHANGES:
        rows = fetch('company-screener', exchange=ex, isActivelyTrading='true',
                     isEtf='false', isFund='false', marketCapMoreThan=int(min_mcap),
                     priceMoreThan=1, limit=10000)
        if not isinstance(rows, list) or len(rows) < MIN_SCREENER_ROWS.get(ex, 0):
            sys.exit(f"ERROR: screener {ex} returned {len(rows) if isinstance(rows, list) else rows!r} "
                     f"— aborting rather than acting on a partial universe")
        for r in rows:
            sym = (r.get('symbol') or '').upper()
            if sym and '.' not in sym:
                found[sym] = r
        print(f"  screener {ex}: {len(rows)} rows")
        time.sleep(0.3)
    return found


def load_tracked(db: str) -> dict[str, dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT symbol, company_name, last_updated FROM stock_consensus").fetchall()
    conn.close()
    return {r['symbol'].upper(): dict(r) for r in rows if r['symbol']}


def is_stale_or_unrefreshed(last_updated: str | None, now: datetime) -> bool:
    """NULL counts too: in production it means 'never gets a full analyst refresh'
    (no price target), not 'brand-new placeholder' — the FMP liveness check is
    the real guard before any rename."""
    if not last_updated:
        return True
    try:
        ts = datetime.fromisoformat(str(last_updated)[:19])
    except ValueError:
        return False
    return now - ts > timedelta(days=STALE_DAYS)


def ticker_is_live(symbol: str) -> bool:
    """Ask FMP whether an old ticker still trades. Anything ambiguous (fetch
    error, non-list payload) counts as LIVE so nothing is ever renamed on doubt.
    An empty list means FMP no longer resolves the ticker at all → not live."""
    try:
        resp = fetch('profile', symbol=symbol)
    except Exception:
        return True
    if not isinstance(resp, list):
        return True
    if not resp:
        return False
    return bool(resp[0].get('isActivelyTrading'))


def reused_ticker_history(bt: sqlite3.Connection | None, symbol: str, ipo_date: str | None) -> str | None:
    """Return a description if backtest.db holds a DIFFERENT (dead) company's
    history under this symbol, else None."""
    if bt is None:
        return None
    try:
        delisted = bt.execute("SELECT company_name FROM delisted_companies WHERE symbol = ?",
                              (symbol,)).fetchone()
        span = bt.execute("SELECT MIN(date), MAX(date) FROM historical_prices WHERE symbol = ?",
                          (symbol,)).fetchone()
    except sqlite3.OperationalError:
        return None
    last = span[1] if span else None
    if delisted:
        return f"delisted_companies has {symbol} ({delisted[0]})"
    if last and ipo_date:
        try:
            gap = datetime.strptime(ipo_date[:10], '%Y-%m-%d') - datetime.strptime(last[:10], '%Y-%m-%d')
        except ValueError:
            return None
        if gap.days > REUSED_TICKER_GAP_DAYS:
            return f"stored history {span[0]}→{last} ends {gap.days}d before IPO {ipo_date[:10]}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='report only, no writes or collection')
    ap.add_argument('--db', default=DATABASE_NAME)
    ap.add_argument('--backtest-db', default=BACKTEST_DB)
    ap.add_argument('--min-mcap', type=float, default=DEFAULT_MIN_MCAP)
    ap.add_argument('--max-add', type=int, default=DEFAULT_MAX_ADD)
    args = ap.parse_args()
    if not FMP_API_KEY:
        sys.exit('FMP_API_KEY not set')

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    tracked = load_tracked(args.db)
    print(f"Tracked universe: {len(tracked)} symbols")
    found = discover(args.min_mcap)
    print(f"Discovered {len(found)} US stocks >= ${args.min_mcap / 1e6:.0f}M")

    tracked_by_name: dict[str, list[str]] = {}
    for sym, row in tracked.items():
        key = norm_name(row.get('company_name'))
        if key:
            tracked_by_name.setdefault(key, []).append(sym)

    # --- filter + dedupe ---------------------------------------------------
    best_by_name: dict[str, tuple] = {}
    skipped = {'noise': 0, 'secondary': 0, 'duplicate': 0}
    for sym, r in found.items():
        if sym in tracked:
            continue
        name = r.get('companyName') or ''
        if (NOISE_NAME.search(name) or COUPON_NAME.search(name) or SPAC_NAME.search(name)
                or NOISE_SYM.search(sym) or NOISE_SYM_BARE.match(sym)):
            skipped['noise'] += 1
            continue
        key = norm_name(name)
        same_name = tracked_by_name.get(key, [])
        old_tickers = []
        if same_name:
            # Rename candidates only: every same-name tracked ticker must be absent
            # from today's live screener and not freshly refreshed. Confirmed dead
            # at FMP below before anything changes.
            dead = [s for s in same_name
                    if s not in found and is_stale_or_unrefreshed(tracked[s].get('last_updated'), now)]
            if len(dead) != len(same_name):
                skipped['secondary'] += 1
                continue
            old_tickers = dead
        entry = (r.get('marketCap') or 0, sym, r, old_tickers)
        if key in best_by_name:
            skipped['duplicate'] += 1
            if entry[0] <= best_by_name[key][0]:
                continue
        best_by_name[key] = entry
    candidates = sorted(best_by_name.values(), key=lambda e: e[0], reverse=True)
    print(f"Candidates after filters: {len(candidates)} (dropped {skipped['noise']} non-common, "
          f"{skipped['secondary']} second securities, {skipped['duplicate']} same-company duplicates)")

    # --- verify ------------------------------------------------------------
    bt = sqlite3.connect(args.backtest_db) if Path(args.backtest_db).exists() else None
    to_add, renames, manual = [], [], []
    for mcap, sym, r, old_tickers in candidates:
        if len(to_add) >= args.max_add:
            break
        try:
            prof = fetch('profile', symbol=sym)
        except Exception as e:
            print(f"  {sym}: profile fetch failed ({e}) — retry next run")
            continue
        p = prof[0] if isinstance(prof, list) and prof else {}
        if not p or p.get('isEtf') or p.get('isFund'):
            continue
        if not is_us(p.get('exchange') or p.get('exchangeShortName')):
            continue
        desc = p.get('description') or ''
        if not p.get('sector') or not p.get('industry') or len(desc) < MIN_DESC_LEN:
            continue
        if SPAC_DESC.search(desc) or SPAC_DESC_EXTRA.search(desc):
            continue
        price, mcap_p, avg_vol = p.get('price') or 0, p.get('marketCap') or 0, p.get('averageVolume') or 0
        if mcap_p <= 0 or (price * avg_vol) / mcap_p < MIN_DAILY_TURNOVER:
            print(f"  {sym}: skipped — market cap ${mcap_p / 1e9:.1f}B inconsistent with "
                  f"${price * avg_vol / 1e6:.1f}M/day traded (bad vendor data)")
            continue
        if looks_like_par_security(sym, p):
            print(f"  {sym}: skipped — $25-par note/preferred pattern "
                  f"(price {price}, 52-wk {p.get('range')})")
            continue
        reused = reused_ticker_history(bt, sym, p.get('ipoDate'))
        if reused:
            manual.append((sym, p.get('companyName'), reused))
            continue
        if old_tickers and any(ticker_is_live(o) for o in old_tickers):
            print(f"  {sym}: skipped — second security of covered {','.join(old_tickers)}")
            continue
        to_add.append((sym, p, old_tickers))
        renames.extend((o, sym) for o in old_tickers)
        time.sleep(0.1)
    if bt is not None:
        bt.close()

    print(f"\nAdding {len(to_add)} symbols:")
    for sym, p, old in to_add:
        tag = f"  (renamed from {','.join(old)})" if old else ''
        print(f"  {sym:<6} ${(p.get('marketCap') or 0) / 1e9:>8.1f}B  {p.get('companyName')}  "
              f"[IPO {p.get('ipoDate') or '?'}]{tag}")
    for sym, name, why in manual:
        print(f"::warning::{sym} ({name}) not added — reused ticker: {why}. Needs manual handling.")
    if args.dry_run:
        print('\n(dry-run: no writes)')
        return
    if not to_add:
        return

    conn = sqlite3.connect(args.db)
    conn.execute("""CREATE TABLE IF NOT EXISTS retired_symbols (
        symbol TEXT PRIMARY KEY, successor TEXT, retired_at TEXT)""")
    for sym, p, old_tickers in to_add:
        if len(old_tickers) == 1:
            # Rename in place: score + price carry over until the next refresh so
            # the company never drops off the site between runs. Identity fields
            # come from the NEW profile — exchange/country are never refreshed by
            # the bulk path (BMOOF→BMM would stay 'OTC' and score as non-US).
            conn.execute("""
                UPDATE stock_consensus
                   SET symbol = ?, company_name = ?, sector = ?, industry = ?,
                       company_description = ?, market_cap = ?, exchange = ?, country = ?
                 WHERE symbol = ?
            """, (sym, p.get('companyName'), p.get('sector'), p.get('industry'),
                  p.get('description'), p.get('marketCap'),
                  p.get('exchange') or p.get('exchangeShortName'), p.get('country'),
                  old_tickers[0]))
        else:
            for o in old_tickers:  # several old rows → keep none, insert fresh
                conn.execute("DELETE FROM stock_consensus WHERE symbol = ?", (o,))
            conn.execute("""
                INSERT OR IGNORE INTO stock_consensus
                    (symbol, company_name, sector, industry, company_description,
                     market_cap, exchange, country, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """, (sym, p.get('companyName'), p.get('sector'), p.get('industry'),
                  p.get('description'), p.get('marketCap'),
                  p.get('exchange') or p.get('exchangeShortName'), p.get('country')))
        for o in old_tickers:
            conn.execute("INSERT OR REPLACE INTO retired_symbols VALUES (?, ?, ?)",
                         (o, sym, now.isoformat()))
        # A symbol coming (back) into the universe is never retired.
        conn.execute("DELETE FROM retired_symbols WHERE symbol = ?", (sym,))
    conn.commit()
    conn.close()

    # Carry the catalyst inputs across ticker changes: earnings surprises and
    # analyst grades are collected by announcement date, so without this a
    # renamed company restarts its beat streak from zero. OR IGNORE leaves any
    # (symbol, date, …) collisions under the old ticker.
    renamed_pairs = [(sym, old[0]) for sym, _, old in to_add if len(old) == 1]
    if renamed_pairs and Path(args.backtest_db).exists():
        with sqlite3.connect(args.backtest_db) as bt_w:
            for table in ('earnings_surprises', 'historical_grades'):
                try:
                    bt_w.executemany(f"UPDATE OR IGNORE {table} SET symbol = ? WHERE symbol = ?",
                                     renamed_pairs)
                except sqlite3.OperationalError:
                    pass  # table absent on a fresh DB
    print(f"\n✓ Added {len(to_add)} symbols ({len(renames)} ticker changes renamed in place)")

    syms = ','.join(sym for sym, _, _ in to_add)
    print(f"Collecting history for {len(to_add)} symbols...")
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / 'collectors' / 'collect_historical_data.py'),
         '--symbols', syms, '--force'],
        cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print(f"::warning::history collection exited {res.returncode}; "
              f"backfill_recent_prices / weekly fundamentals will fill gaps")


if __name__ == '__main__':
    main()

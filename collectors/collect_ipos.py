#!/usr/bin/env python3
"""
Recently-IPO'd collector — surfaces brand-new public companies that can't be
Compass-scored yet, so a search for e.g. SPCX / SpaceX returns the real company
instead of nothing.

A fresh IPO lacks the ~30 days of prices + several quarters of fundamentals the
Compass scorer needs, so it has no score and `export_website_stocks.py` drops it
(WHERE compass_score IS NOT NULL). This collector pulls FMP's ipos-calendar for
the trailing window, keeps real US operating-company COMMON STOCK (drops
SPAC/ETF/unit/right/warrant/notes by name), drops anything that's ALREADY
Compass-scored (those ship via the normal export — no duplicates), enriches the
survivors with current price / market cap / sector via /stable/profile, applies a
strict quality bar (price > $1, market cap > $50M, real metadata), and writes
recent-ipos.json next to stocks.json. The site merges these into the browse
search as "score pending" cards.

Usage:
    python collectors/collect_ipos.py                 # 6-month window
    python collectors/collect_ipos.py --months 6
    python collectors/collect_ipos.py --dry-run       # report only, no write
"""
from __future__ import annotations
import os, sys, re, json, argparse, sqlite3, urllib.request
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FMP_API_KEY, FMP_BASE_URL, DATABASE_NAME

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'

# Non-operating / non-common-stock securities to drop (name-based — the reliable
# signal). Kept narrow on purpose: ADRs/ordinary-shares/royalty cos are real and
# tradeable on US exchanges, so they're NOT dropped here (the site already lists
# ADRs). The strict price/market-cap bar + FMP isEtf/isFund flags catch the rest.
NOISE_NAME = re.compile(
    r'\b(ETF|ETN|Index Fund|\bFund\b|SPAC|\bAcquisitions?\b|Merger Corp|'
    r'Equity Partners|Blank Check|When Issued|Warrants?|\bRights?\b|'
    r'\bUnits?\b|Preferred|\bPref\b|\bNotes?\b|Debenture|Subordinated|'
    r'Floating Rate|Series [A-Z]\b|Capital Trust|\bCLO\b)\b', re.I)
# NOTE: deliberately NOT matching bare "Trust" or "Capital Corp" — genuinely new
# REIT/BDC IPOs carry those words (Camden Property Trust-class names) and are real
# operating companies. SPAC shells still get caught by Acquisition/Merger/Blank
# Check above plus the SPAC_DESC description check below.
# Dotted/dashed ticker suffixes that denote warrants/units/rights/preferreds.
NOISE_SYM = re.compile(r'[.\-](WS|WT|U|UN|R|RT|W|P[A-Z]?)$', re.I)
# Bare 4-5 char tickers ending in U/UN = SPAC units (TVIVU, WLCOU, XYZU).
NOISE_SYM_BARE = re.compile(r'^[A-Z]{3,}(U|UN)$')
# SPAC shells read as blank-check companies in their FMP description.
SPAC_DESC = re.compile(r'blank[ -]check|special[ -]purpose acquisition', re.I)

QUALITY_MIN_PRICE = 1.0
QUALITY_MIN_MCAP = 50_000_000   # $50M floor — cut penny/micro-junk IPOs
MIN_DESC_LEN = 50               # mirror export_website_stocks.py metadata bar

# Write next to stocks.json (CI overrides the dir via STOCKS_JSON_OUTPUT).
OUTPUT_FILE = Path(os.environ.get(
    'STOCKS_JSON_OUTPUT',
    str(Path(__file__).resolve().parent.parent.parent /
        'compass-score-site' / 'src' / 'data' / 'stocks.json')))
RECENT_IPOS_FILE = OUTPUT_FILE.parent / 'recent-ipos.json'


def fetch(path, **p):
    p['apikey'] = FMP_API_KEY
    qs = '&'.join(f'{k}={v}' for k, v in p.items())
    req = urllib.request.Request(f'{FMP_BASE_URL}/stable/{path}?{qs}', headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def is_us(exchange: str) -> bool:
    e = (exchange or '').upper()
    return ('NASDAQ' in e) or ('NYSE' in e) or ('AMEX' in e)


def fetch_ipo_calendar(months: int) -> list:
    today = datetime.now(timezone.utc).date()
    frm = (today - timedelta(days=months * 31)).isoformat()
    to = today.isoformat()
    rows = fetch('ipos-calendar', **{'from': frm, 'to': to})
    rows = rows if isinstance(rows, list) else []
    print(f"ipos-calendar {frm} -> {to}: {len(rows)} rows")
    return rows


def scored_symbols() -> set:
    """Symbols that already have a Compass score (ship via the normal export)."""
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        rows = conn.execute(
            "SELECT symbol FROM stock_consensus WHERE compass_score IS NOT NULL").fetchall()
        conn.close()
        return {r[0].upper() for r in rows if r[0]}
    except sqlite3.OperationalError:
        return set()   # fresh/partial DB — nothing scored yet


def candidate_symbols(rows: list, scored: set) -> dict:
    """First-pass name/exchange/dup filter. Returns {symbol: ipo_date}."""
    out = {}
    for r in rows:
        sym = (r.get('symbol') or '').upper()
        name = r.get('company') or ''
        date = r.get('date') or ''
        if not sym or not name or sym in scored:
            continue
        if not is_us(r.get('exchange')):
            continue
        if NOISE_NAME.search(name) or NOISE_SYM.search(sym) or NOISE_SYM_BARE.match(sym):
            continue
        # keep the earliest IPO date if a symbol appears on multiple rows
        if sym not in out or date < out[sym]:
            out[sym] = date
    return out


def enrich(sym: str, ipo_date: str) -> dict | None:
    """Profile lookup + strict quality bar. Returns a site record or None."""
    try:
        prof = fetch('profile', symbol=sym)
    except Exception as e:
        print(f"  {sym}: profile fetch failed ({e})"); return None
    p = prof[0] if isinstance(prof, list) and prof else {}
    if not p or p.get('isEtf') or p.get('isFund'):
        return None
    price = p.get('price') or 0
    mcap = p.get('marketCap') or 0
    sector = p.get('sector') or ''
    industry = p.get('industry') or ''
    desc = p.get('description') or ''
    if price <= QUALITY_MIN_PRICE or mcap <= QUALITY_MIN_MCAP:
        return None
    if not sector or not industry or len(desc) < MIN_DESC_LEN:
        return None
    if SPAC_DESC.search(desc):   # blank-check shell that slipped the name filter
        return None
    eff_ipo = ipo_date or p.get('ipoDate') or ''
    days = None
    if eff_ipo[:4].isdigit():
        try:
            days = (datetime.now(timezone.utc).date()
                    - datetime.strptime(eff_ipo[:10], '%Y-%m-%d').date()).days
        except ValueError:
            days = None
    return {
        'symbol': sym,
        'name': p.get('companyName') or sym,
        'ipoDate': eff_ipo[:10],
        'daysPublic': days,
        'price': round(float(price), 2),
        'marketCap': round(float(mcap) / 1e9, 6),   # billions, matching stocks-public.json
        'sector': sector,
        'industry': industry,
        'description': desc[:400],
        'exchange': p.get('exchangeShortName') or p.get('exchange') or '',
        'scoreStatus': 'pending',
    }


def run(args):
    if not FMP_API_KEY:
        print("FMP_API_KEY not set"); sys.exit(1)
    rows = fetch_ipo_calendar(args.months)
    scored = scored_symbols()
    cands = candidate_symbols(rows, scored)
    print(f"candidates after name/exchange/dup filter: {len(cands)} "
          f"(excluded {len(scored)} already-scored)")

    records = []
    for sym, ipo_date in sorted(cands.items()):
        rec = enrich(sym, ipo_date)
        if rec:
            records.append(rec)
    records.sort(key=lambda r: (r.get('ipoDate') or ''), reverse=True)
    print(f"recent IPOs passing strict quality bar: {len(records)}")
    for r in records[:25]:
        print(f"  {r['ipoDate']}  {r['symbol']:6s} ${r['price']:<8} "
              f"{r['marketCap']:.2f}B  {r['name'][:40]}")

    if args.dry_run:
        print("(dry-run, no write)"); return
    RECENT_IPOS_FILE.write_text(json.dumps(records, indent=2))
    print(f"\nWrote {len(records)} records to {RECENT_IPOS_FILE}")


def main():
    ap = argparse.ArgumentParser(description="Collect recently-IPO'd US common stock (score-pending)")
    ap.add_argument('--months', type=int, default=6, help='Trailing window in months (default: 6)')
    ap.add_argument('--dry-run', action='store_true', help='Report only, no file write')
    run(ap.parse_args())


if __name__ == '__main__':
    main()

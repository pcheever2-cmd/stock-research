#!/usr/bin/env python3
"""
Delisted-company collector (survivorship correction).

Pulls FMP's delisted-companies list, keeps US-listed OPERATING companies (drops
ETFs/SPACs/warrants/units/preferreds/notes by name), and persists them to a
`delisted_companies` table in backtest.db. Reason (acquired vs bankrupt vs …) is
left NULL here and filled by the reason-classification step (web research /
price-path heuristic) — it biases returns in opposite directions so it must be
explicit per name.

NOTE: FMP's delisted coverage is dense only from ~2016 onward (≤5 names/year
before that). Survivorship can therefore be corrected for the 2020+ OOS window
but NOT for pre-2016 IS selection / 2008 / 2015 regimes — caveat those.

Usage:
    python collectors/collect_delisted.py            # fetch + persist the list
    python collectors/collect_delisted.py --status   # show table counts
"""
import sys, re, json, argparse, sqlite3, urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FMP_API_KEY, FMP_BASE_URL, BACKTEST_DB

UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
US_EXCHANGES = {'NASDAQ', 'NYSE', 'AMEX', 'NYSEAMERICAN'}
MIN_DELIST_YEAR = 2016  # FMP coverage floor

# Non-operating securities to drop (name-based — the reliable signal).
NOISE_NAME = re.compile(
    r'\b(ETF|ETN|Fund|Trust|Index|SPAC|Acquisition Corp|Warrant|Units?|Preferred|'
    r'Pref\b|Notes?|Debenture|Bond|Depositary|Senior|Subordinated|Floating Rate|'
    r'Series [A-Z]\b|Capital Trust)\b', re.I)
NOISE_SYM = re.compile(r'[.\-](WS|WT|U|UN|R|RT|P[A-Z]?)$', re.I)  # dotted/dashed suffixes only


def fetch(path, **p):
    p['apikey'] = FMP_API_KEY
    qs = '&'.join(f'{k}={v}' for k, v in p.items())
    req = urllib.request.Request(f'{FMP_BASE_URL}/stable/{path}?{qs}', headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_delisted_list():
    rows = []
    for page in range(0, 200):
        batch = fetch('delisted-companies', page=page, limit=100)
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
    print(f"Pulled {len(rows)} delisted rows from FMP")
    return rows


def is_operating(r):
    name = r.get('companyName') or ''
    sym = r.get('symbol') or ''
    if r.get('exchange') not in US_EXCHANGES:
        return False
    d = r.get('delistedDate') or ''
    if not (d[:4].isdigit() and int(d[:4]) >= MIN_DELIST_YEAR):
        return False
    if not name or NOISE_NAME.search(name) or NOISE_SYM.search(sym):
        return False
    return True


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS delisted_companies (
            symbol TEXT PRIMARY KEY,
            company_name TEXT,
            exchange TEXT,
            ipo_date TEXT,
            delisted_date TEXT,
            reason TEXT,              -- acquired|merged|went_private|bankrupt|liquidated|distress_delist|other|unknown
            reason_detail TEXT,
            reason_source TEXT,       -- URL, or 'price-path-heuristic'
            reason_confidence TEXT,   -- high|medium|low
            classified_at TEXT
        )
    """)
    conn.commit()


def persist(conn, companies):
    ensure_table(conn)
    n = 0
    for r in companies:
        conn.execute("""
            INSERT INTO delisted_companies (symbol, company_name, exchange, ipo_date, delisted_date)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                company_name=excluded.company_name, exchange=excluded.exchange,
                ipo_date=excluded.ipo_date, delisted_date=excluded.delisted_date
        """, (r['symbol'], r.get('companyName'), r.get('exchange'),
              r.get('ipoDate'), r.get('delistedDate')))
        n += 1
    conn.commit()
    return n


def status(conn):
    ensure_table(conn)
    total = conn.execute("SELECT COUNT(*) FROM delisted_companies").fetchone()[0]
    classified = conn.execute("SELECT COUNT(*) FROM delisted_companies WHERE reason IS NOT NULL").fetchone()[0]
    print(f"delisted_companies: {total} rows, {classified} classified ({total-classified} pending)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    conn = sqlite3.connect(str(BACKTEST_DB))
    if args.status:
        status(conn); conn.close(); return
    rows = fetch_delisted_list()
    companies = [r for r in rows if is_operating(r)]
    print(f"Operating US companies (>= {MIN_DELIST_YEAR}): {len(companies)}")
    n = persist(conn, companies)
    print(f"Persisted {n} rows to delisted_companies")
    status(conn)
    conn.close()


if __name__ == '__main__':
    main()

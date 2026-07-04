#!/usr/bin/env python3
"""
Export daily valuation-multiple overlays per stock to Cloudflare KV, to layer on
the price chart: trailing P/E and trailing EV/EBITDA over ~5 years, plus the
current forward P/E as a scalar (drawn as a reference line client-side).

Correctness choices:
- Multiples use RAW `close` (not adjusted_close) with as-reported EPS/shares so
  the line stays continuous across in-window splits (NVDA/TSLA). The price line
  on the chart still uses adjusted_close — different bases, each correct for its
  job.
- Fundamentals are aligned to each price day by FILING_DATE (when the market knew
  them), via merge_asof backward — no look-ahead bias.
- TTM = rolling 4-quarter sum of eps_diluted / ebitda. P/E and EV/EBITDA are
  emitted only where the denominator is positive (negative-earnings quarters gap).

KV value `overlays:SYM` = {"pe":[{time,value}], "evEbitda":[{time,value}], "fwdPe":<num|null>}
(series ascending by date). Output: byte-chunked /tmp/overlays_kv_<N>.json bulk files.

Usage: python scripts/export_overlays_kv.py [--out-dir /tmp] [--years 5]
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'

DEFAULT_STOCKS_JSON = PROJECT_ROOT.parent / 'compass-score-site' / 'src' / 'data' / 'stocks.json'
STOCKS_JSON = Path(os.environ.get('STOCKS_JSON_OUTPUT', str(DEFAULT_STOCKS_JSON)))
PUBLIC_FILE = STOCKS_JSON.parent / 'stocks-public.json'
PREMIUM_FILE = STOCKS_JSON.parent / 'stocks-premium.json'

MAX_CHUNK_BYTES = 80_000_000
MAX_CHUNK_KEYS = 5_000


def load_published_symbols() -> set[str]:
    path = PUBLIC_FILE if PUBLIC_FILE.exists() else STOCKS_JSON
    if not path.exists():
        sys.exit(f"ERROR: no published symbol list at {PUBLIC_FILE} or {STOCKS_JSON}")
    syms = {r['symbol'].upper() for r in json.loads(path.read_text()) if r.get('symbol')}
    print(f"Loaded {len(syms)} published symbols from {path.name}")
    return syms


def load_current_fwd_pe() -> dict:
    """Current forward P/E snapshot per symbol (for the reference line)."""
    if not PREMIUM_FILE.exists():
        return {}
    data = json.loads(PREMIUM_FILE.read_text())
    out = {}
    for sym, fields in data.items():
        v = fields.get('forwardPe')
        if isinstance(v, (int, float)) and v > 0:
            out[sym.upper()] = round(float(v), 2)
    return out


def load_quarterly_fundamentals(conn, symbols: set[str]) -> pd.DataFrame:
    """Per-quarter TTM EPS/EBITDA + shares + net debt, keyed by filing date."""
    # Standalone-safe: setup_database.py owns the canonical schema, but this
    # guard keeps the LEFT JOIN below working on a fresh/partial DB.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_flags (
            symbol TEXT NOT NULL, table_name TEXT NOT NULL, date TEXT NOT NULL,
            field TEXT NOT NULL, reported REAL, computed REAL, flagged_at TEXT,
            PRIMARY KEY (symbol, table_name, date, field)
        )
    """)
    # Exclude rows flagged by the EPS consistency guard (data_quality_flags):
    # a quarter whose reported EPS disagrees >10x with net_income/shares would
    # poison four TTM windows of the P/E line (OPFI charted P/E 0.2x for
    # months off one bad vendor row). NULLing eps_diluted keeps the quarter
    # for EBITDA but gaps the P/E honestly — a gap beats a lie.
    # A flag can't tell WHICH side of the eps-vs-NI/shares mismatch is wrong
    # (eps, or the shares field EV/EBITDA's market cap also uses) — so a
    # flagged quarter gaps BOTH overlay series, not just P/E.
    inc = pd.read_sql_query(
        """SELECT i.symbol, i.date,
                  CASE WHEN f.symbol IS NULL THEN i.eps_diluted ELSE NULL END AS eps_diluted,
                  CASE WHEN f.symbol IS NULL THEN i.ebitda ELSE NULL END AS ebitda,
                  i.weighted_avg_shares_diluted AS shares,
                  i.filing_date
           FROM historical_income_statements i
           LEFT JOIN data_quality_flags f
             ON f.symbol = i.symbol AND f.date = i.date
            AND f.table_name = 'historical_income_statements'
            AND f.field = 'eps_diluted'
           WHERE i.period LIKE 'Q%'""", conn)
    bal = pd.read_sql_query(
        """SELECT symbol, date, net_debt FROM historical_balance_sheets WHERE period LIKE 'Q%'""",
        conn)

    inc['symbol'] = inc['symbol'].str.upper()
    bal['symbol'] = bal['symbol'].str.upper()
    inc = inc[inc['symbol'].isin(symbols)].copy()
    bal = bal[bal['symbol'].isin(symbols)].copy()

    df = inc.merge(bal, on=['symbol', 'date'], how='left')
    # filing_date is when the market knew the quarter; fall back to period-end + ~75d.
    df['asof'] = pd.to_datetime(df['filing_date'], errors='coerce')
    fallback = pd.to_datetime(df['date'], errors='coerce') + pd.Timedelta(days=75)
    df['asof'] = df['asof'].fillna(fallback)
    df = df.dropna(subset=['asof']).sort_values(['symbol', 'date'])

    # TTM via trailing 4 quarters (needs 4 to be meaningful).
    g = df.groupby('symbol', group_keys=False)
    df['ttmEps'] = g['eps_diluted'].apply(lambda s: s.rolling(4, min_periods=4).sum())
    df['ttmEbitda'] = g['ebitda'].apply(lambda s: s.rolling(4, min_periods=4).sum())
    return df[['symbol', 'asof', 'ttmEps', 'ttmEbitda', 'shares', 'net_debt']].rename(
        columns={'net_debt': 'netDebt'})


def build_overlays(conn, symbols: set[str], cutoff: str, fwd_pe: dict) -> dict:
    fund = load_quarterly_fundamentals(conn, symbols)

    prices = pd.read_sql_query(
        "SELECT symbol, date, close FROM historical_prices WHERE date >= ?",
        conn, params=(cutoff,))
    prices['symbol'] = prices['symbol'].str.upper()
    prices = prices[prices['symbol'].isin(symbols) & prices['close'].notna()].copy()
    prices['dt'] = pd.to_datetime(prices['date'], errors='coerce')
    prices = prices.dropna(subset=['dt']).sort_values('dt')

    fund = fund.dropna(subset=['asof']).sort_values('asof')
    # merge_asof backward: each price day gets the most recent already-filed quarter.
    merged = pd.merge_asof(
        prices, fund, left_on='dt', right_on='asof', by='symbol', direction='backward')

    # Multiples from RAW close; only where the denominator is positive.
    mc = merged['close'] * merged['shares']
    merged['pe'] = (merged['close'] / merged['ttmEps']).where(merged['ttmEps'] > 0)
    merged['evEbitda'] = ((mc + merged['netDebt']) / merged['ttmEbitda']).where(
        merged['ttmEbitda'] > 0)

    overlays: dict[str, dict] = {}
    for sym, grp in merged.groupby('symbol'):
        pe = [{'time': d, 'value': round(float(v), 2)}
              for d, v in zip(grp['date'], grp['pe']) if pd.notna(v) and 0 < v < 1000]
        ev = [{'time': d, 'value': round(float(v), 2)}
              for d, v in zip(grp['date'], grp['evEbitda']) if pd.notna(v) and 0 < v < 1000]
        if len(pe) < 2 and len(ev) < 2:
            continue
        overlays[sym] = {'pe': pe, 'evEbitda': ev, 'fwdPe': fwd_pe.get(sym)}
    return overlays


def write_chunks(overlays: dict, out_dir: Path) -> list[Path]:
    pairs = [{'key': f'overlays:{sym}', 'value': json.dumps(o, separators=(',', ':'))}
             for sym, o in sorted(overlays.items())]
    files: list[Path] = []
    batch: list[dict] = []
    batch_bytes = 0

    def flush():
        nonlocal batch, batch_bytes
        if not batch:
            return
        part = out_dir / f'overlays_kv_{len(files)}.json'
        part.write_text(json.dumps(batch, separators=(',', ':')))
        files.append(part)
        batch, batch_bytes = [], 0

    for p in pairs:
        sz = len(json.dumps(p, separators=(',', ':'))) + 1
        if batch and (batch_bytes + sz > MAX_CHUNK_BYTES or len(batch) >= MAX_CHUNK_KEYS):
            flush()
        batch.append(p)
        batch_bytes += sz
    flush()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp')
    ap.add_argument('--years', type=int, default=5)
    args = ap.parse_args()

    if not BACKTEST_DB.exists():
        sys.exit(f"ERROR: {BACKTEST_DB} not found (download the data release first)")

    symbols = load_published_symbols()
    fwd_pe = load_current_fwd_pe()
    cutoff = (datetime.utcnow() - timedelta(days=args.years * 365)).strftime('%Y-%m-%d')
    print(f"Building daily P/E + EV/EBITDA overlays since {cutoff}...")

    conn = sqlite3.connect(str(BACKTEST_DB))
    overlays = build_overlays(conn, symbols, cutoff, fwd_pe)
    conn.close()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = write_chunks(overlays, out_dir)

    pe_pts = sum(len(o['pe']) for o in overlays.values())
    ev_pts = sum(len(o['evEbitda']) for o in overlays.values())
    with_fwd = sum(1 for o in overlays.values() if o['fwdPe'] is not None)
    print(f"\n✓ Wrote {len(overlays)} symbols across {len(files)} file(s):")
    for f in files:
        print(f"    {f} ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"  P/E points: {pe_pts:,} | EV/EBITDA points: {ev_pts:,} | with fwd P/E: {with_fwd}")
    missing = len(symbols) - len(overlays)
    if missing:
        print(f"  ⚠ {missing} symbols had no usable multiples (sparse/negative fundamentals)")


if __name__ == '__main__':
    main()

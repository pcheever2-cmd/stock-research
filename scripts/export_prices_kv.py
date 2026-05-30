#!/usr/bin/env python3
"""
Export ~5 years of daily closing prices per stock to Cloudflare KV bulk files.

Reads daily history from backtest.db (historical_prices) for the published
universe (the symbols in stocks-public.json), and writes one KV entry per
symbol keyed `prices:SYM`. Each value is the exact shape lightweight-charts
consumes, so the serving Function is a pure pass-through:

    [{ "time": "2021-05-28", "value": 148.21 }, ...]   # ascending by date

Output: chunked bulk files /tmp/prices_kv_<N>.json, each a list of
{"key": "prices:SYM", "value": "<json string>"} — the same format the premium
KV-sync step produces. The workflow then runs `wrangler kv bulk put` per file.

Usage:
    python scripts/export_prices_kv.py
    python scripts/export_prices_kv.py --out-dir /tmp --years 5 --chunk 1000
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'

# Where the published universe lives. Mirrors export_website_stocks.py: the
# split files sit alongside STOCKS_JSON_OUTPUT (CI: data/stocks.json → data/).
DEFAULT_STOCKS_JSON = PROJECT_ROOT.parent / 'compass-score-site' / 'src' / 'data' / 'stocks.json'
STOCKS_JSON = Path(os.environ.get('STOCKS_JSON_OUTPUT', str(DEFAULT_STOCKS_JSON)))
PUBLIC_FILE = STOCKS_JSON.parent / 'stocks-public.json'


def load_published_symbols() -> set[str]:
    """Symbols that have a built detail page (the source of truth for which
    pages exist). Falls back to stocks.json if the public split is absent."""
    path = PUBLIC_FILE if PUBLIC_FILE.exists() else STOCKS_JSON
    if not path.exists():
        sys.exit(f"ERROR: no published symbol list found at {PUBLIC_FILE} or {STOCKS_JSON}")
    records = json.loads(path.read_text())
    syms = {r['symbol'].upper() for r in records if r.get('symbol')}
    print(f"Loaded {len(syms)} published symbols from {path.name}")
    return syms


def build_series(db_path: Path, published: set[str], cutoff: str):
    """Stream historical_prices since cutoff, grouped by symbol, into
    ascending [{time, value}] arrays. Prefers adjusted_close (continuous across
    splits/dividends), falls back to close. Returns (series, sparse, dropped)."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = None
    cur = conn.execute(
        """
        SELECT symbol, date, COALESCE(adjusted_close, close) AS px
        FROM historical_prices
        WHERE date >= ?
        ORDER BY symbol, date
        """,
        (cutoff,),
    )

    series: dict[str, list[dict]] = {}
    cur_sym = None
    points: list[dict] = []

    def flush(sym, pts):
        if sym is not None and pts:
            series[sym] = pts

    for symbol, date, px in cur:
        sym = symbol.upper()
        if sym not in published or px is None:
            continue
        if sym != cur_sym:
            flush(cur_sym, points)
            cur_sym, points = sym, []
        points.append({'time': date, 'value': round(float(px), 4)})
    flush(cur_sym, points)
    conn.close()

    # A chart needs >= 2 points to draw a line; sparser names degrade (hidden client-side).
    sparse = sorted(s for s, p in series.items() if len(p) < 2)
    for s in sparse:
        del series[s]
    dropped = sorted(published - set(series) - set(sparse))  # published but no rows at all
    return series, sparse, dropped


# wrangler `kv bulk put` limits: 10k keys/op and ~100MB request body. A full
# 5yr daily series serializes to ~45-50KB, so we cap each file well under both
# limits by BYTES (not key count) — robust even if series grow (more history /
# density) without silently producing a chunk wrangler would reject whole.
MAX_CHUNK_BYTES = 80_000_000
MAX_CHUNK_KEYS = 5_000


def write_chunks(series: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    """Write {key,value} bulk files, splitting on accumulated byte size or key count."""
    pairs = [{'key': f'prices:{sym}', 'value': json.dumps(pts, separators=(',', ':'))}
             for sym, pts in sorted(series.items())]
    files: list[Path] = []
    batch: list[dict] = []
    batch_bytes = 0

    def flush():
        nonlocal batch, batch_bytes
        if not batch:
            return
        part = out_dir / f'prices_kv_{len(files)}.json'
        part.write_text(json.dumps(batch, separators=(',', ':')))
        files.append(part)
        batch, batch_bytes = [], 0

    for p in pairs:
        # Exact serialized size incl. quote-escaping of the value string, + 1 for
        # the entry's comma — estimating undercounts the escaping (~15%).
        sz = len(json.dumps(p, separators=(',', ':'))) + 1
        if batch and (batch_bytes + sz > MAX_CHUNK_BYTES or len(batch) >= MAX_CHUNK_KEYS):
            flush()
        batch.append(p)
        batch_bytes += sz
    flush()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp', help='Directory for bulk KV files')
    ap.add_argument('--years', type=int, default=5, help='Lookback window in years')
    args = ap.parse_args()

    if not BACKTEST_DB.exists():
        sys.exit(f"ERROR: {BACKTEST_DB} not found (download the data release first)")

    published = load_published_symbols()
    cutoff = (datetime.utcnow() - timedelta(days=args.years * 365)).strftime('%Y-%m-%d')
    print(f"Building daily series since {cutoff} from {BACKTEST_DB.name}...")

    series, sparse, dropped = build_series(BACKTEST_DB, published, cutoff)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = write_chunks(series, out_dir)

    total_pts = sum(len(p) for p in series.values())
    print(f"\n✓ Wrote {len(series)} symbols ({total_pts:,} points) across {len(files)} file(s):")
    for f in files:
        print(f"    {f} ({f.stat().st_size / 1e6:.1f} MB)")

    # Freshness guard: a quietly-stalled backfill would publish old closes as the
    # latest point. Warn (don't fail) so erosion isn't silent. See the price-coverage
    # hardening work — silent staleness is the failure mode we're guarding against.
    latest = max((p[-1]['time'] for p in series.values()), default=None)
    if latest:
        stale_days = (datetime.utcnow() - datetime.strptime(latest, '%Y-%m-%d')).days
        flag = '  ⚠ STALE — check the backfill' if stale_days > 5 else ''
        print(f"  Latest price date: {latest} ({stale_days}d old){flag}")
    # No silent truncation — surface coverage gaps.
    if sparse:
        print(f"  ⚠ {len(sparse)} symbols had <2 points (skipped): {', '.join(sparse[:20])}"
              + (" …" if len(sparse) > 20 else ""))
    if dropped:
        print(f"  ⚠ {len(dropped)} published symbols had NO price rows since {cutoff}: "
              + ', '.join(dropped[:20]) + (" …" if len(dropped) > 20 else ""))


if __name__ == '__main__':
    main()

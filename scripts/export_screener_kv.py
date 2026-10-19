#!/usr/bin/env python3
"""
Export the private-API screening index to Cloudflare KV.

Joins the full website export (data/stocks.json: public + premium fields per
stock) with the valuation-history summary emitted by export_overlays_kv.py
(--summary-out), into ONE compact KV value under the key `screener:index`.
The site's /api/private/* functions read this single key to run server-side
screens — the public fields (name/price/compass/sector/analysts) exist nowhere
else in KV (they are baked into the static build).

Row policy: every stock in stocks.json gets a row; missing metrics are null so
the consuming function sees a consistent shape. No descriptions or ratings text
— the value must stay far below the 25 MB KV limit (abort guard at 20 MB).

Usage: python scripts/export_screener_kv.py [--out-dir /tmp]
       [--valuation-summary /tmp/valuation_history.json]
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STOCKS_JSON = PROJECT_ROOT.parent / 'compass-score-site' / 'src' / 'data' / 'stocks.json'
STOCKS_JSON = Path(os.environ.get('STOCKS_JSON_OUTPUT', str(DEFAULT_STOCKS_JSON)))

MAX_VALUE_BYTES = 20_000_000  # KV hard limit is 25 MB; abort well before it

# Valuation-history fields copied through from valuation_history.json verbatim.
VAL_FIELDS = ('peNow', 'pe5yMedian', 'pePctile', 'peVsMedPct', 'peN',
              'evNow', 'ev5yMedian', 'evPctile', 'evVsMedPct', 'evN')


def _num(v, nd=2):
    """Round a numeric field; pass None/non-numbers through as null."""
    if not isinstance(v, (int, float)):
        return None
    r = round(float(v), nd)
    return int(r) if nd == 0 else r


def build_row(stock: dict, val: dict) -> dict:
    fair = stock.get('fairValue') or {}
    price = _num(stock.get('price'))
    sma200 = _num(stock.get('sma200'))
    vs_sma200 = (round((price / sma200 - 1) * 100, 1)
                 if price and sma200 and sma200 > 0 else None)
    row = {
        'symbol': stock.get('symbol'),
        'name': stock.get('name'),
        'price': price,
        'compass': _num(stock.get('compassScore'), 0),
        'grade': stock.get('grade'),
        'sector': stock.get('sector'),
        'industry': stock.get('industry'),
        'mcapB': _num(stock.get('marketCap')),
        'analysts': _num(stock.get('numAnalysts'), 0),
        'consensus': stock.get('consensus'),
        'golden': bool(stock.get('isGolden')),
        'catalyst': _num(stock.get('catalystScore'), 0),
        'catalystTag': bool(stock.get('catalystTag')),
        'beatStreak': _num(stock.get('beatStreak'), 0),
        'epsSurprisePct': _num(stock.get('epsSurprisePct'), 1),
        'netUpgrades180d': _num(stock.get('netUpgrades180d'), 0),
        'fvUpsidePct': _num(fair.get('upsidePct'), 1),
        'fvVerdict': fair.get('verdict'),
        'evEbitda': _num(stock.get('evEbitda'), 1),
        'fwdPe': _num(stock.get('forwardPe'), 1),
        'peg': _num(stock.get('pegRatio'), 2),
        'rsi': _num(stock.get('rsi'), 1),
        'trendSignal': stock.get('trendSignal'),
        'priceVsSma200Pct': vs_sma200,
    }
    for f in VAL_FIELDS:
        row[f] = val.get(f) if val else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='/tmp')
    ap.add_argument('--valuation-summary', default='/tmp/valuation_history.json')
    args = ap.parse_args()

    if not STOCKS_JSON.exists():
        sys.exit(f"ERROR: {STOCKS_JSON} not found (run export_website_stocks.py first)")
    stocks = json.loads(STOCKS_JSON.read_text())
    if not isinstance(stocks, list) or not stocks:
        sys.exit(f"ERROR: {STOCKS_JSON} is empty or not a list")

    summary_path = Path(args.valuation_summary)
    if summary_path.exists():
        valuation = json.loads(summary_path.read_text())
        print(f"Loaded valuation history for {len(valuation)} symbols from {summary_path}")
    else:
        valuation = {}
        print(f"WARNING: {summary_path} missing — valuation-history fields will be null "
              f"(run export_overlays_kv.py first)")

    rows = []
    for s in stocks:
        sym = (s.get('symbol') or '').upper()
        if not sym:
            continue
        row = build_row(s, valuation.get(sym))
        row['symbol'] = sym  # canonical upper-case — KV lookups compare against it
        rows.append(row)

    index = {
        'asOf': datetime.utcnow().strftime('%Y-%m-%d'),
        'count': len(rows),
        'stocks': rows,
    }
    value = json.dumps(index, separators=(',', ':'))
    if len(value) > MAX_VALUE_BYTES:
        sys.exit(f"ERROR: screener:index is {len(value) / 1e6:.1f} MB (> {MAX_VALUE_BYTES / 1e6:.0f} MB "
                 f"guard; KV limit 25 MB) — trim fields before shipping")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'screener_kv_0.json'
    out.write_text(json.dumps([{'key': 'screener:index', 'value': value}],
                              separators=(',', ':')))

    with_val = sum(1 for r in rows if r['pePctile'] is not None or r['evPctile'] is not None)
    print(f"\n✓ screener:index — {len(rows)} symbols, {len(value) / 1e6:.2f} MB value → {out}")
    print(f"  with valuation history: {with_val} | without: {len(rows) - with_val}")


if __name__ == '__main__':
    main()

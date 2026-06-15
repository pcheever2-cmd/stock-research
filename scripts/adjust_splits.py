#!/usr/bin/env python3
"""
Split-adjust the CONTINUOUS price series (adjusted_close) in backtest.db so it
stays smooth across stock splits.

adjusted_close is captured once at fetch time and goes STALE after a split:
pre-split rows keep the old (high) basis while post-split rows (appended by
backfill_recent_prices.py) arrive at the new (low) basis -> a cliff. That cliff
breaks every consumer of the CONTINUOUS series: the 5yr price chart
(COALESCE(adjusted_close, close)) and the production scorers that compute
returns/momentum/volatility off adjusted_close (compass, momentum, valuation,
long-term) -- a stale cliff reads as a fake ~-90% return on the split date.

We rescale ONLY adjusted_close (and volume, to keep dollar-volume = price*vol
continuous with it). We deliberately leave raw open/high/low/CLOSE on their
NATIVE per-date basis: the P/E overlays (export_overlays_kv.py) and per-share
Fair Value (compute_fair_value_v2.py) pair RAW close with as-reported
EPS/shares, so the ratio is split-invariant ONLY while close stays native --
rescaling close would corrupt both. Different columns, different bases, each
correct for its consumer.

Flow:
  1. DETECT recent splits via FMP splits-calendar (market-wide), filter to our
     universe (symbols present in historical_prices).
  2. APPLY each split to rows dated BEFORE the split date:
        factor = denominator / numerator
        (a 10:1 split is numerator=10, denominator=1 -> factor=0.1)
     UPDATE adjusted_close *= factor (and volume /= factor for continuity).
  3. RECORD it in applied_splits (PRIMARY KEY symbol,date) and skip anything
     already there -> idempotent, no double-adjusting.

Usage:
    python scripts/adjust_splits.py                 # 35-day window
    python scripts/adjust_splits.py --window-days 7
    python scripts/adjust_splits.py --dry-run       # detect only, no writes
"""
import argparse
import logging
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import requests

# Add project root to path (mirror scripts/backfill_recent_prices.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import FMP_API_KEY, FMP_BASE_URL, BACKTEST_DB

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def ensure_applied_splits_table(conn: sqlite3.Connection) -> None:
    """Create applied_splits if missing (setup_database.py owns the canonical
    schema; this guards against a fresh/partial DB so the script is standalone)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applied_splits (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            numerator REAL,
            denominator REAL,
            applied_at TEXT,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()


def fetch_recent_splits(window_days: int) -> list[dict]:
    """Fetch market-wide splits from FMP for the trailing window.
    Returns list of {symbol, date, numerator, denominator, splitType}."""
    today = datetime.now()
    from_date = (today - timedelta(days=window_days)).strftime('%Y-%m-%d')
    to_date = today.strftime('%Y-%m-%d')
    url = f"{FMP_BASE_URL}/stable/splits-calendar"
    params = {'from': from_date, 'to': to_date, 'apikey': FMP_API_KEY}
    log.info(f"Fetching splits-calendar {from_date} -> {to_date}")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        log.warning(f"Unexpected splits-calendar payload: {type(data)}")
        return []
    return data


def get_universe(conn: sqlite3.Connection) -> set[str]:
    """Symbols present in historical_prices."""
    rows = conn.execute("SELECT DISTINCT symbol FROM historical_prices").fetchall()
    return {r[0] for r in rows}


def get_applied(conn: sqlite3.Connection) -> set[tuple[str, str]]:
    """(symbol, date) pairs already adjusted."""
    rows = conn.execute("SELECT symbol, date FROM applied_splits").fetchall()
    return {(r[0], r[1]) for r in rows}


def apply_split(conn: sqlite3.Connection, symbol: str, split_date: str,
                numerator: float, denominator: float) -> int:
    """Rescale the CONTINUOUS series (adjusted_close + volume) for pre-split rows
    of one (symbol, split_date) and record it. Raw open/high/low/close are left
    on their native basis on purpose -- see module docstring. Returns the number
    of price rows rescaled. Caller guarantees this split is not already in
    applied_splits."""
    factor = denominator / numerator  # 10:1 -> 1/10 = 0.1
    cur = conn.execute(
        """
        UPDATE historical_prices
           SET adjusted_close = adjusted_close * ?,
               volume = CASE WHEN volume IS NOT NULL THEN volume / ? ELSE volume END
         WHERE symbol = ? AND date < ?
        """,
        (factor, factor, symbol, split_date),
    )
    rescaled = cur.rowcount
    conn.execute(
        """
        INSERT INTO applied_splits (symbol, date, numerator, denominator, applied_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (symbol, split_date, numerator, denominator, datetime.now().isoformat()),
    )
    return rescaled


def run(args) -> None:
    if not FMP_API_KEY:
        log.error("FMP_API_KEY not set")
        sys.exit(1)

    splits = fetch_recent_splits(args.window_days)
    log.info(f"splits-calendar returned {len(splits)} market-wide splits")

    conn = sqlite3.connect(BACKTEST_DB)
    try:
        ensure_applied_splits_table(conn)
        universe = get_universe(conn)
        already = get_applied(conn)

        # Filter market-wide splits to our universe with usable factors.
        in_universe = []
        for s in splits:
            sym = s.get('symbol')
            date = s.get('date')
            num = s.get('numerator')
            den = s.get('denominator')
            if not sym or not date or num in (None, 0) or den in (None, 0):
                continue
            if sym not in universe:
                continue
            in_universe.append((sym, date, float(num), float(den)))

        newly_applied = 0
        already_applied = 0
        for sym, date, num, den in in_universe:
            if (sym, date) in already:
                already_applied += 1
                continue
            if args.dry_run:
                factor = den / num
                log.info(f"[dry-run] would adjust {sym} pre-{date} by factor {factor:.6g}")
                newly_applied += 1
                continue
            rescaled = apply_split(conn, sym, date, num, den)
            factor = den / num
            log.info(f"Adjusted {sym}: {rescaled} pre-{date} rows * {factor:.6g} "
                     f"({num:g}:{den:g} split)")
            newly_applied += 1

        if not args.dry_run:
            conn.commit()

        log.info(
            f"Summary: {len(in_universe)} splits found in universe, "
            f"{newly_applied} newly applied, {already_applied} already applied"
            + (" (dry-run, no writes)" if args.dry_run else "")
        )
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description='Split-adjust historical_prices in backtest.db')
    parser.add_argument('--window-days', type=int, default=35,
                        help='Look back this many days for recent splits (default: 35)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Detect and report splits without writing any changes')
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()

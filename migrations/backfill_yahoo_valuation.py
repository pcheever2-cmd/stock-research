#!/usr/bin/env python3
"""
Yahoo Finance Fallback for Missing Valuation Data
==================================================
Uses yfinance with curl_cffi impersonation to fetch valuation metrics
for stocks where FMP API returned NULL values.

Only processes stocks with missing: forward_pe, peg_ratio, or ev_ebitda
"""

from curl_cffi import requests
import yfinance as yf
import sqlite3
import time
import random
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Rate limiting settings (conservative to avoid blocks)
MIN_DELAY = 3.0
MAX_DELAY = 6.0

# Create impersonating session (mimics real Chrome browser)
session = requests.Session(impersonate="chrome110")


def get_stocks_missing_valuation():
    """Get stocks that are missing valuation metrics."""
    conn = sqlite3.connect(DATABASE_NAME)

    # Find stocks with compass_score but missing valuation data
    query = """
        SELECT symbol, company_name, forward_pe, peg_ratio, ev_ebitda
        FROM stock_consensus
        WHERE compass_score IS NOT NULL
          AND (forward_pe IS NULL OR peg_ratio IS NULL OR ev_ebitda IS NULL)
        ORDER BY compass_score DESC
    """

    stocks = conn.execute(query).fetchall()
    conn.close()

    return stocks


def fetch_yahoo_valuation(symbol: str) -> dict:
    """Fetch valuation metrics from Yahoo Finance."""
    try:
        stock = yf.Ticker(symbol, session=session)
        info = stock.info

        result = {
            'symbol': symbol,
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'trailing_pe': info.get('trailingPE'),
            'price_to_book': info.get('priceToBook'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'enterprise_value': info.get('enterpriseValue'),
            'ebitda': info.get('ebitda'),
            'earnings_growth': info.get('earningsGrowth'),
            'revenue_growth': info.get('revenueGrowth'),
        }

        return result

    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            print(f"  Rate limited - sleeping 5 minutes...")
            time.sleep(300)
            return fetch_yahoo_valuation(symbol)  # Retry once
        print(f"  Error fetching {symbol}: {e}")
        return {}


def update_stock_valuation(symbol: str, data: dict):
    """Update stock valuation in database (only NULL fields)."""
    if not data:
        return False

    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    # Build dynamic update - only set values that aren't NULL in new data
    updates = []
    params = []

    if data.get('forward_pe') is not None:
        updates.append("forward_pe = COALESCE(forward_pe, ?)")
        params.append(round(data['forward_pe'], 2))

    if data.get('peg_ratio') is not None:
        updates.append("peg_ratio = COALESCE(peg_ratio, ?)")
        params.append(round(data['peg_ratio'], 2))

    if data.get('ev_ebitda') is not None:
        updates.append("ev_ebitda = COALESCE(ev_ebitda, ?)")
        params.append(round(data['ev_ebitda'], 2))

    if data.get('enterprise_value') is not None:
        updates.append("enterprise_value = COALESCE(enterprise_value, ?)")
        params.append(data['enterprise_value'])

    if data.get('ebitda') is not None:
        updates.append("ebitda = COALESCE(ebitda, ?)")
        params.append(data['ebitda'])

    if data.get('trailing_pe') is not None:
        updates.append("trailing_pe = COALESCE(trailing_pe, ?)")
        params.append(round(data['trailing_pe'], 2))

    if data.get('price_to_book') is not None:
        updates.append("price_to_book = COALESCE(price_to_book, ?)")
        params.append(round(data['price_to_book'], 2))

    if data.get('earnings_growth') is not None:
        updates.append("earnings_growth = COALESCE(earnings_growth, ?)")
        params.append(data['earnings_growth'])

    if not updates:
        conn.close()
        return False

    params.append(symbol)
    query = f"UPDATE stock_consensus SET {', '.join(updates)} WHERE symbol = ?"

    cur.execute(query, params)
    conn.commit()
    conn.close()

    return cur.rowcount > 0


def main():
    print("=" * 60)
    print("YAHOO FINANCE VALUATION BACKFILL")
    print("=" * 60)
    print(f"Database: {DATABASE_NAME}")
    print()

    # Get stocks with missing valuation
    stocks = get_stocks_missing_valuation()
    print(f"Found {len(stocks)} stocks with missing valuation data")

    if not stocks:
        print("All stocks have valuation data!")
        return

    # Show what's missing
    print("\nSample of stocks to process:")
    for symbol, name, fwd_pe, peg, ev_eb in stocks[:10]:
        missing = []
        if fwd_pe is None:
            missing.append("forward_pe")
        if peg is None:
            missing.append("peg_ratio")
        if ev_eb is None:
            missing.append("ev_ebitda")
        print(f"  {symbol:<8} - Missing: {', '.join(missing)}")

    if len(stocks) > 10:
        print(f"  ... and {len(stocks) - 10} more")

    print()
    print("Starting Yahoo Finance lookups (with rate limiting)...")
    print("-" * 60)

    updated = 0
    failed = 0

    for i, (symbol, name, _, _, _) in enumerate(stocks):
        # Progress indicator
        if i > 0 and i % 50 == 0:
            print(f"\n--- Progress: {i}/{len(stocks)} ({updated} updated, {failed} failed) ---\n")

        data = fetch_yahoo_valuation(symbol)

        if data:
            success = update_stock_valuation(symbol, data)
            if success:
                fwd_pe = data.get('forward_pe')
                peg = data.get('peg_ratio')
                ev_eb = data.get('ev_ebitda')
                print(f"{symbol:<8} ✓ Forward P/E: {fwd_pe or 'N/A'}, PEG: {peg or 'N/A'}, EV/EBITDA: {ev_eb or 'N/A'}")
                updated += 1
            else:
                print(f"{symbol:<8} - No new data from Yahoo")
                failed += 1
        else:
            print(f"{symbol:<8} ✗ Failed to fetch")
            failed += 1

        # Rate limiting
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)

    print()
    print("=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Total processed: {len(stocks)}")
    print(f"Successfully updated: {updated}")
    print(f"Failed/No data: {failed}")

    # Show remaining missing
    remaining = get_stocks_missing_valuation()
    print(f"\nStocks still missing valuation data: {len(remaining)}")


if __name__ == '__main__':
    main()

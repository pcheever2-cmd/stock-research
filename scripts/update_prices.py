#!/usr/bin/env python3
"""
Lightweight price-only update for the website.
Fetches bulk quotes from FMP and updates stocks-public.json and stocks-premium.json
without running the full pipeline. Designed to run in < 2 minutes.

Usage:
    python scripts/update_prices.py [--output-dir DIR]
"""
import json
import os
import sys
import requests
from pathlib import Path

FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
FMP_BASE = 'https://financialmodelingprep.com'
BATCH_SIZE = 500  # FMP batch quote supports up to ~1000 symbols

def fetch_bulk_quotes(symbols: list[str]) -> dict[str, float]:
    """Fetch current prices for all symbols using FMP batch quote endpoint."""
    prices = {}
    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        url = f"{FMP_BASE}/stable/batch-quote?symbols={','.join(batch)}&apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  WARNING: Batch quote returned {resp.status_code} for batch {i//BATCH_SIZE + 1}")
            continue
        data = resp.json()
        if isinstance(data, list):
            for quote in data:
                sym = quote.get('symbol')
                price = quote.get('price')
                if sym and price is not None and price > 0:
                    prices[sym] = round(price, 2)
        print(f"  Fetched {len(batch)} quotes (batch {i//BATCH_SIZE + 1}), got {len([q for q in data if isinstance(q, dict) and q.get('price')])} prices")
    return prices


def update_json_prices(file_path: Path, prices: dict[str, float], is_public: bool) -> int:
    """Update prices in a stocks JSON file. Returns count of updated stocks."""
    with open(file_path) as f:
        data = json.load(f)

    updated = 0
    skipped_suspect = []
    if is_public:
        # Public file is a list of stock objects
        for stock in data:
            sym = stock.get('symbol')
            if sym in prices:
                # Same sanity guard as the daily pipeline: one garbage hourly
                # quote (0.0 halt, decimal shift) must not hit the live site.
                # Compare vs the price already in the JSON. A legit split-day
                # move gets skipped here for the day and heals on the next
                # daily-pipeline run (which checks FMP's restated previousClose).
                old = stock.get('price')
                new = prices[sym]
                if old and old > 0 and not (0.5 <= new / old <= 2.0):
                    skipped_suspect.append((sym, old, new))
                    continue
                stock['price'] = new
                updated += 1
        if skipped_suspect:
            print(f"  WARNING: skipped {len(skipped_suspect)} suspect quotes (>2x/<0.5x vs current): "
                  + ', '.join(f"{s}({o}->{n})" for s, o, n in skipped_suspect[:10]))
    else:
        # Premium file is a dict keyed by symbol
        # Premium doesn't have price field, but valuationData uses price-relative metrics
        # We don't update premium here — prices are only in public
        pass

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

    return updated


def main():
    if not FMP_API_KEY:
        print("ERROR: FMP_API_KEY not set")
        sys.exit(1)

    # Determine output directory
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    # Find the JSON files — either in a specified dir or the compass-site clone
    search_paths = [
        output_dir,
        Path('compass-site/src/data'),
        Path('src/data'),
        Path('data'),
    ]

    public_file = None
    for p in search_paths:
        if p and (p / 'stocks-public.json').exists():
            public_file = p / 'stocks-public.json'
            break

    if not public_file:
        print("ERROR: Cannot find stocks-public.json")
        sys.exit(1)

    print(f"Updating prices in {public_file.parent}")

    # Load current symbols
    with open(public_file) as f:
        stocks = json.load(f)
    symbols = [s['symbol'] for s in stocks]
    print(f"Fetching prices for {len(symbols)} stocks...")

    # Fetch bulk quotes
    prices = fetch_bulk_quotes(symbols)
    print(f"Got {len(prices)} prices from FMP")

    if len(prices) < len(symbols) * 0.5:
        print(f"WARNING: Only got {len(prices)}/{len(symbols)} prices — something may be wrong")

    # Update public JSON
    updated = update_json_prices(public_file, prices, is_public=True)
    print(f"Updated {updated} prices in stocks-public.json")

    # Also update the full stocks.json if it exists — same list format, so it
    # goes through the same guarded path (a suspect quote must be skipped in
    # BOTH files or they diverge until the next daily run).
    full_file = public_file.parent / 'stocks.json'
    if full_file.exists():
        count = update_json_prices(full_file, prices, is_public=True)
        print(f"Updated {count} prices in stocks.json")

    print("Done!")


if __name__ == '__main__':
    main()

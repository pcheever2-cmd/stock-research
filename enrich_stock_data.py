#!/usr/bin/env python3
"""
Comprehensive Stock Data Enrichment
====================================
Fills missing data in the database:
1. Forward P/E and PEG ratios (calculated from analyst estimates)
2. Financial Scores (Altman Z-Score, Piotroski Score) from FMP API
3. Missing company profiles (name, sector, industry) from FMP API
4. Market cap data from FMP API

Uses concurrent requests for efficiency.
"""

import sqlite3
import requests
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

# Rate limiting: FMP allows ~300 requests/minute on standard plans
MAX_WORKERS = 10  # Parallel threads
RATE_LIMIT_DELAY = 0.2  # 5 requests/second per thread = 50 total req/sec

# ============================================================================
# PART 1: Calculate Forward P/E and PEG from Analyst Estimates
# ============================================================================

def calculate_forward_pe_peg():
    """Calculate forward P/E and PEG ratios from analyst estimates."""
    print("\n" + "=" * 60)
    print("CALCULATING FORWARD P/E AND PEG RATIOS")
    print("=" * 60)

    # Check if backtest.db exists
    if not Path(BACKTEST_DB).exists():
        print("No backtest.db found - skipping forward P/E calculation")
        return 0

    conn_main = sqlite3.connect(DATABASE_NAME)
    conn_bt = sqlite3.connect(BACKTEST_DB)

    # Get current prices from main db
    prices = dict(conn_main.execute(
        "SELECT symbol, current_price FROM stock_consensus WHERE current_price > 0"
    ).fetchall())

    current_year = datetime.now().year
    next_year = current_year + 1

    # Get EPS estimates for current and next year to calculate growth
    estimates_query = """
        SELECT symbol, fiscal_year, eps_avg
        FROM analyst_estimates_snapshot
        WHERE fiscal_year IN (?, ?) AND eps_avg IS NOT NULL AND eps_avg > 0
    """

    # Build dict of {symbol: {year: eps}}
    eps_by_symbol = {}
    for row in conn_bt.execute(estimates_query, (current_year, next_year)).fetchall():
        symbol, year, eps = row
        if symbol not in eps_by_symbol:
            eps_by_symbol[symbol] = {}
        eps_by_symbol[symbol][year] = eps

    print(f"Found {len(prices)} stocks with prices")
    print(f"Found {len(eps_by_symbol)} stocks with EPS estimates")

    # Calculate forward P/E and PEG
    cur = conn_main.cursor()
    updated = 0
    peg_calculated = 0

    for symbol, eps_data in eps_by_symbol.items():
        if symbol not in prices:
            continue

        # Get next year EPS for forward P/E (or current year as fallback)
        eps_next = eps_data.get(next_year) or eps_data.get(current_year)
        eps_current = eps_data.get(current_year)

        if not eps_next or eps_next <= 0:
            continue

        price = prices[symbol]
        forward_pe = price / eps_next

        # Calculate EPS growth rate and PEG
        peg_ratio = None
        eps_growth_pct = None

        if eps_current and eps_current > 0 and eps_data.get(next_year):
            # EPS growth = (next_year - current_year) / current_year * 100
            eps_growth_pct = ((eps_data[next_year] - eps_current) / eps_current) * 100

            if eps_growth_pct > 0:
                # PEG = Forward P/E / EPS Growth %
                peg_ratio = forward_pe / eps_growth_pct
                peg_calculated += 1

        cur.execute("""
            UPDATE stock_consensus
            SET forward_pe = ?,
                peg_ratio = ?,
                projected_eps_next_year = ?,
                projected_eps_growth = ?
            WHERE symbol = ?
        """, (
            round(forward_pe, 2),
            round(peg_ratio, 2) if peg_ratio else None,
            eps_next,
            round(eps_growth_pct, 2) if eps_growth_pct else None,
            symbol
        ))

        if cur.rowcount > 0:
            updated += 1

    conn_main.commit()
    conn_main.close()
    conn_bt.close()

    print(f"Updated {updated} stocks with forward P/E")
    print(f"Calculated PEG for {peg_calculated} stocks with positive EPS growth")
    return updated


# ============================================================================
# PART 2: Fetch Financial Scores from FMP API
# ============================================================================

def fetch_financial_score(symbol: str) -> dict:
    """Fetch financial scores for a single symbol."""
    if not FMP_API_KEY:
        return {}

    url = f"https://financialmodelingprep.com/stable/financial-scores?symbol={symbol}&apikey={FMP_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                return {
                    'symbol': symbol,
                    'altman_z_score': item.get('altmanZScore'),
                    'piotroski_score': item.get('piotroskiScore'),
                }
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        pass

    return {}


def fetch_financial_scores_batch(symbols: list) -> list:
    """Fetch financial scores for multiple symbols using parallel requests."""
    print(f"\nFetching financial scores for {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_financial_score, sym): sym for sym in symbols}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result:
                results.append(result)

            if completed % 500 == 0:
                print(f"  Progress: {completed}/{len(symbols)} ({len(results)} found)")

    return results


def update_financial_scores():
    """Fetch and update financial scores for all stocks."""
    print("\n" + "=" * 60)
    print("FETCHING FINANCIAL SCORES (Altman Z, Piotroski)")
    print("=" * 60)

    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY set - skipping financial scores")
        return 0

    conn = sqlite3.connect(DATABASE_NAME)

    # Get symbols missing financial scores
    symbols = [r[0] for r in conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE altman_z_score IS NULL OR piotroski_score IS NULL
    """).fetchall()]

    print(f"Found {len(symbols)} stocks missing financial scores")

    if not symbols:
        conn.close()
        return 0

    # Fetch scores
    scores = fetch_financial_scores_batch(symbols)
    print(f"Retrieved financial scores for {len(scores)} stocks")

    # Update database
    cur = conn.cursor()
    updated = 0
    for score in scores:
        cur.execute("""
            UPDATE stock_consensus
            SET altman_z_score = COALESCE(?, altman_z_score),
                piotroski_score = COALESCE(?, piotroski_score)
            WHERE symbol = ?
        """, (score.get('altman_z_score'), score.get('piotroski_score'), score['symbol']))
        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"Updated {updated} stocks with financial scores")
    return updated


# ============================================================================
# PART 3: Fetch Missing Company Profiles
# ============================================================================

def fetch_profile(symbol: str) -> dict:
    """Fetch company profile for a single symbol."""
    if not FMP_API_KEY:
        return {}

    url = f"https://financialmodelingprep.com/stable/profile?symbol={symbol}&apikey={FMP_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                item = data[0]
                return {
                    'symbol': symbol,
                    'company_name': item.get('companyName'),
                    'sector': item.get('sector'),
                    'industry': item.get('industry'),
                    'description': (item.get('description', '') or '')[:500],
                    'market_cap': item.get('mktCap') or item.get('marketCap'),
                    'exchange': item.get('exchange'),
                }
        time.sleep(RATE_LIMIT_DELAY)
    except Exception as e:
        pass

    return {}


def fetch_profiles_batch(symbols: list) -> list:
    """Fetch profiles for multiple symbols using parallel requests."""
    print(f"\nFetching profiles for {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_profile, sym): sym for sym in symbols}

        completed = 0
        for future in as_completed(futures):
            completed += 1
            result = future.result()
            if result and result.get('company_name'):
                results.append(result)

            if completed % 500 == 0:
                print(f"  Progress: {completed}/{len(symbols)} ({len(results)} found)")

    return results


def fill_missing_profiles():
    """Fetch and update missing company profiles."""
    print("\n" + "=" * 60)
    print("FILLING MISSING COMPANY PROFILES")
    print("=" * 60)

    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY set - skipping profiles")
        return 0

    conn = sqlite3.connect(DATABASE_NAME)

    # Get symbols missing profile data
    symbols = [r[0] for r in conn.execute("""
        SELECT symbol FROM stock_consensus
        WHERE company_name IS NULL OR company_name = '' OR sector IS NULL
    """).fetchall()]

    print(f"Found {len(symbols)} stocks missing profile data")

    if not symbols:
        conn.close()
        return 0

    # Fetch profiles in batches
    profiles = fetch_profiles_batch(symbols)
    print(f"Retrieved profiles for {len(profiles)} stocks")

    # Update database
    cur = conn.cursor()
    updated = 0
    for profile in profiles:
        cur.execute("""
            UPDATE stock_consensus
            SET company_name = COALESCE(?, company_name),
                sector = COALESCE(?, sector),
                industry = COALESCE(?, industry),
                company_description = COALESCE(?, company_description),
                market_cap = COALESCE(?, market_cap),
                exchange = COALESCE(?, exchange)
            WHERE symbol = ?
        """, (
            profile.get('company_name'),
            profile.get('sector'),
            profile.get('industry'),
            profile.get('description'),
            profile.get('market_cap'),
            profile.get('exchange'),
            profile['symbol']
        ))
        if cur.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"Updated {updated} stocks with profile data")
    return updated


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def print_summary():
    """Print summary of data coverage."""
    conn = sqlite3.connect(DATABASE_NAME)

    total = conn.execute("SELECT COUNT(*) FROM stock_consensus").fetchone()[0]
    has_name = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE company_name IS NOT NULL AND company_name != ''").fetchone()[0]
    has_sector = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE sector IS NOT NULL").fetchone()[0]
    has_fwd_pe = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE forward_pe IS NOT NULL AND forward_pe > 0").fetchone()[0]
    has_peg = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE peg_ratio IS NOT NULL AND peg_ratio > 0").fetchone()[0]
    has_altman = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE altman_z_score IS NOT NULL").fetchone()[0]
    has_piotroski = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE piotroski_score IS NOT NULL").fetchone()[0]
    has_analysts = conn.execute("SELECT COUNT(*) FROM stock_consensus WHERE num_analysts >= 1").fetchone()[0]

    conn.close()

    print("\n" + "=" * 60)
    print("DATA COVERAGE SUMMARY")
    print("=" * 60)
    print(f"Total stocks:           {total:,}")
    print(f"With company name:      {has_name:,} ({100*has_name/total:.1f}%)")
    print(f"With sector:            {has_sector:,} ({100*has_sector/total:.1f}%)")
    print(f"With forward P/E:       {has_fwd_pe:,} ({100*has_fwd_pe/total:.1f}%)")
    print(f"With PEG ratio:         {has_peg:,} ({100*has_peg/total:.1f}%)")
    print(f"With Altman Z-Score:    {has_altman:,} ({100*has_altman/total:.1f}%)")
    print(f"With Piotroski Score:   {has_piotroski:,} ({100*has_piotroski/total:.1f}%)")
    print(f"With analyst coverage:  {has_analysts:,} ({100*has_analysts/total:.1f}%)")


def main():
    print("=" * 60)
    print("COMPREHENSIVE STOCK DATA ENRICHMENT")
    print("=" * 60)
    print(f"Database: {DATABASE_NAME}")
    print(f"API Key: {'Set' if FMP_API_KEY else 'NOT SET'}")

    # Ensure schema is up to date
    print("\nEnsuring database schema is current...")
    import setup_database
    setup_database.ensure_database()

    # Part 1: Calculate Forward P/E and PEG from existing data
    calculate_forward_pe_peg()

    # Part 2: Fetch financial scores
    update_financial_scores()

    # Part 3: Fill missing profiles
    fill_missing_profiles()

    # Print final summary
    print_summary()

    print("\n" + "=" * 60)
    print("DATA ENRICHMENT COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()

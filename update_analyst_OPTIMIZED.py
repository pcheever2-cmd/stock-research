#!/usr/bin/env python3
"""
OPTIMIZED Stock Data Updater
- Async/parallel API calls (6x faster)
- Bulk batch processing 
- Smart caching to avoid redundant calls
- Better error handling with retries
- Estimated time: ~5-10 min for 1000 stocks (vs 30-45 min before)
"""

import asyncio
import aiohttp
import sqlite3
import pandas as pd
import logging
import csv
import io
import time
import argparse
import random
import requests
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

# ==================== CONFIGURATION ====================
from config import FMP_API_KEY, DATABASE_NAME, CACHE_DIR

# Performance settings (used by the residual per-symbol phase + --compare)
BATCH_SIZE = 50  # Process in chunks
MAX_CONCURRENT = 40  # Parallel requests (FMP allows 3,000/min; old value of 10 strangled it)
CALLS_PER_MINUTE = 2500  # FMP limit is 3,000/min, leave headroom
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3

# Residual per-symbol rotation: stocks whose analyst-detail (price target / grades / news) is
# older than this are refreshed today (≈ oldest 1/ROTATION_DAYS of the universe per run). The
# bulk fields (profile/rating/metrics/ratios/estimates) refresh for the WHOLE universe daily.
ROTATION_DAYS = 7

# Endpoints - using /stable (current API)
BASE_URL = 'https://financialmodelingprep.com'
ENDPOINTS = {
    'profile': f'{BASE_URL}/stable/profile',
    'quote': f'{BASE_URL}/stable/quote',
    'target': f'{BASE_URL}/stable/price-target-consensus',
    'price_targets': f'{BASE_URL}/stable/price-target-news',  # Individual analyst price targets (FMP retired /stable/price-target → 404; price-target-news returns the same fields: publishedDate, analystCompany, priceTarget, priceWhenPosted)
    'estimates': f'{BASE_URL}/stable/analyst-estimates',
    'rating': f'{BASE_URL}/stable/ratings-snapshot',
    'grades': f'{BASE_URL}/stable/grades',
    'metrics': f'{BASE_URL}/stable/key-metrics-ttm',
    'ratios': f'{BASE_URL}/stable/ratios-ttm',
}

# Whole-universe BULK CSV endpoints (replace the per-symbol loop for these fields).
# profile-bulk is paginated by ?part=0..N. The *-bulk endpoints have a tighter rate cap than the
# per-symbol budget (observed 429s under rapid calls) → fetch them spaced, with backoff.
# (analyst-estimates is deliberately fetched per-symbol, not bulk — see fetch_bulk_universe.)
BULK_ENDPOINTS = {
    'profile': f'{BASE_URL}/stable/profile-bulk',          # paginated ?part=0..3
    'rating': f'{BASE_URL}/stable/rating-bulk',
    'metrics': f'{BASE_URL}/stable/key-metrics-ttm-bulk',
    'ratios': f'{BASE_URL}/stable/ratios-ttm-bulk',
}
PROFILE_BULK_PARTS = 4          # parts 0..3 (part=4 → 400)
BULK_CALL_SPACING_SEC = 5.0     # spacing between bulk CSV calls to respect the tight bulk rate cap
BULK_MAX_RETRIES = 6

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ==================== CACHE SYSTEM ====================
class DataCache:
    """Cache infrequently changing data (profiles, industries)"""
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_duration = timedelta(days=7)  # Profiles rarely change
    
    def get(self, key: str) -> Optional[dict]:
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        
        # Check if cache is still valid
        mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - mtime > self.cache_duration:
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def set(self, key: str, data: dict):
        cache_file = self.cache_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)

cache = DataCache(CACHE_DIR)

# ==================== RATE LIMITER ====================
class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, calls_per_minute: int):
        self.min_interval = 60.0 / calls_per_minute
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()

# ==================== ASYNC API FETCHER ====================
class AsyncFMPFetcher:
    """Async API fetcher with retry logic and rate limiting"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.rate_limiter = RateLimiter(CALLS_PER_MINUTE)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch(self, url: str, params: dict = None, retry_count: int = 0) -> Optional[dict]:
        """Fetch with retry logic - releases semaphore before retrying to avoid deadlock"""
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        should_retry = False
        await self.rate_limiter.acquire()
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429 and retry_count < MAX_RETRIES:
                        should_retry = True
                    else:
                        log.warning(f"API error {response.status} for {url}")
                        return None
            except asyncio.TimeoutError:
                if retry_count < MAX_RETRIES:
                    log.debug(f"Timeout on {url}, retrying...")
                    should_retry = True
                else:
                    return None
            except Exception as e:
                log.error(f"Fetch error for {url}: {e}")
                return None

        # Retry OUTSIDE the semaphore to avoid deadlock
        if should_retry:
            await asyncio.sleep(2 ** retry_count)
            return await self.fetch(url, params, retry_count + 1)
        return None
    
    async def fetch_stock_data(self, ticker: str) -> Dict:
        """Fetch all needed data for one stock in parallel"""
        tasks = {
            'quote': self.fetch(ENDPOINTS['quote'], {'symbol': ticker}),
            'target': self.fetch(ENDPOINTS['target'], {'symbol': ticker}),
            'price_targets': self.fetch(ENDPOINTS['price_targets'], {'symbol': ticker, 'limit': 10}),
            'estimates': self.fetch(ENDPOINTS['estimates'], {'symbol': ticker, 'period': 'annual'}),
            'rating': self.fetch(ENDPOINTS['rating'], {'symbol': ticker}),
            'grades': self.fetch(ENDPOINTS['grades'], {'symbol': ticker, 'limit': 5}),
            'metrics': self.fetch(ENDPOINTS['metrics'], {'symbol': ticker}),
            'ratios': self.fetch(ENDPOINTS['ratios'], {'symbol': ticker}),
        }

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

# ==================== PROFILE FETCHER ====================
async def fetch_profile(fetcher: AsyncFMPFetcher, ticker: str) -> dict:
    """Fetch individual profile (bulk endpoint is deprecated)"""
    profile_data = await fetcher.fetch(ENDPOINTS['profile'], {'symbol': ticker})
    if isinstance(profile_data, list) and len(profile_data) > 0:
        return profile_data[0]
    return {}

# ==================== DATA PROCESSING ====================
def safe_get(data, *keys, default=None):
    """Safely extract nested dict values with optional default"""
    if data is None:
        return default
    if isinstance(data, Exception):
        return default
    if isinstance(data, list):
        data = data[0] if data else {}
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
            if data is None:
                return default
        else:
            return default
    return data if data is not None else default

def calculate_growth_metrics(estimates_data: list) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """Extract growth metrics and analyst count"""
    if not estimates_data or len(estimates_data) < 2:
        return None, None, None, 0
    
    next_year = estimates_data[0]
    current_year = estimates_data[1]
    
    # Try multiple field names (FMP API inconsistencies)
    eps_curr = safe_get(current_year, 'estimatedEpsAvg', default=0) or safe_get(current_year, 'epsAvg', default=0) or 0
    eps_next = safe_get(next_year, 'estimatedEpsAvg', default=0) or safe_get(next_year, 'epsAvg', default=0) or 0
    
    rev_curr = safe_get(current_year, 'estimatedRevenueAvg', default=0) or safe_get(current_year, 'revenueAvg', default=0) or 0
    rev_next = safe_get(next_year, 'estimatedRevenueAvg', default=0) or safe_get(next_year, 'revenueAvg', default=0) or 0
    
    ebitda_curr = safe_get(current_year, 'estimatedEbitdaAvg', default=0) or safe_get(current_year, 'ebitdaAvg', default=0) or 0
    ebitda_next = safe_get(next_year, 'estimatedEbitdaAvg', default=0) or safe_get(next_year, 'ebitdaAvg', default=0) or 0
    
    # Calculate growth rates
    eps_growth = ((eps_next / eps_curr - 1) * 100) if eps_curr > 0 and eps_next > 0 else None
    rev_growth = ((rev_next / rev_curr - 1) * 100) if rev_curr > 0 and rev_next > 0 else None
    ebitda_growth = ((ebitda_next / ebitda_curr - 1) * 100) if ebitda_curr > 0 and ebitda_next > 0 else None
    
    # Get analyst count from most recent year
    num_analysts = 0
    for year in estimates_data[:3]:
        count = (safe_get(year, 'numAnalystsRevenue', default=0) or
                safe_get(year, 'numAnalystsEps', default=0) or
                safe_get(year, 'numberOfAnalysts', default=0) or
                safe_get(year, 'numAnalysts', default=0) or 0)
        num_analysts = max(num_analysts, count)
    
    return rev_growth, eps_growth, ebitda_growth, max(num_analysts, 1)

def format_recent_ratings(grades_data: list) -> str:
    """Format analyst rating changes"""
    if not grades_data:
        return "No recent rating changes"

    lines = []
    for g in grades_data[:5]:
        if not isinstance(g, dict):
            continue
        action = safe_get(g, 'action', default='')
        action_lower = action.lower() if action else ''
        emoji = "⬆️" if 'upgrade' in action_lower else "⬇️" if 'downgrade' in action_lower else "◾"
        date_val = safe_get(g, 'date', default='')
        date_str = str(date_val)[:10] if date_val else ''
        company = safe_get(g, 'gradingCompany', default='Unknown')
        grade = safe_get(g, 'newGrade', default='N/A')
        lines.append(f"{emoji} {date_str}: {company} → {grade}")

    return "\n".join(lines) if lines else "No recent rating changes"


def format_analyst_price_targets(price_targets_data: list) -> str:
    """Format individual analyst price targets (no analyst names, just firm)"""
    if not price_targets_data:
        return ""

    lines = []
    for pt in price_targets_data[:10]:  # Keep up to 10 most recent
        if not isinstance(pt, dict):
            continue
        date_val = safe_get(pt, 'publishedDate', default='')
        date_str = str(date_val)[:10] if date_val else ''
        firm = safe_get(pt, 'analystCompany', default='Unknown')
        target = safe_get(pt, 'priceTarget')
        price_when_posted = safe_get(pt, 'priceWhenPosted')

        if not target or not firm:
            continue

        # Format: "2026-01-15|Goldman Sachs|200|185" (date|firm|target|priceWhenPosted)
        line = f"{date_str}|{firm}|{target}|{price_when_posted or ''}"
        lines.append(line)

    return "\n".join(lines) if lines else ""

# ==================== ROW ASSEMBLY (shared by per-symbol + bulk paths) ====================
def assemble_row(ticker: str, profile_data, quote_data, target_data, estimates,
                 rating_data, grades, price_targets_data, metrics_data, ratios_data,
                 require_quote: bool = True, require_target: bool = True) -> Optional[Dict]:
    """Build the stock_consensus column dict from raw endpoint responses.

    This is the SINGLE source of field-extraction truth — both the per-symbol path
    (process_ticker) and the bulk path feed it, so their outputs are identical by
    construction (only the data SOURCE differs). The require_* flags reproduce the
    original all-or-nothing validation gates; the bulk path keeps them True so the
    exported universe matches today's (a row needs fresh price + market cap + a price
    target — supplied for non-rotation stocks from their existing DB target).
    """
    try:
        # Extract profile safely
        profile = None
        if isinstance(profile_data, list) and profile_data:
            profile = profile_data[0]
        elif isinstance(profile_data, dict):
            profile = profile_data
        profile = profile or {}

        # Extract quote data with safety checks
        quote = None
        if isinstance(quote_data, list) and quote_data:
            quote = quote_data[0]
        elif isinstance(quote_data, dict):
            quote = quote_data
        quote = quote or {}

        current_price = safe_get(quote, 'price')
        market_cap = safe_get(quote, 'marketCap')

        if require_quote and (not current_price or not market_cap):
            log.debug(f"  {ticker}: Missing price or market cap")
            return None

        # Handle target as list or dict
        target = None
        if isinstance(target_data, list) and target_data:
            target = target_data[0]
        elif isinstance(target_data, dict):
            target = target_data
        target = target or {}

        mean_target = safe_get(target, 'targetConsensus') or safe_get(target, 'targetMeanPrice')
        if require_target and not mean_target:
            log.debug(f"  {ticker}: No price target")
            return None

        upside = ((mean_target - current_price) / current_price * 100
                  if mean_target and current_price else None)

        # Growth metrics
        if isinstance(estimates, Exception) or not estimates:
            estimates = []
        rev_growth, eps_growth, ebitda_growth, num_analysts = calculate_growth_metrics(estimates)

        # Other metrics - rating
        if isinstance(rating_data, Exception):
            rating_data = None
        rating = None
        if isinstance(rating_data, list) and rating_data:
            rating = rating_data[0]
        elif isinstance(rating_data, dict):
            rating = rating_data
        rating = rating or {}
        recommendation = safe_get(rating, 'rating', default='N/A')

        # Grades
        if isinstance(grades, Exception) or not grades:
            grades = []
        recent_ratings = format_recent_ratings(grades)

        # Individual analyst price targets
        if isinstance(price_targets_data, Exception) or not price_targets_data:
            price_targets_data = []
        analyst_price_targets = format_analyst_price_targets(price_targets_data)

        # Key metrics
        if isinstance(metrics_data, Exception):
            metrics_data = None
        metrics = None
        if isinstance(metrics_data, list) and metrics_data:
            metrics = metrics_data[0]
        elif isinstance(metrics_data, dict):
            metrics = metrics_data
        metrics = metrics or {}
        ev_ebitda = safe_get(metrics, 'evToEBITDATTM')
        enterprise_value = safe_get(metrics, 'enterpriseValueTTM')

        # Debt/EBITDA from key-metrics-ttm (net debt / EBITDA)
        debt_ebitda = safe_get(metrics, 'netDebtToEBITDATTM')
        if debt_ebitda is not None and debt_ebitda < 0:
            debt_ebitda = None  # Negative means net cash position, not useful as debt metric

        # OCF/EV from key-metrics-ttm (inverse of EV/OCF)
        ev_to_ocf = safe_get(metrics, 'evToOperatingCashFlowTTM')
        ocf_ev = round(1.0 / ev_to_ocf, 4) if ev_to_ocf and ev_to_ocf > 0 else None

        # Ratios
        if isinstance(ratios_data, Exception):
            ratios_data = None
        ratios = None
        if isinstance(ratios_data, list) and ratios_data:
            ratios = ratios_data[0]
        elif isinstance(ratios_data, dict):
            ratios = ratios_data
        ratios = ratios or {}
        ebitda = safe_get(ratios, 'ebitdaTTM')

        # Company name and description from profile
        company_name = safe_get(profile, 'companyName') or ticker
        company_description = safe_get(profile, 'description')

        # Industry and sector from profile (keep separate)
        industry = safe_get(profile, 'industry') or 'N/A'
        sector = safe_get(profile, 'sector') or 'N/A'

        # Market cap category (market_cap may be None when require_quote=False)
        cap_category = None
        if market_cap:
            cap_category = (
                'Micro/Penny Cap (<$300M)' if market_cap < 300e6 else
                'Small Cap ($300M–$2B)' if market_cap < 2e9 else
                'Mid Cap ($2B–$10B)' if market_cap < 10e9 else
                'Large Cap (>$10B)'
            )

        return {
            'symbol': ticker,
            'current_price': current_price,
            'avg_price_target': mean_target,
            'median_price_target': safe_get(target, 'targetMedian'),
            'min_price_target': safe_get(target, 'targetLow'),
            'max_price_target': safe_get(target, 'targetHigh'),
            'upside_percent': upside,
            'num_analysts': num_analysts,
            'recommendation': recommendation,
            'consensus_rating': recommendation,
            'cap_category': cap_category,
            'industry': industry,
            'sector': sector,
            'company_name': company_name,
            'company_description': company_description,
            'recent_ratings': recent_ratings,
            'analyst_price_targets': analyst_price_targets,
            'enterprise_value': enterprise_value,
            'ebitda': ebitda,
            'ev_ebitda': ev_ebitda,
            'debt_ebitda': debt_ebitda,
            'ocf_ev': ocf_ev,
            'projected_revenue_growth': rev_growth,
            'projected_eps_growth': eps_growth,
            'projected_ebitda_growth': ebitda_growth,
            'last_updated': datetime.utcnow().isoformat(),
        }

    except Exception as e:
        log.error(f"Error assembling {ticker}: {e}")
        import traceback
        log.debug(traceback.format_exc())
        return None


# ==================== PER-SYMBOL PATH (legacy; used by --compare) ====================
async def process_ticker(ticker: str, fetcher: AsyncFMPFetcher) -> Optional[Dict]:
    """Process one ticker via per-symbol API calls, then assemble. Kept as the parity
    reference for --compare (and as a fallback). The daily pipeline now uses run_bulk()."""
    try:
        profile_task = fetcher.fetch(ENDPOINTS['profile'], {'symbol': ticker})
        data_task = fetcher.fetch_stock_data(ticker)
        profile_data, data = await asyncio.gather(profile_task, data_task)

        if not data or not isinstance(data, dict):
            log.debug(f"  {ticker}: No data returned")
            return None

        return assemble_row(
            ticker, profile_data, data.get('quote'), data.get('target'),
            data.get('estimates'), data.get('rating'), data.get('grades'),
            data.get('price_targets'), data.get('metrics'), data.get('ratios'),
        )
    except Exception as e:
        log.error(f"Error processing {ticker}: {e}")
        return None

def save_batch_to_db(results: List[Dict]):
    """Bulk upsert to database - preserves columns not in the update (e.g. scores).

    Rows may be HETEROGENEOUS (the bulk path pops detail columns for non-rotation stocks, so
    different rows carry different key sets). We therefore group rows by their exact key set and
    run one INSERT…ON CONFLICT per group, with each statement touching ONLY that group's columns —
    so a column a row doesn't provide is never inserted/clobbered. (Deriving columns from
    results[0] alone would silently drop the rotation columns for every other row.)"""
    if not results:
        return

    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    # Ensure every column present across ALL rows exists in the table (schema-drift safe)
    existing_cols = {row[1] for row in cur.execute("PRAGMA table_info(stock_consensus)").fetchall()}
    all_cols = set()
    for r in results:
        all_cols.update(r.keys())
    for col in sorted(all_cols):
        if col not in existing_cols:
            sample = next((r[col] for r in results if r.get(col) is not None), None)
            col_type = 'TEXT' if isinstance(sample, str) else 'REAL'
            cur.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col} {col_type}")
            log.info(f"  Added missing column: {col} ({col_type})")

    # Group by key set so each statement's column list matches every row it writes
    from collections import defaultdict
    groups = defaultdict(list)
    for row in results:
        groups[tuple(sorted(row.keys()))].append(row)

    for cols_tuple, group in groups.items():
        columns = list(cols_tuple)
        placeholders = ', '.join(['?'] * len(columns))
        cols_str = ', '.join(columns)
        # ON CONFLICT updates only the columns this group provides → untouched columns preserved
        update_cols = [c for c in columns if c != 'symbol']
        update_clause = ', '.join(f'{c} = excluded.{c}' for c in update_cols)
        sql = (f"INSERT INTO stock_consensus ({cols_str}) VALUES ({placeholders}) "
               f"ON CONFLICT(symbol) DO UPDATE SET {update_clause}")
        cur.executemany(sql, [tuple(row.get(col) for col in columns) for row in group])

    conn.commit()
    conn.close()
    log.info(f"  ✓ Saved {len(results)} stocks to database ({len(groups)} column-set group(s))")

# ==================== BULK CSV FETCH (whole-universe) ====================
def _to_float(v):
    """Coerce a CSV string cell to float, or None. Empty/invalid → None (never '' into REAL)."""
    if v is None:
        return None
    s = str(v).strip()
    if s == '' or s.lower() in ('na', 'n/a', 'none', 'null'):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _clean_row(row: dict) -> dict:
    """Map empty-string CSV cells to None so downstream safe_get/derivations behave like JSON."""
    return {k: (None if (isinstance(v, str) and v.strip() == '') else v) for k, v in row.items()}


def _fetch_bulk_csv(url: str, params: dict) -> List[dict]:
    """Download one bulk CSV endpoint → list of cleaned dict rows, with backoff for 429s.

    The *-bulk endpoints have a tighter rate cap than the per-symbol budget, so we space and
    retry. Returns [] on persistent failure (caller degrades that field to existing/None)."""
    p = dict(params)
    p['apikey'] = FMP_API_KEY
    for attempt in range(BULK_MAX_RETRIES):
        try:
            r = requests.get(url, params=p, timeout=180)
            if r.status_code == 200:
                r.encoding = 'utf-8'  # FMP bulk CSV is UTF-8; requests else guesses Latin-1 → mojibake
                if not r.text.strip():
                    return []
                return [_clean_row(row) for row in csv.DictReader(io.StringIO(r.text))]
            if r.status_code == 429:
                wait = min(90.0, 30.0 * (attempt + 1))
                log.warning(f"  bulk 429 on {url} (attempt {attempt+1}); backing off {wait:.0f}s")
                time.sleep(wait)
                continue
            log.warning(f"  bulk {r.status_code} on {url}: {r.text[:120]}")
            return []
        except Exception as e:
            log.warning(f"  bulk fetch error on {url}: {e}")
            time.sleep(BULK_CALL_SPACING_SEC * (2 ** attempt))
    log.error(f"  bulk fetch FAILED after {BULK_MAX_RETRIES} attempts: {url}")
    return []


def fetch_bulk_universe() -> Dict[str, Dict[str, dict]]:
    """Fetch the whole-universe bulk feeds that reproduce per-symbol values exactly. Returns dicts
    keyed by symbol: {'profile': {sym: row}, 'rating': {...}, 'metrics': {...}, 'ratios': {...}}.
    Calls are spaced to respect the tight bulk rate cap.

    NOTE: analyst-estimates is intentionally NOT bulk-fetched. estimates-bulk filters by CALENDAR
    year, which can't reconstruct the per-symbol descending array for non-December fiscal-year-end
    names (e.g. YEXT) and is severely rate-limited (~1 call/min). Estimates are fetched per-symbol
    in the residual rotation instead — byte-identical to the legacy path."""
    out = {'profile': {}, 'rating': {}, 'metrics': {}, 'ratios': {}}

    # profile-bulk is paginated by part=0..N
    for part in range(PROFILE_BULK_PARTS):
        rows = _fetch_bulk_csv(BULK_ENDPOINTS['profile'], {'part': str(part)})
        for row in rows:
            sym = row.get('symbol')
            if sym:
                out['profile'][sym] = row
        log.info(f"  profile-bulk part {part}: {len(rows)} rows (total {len(out['profile'])})")
        time.sleep(BULK_CALL_SPACING_SEC)

    for key in ('rating', 'metrics', 'ratios'):
        rows = _fetch_bulk_csv(BULK_ENDPOINTS[key], {})
        for row in rows:
            sym = row.get('symbol')
            if sym:
                out[key][sym] = row
        log.info(f"  {key}-bulk: {len(out[key])} symbols")
        time.sleep(BULK_CALL_SPACING_SEC)

    return out


def _is_tradable(profile: dict) -> bool:
    """Mirror the spirit of the old validation gate: real, actively-traded equities only."""
    def truthy(v):
        return str(v).strip().lower() in ('true', '1', 'yes')
    return (truthy(profile.get('isActivelyTrading'))
            and not truthy(profile.get('isEtf'))
            and not truthy(profile.get('isFund')))


def _load_existing() -> Dict[str, dict]:
    """Pull the columns the bulk path must read (current_price, freshness, existing targets/details)."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT symbol, current_price, price_updated_at, avg_price_target, median_price_target,
               min_price_target, max_price_target, analyst_detail_updated_at
        FROM stock_consensus
    """).fetchall()
    conn.close()
    return {r['symbol']: dict(r) for r in rows}


def _bulk_quote(profile: dict, existing: dict) -> dict:
    """Synthesize the 'quote' shape: price from STEP 0 (DB) ONLY if fresh today; market cap from
    profile-bulk. Stale/absent price → None so upside isn't computed off an old price (I4)."""
    price = None
    if existing:
        pu = existing.get('price_updated_at')
        if pu and str(pu)[:10] == date.today().isoformat():
            price = existing.get('current_price')
    return {'price': price, 'marketCap': _to_float(profile.get('marketCap'))}


def _coerce_metrics(row: dict) -> dict:
    if not row:
        return {}
    rr = dict(row)
    for f in ('evToEBITDATTM', 'enterpriseValueTTM', 'netDebtToEBITDATTM', 'evToOperatingCashFlowTTM'):
        if f in rr:
            rr[f] = _to_float(rr[f])
    return rr


def _coerce_ratios(row: dict) -> dict:
    if not row:
        return {}
    rr = dict(row)
    if 'ebitdaTTM' in rr:
        rr['ebitdaTTM'] = _to_float(rr['ebitdaTTM'])
    return rr


# ==================== BULK ORCHESTRATOR ====================
async def run_bulk():
    """Daily STEP 1: whole-universe bulk fields + weekly per-symbol rotation for price
    target / grades / news. Replaces the per-symbol-for-everything loop (~87 min → ~5 min)."""
    log.info("=" * 60)
    log.info("STEP 1 (BULK): analyst + price-target update")
    log.info("=" * 60)

    bulk = fetch_bulk_universe()
    profiles = bulk['profile']
    if not profiles:
        log.error("profile-bulk returned nothing — aborting (would mass-null the universe)")
        return

    # Universe = symbols we track that are tradable equities in profile-bulk
    conn = sqlite3.connect(DATABASE_NAME)
    tracked = [r[0] for r in conn.execute("SELECT symbol FROM stock_consensus").fetchall()]
    conn.close()
    existing = _load_existing()
    universe = [s for s in tracked if s in profiles and _is_tradable(profiles[s])]
    # M2: surface tracked symbols that bulk no longer covers (foreign/ADR/delisted/non-tradable) —
    # they keep their existing columns (not nulled) but no longer refresh via STEP 1.
    not_in_bulk = [s for s in tracked if s not in profiles]
    log.info(f"Universe: {len(universe)} tradable symbols (of {len(tracked)} tracked; "
             f"{len(not_in_bulk)} not in profile-bulk → not refreshed)")

    # Rotation slice: detail NULL or older than ROTATION_DAYS (oldest first)
    cutoff = (datetime.utcnow() - timedelta(days=ROTATION_DAYS)).isoformat()
    def detail_age_key(s):
        return existing.get(s, {}).get('analyst_detail_updated_at') or ''
    rotation = sorted(
        [s for s in universe
         if not existing.get(s, {}).get('analyst_detail_updated_at')
         or existing[s]['analyst_detail_updated_at'] < cutoff],
        key=detail_age_key
    )
    rotation_set = set(rotation)
    log.info(f"Rotation slice (detail > {ROTATION_DAYS}d old): {len(rotation)} symbols")

    # Residual per-symbol fetch (target / news / grades / estimates) for the rotation slice only.
    # estimates is per-symbol (not bulk) so num_analysts + projected growth stay byte-identical to
    # the legacy path (estimates-bulk's calendar-year filter misaligns non-Dec fiscal years).
    fresh_target, fresh_news, fresh_grades, fresh_est = {}, {}, {}, {}
    if rotation:
        async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
            async def fetch_detail(sym):
                t, n, g, e = await asyncio.gather(
                    fetcher.fetch(ENDPOINTS['target'], {'symbol': sym}),
                    fetcher.fetch(ENDPOINTS['price_targets'], {'symbol': sym, 'limit': 10}),
                    fetcher.fetch(ENDPOINTS['grades'], {'symbol': sym, 'limit': 5}),
                    fetcher.fetch(ENDPOINTS['estimates'], {'symbol': sym, 'period': 'annual'}),
                )
                return sym, t, n, g, e
            for i in range(0, len(rotation), BATCH_SIZE):
                chunk = rotation[i:i + BATCH_SIZE]
                for sym, t, n, g, e in await asyncio.gather(*[fetch_detail(s) for s in chunk]):
                    fresh_target[sym] = t if not isinstance(t, Exception) else None
                    fresh_news[sym] = n if not isinstance(n, Exception) else None
                    fresh_grades[sym] = g if not isinstance(g, Exception) else None
                    fresh_est[sym] = e if not isinstance(e, Exception) else None
                log.info(f"  residual {min(i+BATCH_SIZE, len(rotation))}/{len(rotation)}")

    now_iso = datetime.utcnow().isoformat()
    rows, rotation_attempted, stale_price = [], [], 0
    for sym in universe:
        profile = profiles[sym]
        ex = existing.get(sym, {})
        quote = _bulk_quote(profile, ex)
        # I1: if STEP 0 didn't price this symbol today, still refresh the bulk fields (profile,
        # rating, metrics, estimates) — just don't gate on / write current_price + upside.
        price_fresh = quote.get('price') is not None
        if not price_fresh:
            stale_price += 1
        rating = bulk['rating'].get(sym) or {}
        metrics = _coerce_metrics(bulk['metrics'].get(sym))
        ratios = _coerce_ratios(bulk['ratios'].get(sym))

        in_rotation = sym in rotation_set
        if in_rotation:
            rotation_attempted.append(sym)
            target = fresh_target.get(sym)
            grades = fresh_grades.get(sym)
            price_targets = fresh_news.get(sym)
            estimates = fresh_est.get(sym)
        else:
            # Non-rotation: synthesize target from existing DB values so the gate passes and
            # upside recomputes off today's price. grades/news/estimates aren't refreshed → the
            # derived columns (recent_ratings, analyst_price_targets, num_analysts, growth) are
            # popped below so existing values persist.
            target = {
                'targetConsensus': ex.get('avg_price_target'),
                'targetMedian': ex.get('median_price_target'),
                'targetLow': ex.get('min_price_target'),
                'targetHigh': ex.get('max_price_target'),
            }
            grades = None
            price_targets = None
            estimates = None

        row = assemble_row(sym, profile, quote, target, estimates, rating, grades,
                           price_targets, metrics, ratios,
                           require_quote=price_fresh, require_target=True)
        if row is None:
            continue
        if not price_fresh:
            # No fresh price → don't clobber the existing current_price/upside with None
            row.pop('current_price', None)
            row.pop('upside_percent', None)
        if in_rotation:
            row['analyst_detail_updated_at'] = now_iso
        else:
            # Don't overwrite the existing detail-derived columns with empty re-derivations
            for k in ('recent_ratings', 'analyst_price_targets',
                      'num_analysts', 'projected_revenue_growth',
                      'projected_eps_growth', 'projected_ebitda_growth'):
                row.pop(k, None)
        rows.append(row)

    save_batch_to_db(rows)

    # I1: stamp analyst_detail_updated_at on EVERY rotation attempt (incl. those that produced no
    # row, e.g. no price target) so no-coverage names rotate weekly instead of retrying daily.
    written = {r['symbol'] for r in rows if 'analyst_detail_updated_at' in r}
    unstamped = [s for s in rotation_attempted if s not in written]
    if unstamped:
        conn = sqlite3.connect(DATABASE_NAME)
        conn.executemany(
            "UPDATE stock_consensus SET analyst_detail_updated_at = ? WHERE symbol = ?",
            [(now_iso, s) for s in unstamped],
        )
        conn.commit()
        conn.close()
        log.info(f"  Stamped {len(unstamped)} rotation symbols with no new row (weekly retry)")

    log.info("\n" + "=" * 60)
    log.info(f"✓ STEP 1 bulk complete: {len(rows)} rows written, "
             f"{len(rotation)} detail-refreshed, {stale_price} kept existing price (STEP 0 miss)")
    log.info("=" * 60)


# ==================== PARITY COMPARE (--compare) ====================
def _select_compare_symbols(per_bucket: int) -> List[str]:
    """Stratified sample for parity testing: oversample the gate-boundary cohorts that random
    sampling would miss (no-target, micro-cap, ADR/foreign, ETF/fund) plus a random tranche."""
    conn = sqlite3.connect(DATABASE_NAME)
    conn.row_factory = sqlite3.Row

    def q(sql):
        return [r[0] for r in conn.execute(sql).fetchall()]

    buckets = {
        'null_target': q("SELECT symbol FROM stock_consensus WHERE avg_price_target IS NULL"),
        'micro_cap':   q("SELECT symbol FROM stock_consensus WHERE market_cap > 0 AND market_cap < 300e6"),
        'large_cap':   q("SELECT symbol FROM stock_consensus WHERE market_cap >= 10e9"),
        'has_target':  q("SELECT symbol FROM stock_consensus WHERE avg_price_target IS NOT NULL"),
        'random':      q("SELECT symbol FROM stock_consensus"),
    }
    conn.close()
    chosen = []
    rng = random.Random(42)
    for name, pool in buckets.items():
        if not pool:
            continue
        rng.shuffle(pool)
        chosen.extend(pool[:per_bucket])
    # de-dup, preserve order
    seen, out = set(), []
    for s in chosen:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


async def run_compare(per_bucket: int):
    """Prove the BULK data source yields the same assemble_row() output as the per-symbol source.
    For each sampled symbol: OLD = per-symbol fetch→assemble; NEW = bulk feeds (+ same per-symbol
    target/grades/news, which have no bulk) → assemble. Diff field-by-field.

    current_price/upside legitimately differ by source/timing → in compare we feed the NEW path a
    freshly-fetched quote so those match and we isolate the real question (profile/rating/metrics/
    ratios/estimates bulk parity)."""
    symbols = _select_compare_symbols(per_bucket)
    log.info(f"PARITY: comparing {len(symbols)} symbols (stratified)")

    bulk = fetch_bulk_universe()

    SOURCE_FIELDS = [  # fields whose SOURCE changes (bulk vs per-symbol) — must match exactly
        'company_name', 'company_description', 'sector', 'industry', 'cap_category',
        'num_analysts', 'recommendation', 'consensus_rating', 'ev_ebitda', 'enterprise_value',
        'debt_ebitda', 'ocf_ev', 'ebitda',
        'projected_revenue_growth', 'projected_eps_growth', 'projected_ebitda_growth',
    ]
    GROWTH = {'projected_revenue_growth', 'projected_eps_growth', 'projected_ebitda_growth'}
    # These embed CURRENT PRICE (EV = mktcap+debt-cash). Bulk CSV is a daily snapshot, per-symbol
    # is live → they drift by intraday price movement. A ~1% band is immaterial and expected.
    PRICE_DEPENDENT = {'ev_ebitda', 'enterprise_value', 'ocf_ev'}

    def approx_equal(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        if isinstance(a, float) or isinstance(b, float):
            return abs(float(a) - float(b)) <= max(1e-6, abs(float(a)) * 1e-4)
        return a == b

    def field_equal(f, a, b):
        if f in GROWTH and (a is not None or b is not None):
            if a is None or b is None:
                return False
            return abs(float(a) - float(b)) <= 0.1          # ≤0.1 percentage point
        if f in PRICE_DEPENDENT and a is not None and b is not None:
            return abs(float(a) - float(b)) <= max(1e-6, abs(float(a)) * 0.02)  # ≤2% rel (snapshot drift)
        return approx_equal(a, b)

    mismatches, old_none, new_none, both_none, ok = [], 0, 0, 0, 0
    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        for i in range(0, len(symbols), BATCH_SIZE):
            chunk = symbols[i:i + BATCH_SIZE]

            async def build_pair(sym):
                # OLD: full per-symbol
                old = await process_ticker(sym, fetcher)
                # NEW: bulk profile/rating/metrics/ratios + per-symbol target/news/grades/estimates
                # + fresh quote (so current_price matches and we isolate bulk-field parity)
                prof = bulk['profile'].get(sym)
                if prof is None:
                    return sym, old, None
                quote, target, news, grades, estimates = await asyncio.gather(
                    fetcher.fetch(ENDPOINTS['quote'], {'symbol': sym}),
                    fetcher.fetch(ENDPOINTS['target'], {'symbol': sym}),
                    fetcher.fetch(ENDPOINTS['price_targets'], {'symbol': sym, 'limit': 10}),
                    fetcher.fetch(ENDPOINTS['grades'], {'symbol': sym, 'limit': 5}),
                    fetcher.fetch(ENDPOINTS['estimates'], {'symbol': sym, 'period': 'annual'}),
                )
                new = assemble_row(
                    sym, prof, quote, target, estimates,
                    bulk['rating'].get(sym) or {}, grades, news,
                    _coerce_metrics(bulk['metrics'].get(sym)),
                    _coerce_ratios(bulk['ratios'].get(sym)),
                )
                return sym, old, new

            for sym, old, new in await asyncio.gather(*[build_pair(s) for s in chunk]):
                if old is None and new is None:
                    both_none += 1
                    continue
                if old is None:
                    old_none += 1
                    continue
                if new is None:
                    new_none += 1
                    mismatches.append((sym, 'NEW_DROPPED', None, None))
                    continue
                row_ok = True
                for f in SOURCE_FIELDS:
                    a, b = old.get(f), new.get(f)
                    if not field_equal(f, a, b):
                        mismatches.append((sym, f, a, b))
                        row_ok = False
                ok += row_ok

    log.info("\n" + "=" * 60)
    log.info(f"PARITY RESULT: {ok} fully-matched | {len(mismatches)} field mismatches | "
             f"old_dropped&matched={old_none} | new_dropped(old kept)={new_none} | both_dropped={both_none}")
    if mismatches:
        log.warning("First 40 mismatches (symbol, field, old, new):")
        for m in mismatches[:40]:
            log.warning(f"  {m[0]:8} {m[1]:26} old={m[2]!r:30} new={m[3]!r}")
    else:
        log.info("✓ All source-changed fields match between bulk and per-symbol.")
    return mismatches


# ==================== ENTRYPOINT ====================
async def main():
    parser = argparse.ArgumentParser(description="STEP 1 analyst/target updater (bulk-first)")
    parser.add_argument('--compare', type=int, metavar='N',
                        help="Parity mode: compare bulk vs per-symbol on ~N symbols per stratum (no DB writes)")
    parser.add_argument('--legacy', action='store_true',
                        help="Run the old per-symbol-for-everything path (fallback)")
    args = parser.parse_args()

    if args.compare:
        await run_compare(args.compare)
    elif args.legacy:
        await run_legacy()
    else:
        await run_bulk()


async def run_legacy():
    """Original per-symbol path (kept as a fallback)."""
    conn = sqlite3.connect(DATABASE_NAME)
    query = """
        SELECT symbol FROM stock_consensus
        WHERE DATE(last_updated) < DATE('now') OR last_updated IS NULL OR avg_price_target IS NULL
        ORDER BY last_updated ASC NULLS LAST
    """
    tickers = pd.read_sql_query(query, conn)['symbol'].dropna().tolist()
    conn.close()
    if not tickers:
        log.info("✓ All stocks are up to date!")
        return
    log.info(f"[legacy] Found {len(tickers)} stocks to update")
    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            results = await asyncio.gather(*[process_ticker(t, fetcher) for t in batch])
            save_batch_to_db([r for r in results if r is not None])
            log.info(f"[legacy] {min(i+BATCH_SIZE, len(tickers))}/{len(tickers)}")


if __name__ == "__main__":
    asyncio.run(main())
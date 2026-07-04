#!/usr/bin/env python3
"""
Historical Data Collector for Backtesting
Collects price + quarterly financial data from FMP
Stores in separate backtest.db to keep daily pipeline fast

Usage:
    python collect_historical_data.py                     # Collect 5 years (default)
    python collect_historical_data.py --extended          # Collect 30 years (1995-2025)
    python collect_historical_data.py --prices-only       # Just prices
    python collect_historical_data.py --financials-only   # Just financial statements
    python collect_historical_data.py --symbols AAPL,MSFT # Specific symbols
    python collect_historical_data.py --status            # Show progress
    python collect_historical_data.py --retry-failed      # Retry failed symbols
"""

import asyncio
import aiohttp
import sqlite3
import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    FMP_API_KEY, DATABASE_NAME, BACKTEST_DB,
    HISTORICAL_BATCH_SIZE, HISTORICAL_MAX_CONCURRENT,
    HISTORICAL_CALLS_PER_MINUTE, HISTORICAL_REQUEST_TIMEOUT,
    HISTORICAL_ENDPOINTS,
)
from setup_database import setup_backtest_tables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

MAX_RETRIES = 3

# Extended mode settings (30 years: 1995-2025)
EXTENDED_START_YEAR = 1995
EXTENDED_END_YEAR = 2026
EXTENDED_CHUNK_SIZE = 5  # Years per API call (FMP limit)
EXTENDED_FINANCIAL_LIMIT = 120  # ~30 years of quarterly data


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


# ==================== ASYNC FETCHER ====================
class AsyncFMPFetcher:
    """Async API fetcher with retry logic and rate limiting"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(HISTORICAL_MAX_CONCURRENT)
        self.rate_limiter = RateLimiter(HISTORICAL_CALLS_PER_MINUTE)
        self.total_calls = 0

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=HISTORICAL_REQUEST_TIMEOUT)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch(self, url: str, params: dict = None, retry_count: int = 0) -> Optional[dict]:
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        should_retry = False
        await self.rate_limiter.acquire()
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        self.total_calls += 1
                        return await response.json()
                    elif response.status == 429 and retry_count < MAX_RETRIES:
                        should_retry = True
                    else:
                        log.warning(f"API error {response.status} for {url} params={params.get('symbol', '')}")
                        return None
            except asyncio.TimeoutError:
                if retry_count < MAX_RETRIES:
                    should_retry = True
                else:
                    return None
            except Exception as e:
                log.error(f"Fetch error: {e}")
                return None

        if should_retry:
            await asyncio.sleep(2 ** retry_count)
            return await self.fetch(url, params, retry_count + 1)
        return None


# ==================== PROGRESS TRACKER ====================
class ProgressTracker:
    """Tracks collection progress in backtest.db for resumability"""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_completed(self, data_type: str) -> set:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT symbol FROM collection_progress WHERE data_type = ? AND status = 'completed'",
            (data_type,)
        ).fetchall()
        conn.close()
        return {r[0] for r in rows}

    def get_failed(self, data_type: str) -> list:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT symbol FROM collection_progress WHERE data_type = ? AND status = 'failed'",
            (data_type,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def mark_completed(self, symbol: str, data_type: str, rows_collected: int,
                       min_date: str = None, max_date: str = None):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO collection_progress
            (symbol, data_type, status, rows_collected, min_date, max_date, collected_at, error_message)
            VALUES (?, ?, 'completed', ?, ?, ?, ?, NULL)
        """, (symbol, data_type, rows_collected, min_date, max_date,
              datetime.now(timezone.utc).isoformat()))
        conn.commit()
        conn.close()

    def mark_failed(self, symbol: str, data_type: str, error: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO collection_progress
            (symbol, data_type, status, rows_collected, min_date, max_date, collected_at, error_message)
            VALUES (?, ?, 'failed', 0, NULL, NULL, ?, ?)
        """, (symbol, data_type, datetime.now(timezone.utc).isoformat(), error))
        conn.commit()
        conn.close()

    def get_summary(self) -> dict:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT data_type, status, COUNT(*), SUM(rows_collected)
            FROM collection_progress
            GROUP BY data_type, status
            ORDER BY data_type, status
        """).fetchall()
        conn.close()
        summary = {}
        for data_type, status, count, total_rows in rows:
            if data_type not in summary:
                summary[data_type] = {}
            summary[data_type][status] = {'count': count, 'rows': total_rows or 0}
        return summary


# ==================== COLLECTION FUNCTIONS ====================

async def collect_prices_for_symbol(fetcher: AsyncFMPFetcher, symbol: str,
                                     extended: bool = False) -> Optional[list]:
    """
    Fetch daily OHLCV for one symbol.

    Args:
        fetcher: API fetcher
        symbol: Stock symbol
        extended: If True, fetch 30 years (1995-2025) by looping 5-year chunks
                  If False, fetch default 5 years
    """
    all_historical = []

    if extended:
        # Loop through 5-year chunks for 30-year coverage
        for start_year in range(EXTENDED_START_YEAR, EXTENDED_END_YEAR, EXTENDED_CHUNK_SIZE):
            end_year = min(start_year + EXTENDED_CHUNK_SIZE, EXTENDED_END_YEAR)
            params = {
                'symbol': symbol,
                'from': f'{start_year}-01-01',
                'to': f'{end_year}-12-31'
            }
            data = await fetcher.fetch(HISTORICAL_ENDPOINTS['prices'], params)

            if data:
                if isinstance(data, dict):
                    chunk = data.get('historical', [])
                elif isinstance(data, list):
                    chunk = data
                else:
                    chunk = []

                all_historical.extend(chunk)

        return all_historical if all_historical else None
    else:
        # Default: single call for ~5 years
        data = await fetcher.fetch(
            HISTORICAL_ENDPOINTS['prices'],
            {'symbol': symbol}
        )
        if not data:
            return None

        # FMP returns {"historical": [...]} or just a list
        if isinstance(data, dict):
            historical = data.get('historical', [])
        elif isinstance(data, list):
            historical = data
        else:
            return None

        return historical


async def collect_financial_for_symbol(fetcher: AsyncFMPFetcher, symbol: str,
                                       endpoint_key: str,
                                       extended: bool = False) -> Optional[list]:
    """
    Fetch quarterly financial statements for one symbol.

    Args:
        fetcher: API fetcher
        symbol: Stock symbol
        endpoint_key: API endpoint key (income, balance, etc.)
        extended: If True, fetch 30 years (limit=120), else 5 years (limit=20)
    """
    limit = EXTENDED_FINANCIAL_LIMIT if extended else 20

    data = await fetcher.fetch(
        HISTORICAL_ENDPOINTS[endpoint_key],
        {'symbol': symbol, 'period': 'quarter', 'limit': limit}
    )
    if not data or not isinstance(data, list):
        return None
    return data


# ==================== BATCH PROCESSORS ====================

def _to_float(val):
    """Convert numeric values to float for SQLite compatibility (avoids int overflow)"""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def save_prices_batch(symbols_data: Dict[str, list]):
    """Bulk insert price data into backtest.db"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            rows.append((
                symbol,
                r.get('date'),
                _to_float(r.get('open')),
                _to_float(r.get('high')),
                _to_float(r.get('low')),
                _to_float(r.get('close')),
                r.get('volume'),
                _to_float(r.get('adjClose') or r.get('adjustedClose') or r.get('close')),
            ))
    cur.executemany("""
        INSERT OR IGNORE INTO historical_prices
        (symbol, date, open, high, low, close, volume, adjusted_close)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    return len(rows)


# Insert semantics: IGNORE for backfills (never clobber history); true UPSERT
# when we deliberately re-fetch (--recent-earnings-days / --force / --symbols)
# so vendor CORRECTIONS propagate. FMP fixes bad rows upstream (OPFI's Q3-2025
# eps_diluted was served as 45.73, later corrected to 0.77) — with IGNORE the
# first value we ever saw was frozen forever and poisoned TTM P/E site-wide.
# NOT `INSERT OR REPLACE`: that deletes+reinserts and NULLs every column not in
# the insert list — the statement tables carry ~19 populated enrichment columns
# (cost_of_revenue, retained_earnings, stock_based_comp...) that a REPLACE
# would silently erode out of the canonical release DB on every refresh.
REPLACE_MODE = False


def _insert_sql(table: str, cols) -> str:
    placeholders = ', '.join('?' for _ in cols)
    col_list = ', '.join(cols)
    if not REPLACE_MODE:
        return f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"
    key = ('symbol', 'date', 'period')
    updates = ', '.join(f"{c} = excluded.{c}" for c in cols if c not in key)
    return (f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT(symbol, date, period) DO UPDATE SET {updates}")


def _eps_inconsistent(reported, computed):
    """True only when as-reported EPS disagrees with net_income/shares by a
    MAGNITUDE no accounting explains — the OPFI signature (45.73 vs 1.56, 29x)
    or EPS fields holding whole net-income figures (millions). Deliberately
    tolerant below 10x: preferred dividends, minority interests, and diluted
    adjustments legitimately move GAAP EPS several-fold (and can flip its
    sign) relative to naive NI/shares — those must NEVER be 'corrected'.
    Known blind spots (accepted): tiny values (<0.02) never flag, so a
    too-SMALL vendor error (0.01 vs true 1.50) passes; 2-10x errors pass;
    and rows where eps AND net_income are wrong together look consistent."""
    if reported is None or computed is None:
        return False
    r, c = abs(reported), abs(computed)
    if r < 0.02 or c < 0.02:
        return False
    return max(r, c) / min(r, c) >= 10.0


def _ensure_flags_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_quality_flags (
            symbol TEXT NOT NULL,
            table_name TEXT NOT NULL,
            date TEXT NOT NULL,
            field TEXT NOT NULL,
            reported REAL,
            computed REAL,
            flagged_at TEXT,
            PRIMARY KEY (symbol, table_name, date, field)
        )
    """)


def save_income_batch(symbols_data: Dict[str, list]):
    """Bulk insert income statement data (with EPS internal-consistency guard)."""
    conn = sqlite3.connect(BACKTEST_DB)
    _ensure_flags_table(conn)
    cur = conn.cursor()
    rows, flags = [], []
    now = datetime.now(timezone.utc).isoformat()
    for symbol, records in symbols_data.items():
        for r in records:
            eps = _to_float(r.get('eps'))
            eps_dil = _to_float(r.get('epsdiluted') or r.get('epsDiluted'))
            net_income = _to_float(r.get('netIncome'))
            shares = _to_float(r.get('weightedAverageShsOutDil') or r.get('weightedAverageSharesDiluted'))
            # Guard: EPS must be within 10x of net_income / shares. When it
            # isn't, FLAG the row — never rewrite it. The detector can't know
            # WHICH side is wrong (OPFI: eps was garbage, computed right;
            # BRK-A: the shares field is the bad one and computed is garbage),
            # so downstream TTM consumers exclude flagged rows instead.
            computed = (net_income / shares) if (net_income is not None and shares and shares > 1000) else None
            for field, val in (('eps', eps), ('eps_diluted', eps_dil)):
                if computed is not None and _eps_inconsistent(val, computed):
                    flags.append((symbol, 'historical_income_statements', r.get('date'), field, val, round(computed, 4), now))
            rows.append((
                symbol,
                r.get('date'),
                r.get('period', 'Q'),
                r.get('calendarYear') or r.get('fiscalYear'),
                _to_float(r.get('revenue')),
                _to_float(r.get('grossProfit')),
                _to_float(r.get('operatingIncome')),
                net_income,
                _to_float(r.get('ebitda')),
                eps,
                eps_dil,
                shares,
                _to_float(r.get('researchAndDevelopmentExpenses')),
                _to_float(r.get('sellingGeneralAndAdministrativeExpenses')),
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    INCOME_COLS = ('symbol', 'date', 'period', 'fiscal_year', 'revenue', 'gross_profit',
                   'operating_income', 'net_income', 'ebitda', 'eps', 'eps_diluted',
                   'weighted_avg_shares_diluted', 'rd_expense', 'sga_expense',
                   'filing_date', 'accepted_date')
    cur.executemany(_insert_sql('historical_income_statements', INCOME_COLS), rows)
    # Flag maintenance only when stored rows actually changed (upsert mode).
    # In IGNORE mode the payload doesn't touch stored rows, so payload-derived
    # flags would desync from what's actually in the table.
    if REPLACE_MODE:
        # Clear flags for EXACTLY the (symbol, date) pairs this payload
        # re-checked — never a range: vendors drop/re-date quarters, and a
        # range delete would clear flags on stored rows the fetch never saw.
        pairs = sorted({(symbol, r.get('date'))
                        for symbol, records in symbols_data.items()
                        for r in records if r.get('date')})
        cur.executemany(
            "DELETE FROM data_quality_flags WHERE symbol = ? AND table_name = 'historical_income_statements' AND date = ?",
            pairs)
        cur.executemany("""
            INSERT OR REPLACE INTO data_quality_flags
            (symbol, table_name, date, field, reported, computed, flagged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, flags)
    if flags and REPLACE_MODE:
        log.warning(f"  ⚠ EPS consistency guard: {len(flags)} field(s) disagree >10x with "
                    f"net_income/shares — flagged in data_quality_flags (TTM consumers "
                    f"exclude these rows): "
                    + ', '.join(sorted({f'{f[0]} {f[2]}' for f in flags})[:8]))
    conn.commit()
    conn.close()
    return len(rows)


def save_balance_batch(symbols_data: Dict[str, list]):
    """Bulk insert balance sheet data"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            rows.append((
                symbol,
                r.get('date'),
                r.get('period', 'Q'),
                r.get('calendarYear') or r.get('fiscalYear'),
                _to_float(r.get('totalAssets')),
                _to_float(r.get('totalLiabilities')),
                _to_float(r.get('totalStockholdersEquity') or r.get('totalEquity')),
                _to_float(r.get('totalDebt')),
                _to_float(r.get('netDebt')),
                _to_float(r.get('cashAndCashEquivalents') or r.get('cashAndShortTermInvestments')),
                _to_float(r.get('totalCurrentAssets')),
                _to_float(r.get('totalCurrentLiabilities')),
                _to_float(r.get('goodwill')),
                _to_float(r.get('intangibleAssets')),
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    cur.executemany(_insert_sql('historical_balance_sheets', ('symbol', 'date', 'period', 'fiscal_year', 'total_assets', 'total_liabilities', 'total_equity', 'total_debt', 'net_debt', 'cash_and_equivalents', 'total_current_assets', 'total_current_liabilities', 'goodwill', 'intangible_assets', 'filing_date', 'accepted_date')), rows)
    conn.commit()
    conn.close()
    return len(rows)


def save_cashflow_batch(symbols_data: Dict[str, list]):
    """Bulk insert cash flow data"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            rows.append((
                symbol,
                r.get('date'),
                r.get('period', 'Q'),
                r.get('calendarYear') or r.get('fiscalYear'),
                _to_float(r.get('operatingCashFlow')),
                _to_float(r.get('capitalExpenditure')),
                _to_float(r.get('freeCashFlow')),
                # FMP /stable cash-flow uses netDividendsPaid (no 'dividendsPaid' key); fall back to common.
                _to_float(r.get('netDividendsPaid') or r.get('commonDividendsPaid')),
                _to_float(r.get('commonStockRepurchased')),
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    cur.executemany(_insert_sql('historical_cash_flows', ('symbol', 'date', 'period', 'fiscal_year', 'operating_cash_flow', 'capital_expenditure', 'free_cash_flow', 'dividends_paid', 'common_stock_repurchased', 'filing_date', 'accepted_date')), rows)
    conn.commit()
    conn.close()
    return len(rows)


def save_metrics_batch(symbols_data: Dict[str, list]):
    """Bulk insert key metrics data"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            rows.append((
                symbol,
                r.get('date'),
                r.get('period', 'Q'),
                r.get('calendarYear') or r.get('fiscalYear'),
                _to_float(r.get('enterpriseValue')),
                _to_float(r.get('evToEBITDA') or r.get('enterpriseValueOverEBITDA')),
                _to_float(r.get('marketCap')),
                _to_float(r.get('peRatio')),
                _to_float(r.get('pbRatio')),
                _to_float(r.get('debtToEquity')),
                _to_float(r.get('roe') or r.get('returnOnEquity')),
                _to_float(r.get('revenuePerShare')),
                _to_float(r.get('netIncomePerShare')),
                _to_float(r.get('operatingCashFlowPerShare')),
            ))
    cur.executemany(_insert_sql('historical_key_metrics', ('symbol', 'date', 'period', 'fiscal_year', 'enterprise_value', 'ev_to_ebitda', 'market_cap', 'pe_ratio', 'pb_ratio', 'debt_to_equity', 'roe', 'revenue_per_share', 'net_income_per_share', 'operating_cash_flow_per_share')), rows)
    conn.commit()
    conn.close()
    return len(rows)


# ==================== ORCHESTRATOR ====================

async def collect_data_type(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                            symbols: List[str], data_type: str,
                            fetch_fn, save_fn, label: str,
                            extended: bool = False):
    """Generic collection orchestrator for any data type"""
    completed = tracker.get_completed(data_type)
    remaining = [s for s in symbols if s not in completed]

    if not remaining:
        log.info(f"  {label}: All {len(symbols)} symbols already collected")
        return

    log.info(f"  {label}: {len(remaining)} symbols to collect "
             f"({len(completed)} already done)")

    total_rows = 0
    for i in range(0, len(remaining), HISTORICAL_BATCH_SIZE):
        batch = remaining[i:i + HISTORICAL_BATCH_SIZE]

        # Fetch batch in parallel
        tasks = {symbol: fetch_fn(fetcher, symbol) for symbol in batch}
        results = dict(zip(tasks.keys(),
                           await asyncio.gather(*tasks.values(), return_exceptions=True)))

        # Separate successes and failures
        success_data = {}
        for symbol, result in results.items():
            if isinstance(result, Exception):
                tracker.mark_failed(symbol, data_type, str(result))
            elif result and len(result) > 0:
                success_data[symbol] = result
            else:
                # No data returned — mark completed with 0 rows (stock may not have data)
                tracker.mark_completed(symbol, data_type, 0)

        # Bulk save
        if success_data:
            rows_saved = save_fn(success_data)
            total_rows += rows_saved

            # Mark progress
            for symbol, records in success_data.items():
                dates = [r.get('date', '') for r in records if r.get('date')]
                tracker.mark_completed(
                    symbol, data_type, len(records),
                    min_date=min(dates) if dates else None,
                    max_date=max(dates) if dates else None,
                )

        done = min(i + HISTORICAL_BATCH_SIZE, len(remaining))
        log.info(f"    {label}: {done}/{len(remaining)} "
                 f"({total_rows:,} rows saved, {fetcher.total_calls} API calls)")

    log.info(f"  {label} complete: {total_rows:,} total rows saved")


async def collect_prices(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                         symbols: List[str], extended: bool = False):
    """Collect historical prices for all symbols + SPY"""
    years_desc = "30 years (1995-2025)" if extended else "5 years"
    log.info("\n" + "=" * 60)
    log.info(f"COLLECTING: Historical Prices ({years_desc})")
    log.info("=" * 60)

    # SPY first (for market regime detection)
    if 'SPY' not in tracker.get_completed('prices'):
        log.info("  Fetching SPY (market benchmark)...")
        spy_data = await collect_prices_for_symbol(fetcher, 'SPY', extended=extended)
        if spy_data:
            save_prices_batch({'SPY': spy_data})
            dates = [r.get('date', '') for r in spy_data if r.get('date')]
            tracker.mark_completed('SPY', 'prices', len(spy_data),
                                   min(dates) if dates else None,
                                   max(dates) if dates else None)
            log.info(f"  SPY: {len(spy_data)} days saved")

    await collect_data_type(
        fetcher, tracker, symbols, 'prices',
        lambda f, s: collect_prices_for_symbol(f, s, extended=extended),
        save_prices_batch,
        "Prices",
        extended=extended
    )


async def collect_financials(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                              symbols: List[str], extended: bool = False):
    """Collect all financial statement types"""
    years_desc = "30 years" if extended else "5 years"

    financial_types = [
        ('income', lambda f, s: collect_financial_for_symbol(f, s, 'income', extended=extended),
         save_income_batch, "Income Statements"),
        ('key_metrics', lambda f, s: collect_financial_for_symbol(f, s, 'metrics', extended=extended),
         save_metrics_batch, "Key Metrics"),
        ('balance_sheet', lambda f, s: collect_financial_for_symbol(f, s, 'balance', extended=extended),
         save_balance_batch, "Balance Sheets"),
        ('cash_flow', lambda f, s: collect_financial_for_symbol(f, s, 'cashflow', extended=extended),
         save_cashflow_batch, "Cash Flows"),
    ]

    for data_type, fetch_fn, save_fn, label in financial_types:
        log.info("\n" + "=" * 60)
        log.info(f"COLLECTING: {label} (quarterly, {years_desc})")
        log.info("=" * 60)

        await collect_data_type(
            fetcher, tracker, symbols, data_type,
            fetch_fn, save_fn, label,
            extended=extended
        )


# ==================== SYMBOL LOADING ====================

def get_symbols(specific_symbols: str = None) -> List[str]:
    """Get symbol list from main database or CLI argument"""
    if specific_symbols:
        return [s.strip().upper() for s in specific_symbols.split(',')]

    conn = sqlite3.connect(DATABASE_NAME)
    df_symbols = conn.execute("SELECT symbol FROM stock_consensus ORDER BY symbol").fetchall()
    conn.close()
    return [r[0] for r in df_symbols]


def get_recent_earnings_symbols(days: int) -> List[str]:
    """Incremental selection: universe symbols that announced a quarter we DON'T yet have stored.

    Signal = earnings_surprises (announcement-dated, refreshed every pipeline run). A symbol is
    selected only when its latest announcement (within the last `days`) is newer than the latest
    income-statement filing_date we've stored — i.e. a genuinely missing quarter. Once that
    quarter is collected, filing_date catches up and the symbol drops out next run, so this stays
    small in steady state and self-corrects across weekly runs while FMP lags in filing the 10-Q."""
    universe = set(get_symbols())
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    try:
        conn = sqlite3.connect(BACKTEST_DB)
        rows = conn.execute("""
            SELECT es.symbol
            FROM (SELECT symbol, MAX(date) AS announced
                  FROM earnings_surprises WHERE date >= ? GROUP BY symbol) es
            LEFT JOIN (SELECT symbol, MAX(filing_date) AS filed
                       FROM historical_income_statements GROUP BY symbol) inc
              ON es.symbol = inc.symbol
            WHERE inc.filed IS NULL OR es.announced > inc.filed
        """, (cutoff,)).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        log.warning("earnings_surprises table missing — no incremental symbols selected.")
        return []
    recent = sorted({r[0].upper() for r in rows} & universe)
    log.info(f"Incremental: {len(recent)} symbols have a quarter newer than stored (announced since {cutoff})")
    return recent


def scan_eps_inconsistencies() -> List[str]:
    """Find EXISTING stored income rows whose EPS wildly disagrees with
    net_income/shares (frozen vendor errors, e.g. OPFI's 45.73 vs 1.56), flag
    them in data_quality_flags, and return their symbols so the incremental
    refresh re-fetches (and, via the upsert, heals) them."""
    try:
        conn = sqlite3.connect(BACKTEST_DB)
        _ensure_flags_table(conn)
        rows = conn.execute("""
            SELECT symbol, date, eps_diluted, eps,
                   net_income / weighted_avg_shares_diluted AS computed
            FROM historical_income_statements
            WHERE period LIKE 'Q%'
              AND net_income IS NOT NULL
              AND weighted_avg_shares_diluted > 1000
        """).fetchall()
        now = datetime.now(timezone.utc).isoformat()
        flags = []
        for symbol, date, eps_dil, eps, computed in rows:
            for field, val in (('eps_diluted', eps_dil), ('eps', eps)):
                if _eps_inconsistent(val, computed):
                    flags.append((symbol, 'historical_income_statements', date, field, val, round(computed, 4), now))
        conn.executemany("""
            INSERT OR REPLACE INTO data_quality_flags
            (symbol, table_name, date, field, reported, computed, flagged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, flags)
        conn.commit()
        conn.close()
        # Only symbols with flags INSIDE the ~5y fetch horizon are worth
        # re-fetching (a refresh can't touch older rows; their flags persist
        # for the exporter's exclusion join). Vendor-corrected rows converge
        # and drop out; never-corrected in-horizon rows keep re-fetching —
        # bounded by the horizon and cheap relative to the weekly run.
        horizon = (datetime.now() - timedelta(days=5 * 365)).strftime('%Y-%m-%d')
        symbols = sorted({f[0] for f in flags if (f[2] or '') >= horizon})
        if flags:
            log.warning(f"EPS consistency scan: {len(flags)} inconsistent field(s); "
                        f"{len(symbols)} symbols have in-horizon flags eligible for re-fetch: "
                        f"{', '.join(symbols[:15])}" + ('...' if len(symbols) > 15 else ''))
        return symbols
    except sqlite3.OperationalError as e:
        log.warning(f"EPS consistency scan skipped: {e}")
        return []


# ==================== STATUS DISPLAY ====================

def show_status():
    """Display collection progress summary"""
    tracker = ProgressTracker(BACKTEST_DB)
    summary = tracker.get_summary()

    if not summary:
        print("No collection progress recorded yet.")
        return

    print("\n" + "=" * 70)
    print("Historical Data Collection Status")
    print("=" * 70)

    data_types = ['prices', 'income', 'key_metrics', 'balance_sheet', 'cash_flow']
    for dt in data_types:
        if dt not in summary:
            print(f"\n  {dt:20s}: Not started")
            continue

        stats = summary[dt]
        completed = stats.get('completed', {})
        failed = stats.get('failed', {})
        print(f"\n  {dt:20s}: "
              f"{completed.get('count', 0):,} completed "
              f"({completed.get('rows', 0):,} rows) | "
              f"{failed.get('count', 0):,} failed")

    # DB file size
    db_path = Path(BACKTEST_DB)
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"\n  Database size: {size_mb:.1f} MB")

    print("=" * 70)


# ==================== MAIN ====================

async def main():
    parser = argparse.ArgumentParser(description='Collect historical data for backtesting')
    parser.add_argument('--extended', action='store_true',
                        help='Collect 30 years of data (1995-2025) instead of default 5 years')
    parser.add_argument('--force', action='store_true',
                        help='Force recollection even if symbols were previously collected')
    parser.add_argument('--prices-only', action='store_true', help='Only collect price data')
    parser.add_argument('--financials-only', action='store_true', help='Only collect financial statements')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g. AAPL,MSFT)')
    parser.add_argument('--recent-earnings-days', type=int, default=None,
                        help='Incremental: only collect symbols that announced earnings in the last N days '
                             '(uses earnings_surprises). Implies --force for the selected subset.')
    parser.add_argument('--status', action='store_true', help='Show collection progress')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed symbols')
    args = parser.parse_args()

    # Status mode
    if args.status:
        show_status()
        return

    # Ensure backtest tables exist
    setup_backtest_tables()

    # Deliberate re-fetch modes UPSERT so vendor corrections propagate
    # (see the _insert_sql note — never INSERT OR REPLACE, which would NULL
    # the enrichment columns outside the insert list). Backfills keep IGNORE.
    global REPLACE_MODE
    if args.recent_earnings_days is not None or args.force or args.symbols:
        REPLACE_MODE = True
        log.info("Re-fetch mode: UPSERT (vendor corrections overwrite the fetched columns only)")

    # Get symbols — incremental (recent earnings) selection takes precedence over the full universe.
    if args.recent_earnings_days is not None and not args.symbols:
        recent = get_recent_earnings_symbols(args.recent_earnings_days)
        # Union in symbols whose STORED rows fail the EPS consistency check —
        # frozen vendor errors heal on the next refresh instead of never.
        # Intersect with the ACTIVE universe: the backtest DB holds thousands of
        # delisted names with bad old rows; re-fetching those weekly forever
        # would bloat the run without helping the site. Vendor-corrected rows
        # converge and their flags clear; never-corrected in-horizon rows stay
        # flagged (exporters exclude them) and re-fetch weekly, bounded by the
        # 5y horizon filter in scan_eps_inconsistencies.
        flagged = set(scan_eps_inconsistencies()) & set(get_symbols())
        if flagged:
            log.info(f"Including {len(flagged)} flagged universe symbols for correction re-fetch")
        symbols = sorted(set(recent) | flagged)
        if not symbols:
            log.info("No symbols announced earnings in the window and none flagged — nothing to collect.")
            return
        args.force = True  # recent reporters have a new quarter; re-fetch even if previously collected
    else:
        symbols = get_symbols(args.symbols)
    if not symbols:
        log.error("No symbols found. Run the main pipeline first.")
        return

    tracker = ProgressTracker(BACKTEST_DB)

    # Retry mode: replace symbol list with failed ones
    if args.retry_failed:
        failed_symbols = set()
        for dt in ['prices', 'income', 'key_metrics', 'balance_sheet', 'cash_flow']:
            failed_symbols.update(tracker.get_failed(dt))
            # Clear failed status so they get retried
            conn = sqlite3.connect(BACKTEST_DB)
            conn.execute(
                "DELETE FROM collection_progress WHERE data_type = ? AND status = 'failed'",
                (dt,)
            )
            conn.commit()
            conn.close()
        symbols = sorted(failed_symbols)
        if not symbols:
            log.info("No failed symbols to retry.")
            return
        log.info(f"Retrying {len(symbols)} previously failed symbols")

    start_time = datetime.now()
    log.info("=" * 60)
    log.info("Historical Data Collection for Backtesting")
    log.info("=" * 60)
    log.info(f"Symbols: {len(symbols)}")
    log.info(f"Database: {BACKTEST_DB}")
    mode_str = 'prices only' if args.prices_only else 'financials only' if args.financials_only else 'full collection'
    range_str = '30 years (1995-2025)' if args.extended else '5 years (default)'
    log.info(f"Mode: {mode_str}")
    log.info(f"Date range: {range_str}")

    if args.extended:
        log.info(f"NOTE: Extended mode will make ~{len(symbols) * 7}+ API calls for prices (7 chunks x {len(symbols)} symbols)")
        log.info(f"      Estimated time: ~{len(symbols) * 7 / HISTORICAL_CALLS_PER_MINUTE:.0f} minutes for prices alone")

    # Force mode: clear progress for specified symbols to allow recollection
    if args.force:
        log.info("FORCE MODE: Clearing previous collection progress...")
        conn = sqlite3.connect(BACKTEST_DB)
        data_types_to_clear = []
        if not args.financials_only:
            data_types_to_clear.append('prices')
        if not args.prices_only:
            data_types_to_clear.extend(['income', 'key_metrics', 'balance_sheet', 'cash_flow'])

        for dt in data_types_to_clear:
            if args.symbols:
                # Clear specific symbols
                for s in symbols:
                    conn.execute("DELETE FROM collection_progress WHERE symbol = ? AND data_type = ?", (s, dt))
            else:
                # Clear all for this data type
                conn.execute("DELETE FROM collection_progress WHERE data_type = ?", (dt,))
        conn.commit()
        conn.close()
        log.info(f"  Cleared progress for: {', '.join(data_types_to_clear)}")

    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        if not args.financials_only:
            await collect_prices(fetcher, tracker, symbols, extended=args.extended)

        if not args.prices_only:
            await collect_financials(fetcher, tracker, symbols, extended=args.extended)

        log.info(f"\nTotal API calls made: {fetcher.total_calls:,}")

    elapsed = datetime.now() - start_time
    log.info("\n" + "=" * 60)
    log.info(f"Collection complete! Runtime: {elapsed}")
    log.info("=" * 60)

    # Show final status
    show_status()


if __name__ == "__main__":
    asyncio.run(main())

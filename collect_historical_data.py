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
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
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


def save_income_batch(symbols_data: Dict[str, list]):
    """Bulk insert income statement data"""
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
                _to_float(r.get('revenue')),
                _to_float(r.get('grossProfit')),
                _to_float(r.get('operatingIncome')),
                _to_float(r.get('netIncome')),
                _to_float(r.get('ebitda')),
                _to_float(r.get('eps')),
                _to_float(r.get('epsdiluted') or r.get('epsDiluted')),
                _to_float(r.get('weightedAverageShsOutDil') or r.get('weightedAverageSharesDiluted')),
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    cur.executemany("""
        INSERT OR IGNORE INTO historical_income_statements
        (symbol, date, period, fiscal_year, revenue, gross_profit, operating_income,
         net_income, ebitda, eps, eps_diluted, weighted_avg_shares_diluted,
         filing_date, accepted_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
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
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    cur.executemany("""
        INSERT OR IGNORE INTO historical_balance_sheets
        (symbol, date, period, fiscal_year, total_assets, total_liabilities,
         total_equity, total_debt, net_debt, cash_and_equivalents,
         filing_date, accepted_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
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
                r.get('filingDate'),
                r.get('acceptedDate'),
            ))
    cur.executemany("""
        INSERT OR IGNORE INTO historical_cash_flows
        (symbol, date, period, fiscal_year, operating_cash_flow,
         capital_expenditure, free_cash_flow, filing_date, accepted_date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
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
    cur.executemany("""
        INSERT OR IGNORE INTO historical_key_metrics
        (symbol, date, period, fiscal_year, enterprise_value, ev_to_ebitda,
         market_cap, pe_ratio, pb_ratio, debt_to_equity, roe,
         revenue_per_share, net_income_per_share, operating_cash_flow_per_share)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
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
    parser.add_argument('--status', action='store_true', help='Show collection progress')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed symbols')
    args = parser.parse_args()

    # Status mode
    if args.status:
        show_status()
        return

    # Ensure backtest tables exist
    setup_backtest_tables()

    # Get symbols
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

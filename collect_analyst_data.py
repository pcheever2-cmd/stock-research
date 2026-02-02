#!/usr/bin/env python3
"""
Analyst Data Collector for Backtesting
Collects grades (historical), analyst estimates (current snapshot),
and price target summaries from FMP into backtest.db.

Usage:
    python collect_analyst_data.py                    # Collect everything
    python collect_analyst_data.py --grades-only      # Just historical grades
    python collect_analyst_data.py --estimates-only   # Just analyst estimates
    python collect_analyst_data.py --targets-only     # Just price target summaries
    python collect_analyst_data.py --symbols AAPL,MSFT
    python collect_analyst_data.py --status
    python collect_analyst_data.py --retry-failed
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
    ANALYST_ENDPOINTS,
)
from setup_database import setup_backtest_tables
from collect_historical_data import (
    RateLimiter, AsyncFMPFetcher, ProgressTracker, collect_data_type, _to_float,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


# ==================== FETCH FUNCTIONS ====================

async def fetch_grades(fetcher: AsyncFMPFetcher, symbol: str) -> Optional[list]:
    """Fetch historical analyst grades/rating changes for one symbol"""
    data = await fetcher.fetch(
        ANALYST_ENDPOINTS['grades'],
        {'symbol': symbol}
    )
    if not data or not isinstance(data, list):
        return None
    return data if len(data) > 0 else None


async def fetch_analyst_estimates(fetcher: AsyncFMPFetcher, symbol: str) -> Optional[list]:
    """Fetch analyst consensus estimates for one symbol (annual periods)"""
    data = await fetcher.fetch(
        ANALYST_ENDPOINTS['analyst_estimates'],
        {'symbol': symbol, 'period': 'annual'}
    )
    if not data or not isinstance(data, list):
        return None
    return data if len(data) > 0 else None


async def fetch_price_target_summary(fetcher: AsyncFMPFetcher, symbol: str) -> Optional[list]:
    """Fetch price target summary for one symbol"""
    data = await fetcher.fetch(
        ANALYST_ENDPOINTS['price_target_summary'],
        {'symbol': symbol}
    )
    if not data:
        return None
    # API may return a single dict or a list
    if isinstance(data, dict):
        return [data] if data else None
    if isinstance(data, list):
        return data if len(data) > 0 else None
    return None


# ==================== SAVE FUNCTIONS ====================

def save_grades_batch(symbols_data: Dict[str, list]) -> int:
    """Bulk insert grade data into backtest.db"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            date = r.get('date', '')
            company = r.get('gradingCompany') or r.get('grading_company', '')
            if not date or not company:
                continue
            rows.append((
                symbol,
                date,
                company,
                r.get('previousGrade') or r.get('previous_grade', ''),
                r.get('newGrade') or r.get('new_grade', ''),
                r.get('action', ''),
            ))
    if rows:
        cur.executemany("""
            INSERT OR IGNORE INTO historical_grades
            (symbol, date, grading_company, previous_grade, new_grade, action)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows)
    conn.commit()
    conn.close()
    return len(rows)


def save_estimates_batch(symbols_data: Dict[str, list]) -> int:
    """Bulk insert analyst estimates into backtest.db"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            # Extract fiscal year from calendarYear or date field
            fiscal_year = r.get('calendarYear') or r.get('calendar_year')
            if fiscal_year is None:
                # Fall back to extracting year from date (e.g., "2026-09-27" -> 2026)
                date_str = r.get('date', '')
                if date_str and len(date_str) >= 4:
                    fiscal_year = date_str[:4]
                else:
                    continue
            try:
                fiscal_year = int(fiscal_year)
            except (TypeError, ValueError):
                continue
            rows.append((
                symbol,
                fiscal_year,
                _to_float(r.get('revenueLow') or r.get('revenue_low')),
                _to_float(r.get('revenueHigh') or r.get('revenue_high')),
                _to_float(r.get('revenueAvg') or r.get('revenue_avg')),
                _to_float(r.get('ebitdaLow') or r.get('ebitda_low')),
                _to_float(r.get('ebitdaHigh') or r.get('ebitda_high')),
                _to_float(r.get('ebitdaAvg') or r.get('ebitda_avg')),
                _to_float(r.get('epsLow') or r.get('eps_low')),
                _to_float(r.get('epsHigh') or r.get('eps_high')),
                _to_float(r.get('epsAvg') or r.get('eps_avg')),
                r.get('numAnalystsRevenue') or r.get('numberAnalystsEstimatedRevenue'),
                r.get('numAnalystsEps') or r.get('numberAnalystEstimatedEps'),
                now,
            ))
    if rows:
        cur.executemany("""
            INSERT OR REPLACE INTO analyst_estimates_snapshot
            (symbol, fiscal_year, revenue_low, revenue_high, revenue_avg,
             ebitda_low, ebitda_high, ebitda_avg, eps_low, eps_high, eps_avg,
             num_analysts_revenue, num_analysts_eps, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    conn.commit()
    conn.close()
    return len(rows)


def save_price_target_summary_batch(symbols_data: Dict[str, list]) -> int:
    """Bulk insert price target summaries into backtest.db"""
    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for symbol, records in symbols_data.items():
        for r in records:
            rows.append((
                symbol,
                _to_float(r.get('lastMonthAvgPriceTarget') or r.get('last_month_avg')),
                r.get('lastMonthCount') or r.get('last_month_count'),
                _to_float(r.get('lastQuarterAvgPriceTarget') or r.get('last_quarter_avg')),
                r.get('lastQuarterCount') or r.get('last_quarter_count'),
                _to_float(r.get('lastYearAvgPriceTarget') or r.get('last_year_avg')),
                r.get('lastYearCount') or r.get('last_year_count'),
                _to_float(r.get('allTimeAvgPriceTarget') or r.get('all_time_avg')),
                r.get('allTimeCount') or r.get('all_time_count'),
                now,
            ))
    if rows:
        cur.executemany("""
            INSERT OR REPLACE INTO price_target_summary
            (symbol, last_month_avg, last_month_count, last_quarter_avg,
             last_quarter_count, last_year_avg, last_year_count,
             all_time_avg, all_time_count, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    conn.commit()
    conn.close()
    return len(rows)


# ==================== ORCHESTRATORS ====================

async def collect_grades(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                         symbols: List[str]):
    """Collect historical grades for all symbols"""
    log.info("\n" + "=" * 60)
    log.info("COLLECTING: Historical Analyst Grades")
    log.info("=" * 60)

    await collect_data_type(
        fetcher, tracker, symbols, 'grades',
        lambda f, s: fetch_grades(f, s),
        save_grades_batch,
        "Grades"
    )


async def collect_estimates(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                            symbols: List[str]):
    """Collect analyst estimates for all symbols"""
    log.info("\n" + "=" * 60)
    log.info("COLLECTING: Analyst Estimates (current snapshot)")
    log.info("=" * 60)

    await collect_data_type(
        fetcher, tracker, symbols, 'analyst_estimates',
        lambda f, s: fetch_analyst_estimates(f, s),
        save_estimates_batch,
        "Analyst Estimates"
    )


async def collect_targets(fetcher: AsyncFMPFetcher, tracker: ProgressTracker,
                           symbols: List[str]):
    """Collect price target summaries for all symbols"""
    log.info("\n" + "=" * 60)
    log.info("COLLECTING: Price Target Summaries")
    log.info("=" * 60)

    await collect_data_type(
        fetcher, tracker, symbols, 'price_target_summary',
        lambda f, s: fetch_price_target_summary(f, s),
        save_price_target_summary_batch,
        "Price Target Summary"
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
    """Display analyst data collection progress"""
    tracker = ProgressTracker(BACKTEST_DB)
    summary = tracker.get_summary()

    if not summary:
        print("No collection progress recorded yet.")
        return

    print("\n" + "=" * 70)
    print("Analyst Data Collection Status")
    print("=" * 70)

    data_types = ['grades', 'analyst_estimates', 'price_target_summary']
    for dt in data_types:
        if dt not in summary:
            print(f"\n  {dt:25s}: Not started")
            continue

        stats = summary[dt]
        completed = stats.get('completed', {})
        failed = stats.get('failed', {})
        print(f"\n  {dt:25s}: "
              f"{completed.get('count', 0):,} completed "
              f"({completed.get('rows', 0):,} rows) | "
              f"{failed.get('count', 0):,} failed")

    # Row counts from actual tables
    try:
        conn = sqlite3.connect(BACKTEST_DB)
        for table in ['historical_grades', 'analyst_estimates_snapshot', 'price_target_summary']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"\n  {table:25s}: {count:,} rows in table")
        conn.close()
    except Exception:
        pass

    print("=" * 70)


# ==================== MAIN ====================

async def main():
    parser = argparse.ArgumentParser(description='Collect analyst data for backtesting')
    parser.add_argument('--grades-only', action='store_true', help='Only collect grades')
    parser.add_argument('--estimates-only', action='store_true', help='Only collect analyst estimates')
    parser.add_argument('--targets-only', action='store_true', help='Only collect price target summaries')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g. AAPL,MSFT)')
    parser.add_argument('--status', action='store_true', help='Show collection progress')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed symbols')
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    # Ensure backtest tables exist
    setup_backtest_tables()

    symbols = get_symbols(args.symbols)
    if not symbols:
        log.error("No symbols found. Run the main pipeline first.")
        return

    tracker = ProgressTracker(BACKTEST_DB)

    # Retry mode
    if args.retry_failed:
        failed_symbols = set()
        for dt in ['grades', 'analyst_estimates', 'price_target_summary']:
            failed_symbols.update(tracker.get_failed(dt))
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

    # Determine which types to collect
    collect_all = not (args.grades_only or args.estimates_only or args.targets_only)

    start_time = datetime.now()
    log.info("=" * 60)
    log.info("Analyst Data Collection")
    log.info("=" * 60)
    log.info(f"Symbols: {len(symbols)}")
    log.info(f"Database: {BACKTEST_DB}")

    mode_parts = []
    if collect_all or args.grades_only:
        mode_parts.append('grades')
    if collect_all or args.estimates_only:
        mode_parts.append('estimates')
    if collect_all or args.targets_only:
        mode_parts.append('targets')
    log.info(f"Mode: {', '.join(mode_parts)}")

    async with AsyncFMPFetcher(FMP_API_KEY) as fetcher:
        if collect_all or args.grades_only:
            await collect_grades(fetcher, tracker, symbols)

        if collect_all or args.estimates_only:
            await collect_estimates(fetcher, tracker, symbols)

        if collect_all or args.targets_only:
            await collect_targets(fetcher, tracker, symbols)

        log.info(f"\nTotal API calls made: {fetcher.total_calls:,}")

    elapsed = datetime.now() - start_time
    log.info("\n" + "=" * 60)
    log.info(f"Collection complete! Runtime: {elapsed}")
    log.info("=" * 60)

    show_status()


if __name__ == "__main__":
    asyncio.run(main())

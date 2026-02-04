#!/usr/bin/env python3
"""
Mock Portfolio Tracker (v2 - Stocks & Options)
===============================================
Paper trading system that tracks both stock and options positions.

Features:
- Scans for new qualifying signals
- Opens paper positions (stocks or options) with entry tracking
- Monitors for exits (stop-loss, hold period, expiration, manual)
- Calculates real-time P&L
- Stores everything in SQLite for dashboard integration

Commands:
  python mock_portfolio.py scan      - Scan for new signals and open positions
  python mock_portfolio.py status    - Show current portfolio status
  python mock_portfolio.py history   - Show trade history
  python mock_portfolio.py reset     - Reset portfolio (careful!)
  python mock_portfolio.py daily     - Daily update: check stops, update prices, report
  python mock_portfolio.py compare   - Compare stock vs options performance

Stock Strategy: Strong Sectors (E_StrongSectors)
- Only trade in each bucket's strong sectors
- Position sizing: 3.5%/4.5%/3.0% per bucket
- Stops: -15%/-20%/-12%

Options Strategy: OTM5_EarningsBeat
- 5% OTM calls on earnings beat stocks
- Position sizing: 2.5% per position
- Stop: -25%, Target: +100%
"""

import sqlite3
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import requests

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')
PORTFOLIO_DB = str(PROJECT_ROOT / 'mock_portfolio.db')

# Import bucket config
from bucket_config import BUCKETS, BUCKET_1, BUCKET_2, BUCKET_3, BucketConfig
from config import FMP_API_KEY

# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATIONS
# ══════════════════════════════════════════════════════════════════════════════

# Stock strategy: Strong Sectors (best performing)
STOCK_STRATEGY = {
    'name': 'StrongSectors',
    'starting_capital': 70_000,  # 70% of total
    'position_pct': {1: 3.5, 2: 4.5, 3: 3.0},
    'max_positions': {1: 8, 2: 6, 3: 10},
    'stop_loss_pct': {1: -15.0, 2: -20.0, 3: -12.0},
    'hold_months': {1: 6, 2: 6, 3: 3},
    'require_strong_sector': True,
    'require_earnings_beat': False,
}

# Options strategy: OTM5_EarningsBeat (highest return)
OPTIONS_STRATEGY = {
    'name': 'OTM5_EarningsBeat',
    'starting_capital': 30_000,  # 30% of total
    'position_pct': 2.5,
    'max_positions': 10,
    'stop_loss_pct': -25.0,
    'profit_target_pct': 100.0,
    'hold_days': 60,
    'option_type': 'OTM_5',  # 5% out-of-the-money calls
    'require_earnings_beat': True,
    'require_strong_sector': False,
    'premium_pct': 0.035,  # 3.5% of stock price for OTM_5
}

# Combined for backward compatibility
STRATEGY_CONFIG = STOCK_STRATEGY

# Total portfolio allocation
TOTAL_STARTING_CAPITAL = 100_000
STOCK_ALLOCATION_PCT = 70  # 70% stocks
OPTIONS_ALLOCATION_PCT = 30  # 30% options


# ══════════════════════════════════════════════════════════════════════════════
# DATABASE SETUP
# ══════════════════════════════════════════════════════════════════════════════

def setup_portfolio_db():
    """Create portfolio database tables if they don't exist."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()

    # Portfolio configuration
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_config (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Active positions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bucket_id INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            entry_price REAL NOT NULL,
            shares REAL NOT NULL,
            position_value REAL NOT NULL,
            stop_loss_price REAL NOT NULL,
            target_exit_date DATE NOT NULL,
            current_price REAL,
            current_value REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            status TEXT DEFAULT 'open',  -- open, closed
            exit_date DATE,
            exit_price REAL,
            exit_reason TEXT,  -- stop_loss, hold_complete, manual
            realized_pnl REAL,
            realized_pnl_pct REAL,
            signal_type TEXT,
            signal_date DATE,
            earnings_beat INTEGER,
            sector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Trade history (completed trades)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            bucket_id INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            entry_price REAL NOT NULL,
            exit_date DATE NOT NULL,
            exit_price REAL NOT NULL,
            shares REAL NOT NULL,
            position_value REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            realized_pnl_pct REAL NOT NULL,
            hold_days INTEGER NOT NULL,
            exit_reason TEXT NOT NULL,
            signal_type TEXT,
            sector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Daily snapshots for equity curve
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_snapshots (
            date DATE PRIMARY KEY,
            cash REAL NOT NULL,
            positions_value REAL NOT NULL,
            total_equity REAL NOT NULL,
            open_positions INTEGER NOT NULL,
            unrealized_pnl REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Signal log (all signals scanned, whether traded or not)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date DATE NOT NULL,
            bucket_id INTEGER NOT NULL,
            signal_type TEXT,
            action TEXT,  -- opened, skipped_max_positions, skipped_already_held, skipped_sector, etc.
            reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Options positions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            underlying_price REAL NOT NULL,
            strike_price REAL NOT NULL,
            option_type TEXT NOT NULL,  -- OTM_5, OTM_10, ATM
            contracts INTEGER NOT NULL,
            premium_per_contract REAL NOT NULL,
            total_premium REAL NOT NULL,
            entry_date DATE NOT NULL,
            expiration_date DATE NOT NULL,
            stop_loss_pct REAL NOT NULL,
            profit_target_pct REAL NOT NULL,
            current_value REAL,
            unrealized_pnl REAL,
            unrealized_pnl_pct REAL,
            status TEXT DEFAULT 'open',  -- open, closed
            exit_date DATE,
            exit_value REAL,
            exit_reason TEXT,  -- stop_loss, profit_target, expiration, manual
            realized_pnl REAL,
            realized_pnl_pct REAL,
            signal_type TEXT,
            signal_date DATE,
            earnings_beat INTEGER,
            sector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Options trade history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS options_trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            option_type TEXT NOT NULL,
            contracts INTEGER NOT NULL,
            entry_date DATE NOT NULL,
            entry_premium REAL NOT NULL,
            exit_date DATE NOT NULL,
            exit_value REAL NOT NULL,
            total_premium REAL NOT NULL,
            realized_pnl REAL NOT NULL,
            realized_pnl_pct REAL NOT NULL,
            hold_days INTEGER NOT NULL,
            exit_reason TEXT NOT NULL,
            signal_type TEXT,
            sector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Separate cash tracking for stocks and options
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_allocations (
            allocation_type TEXT PRIMARY KEY,  -- 'stocks', 'options'
            cash REAL NOT NULL,
            starting_capital REAL NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


def get_config(key: str, default: str = None) -> Optional[str]:
    """Get a config value."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM portfolio_config WHERE key = ?", (key,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else default


def set_config(key: str, value: str):
    """Set a config value."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO portfolio_config (key, value, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
    """, (key, value))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# PRICE DATA
# ══════════════════════════════════════════════════════════════════════════════

def get_current_price(symbol: str) -> Optional[float]:
    """Get current price from FMP API."""
    try:
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbol}?apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                return data[0].get('price')
    except Exception as e:
        print(f"    Warning: Could not get price for {symbol}: {e}")
    return None


def get_batch_prices(symbols: List[str]) -> Dict[str, float]:
    """Get current prices for multiple symbols."""
    if not symbols:
        return {}

    prices = {}
    # FMP allows batch quotes
    try:
        symbols_str = ','.join(symbols[:50])  # max 50 at a time
        url = f"https://financialmodelingprep.com/api/v3/quote-short/{symbols_str}?apikey={FMP_API_KEY}"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for item in data:
                prices[item['symbol']] = item.get('price')
    except Exception as e:
        print(f"    Warning: Batch price fetch failed: {e}")

    # Fetch any missing individually
    for sym in symbols:
        if sym not in prices:
            p = get_current_price(sym)
            if p:
                prices[sym] = p

    return prices


# ══════════════════════════════════════════════════════════════════════════════
# SIGNAL SCANNING
# ══════════════════════════════════════════════════════════════════════════════

def load_recent_signals(days_back: int = 7) -> pd.DataFrame:
    """Load recent signals from backtest database."""
    conn = sqlite3.connect(BACKTEST_DB)
    cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    df = pd.read_sql_query(f"""
        SELECT
            s.symbol, s.date, s.signal_type,
            s.lt_score, s.value_score_v2, s.close_price,
            d.rsi, d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth,
            d.fundamentals_score, d.market_risk_score
        FROM backtest_signals s
        LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
        WHERE s.date >= '{cutoff}'
    """, conn)

    # Get supplementary data
    mcaps = pd.read_sql_query("""
        SELECT symbol, market_cap FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol HAVING date = MAX(date)
    """, conn).set_index('symbol')['market_cap']

    acov = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot GROUP BY symbol
    """, conn).set_index('symbol')['analyst_count']

    # Earnings surprise
    income = pd.read_sql_query("""
        SELECT symbol, fiscal_year, SUM(eps_diluted) as actual_eps, COUNT(*) as quarters
        FROM historical_income_statements
        WHERE eps_diluted IS NOT NULL AND fiscal_year IS NOT NULL
        GROUP BY symbol, fiscal_year HAVING quarters >= 4
    """, conn)
    estimates = pd.read_sql_query("""
        SELECT symbol, fiscal_year, eps_avg FROM analyst_estimates_snapshot
        WHERE eps_avg IS NOT NULL AND fiscal_year IS NOT NULL
    """, conn)
    conn.close()

    conn2 = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query(
        "SELECT symbol, sector FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    # Enrich
    df['market_cap'] = df['symbol'].map(mcaps)
    df['analyst_count'] = df['symbol'].map(acov)
    df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)
    df['date_dt'] = pd.to_datetime(df['date'])

    # Earnings surprise
    if not income.empty and not estimates.empty:
        merged = income.merge(estimates, on=['symbol', 'fiscal_year'], how='inner')
        if not merged.empty:
            merged = merged.sort_values('fiscal_year').groupby('symbol').last().reset_index()
            merged['earnings_surprise_pct'] = np.where(
                merged['eps_avg'].abs() > 0.01,
                (merged['actual_eps'] - merged['eps_avg']) / merged['eps_avg'].abs() * 100,
                np.nan
            )
            surprise_map = merged.set_index('symbol')['earnings_surprise_pct'].clip(-200, 500)
            df['earnings_surprise_pct'] = df['symbol'].map(surprise_map)
        else:
            df['earnings_surprise_pct'] = np.nan
    else:
        df['earnings_surprise_pct'] = np.nan

    return df


def filter_bucket_signals(df: pd.DataFrame, bucket: BucketConfig) -> pd.DataFrame:
    """Apply bucket filter with strategy overlay."""
    # Base filter
    if bucket.id == 1:
        mask = (
            (df['value_score_v2'] >= 55) &
            (df['fundamentals_score'] >= 18) &
            (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 20) &
            (df['rev_growth'] > 10) &
            (df['market_cap'] > 2e9) &
            (df['analyst_count'] >= 6)
        )
    elif bucket.id == 2:
        mask = (
            (df['market_bearish'] == 1) &
            (df['rsi'] < 40) &
            (df['fundamentals_score'] >= 15) &
            (df['value_score_v2'] >= 40) &
            (df['market_cap'] > 1e9)
        )
    else:  # bucket 3
        ev = df['ev_ebitda_clean']
        mask = (
            (df['eps_growth'] >= 35) &
            (df['ebitda_growth'] >= 33) &
            (ev >= 12) & (ev <= 27) &
            (df['rsi'] < 43)
        )

    result = df[mask].copy()

    # Apply strategy overlay: strong sectors only
    if STRATEGY_CONFIG['require_strong_sector']:
        result = result[result['sector'].isin(bucket.strong_sectors)]

    if STRATEGY_CONFIG['require_earnings_beat']:
        result = result[result['earnings_surprise_pct'] > 0]

    return result


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO OPERATIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_cash() -> float:
    """Get current cash balance."""
    val = get_config('cash', str(STRATEGY_CONFIG['starting_capital']))
    return float(val)


def set_cash(amount: float):
    """Set cash balance."""
    set_config('cash', str(amount))


def get_open_positions() -> pd.DataFrame:
    """Get all open positions."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query("""
        SELECT * FROM positions WHERE status = 'open' ORDER BY entry_date
    """, conn)
    conn.close()
    return df


def get_position_count(bucket_id: int) -> int:
    """Get count of open positions for a bucket."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM positions WHERE bucket_id = ? AND status = 'open'
    """, (bucket_id,))
    count = cursor.fetchone()[0]
    conn.close()
    return count


def is_symbol_held(symbol: str) -> bool:
    """Check if we already hold this symbol."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM positions WHERE symbol = ? AND status = 'open'
    """, (symbol,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def open_position(symbol: str, bucket_id: int, entry_price: float,
                  signal_type: str, signal_date: str, sector: str,
                  earnings_beat: bool) -> bool:
    """Open a new paper position."""
    bucket = BUCKETS[bucket_id - 1]

    # Check position limits
    current_count = get_position_count(bucket_id)
    max_pos = STRATEGY_CONFIG['max_positions'].get(bucket_id, bucket.max_positions)
    if current_count >= max_pos:
        log_signal(symbol, signal_date, bucket_id, signal_type,
                   'skipped_max_positions', f'Already at max {max_pos} for bucket {bucket_id}')
        return False

    # Check if already held
    if is_symbol_held(symbol):
        log_signal(symbol, signal_date, bucket_id, signal_type,
                   'skipped_already_held', f'{symbol} already in portfolio')
        return False

    # Calculate position
    cash = get_cash()
    pos_pct = STRATEGY_CONFIG['position_pct'].get(bucket_id, bucket.position_pct) / 100
    position_value = cash * pos_pct
    shares = position_value / entry_price

    # Calculate stop loss
    stop_pct = STRATEGY_CONFIG['stop_loss_pct'].get(bucket_id, bucket.stop_loss_pct) / 100
    stop_loss_price = entry_price * (1 + stop_pct)

    # Calculate target exit date
    hold_months = STRATEGY_CONFIG['hold_months'].get(bucket_id, bucket.hold_months)
    entry_date = datetime.now().date()
    target_exit = entry_date + timedelta(days=hold_months * 30)

    # Insert position
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO positions (
            symbol, bucket_id, entry_date, entry_price, shares, position_value,
            stop_loss_price, target_exit_date, current_price, current_value,
            unrealized_pnl, unrealized_pnl_pct, signal_type, signal_date,
            earnings_beat, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol, bucket_id, entry_date.isoformat(), entry_price, shares, position_value,
        stop_loss_price, target_exit.isoformat(), entry_price, position_value,
        0, 0, signal_type, signal_date, 1 if earnings_beat else 0, sector
    ))
    conn.commit()
    conn.close()

    # Deduct cash
    set_cash(cash - position_value)

    log_signal(symbol, signal_date, bucket_id, signal_type, 'opened',
               f'Opened {shares:.2f} shares @ ${entry_price:.2f}')

    return True


def close_position(position_id: int, exit_price: float, exit_reason: str):
    """Close a position."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()

    # Get position details
    cursor.execute("SELECT * FROM positions WHERE id = ?", (position_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return

    cols = [d[0] for d in cursor.description]
    pos = dict(zip(cols, row))

    # Calculate P&L
    exit_value = pos['shares'] * exit_price
    realized_pnl = exit_value - pos['position_value']
    realized_pnl_pct = (realized_pnl / pos['position_value']) * 100
    hold_days = (datetime.now().date() - datetime.fromisoformat(pos['entry_date']).date()).days

    # Update position
    cursor.execute("""
        UPDATE positions SET
            status = 'closed',
            exit_date = ?,
            exit_price = ?,
            exit_reason = ?,
            realized_pnl = ?,
            realized_pnl_pct = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (datetime.now().date().isoformat(), exit_price, exit_reason,
          realized_pnl, realized_pnl_pct, position_id))

    # Add to trade history
    cursor.execute("""
        INSERT INTO trade_history (
            symbol, bucket_id, entry_date, entry_price, exit_date, exit_price,
            shares, position_value, realized_pnl, realized_pnl_pct, hold_days,
            exit_reason, signal_type, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pos['symbol'], pos['bucket_id'], pos['entry_date'], pos['entry_price'],
        datetime.now().date().isoformat(), exit_price, pos['shares'],
        pos['position_value'], realized_pnl, realized_pnl_pct, hold_days,
        exit_reason, pos['signal_type'], pos['sector']
    ))

    conn.commit()
    conn.close()

    # Add cash back
    cash = get_cash()
    set_cash(cash + exit_value)


def log_signal(symbol: str, date: str, bucket_id: int, signal_type: str,
               action: str, reason: str):
    """Log a signal event."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO signal_log (symbol, date, bucket_id, signal_type, action, reason)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (symbol, date, bucket_id, signal_type, action, reason))
    conn.commit()
    conn.close()


def save_daily_snapshot():
    """Save daily portfolio snapshot."""
    positions = get_open_positions()
    cash = get_cash()

    if len(positions) > 0:
        positions_value = positions['current_value'].sum()
        unrealized_pnl = positions['unrealized_pnl'].sum()
    else:
        positions_value = 0
        unrealized_pnl = 0

    # Include options
    options_positions = get_open_options_positions()
    options_cash = get_options_cash()

    if len(options_positions) > 0:
        options_value = options_positions['current_value'].sum()
        options_unrealized = options_positions['unrealized_pnl'].sum()
    else:
        options_value = 0
        options_unrealized = 0

    total_cash = cash + options_cash
    total_positions = positions_value + options_value
    total_equity = total_cash + total_positions
    total_unrealized = unrealized_pnl + options_unrealized

    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO daily_snapshots (
            date, cash, positions_value, total_equity, open_positions, unrealized_pnl
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.now().date().isoformat(), total_cash, total_positions, total_equity,
          len(positions) + len(options_positions), total_unrealized))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS-SPECIFIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_options_cash() -> float:
    """Get current options cash balance."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT cash FROM portfolio_allocations WHERE allocation_type = 'options'")
    row = cursor.fetchone()
    conn.close()
    if row:
        return float(row[0])
    # Initialize if not exists
    set_options_cash(OPTIONS_STRATEGY['starting_capital'])
    return OPTIONS_STRATEGY['starting_capital']


def set_options_cash(amount: float):
    """Set options cash balance."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO portfolio_allocations (allocation_type, cash, starting_capital, updated_at)
        VALUES ('options', ?, ?, CURRENT_TIMESTAMP)
    """, (amount, OPTIONS_STRATEGY['starting_capital']))
    conn.commit()
    conn.close()


def get_open_options_positions() -> pd.DataFrame:
    """Get all open options positions."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query("""
        SELECT * FROM options_positions WHERE status = 'open' ORDER BY entry_date
    """, conn)
    conn.close()
    return df


def get_options_position_count() -> int:
    """Get count of open options positions."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM options_positions WHERE status = 'open'")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def is_options_symbol_held(symbol: str) -> bool:
    """Check if we already hold options on this symbol."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM options_positions WHERE symbol = ? AND status = 'open'", (symbol,))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def open_options_position(symbol: str, underlying_price: float, signal_type: str,
                          signal_date: str, sector: str, earnings_beat: bool) -> bool:
    """Open a new options paper position."""
    # Check position limits
    current_count = get_options_position_count()
    if current_count >= OPTIONS_STRATEGY['max_positions']:
        log_signal(symbol, signal_date, 0, signal_type,
                   'skipped_options_max', f'Already at max {OPTIONS_STRATEGY["max_positions"]} options positions')
        return False

    # Check if already held
    if is_options_symbol_held(symbol):
        log_signal(symbol, signal_date, 0, signal_type,
                   'skipped_options_held', f'{symbol} options already in portfolio')
        return False

    # Calculate position
    cash = get_options_cash()
    pos_pct = OPTIONS_STRATEGY['position_pct'] / 100
    position_value = cash * pos_pct

    # Calculate option parameters
    premium_pct = OPTIONS_STRATEGY['premium_pct']  # 3.5% for OTM_5
    premium_per_share = underlying_price * premium_pct
    premium_per_contract = premium_per_share * 100  # 100 shares per contract

    contracts = max(1, int(position_value / premium_per_contract))
    total_premium = contracts * premium_per_contract

    # Strike price (5% OTM for calls)
    strike_price = underlying_price * 1.05

    # Expiration (60 days out)
    entry_date = datetime.now().date()
    expiration_date = entry_date + timedelta(days=OPTIONS_STRATEGY['hold_days'])

    # Insert position
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO options_positions (
            symbol, underlying_price, strike_price, option_type, contracts,
            premium_per_contract, total_premium, entry_date, expiration_date,
            stop_loss_pct, profit_target_pct, current_value, unrealized_pnl,
            unrealized_pnl_pct, signal_type, signal_date, earnings_beat, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        symbol, underlying_price, strike_price, OPTIONS_STRATEGY['option_type'],
        contracts, premium_per_contract, total_premium, entry_date.isoformat(),
        expiration_date.isoformat(), OPTIONS_STRATEGY['stop_loss_pct'],
        OPTIONS_STRATEGY['profit_target_pct'], total_premium, 0, 0,
        signal_type, signal_date, 1 if earnings_beat else 0, sector
    ))
    conn.commit()
    conn.close()

    # Deduct cash
    set_options_cash(cash - total_premium)

    log_signal(symbol, signal_date, 0, signal_type, 'opened_options',
               f'Opened {contracts} contracts @ ${premium_per_contract:.2f}/contract (total ${total_premium:.2f})')

    return True


def close_options_position(position_id: int, exit_value: float, exit_reason: str):
    """Close an options position."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()

    # Get position details
    cursor.execute("SELECT * FROM options_positions WHERE id = ?", (position_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return

    cols = [d[0] for d in cursor.description]
    pos = dict(zip(cols, row))

    # Calculate P&L
    realized_pnl = exit_value - pos['total_premium']
    realized_pnl_pct = (realized_pnl / pos['total_premium']) * 100
    hold_days = (datetime.now().date() - datetime.fromisoformat(pos['entry_date']).date()).days

    # Update position
    cursor.execute("""
        UPDATE options_positions SET
            status = 'closed',
            exit_date = ?,
            exit_value = ?,
            exit_reason = ?,
            realized_pnl = ?,
            realized_pnl_pct = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ?
    """, (datetime.now().date().isoformat(), exit_value, exit_reason,
          realized_pnl, realized_pnl_pct, position_id))

    # Add to trade history
    cursor.execute("""
        INSERT INTO options_trade_history (
            symbol, option_type, contracts, entry_date, entry_premium, exit_date,
            exit_value, total_premium, realized_pnl, realized_pnl_pct, hold_days,
            exit_reason, signal_type, sector
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pos['symbol'], pos['option_type'], pos['contracts'], pos['entry_date'],
        pos['total_premium'], datetime.now().date().isoformat(), exit_value,
        pos['total_premium'], realized_pnl, realized_pnl_pct, hold_days,
        exit_reason, pos['signal_type'], pos['sector']
    ))

    conn.commit()
    conn.close()

    # Add cash back
    cash = get_options_cash()
    set_options_cash(cash + exit_value)


def estimate_option_value(pos: dict, current_stock_price: float) -> float:
    """Estimate current option value based on stock price movement."""
    # Simple model: delta * stock_move + time_decay
    underlying_price = pos['underlying_price']
    stock_return = (current_stock_price - underlying_price) / underlying_price

    # OTM_5 parameters
    delta = 0.35
    premium_pct = 0.035
    leverage = 1.0 / premium_pct  # ~28x

    # Days held
    entry_date = datetime.fromisoformat(pos['entry_date'])
    days_held = (datetime.now().date() - entry_date.date()).days
    hold_fraction = min(1.0, days_held / 90)
    time_decay = 0.35 * hold_fraction  # 35% decay over full period

    if stock_return > 0:
        # Winning trade - less time decay
        effective_leverage = delta * leverage
        option_return = effective_leverage * stock_return - (time_decay * 0.3)
        if stock_return > 0.15:
            option_return *= 1.3  # Gamma boost
    else:
        # Losing trade
        delta_loss = delta * abs(stock_return) / premium_pct
        option_return = -(delta_loss + time_decay)

    option_return = max(-1.0, min(4.0, option_return))  # Cap at -100% to +400%

    current_value = pos['total_premium'] * (1 + option_return)
    return max(0, current_value)  # Can't be negative


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_scan(days_back: int = 7, asset_type: str = 'both'):
    """Scan for new signals and open positions."""
    print("\n" + "=" * 80)
    print("  SCANNING FOR NEW SIGNALS")
    print(f"  Asset type: {asset_type.upper()}")
    print("=" * 80)

    print(f"\n  Loading signals from last {days_back} days...")
    df = load_recent_signals(days_back)
    print(f"  Found {len(df):,} total signals")

    stock_opened = 0
    stock_skipped = 0
    options_opened = 0
    options_skipped = 0

    # Scan for STOCK positions
    if asset_type in ['both', 'stocks']:
        print("\n" + "-" * 40)
        print("  STOCK POSITIONS")
        print("-" * 40)

        for bucket in BUCKETS:
            print(f"\n  Bucket {bucket.id}: {bucket.name}")
            signals = filter_bucket_signals(df, bucket)
            print(f"    {len(signals)} signals pass filters")

            # Sort by value_score_v2 descending
            signals = signals.sort_values('value_score_v2', ascending=False)

            for _, row in signals.iterrows():
                # Get current price
                price = get_current_price(row['symbol'])
                if not price:
                    price = row['close_price']

                if price and price > 0:
                    earnings_beat = pd.notna(row.get('earnings_surprise_pct')) and row['earnings_surprise_pct'] > 0

                    success = open_position(
                        symbol=row['symbol'],
                        bucket_id=bucket.id,
                        entry_price=price,
                        signal_type=row['signal_type'],
                        signal_date=row['date'],
                        sector=row.get('sector', ''),
                        earnings_beat=earnings_beat
                    )

                    if success:
                        stock_opened += 1
                        print(f"    ✓ Opened: {row['symbol']} @ ${price:.2f} "
                              f"({row['signal_type']}, V2={row['value_score_v2']:.0f})")
                    else:
                        stock_skipped += 1

    # Scan for OPTIONS positions
    if asset_type in ['both', 'options']:
        print("\n" + "-" * 40)
        print("  OPTIONS POSITIONS (OTM5_EarningsBeat)")
        print("-" * 40)

        # Filter for options: earnings beat + strong sectors
        options_signals = df[
            (df['earnings_surprise_pct'] > 0) &  # Earnings beat required
            (df['sector'].isin(['Technology', 'Consumer Cyclical', 'Financial Services', 'Industrials']))
        ].copy()

        # Additional quality filters
        options_signals = options_signals[
            (options_signals['value_score_v2'] >= 50) &
            (options_signals['fundamentals_score'] >= 15)
        ]

        options_signals = options_signals.sort_values('earnings_surprise_pct', ascending=False)
        print(f"    {len(options_signals)} signals pass options filters")

        for _, row in options_signals.iterrows():
            # Get current price
            price = get_current_price(row['symbol'])
            if not price:
                price = row['close_price']

            if price and price > 0:
                success = open_options_position(
                    symbol=row['symbol'],
                    underlying_price=price,
                    signal_type=row['signal_type'],
                    signal_date=row['date'],
                    sector=row.get('sector', ''),
                    earnings_beat=True
                )

                if success:
                    options_opened += 1
                    print(f"    ✓ Opened OPTIONS: {row['symbol']} @ ${price:.2f} "
                          f"(EPS surprise: {row['earnings_surprise_pct']:+.1f}%)")
                else:
                    options_skipped += 1

    print(f"\n  Summary:")
    if asset_type in ['both', 'stocks']:
        print(f"    Stocks: {stock_opened} opened, {stock_skipped} skipped")
    if asset_type in ['both', 'options']:
        print(f"    Options: {options_opened} opened, {options_skipped} skipped")
    print("=" * 80)


def cmd_status():
    """Show current portfolio status (stocks + options)."""
    print("\n" + "=" * 100)
    print("  PORTFOLIO STATUS (Stocks & Options)")
    print("=" * 100)

    # ═══════════════════════════════════════════════════════════════════════════
    # STOCKS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    positions = get_open_positions()
    cash = get_cash()

    # Update prices
    if len(positions) > 0:
        print("\n  Updating stock prices...")
        symbols = positions['symbol'].tolist()
        prices = get_batch_prices(symbols)

        conn = sqlite3.connect(PORTFOLIO_DB)
        for _, pos in positions.iterrows():
            price = prices.get(pos['symbol'])
            if price:
                current_value = pos['shares'] * price
                unrealized_pnl = current_value - pos['position_value']
                unrealized_pnl_pct = (unrealized_pnl / pos['position_value']) * 100

                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE positions SET
                        current_price = ?,
                        current_value = ?,
                        unrealized_pnl = ?,
                        unrealized_pnl_pct = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (price, current_value, unrealized_pnl, unrealized_pnl_pct, pos['id']))
        conn.commit()
        conn.close()
        positions = get_open_positions()

    stock_value = positions['current_value'].sum() if len(positions) > 0 else 0
    stock_unrealized = positions['unrealized_pnl'].sum() if len(positions) > 0 else 0
    stock_equity = cash + stock_value

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIONS SECTION
    # ═══════════════════════════════════════════════════════════════════════════
    options_positions = get_open_options_positions()
    options_cash = get_options_cash()

    # Update options values
    if len(options_positions) > 0:
        print("  Updating options values...")
        symbols = options_positions['symbol'].tolist()
        prices = get_batch_prices(symbols)

        conn = sqlite3.connect(PORTFOLIO_DB)
        for _, pos in options_positions.iterrows():
            price = prices.get(pos['symbol'])
            if price:
                current_value = estimate_option_value(pos, price)
                unrealized_pnl = current_value - pos['total_premium']
                unrealized_pnl_pct = (unrealized_pnl / pos['total_premium']) * 100

                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE options_positions SET
                        current_value = ?,
                        unrealized_pnl = ?,
                        unrealized_pnl_pct = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (current_value, unrealized_pnl, unrealized_pnl_pct, pos['id']))
        conn.commit()
        conn.close()
        options_positions = get_open_options_positions()

    options_value = options_positions['current_value'].sum() if len(options_positions) > 0 else 0
    options_unrealized = options_positions['unrealized_pnl'].sum() if len(options_positions) > 0 else 0
    options_equity = options_cash + options_value

    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINED SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    total_equity = stock_equity + options_equity
    total_unrealized = stock_unrealized + options_unrealized

    print(f"\n  ┌{'─'*70}┐")
    print(f"  │  {'':50s} STOCKS    OPTIONS    TOTAL  │")
    print(f"  │  {'-'*66}  │")
    print(f"  │  Cash:            ${cash:>12,.0f}  ${options_cash:>10,.0f}  ${cash+options_cash:>10,.0f}  │")
    print(f"  │  Positions Value: ${stock_value:>12,.0f}  ${options_value:>10,.0f}  ${stock_value+options_value:>10,.0f}  │")
    print(f"  │  Total Equity:    ${stock_equity:>12,.0f}  ${options_equity:>10,.0f}  ${total_equity:>10,.0f}  │")
    print(f"  │  Unrealized P&L:  ${stock_unrealized:>+11,.0f}  ${options_unrealized:>+9,.0f}  ${total_unrealized:>+9,.0f}  │")
    stock_ret = (stock_equity/STOCK_STRATEGY['starting_capital']-1)*100 if STOCK_STRATEGY['starting_capital'] > 0 else 0
    opt_ret = (options_equity/OPTIONS_STRATEGY['starting_capital']-1)*100 if OPTIONS_STRATEGY['starting_capital'] > 0 else 0
    total_ret = (total_equity/TOTAL_STARTING_CAPITAL-1)*100
    print(f"  │  Return:          {stock_ret:>+12.1f}%  {opt_ret:>+10.1f}%  {total_ret:>+10.1f}%  │")
    print(f"  │  Open Positions:  {len(positions):>12}  {len(options_positions):>10}  {len(positions)+len(options_positions):>10}  │")
    print(f"  └{'─'*70}┘")

    # Stock positions detail
    if len(positions) > 0:
        print("\n  STOCK POSITIONS:")
        print(f"  {'Symbol':<8s} {'Bucket':>6s} {'Entry':>8s} {'Current':>8s} {'P&L':>10s} {'P&L%':>7s} {'Stop':>8s} {'Days':>5s}")
        print(f"  {'-'*68}")

        for _, pos in positions.iterrows():
            entry_date = datetime.fromisoformat(pos['entry_date'])
            days_held = (datetime.now().date() - entry_date.date()).days
            stop_status = "⚠" if pos['current_price'] and pos['current_price'] <= pos['stop_loss_price'] else ""

            print(f"  {pos['symbol']:<8s} {pos['bucket_id']:>6} ${pos['entry_price']:>7.2f} "
                  f"${pos['current_price'] or 0:>7.2f} ${pos['unrealized_pnl'] or 0:>+9.2f} "
                  f"{pos['unrealized_pnl_pct'] or 0:>+6.1f}% ${pos['stop_loss_price']:>7.2f} {days_held:>5} {stop_status}")

    # Options positions detail
    if len(options_positions) > 0:
        print("\n  OPTIONS POSITIONS:")
        print(f"  {'Symbol':<8s} {'Type':<8s} {'Contracts':>9s} {'Premium':>10s} {'Current':>10s} {'P&L':>10s} {'P&L%':>7s} {'Days':>5s}")
        print(f"  {'-'*78}")

        for _, pos in options_positions.iterrows():
            entry_date = datetime.fromisoformat(pos['entry_date'])
            days_held = (datetime.now().date() - entry_date.date()).days
            exp_date = datetime.fromisoformat(pos['expiration_date']).date()
            days_to_exp = (exp_date - datetime.now().date()).days

            print(f"  {pos['symbol']:<8s} {pos['option_type']:<8s} {pos['contracts']:>9} "
                  f"${pos['total_premium']:>9.0f} ${pos['current_value'] or 0:>9.0f} "
                  f"${pos['unrealized_pnl'] or 0:>+9.0f} {pos['unrealized_pnl_pct'] or 0:>+6.1f}% "
                  f"{days_held:>3}/{days_to_exp:>2}d")

    # Per-bucket breakdown (stocks)
    print("\n  STOCKS BY BUCKET:")
    for bucket in BUCKETS:
        bucket_pos = positions[positions['bucket_id'] == bucket.id]
        count = len(bucket_pos)
        max_pos = STRATEGY_CONFIG['max_positions'].get(bucket.id, bucket.max_positions)
        pnl = bucket_pos['unrealized_pnl'].sum() if count > 0 else 0
        print(f"    Bucket {bucket.id} ({bucket.short_name}): {count}/{max_pos} positions, "
              f"${pnl:+,.2f} unrealized")

    print(f"\n  OPTIONS: {len(options_positions)}/{OPTIONS_STRATEGY['max_positions']} positions, "
          f"${options_unrealized:+,.2f} unrealized")

    print("\n" + "=" * 100)


def cmd_compare():
    """Show comparison of stock vs options performance."""
    print("\n" + "=" * 100)
    print("  STOCK vs OPTIONS COMPARISON")
    print("=" * 100)

    # Get stock data
    positions = get_open_positions()
    cash = get_cash()
    stock_value = positions['current_value'].sum() if len(positions) > 0 else 0
    stock_equity = cash + stock_value
    stock_start = STOCK_STRATEGY['starting_capital']
    stock_return = (stock_equity / stock_start - 1) * 100 if stock_start > 0 else 0

    # Get stock trade history
    conn = sqlite3.connect(PORTFOLIO_DB)
    stock_trades = pd.read_sql_query("SELECT * FROM trade_history", conn)
    stock_win_rate = (stock_trades['realized_pnl'] > 0).mean() * 100 if len(stock_trades) > 0 else 0

    # Get options data
    options_positions = get_open_options_positions()
    options_cash = get_options_cash()
    options_value = options_positions['current_value'].sum() if len(options_positions) > 0 else 0
    options_equity = options_cash + options_value
    options_start = OPTIONS_STRATEGY['starting_capital']
    options_return = (options_equity / options_start - 1) * 100 if options_start > 0 else 0

    # Get options trade history
    options_trades = pd.read_sql_query("SELECT * FROM options_trade_history", conn)
    options_win_rate = (options_trades['realized_pnl'] > 0).mean() * 100 if len(options_trades) > 0 else 0
    conn.close()

    # Combined
    total_start = TOTAL_STARTING_CAPITAL
    total_equity = stock_equity + options_equity
    total_return = (total_equity / total_start - 1) * 100

    print(f"\n  {'Metric':<25s} {'STOCKS':>15s} {'OPTIONS':>15s} {'COMBINED':>15s}")
    print(f"  {'-'*70}")
    print(f"  {'Starting Capital':<25s} ${stock_start:>13,.0f} ${options_start:>13,.0f} ${total_start:>13,.0f}")
    print(f"  {'Current Equity':<25s} ${stock_equity:>13,.0f} ${options_equity:>13,.0f} ${total_equity:>13,.0f}")
    print(f"  {'Return':<25s} {stock_return:>+13.1f}% {options_return:>+13.1f}% {total_return:>+13.1f}%")
    print(f"  {'Open Positions':<25s} {len(positions):>14} {len(options_positions):>14} {len(positions)+len(options_positions):>14}")
    print(f"  {'Completed Trades':<25s} {len(stock_trades):>14} {len(options_trades):>14} {len(stock_trades)+len(options_trades):>14}")
    print(f"  {'Win Rate':<25s} {stock_win_rate:>13.1f}% {options_win_rate:>13.1f}% {'N/A':>14s}")

    # Backtest comparison
    print(f"\n  BACKTEST REFERENCE (2024-2025):")
    print(f"  {'-'*70}")
    print(f"  {'Best Stock Strategy':<25s} {'H_RegimeAdaptive':>15s} {'+15.8%':>15s}")
    print(f"  {'Best Options Strategy':<25s} {'OTM5_EarningsBeat':>15s} {'+242.2%':>15s}")
    print(f"  {'Options Multiplier':<25s} {'':>15s} {'15.3x':>15s}")

    print("\n" + "=" * 100)


def cmd_history():
    """Show trade history."""
    print("\n" + "=" * 80)
    print("  TRADE HISTORY")
    print("=" * 80)

    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query("""
        SELECT * FROM trade_history ORDER BY exit_date DESC LIMIT 50
    """, conn)
    conn.close()

    if len(df) == 0:
        print("\n  No completed trades yet.")
        return

    total_pnl = df['realized_pnl'].sum()
    win_rate = (df['realized_pnl'] > 0).mean() * 100
    avg_win = df[df['realized_pnl'] > 0]['realized_pnl_pct'].mean() if len(df[df['realized_pnl'] > 0]) > 0 else 0
    avg_loss = df[df['realized_pnl'] <= 0]['realized_pnl_pct'].mean() if len(df[df['realized_pnl'] <= 0]) > 0 else 0

    print(f"\n  Total Trades: {len(df)}")
    print(f"  Total P&L: ${total_pnl:+,.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Avg Win: {avg_win:+.1f}% | Avg Loss: {avg_loss:+.1f}%")

    print(f"\n  {'Symbol':<8s} {'Bucket':>6s} {'Entry':>10s} {'Exit':>10s} {'P&L':>10s} {'P&L%':>7s} {'Days':>5s} {'Reason':<12s}")
    print(f"  {'-'*78}")

    for _, row in df.head(20).iterrows():
        print(f"  {row['symbol']:<8s} {row['bucket_id']:>6} "
              f"{row['entry_date'][:10]:>10s} {row['exit_date'][:10]:>10s} "
              f"${row['realized_pnl']:>+9.2f} {row['realized_pnl_pct']:>+6.1f}% "
              f"{row['hold_days']:>5} {row['exit_reason']:<12s}")

    print("\n" + "=" * 80)


def cmd_daily():
    """Daily update: check stops, update prices, save snapshot."""
    print("\n" + "=" * 80)
    print("  DAILY UPDATE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    positions = get_open_positions()
    if len(positions) == 0:
        print("\n  No open positions.")
        save_daily_snapshot()
        return

    # Update prices
    print("\n  Updating prices...")
    symbols = positions['symbol'].tolist()
    prices = get_batch_prices(symbols)

    stops_hit = []
    holds_complete = []

    conn = sqlite3.connect(PORTFOLIO_DB)
    for _, pos in positions.iterrows():
        price = prices.get(pos['symbol'])
        if not price:
            continue

        current_value = pos['shares'] * price
        unrealized_pnl = current_value - pos['position_value']
        unrealized_pnl_pct = (unrealized_pnl / pos['position_value']) * 100

        cursor = conn.cursor()
        cursor.execute("""
            UPDATE positions SET
                current_price = ?,
                current_value = ?,
                unrealized_pnl = ?,
                unrealized_pnl_pct = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (price, current_value, unrealized_pnl, unrealized_pnl_pct, pos['id']))

        # Check stop loss
        if price <= pos['stop_loss_price']:
            stops_hit.append((pos['id'], pos['symbol'], price, unrealized_pnl_pct))

        # Check hold complete
        target_exit = datetime.fromisoformat(pos['target_exit_date']).date()
        if datetime.now().date() >= target_exit:
            holds_complete.append((pos['id'], pos['symbol'], price, unrealized_pnl_pct))

    conn.commit()
    conn.close()

    # Process stops
    if stops_hit:
        print(f"\n  STOP LOSSES HIT ({len(stops_hit)}):")
        for pos_id, symbol, price, pnl_pct in stops_hit:
            print(f"    ⚠ {symbol}: Stopped out @ ${price:.2f} ({pnl_pct:+.1f}%)")
            close_position(pos_id, price, 'stop_loss')

    # Process holds complete
    if holds_complete:
        print(f"\n  HOLD PERIODS COMPLETE ({len(holds_complete)}):")
        for pos_id, symbol, price, pnl_pct in holds_complete:
            print(f"    ✓ {symbol}: Hold complete @ ${price:.2f} ({pnl_pct:+.1f}%)")
            close_position(pos_id, price, 'hold_complete')

    # Save snapshot
    save_daily_snapshot()

    # Show summary
    cmd_status()


def cmd_reset():
    """Reset portfolio to starting state."""
    confirm = input("\n  Are you sure you want to reset the portfolio? (type 'yes'): ")
    if confirm.lower() != 'yes':
        print("  Reset cancelled.")
        return

    conn = sqlite3.connect(PORTFOLIO_DB)
    cursor = conn.cursor()
    # Stock tables
    cursor.execute("DELETE FROM positions")
    cursor.execute("DELETE FROM trade_history")
    cursor.execute("DELETE FROM daily_snapshots")
    cursor.execute("DELETE FROM signal_log")
    cursor.execute("DELETE FROM portfolio_config")
    # Options tables
    cursor.execute("DELETE FROM options_positions")
    cursor.execute("DELETE FROM options_trade_history")
    cursor.execute("DELETE FROM portfolio_allocations")
    conn.commit()
    conn.close()

    # Reset cash for both
    set_cash(STOCK_STRATEGY['starting_capital'])
    set_options_cash(OPTIONS_STRATEGY['starting_capital'])

    print(f"\n  Portfolio reset.")
    print(f"    Stock allocation: ${STOCK_STRATEGY['starting_capital']:,.0f} ({STOCK_ALLOCATION_PCT}%)")
    print(f"    Options allocation: ${OPTIONS_STRATEGY['starting_capital']:,.0f} ({OPTIONS_ALLOCATION_PCT}%)")
    print(f"    Total: ${TOTAL_STARTING_CAPITAL:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    setup_portfolio_db()

    parser = argparse.ArgumentParser(description='Mock Portfolio Tracker (Stocks & Options)')
    parser.add_argument('command', choices=['scan', 'status', 'history', 'daily', 'reset', 'compare'],
                        help='Command to run')
    parser.add_argument('--days', type=int, default=7, help='Days back to scan (for scan command)')
    parser.add_argument('--type', choices=['stocks', 'options', 'both'], default='both',
                        help='Asset type to scan (for scan command)')

    args = parser.parse_args()

    if args.command == 'scan':
        cmd_scan(args.days, args.type)
    elif args.command == 'status':
        cmd_status()
    elif args.command == 'history':
        cmd_history()
    elif args.command == 'daily':
        cmd_daily()
    elif args.command == 'reset':
        cmd_reset()
    elif args.command == 'compare':
        cmd_compare()


if __name__ == "__main__":
    main()

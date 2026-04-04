#!/usr/bin/env python3
"""
Custom Portfolio Manager
========================
Allows users to create and track multiple custom portfolios.
"""

import sqlite3
import pandas as pd
import requests
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

PROJECT_ROOT = Path(__file__).parent
PORTFOLIO_DB = str(PROJECT_ROOT / 'custom_portfolios.db')


def get_fmp_api_key():
    """Get FMP API key from environment or secrets."""
    try:
        import streamlit as st
        return st.secrets.get("FMP_API_KEY", os.environ.get("FMP_API_KEY", ""))
    except Exception:
        return os.environ.get("FMP_API_KEY", "")


def init_database():
    """Initialize the custom portfolio database."""
    conn = sqlite3.connect(PORTFOLIO_DB)

    # Portfolios table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    ''')

    # Positions table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            shares REAL NOT NULL,
            entry_price REAL NOT NULL,
            entry_date TEXT,
            notes TEXT,
            created_at TEXT,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            UNIQUE(portfolio_id, symbol)
        )
    ''')

    # Daily snapshots for tracking
    conn.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            total_value REAL,
            total_cost REAL,
            pnl REAL,
            pnl_pct REAL,
            FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
            UNIQUE(portfolio_id, date)
        )
    ''')

    conn.commit()
    conn.close()


def create_portfolio(name: str, description: str = "") -> int:
    """Create a new portfolio. Returns portfolio ID."""
    init_database()
    conn = sqlite3.connect(PORTFOLIO_DB)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    try:
        cur = conn.execute('''
            INSERT INTO portfolios (name, description, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        ''', (name, description, now, now))
        portfolio_id = cur.lastrowid
        conn.commit()
    except sqlite3.IntegrityError:
        # Portfolio already exists, get its ID
        portfolio_id = conn.execute(
            'SELECT id FROM portfolios WHERE name = ?', (name,)
        ).fetchone()[0]
    finally:
        conn.close()

    return portfolio_id


def delete_portfolio(portfolio_id: int):
    """Delete a portfolio and all its positions."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    conn.execute('DELETE FROM positions WHERE portfolio_id = ?', (portfolio_id,))
    conn.execute('DELETE FROM portfolio_snapshots WHERE portfolio_id = ?', (portfolio_id,))
    conn.execute('DELETE FROM portfolios WHERE id = ?', (portfolio_id,))
    conn.commit()
    conn.close()


def get_portfolios() -> pd.DataFrame:
    """Get all portfolios."""
    init_database()
    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query('''
        SELECT p.id, p.name, p.description, p.created_at,
               COUNT(pos.id) as num_positions
        FROM portfolios p
        LEFT JOIN positions pos ON p.id = pos.portfolio_id
        GROUP BY p.id
        ORDER BY p.name
    ''', conn)
    conn.close()
    return df


def add_position(portfolio_id: int, symbol: str, shares: float,
                 entry_price: float, entry_date: str = None, notes: str = "") -> bool:
    """Add a position to a portfolio."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry_date = entry_date or datetime.now().strftime('%Y-%m-%d')

    try:
        conn.execute('''
            INSERT OR REPLACE INTO positions
            (portfolio_id, symbol, shares, entry_price, entry_date, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (portfolio_id, symbol.upper(), shares, entry_price, entry_date, notes, now))
        conn.commit()
        success = True
    except Exception as e:
        print(f"Error adding position: {e}")
        success = False
    finally:
        conn.close()

    return success


def remove_position(portfolio_id: int, symbol: str):
    """Remove a position from a portfolio."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    conn.execute('''
        DELETE FROM positions WHERE portfolio_id = ? AND symbol = ?
    ''', (portfolio_id, symbol.upper()))
    conn.commit()
    conn.close()


def get_positions(portfolio_id: int) -> pd.DataFrame:
    """Get all positions in a portfolio."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query('''
        SELECT symbol, shares, entry_price, entry_date, notes,
               (shares * entry_price) as cost_basis
        FROM positions
        WHERE portfolio_id = ?
        ORDER BY symbol
    ''', conn, params=(portfolio_id,))
    conn.close()
    return df


def fetch_current_prices(symbols: list) -> dict:
    """Fetch current prices for symbols."""
    api_key = get_fmp_api_key()
    if not api_key or not symbols:
        return {}

    prices = {}
    symbols_str = ','.join(symbols)
    url = f"https://financialmodelingprep.com/stable/batch-quote?symbols={symbols_str}&apikey={api_key}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            for item in resp.json():
                if 'symbol' in item and 'price' in item:
                    prices[item['symbol']] = item['price']
    except Exception as e:
        print(f"Error fetching prices: {e}")

    return prices


def get_portfolio_summary(portfolio_id: int) -> dict:
    """Get portfolio summary with current values and P&L."""
    positions = get_positions(portfolio_id)

    if positions.empty:
        return {
            'total_cost': 0,
            'total_value': 0,
            'pnl': 0,
            'pnl_pct': 0,
            'positions': positions
        }

    # Fetch current prices
    symbols = positions['symbol'].tolist()
    prices = fetch_current_prices(symbols)

    # Calculate current values
    positions['current_price'] = positions['symbol'].map(prices)
    positions['current_value'] = positions['shares'] * positions['current_price']
    positions['pnl'] = positions['current_value'] - positions['cost_basis']
    positions['pnl_pct'] = (positions['pnl'] / positions['cost_basis'] * 100).round(2)

    total_cost = positions['cost_basis'].sum()
    total_value = positions['current_value'].sum()
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0

    return {
        'total_cost': total_cost,
        'total_value': total_value,
        'pnl': total_pnl,
        'pnl_pct': total_pnl_pct,
        'positions': positions
    }


def save_portfolio_snapshot(portfolio_id: int):
    """Save daily snapshot of portfolio value."""
    summary = get_portfolio_summary(portfolio_id)
    today = datetime.now().strftime('%Y-%m-%d')

    conn = sqlite3.connect(PORTFOLIO_DB)
    conn.execute('''
        INSERT OR REPLACE INTO portfolio_snapshots
        (portfolio_id, date, total_value, total_cost, pnl, pnl_pct)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (portfolio_id, today, summary['total_value'], summary['total_cost'],
          summary['pnl'], summary['pnl_pct']))
    conn.commit()
    conn.close()


def get_portfolio_history(portfolio_id: int) -> pd.DataFrame:
    """Get historical snapshots for a portfolio."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    df = pd.read_sql_query('''
        SELECT date, total_value, total_cost, pnl, pnl_pct
        FROM portfolio_snapshots
        WHERE portfolio_id = ?
        ORDER BY date
    ''', conn, params=(portfolio_id,))
    conn.close()
    return df


# Initialize database on import
init_database()

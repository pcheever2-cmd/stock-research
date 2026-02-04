#!/usr/bin/env python3
"""
Update Hybrid Portfolio
=======================
Fetches current prices and updates P&L for all hybrid portfolio positions.
Run this daily (or during market hours) to track performance.
"""

import sqlite3
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import os

try:
    from config import FMP_API_KEY
except ImportError:
    FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

PROJECT_ROOT = Path(__file__).parent
PORTFOLIO_DB = str(PROJECT_ROOT / 'mock_portfolio.db')


def fetch_prices(symbols: list) -> dict:
    """Fetch current prices from FMP."""
    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY - using placeholder prices")
        return {}

    symbols_str = ','.join(symbols)
    url = f"https://financialmodelingprep.com/stable/batch-quote?symbols={symbols_str}&apikey={FMP_API_KEY}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {item['symbol']: item['price'] for item in data if 'price' in item}
    except Exception as e:
        print(f"Error fetching prices: {e}")

    return {}


def update_portfolio():
    """Update all positions with current prices."""
    conn = sqlite3.connect(PORTFOLIO_DB)

    # Get all open positions
    positions = pd.read_sql_query('''
        SELECT id, symbol, position_type, entry_date, cost_basis, entry_price
        FROM hybrid_positions WHERE status = 'open'
    ''', conn)

    if positions.empty:
        print("No open positions found.")
        conn.close()
        return

    # Fetch current prices
    symbols = positions['symbol'].unique().tolist()
    prices = fetch_prices(symbols)

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today = datetime.now().strftime('%Y-%m-%d')

    print("=" * 70)
    print(f"HYBRID PORTFOLIO UPDATE - {now}")
    print("=" * 70)

    # Track totals by position type
    totals = {'mega_cap': 0, 'options': 0, 'growth': 0}
    costs = {'mega_cap': 0, 'options': 0, 'growth': 0}

    for pos_type in ['mega_cap', 'options', 'growth']:
        type_positions = positions[positions['position_type'] == pos_type]
        if type_positions.empty:
            continue

        emoji = {'mega_cap': '📊', 'options': '🎯', 'growth': '🌱'}[pos_type]
        print(f"\n{emoji} {pos_type.upper()}")
        print("-" * 50)
        print(f"{'Symbol':<8} {'Cost':>10} {'Price':>10} {'Value':>10} {'P&L':>10} {'%':>8}")

        for _, row in type_positions.iterrows():
            symbol = row['symbol']
            cost_basis = row['cost_basis']
            entry_price = row['entry_price']

            current_price = prices.get(symbol)

            if current_price and entry_price:
                # Calculate based on shares
                shares = cost_basis / entry_price
                current_value = shares * current_price
            elif current_price and cost_basis:
                # For options, use simplified model (premium-based)
                if pos_type == 'options':
                    # Options P&L approximation: if underlying up X%, option up ~3X%
                    # This is simplified - real tracking would need option prices
                    current_value = cost_basis  # Placeholder
                else:
                    current_value = cost_basis  # No entry price, use cost as placeholder
            else:
                current_value = cost_basis  # No price data

            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0

            # Update database
            conn.execute('''
                UPDATE hybrid_positions
                SET current_price = ?, current_value = ?, pnl = ?, pnl_pct = ?, updated_at = ?
                WHERE id = ?
            ''', (current_price, current_value, pnl, pnl_pct, now, row['id']))

            totals[pos_type] += current_value
            costs[pos_type] += cost_basis

            price_str = f"${current_price:.2f}" if current_price else "N/A"
            print(f"{symbol:<8} ${cost_basis:>9,.0f} {price_str:>10} ${current_value:>9,.0f} ${pnl:>+9,.0f} {pnl_pct:>+7.1f}%")

        type_pnl = totals[pos_type] - costs[pos_type]
        type_pnl_pct = (type_pnl / costs[pos_type] * 100) if costs[pos_type] else 0
        print(f"{'SUBTOTAL':<8} ${costs[pos_type]:>9,.0f} {'':>10} ${totals[pos_type]:>9,.0f} ${type_pnl:>+9,.0f} {type_pnl_pct:>+7.1f}%")

    # Overall summary
    total_cost = sum(costs.values())
    total_value = sum(totals.values())
    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

    print("\n" + "=" * 70)
    print("PORTFOLIO SUMMARY")
    print("=" * 70)
    print(f"Total Cost Basis:  ${total_cost:>12,.0f}")
    print(f"Current Value:     ${total_value:>12,.0f}")
    print(f"P&L:               ${total_pnl:>+12,.0f} ({total_pnl_pct:+.1f}%)")

    # Save daily snapshot
    # Get SPY price for comparison
    spy_price = prices.get('SPY', 0)
    conn.execute('''
        INSERT OR REPLACE INTO hybrid_daily_snapshot
        (date, mega_cap_value, options_value, growth_value, total_value, spy_value, daily_pnl, cumulative_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (today, totals['mega_cap'], totals['options'], totals['growth'],
          total_value, spy_price, total_pnl, total_pnl))

    conn.commit()
    conn.close()

    print(f"\nSnapshot saved for {today}")


if __name__ == '__main__':
    update_portfolio()

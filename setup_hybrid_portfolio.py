#!/usr/bin/env python3
"""
Setup Hybrid Portfolio for Forward Testing
==========================================
Creates the initial portfolio positions for the hybrid strategy:
- 55% Mega-Cap Core (buy and hold)
- 25% Options Alpha (OTM5 calls on signals)
- 20% Growth Picks (long-term holds from Buckets 1/3)

Run this script to initialize the portfolio for tomorrow's market open.
"""

import sqlite3
import pandas as pd
import requests
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    from config import FMP_API_KEY
except ImportError:
    FMP_API_KEY = os.environ.get('FMP_API_KEY', '')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
PORTFOLIO_DB = str(PROJECT_ROOT / 'mock_portfolio.db')


def fetch_prices(symbols: list) -> dict:
    """Fetch current prices from FMP."""
    if not FMP_API_KEY:
        print("Warning: No FMP_API_KEY - entry prices will be set on first update")
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

# Portfolio settings
TOTAL_CAPITAL = 100000
ALLOCATION = {
    'mega_cap': 0.55,    # $55,000
    'options': 0.25,     # $25,000
    'growth': 0.20,      # $20,000
}

# Mega-cap core allocation
MEGA_CAPS = [
    ('AAPL', 0.15, 'Apple'),
    ('MSFT', 0.15, 'Microsoft'),
    ('NVDA', 0.15, 'NVIDIA'),
    ('GOOG', 0.12, 'Alphabet'),
    ('AMZN', 0.12, 'Amazon'),
    ('META', 0.10, 'Meta'),
    ('TSLA', 0.08, 'Tesla'),
    ('BRK-B', 0.05, 'Berkshire Hathaway'),
    ('AVGO', 0.04, 'Broadcom'),
    ('LLY', 0.04, 'Eli Lilly'),
]


def setup_database():
    """Create portfolio database tables."""
    conn = sqlite3.connect(PORTFOLIO_DB)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS hybrid_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            position_type TEXT NOT NULL,  -- 'mega_cap', 'options', 'growth'
            entry_date TEXT NOT NULL,
            entry_price REAL,
            shares REAL,
            cost_basis REAL,
            current_price REAL,
            current_value REAL,
            pnl REAL,
            pnl_pct REAL,
            status TEXT DEFAULT 'open',  -- 'open', 'closed'
            exit_date TEXT,
            exit_price REAL,
            notes TEXT,
            updated_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS hybrid_daily_snapshot (
            date TEXT PRIMARY KEY,
            mega_cap_value REAL,
            options_value REAL,
            growth_value REAL,
            total_value REAL,
            spy_value REAL,
            daily_pnl REAL,
            cumulative_pnl REAL
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS hybrid_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            position_type TEXT NOT NULL,
            trade_type TEXT NOT NULL,  -- 'buy', 'sell', 'expire'
            trade_date TEXT NOT NULL,
            price REAL,
            shares REAL,
            amount REAL,
            notes TEXT
        )
    ''')
    conn.commit()
    conn.close()


def get_current_signals():
    """Get current bucket signals for options and growth picks."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Get most recent date
    latest = conn.execute('SELECT MAX(date) FROM backtest_daily_scores').fetchone()[0]

    # Get signals with scores
    signals = pd.read_sql_query(f'''
        SELECT d.symbol, d.date, d.lt_score, d.value_score_v2, d.fundamentals_score,
               d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth, d.rsi,
               m.market_cap
        FROM backtest_daily_scores d
        LEFT JOIN (SELECT symbol, market_cap FROM historical_key_metrics
                   WHERE market_cap IS NOT NULL GROUP BY symbol HAVING date = MAX(date)) m
            ON d.symbol = m.symbol
        WHERE d.date = '{latest}'
    ''', conn)

    # Get analyst coverage
    acov = pd.read_sql_query('''
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot WHERE num_analysts_eps IS NOT NULL
        GROUP BY symbol
    ''', conn).set_index('symbol')['analyst_count']
    signals['analyst_count'] = signals['symbol'].map(acov).fillna(0)

    conn.close()
    return signals, latest


def filter_bucket_signals(signals):
    """Apply bucket filters to get options and growth candidates."""
    # Bucket 1: Quality Growth Compounder
    b1 = signals[
        (signals['value_score_v2'] >= 55) &
        (signals['fundamentals_score'] >= 18) &
        (signals['ev_ebitda'] >= 5) & (signals['ev_ebitda'] <= 20) &
        (signals['rev_growth'] > 10) &
        (signals['market_cap'] > 2e9) &
        (signals['analyst_count'] >= 6)
    ].copy()
    b1['bucket'] = 'B1'

    # Bucket 3: High-Growth Momentum
    b3 = signals[
        (signals['eps_growth'] >= 35) &
        (signals['ebitda_growth'] >= 33) &
        (signals['ev_ebitda'] >= 12) & (signals['ev_ebitda'] <= 27) &
        (signals['rsi'] < 43)
    ].copy()
    b3['bucket'] = 'B3'

    # Options: Bucket 1/3 with earnings beat (eps_growth > 15)
    options = pd.concat([b1, b3]).drop_duplicates(subset=['symbol'])
    options = options[options['eps_growth'] > 15]
    options = options.sort_values('fundamentals_score', ascending=False)

    # Growth: All Bucket 1/3 picks
    growth = pd.concat([b1, b3]).drop_duplicates(subset=['symbol'])
    growth = growth.sort_values(['fundamentals_score', 'eps_growth'], ascending=[False, False])

    return options, growth


def setup_portfolio():
    """Setup the hybrid portfolio with initial positions."""
    setup_database()

    signals, latest_date = get_current_signals()
    options_picks, growth_picks = filter_bucket_signals(signals)

    conn = sqlite3.connect(PORTFOLIO_DB)
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry_date = datetime.now().strftime('%Y-%m-%d')

    # Clear existing positions (fresh start)
    conn.execute("DELETE FROM hybrid_positions")
    conn.execute("DELETE FROM hybrid_trades")

    # Fetch current prices for all symbols
    all_symbols = [s[0] for s in MEGA_CAPS] + options_picks['symbol'].tolist() + growth_picks['symbol'].head(10).tolist()
    all_symbols = list(set(all_symbols))  # Dedupe
    prices = fetch_prices(all_symbols)

    print("=" * 70)
    print("SETTING UP HYBRID PORTFOLIO")
    print("=" * 70)
    print(f"Entry Date: {entry_date}")
    print(f"Signal Date: {latest_date}")
    print(f"Total Capital: ${TOTAL_CAPITAL:,}")
    print()

    # 1. MEGA-CAP CORE
    mega_cap_capital = TOTAL_CAPITAL * ALLOCATION['mega_cap']
    print(f"📊 MEGA-CAP CORE: ${mega_cap_capital:,.0f}")
    print("-" * 50)

    for symbol, weight, name in MEGA_CAPS:
        amount = mega_cap_capital * weight
        entry_price = prices.get(symbol)
        shares = amount / entry_price if entry_price else None

        conn.execute('''
            INSERT INTO hybrid_positions
            (symbol, position_type, entry_date, entry_price, shares, cost_basis, current_price, current_value, status, notes, updated_at)
            VALUES (?, 'mega_cap', ?, ?, ?, ?, ?, ?, 'open', ?, ?)
        ''', (symbol, entry_date, entry_price, shares, amount, entry_price, amount, f'{name} - {weight*100:.0f}% weight', now))

        conn.execute('''
            INSERT INTO hybrid_trades
            (symbol, position_type, trade_type, trade_date, price, shares, amount, notes)
            VALUES (?, 'mega_cap', 'buy', ?, ?, ?, ?, ?)
        ''', (symbol, entry_date, entry_price, shares, amount, f'Initial buy - {name}'))

        price_str = f"@ ${entry_price:.2f}" if entry_price else "(price pending)"
        print(f"  {symbol:<6} ${amount:>8,.0f}  ({weight*100:.0f}%)  {name} {price_str}")

    # 2. OPTIONS ALPHA
    options_capital = TOTAL_CAPITAL * ALLOCATION['options']
    print(f"\n🎯 OPTIONS ALPHA: ${options_capital:,.0f}")
    print("-" * 50)

    # Take top 10 options candidates
    top_options = options_picks.head(10)
    options_per_position = options_capital / len(top_options) if len(top_options) > 0 else 0

    for _, row in top_options.iterrows():
        symbol = row['symbol']
        entry_price = prices.get(symbol)

        conn.execute('''
            INSERT INTO hybrid_positions
            (symbol, position_type, entry_date, entry_price, cost_basis, current_price, current_value, status, notes, updated_at)
            VALUES (?, 'options', ?, ?, ?, ?, ?, 'open', ?, ?)
        ''', (symbol, entry_date, entry_price, options_per_position, entry_price, options_per_position,
              f'OTM5 Call - Fund:{row["fundamentals_score"]:.0f} EPS_G:{row["eps_growth"]:.0f}', now))

        conn.execute('''
            INSERT INTO hybrid_trades
            (symbol, position_type, trade_type, trade_date, price, amount, notes)
            VALUES (?, 'options', 'buy', ?, ?, ?, ?)
        ''', (symbol, entry_date, entry_price, options_per_position, f'OTM5 Call entry'))

        price_str = f"@ ${entry_price:.2f}" if entry_price else ""
        print(f"  {symbol:<6} ${options_per_position:>8,.0f}  Fund:{row['fundamentals_score']:.0f} EPS_G:{row['eps_growth']:.0f} {price_str}")

    # 3. GROWTH PICKS
    growth_capital = TOTAL_CAPITAL * ALLOCATION['growth']
    print(f"\n🌱 GROWTH PICKS: ${growth_capital:,.0f}")
    print("-" * 50)

    # Take top 10 growth candidates (different from options if possible)
    options_symbols = set(top_options['symbol'].tolist())
    growth_filtered = growth_picks[~growth_picks['symbol'].isin(options_symbols)]
    if len(growth_filtered) < 10:
        growth_filtered = growth_picks  # Fall back to all if not enough unique
    top_growth = growth_filtered.head(10)
    growth_per_position = growth_capital / len(top_growth) if len(top_growth) > 0 else 0

    for _, row in top_growth.iterrows():
        symbol = row['symbol']
        entry_price = prices.get(symbol)
        shares = growth_per_position / entry_price if entry_price else None

        conn.execute('''
            INSERT INTO hybrid_positions
            (symbol, position_type, entry_date, entry_price, shares, cost_basis, current_price, current_value, status, notes, updated_at)
            VALUES (?, 'growth', ?, ?, ?, ?, ?, ?, 'open', ?, ?)
        ''', (symbol, entry_date, entry_price, shares, growth_per_position, entry_price, growth_per_position,
              f'Long-term hold - {row["bucket"]} Fund:{row["fundamentals_score"]:.0f}', now))

        conn.execute('''
            INSERT INTO hybrid_trades
            (symbol, position_type, trade_type, trade_date, price, shares, amount, notes)
            VALUES (?, 'growth', 'buy', ?, ?, ?, ?, ?)
        ''', (symbol, entry_date, entry_price, shares, growth_per_position, f'Growth pick entry'))

        price_str = f"@ ${entry_price:.2f}" if entry_price else ""
        print(f"  {symbol:<6} ${growth_per_position:>8,.0f}  {row['bucket']} Fund:{row['fundamentals_score']:.0f} {price_str}")

    conn.commit()
    conn.close()

    print("\n" + "=" * 70)
    print("PORTFOLIO SETUP COMPLETE")
    print("=" * 70)
    print(f"Total positions: {len(MEGA_CAPS)} mega-cap + {len(top_options)} options + {len(top_growth)} growth")
    print(f"Database: {PORTFOLIO_DB}")
    print("\nNext steps:")
    print("1. Run 'update_hybrid_portfolio.py' daily to track P&L")
    print("2. View in dashboard under new 'Hybrid Portfolio' tab")


if __name__ == '__main__':
    setup_portfolio()

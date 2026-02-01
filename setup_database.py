#!/usr/bin/env python3
# File: setup_database.py — Full enhanced version with projection fields
import sqlite3
import os
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_NAME = os.path.join(SCRIPT_DIR, "nasdaq_stocks.db")

def ensure_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    print("Setting up/updating nasdaq_stocks.db ...")

    # 1. CREATE TABLES (IF NOT EXISTS)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickers (
            symbol TEXT PRIMARY KEY,
            name TEXT,
            market_cap REAL,
            fetched_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_data (
            symbol TEXT,
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Volume INTEGER,
            PRIMARY KEY (symbol, Date)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS article_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            date TEXT,
            title TEXT,
            sentiment REAL,
            UNIQUE(symbol, date, title)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_news_sentiment (
            symbol TEXT,
            date TEXT,
            avg_sentiment REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyst_ratings (
            benzinga_id TEXT PRIMARY KEY,
            symbol TEXT,
            date TEXT,
            firm TEXT,
            analyst TEXT,
            rating_current TEXT,
            rating_previous TEXT,
            price_target REAL,
            previous_price_target REAL,
            adjusted_price_target REAL,
            currency TEXT,
            importance INTEGER,
            fetched_at TEXT
        )
    """)

    # Enhanced stock_consensus with all value + projection fields
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_consensus (
            symbol TEXT PRIMARY KEY,
            avg_price_target REAL,
            median_price_target REAL,
            min_price_target REAL,
            max_price_target REAL,
            num_analysts INTEGER,
            consensus_rating TEXT,
            last_updated TEXT,
            current_price REAL,
            upside_percent REAL,
            cap_category TEXT,
            industry TEXT,
            recent_ratings TEXT,
            recommendation TEXT,
            enterprise_value REAL,
            ebitda REAL,
            ev_ebitda REAL,
            total_debt REAL,
            debt_ebitda REAL,
            earnings_growth REAL,
            projected_revenue_next_year REAL,
            projected_eps_next_year REAL,
            projected_eps_growth REAL,
            peg_ratio REAL,
            forward_pe REAL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)

    # 2. ADD MISSING COLUMNS SAFELY (for existing databases)
    new_columns = [
        ("enterprise_value", "REAL"),
        ("ebitda", "REAL"),
        ("ev_ebitda", "REAL"),
        ("total_debt", "REAL"),
        ("debt_ebitda", "REAL"),
        ("earnings_growth", "REAL"),
        ("projected_revenue_next_year", "REAL"),
        ("projected_eps_next_year", "REAL"),
        ("projected_eps_growth", "REAL"),
        ("peg_ratio", "REAL"),
        ("forward_pe", "REAL")
    ]

    cur.execute("PRAGMA table_info(stock_consensus)")
    existing_columns = [row[1] for row in cur.fetchall()]

    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            print(f"Adding new column to stock_consensus: {col_name}")
            cur.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col_name} {col_type}")

    # 3. CREATE INDEXES (IF NOT EXISTS)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_article_symbol_date ON article_sentiment(symbol, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_symbol_date ON daily_news_sentiment(symbol, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ratings_symbol_date ON analyst_ratings(symbol, date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ratings_symbol_firm ON analyst_ratings(symbol, firm)")

    # Performance indexes for dashboard
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_upside ON stock_consensus(upside_percent)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_cap ON stock_consensus(cap_category)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_industry ON stock_consensus(industry)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_analysts ON stock_consensus(num_analysts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_ev_ebitda ON stock_consensus(ev_ebitda)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_debt_ebitda ON stock_consensus(debt_ebitda)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_peg ON stock_consensus(peg_ratio)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_forward_pe ON stock_consensus(forward_pe)")

    # Metadata update
    cur.execute("""
        INSERT OR REPLACE INTO metadata (key, value, updated_at)
        VALUES ('database_schema_version', '3.0', ?)
    """, (datetime.utcnow().isoformat(),))

    conn.commit()
    conn.close()

    print(f"Database ready: {DATABASE_NAME}")
    print("All existing data preserved.")
    print("New projection columns added: projected_revenue_next_year, projected_eps_next_year,")
    print("projected_eps_growth, peg_ratio, forward_pe")

if __name__ == "__main__":
    ensure_database()
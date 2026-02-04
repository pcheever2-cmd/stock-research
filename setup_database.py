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
        ("forward_pe", "REAL"),
        # Phase 1: New data fields
        ("company_name", "TEXT"),
        ("sector", "TEXT"),
        ("ocf_ev", "REAL"),
        # Phase 3: Raw technical indicators for trend detection
        ("sma50", "REAL"),
        ("sma200", "REAL"),
        ("rsi", "REAL"),
        ("adx", "REAL"),
        ("close_price_technical", "REAL"),
        ("prev_sma50", "REAL"),
        ("prev_sma200", "REAL"),
        ("prev_rsi", "REAL"),
        ("prev_close_technical", "REAL"),
        ("trend_signal", "TEXT"),
        ("trend_signal_count", "INTEGER"),
        # Company description from FMP profile
        ("company_description", "TEXT"),
        # V2 scoring (continuous value score, backtested)
        ("value_score_v2", "INTEGER"),
        # Financial scores from FMP Financial Scores API
        ("altman_z_score", "REAL"),
        ("piotroski_score", "INTEGER"),
        # Market cap from profile (for stocks without analyst coverage)
        ("market_cap", "REAL"),
        # Exchange info
        ("exchange", "TEXT"),
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
    cur.execute("CREATE INDEX IF NOT EXISTS idx_consensus_sector ON stock_consensus(sector)")

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

def setup_backtest_tables():
    """Create tables in backtest.db for historical data collection"""
    from config import BACKTEST_DB

    conn = sqlite3.connect(BACKTEST_DB)
    cur = conn.cursor()

    print(f"Setting up/updating {BACKTEST_DB} ...")

    # Daily OHLCV prices (5 years)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_prices (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            adjusted_close REAL,
            PRIMARY KEY (symbol, date)
        )
    """)

    # Quarterly income statements
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_income_statements (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            period TEXT NOT NULL,
            fiscal_year INTEGER,
            revenue REAL,
            gross_profit REAL,
            operating_income REAL,
            net_income REAL,
            ebitda REAL,
            eps REAL,
            eps_diluted REAL,
            weighted_avg_shares_diluted REAL,
            filing_date TEXT,
            accepted_date TEXT,
            PRIMARY KEY (symbol, date, period)
        )
    """)

    # Quarterly balance sheets
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_balance_sheets (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            period TEXT NOT NULL,
            fiscal_year INTEGER,
            total_assets REAL,
            total_liabilities REAL,
            total_equity REAL,
            total_debt REAL,
            net_debt REAL,
            cash_and_equivalents REAL,
            filing_date TEXT,
            accepted_date TEXT,
            PRIMARY KEY (symbol, date, period)
        )
    """)

    # Quarterly cash flow statements
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_cash_flows (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            period TEXT NOT NULL,
            fiscal_year INTEGER,
            operating_cash_flow REAL,
            capital_expenditure REAL,
            free_cash_flow REAL,
            filing_date TEXT,
            accepted_date TEXT,
            PRIMARY KEY (symbol, date, period)
        )
    """)

    # Quarterly key metrics
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_key_metrics (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            period TEXT NOT NULL,
            fiscal_year INTEGER,
            enterprise_value REAL,
            ev_to_ebitda REAL,
            market_cap REAL,
            pe_ratio REAL,
            pb_ratio REAL,
            debt_to_equity REAL,
            roe REAL,
            revenue_per_share REAL,
            net_income_per_share REAL,
            operating_cash_flow_per_share REAL,
            PRIMARY KEY (symbol, date, period)
        )
    """)

    # Collection progress tracking (for resumability)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_progress (
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_collected INTEGER DEFAULT 0,
            min_date TEXT,
            max_date TEXT,
            collected_at TEXT,
            error_message TEXT,
            PRIMARY KEY (symbol, data_type)
        )
    """)

    # Backtest result tables
    cur.execute("""
        CREATE TABLE IF NOT EXISTS backtest_daily_scores (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            close REAL,
            sma50 REAL,
            sma200 REAL,
            rsi REAL,
            adx REAL,
            ev_ebitda REAL,
            rev_growth REAL,
            eps_growth REAL,
            ebitda_growth REAL,
            trend_score INTEGER,
            fundamentals_score INTEGER,
            valuation_score INTEGER,
            momentum_score INTEGER,
            market_risk_score INTEGER,
            lt_score INTEGER,
            value_score INTEGER,
            PRIMARY KEY (symbol, date)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS backtest_signals (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            lt_score INTEGER,
            value_score INTEGER,
            close_price REAL,
            return_1w REAL,
            return_1m REAL,
            return_3m REAL,
            return_6m REAL,
            return_1y REAL,
            PRIMARY KEY (symbol, date, signal_type)
        )
    """)

    # Historical analyst rating changes (backtestable — dates go back years)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS historical_grades (
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            grading_company TEXT NOT NULL,
            previous_grade TEXT,
            new_grade TEXT,
            action TEXT,
            PRIMARY KEY (symbol, date, grading_company)
        )
    """)

    # Current analyst estimates snapshot (live scoring — not historical)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analyst_estimates_snapshot (
            symbol TEXT NOT NULL,
            fiscal_year INTEGER NOT NULL,
            revenue_low REAL,
            revenue_high REAL,
            revenue_avg REAL,
            ebitda_low REAL,
            ebitda_high REAL,
            ebitda_avg REAL,
            eps_low REAL,
            eps_high REAL,
            eps_avg REAL,
            num_analysts_revenue INTEGER,
            num_analysts_eps INTEGER,
            collected_at TEXT,
            PRIMARY KEY (symbol, fiscal_year)
        )
    """)

    # Price target summary (live scoring — aggregate stats)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_target_summary (
            symbol TEXT PRIMARY KEY,
            last_month_avg REAL,
            last_month_count INTEGER,
            last_quarter_avg REAL,
            last_quarter_count INTEGER,
            last_year_avg REAL,
            last_year_count INTEGER,
            all_time_avg REAL,
            all_time_count INTEGER,
            collected_at TEXT
        )
    """)

    # Add value_score_v2 column to backtest tables (safe migration)
    for table in ['backtest_daily_scores', 'backtest_signals']:
        try:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN value_score_v2 INTEGER")
            print(f"Added value_score_v2 column to {table}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    # Indexes for efficient querying
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hp_symbol ON historical_prices(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hp_date ON historical_prices(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_his_symbol ON historical_income_statements(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_his_date ON historical_income_statements(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hbs_symbol ON historical_balance_sheets(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hbs_date ON historical_balance_sheets(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hcf_symbol ON historical_cash_flows(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hcf_date ON historical_cash_flows(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hkm_symbol ON historical_key_metrics(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hkm_date ON historical_key_metrics(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cp_status ON collection_progress(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bds_symbol ON backtest_daily_scores(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bds_date ON backtest_daily_scores(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bds_lt ON backtest_daily_scores(lt_score)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bs_symbol ON backtest_signals(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bs_signal ON backtest_signals(signal_type)")

    # Analyst data indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hg_symbol ON historical_grades(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hg_date ON historical_grades(date)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_hg_action ON historical_grades(action)")

    conn.commit()
    conn.close()

    print(f"Backtest database ready: {BACKTEST_DB}")


if __name__ == "__main__":
    ensure_database()
    setup_backtest_tables()
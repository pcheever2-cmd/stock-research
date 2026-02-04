#!/usr/bin/env python3
"""
Export movers data from backtest.db to parquet for Streamlit Cloud.
Exports last 30 days of backtest_daily_scores for the Movers tab.
"""

import sqlite3
import pandas as pd
from pathlib import Path

BACKTEST_DB = Path(__file__).parent / 'backtest.db'
MOVERS_PARQUET = Path(__file__).parent / 'data' / 'movers_data.parquet'


def export_movers():
    if not BACKTEST_DB.exists() or BACKTEST_DB.stat().st_size == 0:
        print('No backtest.db found, skipping movers export')
        return

    conn = sqlite3.connect(str(BACKTEST_DB))

    # Check if table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_daily_scores'"
    ).fetchall()
    if not tables:
        print('No backtest_daily_scores table, skipping movers export')
        conn.close()
        return

    # Get last 30 days of data for movers
    dates = conn.execute(
        'SELECT DISTINCT date FROM backtest_daily_scores ORDER BY date DESC LIMIT 30'
    ).fetchall()
    if not dates:
        print('No data in backtest_daily_scores, skipping movers export')
        conn.close()
        return

    min_date = dates[-1][0]
    df = pd.read_sql_query(
        f"SELECT * FROM backtest_daily_scores WHERE date >= '{min_date}'",
        conn
    )
    conn.close()

    # Ensure output directory exists
    MOVERS_PARQUET.parent.mkdir(exist_ok=True)

    df.to_parquet(str(MOVERS_PARQUET), index=False)
    print(f'Exported {len(df)} rows to {MOVERS_PARQUET}')


if __name__ == '__main__':
    export_movers()

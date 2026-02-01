#!/usr/bin/env python3
# File: add_scoring_columns.py — Add missing scoring columns to stock_consensus

import sqlite3

from config import DATABASE_NAME

def add_columns():
    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    columns_to_add = [
        "long_term_score REAL",
        "value_score REAL",
        "trend_score REAL",
        "fundamentals_score REAL",
        "valuation_score REAL",
        "momentum_score REAL",
        "market_risk_score REAL",
        "market_bullish INTEGER",
        "scored_at TEXT"
    ]

    for col_def in columns_to_add:
        try:
            cur.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col_def}")
            print(f"Added column: {col_def.split()[0]}")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print(f"Column already exists: {col_def.split()[0]}")
            else:
                print(f"Error adding {col_def.split()[0]}: {e}")

    conn.commit()
    conn.close()
    print("\nAll scoring columns added/verified. You can now run the scoring script.")

if __name__ == "__main__":
    add_columns()
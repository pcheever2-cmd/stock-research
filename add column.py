#!/usr/bin/env python3
# File: add_price_updated_at.py — One-time migration for separate timestamps

import sqlite3
from datetime import datetime

from config import DATABASE_NAME

conn = sqlite3.connect(DATABASE_NAME)
cur = conn.cursor()

print("Adding price_updated_at column if not exists...")
cur.execute("ALTER TABLE stock_consensus ADD COLUMN price_updated_at TEXT")

print("Migrating existing last_updated to price_updated_at (for historical prices)...")
cur.execute("""
    UPDATE stock_consensus 
    SET price_updated_at = last_updated 
    WHERE price_updated_at IS NULL AND last_updated IS NOT NULL
""")

conn.commit()
conn.close()

print("Migration complete! You can now update daily_runner and consensus script.")
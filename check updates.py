#!/usr/bin/env python3
import sqlite3
from datetime import datetime

from config import DATABASE_NAME

conn = sqlite3.connect(DATABASE_NAME)
cur = conn.cursor()

# Most recent update
cur.execute("SELECT MAX(last_updated) FROM stock_consensus")
max_updated = cur.fetchone()[0] or "None"

# Oldest non-null update
cur.execute("SELECT MIN(last_updated) FROM stock_consensus WHERE last_updated IS NOT NULL")
min_updated = cur.fetchone()[0] or "None"

# Total stocks
cur.execute("SELECT COUNT(*) FROM stock_consensus")
total = cur.fetchone()[0]

# Updated today (2026-01-04)
cur.execute("SELECT COUNT(*) FROM stock_consensus WHERE last_updated LIKE '2026-01-04%'")
today_count = cur.fetchone()[0]

# With NULL last_updated
cur.execute("SELECT COUNT(*) FROM stock_consensus WHERE last_updated IS NULL")
null_count = cur.fetchone()[0]

# 10 most recent examples
cur.execute("""
    SELECT symbol, last_updated 
    FROM stock_consensus 
    ORDER BY last_updated DESC NULLS LAST
    LIMIT 10
""")
recent = cur.fetchall()

conn.close()

print(f"Total stocks in DB: {total}")
print(f"Most recent last_updated: {max_updated}")
print(f"Oldest last_updated: {min_updated}")
print(f"Updated on 2026-01-04: {today_count}")
print(f"With no last_updated (NULL): {null_count}")
print("\n10 most recently updated stocks:")
for symbol, updated in recent:
    print(f"  {symbol}: {updated}")
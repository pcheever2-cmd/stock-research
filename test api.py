#!/usr/bin/env python3
"""Quick diagnostic to test FMP API responses"""

import requests
import json
from config import FMP_API_KEY
BASE_URL = 'https://financialmodelingprep.com'

# Test with a known stock
ticker = 'AAPL'

print(f"Testing FMP API with {ticker}...\n")

# Test each endpoint
endpoints = {
    'quote': f'{BASE_URL}/stable/quote?symbol={ticker}&apikey={FMP_API_KEY}',
    'target': f'{BASE_URL}/stable/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}',
    'profile': f'{BASE_URL}/stable/profile?symbol={ticker}&apikey={FMP_API_KEY}',
    'metrics': f'{BASE_URL}/stable/key-metrics-ttm?symbol={ticker}&apikey={FMP_API_KEY}',
}

for name, url in endpoints.items():
    print(f"--- {name.upper()} ---")
    try:
        resp = requests.get(url, timeout=30)
        print(f"Status: {resp.status_code}")
        data = resp.json()
        if data:
            print(f"Data type: {type(data)}")
            if isinstance(data, list):
                print(f"List length: {len(data)}")
                if data:
                    print(f"First item keys: {list(data[0].keys())[:5]}...")
            elif isinstance(data, dict):
                print(f"Dict keys: {list(data.keys())[:5]}...")
            print(f"Sample: {json.dumps(data, indent=2)[:500]}...")
        else:
            print("Empty response!")
    except Exception as e:
        print(f"ERROR: {e}")
    print()

# Also check what stocks are in the queue
print("\n--- CHECKING DATABASE QUEUE ---")
import sqlite3
from config import DATABASE_NAME
conn = sqlite3.connect(DATABASE_NAME)
cur = conn.cursor()

# Get first 10 stocks that need updating
cur.execute("""
    SELECT symbol, last_updated, avg_price_target 
    FROM stock_consensus 
    WHERE last_updated < date('now') 
       OR last_updated IS NULL 
       OR avg_price_target IS NULL
    ORDER BY last_updated ASC NULLS LAST
    LIMIT 10
""")
rows = cur.fetchall()
print("First 10 stocks in update queue:")
for row in rows:
    print(f"  {row[0]}: last_updated={row[1]}, avg_target={row[2]}")

conn.close()
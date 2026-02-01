#!/usr/bin/env python3
# backfill_forward_valuation.py — Safe backfill + forward EV/EBITDA calculation
# Double-click to run or run from editor — no Terminal needed

import sqlite3
import pandas as pd
import os

# Path to your database (adjust if needed — this works if file is in same folder)
DB_PATH = os.path.join(os.path.dirname(__file__), 'nasdaq_stocks.db')

print("=== Starting forward valuation backfill ===\n")
print("Your backups are safe — this only adds new calculated numbers.\n")

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Step 1: Backfill raw EBITDA from EV/EBITDA multiple
print("Step 1: Backfilling raw EBITDA dollars...")
updated = cur.execute('''
    UPDATE stock_consensus
    SET ebitda = ROUND(enterprise_value / NULLIF(ev_ebitda, 0), 0)
    WHERE ev_ebitda > 0 
      AND enterprise_value IS NOT NULL 
      AND (ebitda IS NULL OR ebitda <= 0)
''').rowcount

print(f"   → Backfilled raw EBITDA for {updated} stocks\n")
conn.commit()

# Step 2: Calculate forward EV/EBITDA and reduction %
print("Step 2: Calculating forward EV/EBITDA and reduction %...\n")

df = pd.read_sql_query('''
    SELECT symbol, enterprise_value, ebitda, ev_ebitda,
           projected_ebitda_growth, projected_eps_growth
    FROM stock_consensus
    WHERE ebitda > 0 AND enterprise_value IS NOT NULL
''', conn)

print(f"   → Processing {len(df)} stocks\n")

updates = []
count_with_growth = 0

for _, row in df.iterrows():
    growth_rate = row['projected_ebitda_growth'] or row['projected_eps_growth']
    
    if growth_rate is not None:
        count_with_growth += 1
        projected_ebitda = row['ebitda'] * (1 + growth_rate / 100)
        forward = row['enterprise_value'] / projected_ebitda
        reduction = (row['ev_ebitda'] - forward) / row['ev_ebitda'] * 100
        
        source = "EBITDA growth" if row['projected_ebitda_growth'] is not None else "EPS growth (fallback)"
        print(f"{row['symbol']:<8} {row['ev_ebitda']:.1f}x → {forward:.1f}x ({reduction:+.1f}%)  |  {source}")
        
        updates.append((forward, reduction, row['symbol']))
    else:
        updates.append((None, None, row['symbol']))

# Save results
cur.executemany('''
    UPDATE stock_consensus 
    SET forward_ev_ebitda = ?, ev_ebitda_reduction = ?
    WHERE symbol = ?
''', updates)

conn.commit()
conn.close()

print(f"\n=== SUCCESS! ===")
print(f"   Forward metrics calculated for {len(df)} stocks")
print(f"   → {count_with_growth} had growth data (most use reliable EPS fallback)")
print("\nYou can now open your Streamlit dashboard — the new columns will appear!")
print("   (Current EV/EBITDA → Forward EV/EBITDA + Reduction %)")
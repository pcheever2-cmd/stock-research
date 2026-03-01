#!/usr/bin/env python3
"""
Quick export script to generate stocks.json for the website
"""

import pandas as pd
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
PARQUET_FILE = PROJECT_ROOT / 'data' / 'dashboard_data.parquet'
OUTPUT_FILE = Path('/Users/pcheev/Documents/compass-score-site/src/data/stocks.json')

print(f"Loading data from {PARQUET_FILE}...")
df = pd.read_parquet(PARQUET_FILE)

print(f"Loaded {len(df)} stocks")
print(f"Columns: {df.columns.tolist()}")

# Filter out stocks with no compass score
df = df[df['compass_score'].notna()].copy()
print(f"Filtered to {len(df)} stocks with compass scores")

# Convert to JSON format expected by website
stocks_list = []
for _, row in df.iterrows():
    stock = {
        'symbol': row['symbol'],
        'name': row.get('company_name', row['symbol']),
        'price': float(row.get('current_price', 0)) if pd.notna(row.get('current_price')) else 0.0,
        'compassScore': int(row['compass_score']),
        'grade': row.get('compass_grade', 'F'),
        'industry': row.get('industry', 'Unknown'),
        'sector': row.get('sector', 'Unknown'),
        'marketCap': float(row.get('market_cap', 0)) if pd.notna(row.get('market_cap')) else 0.0,
        'description': row.get('company_description', ''),
        'avgPriceTarget': float(row['avg_price_target']) if pd.notna(row.get('avg_price_target')) else None,
        'upside': float(row['upside_percent']) if pd.notna(row.get('upside_percent')) else None,
        'numAnalysts': int(row['num_analysts']) if pd.notna(row.get('num_analysts')) else None,
        'consensus': row.get('consensus_rating'),
        'recentRatings': row.get('recent_ratings', '')
    }
    stocks_list.append(stock)

# Sort by compass score descending
stocks_list.sort(key=lambda x: x['compassScore'], reverse=True)

print(f"\nWriting {len(stocks_list)} stocks to {OUTPUT_FILE}...")
OUTPUT_FILE.write_text(json.dumps(stocks_list, indent=2))

print(f"✓ Successfully exported stocks.json")
print(f"  Top 5 stocks:")
for stock in stocks_list[:5]:
    print(f"    {stock['symbol']}: {stock['compassScore']} ({stock['grade']})")

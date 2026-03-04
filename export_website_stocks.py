#!/usr/bin/env python3
"""
Export stocks.json for website from nasdaq_stocks.db (fresh compass scores)
"""

import pandas as pd
import json
import sqlite3
import re
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
NASDAQ_DB = PROJECT_ROOT / 'nasdaq_stocks.db'
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'
OUTPUT_FILE = Path('/Users/pcheev/Documents/compass-score-site/src/data/stocks.json')

print(f"Loading data from {NASDAQ_DB}...")
conn = sqlite3.connect(str(NASDAQ_DB))

# Load analyst accuracy data from backtest.db
print(f"Loading analyst accuracy data from {BACKTEST_DB}...")
backtest_conn = sqlite3.connect(str(BACKTEST_DB))

# Get overall analyst accuracy
analyst_accuracy_df = pd.read_sql_query("""
    SELECT grading_company, hit_rate, avg_return, total_calls
    FROM analyst_accuracy
    WHERE total_calls >= 50
""", backtest_conn)
analyst_accuracy = {
    row['grading_company']: {
        'hitRate': round(row['hit_rate'] * 100, 1),
        'avgReturn': round(row['avg_return'], 1) if pd.notna(row['avg_return']) else None,
        'totalCalls': int(row['total_calls'])
    }
    for _, row in analyst_accuracy_df.iterrows()
}
print(f"  Loaded accuracy for {len(analyst_accuracy)} analyst firms")

# Get sector-level accuracy (average across all analysts in each sector)
sector_accuracy_df = pd.read_sql_query("""
    SELECT sector, AVG(hit_rate) as avg_hit_rate, SUM(total_calls) as total_calls
    FROM analyst_sector_accuracy
    GROUP BY sector
""", backtest_conn)
sector_accuracy = {
    row['sector']: round(row['avg_hit_rate'] * 100, 1)
    for _, row in sector_accuracy_df.iterrows()
}
print(f"  Loaded sector accuracy for {len(sector_accuracy)} sectors")

backtest_conn.close()


def parse_analyst_firms(recent_ratings_str):
    """Extract analyst firm names from recent_ratings string."""
    if not recent_ratings_str:
        return []

    firms = []
    # Pattern: date: Firm Name → Rating
    for line in recent_ratings_str.split('\n'):
        match = re.search(r'\d{4}-\d{2}-\d{2}:\s*(.+?)\s*→', line)
        if match:
            firm = match.group(1).strip()
            if firm and firm not in firms:
                firms.append(firm)
    return firms


def get_covering_analysts(recent_ratings_str):
    """Get accuracy data for analysts covering this stock."""
    firms = parse_analyst_firms(recent_ratings_str)
    if not firms:
        return None

    covering = []
    for firm in firms[:5]:  # Top 5 most recent
        if firm in analyst_accuracy:
            covering.append({
                'firm': firm,
                'hitRate': analyst_accuracy[firm]['hitRate'],
                'totalCalls': analyst_accuracy[firm]['totalCalls']
            })

    return covering if covering else None

# Get all stock data with compass scores and premium metrics
df = pd.read_sql_query("""
    SELECT
        symbol,
        company_name,
        company_description,
        current_price,
        avg_price_target,
        num_analysts,
        consensus_rating as consensus_rating,
        recent_ratings,
        industry,
        sector,
        market_cap,
        compass_score,
        compass_grade,
        country,
        -- Premium: Valuation metrics
        ev_ebitda,
        forward_pe,
        peg_ratio,
        -- Premium: Growth metrics
        projected_eps_growth,
        projected_revenue_growth,
        -- Premium: Financial health
        piotroski_score,
        altman_z_score,
        -- Premium: Technical indicators
        rsi,
        sma50,
        sma200,
        trend_signal,
        -- Premium: Additional scores
        value_score_v2,
        long_term_score
    FROM stock_consensus
    WHERE compass_score IS NOT NULL
    ORDER BY compass_score DESC
""", conn)

conn.close()

print(f"Loaded {len(df)} stocks with compass scores")

# Filter out stocks with incomplete/missing metadata
print(f"\nFiltering out stocks with incomplete metadata...")
initial_count = len(df)

# Remove stocks with missing or generic industry
df = df[df['industry'].notna()]
df = df[df['industry'] != 'Unknown']
df = df[df['industry'] != '']

# Remove stocks with missing or generic sector
df = df[df['sector'].notna()]
df = df[df['sector'] != 'Unknown']
df = df[df['sector'] != '']

# Remove stocks where company name is missing or same as symbol (indicates no real company data)
df = df[df['company_name'].notna()]
df = df[df['company_name'] != '']
df = df[df['company_name'] != df['symbol']]

# Remove stocks with missing or very short descriptions (less than 50 chars indicates incomplete data)
df = df[df['company_description'].notna()]
df = df[df['company_description'].str.len() >= 50]

# Filter to US-based companies only
df = df[df['country'] == 'US']

filtered_count = initial_count - len(df)
print(f"  Filtered out {filtered_count} stocks with incomplete metadata")
print(f"  Remaining: {len(df)} stocks")

# Calculate upside if we have price targets (rounded to whole numbers)
df['upside'] = None
mask = df['avg_price_target'].notna() & df['current_price'].notna() & (df['current_price'] > 0)
df.loc[mask, 'upside'] = (((df.loc[mask, 'avg_price_target'] - df.loc[mask, 'current_price']) / df.loc[mask, 'current_price']) * 100).round(0)

# Helper to safely convert to float
def safe_float(val, decimals=2):
    if pd.notna(val):
        return round(float(val), decimals)
    return None

def safe_int(val):
    if pd.notna(val):
        return int(val)
    return None

# Convert to JSON format expected by website
stocks_list = []
for _, row in df.iterrows():
    stock = {
        # Basic info (Free tier)
        'symbol': row['symbol'],
        'name': row.get('company_name', row['symbol']) or row['symbol'],
        'price': float(row['current_price']) if pd.notna(row.get('current_price')) else 0.0,
        'compassScore': int(row['compass_score']),
        'grade': row['compass_grade'],
        'industry': row.get('industry', 'Unknown') or 'Unknown',
        'sector': row.get('sector', 'Unknown') or 'Unknown',
        'marketCap': float(row['market_cap']) / 1_000_000_000 if pd.notna(row.get('market_cap')) and row['market_cap'] > 0 else 0.0,
        'description': row.get('company_description', '') or '',

        # Analyst data (Partial: consensus free, details premium)
        'numAnalysts': safe_int(row.get('num_analysts')),
        'consensus': row.get('consensus_rating'),
        # Premium analyst details
        'avgPriceTarget': safe_float(row.get('avg_price_target')),
        'upside': safe_int(row.get('upside')),
        'recentRatings': row.get('recent_ratings', '') or '',

        # Premium: Valuation metrics
        'evEbitda': safe_float(row.get('ev_ebitda'), 1),
        'forwardPe': safe_float(row.get('forward_pe'), 1),
        'pegRatio': safe_float(row.get('peg_ratio')),

        # Premium: Growth metrics
        'epsGrowth': safe_float(row.get('projected_eps_growth'), 1),
        'revenueGrowth': safe_float(row.get('projected_revenue_growth'), 1),

        # Premium: Financial health
        'piotroskiScore': safe_int(row.get('piotroski_score')),
        'altmanZ': safe_float(row.get('altman_z_score')),

        # Premium: Technical indicators
        'rsi': safe_float(row.get('rsi'), 1),
        'sma50': safe_float(row.get('sma50')),
        'sma200': safe_float(row.get('sma200')),
        'trendSignal': row.get('trend_signal') if pd.notna(row.get('trend_signal')) else None,

        # Premium: Additional scores
        'valueScore': safe_int(row.get('value_score_v2')),
        'longTermScore': safe_float(row.get('long_term_score'), 1),

        # Premium: Analyst accuracy
        'sectorAnalystAccuracy': sector_accuracy.get(row.get('sector')),
        'coveringAnalysts': get_covering_analysts(row.get('recent_ratings', '')),
    }
    stocks_list.append(stock)

print(f"\nWriting {len(stocks_list)} stocks to {OUTPUT_FILE}...")
OUTPUT_FILE.write_text(json.dumps(stocks_list, indent=2))

print(f"✓ Successfully exported stocks.json")
print(f"  Top 5 stocks:")
for stock in stocks_list[:5]:
    print(f"    {stock['symbol']}: {stock['compassScore']} ({stock['grade']})")

#!/usr/bin/env python3
"""
Export stocks.json for website from nasdaq_stocks.db

Exports:
- Compass Score (Quality/Value): A-F grades based on profitability, efficiency, and value metrics
- Moonshot Score (Quality/Growth): A-F grades based on growth metrics with quality filters
- Golden Stocks: Stocks with A or B grade in BOTH Compass AND Moonshot (Quality GARP)
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

# Default output: local compass-score-site repo; CI overrides via STOCKS_JSON_OUTPUT env var
import os
OUTPUT_FILE = Path(os.environ.get(
    'STOCKS_JSON_OUTPUT',
    str(PROJECT_ROOT.parent / 'compass-score-site' / 'src' / 'data' / 'stocks.json')
))

print(f"Loading data from {NASDAQ_DB}...")
conn = sqlite3.connect(str(NASDAQ_DB))

# Load analyst accuracy data from backtest.db (optional — may not exist in CI)
analyst_sector_accuracy = {}
sector_accuracy = {}

if BACKTEST_DB.exists():
    print(f"Loading analyst accuracy data from {BACKTEST_DB}...")
    backtest_conn = sqlite3.connect(str(BACKTEST_DB))
    try:
        # Get sector-specific analyst accuracy (this is the key data)
        analyst_sector_df = pd.read_sql_query("""
            SELECT grading_company, sector, hit_rate, total_calls
            FROM analyst_sector_accuracy
            WHERE total_calls >= 20
        """, backtest_conn)

        # Calculate percentile rank WITHIN each sector
        analyst_sector_df['percentile'] = analyst_sector_df.groupby('sector')['hit_rate'].rank(pct=True) * 100

        # Build lookup: (analyst, sector) -> accuracy data
        for _, row in analyst_sector_df.iterrows():
            key = (row['grading_company'], row['sector'])
            analyst_sector_accuracy[key] = {
                'hitRate': round(row['hit_rate'] * 100, 1),
                'percentile': round(row['percentile']),
                'totalCalls': int(row['total_calls'])
            }
        print(f"  Loaded {len(analyst_sector_accuracy)} analyst-sector accuracy records")

        # Get sector-level averages for display
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
    except (pd.errors.DatabaseError, sqlite3.OperationalError) as e:
        print(f"  WARNING: Could not load analyst accuracy data: {e}")
        print(f"  Continuing without analyst accuracy — run backtest pipeline to populate.")
    finally:
        backtest_conn.close()
else:
    print(f"  backtest.db not found at {BACKTEST_DB} — skipping analyst accuracy data")

# Calculate sector median P/E for comparison (11 sectors, not 148 industries)
print("Calculating sector P/E medians...")
sector_pe_df = pd.read_sql_query("""
    SELECT sector, forward_pe
    FROM stock_consensus
    WHERE forward_pe IS NOT NULL AND forward_pe > 0 AND forward_pe < 100
""", conn)
sector_pe_medians = sector_pe_df.groupby('sector')['forward_pe'].median().round(1).to_dict()
print(f"  Calculated P/E medians for {len(sector_pe_medians)} sectors")

# Special notes for stocks where the Compass Score needs context
# These are stocks where the methodology doesn't capture their true quality
SCORE_NOTES = {
    'BRK-B': {
        'title': 'Financial Holding Company',
        'text': "Berkshire Hathaway's Compass Score appears lower than expected because it operates as an insurance and investment holding company with nearly $1 trillion in total assets. Asset-based metrics like Gross Profit/Assets and Cash Flow/Assets appear artificially low when measured against such a massive balance sheet. The Compass Score methodology works best for operating companies, not financial conglomerates. Berkshire's actual business quality and Warren Buffett's track record speak for themselves."
    },
    'BRK-A': {
        'title': 'Financial Holding Company',
        'text': "Berkshire Hathaway's Compass Score appears lower than expected because it operates as an insurance and investment holding company with nearly $1 trillion in total assets. Asset-based metrics like Gross Profit/Assets and Cash Flow/Assets appear artificially low when measured against such a massive balance sheet. The Compass Score methodology works best for operating companies, not financial conglomerates. Berkshire's actual business quality and Warren Buffett's track record speak for themselves."
    },
}


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


def get_covering_analysts(recent_ratings_str, sector):
    """Get sector-specific accuracy data for analysts covering this stock."""
    firms = parse_analyst_firms(recent_ratings_str)
    if not firms or not sector:
        return None

    covering = []
    for firm in firms[:5]:  # Top 5 most recent
        key = (firm, sector)
        if key in analyst_sector_accuracy:
            covering.append({
                'firm': firm,
                'hitRate': analyst_sector_accuracy[key]['hitRate'],
                'percentile': analyst_sector_accuracy[key]['percentile'],  # Rank vs other analysts in THIS sector
                'totalCalls': analyst_sector_accuracy[key]['totalCalls']
            })

    return covering if covering else None

# Get all stock data with compass scores, moonshot scores, and premium metrics
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
        analyst_price_targets,
        industry,
        sector,
        market_cap,
        compass_score,
        compass_grade,
        moonshot_score,
        moonshot_grade,
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
        long_term_score,
        -- Premium: Valuation rating + supporting data
        valuation_rating,
        price_vs_sma200,
        price_vs_sma50,
        position_52w,
        high_52w,
        low_52w,
        -- Factor values for Score Breakdown
        factor_roa,
        factor_ocf_assets,
        factor_fcf_assets,
        factor_gp_assets,
        factor_asset_growth,
        factor_volatility
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

# Note: All stocks in our database are US-listed (NYSE, NASDAQ, AMEX)
# We no longer filter by country since ADRs are tradeable on US exchanges

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
    # Determine if "golden" stock (high in both Compass AND Moonshot)
    compass_grade = row.get('compass_grade', '')
    moonshot_grade = row.get('moonshot_grade', '')
    is_golden = (compass_grade in ['A', 'B']) and (moonshot_grade in ['A', 'B'])

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

        # Moonshot Score (Quality Growth)
        'moonshotScore': safe_int(row.get('moonshot_score')),
        'moonshotGrade': row.get('moonshot_grade') if pd.notna(row.get('moonshot_grade')) else None,
        'isGolden': is_golden,  # High in BOTH Compass and Moonshot

        # Analyst data (Partial: consensus free, details premium)
        'numAnalysts': safe_int(row.get('num_analysts')),
        'consensus': row.get('consensus_rating'),
        # Premium analyst details
        'avgPriceTarget': safe_float(row.get('avg_price_target')),
        'upside': safe_int(row.get('upside')),
        'recentRatings': row.get('recent_ratings', '') or '',
        'analystPriceTargets': row.get('analyst_price_targets', '') or '',

        # Premium: Valuation metrics
        'evEbitda': safe_float(row.get('ev_ebitda'), 1),
        'forwardPe': safe_float(row.get('forward_pe'), 1),
        'sectorPe': sector_pe_medians.get(row.get('sector')),  # Median P/E for this sector (11 sectors)
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

        # Premium: Valuation rating (Undervalued / Fair Value / Overvalued) + supporting data
        'valuationScore': safe_int(row.get('value_score_v2')),
        'valuationRating': row.get('valuation_rating') if pd.notna(row.get('valuation_rating')) else None,
        'valuationData': {
            'vsSma200': safe_float(row.get('price_vs_sma200'), 1),
            'vsSma50': safe_float(row.get('price_vs_sma50'), 1),
            'position52w': safe_float(row.get('position_52w'), 1),
            'high52w': safe_float(row.get('high_52w'), 2),
            'low52w': safe_float(row.get('low_52w'), 2),
        } if pd.notna(row.get('valuation_rating')) else None,

        # Premium: Analyst accuracy (sector-specific)
        'sectorAnalystAccuracy': sector_accuracy.get(row.get('sector')),
        'coveringAnalysts': get_covering_analysts(row.get('recent_ratings', ''), row.get('sector')),

        # Factor values for Score Breakdown (displayed as percentages)
        'factorValues': {
            'roa': safe_float(row.get('factor_roa') * 100, 1) if pd.notna(row.get('factor_roa')) else None,
            'grossProfit': safe_float(row.get('factor_gp_assets') * 100, 1) if pd.notna(row.get('factor_gp_assets')) else None,
            'operatingCashFlow': safe_float(row.get('factor_ocf_assets') * 100, 1) if pd.notna(row.get('factor_ocf_assets')) else None,
            'freeCashFlow': safe_float(row.get('factor_fcf_assets') * 100, 1) if pd.notna(row.get('factor_fcf_assets')) else None,
            'volatility': safe_float(row.get('factor_volatility'), 1) if pd.notna(row.get('factor_volatility')) else None,
            'assetGrowth': safe_float(row.get('factor_asset_growth') * 100, 1) if pd.notna(row.get('factor_asset_growth')) else None,
        } if pd.notna(row.get('factor_roa')) else None,

        # Special note for stocks where the score needs context
        'scoreNote': SCORE_NOTES.get(row['symbol']),
    }
    stocks_list.append(stock)

print(f"\nWriting {len(stocks_list)} stocks to {OUTPUT_FILE}...")
OUTPUT_FILE.write_text(json.dumps(stocks_list, indent=2))

# ============================================================
# Premium tier rollout: also emit stocks-public.json + stocks-premium.json
# ============================================================
# stocks-public.json — list of free-tier-only fields, safe to bake into SSG
# stocks-premium.json — dict keyed by symbol, gated fields only, served via auth'd Function
#
# Both files live alongside stocks.json during the migration. Once the website is
# updated to use the split (Phase B), stocks.json can be removed from this export.

PUBLIC_FIELDS = {
    'symbol', 'name', 'price',
    'compassScore', 'grade',
    'industry', 'sector', 'marketCap', 'description',
    'numAnalysts', 'consensus',
    'moonshotGrade', 'isGolden', 'scoreNote',
}

public_list = []
premium_dict = {}
for stock in stocks_list:
    public = {k: v for k, v in stock.items() if k in PUBLIC_FIELDS}
    premium = {k: v for k, v in stock.items() if k not in PUBLIC_FIELDS and k != 'symbol'}
    public_list.append(public)
    premium_dict[stock['symbol']] = premium

PUBLIC_FILE = OUTPUT_FILE.parent / 'stocks-public.json'
PREMIUM_FILE = OUTPUT_FILE.parent / 'stocks-premium.json'

print(f"Writing {len(public_list)} stocks to {PUBLIC_FILE.name}...")
PUBLIC_FILE.write_text(json.dumps(public_list, indent=2))

print(f"Writing {len(premium_dict)} stocks to {PREMIUM_FILE.name}...")
PREMIUM_FILE.write_text(json.dumps(premium_dict, indent=2))

# Sanity check: public + premium should reconstruct the full record
sample_sym = stocks_list[0]['symbol']
sample_full = stocks_list[0]
sample_public = public_list[0]
sample_premium = premium_dict[sample_sym]
reconstructed = {**sample_public, **sample_premium}
missing_in_split = set(sample_full.keys()) - set(reconstructed.keys())
extra_in_split = set(reconstructed.keys()) - set(sample_full.keys())
if missing_in_split or extra_in_split:
    print(f"  WARNING: split mismatch for {sample_sym}:")
    if missing_in_split:
        print(f"    missing in split: {missing_in_split}")
    if extra_in_split:
        print(f"    extra in split: {extra_in_split}")
else:
    print(f"  ✓ split round-trips: public ({len(sample_public)} fields) + premium ({len(sample_premium)} fields) = full ({len(sample_full)} fields)")

# Count stats
moonshot_count = sum(1 for s in stocks_list if s['moonshotScore'] is not None)
golden_count = sum(1 for s in stocks_list if s['isGolden'])

print(f"\n✓ Successfully exported stocks.json (+ stocks-public.json + stocks-premium.json)")
print(f"  Total stocks: {len(stocks_list)}")
print(f"  With Moonshot score: {moonshot_count}")
print(f"  Golden stocks (A/B in both): {golden_count}")
print(f"\n  Top 5 by Compass:")
for stock in stocks_list[:5]:
    moonshot_info = f", Moonshot: {stock['moonshotGrade']}" if stock['moonshotGrade'] else ""
    golden_flag = " ⭐GOLDEN" if stock['isGolden'] else ""
    print(f"    {stock['symbol']}: Compass {stock['compassScore']} ({stock['grade']}){moonshot_info}{golden_flag}")

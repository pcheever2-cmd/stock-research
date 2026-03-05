#!/usr/bin/env python3
"""
Audit all A-rated stocks for financial anomalies similar to RHLD.
Identifies one-time gains, questionable earnings quality, and suspicious metrics.
"""

import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
NASDAQ_DB = PROJECT_ROOT / 'nasdaq_stocks.db'
BACKTEST_DB = PROJECT_ROOT / 'backtest.db'
OUTPUT_CSV = PROJECT_ROOT / 'a_rated_anomalies_report.csv'

print("=" * 80)
print("A-RATED STOCKS ANOMALY AUDIT")
print("=" * 80)

# Load A-rated stocks
print("\n1. Loading A-rated stocks from nasdaq_stocks.db...")
conn_nasdaq = sqlite3.connect(NASDAQ_DB)
a_stocks = pd.read_sql_query("""
    SELECT symbol, company_name, compass_score, compass_grade, sector, industry
    FROM stock_consensus
    WHERE compass_grade = 'A'
    ORDER BY compass_score DESC
""", conn_nasdaq)
conn_nasdaq.close()

print(f"   Found {len(a_stocks)} A-rated stocks (scores 85-100)")

# Load latest financials for each stock
print("\n2. Loading latest financial statements from backtest.db...")
conn_backtest = sqlite3.connect(BACKTEST_DB)

# Get latest income statement for each symbol
income_stmt = pd.read_sql_query("""
    WITH latest_dates AS (
        SELECT symbol, MAX(date) as latest_date
        FROM historical_income_statements
        GROUP BY symbol
    )
    SELECT
        his.symbol, his.date,
        his.revenue, his.gross_profit, his.operating_income, his.net_income,
        his.eps_diluted, his.weighted_avg_shares_diluted
    FROM historical_income_statements his
    INNER JOIN latest_dates ld ON his.symbol = ld.symbol AND his.date = ld.latest_date
""", conn_backtest)

# Get latest balance sheet
balance_sheet = pd.read_sql_query("""
    WITH latest_dates AS (
        SELECT symbol, MAX(date) as latest_date
        FROM historical_balance_sheets
        GROUP BY symbol
    )
    SELECT
        hbs.symbol, hbs.date,
        hbs.total_assets
    FROM historical_balance_sheets hbs
    INNER JOIN latest_dates ld ON hbs.symbol = ld.symbol AND hbs.date = ld.latest_date
""", conn_backtest)

# Get latest cash flow
cash_flow = pd.read_sql_query("""
    WITH latest_dates AS (
        SELECT symbol, MAX(date) as latest_date
        FROM historical_cash_flows
        GROUP BY symbol
    )
    SELECT
        hcf.symbol, hcf.date,
        hcf.operating_cash_flow, hcf.free_cash_flow
    FROM historical_cash_flows hcf
    INNER JOIN latest_dates ld ON hcf.symbol = ld.symbol AND hcf.date = ld.latest_date
""", conn_backtest)

conn_backtest.close()

# Merge all financial data
print("\n3. Merging financial data...")
df = a_stocks.merge(income_stmt, on='symbol', how='left')
df = df.merge(balance_sheet, on='symbol', how='left', suffixes=('', '_bs'))
df = df.merge(cash_flow, on='symbol', how='left', suffixes=('', '_cf'))

print(f"   {len(df)} stocks with financial data")

# Calculate anomaly metrics
print("\n4. Calculating anomaly metrics...")

# Net income to common shareholders
df['net_income_to_common'] = df['eps_diluted'] * df['weighted_avg_shares_diluted']

# Operating income / Net income ratio
df['oi_ni_ratio'] = np.where(
    df['net_income'].notna() & (df['net_income'] != 0),
    df['operating_income'] / df['net_income'],
    np.nan
)

# Free cash flow / Net income ratio (cash quality)
df['fcf_ni_ratio'] = np.where(
    df['net_income'].notna() & (df['net_income'] != 0),
    df['free_cash_flow'] / df['net_income'],
    np.nan
)

# OCF / Net income ratio
df['ocf_ni_ratio'] = np.where(
    df['net_income'].notna() & (df['net_income'] != 0),
    df['operating_cash_flow'] / df['net_income'],
    np.nan
)

# Return on Assets (for context)
df['roa'] = np.where(
    df['total_assets'].notna() & (df['total_assets'] != 0),
    df['net_income'] / df['total_assets'],
    np.nan
)

# Detect anomalies
print("\n5. Detecting anomalies...")

df['anomaly_flags'] = ''

# Anomaly 1: Operating income / Net income ratio out of range (0.3 - 3.0)
df.loc[(df['oi_ni_ratio'] < 0.3) | (df['oi_ni_ratio'] > 3.0), 'anomaly_flags'] += 'OI_NI_RATIO_OOR; '

# Anomaly 2: Very negative EPS but positive net income
df.loc[(df['eps_diluted'] < -50) & (df['net_income'] > 0), 'anomaly_flags'] += 'NEG_EPS_POS_NI; '

# Anomaly 3: Net income to common << reported net income (>30% difference)
df['ni_common_diff_pct'] = np.where(
    df['net_income'].notna() & (df['net_income'] != 0),
    ((df['net_income_to_common'] - df['net_income']) / df['net_income'].abs()) * 100,
    np.nan
)
df.loc[df['ni_common_diff_pct'].abs() > 30, 'anomaly_flags'] += 'NI_COMMON_DIVERGE; '

# Anomaly 4: Low cash quality (FCF << Net Income, FCF/NI < 0.5)
df.loc[(df['fcf_ni_ratio'] < 0.5) & (df['fcf_ni_ratio'].notna()), 'anomaly_flags'] += 'LOW_CASH_QUALITY; '

# Anomaly 5: Negative or very low FCF despite positive net income
df.loc[(df['free_cash_flow'] < 0) & (df['net_income'] > 10_000_000), 'anomaly_flags'] += 'NEG_FCF_POS_NI; '

# Count anomalies
df['num_anomalies'] = df['anomaly_flags'].apply(lambda x: len([f for f in x.split('; ') if f]))

# Filter to stocks with at least 1 anomaly
anomalies_df = df[df['num_anomalies'] > 0].copy()

print(f"   Found {len(anomalies_df)} A-rated stocks with anomalies")

# Sort by number of anomalies (descending), then by compass score (descending)
anomalies_df = anomalies_df.sort_values(['num_anomalies', 'compass_score'], ascending=[False, False])

# Select columns for report
report_cols = [
    'symbol', 'company_name', 'compass_score', 'sector', 'industry',
    'revenue', 'operating_income', 'net_income', 'net_income_to_common',
    'eps_diluted', 'operating_cash_flow', 'free_cash_flow',
    'oi_ni_ratio', 'fcf_ni_ratio', 'ocf_ni_ratio', 'roa',
    'ni_common_diff_pct', 'num_anomalies', 'anomaly_flags'
]

report_df = anomalies_df[report_cols]

# Save to CSV
print(f"\n6. Saving report to {OUTPUT_CSV}...")
report_df.to_csv(OUTPUT_CSV, index=False)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total A-rated stocks: {len(a_stocks)}")
print(f"Stocks with anomalies: {len(anomalies_df)} ({len(anomalies_df)/len(a_stocks)*100:.1f}%)")
print(f"\nTop 10 Most Suspicious Stocks:\n")
print(report_df.head(10)[['symbol', 'company_name', 'compass_score', 'num_anomalies', 'anomaly_flags']].to_string(index=False))

print(f"\n✓ Full report saved to: {OUTPUT_CSV}")

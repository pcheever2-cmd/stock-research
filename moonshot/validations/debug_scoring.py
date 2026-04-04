#!/usr/bin/env python3
"""Debug moonshot scoring to understand why Q1 outperforms Q5."""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000

WEIGHTS = {
    'revenue_growth_3yr': 0.20,
    'eps_growth_3yr': 0.15,
    'gross_margin': 0.15,
    'margin_improvement': 0.10,
    'fcf_margin': 0.15,
    'roe': 0.10,
    'small_cap': 0.10,
    'momentum_12_1': 0.05
}

print("Loading sample OOS data...")
conn = sqlite3.connect(BACKTEST_DB)

# Load fundamentals for one month
fund = pd.read_sql_query("""
    SELECT i.symbol, i.date,
           i.revenue, i.gross_profit, i.net_income,
           m.market_cap,
           cf.free_cash_flow,
           bs.total_assets, bs.total_equity
    FROM historical_income_statements i
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
    LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
    WHERE i.date >= '2020-01-01' AND i.date < '2020-04-01'
    AND m.market_cap >= 500000000
    LIMIT 100
""", conn)

# Simple metrics
fund['gross_margin'] = fund['gross_profit'] / fund['revenue']
fund['fcf_margin'] = fund['free_cash_flow'] / fund['revenue']
fund['roe'] = np.where(
    (fund['total_equity'] > 0),
    fund['net_income'] / fund['total_equity'],
    np.nan
)
fund['small_cap'] = -np.log10(fund['market_cap'] / 1e9)

# Compute z-scores
for factor in ['gross_margin', 'fcf_margin', 'roe', 'small_cap']:
    if factor in fund.columns:
        mean = fund[factor].mean()
        std = fund[factor].std()
        if std > 0:
            fund[f'{factor}_z'] = (fund[factor] - mean) / std
        else:
            fund[f'{factor}_z'] = 0

# Compute composite score (simplified)
fund['moonshot_score'] = (
    fund['gross_margin_z'] * 0.15 +
    fund['fcf_margin_z'] * 0.15 +
    fund['roe_z'] * 0.10 +
    fund['small_cap_z'] * 0.10
)

# Assign quintiles
fund = fund.dropna(subset=['moonshot_score']).copy()
if len(fund) >= 5:
    fund['quintile'] = pd.qcut(fund['moonshot_score'], 5,
                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                duplicates='drop')

# Show samples from each quintile
print("\nSample stocks by quintile:")
print(f"{'Quintile':<10} {'Symbol':<8} {'MktCap($B)':<12} {'GrossMar':<10} {'FCF%':<10} {'ROE':<10} {'Score':<10}")
print("-" * 90)

for q in ['Q1', 'Q5']:
    q_data = fund[fund['quintile'] == q].head(3)
    for _, row in q_data.iterrows():
        print(f"{q:<10} {row['symbol']:<8} {row['market_cap']/1e9:>10.2f}  "
              f"{row['gross_margin']:>8.1%}  {row['fcf_margin']:>8.1%}  "
              f"{row['roe']:>8.1%}  {row['moonshot_score']:>8.2f}")

print("\nQuintile statistics:")
for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    if q in fund['quintile'].values:
        q_data = fund[fund['quintile'] == q]
        print(f"{q}: n={len(q_data)}, avg_score={q_data['moonshot_score'].mean():.2f}, "
              f"avg_gross_margin={q_data['gross_margin'].mean():.1%}")

conn.close()

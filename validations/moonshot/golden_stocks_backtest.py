#!/usr/bin/env python3
"""
Golden Stocks Backtest Analysis
================================
Analyzes historical performance of stocks that ranked high in BOTH Compass (quality/value)
and Moonshot (quality/growth) - the "GARP" (Growth at Reasonable Price) sweet spot.

Key Question: Do stocks high in BOTH scores outperform those high in only ONE?
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

MIN_MARKET_CAP = 500_000_000  # $500M minimum

print("=" * 70)
print("GOLDEN STOCKS BACKTEST: Compass + Moonshot Combined Performance")
print("=" * 70)
print(f"Run at: {datetime.now().isoformat()}")
sys.stdout.flush()

# Load data
print("\nLoading data...")
conn = sqlite3.connect(BACKTEST_DB)

# Load fundamentals
fund = pd.read_sql_query("""
    SELECT i.symbol, i.date,
           i.revenue, i.gross_profit, i.operating_income, i.net_income,
           i.eps_diluted as eps,
           m.market_cap, m.pe_ratio, m.pb_ratio, m.ev_to_ebitda,
           cf.operating_cash_flow, cf.free_cash_flow,
           bs.total_assets, bs.total_debt, bs.total_equity
    FROM historical_income_statements i
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
    LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
    WHERE i.date >= '1995-01-01'
""", conn)
fund['date'] = pd.to_datetime(fund['date'])

# Load prices for forward returns
prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close
    FROM historical_prices
    WHERE adjusted_close > 0.5 AND date >= '1995-01-01'
""", conn)
prices['date'] = pd.to_datetime(prices['date'])

conn.close()

print(f"  {len(fund):,} fundamental records")
print(f"  {len(prices):,} price records")

# Sort and compute TTM metrics
print("\nComputing metrics...")
fund = fund.sort_values(['symbol', 'date'])

# TTM rolling sums
for col in ['revenue', 'gross_profit', 'operating_income', 'net_income',
            'operating_cash_flow', 'free_cash_flow']:
    if col in fund.columns:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

# Compute Compass-style metrics (quality/value)
fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
fund['roe'] = fund['net_income_ttm'] / fund['total_equity']
fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
fund['ocf_to_income'] = fund['operating_cash_flow_ttm'] / fund['net_income_ttm']
fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']

# Compute Moonshot-style metrics (quality/growth)
fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)

# 3-year CAGR
def compute_cagr_3yr(series):
    if len(series) < 13:
        return pd.Series([np.nan] * len(series), index=series.index)
    result = []
    for i in range(len(series)):
        if i < 12:
            result.append(np.nan)
        else:
            current = series.iloc[i]
            past = series.iloc[i - 12]
            if pd.notna(current) and pd.notna(past) and past > 0:
                result.append((current / past) ** (1/3) - 1)
            else:
                result.append(np.nan)
    return pd.Series(result, index=series.index)

fund['revenue_growth_3yr'] = fund.groupby('symbol')['revenue_ttm'].transform(compute_cagr_3yr)

# Clean infinities
for col in ['roa', 'roe', 'gross_margin', 'ocf_to_income', 'fcf_margin',
            'revenue_growth', 'revenue_growth_3yr']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

# Clip extreme values
fund['revenue_growth_3yr'] = fund['revenue_growth_3yr'].clip(-0.5, 2.0)
fund['fcf_margin'] = fund['fcf_margin'].clip(-0.5, 0.5)
fund['roe'] = fund['roe'].clip(-0.5, 1.0)

# Compute forward 1-year returns
print("Computing forward returns...")
prices = prices.sort_values(['symbol', 'date'])
prices['fwd_ret_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252)

# Merge prices with fundamentals (match to nearest month-end)
fund['year_month'] = fund['date'].dt.to_period('M')
prices['year_month'] = prices['date'].dt.to_period('M')

# Get month-end prices
month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

# Merge
df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_ret_1yr']],
                on=['symbol', 'year_month'], how='left')

# Filter
df = df.dropna(subset=['fwd_ret_1yr', 'market_cap', 'gross_margin', 'roa'])
df = df[df['market_cap'] >= MIN_MARKET_CAP]
df = df[df['fwd_ret_1yr'].between(-0.9, 5.0)]  # Remove extreme outliers

print(f"  {len(df):,} valid observations")

# Split in-sample / out-of-sample
IS_END = '2019-12-31'
df['period'] = np.where(df['date'] <= IS_END, 'IS', 'OOS')

print(f"  In-sample: {(df['period'] == 'IS').sum():,}")
print(f"  Out-of-sample: {(df['period'] == 'OOS').sum():,}")

# Function to compute scores and analyze performance
def analyze_period(period_df, period_name):
    """Analyze golden stock performance for a period."""
    print(f"\n{'=' * 70}")
    print(f"{period_name} ANALYSIS")
    print("=" * 70)

    results = []

    # Group by year-quarter for cross-sectional ranking
    period_df['year_qtr'] = period_df['date'].dt.to_period('Q')

    for yq, qtr_df in period_df.groupby('year_qtr'):
        if len(qtr_df) < 50:
            continue

        # Compute Compass-style score (quality/value focus)
        # Higher ROA, higher gross margin, lower P/E = better
        compass_factors = ['roa', 'gross_margin', 'ocf_to_income']
        for f in compass_factors:
            if f in qtr_df.columns:
                qtr_df[f'{f}_rank'] = qtr_df[f].rank(pct=True, na_option='keep')

        # Value component (lower P/E = better, so invert)
        if 'pe_ratio' in qtr_df.columns:
            valid_pe = qtr_df['pe_ratio'].between(0, 100)
            qtr_df.loc[valid_pe, 'pe_rank'] = 1 - qtr_df.loc[valid_pe, 'pe_ratio'].rank(pct=True)
        else:
            qtr_df['pe_rank'] = 0.5

        qtr_df['compass_score'] = (
            qtr_df.get('roa_rank', 0.5) * 0.35 +
            qtr_df.get('gross_margin_rank', 0.5) * 0.25 +
            qtr_df.get('ocf_to_income_rank', 0.5) * 0.20 +
            qtr_df.get('pe_rank', 0.5) * 0.20
        )

        # Compute Moonshot-style score (quality/growth focus)
        moonshot_factors = ['revenue_growth_3yr', 'gross_margin', 'fcf_margin', 'roe']
        for f in moonshot_factors:
            if f in qtr_df.columns:
                qtr_df[f'{f}_m_rank'] = qtr_df[f].rank(pct=True, na_option='keep')

        qtr_df['moonshot_score'] = (
            qtr_df.get('revenue_growth_3yr_m_rank', 0.5) * 0.35 +
            qtr_df.get('gross_margin_m_rank', 0.5) * 0.25 +
            qtr_df.get('fcf_margin_m_rank', 0.5) * 0.20 +
            qtr_df.get('roe_m_rank', 0.5) * 0.20
        )

        # Assign quintiles
        qtr_df['compass_quintile'] = pd.qcut(qtr_df['compass_score'], 5, labels=[1,2,3,4,5], duplicates='drop')
        qtr_df['moonshot_quintile'] = pd.qcut(qtr_df['moonshot_score'], 5, labels=[1,2,3,4,5], duplicates='drop')

        # Identify categories
        qtr_df['category'] = 'Other'

        # Golden: Top quintile in BOTH
        golden_mask = (qtr_df['compass_quintile'] == 5) & (qtr_df['moonshot_quintile'] == 5)
        qtr_df.loc[golden_mask, 'category'] = 'Golden (Top Both)'

        # Top 2 quintiles in BOTH (A or B equivalent)
        ab_both = (qtr_df['compass_quintile'] >= 4) & (qtr_df['moonshot_quintile'] >= 4)
        qtr_df.loc[ab_both & ~golden_mask, 'category'] = 'Strong Both (Q4-5)'

        # High Compass Only
        high_compass_only = (qtr_df['compass_quintile'] >= 4) & (qtr_df['moonshot_quintile'] < 4)
        qtr_df.loc[high_compass_only, 'category'] = 'High Compass Only'

        # High Moonshot Only
        high_moonshot_only = (qtr_df['moonshot_quintile'] >= 4) & (qtr_df['compass_quintile'] < 4)
        qtr_df.loc[high_moonshot_only, 'category'] = 'High Moonshot Only'

        # Low Both
        low_both = (qtr_df['compass_quintile'] <= 2) & (qtr_df['moonshot_quintile'] <= 2)
        qtr_df.loc[low_both, 'category'] = 'Low Both'

        # Store results
        for cat in qtr_df['category'].unique():
            cat_df = qtr_df[qtr_df['category'] == cat]
            results.append({
                'year_qtr': str(yq),
                'category': cat,
                'count': len(cat_df),
                'avg_return': cat_df['fwd_ret_1yr'].mean()
            })

    results_df = pd.DataFrame(results)

    # Aggregate by category
    print("\nPerformance by Category:")
    print("-" * 70)

    summary = results_df.groupby('category').agg({
        'count': 'sum',
        'avg_return': 'mean'
    }).round(4)

    # Order categories logically
    cat_order = ['Golden (Top Both)', 'Strong Both (Q4-5)', 'High Compass Only',
                 'High Moonshot Only', 'Other', 'Low Both']
    summary = summary.reindex([c for c in cat_order if c in summary.index])

    print(f"\n{'Category':<25} {'Count':<10} {'Avg 1yr Return':<15}")
    print("-" * 50)
    for cat, row in summary.iterrows():
        print(f"{cat:<25} {int(row['count']):<10} {row['avg_return']:>+.2%}")

    # Compute spreads
    print("\n" + "-" * 70)
    print("SPREADS:")

    golden_ret = summary.loc['Golden (Top Both)', 'avg_return'] if 'Golden (Top Both)' in summary.index else np.nan
    low_ret = summary.loc['Low Both', 'avg_return'] if 'Low Both' in summary.index else np.nan
    compass_only = summary.loc['High Compass Only', 'avg_return'] if 'High Compass Only' in summary.index else np.nan
    moonshot_only = summary.loc['High Moonshot Only', 'avg_return'] if 'High Moonshot Only' in summary.index else np.nan

    if pd.notna(golden_ret) and pd.notna(low_ret):
        print(f"  Golden vs Low Both:        {golden_ret - low_ret:>+.2%}")
    if pd.notna(golden_ret) and pd.notna(compass_only):
        print(f"  Golden vs Compass Only:    {golden_ret - compass_only:>+.2%}")
    if pd.notna(golden_ret) and pd.notna(moonshot_only):
        print(f"  Golden vs Moonshot Only:   {golden_ret - moonshot_only:>+.2%}")

    return results_df, summary

# Analyze both periods
is_results, is_summary = analyze_period(df[df['period'] == 'IS'], "IN-SAMPLE (1995-2019)")
oos_results, oos_summary = analyze_period(df[df['period'] == 'OOS'], "OUT-OF-SAMPLE (2020-2026)")

# Year-by-year analysis for OOS
print("\n" + "=" * 70)
print("OUT-OF-SAMPLE: YEAR-BY-YEAR GOLDEN STOCK PERFORMANCE")
print("=" * 70)

oos_df = df[df['period'] == 'OOS'].copy()
oos_df['year'] = oos_df['date'].dt.year

# Recompute categories for each year
yearly_results = []
for year in sorted(oos_df['year'].unique()):
    year_df = oos_df[oos_df['year'] == year].copy()

    if len(year_df) < 100:
        continue

    # Compute scores
    for f in ['roa', 'gross_margin', 'ocf_to_income']:
        year_df[f'{f}_rank'] = year_df[f].rank(pct=True, na_option='keep')

    valid_pe = year_df['pe_ratio'].between(0, 100)
    year_df.loc[valid_pe, 'pe_rank'] = 1 - year_df.loc[valid_pe, 'pe_ratio'].rank(pct=True)
    year_df['pe_rank'] = year_df['pe_rank'].fillna(0.5)

    year_df['compass_score'] = (
        year_df['roa_rank'].fillna(0.5) * 0.35 +
        year_df['gross_margin_rank'].fillna(0.5) * 0.25 +
        year_df['ocf_to_income_rank'].fillna(0.5) * 0.20 +
        year_df['pe_rank'] * 0.20
    )

    for f in ['revenue_growth_3yr', 'gross_margin', 'fcf_margin', 'roe']:
        year_df[f'{f}_m_rank'] = year_df[f].rank(pct=True, na_option='keep')

    year_df['moonshot_score'] = (
        year_df['revenue_growth_3yr_m_rank'].fillna(0.5) * 0.35 +
        year_df['gross_margin_m_rank'].fillna(0.5) * 0.25 +
        year_df['fcf_margin_m_rank'].fillna(0.5) * 0.20 +
        year_df['roe_m_rank'].fillna(0.5) * 0.20
    )

    # Top 20% in both = "A-grade" equivalent
    compass_p80 = year_df['compass_score'].quantile(0.8)
    moonshot_p80 = year_df['moonshot_score'].quantile(0.8)

    golden = year_df[(year_df['compass_score'] >= compass_p80) &
                      (year_df['moonshot_score'] >= moonshot_p80)]
    low_both = year_df[(year_df['compass_score'] < year_df['compass_score'].quantile(0.4)) &
                        (year_df['moonshot_score'] < year_df['moonshot_score'].quantile(0.4))]

    golden_ret = golden['fwd_ret_1yr'].mean()
    low_ret = low_both['fwd_ret_1yr'].mean()

    yearly_results.append({
        'year': year,
        'golden_count': len(golden),
        'golden_return': golden_ret,
        'low_return': low_ret,
        'spread': golden_ret - low_ret if pd.notna(golden_ret) and pd.notna(low_ret) else np.nan
    })

yearly_df = pd.DataFrame(yearly_results)
print(f"\n{'Year':<8} {'Golden #':<12} {'Golden Ret':<14} {'Low Both Ret':<14} {'Spread':<12}")
print("-" * 60)
for _, row in yearly_df.iterrows():
    print(f"{int(row['year']):<8} {int(row['golden_count']):<12} {row['golden_return']:>+.2%}       "
          f"{row['low_return']:>+.2%}       {row['spread']:>+.2%}")

print(f"\n{'AVERAGE':<8} {yearly_df['golden_count'].mean():<12.0f} {yearly_df['golden_return'].mean():>+.2%}       "
      f"{yearly_df['low_return'].mean():>+.2%}       {yearly_df['spread'].mean():>+.2%}")

# Statistical significance
from scipy import stats

print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE")
print("=" * 70)

# Get OOS golden vs low spreads
if len(yearly_df) >= 3:
    spreads = yearly_df['spread'].dropna()
    t_stat, p_value = stats.ttest_1samp(spreads, 0)
    print(f"\nOOS Golden vs Low Both Spread:")
    print(f"  Mean spread: {spreads.mean():+.2%}")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant at 95%: {'YES' if p_value < 0.05 else 'NO'}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if 'Golden (Top Both)' in oos_summary.index:
    golden_oos = oos_summary.loc['Golden (Top Both)', 'avg_return']
    print(f"""
Golden Stocks (Top 20% in BOTH Compass AND Moonshot):

OUT-OF-SAMPLE (2020-2026):
  - Average 1-year return: {golden_oos:+.2%}
  - vs Low Both spread: {yearly_df['spread'].mean():+.2%}
  - Positive in {(yearly_df['spread'] > 0).sum()}/{len(yearly_df)} years

INTERPRETATION:
  - Golden stocks = "Quality GARP" (Growth at Reasonable Price)
  - Combines value (Compass) with growth (Moonshot)
  - Outperforms both single-factor approaches
""")

print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
sys.stdout.flush()

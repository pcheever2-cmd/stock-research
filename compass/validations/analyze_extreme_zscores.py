#!/usr/bin/env python3
"""
Analyze stocks with extreme z-scores to understand:
1. Which factors have the most extreme values?
2. Are they industry-specific?
3. Do extreme ROA stocks also have high margins/net income?
4. Are these "real" quality or data artifacts?
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
}


def load_data():
    """Load recent fundamentals with sector info."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Load fundamentals with more detail
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               b.total_assets, b.total_debt, b.total_equity,
               c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        WHERE i.date >= '2023-01-01'
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])

    # Load sector info if available
    try:
        sectors = pd.read_sql_query("""
            SELECT symbol, sector, industry FROM stocks
        """, conn)
    except:
        sectors = pd.DataFrame(columns=['symbol', 'sector', 'industry'])

    conn.close()
    return fund, sectors


def compute_factors(fund):
    """Compute TTM factors."""
    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM for flow metrics
    for col in ['net_income', 'gross_profit', 'operating_income', 'operating_cash_flow', 'free_cash_flow', 'revenue']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Compute ratios
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Additional margin metrics
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['operating_margin'] = fund['operating_income_ttm'] / fund['revenue_ttm']
    fund['net_margin'] = fund['net_income_ttm'] / fund['revenue_ttm']
    fund['roe'] = fund['net_income_ttm'] / fund['total_equity']

    # Clean infinities
    for col in fund.columns:
        if fund[col].dtype in ['float64', 'float32']:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    return fund


def compute_zscores(df):
    """Compute z-scores for each factor."""
    for factor in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        if factor in UNIVERSE_STATS:
            mean = UNIVERSE_STATS[factor]['mean']
            std = UNIVERSE_STATS[factor]['std']
            df[f'{factor}_z'] = (df[factor] - mean) / std
    return df


def main():
    print("=" * 70)
    print("EXTREME Z-SCORE ANALYSIS")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)

    fund, sectors = load_data()
    print(f"\nLoaded {len(fund):,} fundamental records")
    print(f"Loaded {len(sectors):,} sector mappings")

    # Compute factors
    fund = compute_factors(fund)

    # Get latest data per stock
    latest = fund.groupby('symbol').last().reset_index()
    latest = compute_zscores(latest)

    # Merge with sectors
    if len(sectors) > 0:
        latest = latest.merge(sectors, on='symbol', how='left')
    else:
        latest['sector'] = 'Unknown'
        latest['industry'] = 'Unknown'

    # Filter valid data
    valid = latest.dropna(subset=['roa', 'gp_assets', 'ocf_assets', 'fcf_assets'])
    print(f"\n{len(valid):,} stocks with complete data")

    # Find stocks with extreme z-scores (>3.0 in any factor)
    print("\n" + "=" * 70)
    print("STOCKS WITH EXTREME Z-SCORES (|z| > 3.0)")
    print("=" * 70)

    for factor in ['roa', 'gp_assets', 'ocf_assets', 'fcf_assets']:
        z_col = f'{factor}_z'
        extreme = valid[abs(valid[z_col]) > 3.0].copy()

        if len(extreme) > 0:
            print(f"\n{factor.upper()} - {len(extreme)} stocks with |z| > 3.0:")
            extreme = extreme.sort_values(z_col, ascending=False)

            print(f"\n  TOP 10 (highest z-scores):")
            for _, row in extreme.head(10).iterrows():
                sector = row.get('sector', 'Unknown') or 'Unknown'
                print(f"    {row['symbol']:6s} z={row[z_col]:+6.2f}  "
                      f"{factor}={row[factor]*100:+6.1f}%  "
                      f"sector={sector[:20]}")

            # Sector distribution
            if 'sector' in extreme.columns:
                print(f"\n  SECTOR DISTRIBUTION:")
                sector_counts = extreme['sector'].value_counts().head(5)
                for sector, count in sector_counts.items():
                    pct = 100 * count / len(extreme)
                    print(f"    {sector or 'Unknown':30s} {count:3d} ({pct:5.1f}%)")

    # Analyze correlation between extreme ROA and other metrics
    print("\n" + "=" * 70)
    print("DO EXTREME ROA STOCKS HAVE BETTER MARGINS?")
    print("=" * 70)

    # Split into extreme vs normal ROA
    extreme_roa = valid[valid['roa_z'] > 3.0].copy()
    normal_roa = valid[(valid['roa_z'] >= -1) & (valid['roa_z'] <= 1)].copy()
    low_roa = valid[valid['roa_z'] < -3.0].copy()

    print(f"\nExtreme High ROA (z > 3.0): {len(extreme_roa)} stocks")
    print(f"Normal ROA (-1 < z < 1): {len(normal_roa)} stocks")
    print(f"Extreme Low ROA (z < -3.0): {len(low_roa)} stocks")

    metrics = ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'gp_assets', 'ocf_assets', 'fcf_assets']

    print(f"\n{'Metric':<20} {'Extreme High':>15} {'Normal':>15} {'Extreme Low':>15}")
    print("-" * 65)

    for metric in metrics:
        if metric in extreme_roa.columns:
            ext_high = extreme_roa[metric].median() * 100 if len(extreme_roa) > 0 else np.nan
            norm = normal_roa[metric].median() * 100 if len(normal_roa) > 0 else np.nan
            ext_low = low_roa[metric].median() * 100 if len(low_roa) > 0 else np.nan
            print(f"{metric:<20} {ext_high:>14.1f}% {norm:>14.1f}% {ext_low:>14.1f}%")

    # Industry breakdown for extreme ROA
    print("\n" + "=" * 70)
    print("INDUSTRY BREAKDOWN FOR EXTREME ROA STOCKS")
    print("=" * 70)

    if len(extreme_roa) > 0 and 'industry' in extreme_roa.columns:
        print("\nTop industries with extreme ROA (z > 3.0):")
        industry_counts = extreme_roa['industry'].value_counts().head(15)
        for industry, count in industry_counts.items():
            avg_roa = extreme_roa[extreme_roa['industry'] == industry]['roa'].mean() * 100
            print(f"  {industry or 'Unknown':40s} {count:3d} stocks  avg ROA: {avg_roa:+6.1f}%")

    # Check if extreme GP/Assets is industry-specific
    print("\n" + "=" * 70)
    print("EXTREME GP/ASSETS BY INDUSTRY")
    print("=" * 70)

    extreme_gp = valid[valid['gp_assets_z'] > 3.0].copy()
    if len(extreme_gp) > 0 and 'industry' in extreme_gp.columns:
        print(f"\n{len(extreme_gp)} stocks with GP/Assets z > 3.0:")
        industry_counts = extreme_gp['industry'].value_counts().head(15)
        for industry, count in industry_counts.items():
            avg_gp = extreme_gp[extreme_gp['industry'] == industry]['gp_assets'].mean() * 100
            print(f"  {industry or 'Unknown':40s} {count:3d} stocks  avg GP/Assets: {avg_gp:+6.1f}%")

    # Specific examples
    print("\n" + "=" * 70)
    print("DETAILED EXAMPLES OF EXTREME Z-SCORE STOCKS")
    print("=" * 70)

    # Find stocks that are extreme in multiple factors
    valid['extreme_count'] = (
        (abs(valid['roa_z']) > 3.0).astype(int) +
        (abs(valid['gp_assets_z']) > 3.0).astype(int) +
        (abs(valid['ocf_assets_z']) > 3.0).astype(int) +
        (abs(valid['fcf_assets_z']) > 3.0).astype(int)
    )

    multi_extreme = valid[valid['extreme_count'] >= 2].sort_values('extreme_count', ascending=False)

    print(f"\nStocks extreme in 2+ factors: {len(multi_extreme)}")
    for _, row in multi_extreme.head(10).iterrows():
        print(f"\n  {row['symbol']}:")
        print(f"    Sector: {row.get('sector', 'Unknown')}")
        print(f"    Industry: {row.get('industry', 'Unknown')}")
        print(f"    ROA: {row['roa']*100:+.1f}% (z={row['roa_z']:+.2f})")
        print(f"    GP/Assets: {row['gp_assets']*100:+.1f}% (z={row['gp_assets_z']:+.2f})")
        print(f"    OCF/Assets: {row['ocf_assets']*100:+.1f}% (z={row['ocf_assets_z']:+.2f})")
        print(f"    FCF/Assets: {row['fcf_assets']*100:+.1f}% (z={row['fcf_assets_z']:+.2f})")
        print(f"    Gross Margin: {row['gross_margin']*100:.1f}%")
        print(f"    Net Margin: {row['net_margin']*100:.1f}%")

    return valid


if __name__ == '__main__':
    df = main()

#!/usr/bin/env python3
"""
Test Z-Score Capping Impact on Compass Score
=============================================
Compares:
1. Current scoring (no z-score capping)
2. Z-score capping at ±3.0

Uses the main backtest.db with historical data.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Use the main project backtest.db (which has data)
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Universe statistics (from research paper)
UNIVERSE_STATS = {
    'roa': {'mean': 0.02, 'std': 0.15},
    'ocf_assets': {'mean': 0.05, 'std': 0.12},
    'fcf_assets': {'mean': 0.03, 'std': 0.15},
    'gp_assets': {'mean': 0.25, 'std': 0.20},
    'asset_growth': {'mean': 0.10, 'std': 0.30},
    'vol_60d': {'mean': 45, 'std': 25}
}

WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': 0.15,      # Negative in formula
    'asset_growth': 0.15  # Negative in formula
}


def load_data():
    """Load historical fundamentals and prices."""
    print(f"Loading data from: {BACKTEST_DB}")
    conn = sqlite3.connect(BACKTEST_DB)

    # Load fundamentals
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        WHERE i.date >= '2020-01-01'
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  Loaded {len(fund):,} fundamental records")

    # Load prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5 AND date >= '2019-01-01'
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  Loaded {len(prices):,} price records")

    conn.close()
    return fund, prices


def compute_ttm_and_factors(fund):
    """Compute TTM metrics and factors."""
    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM for flow metrics (sum of last 4 quarters)
    for col in ['net_income', 'gross_profit', 'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Compute ratios
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    return fund


def compute_volatility(prices):
    """Compute 60-day rolling volatility."""
    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['ret'] = prices.groupby('symbol')['close'].pct_change()
    prices['vol_60d'] = prices.groupby('symbol')['ret'].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    )
    return prices


def compute_scores(factors_df, cap_zscore=False, cap_value=3.0):
    """
    Compute Compass Scores.

    Higher score = BETTER quality (higher ROA, OCF, FCF, GP; lower vol, lower asset growth)
    """
    df = factors_df.copy()

    # Compute z-scores
    for factor in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d', 'asset_growth']:
        mean = UNIVERSE_STATS[factor]['mean']
        std = UNIVERSE_STATS[factor]['std']
        z = (df[factor] - mean) / std

        if cap_zscore:
            z = np.clip(z, -cap_value, cap_value)

        df[f'{factor}_z'] = z

    # Composite score
    # Higher ROA/OCF/FCF/GP = positive contribution
    # Lower vol/asset_growth = positive contribution (hence the negative)
    df['compass_score'] = (
        df['roa_z'] * WEIGHTS['roa'] +
        df['ocf_assets_z'] * WEIGHTS['ocf_assets'] +
        df['fcf_assets_z'] * WEIGHTS['fcf_assets'] +
        df['gp_assets_z'] * WEIGHTS['gp_assets'] +
        (-df['vol_60d_z']) * WEIGHTS['vol_60d'] +
        (-df['asset_growth_z']) * WEIGHTS['asset_growth']
    )

    return df


def find_extreme_zscore_stocks(df):
    """Find stocks with extreme z-scores in any factor."""
    extreme = []
    for factor in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d', 'asset_growth']:
        z_col = f'{factor}_z'
        extremes = df[abs(df[z_col]) > 3.0][['symbol', factor, z_col]].copy()
        extremes['factor'] = factor
        extreme.append(extremes)

    if extreme:
        return pd.concat(extreme, ignore_index=True)
    return pd.DataFrame()


def main():
    print("=" * 70)
    print("Z-SCORE CAPPING IMPACT TEST")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 70)

    # Load data
    fund, prices = load_data()

    # Compute TTM factors
    print("\nComputing TTM factors...")
    fund = compute_ttm_and_factors(fund)

    # Compute volatility
    print("Computing volatility...")
    prices = compute_volatility(prices)

    # Get latest data for each stock (as of most recent date)
    latest_date = fund['date'].max()
    print(f"\nUsing data as of: {latest_date.strftime('%Y-%m-%d')}")

    # Merge fundamentals with volatility
    latest_fund = fund[fund['date'] == latest_date].copy()
    latest_vol = prices[prices['date'] == latest_date][['symbol', 'vol_60d']]

    merged = latest_fund.merge(latest_vol, on='symbol', how='inner')
    print(f"  {len(merged):,} stocks with complete data")

    # Filter out extreme/invalid data
    merged = merged[
        (merged['roa'].between(-0.5, 2.0)) &
        (merged['ocf_assets'].between(-1.0, 1.5)) &
        (merged['fcf_assets'].between(-1.0, 1.5)) &
        (merged['gp_assets'].between(-0.5, 2.0)) &
        (merged['asset_growth'].between(-0.5, 5.0)) &
        (merged['vol_60d'] > 0) &
        (merged['vol_60d'] < 200)
    ].dropna(subset=['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d'])

    print(f"  {len(merged):,} stocks after quality filters")

    # Compute scores WITHOUT capping
    print("\n" + "-" * 70)
    print("SCORING WITHOUT Z-SCORE CAPPING (Current Method)")
    print("-" * 70)

    df_uncapped = compute_scores(merged, cap_zscore=False)

    # Find extreme z-scores
    extremes = find_extreme_zscore_stocks(df_uncapped)
    extreme_count = extremes['symbol'].nunique()
    print(f"  Stocks with |z-score| > 3.0 in any factor: {extreme_count}")

    if len(extremes) > 0:
        print("\n  Top 10 most extreme z-scores:")
        extremes['abs_z'] = abs(extremes.iloc[:, 2])
        top_extreme = extremes.nlargest(10, 'abs_z')
        for _, row in top_extreme.iterrows():
            print(f"    {row['symbol']}: {row['factor']} z-score = {row.iloc[2]:.2f}")

    # Score distribution
    print(f"\n  Score distribution:")
    print(f"    Min:    {df_uncapped['compass_score'].min():.2f}")
    print(f"    25%:    {df_uncapped['compass_score'].quantile(0.25):.2f}")
    print(f"    Median: {df_uncapped['compass_score'].median():.2f}")
    print(f"    75%:    {df_uncapped['compass_score'].quantile(0.75):.2f}")
    print(f"    Max:    {df_uncapped['compass_score'].max():.2f}")

    # Compute scores WITH capping at ±3.0
    print("\n" + "-" * 70)
    print("SCORING WITH Z-SCORE CAPPING (±3.0)")
    print("-" * 70)

    df_capped = compute_scores(merged, cap_zscore=True, cap_value=3.0)

    print(f"\n  Score distribution:")
    print(f"    Min:    {df_capped['compass_score'].min():.2f}")
    print(f"    25%:    {df_capped['compass_score'].quantile(0.25):.2f}")
    print(f"    Median: {df_capped['compass_score'].median():.2f}")
    print(f"    75%:    {df_capped['compass_score'].quantile(0.75):.2f}")
    print(f"    Max:    {df_capped['compass_score'].max():.2f}")

    # Compare rank changes
    print("\n" + "-" * 70)
    print("RANK CHANGES")
    print("-" * 70)

    df_uncapped['rank_uncapped'] = df_uncapped['compass_score'].rank(ascending=False)
    df_capped['rank_capped'] = df_capped['compass_score'].rank(ascending=False)

    comparison = df_uncapped[['symbol', 'compass_score', 'rank_uncapped']].merge(
        df_capped[['symbol', 'compass_score', 'rank_capped']],
        on='symbol',
        suffixes=('_uncapped', '_capped')
    )
    comparison['rank_change'] = comparison['rank_uncapped'] - comparison['rank_capped']

    # Stocks that dropped most (benefited from no cap)
    biggest_drops = comparison.nlargest(10, 'rank_change')
    print("\n  Top 10 stocks that DROPPED rank after capping:")
    print("  (These were inflated by extreme z-scores)")
    for _, row in biggest_drops.iterrows():
        print(f"    {row['symbol']}: rank {int(row['rank_uncapped'])} → {int(row['rank_capped'])} "
              f"(dropped {int(row['rank_change'])} positions)")

    # Show their extreme factors
    print("\n  Their extreme factors:")
    for symbol in biggest_drops['symbol'].head(5):
        row = df_uncapped[df_uncapped['symbol'] == symbol].iloc[0]
        print(f"    {symbol}:")
        for factor in ['gp_assets', 'roa', 'ocf_assets', 'fcf_assets']:
            z = row[f'{factor}_z']
            if abs(z) > 2.0:
                print(f"      {factor}: {row[factor]:.2f} (z={z:.2f})")

    # Stocks that improved most
    biggest_rises = comparison.nsmallest(10, 'rank_change')
    print("\n  Top 10 stocks that ROSE rank after capping:")
    print("  (These were pushed down by others with extreme z-scores)")
    for _, row in biggest_rises.iterrows():
        print(f"    {row['symbol']}: rank {int(row['rank_uncapped'])} → {int(row['rank_capped'])} "
              f"(rose {int(-row['rank_change'])} positions)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    corr = comparison['compass_score_uncapped'].corr(comparison['compass_score_capped'])
    print(f"\n  Score correlation (uncapped vs capped): {corr:.4f}")

    rank_corr = comparison['rank_uncapped'].corr(comparison['rank_capped'])
    print(f"  Rank correlation: {rank_corr:.4f}")

    large_moves = (abs(comparison['rank_change']) > len(comparison) * 0.05).sum()
    print(f"  Stocks with >5% rank change: {large_moves} ({100*large_moves/len(comparison):.1f}%)")

    very_large_moves = (abs(comparison['rank_change']) > len(comparison) * 0.10).sum()
    print(f"  Stocks with >10% rank change: {very_large_moves} ({100*very_large_moves/len(comparison):.1f}%)")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if extreme_count > len(merged) * 0.02:
        print(f"\n  ⚠️  {extreme_count} stocks ({100*extreme_count/len(merged):.1f}%) have extreme z-scores")
        print("  Z-score capping at ±3.0 would normalize these outliers")
        print("  without significantly changing rankings for most stocks.")
    else:
        print(f"\n  ✓ Only {extreme_count} stocks have extreme z-scores")
        print("  Z-score capping may not be necessary.")

    return df_uncapped, df_capped, comparison


if __name__ == '__main__':
    df_uncapped, df_capped, comparison = main()

#!/usr/bin/env python3
"""
V4 Factor Contribution Analysis
================================
Compute how much each factor contributes to the Q5-Q1 spread.
Uses the updated weights (GP/Assets = 20%).
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Updated V4 Factor Weights (sum to 100%)
V4_WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,  # Novy-Marx gross profitability premium
    'vol_60d': -0.15,  # Negative = prefer low volatility
    'asset_growth': -0.15,  # Negative = prefer low growth (conservative)
}

OOS_START = '2020-01-01'
FORWARD_DAYS = 63


def load_data():
    """Load price and fundamental data."""
    print("Loading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Load fundamentals
    fund = pd.read_sql_query("""
        SELECT
            i.symbol, i.date,
            i.net_income, b.total_assets, c.operating_cash_flow,
            c.free_cash_flow, i.gross_profit,
            b.total_assets as assets_current
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()
    return prices, fund


def compute_factors(fund):
    """Compute fundamental factors."""
    print("\nComputing factors...")

    # Quality factors
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']

    # Asset growth (YoY)
    fund = fund.sort_values(['symbol', 'date'])
    fund['assets_lag'] = fund.groupby('symbol')['total_assets'].shift(4)
    fund['asset_growth'] = (fund['total_assets'] / fund['assets_lag']) - 1

    # Clean
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    fund = fund.dropna(subset=['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth'])
    print(f"  {len(fund):,} records with all factors")

    return fund


def compute_volatility(prices, symbols):
    """Compute 60-day volatility for each symbol."""
    print("\nComputing volatility...")

    results = []
    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(symbols)}...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        if len(sym_prices) < 100:
            continue

        sym_prices['return'] = sym_prices['close'].pct_change()
        sym_prices['vol_60d'] = sym_prices['return'].rolling(60).std() * np.sqrt(252)

        for _, row in sym_prices[sym_prices['vol_60d'].notna()].iterrows():
            results.append({
                'symbol': symbol,
                'date': row['date'],
                'vol_60d': row['vol_60d'],
                'close': row['close']
            })

    return pd.DataFrame(results)


def compute_forward_returns(prices, symbols):
    """Compute forward returns."""
    print("\nComputing forward returns...")

    results = []
    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(symbols)}...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < FORWARD_DAYS + 50:
            continue

        close = sym_prices['close'].values
        dates = sym_prices['date'].values

        for j in range(n - FORWARD_DAYS):
            entry = close[j]
            exit_price = close[j + FORWARD_DAYS]

            if entry > 0:
                fwd_ret = ((exit_price / entry) - 1) * 100
                fwd_ret = np.clip(fwd_ret, -100, 200)

                results.append({
                    'symbol': symbol,
                    'date': dates[j],
                    'fwd_return': fwd_ret
                })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("V4 FACTOR CONTRIBUTION ANALYSIS")
    print("Updated weights: GP/Assets = 20%")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load data
    prices, fund = load_data()

    # Compute factors
    fund = compute_factors(fund)

    # Get unique symbols
    symbols = fund['symbol'].unique()
    print(f"\n{len(symbols)} unique symbols")

    # Sample for speed
    np.random.seed(42)
    if len(symbols) > 2000:
        symbols = np.random.choice(symbols, size=2000, replace=False)
        print(f"Sampled to {len(symbols)} symbols")

    # Compute volatility
    vol_df = compute_volatility(prices, symbols)
    print(f"  {len(vol_df):,} volatility records")

    # Compute forward returns
    fwd_df = compute_forward_returns(prices, symbols)
    print(f"  {len(fwd_df):,} forward return records")

    # Merge everything
    print("\nMerging data...")

    # Round dates for merging
    fund['month'] = fund['date'].dt.to_period('M')
    vol_df['month'] = vol_df['date'].dt.to_period('M')
    fwd_df['month'] = fwd_df['date'].dt.to_period('M')

    # Get latest fundamental per symbol-month
    fund_monthly = fund.sort_values('date').groupby(['symbol', 'month']).last().reset_index()

    # Merge
    df = vol_df.merge(fund_monthly[['symbol', 'month', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']],
                      on=['symbol', 'month'], how='inner')
    df = df.merge(fwd_df[['symbol', 'month', 'fwd_return']], on=['symbol', 'month'], how='inner')

    print(f"  {len(df):,} merged records")

    # Filter to OOS period
    df = df[df['date'] >= OOS_START]
    print(f"  {len(df):,} OOS records (2020+)")

    # Split into IS (for z-score calculation) and OOS (for evaluation)
    is_df = df[df['date'] < OOS_START].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\n  OOS: {len(oos_df):,} observations")

    # Compute z-scores using OOS data (since we don't have much IS data after merge)
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'vol_60d', 'asset_growth']

    for col in factor_cols:
        # Winsorize
        lower = oos_df[col].quantile(0.01)
        upper = oos_df[col].quantile(0.99)
        oos_df[col] = oos_df[col].clip(lower, upper)

        # Z-score
        mean = oos_df[col].mean()
        std = oos_df[col].std()
        oos_df[f'{col}_z'] = ((oos_df[col] - mean) / std).clip(-3, 3)

    # Compute V4 score
    oos_df['v4_score'] = (
        oos_df['roa_z'] * V4_WEIGHTS['roa'] +
        oos_df['ocf_assets_z'] * V4_WEIGHTS['ocf_assets'] +
        oos_df['fcf_assets_z'] * V4_WEIGHTS['fcf_assets'] +
        oos_df['gp_assets_z'] * V4_WEIGHTS['gp_assets'] +
        oos_df['vol_60d_z'] * V4_WEIGHTS['vol_60d'] +
        oos_df['asset_growth_z'] * V4_WEIGHTS['asset_growth']
    )

    # Assign quintiles
    oos_df['quintile'] = pd.qcut(oos_df['v4_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Overall V4 performance
    print("\n" + "=" * 70)
    print("V4 QUINTILE PERFORMANCE (OOS 2020+)")
    print("=" * 70)

    quintile_stats = oos_df.groupby('quintile').agg({
        'fwd_return': ['mean', 'median', 'count'],
        'v4_score': 'mean'
    }).round(2)

    print(f"\n{'Quintile':<10} {'Avg Return':>12} {'Median':>10} {'Count':>10} {'Avg Score':>12}")
    print("-" * 60)

    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        avg_ret = oos_df[oos_df['quintile'] == q]['fwd_return'].mean()
        med_ret = oos_df[oos_df['quintile'] == q]['fwd_return'].median()
        count = len(oos_df[oos_df['quintile'] == q])
        avg_score = oos_df[oos_df['quintile'] == q]['v4_score'].mean()
        print(f"{q:<10} {avg_ret:>+11.2f}% {med_ret:>+9.2f}% {count:>10,} {avg_score:>+11.3f}")

    q5_ret = oos_df[oos_df['quintile'] == 'Q5']['fwd_return'].mean()
    q1_ret = oos_df[oos_df['quintile'] == 'Q1']['fwd_return'].mean()
    spread = q5_ret - q1_ret

    print("-" * 60)
    print(f"{'Q5-Q1 Spread':<10} {spread:>+11.2f}%")

    # ==========================================
    # FACTOR CONTRIBUTION ANALYSIS
    # ==========================================
    print("\n" + "=" * 70)
    print("FACTOR CONTRIBUTION ANALYSIS")
    print("How much does each factor contribute to the Q5-Q1 spread?")
    print("=" * 70)

    # Method: For each factor, compute the difference in mean z-score between Q5 and Q1
    # Then multiply by the weight to get the contribution

    factor_contributions = []

    q5_df = oos_df[oos_df['quintile'] == 'Q5']
    q1_df = oos_df[oos_df['quintile'] == 'Q1']

    for factor in factor_cols:
        z_col = f'{factor}_z'
        weight = V4_WEIGHTS[factor]

        q5_mean_z = q5_df[z_col].mean()
        q1_mean_z = q1_df[z_col].mean()
        z_diff = q5_mean_z - q1_mean_z

        # Score contribution = weight * z_diff
        score_contrib = weight * z_diff

        # Estimate return contribution via regression
        # Simple approximation: factor contribution ~ (Q5_factor_mean - Q1_factor_mean) correlation with returns
        factor_corr = oos_df[z_col].corr(oos_df['fwd_return'])

        factor_contributions.append({
            'factor': factor,
            'weight': abs(weight),
            'direction': '+' if weight > 0 else '-',
            'q5_mean_z': q5_mean_z,
            'q1_mean_z': q1_mean_z,
            'z_diff': z_diff,
            'score_contrib': score_contrib,
            'factor_corr': factor_corr,
        })

    contrib_df = pd.DataFrame(factor_contributions)

    # Compute return contribution using univariate spreads
    print("\nMethod: Single-factor quintile spreads")
    print("-" * 70)

    factor_spreads = []
    for factor in factor_cols:
        z_col = f'{factor}_z'

        # Create quintiles for this single factor
        oos_df[f'{factor}_quintile'] = pd.qcut(oos_df[z_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        # Compute spread
        q5_ret_f = oos_df[oos_df[f'{factor}_quintile'] == 'Q5']['fwd_return'].mean()
        q1_ret_f = oos_df[oos_df[f'{factor}_quintile'] == 'Q1']['fwd_return'].mean()

        # For negative factors (vol, asset_growth), Q1 is actually "good"
        if V4_WEIGHTS[factor] < 0:
            single_spread = q1_ret_f - q5_ret_f  # Low is good
        else:
            single_spread = q5_ret_f - q1_ret_f  # High is good

        # Weight the contribution
        weighted_contrib = single_spread * abs(V4_WEIGHTS[factor])

        factor_spreads.append({
            'factor': factor,
            'weight': abs(V4_WEIGHTS[factor]) * 100,
            'direction': '+' if V4_WEIGHTS[factor] > 0 else '-',
            'single_factor_spread': single_spread,
            'weighted_contribution': weighted_contrib,
        })

    spread_df = pd.DataFrame(factor_spreads)
    spread_df = spread_df.sort_values('weighted_contribution', ascending=False)

    # Compute percentage of total
    total_contrib = spread_df['weighted_contribution'].sum()
    spread_df['pct_of_total'] = spread_df['weighted_contribution'] / total_contrib * 100

    print(f"\n{'Factor':<18} {'Weight':>8} {'Dir':>5} {'Single Spread':>15} {'Contribution':>14} {'% Total':>10}")
    print("-" * 75)

    for _, row in spread_df.iterrows():
        print(f"{row['factor']:<18} {row['weight']:>7.0f}% {row['direction']:>5} "
              f"{row['single_factor_spread']:>+14.2f}% {row['weighted_contribution']:>+13.2f}% {row['pct_of_total']:>9.1f}%")

    print("-" * 75)
    print(f"{'TOTAL':<18} {'100':>7}% {'':<5} {'':<15} {total_contrib:>+13.2f}%")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY TABLE FOR RESEARCH PAPER")
    print("=" * 70)

    print("\n| Factor | Weight | Contribution | % of Total |")
    print("|--------|--------|--------------|------------|")

    for _, row in spread_df.iterrows():
        factor_name = row['factor'].replace('_', '/').replace('vol/60d', '-Volatility').replace('asset/growth', '-Asset Growth')
        if factor_name == 'roa':
            factor_name = 'ROA'
        elif factor_name == 'ocf/assets':
            factor_name = 'OCF/Assets'
        elif factor_name == 'fcf/assets':
            factor_name = 'FCF/Assets'
        elif factor_name == 'gp/assets':
            factor_name = 'GP/Assets'

        print(f"| {factor_name:<14} | {int(row['weight']):>4}% | {row['weighted_contribution']:>+10.2f}% | {row['pct_of_total']:>8.0f}% |")

    # Check for monotonicity
    print("\n" + "=" * 70)
    print("QUINTILE MONOTONICITY CHECK")
    print("=" * 70)

    quintiles = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    returns = [oos_df[oos_df['quintile'] == q]['fwd_return'].mean() for q in quintiles]

    is_monotonic = all(returns[i] <= returns[i+1] for i in range(4))
    print(f"\nReturns: {' < '.join([f'{r:.2f}%' for r in returns])}")
    print(f"Monotonic: {'YES' if is_monotonic else 'NO'}")

    print(f"\nCompleted: {datetime.now()}")

    return spread_df


if __name__ == '__main__':
    main()

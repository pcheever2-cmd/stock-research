#!/usr/bin/env python3
"""
Multi-Horizon Compass Score Validation
======================================
Test the Compass Score across multiple holding periods:
- 1-month (21 trading days)
- 3-month (63 trading days) - baseline
- 1-year (252 trading days)
- 2-year (504 trading days)

Uses proper in-sample/out-of-sample methodology to avoid look-ahead bias.
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

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'

# Horizons to test (trading days)
HORIZONS = {
    '1-Month': 21,
    '3-Month': 63,
    '1-Year': 252,
    '2-Year': 504
}

# Compass Score weights (sum to 100%)
COMPASS_WEIGHTS = {
    'roa': 0.20,
    'ocf_assets': 0.15,
    'fcf_assets': 0.15,
    'gp_assets': 0.20,
    'vol_60d': -0.15,      # Negative = prefer low volatility
    'asset_growth': -0.15,  # Negative = prefer low asset growth
}


def load_data():
    """Load price and fundamental data."""
    print("Loading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    # Prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    # Fundamentals
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()
    return prices, fund


def compute_fundamental_factors(fund):
    """Compute fundamental factors."""
    print("\nComputing fundamental factors...")

    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    required = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']
    fund = fund.dropna(subset=required)
    print(f"  {len(fund):,} records with all factors")

    return fund


def compute_price_factors_multi_horizon(prices, symbols, horizons):
    """Compute volatility and forward returns for multiple horizons."""
    print(f"\nComputing price factors for {len(symbols)} symbols...")

    max_horizon = max(horizons.values())
    results = []

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        # Need enough data for max horizon
        if n < 300 + max_horizon:
            continue

        close = sym_prices['close'].values
        dates = sym_prices['date'].values

        # Sample monthly
        for j in range(252, n - max_horizon, 21):
            date = dates[j]

            # 60-day volatility (trailing, no look-ahead)
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            row = {
                'symbol': symbol,
                'date': date,
                'vol_60d': vol_60d,
            }

            # Compute forward returns for each horizon
            for horizon_name, days in horizons.items():
                if j + days < n:
                    fwd_ret = ((close[j + days] / close[j]) - 1) * 100
                    fwd_ret = np.clip(fwd_ret, -100, 500)  # Allow higher upside for longer horizons
                    row[f'fwd_{horizon_name}'] = fwd_ret
                else:
                    row[f'fwd_{horizon_name}'] = np.nan

            results.append(row)

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")
    return df


def merge_fundamentals(price_df, fund):
    """Merge fundamentals using backward-looking only (no look-ahead)."""
    print("\nMerging fundamentals (backward-looking only)...")

    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']
    fund_clean = fund[fund_cols].copy()

    all_merged = []
    symbols = price_df['symbol'].unique()

    for symbol in symbols:
        pf = price_df[price_df['symbol'] == symbol].sort_values('date')
        f = fund_clean[fund_clean['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)

    required = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']
    df = df.dropna(subset=required)
    print(f"  {len(df):,} merged observations")

    return df


def compute_compass_score(is_df, oos_df):
    """Compute Compass Score using in-sample statistics."""
    print("\nComputing Compass Score...")

    # Winsorize using in-sample bounds
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']

    for col in factor_cols:
        lower = is_df[col].quantile(0.01)
        upper = is_df[col].quantile(0.99)
        is_df[col] = is_df[col].clip(lower, upper)
        oos_df[col] = oos_df[col].clip(lower, upper)

    # Z-score using in-sample means/stds
    for col in factor_cols:
        mean = is_df[col].mean()
        std = is_df[col].std()

        is_df[f'{col}_z'] = ((is_df[col] - mean) / std).clip(-3, 3).fillna(0)
        oos_df[f'{col}_z'] = ((oos_df[col] - mean) / std).clip(-3, 3).fillna(0)

    # Compute score
    for dataset in [is_df, oos_df]:
        dataset['compass_score'] = (
            dataset['roa_z'] * COMPASS_WEIGHTS['roa'] +
            dataset['ocf_assets_z'] * COMPASS_WEIGHTS['ocf_assets'] +
            dataset['fcf_assets_z'] * COMPASS_WEIGHTS['fcf_assets'] +
            dataset['gp_assets_z'] * COMPASS_WEIGHTS['gp_assets'] +
            dataset['vol_60d_z'] * COMPASS_WEIGHTS['vol_60d'] +
            dataset['asset_growth_z'] * COMPASS_WEIGHTS['asset_growth']
        )

    return is_df, oos_df


def analyze_horizon(df, horizon_name, return_col):
    """Analyze quintile performance for a specific horizon."""
    clean = df.dropna(subset=['compass_score', return_col])

    if len(clean) < 100:
        return None

    # Correlation
    corr = clean['compass_score'].corr(clean[return_col])

    # Quintiles
    try:
        clean['quintile'] = pd.qcut(clean['compass_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    except ValueError:
        clean['quintile'] = pd.cut(clean['compass_score'].rank(pct=True),
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quintile_returns = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = clean[clean['quintile'] == q]
        quintile_returns[q] = q_data[return_col].mean()

    spread = quintile_returns['Q5'] - quintile_returns['Q1']

    # Check monotonicity
    returns_list = [quintile_returns[q] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    monotonic = all(returns_list[i] <= returns_list[i+1] for i in range(4))

    return {
        'horizon': horizon_name,
        'n_obs': len(clean),
        'correlation': corr,
        'Q1': quintile_returns['Q1'],
        'Q2': quintile_returns['Q2'],
        'Q3': quintile_returns['Q3'],
        'Q4': quintile_returns['Q4'],
        'Q5': quintile_returns['Q5'],
        'spread': spread,
        'monotonic': monotonic
    }


def main():
    print("=" * 70)
    print("MULTI-HORIZON COMPASS SCORE VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nHorizons: {', '.join(HORIZONS.keys())}")
    print(f"In-Sample: ≤ {IN_SAMPLE_END}")
    print(f"Out-of-Sample: ≥ {OOS_START}")

    # Load data
    prices, fund = load_data()
    fund = compute_fundamental_factors(fund)

    # Sample symbols for speed (or use all)
    symbols = fund['symbol'].unique()
    print(f"\n{len(symbols)} symbols with fundamental data")

    np.random.seed(42)
    if len(symbols) > 2500:
        symbols = np.random.choice(symbols, size=2500, replace=False)
        print(f"Sampled to {len(symbols)} symbols for speed")

    # Compute price factors
    price_df = compute_price_factors_multi_horizon(prices, symbols, HORIZONS)

    # Merge
    df = merge_fundamentals(price_df, fund)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Split
    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()

    print(f"\nIn-Sample:  {len(is_df):,} observations")
    print(f"Out-of-Sample: {len(oos_df):,} observations")

    # Compute scores
    is_df, oos_df = compute_compass_score(is_df, oos_df)

    # ==========================================
    # RESULTS
    # ==========================================

    print("\n" + "=" * 70)
    print("MULTI-HORIZON RESULTS (OUT-OF-SAMPLE 2020+)")
    print("=" * 70)

    results = []
    for horizon_name, days in HORIZONS.items():
        return_col = f'fwd_{horizon_name}'
        result = analyze_horizon(oos_df, horizon_name, return_col)
        if result:
            results.append(result)

    # Print summary table
    print(f"\n{'Horizon':<12} {'N Obs':>10} {'Corr':>8} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Q5':>10} {'Spread':>10} {'Mono':>6}")
    print("-" * 110)

    for r in results:
        mono_str = "YES" if r['monotonic'] else "NO"
        print(f"{r['horizon']:<12} {r['n_obs']:>10,} {r['correlation']:>+7.3f} "
              f"{r['Q1']:>+9.2f}% {r['Q2']:>+9.2f}% {r['Q3']:>+9.2f}% "
              f"{r['Q4']:>+9.2f}% {r['Q5']:>+9.2f}% {r['spread']:>+9.2f}% {mono_str:>6}")

    # Annualized spreads
    print("\n" + "=" * 70)
    print("ANNUALIZED Q5-Q1 SPREADS")
    print("=" * 70)

    print(f"\n{'Horizon':<12} {'Raw Spread':>12} {'Annualized':>14}")
    print("-" * 40)

    for r in results:
        horizon = r['horizon']
        spread = r['spread']

        # Annualize
        if '1-Month' in horizon:
            annual = spread * 12
        elif '3-Month' in horizon:
            annual = spread * 4
        elif '1-Year' in horizon:
            annual = spread
        elif '2-Year' in horizon:
            annual = spread / 2
        else:
            annual = spread

        print(f"{horizon:<12} {spread:>+11.2f}% {annual:>+13.2f}%")

    # In-Sample comparison (for paper)
    print("\n" + "=" * 70)
    print("IN-SAMPLE RESULTS (1995-2019) FOR REFERENCE")
    print("=" * 70)

    is_results = []
    for horizon_name, days in HORIZONS.items():
        return_col = f'fwd_{horizon_name}'
        result = analyze_horizon(is_df, horizon_name, return_col)
        if result:
            is_results.append(result)

    print(f"\n{'Horizon':<12} {'N Obs':>10} {'Corr':>8} {'Q5-Q1 Spread':>14} {'Mono':>6}")
    print("-" * 55)

    for r in is_results:
        mono_str = "YES" if r['monotonic'] else "NO"
        print(f"{r['horizon']:<12} {r['n_obs']:>10,} {r['correlation']:>+7.3f} {r['spread']:>+13.2f}% {mono_str:>6}")

    # Table for paper
    print("\n" + "=" * 70)
    print("TABLE FOR RESEARCH PAPER")
    print("=" * 70)

    print("\n| Horizon | N (OOS) | Correlation | Q5-Q1 Spread | Annualized | Monotonic |")
    print("|---------|---------|-------------|--------------|------------|-----------|")

    for r in results:
        horizon = r['horizon']
        spread = r['spread']

        if '1-Month' in horizon:
            annual = spread * 12
        elif '3-Month' in horizon:
            annual = spread * 4
        elif '1-Year' in horizon:
            annual = spread
        elif '2-Year' in horizon:
            annual = spread / 2
        else:
            annual = spread

        mono_str = "Yes" if r['monotonic'] else "No"
        print(f"| {horizon:<7} | {r['n_obs']:>7,} | {r['correlation']:>+10.3f} | {spread:>+11.2f}% | {annual:>+9.2f}% | {mono_str:<9} |")

    print(f"\nCompleted: {datetime.now()}")

    return results


if __name__ == '__main__':
    main()

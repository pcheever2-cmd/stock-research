#!/usr/bin/env python3
"""
V4 Full Validation Suite
========================
1. Multivariate regression (which factors survive together?)
2. Full universe backtest (all symbols, not sample)
3. Liquidity filter (>$1M daily volume)
4. Sector-neutral analysis
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
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Minimum liquidity threshold
MIN_DOLLAR_VOL = 1_000_000  # $1M daily volume


def load_full_dataset():
    """Load full dataset with all symbols."""
    print("=" * 70)
    print("LOADING FULL DATASET")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices with volume
    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    # Load fundamentals
    print("\nLoading fundamentals...")
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap, m.ev_to_ebitda
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        LEFT JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()

    # Load sector mapping
    try:
        main_conn = sqlite3.connect(MAIN_DB)
        sectors = pd.read_sql_query(
            "SELECT symbol, sector FROM stock_consensus", main_conn)
        main_conn.close()
        sector_map = dict(zip(sectors['symbol'], sectors['sector']))
        print(f"  {len(sector_map):,} sector mappings")
    except:
        sector_map = {}
        print("  No sector data available")

    return prices, fund, sector_map


def compute_factors(prices: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Compute all factors for full universe."""
    print("\n" + "=" * 70)
    print("COMPUTING FACTORS (FULL UNIVERSE)")
    print("=" * 70)

    # Compute fundamental factors
    print("\nComputing fundamental factors...")
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['fcf_yield'] = fund['free_cash_flow'] / fund['market_cap']

    # Asset growth
    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
            lower = fund[col].quantile(0.01)
            upper = fund[col].quantile(0.99)
            fund[col] = fund[col].clip(lower, upper)

    # Compute price factors
    print("\nComputing price factors (all symbols)...")
    symbols = prices['symbol'].unique()
    n_symbols = len(symbols)

    results = []

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i:,}/{n_symbols:,} symbols...")

        sym = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym)

        if n < 300:
            continue

        close = sym['close'].values
        volume = sym['volume'].values

        # Sample monthly
        for j in range(252, n - 63, 21):
            date = sym['date'].iloc[j]

            # Forward return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Dollar volume (average 20-day)
            avg_vol = np.mean(volume[j-20:j])
            dollar_vol = close[j] * avg_vol

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'dollar_vol': dollar_vol,
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    print("\nMerging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                 'fcf_yield', 'asset_growth', 'ev_to_ebitda']
    fund = fund[[c for c in fund_cols if c in fund.columns]].dropna(subset=['roa'])

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund[fund['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)

    # Winsorize returns
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)

    # Z-score all factors
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield',
                   'asset_growth', 'vol_60d', 'mom_12_1', 'ev_to_ebitda']

    for col in factor_cols:
        if col in df.columns:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_z'] = df[f'{col}_z'].clip(-3, 3).fillna(0)

    # Compute Static V4 score
    df['static_v4'] = (
        df['roa_z'] * 0.12 +
        df['ocf_assets_z'] * 0.12 +
        df['fcf_assets_z'] * 0.12 +
        df['fcf_yield_z'] * 0.15 +
        df['gp_assets_z'] * 0.10 +
        (-df['vol_60d_z']) * 0.10 +
        (-df['asset_growth_z']) * 0.10 +
        df['mom_12_1_z'] * 0.05
    )

    return df


def run_multivariate_regression(df: pd.DataFrame):
    """Run multivariate OLS to see which factors survive."""
    print("\n" + "=" * 70)
    print("MULTIVARIATE FACTOR REGRESSION")
    print("=" * 70)

    # Clean data
    factor_cols = ['roa_z', 'ocf_assets_z', 'fcf_assets_z', 'fcf_yield_z',
                   'gp_assets_z', 'vol_60d_z', 'asset_growth_z', 'mom_12_1_z']

    available = [c for c in factor_cols if c in df.columns]
    clean = df.dropna(subset=['fwd_3m'] + available).copy()

    print(f"\nObservations: {len(clean):,}")

    # Build design matrix
    X = clean[available].values
    y = clean['fwd_3m'].values

    # Add constant
    X = np.column_stack([np.ones(len(X)), X])

    # OLS: beta = (X'X)^-1 X'y
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y

        # Standard errors
        y_hat = X @ beta
        residuals = y - y_hat
        n, k = X.shape
        mse = np.sum(residuals**2) / (n - k)
        se = np.sqrt(np.diag(XtX_inv) * mse)

        # T-statistics
        t_stats = beta / se

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\nR² = {r_squared:.4f}")
        print(f"\n{'Factor':<20} {'Coef':>10} {'T-Stat':>10} {'Sig':>6}")
        print("-" * 50)

        print(f"{'(Intercept)':<20} {beta[0]:>+9.4f} {t_stats[0]:>+9.2f}")

        results = []
        for i, factor in enumerate(available):
            idx = i + 1
            sig = '***' if abs(t_stats[idx]) > 2.58 else '**' if abs(t_stats[idx]) > 1.96 else '*' if abs(t_stats[idx]) > 1.65 else ''
            results.append((factor, beta[idx], t_stats[idx], sig))

        # Sort by absolute t-stat
        results.sort(key=lambda x: abs(x[2]), reverse=True)

        for factor, coef, t, sig in results:
            print(f"{factor:<20} {coef:>+9.4f} {t:>+9.2f} {sig:>6}")

        # Interpretation
        print("\n--- SURVIVING FACTORS (|t| > 1.96) ---")
        surviving = [(f, c, t) for f, c, t, s in results if abs(t) > 1.96]
        for f, c, t in surviving:
            direction = "POSITIVE" if c > 0 else "NEGATIVE"
            print(f"  {f}: {direction} effect (t={t:+.2f})")

        # Redundancy check
        print("\n--- REDUNDANCY CHECK ---")
        redundant = [(f, c, t) for f, c, t, s in results if abs(t) < 1.5]
        if redundant:
            print("  These factors add little incremental value:")
            for f, c, t in redundant:
                print(f"    {f} (t={t:+.2f}) - consider removing")
        else:
            print("  All factors contribute meaningfully")

    except np.linalg.LinAlgError:
        print("  ERROR: Singular matrix (multicollinearity issue)")


def run_full_backtest(df: pd.DataFrame, liquidity_filter: bool = False):
    """Run backtest on full universe with optional liquidity filter."""
    print("\n" + "=" * 70)
    title = "FULL UNIVERSE BACKTEST" if not liquidity_filter else f"LIQUID STOCKS BACKTEST (>${MIN_DOLLAR_VOL/1e6:.0f}M daily vol)"
    print(title)
    print("=" * 70)

    if liquidity_filter:
        df = df[df['dollar_vol'] >= MIN_DOLLAR_VOL].copy()

    clean = df.dropna(subset=['fwd_3m', 'static_v4'])

    print(f"\nObservations: {len(clean):,}")
    print(f"Symbols: {clean['symbol'].nunique():,}")

    # Correlation
    corr = clean['static_v4'].corr(clean['fwd_3m'])

    # Quintile analysis
    q5 = clean[clean['static_v4'] >= clean['static_v4'].quantile(0.8)]
    q4 = clean[(clean['static_v4'] >= clean['static_v4'].quantile(0.6)) &
               (clean['static_v4'] < clean['static_v4'].quantile(0.8))]
    q3 = clean[(clean['static_v4'] >= clean['static_v4'].quantile(0.4)) &
               (clean['static_v4'] < clean['static_v4'].quantile(0.6))]
    q2 = clean[(clean['static_v4'] >= clean['static_v4'].quantile(0.2)) &
               (clean['static_v4'] < clean['static_v4'].quantile(0.4))]
    q1 = clean[clean['static_v4'] < clean['static_v4'].quantile(0.2)]

    print(f"\n{'Quintile':<10} {'Avg Return':>12} {'Median':>10} {'Std':>10} {'Count':>10}")
    print("-" * 55)

    quintiles = [('Q5 (Best)', q5), ('Q4', q4), ('Q3', q3), ('Q2', q2), ('Q1 (Worst)', q1)]
    for name, q in quintiles:
        print(f"{name:<10} {q['fwd_3m'].mean():>+11.2f}% {q['fwd_3m'].median():>+9.2f}% "
              f"{q['fwd_3m'].std():>9.1f} {len(q):>10,}")

    spread = q5['fwd_3m'].mean() - q1['fwd_3m'].mean()

    print("-" * 55)
    print(f"\nKEY METRICS:")
    print(f"  Correlation:   {corr:>+.4f}")
    print(f"  Q5-Q1 Spread:  {spread:>+.2f}% (per quarter)")
    print(f"  Annualized:    {spread * 4:>+.1f}%")

    # T-test
    n = len(clean)
    t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0
    print(f"  T-statistic:   {t_stat:>+.1f}")

    # Monotonicity
    q_returns = [q['fwd_3m'].mean() for _, q in quintiles]
    monotonic = all(q_returns[i] >= q_returns[i+1] for i in range(len(q_returns)-1))
    print(f"  Monotonic:     {'YES' if monotonic else 'NO'}")

    return {'corr': corr, 'spread': spread, 't_stat': t_stat, 'n': len(clean)}


def run_sector_neutral_analysis(df: pd.DataFrame, sector_map: dict):
    """Run sector-neutral quintile analysis."""
    print("\n" + "=" * 70)
    print("SECTOR-NEUTRAL ANALYSIS")
    print("=" * 70)

    # Add sector
    df['sector'] = df['symbol'].map(sector_map)
    df = df.dropna(subset=['sector', 'fwd_3m', 'static_v4'])

    print(f"\nObservations with sector: {len(df):,}")
    print(f"Sectors: {df['sector'].nunique()}")

    # Rank within each sector
    all_quintiles = []

    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector].copy()

        if len(sector_df) < 100:
            continue

        # Quintile within sector
        try:
            sector_df['sector_quintile'] = pd.qcut(
                sector_df['static_v4'], q=5,
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
            all_quintiles.append(sector_df)
        except:
            continue

    if not all_quintiles:
        print("  Insufficient sector data")
        return

    combined = pd.concat(all_quintiles, ignore_index=True)

    # Aggregate by sector-neutral quintile
    print(f"\nSector-neutral quintiles (ranking within each sector):")
    print(f"{'Quintile':<10} {'Avg Return':>12} {'Std':>10} {'Count':>10}")
    print("-" * 45)

    for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
        q_data = combined[combined['sector_quintile'] == q]
        print(f"{q:<10} {q_data['fwd_3m'].mean():>+11.2f}% "
              f"{q_data['fwd_3m'].std():>9.1f} {len(q_data):>10,}")

    q5 = combined[combined['sector_quintile'] == 'Q5']['fwd_3m'].mean()
    q1 = combined[combined['sector_quintile'] == 'Q1']['fwd_3m'].mean()
    spread = q5 - q1

    print("-" * 45)
    print(f"\nSector-Neutral Q5-Q1 Spread: {spread:>+.2f}%")

    if spread > 5:
        print("  STRONG stock-picking signal (spread > 5%)")
    elif spread > 2:
        print("  MODERATE stock-picking signal (spread 2-5%)")
    elif spread > 0:
        print("  WEAK stock-picking signal (spread 0-2%)")
    else:
        print("  NO stock-picking signal (spread <= 0)")

    # Compare to raw spread
    raw_q5 = df[df['static_v4'] >= df['static_v4'].quantile(0.8)]['fwd_3m'].mean()
    raw_q1 = df[df['static_v4'] < df['static_v4'].quantile(0.2)]['fwd_3m'].mean()
    raw_spread = raw_q5 - raw_q1

    print(f"\nComparison:")
    print(f"  Raw spread:           {raw_spread:>+.2f}%")
    print(f"  Sector-neutral spread: {spread:>+.2f}%")
    print(f"  Difference:           {raw_spread - spread:>+.2f}%")

    if raw_spread - spread > 2:
        print("  Some of the raw spread may be sector timing, not pure stock-picking")
    else:
        print("  V4 signal is primarily stock-picking, not sector timing")


def main():
    print("=" * 70)
    print("V4 FULL VALIDATION SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load data
    prices, fund, sector_map = load_full_dataset()

    # Compute factors
    df = compute_factors(prices, fund)

    # 1. Multivariate regression
    run_multivariate_regression(df)

    # 2. Full universe backtest
    full_results = run_full_backtest(df, liquidity_filter=False)

    # 3. Liquid stocks backtest
    liquid_results = run_full_backtest(df, liquidity_filter=True)

    # 4. Sector-neutral analysis
    run_sector_neutral_analysis(df.copy(), sector_map)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"""
Static V4 Scoring Formula:
  ROA              × 12%
  OCF/Assets       × 12%
  FCF/Assets       × 12%
  FCF Yield        × 15%
  GP/Assets        × 10%
  -Volatility      × 10%
  -Asset Growth    × 10%
  Momentum 12-1    ×  5%
  (Other)          × 14%
  ────────────────────────
  Total            100%

Results:
  Full Universe:
    - Correlation:     {full_results['corr']:>+.4f}
    - Q5-Q1 Spread:    {full_results['spread']:>+.2f}%
    - T-statistic:     {full_results['t_stat']:>+.1f}
    - Observations:    {full_results['n']:>,}

  Liquid Stocks (>${MIN_DOLLAR_VOL/1e6:.0f}M vol):
    - Correlation:     {liquid_results['corr']:>+.4f}
    - Q5-Q1 Spread:    {liquid_results['spread']:>+.2f}%
    - T-statistic:     {liquid_results['t_stat']:>+.1f}
    - Observations:    {liquid_results['n']:>,}
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

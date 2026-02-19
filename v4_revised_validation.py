#!/usr/bin/env python3
"""
V4 Revised Validation
Based on multivariate regression results, revises the V4 formula
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


def main():
    print("=" * 70)
    print("V4 REVISED VALIDATION")
    print("Based on Multivariate Regression Results")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    conn = sqlite3.connect(BACKTEST_DB)

    # Load data (sampling for speed)
    print("\n" + "=" * 70)
    print("LOADING DATA (Full Universe)")
    print("=" * 70)

    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    print("\nLoading fundamentals...")
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()

    # Compute fundamental factors
    print("\n" + "=" * 70)
    print("COMPUTING FACTORS")
    print("=" * 70)

    print("\nComputing fundamental factors...")
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['fcf_yield'] = fund['free_cash_flow'] / fund['market_cap']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
        lower = fund[col].quantile(0.01)
        upper = fund[col].quantile(0.99)
        fund[col] = fund[col].clip(lower, upper)

    # Compute price factors - sample 2000 symbols for speed
    print("\nComputing price factors (sampling 2000 symbols)...")
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(2000, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    results = []
    for i, symbol in enumerate(sample_symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(sample_symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values

        # Sample monthly
        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum 12-1
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Average daily volume ($)
            avg_dollar_vol = np.mean(volume[j-20:j] * close[j-20:j]) if j >= 20 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'avg_dollar_vol': avg_dollar_vol,
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    print("\nMerging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth']
    fund = fund[fund_cols].dropna(subset=['roa'])

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

    # Clean
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)
    df = df.dropna(subset=['fwd_3m', 'roa'])
    print(f"\nClean observations: {len(df):,}")

    # Z-score factors
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield',
                'asset_growth', 'vol_60d', 'mom_12_1']:
        if col in df.columns:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_z'] = df[f'{col}_z'].clip(-3, 3).fillna(0)

    # ==========================================
    # COMPARE V4 ORIGINAL vs V4 REVISED
    # ==========================================

    print("\n" + "=" * 70)
    print("COMPARING V4 FORMULAS")
    print("=" * 70)

    # V4 Original (as designed)
    df['v4_original'] = (
        df['roa_z'] * 0.12 +
        df['ocf_assets_z'] * 0.12 +
        df['fcf_assets_z'] * 0.12 +
        df['fcf_yield_z'] * 0.15 +  # POSITIVE weight
        df['gp_assets_z'] * 0.10 +
        (-df['vol_60d_z']) * 0.10 +
        (-df['asset_growth_z']) * 0.10 +
        df['mom_12_1_z'] * 0.05  # POSITIVE weight
    )

    # V4 Revised (based on multivariate regression)
    # Remove FCF Yield (negative effect in regression)
    # Remove Momentum (negative effect in regression)
    # Increase weights on factors that work
    df['v4_revised'] = (
        df['roa_z'] * 0.20 +  # Strongest predictor
        df['ocf_assets_z'] * 0.15 +
        df['fcf_assets_z'] * 0.15 +
        df['gp_assets_z'] * 0.10 +
        (-df['vol_60d_z']) * 0.15 +  # Low vol works
        (-df['asset_growth_z']) * 0.15  # Asset growth reversal works
    )
    # Note: FCF Yield and Momentum REMOVED

    # V4 Empirical (using regression coefficients directly)
    # Normalize coefficients to sum to 1
    # From regression: ROA=2.9, AG=-1.8, FCF_Y=-0.7, FCF/A=1.7, Mom=-8.1, OCF=0.8, Vol=-13.2, GP=0.09
    # Take absolute values for weighting, use signs from regression
    df['v4_empirical'] = (
        df['roa_z'] * 0.15 +  # positive
        df['fcf_assets_z'] * 0.10 +  # positive
        df['ocf_assets_z'] * 0.05 +  # positive (weak)
        (-df['asset_growth_z']) * 0.15 +  # negative -> invert
        (-df['vol_60d_z']) * 0.25 +  # negative -> invert (strongest factor!)
        (-df['fcf_yield_z']) * 0.10 +  # negative -> invert
        (-df['mom_12_1_z']) * 0.15  # negative -> invert
        # GP/Assets removed (not significant)
    )

    # Analyze each strategy
    strategies = {
        'V4 Original': 'v4_original',
        'V4 Revised': 'v4_revised',
        'V4 Empirical': 'v4_empirical',
    }

    print("\n" + "-" * 70)
    print("FULL SAMPLE RESULTS")
    print("-" * 70)

    for name, col in strategies.items():
        clean = df.dropna(subset=[col, 'fwd_3m'])
        corr = clean[col].corr(clean['fwd_3m'])

        clean['quintile'] = pd.qcut(clean[col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        # Monotonicity check
        qrets = clean.groupby('quintile')['fwd_3m'].mean()
        is_monotonic = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))

        print(f"\n{name}:")
        print(f"  Correlation:    {corr:>+.4f}")
        print(f"  Q5 Return:      {q5_ret:>+.2f}%")
        print(f"  Q1 Return:      {q1_ret:>+.2f}%")
        print(f"  Q5-Q1 Spread:   {spread:>+.2f}%")
        print(f"  Monotonic:      {'YES' if is_monotonic else 'NO'}")

    # Detailed quintile analysis for best strategy
    print("\n" + "-" * 70)
    print("QUINTILE DETAIL FOR EACH STRATEGY")
    print("-" * 70)

    for name, col in strategies.items():
        clean = df.dropna(subset=[col, 'fwd_3m'])
        clean['quintile'] = pd.qcut(clean[col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        print(f"\n{name}:")
        print(f"{'Quintile':<12} {'Avg Return':>12} {'Median':>10} {'Std':>10} {'Count':>10}")
        print("-" * 55)

        for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
            q_data = clean[clean['quintile'] == q]['fwd_3m']
            label = f"{q} ({'Best' if q == 'Q5' else 'Worst' if q == 'Q1' else ''})"
            print(f"{label:<12} {q_data.mean():>+10.2f}% {q_data.median():>+10.2f}% {q_data.std():>10.1f} {len(q_data):>10,}")

    # Liquid stocks analysis
    print("\n" + "-" * 70)
    print("LIQUID STOCKS (>$1M daily volume)")
    print("-" * 70)

    liquid = df[df['avg_dollar_vol'] > 1_000_000].copy()
    print(f"Observations: {len(liquid):,}")

    for name, col in strategies.items():
        clean = liquid.dropna(subset=[col, 'fwd_3m'])
        if len(clean) < 1000:
            print(f"\n{name}: Insufficient data")
            continue

        corr = clean[col].corr(clean['fwd_3m'])

        clean['quintile'] = pd.qcut(clean[col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        print(f"\n{name}:")
        print(f"  Correlation:    {corr:>+.4f}")
        print(f"  Q5-Q1 Spread:   {spread:>+.2f}%")

    # Summary
    print("\n" + "=" * 70)
    print("FORMULA COMPARISON")
    print("=" * 70)

    print("""
V4 ORIGINAL (Based on academic priors):
  ROA              × 12%
  OCF/Assets       × 12%
  FCF/Assets       × 12%
  FCF Yield        × 15%  ← REGRESSION SAYS NEGATIVE EFFECT
  GP/Assets        × 10%
  -Volatility      × 10%
  -Asset Growth    × 10%
  Momentum 12-1    ×  5%  ← REGRESSION SAYS NEGATIVE EFFECT

V4 REVISED (Remove problematic factors):
  ROA              × 20%
  OCF/Assets       × 15%
  FCF/Assets       × 15%
  GP/Assets        × 10%
  -Volatility      × 15%
  -Asset Growth    × 15%
  (FCF Yield removed, Momentum removed)

V4 EMPIRICAL (Use regression coefficients):
  ROA              × 15%
  FCF/Assets       × 10%
  OCF/Assets       ×  5%
  -Asset Growth    × 15%
  -Volatility      × 25%  ← Strongest effect
  -FCF Yield       × 10%  ← INVERTED (was positive)
  -Momentum 12-1   × 15%  ← INVERTED (was positive)
  (GP/Assets removed - not significant)
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

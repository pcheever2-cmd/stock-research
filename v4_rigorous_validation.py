#!/usr/bin/env python3
"""
V4 Rigorous Validation
Addresses common pitfalls:
1. Over-segmentation: Only 2 segments (Liquid vs Illiquid)
2. Data sparsity: Strict fundamental data requirements
3. Look-ahead bias: Rolling segment definitions using past data only
4. Test inflation: In-sample (1995-2019) vs Out-of-sample (2020-2026)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'  # Train on data before this
OOS_START = '2020-01-01'      # Test on data after this
LIQUIDITY_THRESHOLD = 1_000_000  # $1M daily volume


def main():
    print("=" * 70)
    print("V4 RIGOROUS VALIDATION")
    print("Avoiding: Over-segmentation, Look-ahead bias, Test inflation")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nMethodology:")
    print(f"  In-Sample Period:  1995 - 2019 (train)")
    print(f"  Out-of-Sample:     2020 - 2026 (test)")
    print(f"  Segments:          2 (Liquid >$1M vol, Illiquid <$1M)")
    print(f"  Bias Prevention:   Rolling lookback for all metrics")

    conn = sqlite3.connect(BACKTEST_DB)

    # Load ALL data
    print("\n" + "=" * 70)
    print("LOADING DATA")
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
    print("\nComputing fundamental factors...")
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['fcf_yield'] = fund['free_cash_flow'] / fund['market_cap']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities (but DON'T winsorize yet - will do in-sample only)
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # STRICT data requirement: must have all key factors
    required_factors = ['roa', 'ocf_assets', 'fcf_assets', 'asset_growth']
    fund = fund.dropna(subset=required_factors)
    print(f"  After strict filter: {len(fund):,} records with all required factors")

    # Compute price factors for ALL symbols
    print("\n" + "=" * 70)
    print("COMPUTING PRICE FACTORS (ALL SYMBOLS)")
    print("=" * 70)

    symbols = prices['symbol'].unique()

    results = []
    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values
        dates = sym_prices['date'].values

        # Sample monthly
        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum 12-1
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility (trailing 60 days - no look-ahead)
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # ROLLING average daily dollar volume (trailing 60 days - no look-ahead)
            if j >= 60:
                avg_dollar_vol = np.mean(volume[j-60:j] * close[j-60:j])
            else:
                avg_dollar_vol = np.nan

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

    # Merge fundamentals (using most recent PRIOR fundamental data only)
    print("\nMerging fundamentals (backward-looking only)...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                 'fcf_yield', 'asset_growth', 'market_cap']
    fund_clean = fund[fund_cols].copy()

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund_clean[fund_clean['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        # merge_asof with backward direction ensures no look-ahead
        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)

    # Clean
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)
    df = df.dropna(subset=['fwd_3m', 'roa', 'ocf_assets', 'fcf_assets', 'asset_growth'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    print(f"\nClean observations: {len(df):,}")

    # Split into In-Sample and Out-of-Sample
    print("\n" + "=" * 70)
    print("SPLITTING INTO IN-SAMPLE AND OUT-OF-SAMPLE")
    print("=" * 70)

    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()

    print(f"  In-Sample (≤{IN_SAMPLE_END}):  {len(is_df):,} observations")
    print(f"  Out-of-Sample (≥{OOS_START}): {len(oos_df):,} observations")

    # Compute winsorization bounds from IN-SAMPLE ONLY
    print("\nComputing factor bounds from IN-SAMPLE data only...")
    factor_bounds = {}
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth', 'vol_60d']:
        if col in is_df.columns:
            lower = is_df[col].quantile(0.01)
            upper = is_df[col].quantile(0.99)
            factor_bounds[col] = (lower, upper)
            print(f"  {col}: [{lower:.4f}, {upper:.4f}]")

    # Apply bounds to BOTH datasets
    for col, (lower, upper) in factor_bounds.items():
        is_df[col] = is_df[col].clip(lower, upper)
        oos_df[col] = oos_df[col].clip(lower, upper)

    # Compute z-scores using IN-SAMPLE means/stds
    print("\nComputing z-scores using IN-SAMPLE statistics...")
    factor_stats = {}
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth', 'vol_60d', 'mom_12_1']:
        if col in is_df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
            factor_stats[col] = (mean, std)
            print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

    # Apply z-scores to BOTH datasets using IN-SAMPLE stats
    for col, (mean, std) in factor_stats.items():
        for dataset in [is_df, oos_df]:
            dataset[f'{col}_z'] = (dataset[col] - mean) / std
            dataset[f'{col}_z'] = dataset[f'{col}_z'].clip(-3, 3).fillna(0)

    # V4 Revised Score
    for dataset in [is_df, oos_df]:
        dataset['v4_score'] = (
            dataset['roa_z'] * 0.20 +
            dataset['ocf_assets_z'] * 0.15 +
            dataset['fcf_assets_z'] * 0.15 +
            dataset['gp_assets_z'] * 0.10 +
            (-dataset['vol_60d_z']) * 0.15 +
            (-dataset['asset_growth_z']) * 0.15
        )

    # Define liquidity segments using TRAILING volume (no look-ahead)
    for dataset in [is_df, oos_df]:
        dataset['is_liquid'] = dataset['avg_dollar_vol'] >= LIQUIDITY_THRESHOLD

    # ==========================================
    # ANALYSIS
    # ==========================================

    def analyze_dataset(data, name, segments=True):
        """Analyze a dataset and return metrics."""
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        results = {}

        # Overall
        clean = data.dropna(subset=['v4_score', 'fwd_3m'])
        corr = clean['v4_score'].corr(clean['fwd_3m'])

        try:
            clean['quintile'] = pd.qcut(clean['v4_score'], 5,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                       duplicates='drop')
        except ValueError:
            print("  Could not compute quintiles")
            return results

        q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        qrets = clean.groupby('quintile')['fwd_3m'].mean()
        is_monotonic = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))

        print(f"\n  OVERALL ({len(clean):,} observations)")
        print(f"    Correlation:   {corr:>+.4f}")
        print(f"    Q5 Return:     {q5_ret:>+.2f}%")
        print(f"    Q1 Return:     {q1_ret:>+.2f}%")
        print(f"    Q5-Q1 Spread:  {spread:>+.2f}%")
        print(f"    Monotonic:     {'YES' if is_monotonic else 'NO'}")

        results['overall'] = {
            'corr': corr, 'spread': spread, 'monotonic': is_monotonic, 'n': len(clean)
        }

        # Quintile detail
        print(f"\n    {'Quintile':<10} {'Avg Ret':>10} {'Median':>10} {'Count':>10}")
        print(f"    {'-'*45}")
        for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
            q_data = clean[clean['quintile'] == q]['fwd_3m']
            print(f"    {q:<10} {q_data.mean():>+9.2f}% {q_data.median():>+9.2f}% {len(q_data):>10,}")

        if segments:
            # Liquid segment
            liquid = data[data['is_liquid'] == True].dropna(subset=['v4_score', 'fwd_3m'])
            if len(liquid) >= 500:
                corr_l = liquid['v4_score'].corr(liquid['fwd_3m'])
                try:
                    liquid['quintile'] = pd.qcut(liquid['v4_score'], 5,
                                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                duplicates='drop')
                    q5_l = liquid[liquid['quintile'] == 'Q5']['fwd_3m'].mean()
                    q1_l = liquid[liquid['quintile'] == 'Q1']['fwd_3m'].mean()
                    spread_l = q5_l - q1_l

                    qrets_l = liquid.groupby('quintile')['fwd_3m'].mean()
                    mono_l = all(qrets_l.iloc[i] <= qrets_l.iloc[i+1] for i in range(len(qrets_l)-1))

                    print(f"\n  LIQUID (>$1M vol) ({len(liquid):,} observations)")
                    print(f"    Correlation:   {corr_l:>+.4f}")
                    print(f"    Q5-Q1 Spread:  {spread_l:>+.2f}%")
                    print(f"    Monotonic:     {'YES' if mono_l else 'NO'}")

                    results['liquid'] = {
                        'corr': corr_l, 'spread': spread_l, 'monotonic': mono_l, 'n': len(liquid)
                    }
                except:
                    pass

            # Illiquid segment
            illiquid = data[data['is_liquid'] == False].dropna(subset=['v4_score', 'fwd_3m'])
            if len(illiquid) >= 500:
                corr_i = illiquid['v4_score'].corr(illiquid['fwd_3m'])
                try:
                    illiquid['quintile'] = pd.qcut(illiquid['v4_score'], 5,
                                                  labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                  duplicates='drop')
                    q5_i = illiquid[illiquid['quintile'] == 'Q5']['fwd_3m'].mean()
                    q1_i = illiquid[illiquid['quintile'] == 'Q1']['fwd_3m'].mean()
                    spread_i = q5_i - q1_i

                    qrets_i = illiquid.groupby('quintile')['fwd_3m'].mean()
                    mono_i = all(qrets_i.iloc[i] <= qrets_i.iloc[i+1] for i in range(len(qrets_i)-1))

                    print(f"\n  ILLIQUID (<$1M vol) ({len(illiquid):,} observations)")
                    print(f"    Correlation:   {corr_i:>+.4f}")
                    print(f"    Q5-Q1 Spread:  {spread_i:>+.2f}%")
                    print(f"    Monotonic:     {'YES' if mono_i else 'NO'}")

                    results['illiquid'] = {
                        'corr': corr_i, 'spread': spread_i, 'monotonic': mono_i, 'n': len(illiquid)
                    }
                except:
                    pass

        return results

    # Run analysis
    print("\n" + "=" * 70)
    print("IN-SAMPLE ANALYSIS (1995-2019)")
    print("=" * 70)
    is_results = analyze_dataset(is_df, "IN-SAMPLE (Training Period)")

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE ANALYSIS (2020-2026)")
    print("=" * 70)
    oos_results = analyze_dataset(oos_df, "OUT-OF-SAMPLE (Test Period)")

    # Summary comparison
    print("\n" + "=" * 70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("=" * 70)

    print("\nV4 Revised Performance:")
    print(f"{'Period':<25} {'N':>12} {'Correlation':>12} {'Spread':>12} {'Monotonic':>12}")
    print("-" * 75)

    for name, results in [('In-Sample (1995-2019)', is_results),
                          ('Out-of-Sample (2020-2026)', oos_results)]:
        if 'overall' in results:
            r = results['overall']
            print(f"{name:<25} {r['n']:>12,} {r['corr']:>+11.4f} {r['spread']:>+11.2f}% "
                  f"{'YES' if r['monotonic'] else 'NO':>12}")

    # Liquid segment comparison
    print("\nLiquid Segment (>$1M vol):")
    print(f"{'Period':<25} {'N':>12} {'Correlation':>12} {'Spread':>12} {'Monotonic':>12}")
    print("-" * 75)

    for name, results in [('In-Sample (1995-2019)', is_results),
                          ('Out-of-Sample (2020-2026)', oos_results)]:
        if 'liquid' in results:
            r = results['liquid']
            print(f"{name:<25} {r['n']:>12,} {r['corr']:>+11.4f} {r['spread']:>+11.2f}% "
                  f"{'YES' if r['monotonic'] else 'NO':>12}")

    # Degradation check
    print("\n" + "=" * 70)
    print("OVERFITTING CHECK")
    print("=" * 70)

    if 'overall' in is_results and 'overall' in oos_results:
        is_spread = is_results['overall']['spread']
        oos_spread = oos_results['overall']['spread']
        degradation = is_spread - oos_spread

        print(f"\n  In-Sample Spread:      {is_spread:>+.2f}%")
        print(f"  Out-of-Sample Spread:  {oos_spread:>+.2f}%")
        print(f"  Degradation:           {degradation:>+.2f}%")

        if oos_spread >= is_spread * 0.7:
            print(f"\n  PASS: OOS retains ≥70% of IS performance")
        elif oos_spread >= is_spread * 0.5:
            print(f"\n  CAUTION: OOS retains 50-70% of IS performance")
        elif oos_spread > 0:
            print(f"\n  WARNING: OOS retains <50% of IS performance (possible overfitting)")
        else:
            print(f"\n  FAIL: OOS spread is negative (signal doesn't generalize)")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    if 'overall' in oos_results:
        oos = oos_results['overall']
        if oos['spread'] > 5 and oos['monotonic']:
            print("\n  V4 PASSES out-of-sample validation")
            print(f"  - {oos['spread']:.2f}% quarterly spread in unseen 2020-2026 data")
            print(f"  - Monotonic quintile returns confirm ranking validity")
        elif oos['spread'] > 2:
            print("\n  V4 shows MODERATE out-of-sample performance")
            print(f"  - {oos['spread']:.2f}% quarterly spread (weaker than in-sample)")
        elif oos['spread'] > 0:
            print("\n  V4 shows WEAK out-of-sample performance")
            print(f"  - {oos['spread']:.2f}% quarterly spread (may be noise)")
        else:
            print("\n  V4 FAILS out-of-sample validation")
            print(f"  - Negative spread suggests no predictive power on new data")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

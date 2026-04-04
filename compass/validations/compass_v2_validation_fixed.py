#!/usr/bin/env python3
"""
Compass Score V2 Validation - FIXED
====================================
Uses EXACT methodology from v4_rigorous_validation.py that produced +6.79% OOS spread.

Key methodology (copied from original):
- Z-scores computed from IN-SAMPLE data (1995-2019)
- Monthly rebalancing (21 trading days)
- 63-day forward returns computed inline
- Price filter at $1
- 1st/99th percentile winsorization from IS data
- Strict data requirements (dropna)

Tests V2 modifications:
1. 3x penalty for negative ROA
2. V-shaped asset growth (penalize both shrinking and excessive growth)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration (matching v4_rigorous_validation.py EXACTLY)
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'

# V2 modification parameters
IDEAL_GROWTH = 0.05  # 5% annual growth is ideal for V-shaped penalty


def main():
    print("=" * 70)
    print("COMPASS V2 VALIDATION (FIXED METHODOLOGY)")
    print("Using exact methodology from v4_rigorous_validation.py")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nMethodology:")
    print(f"  In-Sample Period:  1995 - 2019 (train z-scores)")
    print(f"  Out-of-Sample:     2020 - 2026 (test)")
    print(f"  Rebalancing:       Monthly (21 trading days)")
    print(f"  Forward Returns:   63 trading days (~3 months)")

    conn = sqlite3.connect(BACKTEST_DB)

    # Load ALL data (same as v4_rigorous_validation.py)
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

    # Compute fundamental factors (same as v4)
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

    # STRICT data requirement
    required_factors = ['roa', 'ocf_assets', 'fcf_assets', 'asset_growth']
    fund = fund.dropna(subset=required_factors)
    print(f"  After strict filter: {len(fund):,} records with all required factors")

    # Compute price factors (same loop as v4)
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

        # Sample monthly
        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum 12-1
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility (trailing 60 days)
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Average daily dollar volume (trailing 60 days)
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

    # Merge fundamentals (backward-looking only)
    print("\nMerging fundamentals (backward-looking only)...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'market_cap']
    fund_clean = fund[fund_cols].copy()

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund_clean[fund_clean['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

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
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
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
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d', 'mom_12_1']:
        if col in is_df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
            factor_stats[col] = (mean, std)
            print(f"  {col}: mean={mean:.4f}, std={std:.4f}")

    # ==========================================
    # COMPUTE ALL 4 CONFIGURATIONS
    # ==========================================

    def compute_scores(dataset, triple_roa_penalty=False, vshaped_asset_growth=False):
        """Compute compass scores with optional V2 modifications."""
        data = dataset.copy()

        # Apply z-scores to dataset using IN-SAMPLE stats
        for col, (mean, std) in factor_stats.items():
            data[f'{col}_z'] = (data[col] - mean) / std
            data[f'{col}_z'] = data[f'{col}_z'].clip(-3, 3).fillna(0)

        # V2 Modification 1: Triple ROA penalty for negative ROA
        if triple_roa_penalty:
            mask = data['roa'] < 0
            data.loc[mask, 'roa_z'] = data.loc[mask, 'roa_z'] * 3
            # Re-clip after multiplying
            data['roa_z'] = data['roa_z'].clip(-3, 3)

        # V2 Modification 2: V-shaped asset growth
        if vshaped_asset_growth:
            # Penalize deviation from ideal growth (5%)
            mean, std = factor_stats['asset_growth']
            # Compute deviation from ideal (absolute difference from 5%)
            deviation = np.abs(data['asset_growth'] - IDEAL_GROWTH)
            # Z-score the deviation using in-sample stats for deviation
            # First compute in-sample deviation stats
            is_deviation = np.abs(is_df['asset_growth'] - IDEAL_GROWTH)
            dev_mean = is_deviation.mean()
            dev_std = is_deviation.std()
            # Apply to current data
            data['asset_growth_z'] = (deviation - dev_mean) / dev_std
            data['asset_growth_z'] = data['asset_growth_z'].clip(-3, 3).fillna(0)

        # Compute score
        data['score'] = (
            data['roa_z'] * 0.20 +
            data['ocf_assets_z'] * 0.15 +
            data['fcf_assets_z'] * 0.15 +
            data['gp_assets_z'] * 0.20 +
            (-data['vol_60d_z']) * 0.15 +
            (-data['asset_growth_z']) * 0.15
        )

        return data

    def analyze_config(data, name):
        """Analyze a configuration and return metrics."""
        clean = data.dropna(subset=['score', 'fwd_3m'])
        corr = clean['score'].corr(clean['fwd_3m'])

        try:
            clean['quintile'] = pd.qcut(clean['score'], 5,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                       duplicates='drop')
        except ValueError:
            return None

        q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        qrets = clean.groupby('quintile')['fwd_3m'].mean()
        is_monotonic = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))

        return {
            'name': name,
            'n': len(clean),
            'corr': corr,
            'spread': spread,
            'monotonic': is_monotonic,
            'q1': q1_ret,
            'q2': clean[clean['quintile'] == 'Q2']['fwd_3m'].mean(),
            'q3': clean[clean['quintile'] == 'Q3']['fwd_3m'].mean(),
            'q4': clean[clean['quintile'] == 'Q4']['fwd_3m'].mean(),
            'q5': q5_ret,
            'data': clean
        }

    # Define configurations
    configs = [
        ('A. V1 Baseline', False, False),
        ('B. +3x ROA penalty', True, False),
        ('C. +V-shape asset growth', False, True),
        ('D. V2 (both mods)', True, True),
    ]

    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE ANALYSIS (2020-2026)")
    print("=" * 70)

    oos_results = {}
    for name, triple_roa, vshaped in configs:
        scored = compute_scores(oos_df, triple_roa_penalty=triple_roa, vshaped_asset_growth=vshaped)
        result = analyze_config(scored, name)
        if result:
            oos_results[name] = result

    # Print results
    print(f"\n{'Configuration':<30} {'N':>10} {'Correlation':>12} {'Spread':>12} {'Monotonic':>10}")
    print("-" * 80)

    baseline_spread = oos_results.get('A. V1 Baseline', {}).get('spread', 0)

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in oos_results:
            r = oos_results[name]
            diff = r['spread'] - baseline_spread if name != 'A. V1 Baseline' else 0
            diff_str = f"({diff:+.2f}%)" if name != 'A. V1 Baseline' else ""
            print(f"{name:<30} {r['n']:>10,} {r['corr']:>+11.4f} {r['spread']:>+10.2f}% {diff_str:>8} "
                  f"{'YES' if r['monotonic'] else 'NO':>8}")

    # Quintile detail
    print("\n" + "=" * 70)
    print("QUINTILE RETURNS BY CONFIGURATION")
    print("=" * 70)
    print(f"  (Q5 = highest quality, Q1 = lowest quality)")

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in oos_results:
            r = oos_results[name]
            print(f"\n  {name}:")
            print(f"    Q5: {r['q5']:>+7.2f}%  Q4: {r['q4']:>+7.2f}%  Q3: {r['q3']:>+7.2f}%  "
                  f"Q2: {r['q2']:>+7.2f}%  Q1: {r['q1']:>+7.2f}%")
            print(f"    Spread (Q5-Q1): {r['spread']:>+.2f}%")

    # Year-by-year breakdown
    print("\n" + "=" * 70)
    print("YEAR-BY-YEAR ANALYSIS")
    print("=" * 70)

    for name in ['A. V1 Baseline', 'D. V2 (both mods)']:
        if name in oos_results:
            data = oos_results[name]['data'].copy()
            data['year'] = data['date'].dt.year

            print(f"\n  {name}:")
            print(f"  {'Year':<6} {'Q5':>10} {'Q4':>10} {'Q3':>10} {'Q2':>10} {'Q1':>10} {'Spread':>10}")
            print(f"  {'-'*66}")

            for year in sorted(data['year'].unique()):
                year_data = data[data['year'] == year].copy()
                if len(year_data) < 100:
                    continue

                try:
                    year_data['quintile'] = pd.qcut(year_data['score'], 5,
                                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                   duplicates='drop')
                except ValueError:
                    continue

                qrets = {}
                for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                    q_data = year_data[year_data['quintile'] == q]
                    if len(q_data) > 0:
                        qrets[q] = q_data['fwd_3m'].mean()

                if 'Q5' in qrets and 'Q1' in qrets:
                    spread = qrets['Q5'] - qrets['Q1']
                    print(f"  {year:<6} {qrets.get('Q5', 0):>+9.2f}% {qrets.get('Q4', 0):>+9.2f}% "
                          f"{qrets.get('Q3', 0):>+9.2f}% {qrets.get('Q2', 0):>+9.2f}% "
                          f"{qrets.get('Q1', 0):>+9.2f}% {spread:>+9.2f}%")

    # ==========================================
    # IN-SAMPLE COMPARISON (sanity check)
    # ==========================================
    print("\n" + "=" * 70)
    print("IN-SAMPLE VERIFICATION (1995-2019)")
    print("=" * 70)

    is_results = {}
    for name, triple_roa, vshaped in configs:
        scored = compute_scores(is_df, triple_roa_penalty=triple_roa, vshaped_asset_growth=vshaped)
        result = analyze_config(scored, name)
        if result:
            is_results[name] = result

    print(f"\n{'Configuration':<30} {'N':>10} {'Correlation':>12} {'Spread':>12} {'Monotonic':>10}")
    print("-" * 80)

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in is_results:
            r = is_results[name]
            print(f"{name:<30} {r['n']:>10,} {r['corr']:>+11.4f} {r['spread']:>+10.2f}% "
                  f"{'YES' if r['monotonic'] else 'NO':>8}")

    # ==========================================
    # COMPARISON SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
    print("=" * 70)

    print(f"\n{'Configuration':<30} {'IS Spread':>12} {'OOS Spread':>12} {'Degradation':>12}")
    print("-" * 70)

    for name in ['A. V1 Baseline', 'B. +3x ROA penalty', 'C. +V-shape asset growth', 'D. V2 (both mods)']:
        if name in is_results and name in oos_results:
            is_spread = is_results[name]['spread']
            oos_spread = oos_results[name]['spread']
            degradation = is_spread - oos_spread
            print(f"{name:<30} {is_spread:>+10.2f}% {oos_spread:>+10.2f}% {degradation:>+10.2f}%")

    # Final verdict
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if 'A. V1 Baseline' in oos_results:
        v1_oos = oos_results['A. V1 Baseline']['spread']
        print(f"\n  V1 Baseline OOS Spread: {v1_oos:+.2f}%")

        if v1_oos > 5:
            print(f"  CONFIRMED: Original methodology produces positive OOS spread!")
        elif v1_oos > 0:
            print(f"  Positive but weaker than expected ({v1_oos:.2f}% vs +6.79% in paper)")
        else:
            print(f"  WARNING: Negative spread - investigate methodology differences")

    if 'D. V2 (both mods)' in oos_results:
        v2_oos = oos_results['D. V2 (both mods)']['spread']
        v1_oos = oos_results.get('A. V1 Baseline', {}).get('spread', 0)
        diff = v2_oos - v1_oos

        print(f"\n  V2 Modifications Impact: {diff:+.2f}%")
        if diff > 0.5:
            print(f"  IMPROVEMENT: V2 outperforms V1 baseline")
        elif diff > -0.5:
            print(f"  NEUTRAL: V2 has similar performance to V1")
        else:
            print(f"  DEGRADATION: V2 underperforms V1 baseline")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

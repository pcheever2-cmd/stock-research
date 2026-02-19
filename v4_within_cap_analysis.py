#!/usr/bin/env python3
"""
V4 Within-Cap Segment Normalization
Z-scores computed within each market cap bucket to improve large-cap performance
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

# Market cap segments
CAP_SEGMENTS = {
    'Micro': (0, 300_000_000),
    'Small': (300_000_000, 2_000_000_000),
    'Mid': (2_000_000_000, 10_000_000_000),
    'Large': (10_000_000_000, float('inf')),
}


def main():
    print("=" * 70)
    print("V4 WITHIN-CAP SEGMENT ANALYSIS")
    print("Z-scores normalized within each market cap bucket")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    conn = sqlite3.connect(BACKTEST_DB)

    # Load data
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
    print(f"  {len(prices):,} price records")

    print("\nLoading fundamentals with market cap...")
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        WHERE m.market_cap IS NOT NULL AND m.market_cap > 0
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records with market cap")

    conn.close()

    # Compute fundamental factors
    print("\n" + "=" * 70)
    print("COMPUTING FACTORS")
    print("=" * 70)

    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Assign market cap segment
    def get_cap_segment(mc):
        for name, (low, high) in CAP_SEGMENTS.items():
            if low <= mc < high:
                return name
        return 'Large'

    fund['cap_segment'] = fund['market_cap'].apply(get_cap_segment)

    print("\nMarket cap distribution:")
    print(fund['cap_segment'].value_counts())

    # Winsorize factors
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
        lower = fund[col].quantile(0.01)
        upper = fund[col].quantile(0.99)
        fund[col] = fund[col].clip(lower, upper)

    # Sample symbols
    print("\nSampling 2500 symbols...")
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(2500, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    # Compute price-based factors and forward returns
    print("Computing price factors...")
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

        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Avg dollar volume
            avg_dollar_vol = np.mean(volume[j-20:j] * close[j-20:j]) if j >= 20 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'vol_60d': vol_60d,
                'avg_dollar_vol': avg_dollar_vol,
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    print("\nMerging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                 'asset_growth', 'market_cap', 'cap_segment']
    fund_subset = fund[fund_cols].dropna(subset=['roa', 'market_cap'])

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund_subset[fund_subset['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)
    df = df.dropna(subset=['fwd_3m', 'roa', 'cap_segment'])
    print(f"  Clean observations: {len(df):,}")

    print("\nObservations by cap segment:")
    print(df['cap_segment'].value_counts())

    # ==========================================
    # METHOD 1: Global Z-scores (current V4)
    # ==========================================
    print("\n" + "=" * 70)
    print("METHOD 1: GLOBAL Z-SCORES (Current V4)")
    print("=" * 70)

    df_global = df.copy()
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
        df_global[f'{col}_z'] = (df_global[col] - df_global[col].mean()) / df_global[col].std()
        df_global[f'{col}_z'] = df_global[f'{col}_z'].clip(-3, 3).fillna(0)

    df_global['v4_global'] = (
        df_global['roa_z'] * 0.20 +
        df_global['ocf_assets_z'] * 0.15 +
        df_global['fcf_assets_z'] * 0.15 +
        df_global['gp_assets_z'] * 0.10 +
        (-df_global['vol_60d_z']) * 0.15 +
        (-df_global['asset_growth_z']) * 0.15
    )

    # Analyze by cap segment
    print("\nGlobal Z-score Results by Cap Segment:")
    print(f"{'Segment':<10} {'Obs':>10} {'Corr':>10} {'Q5 Ret':>10} {'Q1 Ret':>10} {'Spread':>10}")
    print("-" * 65)

    global_results = {}
    for segment in ['Large', 'Mid', 'Small', 'Micro']:
        seg_df = df_global[df_global['cap_segment'] == segment].copy()
        if len(seg_df) < 500:
            continue

        corr = seg_df['v4_global'].corr(seg_df['fwd_3m'])
        seg_df['quintile'] = pd.qcut(seg_df['v4_global'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        q5_ret = seg_df[seg_df['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = seg_df[seg_df['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        global_results[segment] = spread
        print(f"{segment:<10} {len(seg_df):>10,} {corr:>+10.4f} {q5_ret:>+10.2f}% {q1_ret:>+10.2f}% {spread:>+10.2f}%")

    # ==========================================
    # METHOD 2: Within-Cap Z-scores
    # ==========================================
    print("\n" + "=" * 70)
    print("METHOD 2: WITHIN-CAP Z-SCORES")
    print("Z-scores normalized within each market cap segment")
    print("=" * 70)

    df_within = df.copy()

    # Z-score within each cap segment
    def zscore_within_group(group, cols):
        for col in cols:
            group[f'{col}_z'] = (group[col] - group[col].mean()) / group[col].std()
            group[f'{col}_z'] = group[f'{col}_z'].clip(-3, 3).fillna(0)
        return group

    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']
    df_within = df_within.groupby('cap_segment', group_keys=False).apply(
        lambda g: zscore_within_group(g, factor_cols)
    )

    df_within['v4_within'] = (
        df_within['roa_z'] * 0.20 +
        df_within['ocf_assets_z'] * 0.15 +
        df_within['fcf_assets_z'] * 0.15 +
        df_within['gp_assets_z'] * 0.10 +
        (-df_within['vol_60d_z']) * 0.15 +
        (-df_within['asset_growth_z']) * 0.15
    )

    # Analyze by cap segment
    print("\nWithin-Cap Z-score Results by Cap Segment:")
    print(f"{'Segment':<10} {'Obs':>10} {'Corr':>10} {'Q5 Ret':>10} {'Q1 Ret':>10} {'Spread':>10} {'vs Global':>12}")
    print("-" * 80)

    within_results = {}
    for segment in ['Large', 'Mid', 'Small', 'Micro']:
        seg_df = df_within[df_within['cap_segment'] == segment].copy()
        if len(seg_df) < 500:
            continue

        corr = seg_df['v4_within'].corr(seg_df['fwd_3m'])
        seg_df['quintile'] = pd.qcut(seg_df['v4_within'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        q5_ret = seg_df[seg_df['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = seg_df[seg_df['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        within_results[segment] = spread
        improvement = spread - global_results.get(segment, 0)
        print(f"{segment:<10} {len(seg_df):>10,} {corr:>+10.4f} {q5_ret:>+10.2f}% {q1_ret:>+10.2f}% {spread:>+10.2f}% {improvement:>+12.2f}%")

    # ==========================================
    # DETAILED QUINTILE ANALYSIS FOR LARGE-CAP
    # ==========================================
    print("\n" + "=" * 70)
    print("LARGE-CAP QUINTILE DETAIL")
    print("=" * 70)

    large_global = df_global[df_global['cap_segment'] == 'Large'].copy()
    large_within = df_within[df_within['cap_segment'] == 'Large'].copy()

    if len(large_global) > 1000:
        large_global['quintile'] = pd.qcut(large_global['v4_global'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        large_within['quintile'] = pd.qcut(large_within['v4_within'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        print("\nGlobal Z-scores (Large Cap):")
        print(f"{'Quintile':<12} {'Avg Return':>12} {'Median':>10} {'Count':>10}")
        print("-" * 45)
        for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
            q_data = large_global[large_global['quintile'] == q]['fwd_3m']
            label = f"{q} ({'Best' if q == 'Q5' else 'Worst' if q == 'Q1' else ''})"
            print(f"{label:<12} {q_data.mean():>+10.2f}% {q_data.median():>+10.2f}% {len(q_data):>10,}")

        print("\nWithin-Cap Z-scores (Large Cap):")
        print(f"{'Quintile':<12} {'Avg Return':>12} {'Median':>10} {'Count':>10}")
        print("-" * 45)
        for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
            q_data = large_within[large_within['quintile'] == q]['fwd_3m']
            label = f"{q} ({'Best' if q == 'Q5' else 'Worst' if q == 'Q1' else ''})"
            print(f"{label:<12} {q_data.mean():>+10.2f}% {q_data.median():>+10.2f}% {len(q_data):>10,}")

    # ==========================================
    # LIQUID STOCKS ANALYSIS
    # ==========================================
    print("\n" + "=" * 70)
    print("LIQUID STOCKS (>$1M daily volume)")
    print("=" * 70)

    liquid = df_within[df_within['avg_dollar_vol'] > 1_000_000].copy()
    print(f"Liquid observations: {len(liquid):,}")

    print("\nWithin-Cap Z-scores (Liquid Only):")
    print(f"{'Segment':<10} {'Obs':>10} {'Corr':>10} {'Spread':>10}")
    print("-" * 45)

    for segment in ['Large', 'Mid', 'Small', 'Micro']:
        seg_df = liquid[liquid['cap_segment'] == segment].copy()
        if len(seg_df) < 200:
            continue

        corr = seg_df['v4_within'].corr(seg_df['fwd_3m'])
        seg_df['quintile'] = pd.qcut(seg_df['v4_within'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

        q5_ret = seg_df[seg_df['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = seg_df[seg_df['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        print(f"{segment:<10} {len(seg_df):>10,} {corr:>+10.4f} {spread:>+10.2f}%")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY: GLOBAL vs WITHIN-CAP Z-SCORES")
    print("=" * 70)

    print(f"""
COMPARISON BY SEGMENT:

{'Segment':<10} {'Global Z':>12} {'Within-Cap Z':>14} {'Improvement':>14}
{'-' * 55}""")

    for segment in ['Large', 'Mid', 'Small', 'Micro']:
        g = global_results.get(segment, 0)
        w = within_results.get(segment, 0)
        imp = w - g
        print(f"{segment:<10} {g:>+10.2f}% {w:>+12.2f}% {imp:>+12.2f}%")

    print(f"""
FINDINGS:

1. Within-cap z-scoring {'IMPROVES' if sum(within_results.values()) > sum(global_results.values()) else 'does not improve'} overall performance.

2. Large-cap improvement: {within_results.get('Large', 0) - global_results.get('Large', 0):+.2f}%
   - Global: {global_results.get('Large', 0):+.2f}%
   - Within-cap: {within_results.get('Large', 0):+.2f}%

3. The within-cap approach normalizes relative to similar-sized peers,
   potentially capturing different quality dynamics at each size tier.
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

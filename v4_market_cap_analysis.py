#!/usr/bin/env python3
"""
V4 Market Cap Segmentation Analysis
Tests if V4 formula works differently across market cap segments
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
    print("V4 MARKET CAP SEGMENTATION ANALYSIS")
    print("Full Universe - All Data Points")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    conn = sqlite3.connect(BACKTEST_DB)

    # Load ALL data
    print("\n" + "=" * 70)
    print("LOADING FULL UNIVERSE")
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

    print("\nLoading fundamentals with market cap...")
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
    print("COMPUTING FACTORS (FULL UNIVERSE)")
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

    # Compute price factors for ALL symbols
    print("\nComputing price factors (ALL symbols)...")
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

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Average daily dollar volume
            avg_dollar_vol = np.mean(volume[j-20:j] * close[j-20:j]) if j >= 20 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'avg_dollar_vol': avg_dollar_vol,
                'price': close[j],
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    print("\nMerging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                 'fcf_yield', 'asset_growth', 'market_cap']
    fund_clean = fund[fund_cols].dropna(subset=['roa'])

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
    df = df.dropna(subset=['fwd_3m', 'roa', 'market_cap'])
    print(f"\nClean observations with market cap: {len(df):,}")

    # Define market cap segments
    print("\n" + "=" * 70)
    print("MARKET CAP SEGMENTATION")
    print("=" * 70)

    # Calculate market cap percentiles
    mc_p25 = df['market_cap'].quantile(0.25)
    mc_p50 = df['market_cap'].quantile(0.50)
    mc_p75 = df['market_cap'].quantile(0.75)
    mc_p90 = df['market_cap'].quantile(0.90)

    print(f"\nMarket Cap Distribution (in millions):")
    print(f"  25th percentile: ${mc_p25/1e6:,.0f}M")
    print(f"  50th percentile: ${mc_p50/1e6:,.0f}M")
    print(f"  75th percentile: ${mc_p75/1e6:,.0f}M")
    print(f"  90th percentile: ${mc_p90/1e6:,.0f}M")

    # Define segments
    def classify_market_cap(mc):
        if mc >= 10e9:  # $10B+
            return 'Large Cap'
        elif mc >= 2e9:  # $2B-$10B
            return 'Mid Cap'
        elif mc >= 300e6:  # $300M-$2B
            return 'Small Cap'
        else:  # <$300M
            return 'Micro Cap'

    df['market_cap_segment'] = df['market_cap'].apply(classify_market_cap)

    print("\nSegment Distribution:")
    for seg in ['Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']:
        count = len(df[df['market_cap_segment'] == seg])
        pct = count / len(df) * 100
        print(f"  {seg}: {count:,} ({pct:.1f}%)")

    # Z-score factors WITHIN each market cap segment
    print("\n" + "=" * 70)
    print("COMPUTING SEGMENT-SPECIFIC Z-SCORES")
    print("=" * 70)

    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield',
                   'asset_growth', 'vol_60d', 'mom_12_1']

    # Global z-scores (for comparison)
    for col in factor_cols:
        if col in df.columns:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_z'] = df[f'{col}_z'].clip(-3, 3).fillna(0)

    # Segment-specific z-scores
    for seg in df['market_cap_segment'].unique():
        seg_mask = df['market_cap_segment'] == seg
        for col in factor_cols:
            if col in df.columns:
                seg_data = df.loc[seg_mask, col]
                df.loc[seg_mask, f'{col}_seg_z'] = (seg_data - seg_data.mean()) / seg_data.std()
                df[f'{col}_seg_z'] = df[f'{col}_seg_z'].clip(-3, 3).fillna(0)

    # V4 Revised Score (using global z-scores)
    df['v4_revised'] = (
        df['roa_z'] * 0.20 +
        df['ocf_assets_z'] * 0.15 +
        df['fcf_assets_z'] * 0.15 +
        df['gp_assets_z'] * 0.10 +
        (-df['vol_60d_z']) * 0.15 +
        (-df['asset_growth_z']) * 0.15
    )

    # V4 Segment-Normalized (using segment z-scores)
    df['v4_segment'] = (
        df['roa_seg_z'] * 0.20 +
        df['ocf_assets_seg_z'] * 0.15 +
        df['fcf_assets_seg_z'] * 0.15 +
        df['gp_assets_seg_z'] * 0.10 +
        (-df['vol_60d_seg_z']) * 0.15 +
        (-df['asset_growth_seg_z']) * 0.15
    )

    # Analyze each segment
    print("\n" + "=" * 70)
    print("V4 PERFORMANCE BY MARKET CAP SEGMENT")
    print("=" * 70)

    segments = ['Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']

    segment_results = []

    for seg in segments:
        seg_df = df[df['market_cap_segment'] == seg].copy()

        if len(seg_df) < 1000:
            print(f"\n{seg}: Insufficient data ({len(seg_df)} obs)")
            continue

        print(f"\n{'='*60}")
        print(f"{seg.upper()} ({len(seg_df):,} observations)")
        print(f"{'='*60}")

        for score_name, score_col in [('V4 Revised (Global Z)', 'v4_revised'),
                                       ('V4 Segment-Normalized', 'v4_segment')]:
            clean = seg_df.dropna(subset=[score_col, 'fwd_3m'])

            if len(clean) < 500:
                continue

            corr = clean[score_col].corr(clean['fwd_3m'])

            # Quintiles
            try:
                clean['quintile'] = pd.qcut(clean[score_col], 5,
                                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                           duplicates='drop')
            except ValueError:
                continue

            q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
            q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
            spread = q5_ret - q1_ret

            # Check monotonicity
            qrets = clean.groupby('quintile')['fwd_3m'].mean()
            is_monotonic = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))

            print(f"\n  {score_name}:")
            print(f"    Correlation:   {corr:>+.4f}")
            print(f"    Q5 Return:     {q5_ret:>+.2f}%")
            print(f"    Q1 Return:     {q1_ret:>+.2f}%")
            print(f"    Q5-Q1 Spread:  {spread:>+.2f}%")
            print(f"    Monotonic:     {'YES' if is_monotonic else 'NO'}")

            segment_results.append({
                'segment': seg,
                'score': score_name,
                'correlation': corr,
                'spread': spread,
                'monotonic': is_monotonic,
                'n_obs': len(clean),
            })

        # Quintile detail for V4 Segment
        print(f"\n  Quintile Detail (V4 Segment-Normalized):")
        clean = seg_df.dropna(subset=['v4_segment', 'fwd_3m'])
        try:
            clean['quintile'] = pd.qcut(clean['v4_segment'], 5,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                       duplicates='drop')
        except:
            continue

        print(f"  {'Quintile':<10} {'Avg Ret':>10} {'Median':>10} {'Std':>10} {'Count':>10}")
        print(f"  {'-'*50}")
        for q in ['Q5', 'Q4', 'Q3', 'Q2', 'Q1']:
            q_data = clean[clean['quintile'] == q]['fwd_3m']
            print(f"  {q:<10} {q_data.mean():>+9.2f}% {q_data.median():>+9.2f}% {q_data.std():>9.1f} {len(q_data):>10,}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: V4 PERFORMANCE BY SEGMENT")
    print("=" * 70)

    if segment_results:
        results_df = pd.DataFrame(segment_results)

        print("\nV4 Segment-Normalized Results:")
        print(f"{'Segment':<15} {'Correlation':>12} {'Spread':>12} {'Monotonic':>12} {'N':>12}")
        print("-" * 65)

        for seg in segments:
            seg_data = results_df[(results_df['segment'] == seg) &
                                   (results_df['score'] == 'V4 Segment-Normalized')]
            if len(seg_data) > 0:
                row = seg_data.iloc[0]
                print(f"{seg:<15} {row['correlation']:>+11.4f} {row['spread']:>+11.2f}% "
                      f"{'YES' if row['monotonic'] else 'NO':>12} {row['n_obs']:>12,}")

    # Liquidity filter analysis
    print("\n" + "=" * 70)
    print("LIQUIDITY FILTER ANALYSIS")
    print("=" * 70)

    liquidity_thresholds = [100_000, 500_000, 1_000_000, 5_000_000]

    print("\nV4 Segment-Normalized with Liquidity Filters:")
    print(f"{'Min Daily Vol':>15} {'N Obs':>12} {'Correlation':>12} {'Spread':>12} {'Monotonic':>12}")
    print("-" * 65)

    for threshold in liquidity_thresholds:
        liquid_df = df[df['avg_dollar_vol'] >= threshold].copy()

        if len(liquid_df) < 1000:
            print(f"${threshold/1e6:.1f}M+{' ':>10} {len(liquid_df):>12,} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue

        clean = liquid_df.dropna(subset=['v4_segment', 'fwd_3m'])
        corr = clean['v4_segment'].corr(clean['fwd_3m'])

        try:
            clean['quintile'] = pd.qcut(clean['v4_segment'], 5,
                                       labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                       duplicates='drop')
        except:
            continue

        q5_ret = clean[clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = clean[clean['quintile'] == 'Q1']['fwd_3m'].mean()
        spread = q5_ret - q1_ret

        qrets = clean.groupby('quintile')['fwd_3m'].mean()
        is_monotonic = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))

        label = f"${threshold/1e6:.1f}M+"
        print(f"{label:>15} {len(clean):>12,} {corr:>+11.4f} {spread:>+11.2f}% "
              f"{'YES' if is_monotonic else 'NO':>12}")

    # Final recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    print("""
Based on this analysis:

1. SEGMENT-SPECIFIC SCORING: Normalize factors WITHIN each market cap
   segment rather than across the full universe. This prevents large
   caps from dominating the z-score distribution.

2. LIQUIDITY FILTER: Apply a minimum daily volume filter (suggest $1M+)
   to focus on tradeable stocks where the signal is more reliable.

3. SEPARATE MODELS: Consider maintaining separate models/weights for:
   - Large Cap ($10B+): May need different factor weights
   - Mid Cap ($2B-$10B): Core universe, most signal
   - Small/Micro Cap (<$2B): Higher noise, consider excluding or
     using stricter filters

4. V4 REVISED FORMULA (for implementation):

   # Normalize within market cap segment first
   segment_z_score = (factor - segment_mean) / segment_std

   v4_score = (
       roa_z * 0.20 +
       ocf_assets_z * 0.15 +
       fcf_assets_z * 0.15 +
       gp_assets_z * 0.10 +
       (-vol_60d_z) * 0.15 +
       (-asset_growth_z) * 0.15
   )
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

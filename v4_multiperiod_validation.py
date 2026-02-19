#!/usr/bin/env python3
"""
V4 Multi-Period Rolling Validation
Tests V4 signal across multiple non-overlapping time periods
to ensure robustness across different market regimes.

Methodology:
- Multiple train/test splits across different decades
- Each period trains on 5 years, tests on next 5 years
- No overlap between training and test data
- Aggregates results to show consistency
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

# Define validation periods (train_start, train_end, test_start, test_end)
# Fundamentals only available from 2007, so start from 2008
VALIDATION_PERIODS = [
    ('Period 1: GFC Era', '2008-01-01', '2011-12-31', '2012-01-01', '2014-12-31'),
    ('Period 2: Recovery', '2011-01-01', '2014-12-31', '2015-01-01', '2017-12-31'),
    ('Period 3: Pre-COVID', '2014-01-01', '2017-12-31', '2018-01-01', '2020-03-31'),
    ('Period 4: COVID & After', '2017-01-01', '2020-03-31', '2020-04-01', '2022-12-31'),
    ('Period 5: Recent', '2020-01-01', '2022-12-31', '2023-01-01', '2025-12-31'),
]

LIQUIDITY_THRESHOLD = 1_000_000  # $1M daily volume


def load_and_prepare_data():
    """Load all data and compute factors once."""
    print("=" * 70)
    print("LOADING AND PREPARING DATA")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

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

    # Clean infinities
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Sample symbols for speed (2000 symbols)
    print("\nSampling 2000 symbols for computational efficiency...")
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(2000, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    # Compute price factors
    print(f"\nComputing price factors ({len(sample_symbols)} symbols)...")
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

            # Average daily dollar volume (trailing 60 days)
            avg_dollar_vol = np.mean(volume[j-60:j] * close[j-60:j]) if j >= 60 else np.nan

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
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets',
                 'fcf_yield', 'asset_growth']
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
    df = df.dropna(subset=['fwd_3m', 'roa', 'ocf_assets', 'fcf_assets', 'asset_growth'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    print(f"\nTotal clean observations: {len(df):,}")

    return df


def validate_period(df, train_start, train_end, test_start, test_end, period_name):
    """
    Validate V4 on a specific train/test period.

    - Computes factor statistics from TRAINING data only
    - Applies to TEST data
    - Returns test performance metrics
    """
    # Split data
    train_df = df[(df['date_str'] >= train_start) & (df['date_str'] <= train_end)].copy()
    test_df = df[(df['date_str'] >= test_start) & (df['date_str'] <= test_end)].copy()

    if len(train_df) < 1000 or len(test_df) < 500:
        return None

    # Compute winsorization bounds from TRAINING data
    factor_bounds = {}
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth', 'vol_60d']:
        if col in train_df.columns:
            lower = train_df[col].quantile(0.01)
            upper = train_df[col].quantile(0.99)
            factor_bounds[col] = (lower, upper)

    # Apply bounds
    for col, (lower, upper) in factor_bounds.items():
        train_df[col] = train_df[col].clip(lower, upper)
        test_df[col] = test_df[col].clip(lower, upper)

    # Compute z-scores using TRAINING statistics
    factor_stats = {}
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield', 'asset_growth', 'vol_60d', 'mom_12_1']:
        if col in train_df.columns:
            mean = train_df[col].mean()
            std = train_df[col].std()
            if std > 0:
                factor_stats[col] = (mean, std)

    # Apply z-scores to BOTH datasets
    for col, (mean, std) in factor_stats.items():
        for dataset in [train_df, test_df]:
            dataset[f'{col}_z'] = (dataset[col] - mean) / std
            dataset[f'{col}_z'] = dataset[f'{col}_z'].clip(-3, 3).fillna(0)

    # V4 Revised Score
    for dataset in [train_df, test_df]:
        dataset['v4_score'] = (
            dataset['roa_z'] * 0.20 +
            dataset['ocf_assets_z'] * 0.15 +
            dataset['fcf_assets_z'] * 0.15 +
            dataset['gp_assets_z'] * 0.10 +
            (-dataset['vol_60d_z']) * 0.15 +
            (-dataset['asset_growth_z']) * 0.15
        )

    # Liquidity filter
    test_df['is_liquid'] = test_df['avg_dollar_vol'] >= LIQUIDITY_THRESHOLD

    # Calculate metrics for TEST data
    results = {'period': period_name, 'train_n': len(train_df), 'test_n': len(test_df)}

    # Overall test performance
    test_clean = test_df.dropna(subset=['v4_score', 'fwd_3m'])
    results['overall_corr'] = test_clean['v4_score'].corr(test_clean['fwd_3m'])

    try:
        test_clean['quintile'] = pd.qcut(test_clean['v4_score'], 5,
                                         labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                         duplicates='drop')
        q5_ret = test_clean[test_clean['quintile'] == 'Q5']['fwd_3m'].mean()
        q1_ret = test_clean[test_clean['quintile'] == 'Q1']['fwd_3m'].mean()
        results['overall_spread'] = q5_ret - q1_ret
        results['overall_q5'] = q5_ret
        results['overall_q1'] = q1_ret

        qrets = test_clean.groupby('quintile')['fwd_3m'].mean()
        results['overall_monotonic'] = all(qrets.iloc[i] <= qrets.iloc[i+1] for i in range(len(qrets)-1))
    except:
        results['overall_spread'] = np.nan
        results['overall_monotonic'] = False

    # Liquid stocks only
    liquid_test = test_df[test_df['is_liquid'] == True].dropna(subset=['v4_score', 'fwd_3m'])
    if len(liquid_test) >= 200:
        results['liquid_n'] = len(liquid_test)
        results['liquid_corr'] = liquid_test['v4_score'].corr(liquid_test['fwd_3m'])

        try:
            liquid_test['quintile'] = pd.qcut(liquid_test['v4_score'], 5,
                                             labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                             duplicates='drop')
            q5_l = liquid_test[liquid_test['quintile'] == 'Q5']['fwd_3m'].mean()
            q1_l = liquid_test[liquid_test['quintile'] == 'Q1']['fwd_3m'].mean()
            results['liquid_spread'] = q5_l - q1_l

            qrets_l = liquid_test.groupby('quintile')['fwd_3m'].mean()
            results['liquid_monotonic'] = all(qrets_l.iloc[i] <= qrets_l.iloc[i+1] for i in range(len(qrets_l)-1))
        except:
            results['liquid_spread'] = np.nan
            results['liquid_monotonic'] = False

    return results


def main():
    print("=" * 70)
    print("V4 MULTI-PERIOD ROLLING VALIDATION")
    print("Testing consistency across different market regimes")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load data once
    df = load_and_prepare_data()

    # Run validation for each period
    print("\n" + "=" * 70)
    print("RUNNING PERIOD-BY-PERIOD VALIDATION")
    print("=" * 70)

    all_results = []

    for period_name, train_start, train_end, test_start, test_end in VALIDATION_PERIODS:
        print(f"\n--- {period_name} ---")
        print(f"    Train: {train_start} to {train_end}")
        print(f"    Test:  {test_start} to {test_end}")

        result = validate_period(df, train_start, train_end, test_start, test_end, period_name)

        if result:
            all_results.append(result)
            print(f"\n    Training obs:  {result['train_n']:,}")
            print(f"    Test obs:      {result['test_n']:,}")
            print(f"    Correlation:   {result['overall_corr']:+.4f}")
            print(f"    Q5-Q1 Spread:  {result['overall_spread']:+.2f}%")
            print(f"    Monotonic:     {'YES' if result['overall_monotonic'] else 'NO'}")

            if 'liquid_spread' in result:
                print(f"\n    Liquid Only ({result['liquid_n']:,} obs):")
                print(f"      Correlation: {result['liquid_corr']:+.4f}")
                print(f"      Spread:      {result['liquid_spread']:+.2f}%")
        else:
            print(f"    Insufficient data for this period")

    # Summary
    print("\n" + "=" * 70)
    print("MULTI-PERIOD SUMMARY")
    print("=" * 70)

    if all_results:
        results_df = pd.DataFrame(all_results)

        print("\nOverall Performance Across Periods:")
        print(f"{'Period':<40} {'Train N':>10} {'Test N':>10} {'Corr':>10} {'Spread':>10} {'Mono':>8}")
        print("-" * 90)

        for _, r in results_df.iterrows():
            mono = 'YES' if r.get('overall_monotonic', False) else 'NO'
            print(f"{r['period']:<40} {r['train_n']:>10,} {r['test_n']:>10,} "
                  f"{r['overall_corr']:>+9.4f} {r['overall_spread']:>+9.2f}% {mono:>8}")

        # Averages
        avg_corr = results_df['overall_corr'].mean()
        avg_spread = results_df['overall_spread'].mean()
        pct_positive = (results_df['overall_spread'] > 0).mean() * 100
        pct_monotonic = results_df['overall_monotonic'].mean() * 100

        print("-" * 90)
        print(f"{'AVERAGE':<40} {'':<10} {'':<10} {avg_corr:>+9.4f} {avg_spread:>+9.2f}%")

        print(f"\nConsistency Metrics:")
        print(f"  Periods with positive spread: {pct_positive:.0f}%")
        print(f"  Periods with monotonic quintiles: {pct_monotonic:.0f}%")
        print(f"  Average spread: {avg_spread:+.2f}% per quarter")

        # Liquid stocks
        if 'liquid_spread' in results_df.columns:
            liquid_results = results_df.dropna(subset=['liquid_spread'])
            if len(liquid_results) > 0:
                print("\nLiquid Stocks (>$1M vol) Across Periods:")
                print(f"{'Period':<40} {'N':>10} {'Corr':>10} {'Spread':>10}")
                print("-" * 75)

                for _, r in liquid_results.iterrows():
                    print(f"{r['period']:<40} {r['liquid_n']:>10,} "
                          f"{r['liquid_corr']:>+9.4f} {r['liquid_spread']:>+9.2f}%")

                avg_liquid_spread = liquid_results['liquid_spread'].mean()
                print("-" * 75)
                print(f"{'AVERAGE':<40} {'':<10} {'':<10} {avg_liquid_spread:>+9.2f}%")

        # Final verdict
        print("\n" + "=" * 70)
        print("FINAL VERDICT")
        print("=" * 70)

        if pct_positive >= 75 and avg_spread > 3:
            print(f"\n  V4 PASSES multi-period validation")
            print(f"  - Positive spread in {pct_positive:.0f}% of test periods")
            print(f"  - Average {avg_spread:.2f}% quarterly spread across regimes")
        elif pct_positive >= 50 and avg_spread > 1:
            print(f"\n  V4 shows MODERATE consistency")
            print(f"  - Positive spread in {pct_positive:.0f}% of test periods")
            print(f"  - Signal weakens in some market regimes")
        else:
            print(f"\n  V4 shows WEAK consistency")
            print(f"  - Only positive in {pct_positive:.0f}% of periods")
            print(f"  - May be overfitting to specific regimes")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

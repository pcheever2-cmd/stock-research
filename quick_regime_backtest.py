#!/usr/bin/env python3
"""
Quick Regime Backtest - Clean version
Tests static vs adaptive V4 scoring
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
    print("QUICK REGIME BACKTEST: Static V4 vs Adaptive V4")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load price data with computed factors
    print("\nLoading data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Load SPY for regime
    spy = pd.read_sql_query("""
        SELECT date, adjusted_close as spy_close
        FROM historical_prices
        WHERE symbol = 'SPY'
        ORDER BY date
    """, conn)
    spy['date'] = pd.to_datetime(spy['date'])

    # Load fundamentals
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

    conn.close()

    # Compute SPY regime
    print("\nComputing SPY regime...")
    spy = spy.sort_values('date').reset_index(drop=True)
    spy['sma_200'] = spy['spy_close'].rolling(200).mean()
    spy['mom_6m'] = spy['spy_close'].pct_change(126)
    spy['returns'] = spy['spy_close'].pct_change()
    spy['vol_20d'] = spy['returns'].rolling(20).std() * np.sqrt(252) * 100

    def classify(row):
        if pd.isna(row['sma_200']) or pd.isna(row['vol_20d']):
            return 'Unknown'
        high_vol = row['vol_20d'] > 25
        above_sma = row['spy_close'] > row['sma_200']
        pos_mom = row['mom_6m'] > 0 if not pd.isna(row['mom_6m']) else False

        if high_vol:
            return 'High_Vol'
        elif above_sma and pos_mom:
            return 'Bull'
        elif not above_sma and not pos_mom:
            return 'Bear'
        else:
            return 'Transition'

    spy['regime'] = spy.apply(classify, axis=1)

    print("\nRegime Distribution:")
    for regime, count in spy['regime'].value_counts().items():
        print(f"  {regime}: {count:,} days ({count/len(spy)*100:.1f}%)")

    # Compute derived factors
    print("\nComputing factors...")
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
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)
        # Winsorize
        lower = fund[col].quantile(0.01)
        upper = fund[col].quantile(0.99)
        fund[col] = fund[col].clip(lower, upper)

    # Compute price factors per symbol
    print("\nComputing price factors (sampling 1000 symbols)...")

    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(1000, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    results = []
    for i, symbol in enumerate(sample_symbols):
        if i % 200 == 0:
            print(f"  {i}/{len(sample_symbols)}...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values

        # Sample monthly, get forward returns
        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
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

    # Merge regime
    df = pd.merge_asof(df.sort_values('date'),
                       spy[['date', 'regime']].sort_values('date'),
                       on='date', direction='backward')

    # Winsorize returns
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)

    # Clean
    df = df.dropna(subset=['fwd_3m', 'roa', 'regime'])
    print(f"\nClean observations: {len(df):,}")

    # Z-score factors
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'fcf_yield',
                'asset_growth', 'vol_60d', 'mom_12_1']:
        if col in df.columns:
            df[f'{col}_z'] = (df[col] - df[col].mean()) / df[col].std()
            df[f'{col}_z'] = df[f'{col}_z'].clip(-3, 3).fillna(0)

    # Static V4 score (cash-flow focused)
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

    # Adaptive V4 score
    def adaptive_score(row):
        regime = row.get('regime', 'Bull')

        if regime == 'Bull':
            return (
                row['roa_z'] * 0.08 +
                row['ocf_assets_z'] * 0.08 +
                row['fcf_assets_z'] * 0.08 +
                row['fcf_yield_z'] * 0.10 +
                row['gp_assets_z'] * 0.08 +
                (-row['vol_60d_z']) * 0.05 +
                (-row['asset_growth_z']) * 0.05 +
                row['mom_12_1_z'] * 0.20  # Momentum works in bull
            )
        elif regime == 'Bear':
            return (
                row['roa_z'] * 0.15 +
                row['ocf_assets_z'] * 0.15 +
                row['fcf_assets_z'] * 0.15 +
                row['fcf_yield_z'] * 0.15 +
                row['gp_assets_z'] * 0.10 +
                (-row['vol_60d_z']) * 0.20 +  # Low vol critical
                (-row['asset_growth_z']) * 0.10 +
                row['mom_12_1_z'] * 0.00  # Zero momentum
            )
        elif regime == 'High_Vol':
            return (
                row['roa_z'] * 0.10 +
                row['ocf_assets_z'] * 0.10 +
                row['fcf_assets_z'] * 0.10 +
                row['fcf_yield_z'] * 0.10 +
                row['gp_assets_z'] * 0.10 +
                (-row['vol_60d_z']) * 0.25 +  # Very high low-vol weight
                (-row['asset_growth_z']) * 0.10 +
                row['mom_12_1_z'] * 0.00
            )
        else:  # Transition
            return (
                row['roa_z'] * 0.10 +
                row['ocf_assets_z'] * 0.10 +
                row['fcf_assets_z'] * 0.10 +
                row['fcf_yield_z'] * 0.12 +
                row['gp_assets_z'] * 0.08 +
                (-row['vol_60d_z']) * 0.10 +
                (-row['asset_growth_z']) * 0.10 +
                row['mom_12_1_z'] * 0.10
            )

    df['adaptive_v4'] = df.apply(adaptive_score, axis=1)

    # Results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    for strategy in ['static_v4', 'adaptive_v4']:
        name = 'Static V4' if 'static' in strategy else 'Adaptive V4'

        clean = df.dropna(subset=[strategy, 'fwd_3m'])
        corr = clean[strategy].corr(clean['fwd_3m'])

        q5 = clean[clean[strategy] > clean[strategy].quantile(0.8)]['fwd_3m'].mean()
        q1 = clean[clean[strategy] < clean[strategy].quantile(0.2)]['fwd_3m'].mean()
        spread = q5 - q1

        print(f"\n{name}:")
        print(f"  Correlation:   {corr:>+.4f}")
        print(f"  Q5 Return:     {q5:>+.2f}%")
        print(f"  Q1 Return:     {q1:>+.2f}%")
        print(f"  Q5-Q1 Spread:  {spread:>+.2f}%")

    # By regime
    print("\n" + "=" * 70)
    print("PERFORMANCE BY REGIME")
    print("=" * 70)

    for regime in ['Bull', 'Bear', 'High_Vol', 'Transition']:
        regime_df = df[df['regime'] == regime]

        if len(regime_df) < 100:
            continue

        print(f"\n=== {regime} ({len(regime_df):,} obs) ===")

        for strategy in ['static_v4', 'adaptive_v4']:
            name = 'Static' if 'static' in strategy else 'Adaptive'

            clean = regime_df.dropna(subset=[strategy, 'fwd_3m'])
            corr = clean[strategy].corr(clean['fwd_3m'])

            q5 = clean[clean[strategy] > clean[strategy].quantile(0.8)]['fwd_3m'].mean()
            q1 = clean[clean[strategy] < clean[strategy].quantile(0.2)]['fwd_3m'].mean()
            spread = q5 - q1

            print(f"  {name:10s}: r={corr:>+.4f}, spread={spread:>+.2f}%")

    # Improvement summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT SUMMARY")
    print("=" * 70)

    static_spread = df.dropna(subset=['static_v4', 'fwd_3m']).pipe(
        lambda x: x[x['static_v4'] > x['static_v4'].quantile(0.8)]['fwd_3m'].mean() -
                  x[x['static_v4'] < x['static_v4'].quantile(0.2)]['fwd_3m'].mean()
    )

    adaptive_spread = df.dropna(subset=['adaptive_v4', 'fwd_3m']).pipe(
        lambda x: x[x['adaptive_v4'] > x['adaptive_v4'].quantile(0.8)]['fwd_3m'].mean() -
                  x[x['adaptive_v4'] < x['adaptive_v4'].quantile(0.2)]['fwd_3m'].mean()
    )

    improvement = adaptive_spread - static_spread

    print(f"\nStatic V4 Q5-Q1 Spread:   {static_spread:>+.2f}%")
    print(f"Adaptive V4 Q5-Q1 Spread: {adaptive_spread:>+.2f}%")
    print(f"Improvement:              {improvement:>+.2f}%")

    if improvement > 0:
        print(f"\n  Adaptive scoring OUTPERFORMS static by {improvement:.2f}% per quarter")
    else:
        print(f"\n  Static scoring performs better - adaptive adds {improvement:.2f}%")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

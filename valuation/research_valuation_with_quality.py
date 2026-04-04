#!/usr/bin/env python3
"""
Research: Valuation Signals Combined with Quality (Compass Score)

Tests whether combining technical valuation signals with quality filters
improves hit rate and consistency.

Hypothesis: Technical mean-reversion signals work better on high-quality stocks
because quality stocks are more likely to recover from temporary undervaluation.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
NASDAQ_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # 12-month forward returns


def load_data():
    """Load all required data."""
    print("=" * 80)
    print("VALUATION + QUALITY RESEARCH")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    # Income statements
    income = pd.read_sql_query("""
        SELECT symbol, date, gross_profit, net_income, operating_income
        FROM historical_income_statements
        WHERE date >= '1995-01-01'
        ORDER BY symbol, date
    """, conn)
    income['date'] = pd.to_datetime(income['date'])
    print(f"   Income: {len(income):,}")

    # Balance sheets
    balance = pd.read_sql_query("""
        SELECT symbol, date, total_assets
        FROM historical_balance_sheets
        WHERE date >= '1995-01-01' AND total_assets > 1000000
        ORDER BY symbol, date
    """, conn)
    balance['date'] = pd.to_datetime(balance['date'])
    print(f"   Balance: {len(balance):,}")

    # Cash flows
    cashflow = pd.read_sql_query("""
        SELECT symbol, date, operating_cash_flow, free_cash_flow
        FROM historical_cash_flows
        WHERE date >= '1995-01-01'
        ORDER BY symbol, date
    """, conn)
    cashflow['date'] = pd.to_datetime(cashflow['date'])
    print(f"   Cashflow: {len(cashflow):,}")

    print("\n2. Loading prices...")
    sys.stdout.flush()

    prices = pd.read_sql_query("""
        SELECT symbol, date, high, low, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   Prices: {len(prices):,}")

    # Sectors
    print("\n3. Loading sectors...")
    nasdaq_conn = sqlite3.connect(NASDAQ_DB)
    sectors = pd.read_sql_query("""
        SELECT symbol, sector FROM stock_consensus WHERE sector IS NOT NULL
    """, nasdaq_conn)
    nasdaq_conn.close()
    print(f"   Sectors: {len(sectors):,}")

    conn.close()
    return income, balance, cashflow, prices, sectors


def compute_quality_metrics(income, balance, cashflow):
    """Compute Compass-style quality metrics."""
    print("\n4. Computing quality metrics (TTM)...")
    sys.stdout.flush()

    # Merge fundamentals
    fund = income.merge(balance, on=['symbol', 'date'], how='inner')
    fund = fund.merge(cashflow, on=['symbol', 'date'], how='inner')
    fund = fund.sort_values(['symbol', 'date'])

    # TTM calculations (sum of last 4 quarters)
    for col in ['gross_profit', 'net_income', 'operating_income', 'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # Quality ratios (using TTM values / latest assets)
    fund['roa'] = fund['net_income_ttm'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit_ttm'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow_ttm'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow_ttm'] / fund['total_assets']

    # Asset growth (YoY)
    fund['assets_1y_ago'] = fund.groupby('symbol')['total_assets'].shift(4)
    fund['asset_growth'] = (fund['total_assets'] - fund['assets_1y_ago']) / fund['assets_1y_ago']

    print(f"   Quality metrics: {len(fund):,}")
    sys.stdout.flush()

    return fund


def compute_technical_indicators(prices):
    """Compute technical indicators."""
    print("\n5. Computing technical indicators...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()

    # SMA200
    prices['sma200'] = prices.groupby('symbol')['close'].transform(
        lambda x: x.rolling(200, min_periods=100).mean()
    )
    prices['price_vs_sma200'] = (prices['close'] - prices['sma200']) / prices['sma200'] * 100

    # 52-week position
    prices['high_52w'] = prices.groupby('symbol')['high'].transform(
        lambda x: x.rolling(252, min_periods=126).max()
    )
    prices['low_52w'] = prices.groupby('symbol')['low'].transform(
        lambda x: x.rolling(252, min_periods=126).min()
    )
    prices['position_52w'] = (prices['close'] - prices['low_52w']) / (prices['high_52w'] - prices['low_52w']) * 100

    # Volatility (60-day)
    prices['returns'] = prices.groupby('symbol')['close'].pct_change()
    prices['vol_60d'] = prices.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    )

    # Forward returns
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-FWD_HORIZON)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    print(f"   Technicals: {len(prices):,}")
    sys.stdout.flush()

    return prices


def compute_compass_score(df):
    """
    Compute Compass-style quality score from fundamentals.
    Higher score = higher quality.
    """
    # Universe statistics (from research)
    stats = {
        'roa': {'mean': 0.02, 'std': 0.15},
        'ocf_assets': {'mean': 0.05, 'std': 0.12},
        'fcf_assets': {'mean': 0.03, 'std': 0.15},
        'gp_assets': {'mean': 0.25, 'std': 0.20},
        'asset_growth': {'mean': 0.10, 'std': 0.30},
        'vol_60d': {'mean': 45, 'std': 25}
    }

    # Z-scores (capped at ±3)
    for metric, params in stats.items():
        if metric in df.columns:
            z = (df[metric] - params['mean']) / params['std']
            df[f'{metric}_z'] = z.clip(-3, 3)

    # Compass Score = quality factors (positive) - risk factors (negative)
    # Quality: ROA, OCF/Assets, FCF/Assets, GP/Assets
    # Penalty: Asset growth (empire building), Volatility
    df['compass_score'] = (
        df['roa_z'].fillna(0) * 25 +           # Profitability
        df['ocf_assets_z'].fillna(0) * 20 +    # Cash generation
        df['fcf_assets_z'].fillna(0) * 15 +    # Free cash flow
        df['gp_assets_z'].fillna(0) * 15 +     # Gross margins
        -df['asset_growth_z'].fillna(0) * 10 + # Penalize growth
        -df['vol_60d_z'].fillna(0) * 15        # Penalize volatility
    )

    # Normalize to 0-100 scale
    df['compass_score'] = (df['compass_score'] + 50).clip(0, 100)

    return df


def build_dataset(fund, prices, sectors):
    """Build combined dataset with quality + valuation signals."""
    print("\n6. Building dataset...")
    sys.stdout.flush()

    # Quarterly rebalance dates
    rebalance_dates = pd.date_range(start='2000-01-01', end='2025-03-31', freq='QE')

    all_records = []

    for date in rebalance_dates:
        # Latest fundamentals before date
        fund_sub = fund[fund['date'] < date].copy()
        latest_fund = fund_sub.groupby('symbol').last().reset_index()

        # Latest price data at date
        price_sub = prices[prices['date'] <= date].copy()
        latest_price = price_sub.groupby('symbol').last().reset_index()

        # Merge
        df = latest_fund.merge(
            latest_price[['symbol', 'close', 'price_vs_sma200', 'position_52w', 'vol_60d', 'fwd_return']],
            on='symbol', how='inner'
        )
        df = df.merge(sectors, on='symbol', how='left')
        df['rebalance_date'] = date

        # Filter valid
        df = df.dropna(subset=['close', 'fwd_return', 'roa', 'price_vs_sma200'])

        # Compute Compass Score
        df = compute_compass_score(df)

        all_records.append(df)

        if len(all_records) % 20 == 0:
            print(f"   Processed {len(all_records)} quarters...")
            sys.stdout.flush()

    combined = pd.concat(all_records, ignore_index=True)
    print(f"   Total records: {len(combined):,}")
    sys.stdout.flush()

    return combined


def run_analysis(df, name, period):
    """Run quintile analysis with quality interactions."""
    results = []

    # ========================
    # 1. Technical Only (baseline)
    # ========================
    factor_df = df.dropna(subset=['price_vs_sma200', 'fwd_return']).copy()
    factor_df['valuation_factor'] = -factor_df['price_vs_sma200']  # Below SMA = undervalued

    if len(factor_df) >= 500:
        try:
            factor_df['quintile'] = pd.qcut(
                factor_df['valuation_factor'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = factor_df.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            # Hit rate
            quarters = factor_df.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Technical Only (Price vs SMA200)',
                'period': period,
                'n': len(factor_df),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in Technical Only: {e}")

    # ========================
    # 2. Quality Only (Compass Score)
    # ========================
    factor_df = df.dropna(subset=['compass_score', 'fwd_return']).copy()

    if len(factor_df) >= 500:
        try:
            factor_df['quintile'] = pd.qcut(
                factor_df['compass_score'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = factor_df.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            quarters = factor_df.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Quality Only (Compass Score)',
                'period': period,
                'n': len(factor_df),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in Quality Only: {e}")

    # ========================
    # 3. Technical within High Quality (top 40%)
    # ========================
    factor_df = df.dropna(subset=['compass_score', 'price_vs_sma200', 'fwd_return']).copy()
    high_quality = factor_df[factor_df['compass_score'] >= factor_df['compass_score'].quantile(0.6)]

    if len(high_quality) >= 300:
        try:
            high_quality = high_quality.copy()
            high_quality['valuation_factor'] = -high_quality['price_vs_sma200']
            high_quality['quintile'] = pd.qcut(
                high_quality['valuation_factor'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = high_quality.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            quarters = high_quality.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Technical in High Quality (top 40%)',
                'period': period,
                'n': len(high_quality),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in High Quality: {e}")

    # ========================
    # 4. Technical within Top Quality (top 20%)
    # ========================
    top_quality = factor_df[factor_df['compass_score'] >= factor_df['compass_score'].quantile(0.8)]

    if len(top_quality) >= 200:
        try:
            top_quality = top_quality.copy()
            top_quality['valuation_factor'] = -top_quality['price_vs_sma200']
            top_quality['quintile'] = pd.qcut(
                top_quality['valuation_factor'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = top_quality.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            quarters = top_quality.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Technical in Top Quality (top 20%)',
                'period': period,
                'n': len(top_quality),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in Top Quality: {e}")

    # ========================
    # 5. Combined Score (50/50)
    # ========================
    factor_df = df.dropna(subset=['compass_score', 'price_vs_sma200', 'fwd_return']).copy()

    if len(factor_df) >= 500:
        try:
            # Z-score both factors
            val_z = ((-factor_df['price_vs_sma200']) - (-factor_df['price_vs_sma200']).mean()) / (-factor_df['price_vs_sma200']).std()
            qual_z = (factor_df['compass_score'] - factor_df['compass_score'].mean()) / factor_df['compass_score'].std()
            factor_df['combined'] = 0.5 * val_z + 0.5 * qual_z

            factor_df['quintile'] = pd.qcut(
                factor_df['combined'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = factor_df.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            quarters = factor_df.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Combined 50/50 (Quality + Valuation)',
                'period': period,
                'n': len(factor_df),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in Combined: {e}")

    # ========================
    # 6. Combined Score (70/30 Quality Heavy)
    # ========================
    if len(factor_df) >= 500:
        try:
            factor_df['combined_70_30'] = 0.7 * qual_z + 0.3 * val_z

            factor_df['quintile'] = pd.qcut(
                factor_df['combined_70_30'].rank(method='first'),
                5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']
            )

            q_returns = factor_df.groupby('quintile', observed=True)['fwd_return'].mean()
            q1_ret = q_returns.get('Q1', np.nan) * 100
            q5_ret = q_returns.get('Q5', np.nan) * 100
            spread = q1_ret - q5_ret

            quarters = factor_df.groupby('rebalance_date', observed=True).apply(
                lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() >
                         x[x['quintile'] == 'Q5']['fwd_return'].median()
                if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan,
                include_groups=False
            )
            hit_rate = quarters.mean() * 100

            results.append({
                'strategy': 'Combined 70/30 (Quality Heavy)',
                'period': period,
                'n': len(factor_df),
                'q1_return': q1_ret,
                'q5_return': q5_ret,
                'spread': spread,
                'hit_rate': hit_rate
            })
        except Exception as e:
            print(f"   Error in Combined 70/30: {e}")

    return results


def main():
    # Load and process data
    income, balance, cashflow, prices, sectors = load_data()
    fund = compute_quality_metrics(income, balance, cashflow)
    prices = compute_technical_indicators(prices)
    df = build_dataset(fund, prices, sectors)

    # Split IS/OOS
    is_df = df[df['rebalance_date'] <= IS_END].copy()
    oos_df = df[df['rebalance_date'] >= OOS_START].copy()

    print(f"\nDataset splits:")
    print(f"  IS:  {len(is_df):,} records ({IS_START} to {IS_END})")
    print(f"  OOS: {len(oos_df):,} records ({OOS_START}+)")

    # Run analysis
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    all_results = []
    all_results.extend(run_analysis(is_df, 'All', 'IS'))
    all_results.extend(run_analysis(oos_df, 'All', 'OOS'))

    results_df = pd.DataFrame(all_results)

    # Print results
    for period in ['IS', 'OOS']:
        period_name = "In-Sample (2000-2019)" if period == 'IS' else "Out-of-Sample (2020+)"
        print(f"\n{'-' * 80}")
        print(f"{period_name}")
        print(f"{'-' * 80}")

        period_results = results_df[results_df['period'] == period].sort_values('spread', ascending=False)

        print(f"\n{'Strategy':<40} {'Q1 Ret':<10} {'Q5 Ret':<10} {'Spread':<10} {'Hit%':<8} {'N'}")
        print("-" * 90)
        for _, row in period_results.iterrows():
            print(f"{row['strategy']:<40} {row['q1_return']:>+7.1f}%  {row['q5_return']:>+7.1f}%  {row['spread']:>+7.1f}pp  {row['hit_rate']:>5.0f}%  {row['n']:>6,}")

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    oos_results = results_df[results_df['period'] == 'OOS'].sort_values('hit_rate', ascending=False)
    if len(oos_results) > 0:
        best_hr = oos_results.iloc[0]
        best_spread = oos_results.sort_values('spread', ascending=False).iloc[0]

        print(f"\nBest Hit Rate (OOS): {best_hr['strategy']}")
        print(f"  Hit Rate: {best_hr['hit_rate']:.0f}%  |  Spread: {best_hr['spread']:+.1f}pp")

        print(f"\nBest Spread (OOS): {best_spread['strategy']}")
        print(f"  Spread: {best_spread['spread']:+.1f}pp  |  Hit Rate: {best_spread['hit_rate']:.0f}%")

        # Compare technical alone vs with quality
        tech_only = oos_results[oos_results['strategy'].str.contains('Technical Only')]
        tech_qual = oos_results[oos_results['strategy'].str.contains('High Quality|Top Quality')]

        if len(tech_only) > 0 and len(tech_qual) > 0:
            print(f"\nQuality Filter Impact (OOS):")
            print(f"  Technical Alone:         Spread {tech_only.iloc[0]['spread']:+.1f}pp, Hit Rate {tech_only.iloc[0]['hit_rate']:.0f}%")
            for _, row in tech_qual.iterrows():
                print(f"  {row['strategy'][:30]:<30} Spread {row['spread']:+.1f}pp, Hit Rate {row['hit_rate']:.0f}%")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

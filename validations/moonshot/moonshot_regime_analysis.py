#!/usr/bin/env python3
"""
Moonshot Quality-First Regime Analysis
=======================================
Tests if Moonshot performs differently across market regimes.

REGIMES TESTED:
1. Bull vs Bear markets (past 3-month market return)
2. High vs Low volatility periods (VIX proxy)
3. Growth vs Value regime (2020-2021 vs 2022-2023)
4. COVID period (2020) vs Post-COVID (2021-2026)

METHODOLOGY:
- Quality-First v2.0 (≥$500M market cap)
- Z-scores from IS data only
- Monthly rebalancing, 1-month forward returns
- Out-of-sample focus: 2020-2026

EXPECTED OUTCOMES:
- Positive spread in most regimes
- May underperform in extreme risk-off periods (early 2020, late 2022)
- Quality should shine in bear markets vs growth-at-any-cost strategies
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

IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000  # $500M

# Quality-First v2.0 Weights
WEIGHTS = {
    'revenue_growth_3yr': 0.20,
    'eps_growth_3yr': 0.15,
    'gross_margin': 0.15,
    'margin_improvement': 0.10,
    'fcf_margin': 0.15,
    'roe': 0.10,
    'small_cap': 0.10,
    'momentum_12_1': 0.05
}

# Pre-defined regime periods
REGIME_PERIODS = {
    'COVID Crash': ('2020-01-01', '2020-04-30'),
    'COVID Recovery': ('2020-05-01', '2020-12-31'),
    'Growth Boom': ('2021-01-01', '2021-12-31'),
    'Rate Hikes / Bear': ('2022-01-01', '2023-12-31'),
    'Recovery 2024+': ('2024-01-01', '2026-12-31')
}


def passes_quality_filters(row):
    """Check if stock passes Quality-First v2.0 filters."""
    if pd.isna(row['gross_margin']) or row['gross_margin'] < 0.30 or row['gross_margin'] > 0.95:
        return False
    if pd.isna(row['revenue_ttm']) or row['revenue_ttm'] < 50_000_000:
        return False
    if pd.isna(row['revenue_growth']) or row['revenue_growth'] < 0.15 or row['revenue_growth'] > 3.0:
        return False
    if not pd.isna(row['net_income_ttm']) and not pd.isna(row['operating_cash_flow_ttm']):
        if row['net_income_ttm'] > 0:
            if row['operating_cash_flow_ttm'] / row['net_income_ttm'] < 0.7:
                return False
        else:
            if row['operating_cash_flow_ttm'] / row['revenue_ttm'] < -0.5:
                return False
    if not pd.isna(row['total_debt']) and not pd.isna(row['total_assets']) and row['total_assets'] > 0:
        if row['total_debt'] / row['total_assets'] > 2.0:
            return False
    if not pd.isna(row['operating_income_ttm']) and not pd.isna(row['net_income_ttm']):
        if abs(row['net_income_ttm']) > 0 and abs(row['operating_income_ttm']) > 0:
            oi_ni_ratio = abs(row['operating_income_ttm'] / row['net_income_ttm'])
            if oi_ni_ratio < 0.3 or oi_ni_ratio > 3.0:
                return False
    return True


def load_and_score_data():
    """Load data, apply filters, and compute Moonshot scores."""
    print("\n" + "=" * 70)
    print("LOADING AND SCORING DATA")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    print("\nLoading data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted as eps,
               m.market_cap,
               cf.operating_cash_flow, cf.free_cash_flow,
               bs.total_assets, bs.total_debt, bs.total_equity
        FROM historical_income_statements i
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
        LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
        WHERE i.date >= '1990-01-01'
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'date'])

    conn.close()

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")

    # Compute metrics (same as other scripts)
    print("Computing metrics...")
    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    def compute_cagr_3yr(series):
        if len(series) < 13:
            return np.nan
        current = series.iloc[-1]
        three_years_ago = series.iloc[-13]
        if pd.isna(current) or pd.isna(three_years_ago) or three_years_ago <= 0:
            return np.nan
        return (current / three_years_ago) ** (1/3) - 1

    fund['revenue_growth_3yr'] = fund.groupby('symbol')['revenue_ttm'].transform(compute_cagr_3yr)
    fund['eps_growth_3yr'] = fund.groupby('symbol')['eps_ttm'].transform(compute_cagr_3yr)
    fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']
    fund['roe'] = np.where(
        fund['total_equity'].notna() & (fund['total_equity'] > 0),
        fund['net_income_ttm'] / fund['total_equity'],
        np.nan
    )

    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement',
                'revenue_growth_3yr', 'eps_growth_3yr', 'fcf_margin', 'roe']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Compute forward 1-year returns and volatility
    print("Computing forward 1-year returns and volatility...")
    prices = prices.sort_values(['symbol', 'date'])
    prices['fwd_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252) * 100

    # Volatility (60-day rolling)
    prices['returns'] = prices.groupby('symbol')['close'].pct_change()
    prices['vol_60d'] = prices.groupby('symbol')['returns'].transform(
        lambda x: x.rolling(60, min_periods=30).std() * np.sqrt(252) * 100
    )

    # Past 3-month return (for regime classification)
    prices['past_3m'] = prices.groupby('symbol')['close'].pct_change(63) * 100

    # Merge monthly
    fund['year_month'] = fund['date'].dt.to_period('M')
    prices['year_month'] = prices['date'].dt.to_period('M')
    month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

    df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_1yr', 'vol_60d', 'past_3m']],
                    on=['symbol', 'year_month'], how='inner')
    df = df.dropna(subset=['fwd_1yr', 'market_cap'])

    # Filter for ≥$500M
    df = df[df['market_cap'] >= MIN_MARKET_CAP].copy()
    print(f"  {len(df):,} observations after market cap filter")

    # Apply quality filters
    df['passes_quality'] = df.apply(passes_quality_filters, axis=1)
    df = df[df['passes_quality']].copy()
    print(f"  {len(df):,} observations after quality filters")

    # Compute momentum
    prices_sorted = prices.sort_values(['symbol', 'date'])
    prices_sorted['price_12m'] = prices_sorted.groupby('symbol')['close'].shift(252)
    prices_sorted['price_1m'] = prices_sorted.groupby('symbol')['close'].shift(21)
    prices_sorted['momentum_12_1'] = (prices_sorted['price_1m'] - prices_sorted['price_12m']) / prices_sorted['price_12m']

    month_end_momentum = prices_sorted.groupby(['symbol', 'year_month']).last().reset_index()
    df = df.merge(month_end_momentum[['symbol', 'year_month', 'momentum_12_1']],
                  on=['symbol', 'year_month'], how='left')

    # Handle missing factors
    for factor in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
        df[factor] = df[factor].fillna(0)

    # Cap extreme values
    df['revenue_growth_3yr'] = df['revenue_growth_3yr'].clip(-0.3, 1.5)
    df['eps_growth_3yr'] = df['eps_growth_3yr'].clip(-0.5, 2.0)
    df['fcf_margin'] = df['fcf_margin'].clip(-0.5, 0.5)
    df['roe'] = df['roe'].clip(-0.5, 1.0)
    df['momentum_12_1'] = df['momentum_12_1'].clip(-0.5, 2.0)

    # Compute small cap factor
    df['small_cap'] = -np.log10(df['market_cap'] / 1e9)

    # Remove critical missing values
    df = df.dropna(subset=['revenue_growth_3yr', 'gross_margin'])
    print(f"  {len(df):,} observations ready for scoring")

    # Compute z-scores from IS data
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()

    print("\nComputing z-scores from IS data...")
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }

    # Apply z-scores to full dataset
    for factor in WEIGHTS.keys():
        df[f'{factor}_z'] = (df[factor] - factor_stats[factor]['mean']) / factor_stats[factor]['std']

    df['moonshot_score'] = sum(df[f'{factor}_z'] * weight for factor, weight in WEIGHTS.items())

    return df


def analyze_by_predefined_regimes(df):
    """Analyze performance for pre-defined regime periods."""
    print("\n" + "=" * 70)
    print("PRE-DEFINED REGIME ANALYSIS")
    print("=" * 70)

    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\n{'Regime':<25} {'Period':<30} {'Q5-Q1 Spread':>15}")
    print("-" * 75)

    for regime_name, (start_date, end_date) in REGIME_PERIODS.items():
        regime_df = oos_df[(oos_df['date'] >= start_date) & (oos_df['date'] <= end_date)].copy()

        if len(regime_df) == 0:
            print(f"{regime_name:<25} {start_date} to {end_date:<15} {'N/A':>15}")
            continue

        # Assign quintiles across ALL time periods for this regime (time-series ranking)
        if len(regime_df) >= 250:
            try:
                regime_df['quintile'] = pd.qcut(regime_df['moonshot_score'], 5,
                                               labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                               duplicates='drop')
            except ValueError:
                continue
        else:
            continue

        regime_df = regime_df.dropna(subset=['quintile'])

        # Compute spread
        quintile_returns = regime_df.groupby('quintile')['fwd_1yr'].mean()
        if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
            spread = quintile_returns['Q5'] - quintile_returns['Q1']
        else:
            spread = np.nan

        status = "✅ POSITIVE" if not pd.isna(spread) and spread > 0 else "❌ NEGATIVE"
        print(f"{regime_name:<25} {start_date} to {end_date:<15} {spread:>+14.2f}% {status}")


def analyze_by_volatility_regime(df):
    """Analyze performance in high vs low volatility periods."""
    print("\n" + "=" * 70)
    print("VOLATILITY REGIME ANALYSIS")
    print("=" * 70)

    oos_df = df[df['date'] >= OOS_START].copy()

    # Compute monthly average volatility
    monthly_vol = oos_df.groupby('year_month')['vol_60d'].mean()

    # Define high/low vol
    vol_median = monthly_vol.median()
    high_vol_months = monthly_vol[monthly_vol >= vol_median].index
    low_vol_months = monthly_vol[monthly_vol < vol_median].index

    print(f"\n  Median volatility (OOS): {vol_median:.1f}%")
    print(f"  High vol months: {len(high_vol_months)}")
    print(f"  Low vol months: {len(low_vol_months)}")

    # Analyze each regime using time-series ranking (consistent with other analyses)
    for regime_name, regime_months in [('High Volatility', high_vol_months), ('Low Volatility', low_vol_months)]:
        regime_df = oos_df[oos_df['year_month'].isin(regime_months)].copy()

        if len(regime_df) == 0:
            continue

        # Assign quintiles across ALL time periods for this regime (time-series ranking)
        # This is consistent with pre-defined regimes and other validation scripts
        if len(regime_df) >= 250:
            try:
                regime_df['quintile'] = pd.qcut(regime_df['moonshot_score'], 5,
                                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                duplicates='drop')
            except ValueError:
                continue
        else:
            continue

        regime_df = regime_df.dropna(subset=['quintile'])

        # Compute spread
        quintile_returns = regime_df.groupby('quintile')['fwd_1yr'].mean()
        if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
            spread = quintile_returns['Q5'] - quintile_returns['Q1']
        else:
            spread = np.nan

        status = "✅ POSITIVE" if not pd.isna(spread) and spread > 0 else "❌ NEGATIVE"
        print(f"\n  {regime_name}: {spread:>+.2f}% {status}")


def analyze_by_market_return_regime(df):
    """Analyze performance in bull vs bear market months."""
    print("\n" + "=" * 70)
    print("BULL / BEAR MARKET ANALYSIS")
    print("=" * 70)

    oos_df = df[df['date'] >= OOS_START].copy()

    # Compute monthly average past 3-month return
    monthly_past_ret = oos_df.groupby('year_month')['past_3m'].mean()

    # Define bull/bear
    bull_months = monthly_past_ret[monthly_past_ret >= 0].index
    bear_months = monthly_past_ret[monthly_past_ret < 0].index

    print(f"\n  Bull months (past 3m > 0): {len(bull_months)}")
    print(f"  Bear months (past 3m < 0): {len(bear_months)}")

    # Analyze each regime using time-series ranking (consistent with other analyses)
    for regime_name, regime_months in [('Bull Market', bull_months), ('Bear Market', bear_months)]:
        regime_df = oos_df[oos_df['year_month'].isin(regime_months)].copy()

        if len(regime_df) == 0:
            continue

        # Assign quintiles across ALL time periods for this regime (time-series ranking)
        # This is consistent with pre-defined regimes and other validation scripts
        if len(regime_df) >= 250:
            try:
                regime_df['quintile'] = pd.qcut(regime_df['moonshot_score'], 5,
                                                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                                duplicates='drop')
            except ValueError:
                continue
        else:
            continue

        regime_df = regime_df.dropna(subset=['quintile'])

        # Compute spread
        quintile_returns = regime_df.groupby('quintile')['fwd_1yr'].mean()
        if 'Q5' in quintile_returns and 'Q1' in quintile_returns:
            spread = quintile_returns['Q5'] - quintile_returns['Q1']
        else:
            spread = np.nan

        status = "✅ POSITIVE" if not pd.isna(spread) and spread > 0 else "❌ NEGATIVE"
        print(f"\n  {regime_name}: {spread:>+.2f}% {status}")


def main():
    print("=" * 80)
    print("MOONSHOT QUALITY-FIRST REGIME ANALYSIS")
    print("=" * 80)
    print(f"Run at: {datetime.now().isoformat()}")
    print()
    print("Methodology: Quality-First v2.0 (≥$500M market cap)")
    print("Out-of-sample focus: 2020-2026")
    print("=" * 80)

    # Load and score data
    df = load_and_score_data()

    # Analyze by pre-defined regimes
    analyze_by_predefined_regimes(df)

    # Analyze by volatility regime
    analyze_by_volatility_regime(df)

    # Analyze by bull/bear regime
    analyze_by_market_return_regime(df)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n✅ SUCCESS CRITERIA:")
    print("   - Positive spread in most regimes")
    print("   - Quality factors should perform especially well in bear markets")
    print("   - Some underperformance acceptable in extreme risk-off (COVID crash)")
    print("   - Robustness across different market conditions demonstrates strategy viability")


if __name__ == '__main__':
    main()

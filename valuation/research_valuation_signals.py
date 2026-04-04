#!/usr/bin/env python3
"""
Research: Overvalued/Undervalued Stock Signals

Tests fundamental and technical valuation factors to identify which best predict
forward returns. Factors are oriented so higher score = more undervalued.

Quintile Convention:
- Q1 = Top 20% (highest scores = most undervalued) → Expected best returns
- Q5 = Bottom 20% (lowest scores = most overvalued) → Expected worst returns

Success Criteria:
- Q1-Q5 spread > 10% on 12-month forward returns (OOS)
- Hit rate > 55% (Q1 beats Q5 more often than not)
- Monotonic returns (Q1 > Q2 > Q3 > Q4 > Q5)

Note: Using 12-month horizon to match analyst price target timeframes.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # 12-month forward returns (trading days)


def load_data():
    """Load all required data from backtest.db."""
    print("=" * 80)
    print("VALUATION SIGNALS RESEARCH")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    # Income statements for earnings and growth
    income = pd.read_sql_query("""
        SELECT symbol, date, revenue, gross_profit, operating_income, net_income,
               eps_diluted, weighted_avg_shares_diluted
        FROM historical_income_statements
        WHERE date >= '1990-01-01'
        ORDER BY symbol, date
    """, conn)
    income['date'] = pd.to_datetime(income['date'])
    print(f"   Income statements: {len(income):,}")

    # Balance sheets for assets and book value
    balance = pd.read_sql_query("""
        SELECT symbol, date, total_assets, total_equity, total_debt
        FROM historical_balance_sheets
        WHERE date >= '1990-01-01' AND total_assets > 1000000
        ORDER BY symbol, date
    """, conn)
    balance['date'] = pd.to_datetime(balance['date'])
    print(f"   Balance sheets: {len(balance):,}")

    # Key metrics for valuation ratios
    metrics = pd.read_sql_query("""
        SELECT symbol, date, pe_ratio, pb_ratio, ev_to_ebitda,
               enterprise_value, market_cap
        FROM historical_key_metrics
        WHERE date >= '1990-01-01'
        ORDER BY symbol, date
    """, conn)
    metrics['date'] = pd.to_datetime(metrics['date'])
    # Rename for consistency
    metrics = metrics.rename(columns={'pb_ratio': 'price_to_book'})
    print(f"   Key metrics: {len(metrics):,}")

    print("\n2. Loading prices...")
    sys.stdout.flush()

    prices = pd.read_sql_query("""
        SELECT symbol, date, open, high, low, close, adjusted_close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   Prices: {len(prices):,}")

    # Get sector mapping from current stock_consensus (for sector-relative analysis)
    print("\n3. Loading sector mappings...")
    nasdaq_conn = sqlite3.connect(str(PROJECT_ROOT / 'nasdaq_stocks.db'))
    sectors = pd.read_sql_query("""
        SELECT symbol, sector FROM stock_consensus WHERE sector IS NOT NULL
    """, nasdaq_conn)
    nasdaq_conn.close()
    print(f"   Sector mappings: {len(sectors):,}")

    conn.close()
    sys.stdout.flush()

    return income, balance, metrics, prices, sectors


def compute_ttm_metrics(income, balance):
    """Compute trailing twelve month metrics."""
    print("\n4. Computing TTM metrics...")
    sys.stdout.flush()

    income = income.sort_values(['symbol', 'date']).copy()

    # TTM sums
    for col in ['revenue', 'net_income', 'gross_profit', 'operating_income']:
        income[f'{col}_ttm'] = income.groupby('symbol')[col].transform(
            lambda x: x.rolling(4, min_periods=4).sum()
        )

    # EPS TTM
    income['eps_ttm'] = income.groupby('symbol')['eps_diluted'].transform(
        lambda x: x.rolling(4, min_periods=4).sum()
    )

    # Revenue growth YoY
    income['revenue_growth_yoy'] = income.groupby('symbol')['revenue_ttm'].pct_change(4)

    # EPS growth YoY
    income['eps_growth_yoy'] = income.groupby('symbol')['eps_ttm'].pct_change(4)

    print(f"   Computed TTM for {len(income):,} records")
    sys.stdout.flush()

    return income


def compute_technical_indicators(prices):
    """Compute technical indicators from price data."""
    print("\n5. Computing technical indicators...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()

    # Use adjusted_close for calculations
    prices['close'] = prices['adjusted_close']

    # Moving averages
    prices['sma50'] = prices.groupby('symbol')['close'].transform(
        lambda x: x.rolling(50, min_periods=30).mean()
    )
    prices['sma200'] = prices.groupby('symbol')['close'].transform(
        lambda x: x.rolling(200, min_periods=100).mean()
    )

    # Price vs SMA (% deviation)
    prices['price_vs_sma50'] = (prices['close'] - prices['sma50']) / prices['sma50'] * 100
    prices['price_vs_sma200'] = (prices['close'] - prices['sma200']) / prices['sma200'] * 100

    # RSI (14-day)
    delta = prices.groupby('symbol')['close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.groupby(prices['symbol']).transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    )
    avg_loss = loss.groupby(prices['symbol']).transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    )

    rs = avg_gain / avg_loss.replace(0, np.nan)
    prices['rsi'] = 100 - (100 / (1 + rs))

    # 52-week high/low position
    prices['high_52w'] = prices.groupby('symbol')['high'].transform(
        lambda x: x.rolling(252, min_periods=126).max()
    )
    prices['low_52w'] = prices.groupby('symbol')['low'].transform(
        lambda x: x.rolling(252, min_periods=126).min()
    )
    prices['position_52w'] = (prices['close'] - prices['low_52w']) / (prices['high_52w'] - prices['low_52w']) * 100

    print(f"   Computed technicals for {len(prices):,} records")
    sys.stdout.flush()

    return prices


def precompute_forward_returns(prices, horizon=126):
    """Pre-compute forward returns as a DataFrame (more efficient than dict)."""
    print(f"\n6. Pre-computing {horizon}-day forward returns...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-horizon)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    fwd_returns = prices[['symbol', 'date', 'fwd_return']].dropna().copy()
    fwd_returns = fwd_returns.rename(columns={'fwd_return': 'fwd_return_price'})

    print(f"   Forward returns: {len(fwd_returns):,}")
    sys.stdout.flush()

    return fwd_returns


def build_factor_dataset(income, balance, metrics, prices, sectors, fwd_returns_df):
    """Build dataset with all valuation factors per stock/date."""
    print("\n7. Building factor dataset...")
    sys.stdout.flush()

    # Sample on quarterly rebalance dates
    rebalance_dates = pd.date_range(start='2000-01-01', end='2025-06-30', freq='QE')

    all_records = []

    for date in rebalance_dates:
        # Get latest fundamentals before date
        inc_sub = income[income['date'] < date].copy()
        latest_inc = inc_sub.groupby('symbol').last().reset_index()

        bal_sub = balance[balance['date'] < date].copy()
        latest_bal = bal_sub.groupby('symbol').last().reset_index()

        met_sub = metrics[metrics['date'] < date].copy()
        latest_met = met_sub.groupby('symbol').last().reset_index()

        # Get price data at date (or closest before)
        price_sub = prices[prices['date'] <= date].copy()
        latest_price = price_sub.groupby('symbol').last().reset_index()

        # Merge all
        df = latest_met.merge(latest_inc[['symbol', 'net_income_ttm', 'eps_ttm', 'revenue_growth_yoy', 'eps_growth_yoy']],
                              on='symbol', how='left')
        df = df.merge(latest_bal[['symbol', 'total_equity', 'total_assets']],
                      on='symbol', how='left')
        df = df.merge(latest_price[['symbol', 'close', 'rsi', 'price_vs_sma50', 'price_vs_sma200', 'position_52w']],
                      on='symbol', how='left')
        df = df.merge(sectors, on='symbol', how='left')

        # Add date
        df['date'] = date

        # Add forward returns via merge (much faster than apply)
        fwd_at_date = fwd_returns_df[fwd_returns_df['date'] == date][['symbol', 'fwd_return_price']]
        df = df.merge(fwd_at_date, on='symbol', how='left')
        df = df.rename(columns={'fwd_return_price': 'fwd_return'})

        # Filter: need valid data
        df = df.dropna(subset=['close', 'fwd_return'])

        all_records.append(df)

        if len(all_records) % 20 == 0:
            print(f"   Processed {len(all_records)} quarters...")
            sys.stdout.flush()

    combined = pd.concat(all_records, ignore_index=True)
    print(f"   Total records: {len(combined):,}")
    sys.stdout.flush()

    return combined


def calculate_valuation_factors(df):
    """
    Calculate valuation factors oriented so higher = more undervalued.
    """
    print("\n8. Calculating valuation factors...")
    sys.stdout.flush()

    # ===================
    # FUNDAMENTAL FACTORS
    # ===================

    # 1. EV/EBITDA (inverted: lower EV/EBITDA = more undervalued)
    #    Filter extreme values
    df['ev_ebitda_clean'] = df['ev_to_ebitda'].clip(0, 100)
    df['ev_ebitda_factor'] = -df['ev_ebitda_clean']  # Invert: higher = cheaper

    # 2. P/E Ratio (inverted: lower P/E = more undervalued)
    #    Only for positive earnings, handle NaN
    df['pe_clean'] = df['pe_ratio'].clip(0, 100)
    df['pe_factor'] = np.where(
        (df['pe_clean'] > 0) & df['pe_clean'].notna(),
        -df['pe_clean'].fillna(0),
        np.nan
    )

    # 3. Price to Book (inverted: lower P/B = more undervalued)
    df['pb_clean'] = df['price_to_book'].clip(0, 50)
    df['pb_factor'] = np.where(df['pb_clean'].notna(), -df['pb_clean'].fillna(0), np.nan)

    # 4. Earnings Yield (inverse of P/E, already oriented correctly)
    # Higher earnings yield = more undervalued
    df['earnings_yield'] = np.where(
        (df['pe_clean'] > 0) & df['pe_clean'].notna(),
        100 / df['pe_clean'].replace(0, np.nan),
        np.nan
    )
    df['earnings_yield_factor'] = df['earnings_yield']  # No inversion needed

    # ===================
    # TECHNICAL FACTORS
    # ===================

    # 6. RSI (inverted: lower RSI = more oversold = higher score)
    df['rsi_factor'] = -df['rsi']  # Invert: lower RSI = higher factor

    # 7. Price vs SMA200 (inverted: below SMA200 = more oversold)
    df['sma200_factor'] = -df['price_vs_sma200']  # Invert: below SMA = higher factor

    # 8. Price vs SMA50 (inverted: below SMA50 = more oversold)
    df['sma50_factor'] = -df['price_vs_sma50']  # Invert

    # 9. 52-week position (inverted: near 52w low = more oversold)
    df['position_52w_factor'] = -df['position_52w']  # Invert: near low = higher

    print(f"   Calculated factors for {len(df):,} records")
    sys.stdout.flush()

    return df


def run_single_factor_analysis(df, factor_col, factor_name, period='ALL'):
    """Run quintile analysis for a single factor."""
    # Filter to valid data
    factor_df = df.dropna(subset=[factor_col, 'fwd_return']).copy()

    if len(factor_df) < 500:
        return None

    # Assign quintiles (Q1 = top 20% = most undervalued, Q5 = bottom 20%)
    try:
        factor_df['quintile'] = pd.qcut(
            factor_df[factor_col].rank(method='first'),
            5, labels=['Q5', 'Q4', 'Q3', 'Q2', 'Q1']  # Reversed so Q1 = highest
        )
    except ValueError:
        return None

    # Calculate returns by quintile
    q_returns = factor_df.groupby('quintile')['fwd_return'].agg(['mean', 'median', 'count'])

    # Get Q1 and Q5 returns
    q1_return = q_returns.loc['Q1', 'mean'] if 'Q1' in q_returns.index else np.nan
    q5_return = q_returns.loc['Q5', 'mean'] if 'Q5' in q_returns.index else np.nan

    # Calculate spread
    spread = q1_return - q5_return if not pd.isna(q1_return) and not pd.isna(q5_return) else np.nan

    # Calculate hit rate by quarter
    quarters = factor_df.groupby('date').apply(
        lambda x: x[x['quintile'] == 'Q1']['fwd_return'].median() > x[x['quintile'] == 'Q5']['fwd_return'].median()
        if len(x[x['quintile'] == 'Q1']) > 0 and len(x[x['quintile'] == 'Q5']) > 0 else np.nan
    )
    hit_rate = quarters.mean() * 100 if len(quarters.dropna()) > 0 else np.nan

    # Check monotonicity (Q1 > Q2 > Q3 > Q4 > Q5)
    try:
        returns_ordered = [q_returns.loc[f'Q{i}', 'mean'] for i in range(1, 6)]
        monotonic = all(returns_ordered[i] >= returns_ordered[i+1] for i in range(4))
    except:
        monotonic = False

    return {
        'factor': factor_name,
        'period': period,
        'n': len(factor_df),
        'q1_return': q1_return * 100,
        'q2_return': q_returns.loc['Q2', 'mean'] * 100 if 'Q2' in q_returns.index else np.nan,
        'q3_return': q_returns.loc['Q3', 'mean'] * 100 if 'Q3' in q_returns.index else np.nan,
        'q4_return': q_returns.loc['Q4', 'mean'] * 100 if 'Q4' in q_returns.index else np.nan,
        'q5_return': q5_return * 100,
        'spread': spread * 100,
        'hit_rate': hit_rate,
        'monotonic': monotonic
    }


def run_combination_analysis(df, fundamental_factors, technical_factors, weights, combo_name, period='ALL'):
    """Run analysis on a combination of factors with specified weights."""
    combo_df = df.copy()

    # Calculate combined score
    total_weight = sum(weights.values())
    combo_df['combined_score'] = 0

    for factor, weight in weights.items():
        if factor in combo_df.columns:
            # Z-score normalize the factor
            factor_mean = combo_df[factor].mean()
            factor_std = combo_df[factor].std()
            if factor_std > 0:
                combo_df['combined_score'] += ((combo_df[factor] - factor_mean) / factor_std) * (weight / total_weight)

    # Filter valid
    combo_df = combo_df.dropna(subset=['combined_score', 'fwd_return'])

    if len(combo_df) < 500:
        return None

    return run_single_factor_analysis(combo_df, 'combined_score', combo_name, period)


def main():
    # Load data
    income, balance, metrics, prices, sectors = load_data()

    # Process data
    income = compute_ttm_metrics(income, balance)
    prices = compute_technical_indicators(prices)
    fwd_returns = precompute_forward_returns(prices, FWD_HORIZON)

    # Build factor dataset
    df = build_factor_dataset(income, balance, metrics, prices, sectors, fwd_returns)

    # Calculate valuation factors
    df = calculate_valuation_factors(df)

    # Define factors to test
    fundamental_factors = {
        'ev_ebitda_factor': 'EV/EBITDA (inv)',
        'pe_factor': 'P/E Ratio (inv)',
        'pb_factor': 'Price/Book (inv)',
        'earnings_yield_factor': 'Earnings Yield',
    }

    technical_factors = {
        'rsi_factor': 'RSI (inv)',
        'sma200_factor': 'Price vs SMA200 (inv)',
        'sma50_factor': 'Price vs SMA50 (inv)',
        'position_52w_factor': '52W Position (inv)',
    }

    # Split IS and OOS
    is_df = df[df['date'] <= IS_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\nDataset splits:")
    print(f"  IS (In-Sample):  {len(is_df):,} records ({IS_START} to {IS_END})")
    print(f"  OOS (Out-of-Sample): {len(oos_df):,} records ({OOS_START}+)")

    # =========================
    # SINGLE FACTOR ANALYSIS
    # =========================
    print("\n" + "=" * 80)
    print("SINGLE FACTOR ANALYSIS")
    print("=" * 80)

    all_factors = {**fundamental_factors, **technical_factors}
    results = []

    for factor_col, factor_name in all_factors.items():
        # IS
        is_result = run_single_factor_analysis(is_df, factor_col, factor_name, 'IS')
        if is_result:
            results.append(is_result)

        # OOS
        oos_result = run_single_factor_analysis(oos_df, factor_col, factor_name, 'OOS')
        if oos_result:
            results.append(oos_result)

    results_df = pd.DataFrame(results)

    # Print IS results
    print("\n" + "-" * 80)
    print("IN-SAMPLE (2000-2019)")
    print("-" * 80)
    is_results = results_df[results_df['period'] == 'IS'].copy()
    is_results = is_results.sort_values('spread', ascending=False)

    print(f"\n{'Factor':<25} {'Q1 Ret':<10} {'Q5 Ret':<10} {'Spread':<10} {'Hit%':<8} {'Mono'}")
    print("-" * 75)
    for _, row in is_results.iterrows():
        mono_str = "✓" if row['monotonic'] else "✗"
        print(f"{row['factor']:<25} {row['q1_return']:>+7.1f}%  {row['q5_return']:>+7.1f}%  {row['spread']:>+7.1f}pp  {row['hit_rate']:>5.0f}%  {mono_str}")

    # Print OOS results
    print("\n" + "-" * 80)
    print("OUT-OF-SAMPLE (2020+)")
    print("-" * 80)
    oos_results = results_df[results_df['period'] == 'OOS'].copy()
    oos_results = oos_results.sort_values('spread', ascending=False)

    print(f"\n{'Factor':<25} {'Q1 Ret':<10} {'Q5 Ret':<10} {'Spread':<10} {'Hit%':<8} {'Mono'}")
    print("-" * 75)
    for _, row in oos_results.iterrows():
        mono_str = "✓" if row['monotonic'] else "✗"
        print(f"{row['factor']:<25} {row['q1_return']:>+7.1f}%  {row['q5_return']:>+7.1f}%  {row['spread']:>+7.1f}pp  {row['hit_rate']:>5.0f}%  {mono_str}")

    # =========================
    # COMBINATION ANALYSIS
    # =========================
    print("\n" + "=" * 80)
    print("COMBINATION ANALYSIS")
    print("=" * 80)

    # Define combinations to test
    combinations = [
        # Fundamental only
        {
            'name': 'Fundamental Only (Equal)',
            'weights': {k: 1.0 for k in fundamental_factors.keys()}
        },
        # Technical only
        {
            'name': 'Technical Only (Equal)',
            'weights': {k: 1.0 for k in technical_factors.keys()}
        },
        # 50/50 blend
        {
            'name': '50/50 Fund+Tech',
            'weights': {**{k: 0.5 for k in fundamental_factors.keys()},
                       **{k: 0.5 for k in technical_factors.keys()}}
        },
        # 70/30 Fundamental heavy
        {
            'name': '70/30 Fund Heavy',
            'weights': {**{k: 0.7 for k in fundamental_factors.keys()},
                       **{k: 0.3 for k in technical_factors.keys()}}
        },
        # 70/30 Technical heavy
        {
            'name': '70/30 Tech Heavy',
            'weights': {**{k: 0.3 for k in fundamental_factors.keys()},
                       **{k: 0.7 for k in technical_factors.keys()}}
        },
        # Best fundamental + best technical (based on IS results)
        {
            'name': 'Best Fund + Best Tech',
            'weights': {'ev_ebitda_factor': 1.0, 'rsi_factor': 1.0}
        },
    ]

    combo_results = []
    for combo in combinations:
        # IS
        is_result = run_combination_analysis(is_df, fundamental_factors, technical_factors,
                                             combo['weights'], combo['name'], 'IS')
        if is_result:
            combo_results.append(is_result)

        # OOS
        oos_result = run_combination_analysis(oos_df, fundamental_factors, technical_factors,
                                              combo['weights'], combo['name'], 'OOS')
        if oos_result:
            combo_results.append(oos_result)

    combo_df = pd.DataFrame(combo_results)

    # Print combination results
    print("\n" + "-" * 80)
    print("COMBINATION RESULTS")
    print("-" * 80)

    for period in ['IS', 'OOS']:
        period_name = "In-Sample" if period == 'IS' else "Out-of-Sample"
        print(f"\n{period_name}:")
        period_results = combo_df[combo_df['period'] == period].sort_values('spread', ascending=False)

        print(f"{'Combination':<25} {'Q1 Ret':<10} {'Q5 Ret':<10} {'Spread':<10} {'Hit%':<8}")
        print("-" * 65)
        for _, row in period_results.iterrows():
            print(f"{row['factor']:<25} {row['q1_return']:>+7.1f}%  {row['q5_return']:>+7.1f}%  {row['spread']:>+7.1f}pp  {row['hit_rate']:>5.0f}%")

    # =========================
    # SUMMARY
    # =========================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find best OOS performer
    oos_all = pd.concat([
        results_df[results_df['period'] == 'OOS'],
        combo_df[combo_df['period'] == 'OOS']
    ])

    if len(oos_all) > 0:
        best = oos_all.sort_values('spread', ascending=False).iloc[0]
        print(f"\nBest OOS performer: {best['factor']}")
        print(f"  Q1-Q5 Spread: {best['spread']:+.1f}pp")
        print(f"  Hit Rate: {best['hit_rate']:.0f}%")
        print(f"  Q1 Return: {best['q1_return']:+.1f}%")
        print(f"  Q5 Return: {best['q5_return']:+.1f}%")

        # Check success criteria (12-month horizon: expect 10%+ spread)
        print("\nSuccess Criteria Check (12-month horizon):")
        print(f"  [{'✓' if best['spread'] > 10 else '✗'}] Q1-Q5 Spread > 10% (actual: {best['spread']:.1f}%)")
        print(f"  [{'✓' if best['hit_rate'] > 55 else '✗'}] Hit Rate > 55% (actual: {best['hit_rate']:.0f}%)")

    print("\n" + "=" * 80)
    print("RESEARCH COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Quality-First Moonshot Score Validation
========================================
Validates the new Quality-First Moonshot v2.0 with strict quality filters.

Tests:
- In-sample: 1995-2019
- Out-of-sample: 2020-2026
- Market cap segments: Micro, Small, Mid, Large

Target: +15% OOS spread (vs old -4.37%)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # 1-year forward returns

# NEW WEIGHTS (Quality-First v2.0)
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


def passes_quality_filters(row):
    """
    Quality filters from compute_moonshot_scores.py
    Returns True if stock passes ALL filters.
    """
    # Filter 1: Gross margin > 30%
    gross_margin = row.get('gross_margin', np.nan)
    if pd.isna(gross_margin) or gross_margin < 0.30 or gross_margin > 0.95:
        return False

    # Filter 2: Revenue > $50M
    revenue_ttm = row.get('revenue_ttm', 0)
    if pd.isna(revenue_ttm) or revenue_ttm < 50_000_000:
        return False

    # Filter 3: Revenue growth > 15%
    revenue_growth = row.get('revenue_growth', np.nan)
    if pd.isna(revenue_growth) or revenue_growth < 0.15 or revenue_growth > 3.0:
        return False

    # Filter 4: Cash flow quality
    ocf_ttm = row.get('operating_cash_flow_ttm', np.nan)
    net_income_ttm = row.get('net_income_ttm', np.nan)

    if not pd.isna(net_income_ttm) and not pd.isna(ocf_ttm):
        if net_income_ttm > 0:
            if ocf_ttm / net_income_ttm < 0.7:
                return False
        else:
            if ocf_ttm / revenue_ttm < -0.5:
                return False

    # Filter 5: Balance sheet health
    total_debt = row.get('total_debt', 0)
    total_assets = row.get('total_assets', 1)
    if not pd.isna(total_debt) and not pd.isna(total_assets) and total_assets > 0:
        if total_debt / total_assets > 2.0:
            return False

    # Filter 6: Operating income quality
    operating_income_ttm = row.get('operating_income_ttm', np.nan)
    if not pd.isna(operating_income_ttm) and not pd.isna(net_income_ttm):
        if abs(net_income_ttm) > 0 and abs(operating_income_ttm) > 0:
            oi_ni_ratio = abs(operating_income_ttm / net_income_ttm)
            if oi_ni_ratio < 0.3 or oi_ni_ratio > 3.0:
                return False

    return True


def load_data():
    """Load all required data."""
    print("=" * 80)
    print("QUALITY-FIRST MOONSHOT VALIDATION")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    print("\n1. Loading fundamentals...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Load fundamentals with all needed fields
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted as eps, i.weighted_avg_shares_diluted,
               m.market_cap,
               cf.operating_cash_flow, cf.free_cash_flow,
               bs.total_assets, bs.total_debt, bs.total_equity
        FROM historical_income_statements i
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        LEFT JOIN historical_cash_flows cf ON i.symbol = cf.symbol AND i.date = cf.date
        LEFT JOIN historical_balance_sheets bs ON i.symbol = bs.symbol AND i.date = bs.date
        WHERE i.date >= '1990-01-01'
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"   {len(fund):,} fundamental records")

    # Load prices
    print("\n2. Loading prices...")
    sys.stdout.flush()

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"   {len(prices):,} price records")

    conn.close()
    sys.stdout.flush()

    return fund, prices


def compute_fundamentals(fund):
    """Compute all fundamental metrics including 3-year CAGR."""
    print("\n3. Computing TTM metrics...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM metrics
    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Gross margin
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # YoY growth (for quality filters)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4, fill_method=None)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4, fill_method=None)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    print("   Computing 3-year CAGR...")
    sys.stdout.flush()

    # 3-year CAGR
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

    # FCF margin and ROE
    fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']
    fund['roe'] = np.where(
        fund['total_equity'].notna() & (fund['total_equity'] > 0),
        fund['net_income_ttm'] / fund['total_equity'],
        np.nan
    )

    # Clean infinities
    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement',
                'revenue_growth_3yr', 'eps_growth_3yr', 'fcf_margin', 'roe']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"   Computed metrics for {len(fund):,} records")
    sys.stdout.flush()

    return fund


def compute_momentum(prices_df, symbol, date):
    """Compute 12-1 month momentum."""
    sym_prices = prices_df[(prices_df['symbol'] == symbol) & (prices_df['date'] <= date)]
    sym_prices = sym_prices.sort_values('date')

    if len(sym_prices) < 252:
        return np.nan

    price_12m = sym_prices.iloc[-252]['close']
    price_1m = sym_prices.iloc[-21]['close']

    if price_12m <= 0:
        return np.nan

    return (price_1m - price_12m) / price_12m


def compute_moonshot_scores(fund_subset, prices, date):
    """Compute Moonshot scores for a specific date using Quality-First v2.0."""
    results = []

    # Group by symbol and get latest fundamentals for each
    for symbol, group in fund_subset.groupby('symbol'):
        latest = group.sort_values('date').iloc[-1]

        # Apply quality filters
        if not passes_quality_filters(latest):
            continue

        # Get factor values
        factors = {
            'revenue_growth_3yr': latest.get('revenue_growth_3yr', np.nan),
            'eps_growth_3yr': latest.get('eps_growth_3yr', np.nan),
            'gross_margin': latest.get('gross_margin', np.nan),
            'margin_improvement': latest.get('margin_improvement', np.nan),
            'fcf_margin': latest.get('fcf_margin', np.nan),
            'roe': latest.get('roe', np.nan),
            'market_cap': latest.get('market_cap', np.nan)
        }

        # Check critical missing values
        if pd.isna(factors['revenue_growth_3yr']) or pd.isna(factors['gross_margin']):
            continue
        if pd.isna(factors['market_cap']) or factors['market_cap'] <= 0:
            continue

        # Compute momentum
        factors['momentum_12_1'] = compute_momentum(prices, symbol, date)

        # Handle missing optional factors
        for key in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
            if pd.isna(factors[key]):
                factors[key] = 0

        # Cap extreme values
        factors['revenue_growth_3yr'] = np.clip(factors['revenue_growth_3yr'], -0.3, 1.5)
        factors['eps_growth_3yr'] = np.clip(factors['eps_growth_3yr'], -0.5, 2.0)
        factors['fcf_margin'] = np.clip(factors['fcf_margin'], -0.5, 0.5)
        factors['roe'] = np.clip(factors['roe'], -0.5, 1.0)
        factors['momentum_12_1'] = np.clip(factors['momentum_12_1'], -0.5, 2.0)

        # Small cap score
        factors['small_cap'] = -np.log10(factors['market_cap'] / 1e9)

        results.append({
            'symbol': symbol,
            'date': date,
            'market_cap': factors['market_cap'],
            **{k: v for k, v in factors.items() if k != 'market_cap'}
        })

    if len(results) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Compute z-scores
    for factor in WEIGHTS.keys():
        mean = df[factor].mean()
        std = df[factor].std()
        if std > 0:
            df[f'{factor}_z'] = (df[factor] - mean) / std
        else:
            df[f'{factor}_z'] = 0

    # Composite score
    df['moonshot_score'] = sum(
        df[f'{factor}_z'] * weight
        for factor, weight in WEIGHTS.items()
    )

    return df


def compute_forward_returns(prices, symbol, start_date, horizon):
    """Compute forward return for a stock."""
    sym_prices = prices[(prices['symbol'] == symbol) & (prices['date'] >= start_date)]
    sym_prices = sym_prices.sort_values('date')

    if len(sym_prices) < horizon + 1:
        return np.nan

    start_price = sym_prices.iloc[0]['close']
    end_price = sym_prices.iloc[horizon]['close']

    if start_price <= 0:
        return np.nan

    return (end_price - start_price) / start_price


def validate_period(fund, prices, start_date, end_date, period_name):
    """Validate Moonshot score for a specific period."""
    print(f"\n{'=' * 80}")
    print(f"{period_name.upper()} VALIDATION ({start_date} to {end_date})")
    print(f"{'=' * 80}")
    sys.stdout.flush()

    # Get quarterly rebalance dates
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')
    rebalance_dates = [d for d in rebalance_dates if d <= pd.Timestamp(end_date)]

    all_scores = []

    for i, date in enumerate(rebalance_dates):
        if (i + 1) % 4 == 0:
            print(f"  Processing {date.strftime('%Y-%m')} ({i+1}/{len(rebalance_dates)})...")
            sys.stdout.flush()

        # Get fundamentals up to this date
        fund_subset = fund[(fund['date'] <= date) & (fund['date'] >= date - pd.DateOffset(years=5))]

        # Compute scores
        scores = compute_moonshot_scores(fund_subset, prices, date)
        if len(scores) == 0:
            continue

        # Compute forward returns
        fwd_returns = []
        for _, row in scores.iterrows():
            ret = compute_forward_returns(prices, row['symbol'], date, FWD_HORIZON)
            fwd_returns.append(ret)

        scores['fwd_return'] = fwd_returns
        scores = scores.dropna(subset=['fwd_return'])

        all_scores.append(scores)

    if len(all_scores) == 0:
        print(f"  No valid scores for {period_name}")
        return

    # Combine all scores
    combined = pd.concat(all_scores, ignore_index=True)
    print(f"\n  Total observations: {len(combined):,}")
    print(f"  Stocks passing quality filters: {combined['symbol'].nunique():,}")
    sys.stdout.flush()

    # Quintile analysis
    combined['quintile'] = pd.qcut(combined['moonshot_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    print(f"\n  Quintile Performance:")
    print(f"  {'Quintile':<10} {'Avg Return':<12} {'Count':<10}")
    print(f"  {'-' * 35}")

    q_results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = combined[combined['quintile'] == q]
        if len(q_data) > 0:
            avg_ret = q_data['fwd_return'].mean()
            q_results[q] = avg_ret
            print(f"  {q:<10} {avg_ret:>+10.2%}  {len(q_data):>8,}")

    if 'Q5' in q_results and 'Q1' in q_results:
        spread = q_results['Q5'] - q_results['Q1']
        print(f"  {'-' * 35}")
        print(f"  {'Q5 - Q1':<10} {spread:>+10.2%}")
        print(f"  {'=' * 35}")

    # Market cap breakdown
    print(f"\n  Market Cap Breakdown:")
    combined['cap_segment'] = pd.cut(
        combined['market_cap'],
        bins=[0, 500e6, 2e9, 10e9, np.inf],
        labels=['Micro (<$500M)', 'Small ($500M-$2B)', 'Mid ($2B-$10B)', 'Large (>$10B)']
    )

    print(f"  {'Segment':<20} {'Q5-Q1 Spread':<15} {'Observations':<12}")
    print(f"  {'-' * 50}")

    for segment in ['Micro (<$500M)', 'Small ($500M-$2B)', 'Mid ($2B-$10B)', 'Large (>$10B)']:
        seg_data = combined[combined['cap_segment'] == segment]
        if len(seg_data) > 0:
            seg_data['q'] = pd.qcut(seg_data['moonshot_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            q5 = seg_data[seg_data['q'] == 'Q5']['fwd_return'].mean()
            q1 = seg_data[seg_data['q'] == 'Q1']['fwd_return'].mean()
            spread = q5 - q1
            print(f"  {segment:<20} {spread:>+13.2%}  {len(seg_data):>10,}")

    sys.stdout.flush()


def main():
    # Load data
    fund, prices = load_data()

    # Compute fundamentals
    fund = compute_fundamentals(fund)

    # In-sample validation
    validate_period(fund, prices, IS_START, IS_END, "IN-SAMPLE")

    # Out-of-sample validation
    validate_period(fund, prices, OOS_START, OOS_END, "OUT-OF-SAMPLE")

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 80}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

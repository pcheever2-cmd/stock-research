#!/usr/bin/env python3
"""
Hybrid Moonshot Score Validation (OPTIMIZED)
============================================
Validates the Hybrid Moonshot strategy:
- Micro-caps (<$500M): Original Moonshot (no quality filters, YoY growth)
- Small/Mid/Large (≥$500M): Quality-First v2.0 (strict filters, 3yr CAGR)

OPTIMIZATIONS:
- Pre-computes ALL forward returns once (major speedup)
- Pre-computes ALL momentum values once (major speedup)
- Estimated runtime: ~2-3 hours

Tests:
- In-sample: 1995-2019
- Out-of-sample: 2020-2026
- Market cap segments: Micro, Small, Mid, Large

Expected Results:
- Micro-caps: +10-11% OOS (original methodology)
- Small/Mid/Large: +15-18% OOS (quality-first methodology)
- Blended: +14-15% OOS
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
MICRO_CAP_THRESHOLD = 500_000_000  # $500M

# Original Moonshot weights (for micro-caps <$500M)
ORIGINAL_WEIGHTS = {
    'revenue_growth': 0.25,        # YoY growth
    'eps_growth': 0.20,            # YoY growth
    'gross_margin': 0.25,          # Level
    'small_cap': 0.15,             # Market cap factor
    'momentum_12_1': 0.15          # 12-1 month momentum
}

# Quality-First weights (for companies ≥$500M)
QUALITY_FIRST_WEIGHTS = {
    'revenue_growth_3yr': 0.20,    # 3-year CAGR
    'eps_growth_3yr': 0.15,        # 3-year CAGR
    'gross_margin': 0.15,          # Level
    'margin_improvement': 0.10,    # YoY change
    'fcf_margin': 0.15,            # Free Cash Flow / Revenue
    'roe': 0.10,                   # Return on Equity
    'small_cap': 0.10,             # Market cap factor
    'momentum_12_1': 0.05          # 12-1 month momentum
}


def passes_quality_filters(row):
    """
    Quality filters for Quality-First methodology (≥$500M companies).
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
    print("HYBRID MOONSHOT VALIDATION (OPTIMIZED)")
    print("Micro-caps (<$500M): Original | ≥$500M: Quality-First v2.0")
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

    # YoY growth (for quality filters and original methodology)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4, fill_method=None)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4, fill_method=None)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    print("   Computing 3-year CAGR...")
    sys.stdout.flush()

    # 3-year CAGR (for quality-first methodology)
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

    # FCF margin and ROE (for quality-first methodology)
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


def precompute_forward_returns(prices, horizon=252):
    """Pre-compute ALL forward returns at once (major optimization)."""
    print(f"\n4. Pre-computing {horizon}-day forward returns...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()
    prices['future_close'] = prices.groupby('symbol')['close'].shift(-horizon)
    prices['fwd_return'] = (prices['future_close'] - prices['close']) / prices['close']

    fwd_returns = prices[['symbol', 'date', 'fwd_return']].set_index(['symbol', 'date'])['fwd_return'].to_dict()

    print(f"   Computed {len(fwd_returns):,} forward return entries")
    sys.stdout.flush()

    return fwd_returns


def precompute_momentum(prices):
    """Pre-compute ALL 12-1 momentum values at once (major optimization)."""
    print(f"\n5. Pre-computing 12-1 momentum...")
    sys.stdout.flush()

    prices = prices.sort_values(['symbol', 'date']).copy()

    # 12-month price (252 trading days ago)
    prices['price_12m'] = prices.groupby('symbol')['close'].shift(252)
    # 1-month price (21 trading days ago)
    prices['price_1m'] = prices.groupby('symbol')['close'].shift(21)

    # Momentum calculation
    prices['momentum_12_1'] = (prices['price_1m'] - prices['price_12m']) / prices['price_12m']

    momentums = prices[['symbol', 'date', 'momentum_12_1']].set_index(['symbol', 'date'])['momentum_12_1'].to_dict()

    print(f"   Computed {len(momentums):,} momentum entries")
    sys.stdout.flush()

    return momentums


def compute_hybrid_moonshot_scores(fund_subset, prices, date, momentum_dict):
    """
    Compute Hybrid Moonshot scores:
    - Micro-caps (<$500M): Original methodology
    - Larger companies (≥$500M): Quality-First v2.0
    """
    original_results = []
    quality_first_results = []

    # Group by symbol and get latest fundamentals for each
    for symbol, group in fund_subset.groupby('symbol'):
        latest = group.sort_values('date').iloc[-1]
        market_cap = latest.get('market_cap', np.nan)

        if pd.isna(market_cap) or market_cap <= 0:
            continue

        # Branch based on market cap
        if market_cap < MICRO_CAP_THRESHOLD:
            # ORIGINAL MOONSHOT (no quality filters)
            factors = {
                'revenue_growth': latest.get('revenue_growth', np.nan),
                'eps_growth': latest.get('eps_growth', np.nan),
                'gross_margin': latest.get('gross_margin', np.nan),
                'market_cap': market_cap
            }

            # Check critical missing values
            if pd.isna(factors['revenue_growth']) or pd.isna(factors['gross_margin']):
                continue

            # Get momentum
            factors['momentum_12_1'] = momentum_dict.get((symbol, date), np.nan)

            # Handle missing optional factors
            if pd.isna(factors['eps_growth']):
                factors['eps_growth'] = 0
            if pd.isna(factors['momentum_12_1']):
                factors['momentum_12_1'] = 0

            # Cap extreme values (more lenient for early-stage)
            factors['revenue_growth'] = np.clip(factors['revenue_growth'], -0.5, 3.0)
            factors['eps_growth'] = np.clip(factors['eps_growth'], -1.0, 3.0)
            factors['gross_margin'] = np.clip(factors['gross_margin'], 0.0, 0.95)
            factors['momentum_12_1'] = np.clip(factors['momentum_12_1'], -0.5, 2.0)

            # Small cap score
            factors['small_cap'] = -np.log10(market_cap / 1e9)

            original_results.append({
                'symbol': symbol,
                'date': date,
                'market_cap': market_cap,
                'methodology': 'original',
                **{k: v for k, v in factors.items() if k != 'market_cap'}
            })

        else:
            # QUALITY-FIRST v2.0 (strict filters)
            if not passes_quality_filters(latest):
                continue

            factors = {
                'revenue_growth_3yr': latest.get('revenue_growth_3yr', np.nan),
                'eps_growth_3yr': latest.get('eps_growth_3yr', np.nan),
                'gross_margin': latest.get('gross_margin', np.nan),
                'margin_improvement': latest.get('margin_improvement', np.nan),
                'fcf_margin': latest.get('fcf_margin', np.nan),
                'roe': latest.get('roe', np.nan),
                'market_cap': market_cap
            }

            # Check critical missing values
            if pd.isna(factors['revenue_growth_3yr']) or pd.isna(factors['gross_margin']):
                continue

            # Get momentum
            factors['momentum_12_1'] = momentum_dict.get((symbol, date), np.nan)

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
            factors['small_cap'] = -np.log10(market_cap / 1e9)

            quality_first_results.append({
                'symbol': symbol,
                'date': date,
                'market_cap': market_cap,
                'methodology': 'quality_first',
                **{k: v for k, v in factors.items() if k != 'market_cap'}
            })

    # Process original methodology stocks
    df_original = pd.DataFrame(original_results) if original_results else pd.DataFrame()
    if len(df_original) > 0:
        for factor in ORIGINAL_WEIGHTS.keys():
            mean = df_original[factor].mean()
            std = df_original[factor].std()
            if std > 0:
                df_original[f'{factor}_z'] = (df_original[factor] - mean) / std
            else:
                df_original[f'{factor}_z'] = 0

        df_original['moonshot_score'] = sum(
            df_original[f'{factor}_z'] * weight
            for factor, weight in ORIGINAL_WEIGHTS.items()
        )

    # Process quality-first methodology stocks
    df_quality = pd.DataFrame(quality_first_results) if quality_first_results else pd.DataFrame()
    if len(df_quality) > 0:
        for factor in QUALITY_FIRST_WEIGHTS.keys():
            mean = df_quality[factor].mean()
            std = df_quality[factor].std()
            if std > 0:
                df_quality[f'{factor}_z'] = (df_quality[factor] - mean) / std
            else:
                df_quality[f'{factor}_z'] = 0

        df_quality['moonshot_score'] = sum(
            df_quality[f'{factor}_z'] * weight
            for factor, weight in QUALITY_FIRST_WEIGHTS.items()
        )

    # Combine both methodologies
    if len(df_original) > 0 and len(df_quality) > 0:
        combined = pd.concat([df_original, df_quality], ignore_index=True)
    elif len(df_original) > 0:
        combined = df_original
    elif len(df_quality) > 0:
        combined = df_quality
    else:
        return pd.DataFrame()

    return combined


def validate_period(fund, prices, fwd_returns_dict, momentum_dict, start_date, end_date, period_name):
    """Validate Hybrid Moonshot score for a specific period."""
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

        # Compute scores using hybrid methodology
        scores = compute_hybrid_moonshot_scores(fund_subset, prices, date, momentum_dict)
        if len(scores) == 0:
            continue

        # Get pre-computed forward returns
        scores['fwd_return'] = scores.apply(
            lambda row: fwd_returns_dict.get((row['symbol'], date), np.nan),
            axis=1
        )
        scores = scores.dropna(subset=['fwd_return'])

        all_scores.append(scores)

    if len(all_scores) == 0:
        print(f"  No valid scores for {period_name}")
        return

    # Combine all scores
    combined = pd.concat(all_scores, ignore_index=True)
    print(f"\n  Total observations: {len(combined):,}")

    # Count by methodology
    original_count = len(combined[combined['methodology'] == 'original'])
    quality_count = len(combined[combined['methodology'] == 'quality_first'])
    print(f"  Original methodology (micro-caps): {original_count:,} ({original_count/len(combined)*100:.1f}%)")
    print(f"  Quality-First methodology (≥$500M): {quality_count:,} ({quality_count/len(combined)*100:.1f}%)")
    sys.stdout.flush()

    # Overall quintile analysis
    print(f"\n  OVERALL Quintile Performance:")
    combined['quintile'] = pd.qcut(combined['moonshot_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

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

    print(f"  {'Segment':<20} {'Q5-Q1 Spread':<15} {'Count':<10} {'Method':<15}")
    print(f"  {'-' * 65}")

    for segment in ['Micro (<$500M)', 'Small ($500M-$2B)', 'Mid ($2B-$10B)', 'Large (>$10B)']:
        seg_data = combined[combined['cap_segment'] == segment]
        if len(seg_data) > 0:
            seg_data['quintile_seg'] = pd.qcut(seg_data['moonshot_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

            q5_ret = seg_data[seg_data['quintile_seg'] == 'Q5']['fwd_return'].mean()
            q1_ret = seg_data[seg_data['quintile_seg'] == 'Q1']['fwd_return'].mean()
            spread = q5_ret - q1_ret

            # Determine predominant methodology
            method = 'Original' if segment == 'Micro (<$500M)' else 'Quality-First'

            print(f"  {segment:<20} {spread:>+12.2%}   {len(seg_data):>8,}  {method:<15}")

    sys.stdout.flush()


def main():
    # Load data
    fund, prices = load_data()

    # Compute fundamentals
    fund = compute_fundamentals(fund)

    # Pre-compute forward returns and momentum
    fwd_returns_dict = precompute_forward_returns(prices, FWD_HORIZON)
    momentum_dict = precompute_momentum(prices)

    # Validate in-sample period
    validate_period(fund, prices, fwd_returns_dict, momentum_dict, IS_START, IS_END, "In-Sample")

    # Validate out-of-sample period
    validate_period(fund, prices, fwd_returns_dict, momentum_dict, OOS_START, OOS_END, "Out-of-Sample")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

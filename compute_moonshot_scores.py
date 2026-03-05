#!/usr/bin/env python3
"""
Compute Moonshot Scores for Dashboard (Quality-First v2.0)
===========================================================
Identifies high-quality growth stocks with significant long-term potential.

SCOPE: Companies with market cap ≥$500M only
(Micro-caps excluded due to liquidity/coverage concerns)

QUALITY FILTERS (Must Pass ALL):
1. Gross margin > 30% (competitive advantage)
2. Revenue > $50M (real business)
3. Revenue growth > 15% (actually growing)
4. Cash flow quality (OCF/NI > 0.7 if profitable, OCF/Rev > -50% if not)
5. Balance sheet health (Debt/Assets < 2.0)
6. Operating income quality (OI/NI ratio 0.3-3.0, no one-time gains)

GROWTH SCORING (Only for stocks passing quality filters):
1. Revenue Growth (3-year CAGR) - 20% weight
2. EPS Growth (3-year CAGR) - 15% weight
3. Gross Margin Level - 15% weight
4. Gross Margin Improvement - 10% weight
5. FCF Margin - 15% weight
6. ROE - 10% weight
7. Smaller Market Cap - 10% weight
8. 12-1 Month Momentum - 5% weight

VALIDATED OUT-OF-SAMPLE PERFORMANCE (2020-2026):
- Full Universe (≥$500M): +23.58% Q5-Q1 spread
- Small-cap ($500M-$2B): +19.59% spread
- Mid-cap ($2B-$10B): +22.32% spread
- Large-cap (>$10B): +24.35% spread (monotonic quintiles)
- FF6 Alpha: +33.98% (1-year spread, t-stat 4.82***)

Top Contributing Factors:
- Revenue Growth 3yr: +32.04% single-factor spread
- FCF Margin: +9.74% spread
- Small Cap: +3.96% spread
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

# Minimum market cap for scoring (micro-caps excluded)
MIN_MARKET_CAP = 500_000_000  # $500M

# Factor weights (Quality-First v2.0)
WEIGHTS = {
    'revenue_growth_3yr': 0.20,    # 3-year CAGR (smoother than annual)
    'eps_growth_3yr': 0.15,        # 3-year CAGR
    'gross_margin': 0.15,          # Level
    'margin_improvement': 0.10,    # YoY change
    'fcf_margin': 0.15,            # Free Cash Flow / Revenue
    'roe': 0.10,                   # Return on Equity
    'small_cap': 0.10,             # Market cap factor
    'momentum_12_1': 0.05          # 12-1 month momentum
}


def assign_grade(percentile):
    """Assign letter grade based on percentile."""
    if percentile >= 85:
        return 'A'
    elif percentile >= 60:
        return 'B'
    elif percentile >= 40:
        return 'C'
    elif percentile >= 20:
        return 'D'
    else:
        return 'F'


def load_data():
    """Load price and fundamental data from backtest.db."""
    print("Loading data from backtest.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Get prices for momentum calculation (last 14 months)
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 0.5
          AND date >= date('now', '-14 months')
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Get fundamentals (last 4+ years for 3-year CAGR calculations)
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
        WHERE i.date >= date('now', '-5 years')
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    conn.close()

    # Sort by symbol and date
    fund = fund.sort_values(['symbol', 'date'])

    # Compute TTM metrics
    print("  Computing TTM metrics...")
    sys.stdout.flush()

    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Compute gross margin
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']

    # Compute YoY growth (compare to 4 quarters ago)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Compute 3-year CAGR for revenue and EPS (12 quarters = 3 years)
    print("  Computing 3-year CAGR...")
    sys.stdout.flush()

    def compute_cagr_3yr(series):
        """Compute 3-year CAGR (12 quarters)."""
        if len(series) < 13:
            return np.nan
        current = series.iloc[-1]
        three_years_ago = series.iloc[-13]
        if pd.isna(current) or pd.isna(three_years_ago) or three_years_ago <= 0:
            return np.nan
        return (current / three_years_ago) ** (1/3) - 1

    fund['revenue_growth_3yr'] = fund.groupby('symbol')['revenue_ttm'].transform(compute_cagr_3yr)
    fund['eps_growth_3yr'] = fund.groupby('symbol')['eps_ttm'].transform(compute_cagr_3yr)

    # Compute FCF margin and ROE
    fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']
    fund['roe'] = np.where(
        fund['total_equity'].notna() & (fund['total_equity'] > 0),
        fund['net_income_ttm'] / fund['total_equity'],
        np.nan
    )

    # Clean up infinities
    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement',
                'revenue_growth_3yr', 'eps_growth_3yr', 'fcf_margin', 'roe']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")
    sys.stdout.flush()

    return prices, fund


def compute_momentum_12_1(prices_df, symbol):
    """Compute 12-1 month momentum (skip most recent month)."""
    sym_prices = prices_df[prices_df['symbol'] == symbol].sort_values('date')

    if len(sym_prices) < 252:  # Need ~1 year of data
        return np.nan

    # Price 12 months ago
    price_12m = sym_prices.iloc[-252]['close']
    # Price 1 month ago (skip last ~21 trading days)
    price_1m = sym_prices.iloc[-21]['close']

    if price_12m <= 0:
        return np.nan

    return (price_1m - price_12m) / price_12m


def passes_quality_filters(latest_fund):
    """
    Strict quality gates - stock must pass ALL to be scored.
    Returns: (passes: bool, reason: str)
    """
    # Filter 1: Gross margin > 30% (competitive advantage)
    gross_margin = latest_fund.get('gross_margin', 0)
    if pd.isna(gross_margin) or gross_margin < 0.30:
        return False, "Gross margin < 30%"
    if gross_margin > 0.95:
        return False, "Gross margin suspicious (>95%)"

    # Filter 2: Revenue > $50M (real business)
    revenue_ttm = latest_fund.get('revenue_ttm', 0)
    if pd.isna(revenue_ttm) or revenue_ttm < 50_000_000:
        return False, "Revenue < $50M"

    # Filter 3: Revenue growth > 15% (actually growing)
    revenue_growth = latest_fund.get('revenue_growth', 0)
    if pd.isna(revenue_growth) or revenue_growth < 0.15:
        return False, "Revenue growth < 15%"
    if revenue_growth > 3.0:
        return False, "Revenue growth suspicious (>300%)"

    # Filter 4: Cash flow quality
    ocf_ttm = latest_fund.get('operating_cash_flow_ttm', np.nan)
    net_income_ttm = latest_fund.get('net_income_ttm', np.nan)

    if not pd.isna(net_income_ttm) and not pd.isna(ocf_ttm):
        if net_income_ttm > 0:
            # If profitable: OCF/Net Income > 0.7 (high cash conversion)
            if ocf_ttm / net_income_ttm < 0.7:
                return False, "Low cash conversion (OCF/NI < 0.7)"
        else:
            # If unprofitable: OCF/Revenue > -50% (not burning infinite cash)
            if ocf_ttm / revenue_ttm < -0.5:
                return False, "Excessive cash burn (OCF/Rev < -50%)"

    # Filter 5: Balance sheet health
    total_debt = latest_fund.get('total_debt', 0)
    total_assets = latest_fund.get('total_assets', 1)
    if not pd.isna(total_debt) and not pd.isna(total_assets) and total_assets > 0:
        if total_debt / total_assets > 2.0:
            return False, "Overleveraged (Debt > 2x Assets)"

    # Filter 6: Operating income quality (no one-time gains)
    operating_income_ttm = latest_fund.get('operating_income_ttm', np.nan)
    if not pd.isna(operating_income_ttm) and not pd.isna(net_income_ttm):
        if abs(net_income_ttm) > 0 and abs(operating_income_ttm) > 0:
            oi_ni_ratio = abs(operating_income_ttm / net_income_ttm)
            if oi_ni_ratio < 0.3 or oi_ni_ratio > 3.0:
                return False, "One-time gain/loss detected (OI/NI ratio OOR)"

    return True, "Passed all quality filters"


def compute_raw_score(symbol, prices_df, fund_df):
    """Compute raw Moonshot Score for a single symbol (Quality-First v2.0)."""
    # Get latest fundamentals
    sym_fund = fund_df[fund_df['symbol'] == symbol]
    if len(sym_fund) == 0:
        return None, None

    latest_fund = sym_fund.sort_values('date').iloc[-1]
    market_cap = latest_fund.get('market_cap', np.nan)

    # EXCLUDE micro-caps (<$500M)
    if pd.isna(market_cap) or market_cap < MIN_MARKET_CAP:
        return None, None

    # QUALITY FILTERS - Must pass ALL to be scored
    passes, reason = passes_quality_filters(latest_fund)
    if not passes:
        return None, None

    # Get factor values
    factors = {
        'revenue_growth_3yr': latest_fund.get('revenue_growth_3yr', np.nan),
        'eps_growth_3yr': latest_fund.get('eps_growth_3yr', np.nan),
        'gross_margin': latest_fund.get('gross_margin', np.nan),
        'margin_improvement': latest_fund.get('margin_improvement', np.nan),
        'fcf_margin': latest_fund.get('fcf_margin', np.nan),
        'roe': latest_fund.get('roe', np.nan),
        'market_cap': market_cap
    }

    # Compute momentum
    factors['momentum_12_1'] = compute_momentum_12_1(prices_df, symbol)

    # Check for critical missing values
    if pd.isna(factors['revenue_growth_3yr']) or pd.isna(factors['gross_margin']):
        return None, None

    # Handle missing values for optional factors (use 0 as neutral)
    if pd.isna(factors['eps_growth_3yr']):
        factors['eps_growth_3yr'] = 0
    if pd.isna(factors['margin_improvement']):
        factors['margin_improvement'] = 0
    if pd.isna(factors['fcf_margin']):
        factors['fcf_margin'] = 0
    if pd.isna(factors['roe']):
        factors['roe'] = 0
    if pd.isna(factors['momentum_12_1']):
        factors['momentum_12_1'] = 0

    # Cap extreme values
    factors['revenue_growth_3yr'] = np.clip(factors['revenue_growth_3yr'], -0.3, 1.5)
    factors['eps_growth_3yr'] = np.clip(factors['eps_growth_3yr'], -0.5, 2.0)
    factors['fcf_margin'] = np.clip(factors['fcf_margin'], -0.5, 0.5)
    factors['roe'] = np.clip(factors['roe'], -0.5, 1.0)
    factors['momentum_12_1'] = np.clip(factors['momentum_12_1'], -0.5, 2.0)

    # Compute small cap score (inverse of market cap, log-scaled)
    factors['small_cap'] = -np.log10(market_cap / 1e9)

    return factors, factors


def compute_all_scores():
    """Compute Moonshot Scores for all stocks (Quality-First v2.0, ≥$500M only)."""
    prices, fund = load_data()

    print("\nComputing raw scores for all stocks...")
    sys.stdout.flush()

    # Get all symbols that have both price and fundamental data
    price_symbols = set(prices['symbol'].unique())
    fund_symbols = set(fund['symbol'].unique())
    all_symbols = price_symbols & fund_symbols

    print(f"  {len(all_symbols):,} stocks with both price and fundamental data")
    sys.stdout.flush()

    # Compute raw factor values for Quality-First methodology
    results = []
    excluded = 0

    for i, symbol in enumerate(all_symbols):
        factors, _ = compute_raw_score(symbol, prices, fund)
        if factors is not None:
            result = {'symbol': symbol}
            result.update(factors)
            results.append(result)
        else:
            excluded += 1

        if (i + 1) % 1000 == 0:
            print(f"    {i + 1:,}/{len(all_symbols):,} processed...")
            sys.stdout.flush()

    print(f"  {len(results):,} stocks passed quality filters (≥$500M)")
    print(f"  {excluded:,} stocks excluded ({excluded/len(all_symbols)*100:.1f}%)")
    sys.stdout.flush()

    # Compute z-scores and composite scores
    print("  Computing z-scores and composite scores...")
    sys.stdout.flush()

    df = pd.DataFrame(results)
    if len(df) > 0:
        for factor in WEIGHTS.keys():
            mean = df[factor].mean()
            std = df[factor].std()
            if std > 0:
                df[f'{factor}_z'] = (df[factor] - mean) / std
            else:
                df[f'{factor}_z'] = 0

        df['raw_score'] = sum(
            df[f'{factor}_z'] * weight
            for factor, weight in WEIGHTS.items()
        )

    # Convert to percentile-based score (0-100) across ALL stocks
    percentile = df['raw_score'].rank(pct=True)
    df['moonshot_score'] = (percentile * 100).round(0).astype(int)

    # Spread out top tier based on raw score
    df.loc[df['raw_score'] >= 0.8, 'moonshot_score'] = 96
    df.loc[df['raw_score'] >= 1.0, 'moonshot_score'] = 97
    df.loc[df['raw_score'] >= 1.3, 'moonshot_score'] = 98
    df.loc[df['raw_score'] >= 1.6, 'moonshot_score'] = 99
    df.loc[df['raw_score'] >= 2.0, 'moonshot_score'] = 100

    # Assign grades
    df['moonshot_grade'] = df['moonshot_score'].apply(assign_grade)

    return df


def update_nasdaq_db(scores_df):
    """Update nasdaq_stocks.db with Moonshot Scores."""
    print("\nUpdating nasdaq_stocks.db...")
    sys.stdout.flush()

    conn = sqlite3.connect(NASDAQ_DB)
    cursor = conn.cursor()

    # Add columns if they don't exist
    for col, dtype in [('moonshot_score', 'INTEGER'),
                       ('moonshot_grade', 'TEXT'),
                       ('moonshot_updated_at', 'TEXT')]:
        try:
            cursor.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col} {dtype}")
            print(f"  Added {col} column")
        except sqlite3.OperationalError:
            pass

    # Clear existing scores
    cursor.execute("""
        UPDATE stock_consensus
        SET moonshot_score = NULL, moonshot_grade = NULL, moonshot_updated_at = NULL
    """)
    cleared = cursor.rowcount
    print(f"  Cleared {cleared:,} existing scores")

    # Update scores
    updated = 0
    timestamp = datetime.now().isoformat()

    for _, row in scores_df.iterrows():
        cursor.execute("""
            UPDATE stock_consensus
            SET moonshot_score = ?, moonshot_grade = ?, moonshot_updated_at = ?
            WHERE symbol = ?
        """, (row['moonshot_score'], row['moonshot_grade'], timestamp, row['symbol']))
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"  Updated {updated:,} stocks in stock_consensus")
    sys.stdout.flush()


def print_examples(scores_df):
    """Print example stocks for verification."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Example Stocks")
    print("=" * 60)

    famous = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'META', 'PLTR', 'COIN', 'SMCI', 'AMD']

    print(f"\n{'Symbol':<8} {'Score':<8} {'Grade':<6} {'RevGr3yr':<12} {'EPSGr3yr':<12} {'GrossM':<10} {'FCF%':<10}")
    print("-" * 95)

    for symbol in famous:
        row = scores_df[scores_df['symbol'] == symbol]
        if len(row) > 0:
            r = row.iloc[0]
            rev_gr = r.get('revenue_growth_3yr', 0)
            eps_gr = r.get('eps_growth_3yr', 0)
            fcf_m = r.get('fcf_margin', 0)

            print(f"{symbol:<8} {r['moonshot_score']:<8} {r['moonshot_grade']:<6} "
                  f"{rev_gr:+.1%}       {eps_gr:+.1%}       {r.get('gross_margin', 0):+.1%}    {fcf_m:+.1%}")
        else:
            print(f"{symbol:<8} N/A (excluded)")

    # Show top 10 moonshot stocks
    print("\n" + "=" * 80)
    print("TOP 10 MOONSHOT STOCKS (Quality-First v2.0)")
    print("=" * 80)

    top10 = scores_df.nlargest(10, 'raw_score')
    print(f"{'Symbol':<8} {'Score':<8} {'RevGr3yr':<12} {'GrossM':<10} {'FCF%':<10} {'ROE':<10}")
    print("-" * 80)
    for _, r in top10.iterrows():
        print(f"{r['symbol']:<8} {r['moonshot_score']:<8} {r.get('revenue_growth_3yr', 0):+.1%}       "
              f"{r.get('gross_margin', 0):+.1%}    {r.get('fcf_margin', 0):+.1%}    "
              f"{r.get('roe', 0):+.1%}")

    sys.stdout.flush()


def main():
    print("=" * 80)
    print("MOONSHOT SCORE COMPUTATION (Quality-First v2.0)")
    print("Scope: Companies ≥$500M market cap with strong quality filters")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    # Compute scores
    scores_df = compute_all_scores()

    # Update database
    update_nasdaq_db(scores_df)

    # Print examples for verification
    print_examples(scores_df)

    # Summary stats
    print("\n" + "=" * 60)
    print("GRADE DISTRIBUTION")
    print("=" * 60)
    grade_counts = scores_df['moonshot_grade'].value_counts().sort_index()
    for grade, count in grade_counts.items():
        pct = count / len(scores_df) * 100
        print(f"  {grade}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("COMPLETED")
    print("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

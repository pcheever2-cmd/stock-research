#!/usr/bin/env python3
"""
Moonshot Quality-First Sector-Neutral Validation
=================================================
Tests whether Moonshot works WITHIN sectors, not just by picking hot sectors.

Key Question: Is the alpha from stock selection or sector selection?

Methodology:
1. Rank stocks within each sector (not across all sectors)
2. Compute quintile returns within each sector
3. Average across sectors to get sector-neutral performance

If sector-neutral spread is positive, the score has genuine stock selection skill.
If it disappears, the score is just picking hot sectors.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import requests
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000

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

# Sector classifications for major stocks
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'INTC': 'Technology',
    'CRM': 'Technology', 'ADBE': 'Technology', 'ORCL': 'Technology', 'CSCO': 'Technology',
    'AVGO': 'Technology', 'TXN': 'Technology', 'QCOM': 'Technology', 'IBM': 'Technology',
    'NOW': 'Technology', 'INTU': 'Technology', 'AMAT': 'Technology', 'MU': 'Technology',
    'LRCX': 'Technology', 'KLAC': 'Technology', 'SNPS': 'Technology', 'CDNS': 'Technology',
    'MRVL': 'Technology', 'ADI': 'Technology', 'NXPI': 'Technology', 'MCHP': 'Technology',
    'FTNT': 'Technology', 'PANW': 'Technology', 'CRWD': 'Technology', 'ZS': 'Technology',
    'NET': 'Technology', 'DDOG': 'Technology', 'SNOW': 'Technology', 'MDB': 'Technology',
    'PLTR': 'Technology', 'DELL': 'Technology', 'HPE': 'Technology', 'HPQ': 'Technology',
    'SMCI': 'Technology', 'ARM': 'Technology',

    # Financials
    'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials', 'GS': 'Financials',
    'MS': 'Financials', 'C': 'Financials', 'USB': 'Financials', 'PNC': 'Financials',
    'SCHW': 'Financials', 'BLK': 'Financials', 'AXP': 'Financials', 'COF': 'Financials',
    'V': 'Financials', 'MA': 'Financials', 'PYPL': 'Financials',

    # Healthcare
    'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
    'ABBV': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'ABT': 'Healthcare',
    'DHR': 'Healthcare', 'BMY': 'Healthcare', 'AMGN': 'Healthcare', 'GILD': 'Healthcare',
    'VRTX': 'Healthcare', 'REGN': 'Healthcare', 'ISRG': 'Healthcare', 'MDT': 'Healthcare',
    'CVS': 'Healthcare', 'CI': 'Healthcare', 'HUM': 'Healthcare', 'ELV': 'Healthcare',
    'MRNA': 'Healthcare', 'BIIB': 'Healthcare',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'TSLA': 'Consumer Discretionary', 'HD': 'Consumer Discretionary',
    'MCD': 'Consumer Discretionary', 'NKE': 'Consumer Discretionary', 'SBUX': 'Consumer Discretionary',
    'LOW': 'Consumer Discretionary', 'TGT': 'Consumer Discretionary', 'TJX': 'Consumer Discretionary',
    'ROST': 'Consumer Discretionary', 'CMG': 'Consumer Discretionary', 'MAR': 'Consumer Discretionary',
    'GM': 'Consumer Discretionary', 'F': 'Consumer Discretionary', 'ABNB': 'Consumer Discretionary',

    # Consumer Staples
    'WMT': 'Consumer Staples', 'PG': 'Consumer Staples', 'KO': 'Consumer Staples',
    'PEP': 'Consumer Staples', 'COST': 'Consumer Staples', 'PM': 'Consumer Staples',
    'MO': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',
    'MDLZ': 'Consumer Staples', 'GIS': 'Consumer Staples', 'K': 'Consumer Staples',

    # Industrials
    'CAT': 'Industrials', 'DE': 'Industrials', 'UPS': 'Industrials', 'FDX': 'Industrials',
    'HON': 'Industrials', 'UNP': 'Industrials', 'BA': 'Industrials', 'RTX': 'Industrials',
    'LMT': 'Industrials', 'GE': 'Industrials', 'MMM': 'Industrials', 'EMR': 'Industrials',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'SLB': 'Energy',
    'EOG': 'Energy', 'MPC': 'Energy', 'PSX': 'Energy', 'VLO': 'Energy',
    'OXY': 'Energy', 'HAL': 'Energy', 'DVN': 'Energy', 'PXD': 'Energy',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'SHW': 'Materials', 'ECL': 'Materials',
    'NEM': 'Materials', 'FCX': 'Materials', 'NUE': 'Materials', 'DD': 'Materials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'AEP': 'Utilities', 'EXC': 'Utilities', 'SRE': 'Utilities', 'XEL': 'Utilities',

    # Real Estate
    'AMT': 'Real Estate', 'PLD': 'Real Estate', 'CCI': 'Real Estate', 'EQIX': 'Real Estate',
    'PSA': 'Real Estate', 'SPG': 'Real Estate', 'O': 'Real Estate', 'WELL': 'Real Estate',

    # Communication Services
    'DIS': 'Communication Services', 'NFLX': 'Communication Services', 'CMCSA': 'Communication Services',
    'T': 'Communication Services', 'VZ': 'Communication Services', 'TMUS': 'Communication Services',
    'CHTR': 'Communication Services', 'EA': 'Communication Services', 'TTWO': 'Communication Services',
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


def load_and_prepare_data():
    """Load data and compute Moonshot scores."""
    print("=" * 70)
    print("MOONSHOT SECTOR-NEUTRAL VALIDATION")
    print("=" * 70)
    print(f"Run at: {datetime.now().isoformat()}")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Load fundamentals
    print("Loading fundamentals...")
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
    print(f"  {len(fund):,} fundamental records")

    conn.close()

    # Compute TTM metrics
    print("\nComputing metrics...")
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

    # Compute forward returns
    print("Computing forward returns...")
    prices = prices.sort_values(['symbol', 'date'])
    prices['fwd_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252) * 100

    # Merge
    fund['year_month'] = fund['date'].dt.to_period('M')
    prices['year_month'] = prices['date'].dt.to_period('M')
    month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

    df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_1yr']],
                    on=['symbol', 'year_month'], how='inner')
    df = df.dropna(subset=['fwd_1yr', 'market_cap'])
    df = df[df['market_cap'] >= MIN_MARKET_CAP].copy()

    # Apply quality filters
    print("Applying quality filters...")
    df['passes_quality'] = df.apply(passes_quality_filters, axis=1)
    df = df[df['passes_quality']].copy()
    print(f"  {len(df):,} stock-months passed filters")

    # Compute momentum
    prices_sorted = prices.sort_values(['symbol', 'date'])
    prices_sorted['price_12m'] = prices_sorted.groupby('symbol')['close'].shift(252)
    prices_sorted['price_1m'] = prices_sorted.groupby('symbol')['close'].shift(21)
    prices_sorted['momentum_12_1'] = (prices_sorted['price_1m'] - prices_sorted['price_12m']) / prices_sorted['price_12m']

    month_end_momentum = prices_sorted.groupby(['symbol', 'year_month']).last().reset_index()
    df = df.merge(month_end_momentum[['symbol', 'year_month', 'momentum_12_1']],
                  on=['symbol', 'year_month'], how='left')

    # Fill missing and cap
    for factor in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
        df[factor] = df[factor].fillna(0)

    df['revenue_growth_3yr'] = df['revenue_growth_3yr'].clip(-0.3, 1.5)
    df['eps_growth_3yr'] = df['eps_growth_3yr'].clip(-0.5, 2.0)
    df['fcf_margin'] = df['fcf_margin'].clip(-0.5, 0.5)
    df['roe'] = df['roe'].clip(-0.5, 1.0)
    df['momentum_12_1'] = df['momentum_12_1'].clip(-0.5, 2.0)

    df['small_cap'] = -np.log10(df['market_cap'] / 1e9)
    df = df.dropna(subset=['revenue_growth_3yr', 'gross_margin'])

    # Add sector
    df['sector'] = df['symbol'].map(SECTOR_MAP)

    # Split IS/OOS
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    # Compute z-score stats from IS
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }

    # Apply z-scores and compute score
    for period_df in [is_df, oos_df]:
        for factor in WEIGHTS.keys():
            period_df[f'{factor}_z'] = (
                (period_df[factor] - factor_stats[factor]['mean']) /
                factor_stats[factor]['std']
            )
        period_df['moonshot_score'] = sum(
            period_df[f'{factor}_z'] * weight
            for factor, weight in WEIGHTS.items()
        )

    return is_df, oos_df


def sector_neutral_analysis(df, period_name):
    """Compute sector-neutral quintile returns."""
    print(f"\n{'=' * 70}")
    print(f"SECTOR-NEUTRAL ANALYSIS ({period_name})")
    print("=" * 70)

    # Only include stocks with known sectors
    df_with_sector = df[df['sector'].notna()].copy()
    print(f"Stocks with sector classification: {len(df_with_sector):,} ({df_with_sector['symbol'].nunique()} unique)")

    if len(df_with_sector) < 100:
        print("  Insufficient data for sector analysis")
        return None

    # Rank within each sector-month
    df_with_sector['sector_quintile'] = df_with_sector.groupby(['year_month', 'sector'])['moonshot_score'].transform(
        lambda x: pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop') if len(x) >= 5 else np.nan
    )

    df_with_sector = df_with_sector.dropna(subset=['sector_quintile'])

    # Compute average returns by sector-quintile
    print("\nSector-Neutral Quintile Returns:")
    print(f"{'Quintile':<10} {'Avg Return':<12} {'Count':<10}")
    print("-" * 35)

    q_returns = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = df_with_sector[df_with_sector['sector_quintile'] == q]
        if len(q_data) > 0:
            avg_ret = q_data['fwd_1yr'].mean()
            q_returns[q] = avg_ret
            print(f"{q:<10} {avg_ret:>+10.2f}%  {len(q_data):>8,}")

    if 'Q5' in q_returns and 'Q1' in q_returns:
        spread = q_returns['Q5'] - q_returns['Q1']
        print("-" * 35)
        print(f"{'Q5-Q1':<10} {spread:>+10.2f}%")

        # T-test
        q5_returns = df_with_sector[df_with_sector['sector_quintile'] == 'Q5']['fwd_1yr']
        q1_returns = df_with_sector[df_with_sector['sector_quintile'] == 'Q1']['fwd_1yr']
        from scipy import stats
        t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns)
        print(f"{'t-stat':<10} {t_stat:>+10.2f}")
        print(f"{'p-value':<10} {p_val:>10.4f}")

    # By sector breakdown
    print(f"\n{'Sector':<25} {'Q5-Q1 Spread':<15} {'Stocks':<10}")
    print("-" * 55)

    sector_spreads = []
    for sector in sorted(df_with_sector['sector'].unique()):
        sector_data = df_with_sector[df_with_sector['sector'] == sector]
        if len(sector_data) >= 50:
            q5_ret = sector_data[sector_data['sector_quintile'] == 'Q5']['fwd_1yr'].mean()
            q1_ret = sector_data[sector_data['sector_quintile'] == 'Q1']['fwd_1yr'].mean()
            if not pd.isna(q5_ret) and not pd.isna(q1_ret):
                spread = q5_ret - q1_ret
                sector_spreads.append(spread)
                n_stocks = sector_data['symbol'].nunique()
                print(f"{sector:<25} {spread:>+13.2f}%  {n_stocks:>8}")

    if sector_spreads:
        avg_sector_spread = np.mean(sector_spreads)
        print("-" * 55)
        print(f"{'Average Sector Spread':<25} {avg_sector_spread:>+13.2f}%")
        print(f"{'Sectors with Positive':<25} {sum(1 for s in sector_spreads if s > 0)}/{len(sector_spreads)}")

    return q_returns


def compare_overall_vs_sector_neutral(df, period_name):
    """Compare overall quintile analysis to sector-neutral."""
    print(f"\n{'=' * 70}")
    print(f"COMPARISON: OVERALL vs SECTOR-NEUTRAL ({period_name})")
    print("=" * 70)

    # Overall quintiles
    df['overall_quintile'] = pd.qcut(df['moonshot_score'], 5,
                                     labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                     duplicates='drop')

    overall_q5 = df[df['overall_quintile'] == 'Q5']['fwd_1yr'].mean()
    overall_q1 = df[df['overall_quintile'] == 'Q1']['fwd_1yr'].mean()
    overall_spread = overall_q5 - overall_q1

    # Sector-neutral (only for stocks with sectors)
    df_with_sector = df[df['sector'].notna()].copy()
    df_with_sector['sector_quintile'] = df_with_sector.groupby(['year_month', 'sector'])['moonshot_score'].transform(
        lambda x: pd.qcut(x, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop') if len(x) >= 5 else np.nan
    )
    df_with_sector = df_with_sector.dropna(subset=['sector_quintile'])

    sn_q5 = df_with_sector[df_with_sector['sector_quintile'] == 'Q5']['fwd_1yr'].mean()
    sn_q1 = df_with_sector[df_with_sector['sector_quintile'] == 'Q1']['fwd_1yr'].mean()
    sn_spread = sn_q5 - sn_q1

    print(f"\n{'Method':<25} {'Q5-Q1 Spread':<15}")
    print("-" * 45)
    print(f"{'Overall (cross-sectional)':<25} {overall_spread:>+13.2f}%")
    print(f"{'Sector-Neutral':<25} {sn_spread:>+13.2f}%")
    print("-" * 45)
    print(f"{'Difference':<25} {sn_spread - overall_spread:>+13.2f}%")

    if sn_spread > 0:
        retention = (sn_spread / overall_spread * 100) if overall_spread != 0 else 0
        print(f"\nSector-Neutral retains {retention:.0f}% of overall spread")
        if retention >= 50:
            print("=> PASS: Moonshot has genuine stock selection skill within sectors")
        else:
            print("=> CAUTION: Much of the spread may be from sector selection")
    else:
        print("\n=> WARNING: Sector-neutral spread is negative - score may just be picking hot sectors")


def main():
    is_df, oos_df = load_and_prepare_data()

    # In-sample analysis
    sector_neutral_analysis(is_df, "IN-SAMPLE (1995-2019)")
    compare_overall_vs_sector_neutral(is_df, "IN-SAMPLE")

    # Out-of-sample analysis
    sector_neutral_analysis(oos_df, "OUT-OF-SAMPLE (2020-2026)")
    compare_overall_vs_sector_neutral(oos_df, "OUT-OF-SAMPLE")

    print("\n" + "=" * 70)
    print("SECTOR-NEUTRAL VALIDATION COMPLETE")
    print("=" * 70)
    sys.stdout.flush()


if __name__ == '__main__':
    main()

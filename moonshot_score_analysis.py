#!/usr/bin/env python3
"""
Moonshot Score Analysis
=======================
Identify factors that predict 3x (300%) returns in speculative stocks.

Entry Criteria (Relaxed):
- Stock price < $20
- Market cap < $1B

Time Horizons:
- 3-year (756 trading days)
- 5-year (1260 trading days)

Methodology:
1. Find all stocks meeting entry criteria that achieved 3x returns
2. Analyze what factors predicted these winners BEFORE they ran
3. Build "Moonshot Score" using predictive factors
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

# Entry criteria (relaxed)
MAX_PRICE = 20.0       # Stock price < $20
MAX_MARKET_CAP = 1000  # Market cap < $1B (in millions)

# Target return (relaxed)
TARGET_RETURN = 300    # 3x = 300% gain

# Horizons (trading days)
HORIZONS = {
    '3-Year': 756,
    '5-Year': 1260
}

# Analysis period: look for winners from 2010-2020
# (gives us time to see 3-5 year returns)
ANALYSIS_START = '2010-01-01'
ANALYSIS_END = '2020-01-01'


def load_data():
    """Load price and fundamental data."""
    print("Loading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    # Prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    # Fundamentals with market cap
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income, i.revenue,
               b.total_assets, b.total_debt, b.cash_and_equivalents,
               c.operating_cash_flow, c.free_cash_flow, c.capital_expenditure,
               m.market_cap, m.pe_ratio
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()
    return prices, fund


def find_moonshot_candidates(prices, fund):
    """Find stocks meeting entry criteria: price < $10 AND market cap < $500M."""
    print(f"\nFinding moonshot candidates (price < ${MAX_PRICE}, mcap < ${MAX_MARKET_CAP}M)...")

    # Get market cap from fundamentals
    mcap_data = fund[['symbol', 'date', 'market_cap']].dropna()
    mcap_data = mcap_data[mcap_data['market_cap'] > 0]

    # Convert to millions
    mcap_data['market_cap_m'] = mcap_data['market_cap'] / 1_000_000

    # Get price data
    price_check = prices[['symbol', 'date', 'close']].copy()

    # Merge
    merged = pd.merge(price_check, mcap_data, on=['symbol', 'date'], how='inner')

    # Apply filters
    candidates = merged[
        (merged['close'] < MAX_PRICE) &
        (merged['close'] > 0.5) &  # Avoid penny stocks < $0.50
        (merged['market_cap_m'] < MAX_MARKET_CAP) &
        (merged['market_cap_m'] > 10)  # Avoid ultra-micro (>$10M mcap)
    ]

    print(f"  {len(candidates):,} candidate observations")
    print(f"  {candidates['symbol'].nunique():,} unique symbols")

    return candidates


def compute_forward_returns(prices, candidates, horizons):
    """Compute forward returns for each horizon."""
    print("\nComputing forward returns for moonshot candidates...")

    results = []
    symbols = candidates['symbol'].unique()

    for i, symbol in enumerate(symbols):
        if i % 200 == 0:
            print(f"  {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        sym_candidates = candidates[candidates['symbol'] == symbol]

        if len(sym_prices) < 100:
            continue

        close = sym_prices['close'].values
        dates = pd.to_datetime(sym_prices['date'].values)
        n = len(close)

        # Create date-to-index mapping
        date_to_idx = {d: i for i, d in enumerate(dates)}

        for _, cand_row in sym_candidates.iterrows():
            cand_date = pd.Timestamp(cand_row['date'])

            if cand_date not in date_to_idx:
                continue

            j = date_to_idx[cand_date]

            # Check if within analysis period
            if cand_date < pd.Timestamp(ANALYSIS_START) or cand_date > pd.Timestamp(ANALYSIS_END):
                continue

            row = {
                'symbol': symbol,
                'date': cand_date,
                'entry_price': cand_row['close'],
                'market_cap_m': cand_row['market_cap_m'],
            }

            for horizon_name, days in horizons.items():
                if j + days < n:
                    exit_price = close[j + days]
                    fwd_ret = ((exit_price / close[j]) - 1) * 100
                    row[f'fwd_{horizon_name}'] = fwd_ret
                    row[f'is_3x_{horizon_name}'] = fwd_ret >= TARGET_RETURN
                else:
                    row[f'fwd_{horizon_name}'] = np.nan
                    row[f'is_3x_{horizon_name}'] = np.nan

            results.append(row)

    df = pd.DataFrame(results)
    print(f"  {len(df):,} candidate-date observations")
    return df


def compute_predictive_factors(fund, candidates_df):
    """Compute factors that might predict 3x winners."""
    print("\nComputing predictive factors...")

    # Calculate fundamental factors
    fund = fund.copy()

    # Growth factors
    fund = fund.sort_values(['symbol', 'date'])
    fund['revenue_growth'] = fund.groupby('symbol')['revenue'].pct_change(4) * 100  # YoY growth

    # Profitability
    fund['gross_margin'] = (fund['gross_profit'] / fund['revenue']) * 100
    fund['roa'] = (fund['net_income'] / fund['total_assets']) * 100
    fund['ocf_margin'] = (fund['operating_cash_flow'] / fund['revenue']) * 100

    # Balance sheet
    fund['debt_to_assets'] = fund['total_debt'] / fund['total_assets']
    fund['cash_ratio'] = fund['cash_and_equivalents'] / fund['total_assets']

    # Valuation - calculate P/S from market cap and revenue
    fund['price_to_sales'] = fund['market_cap'] / fund['revenue']

    # R&D proxy (capex as % of revenue - proxy for investment intensity)
    fund['capex_intensity'] = abs(fund['capital_expenditure']) / fund['revenue'] * 100

    # Size
    fund['log_mcap'] = np.log10(fund['market_cap'] / 1e6 + 1)  # Log market cap in millions

    # Clean
    for col in ['revenue_growth', 'gross_margin', 'roa', 'ocf_margin', 'debt_to_assets',
                'cash_ratio', 'price_to_sales', 'capex_intensity', 'log_mcap']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Merge with candidates
    factor_cols = ['symbol', 'date', 'revenue_growth', 'gross_margin', 'roa', 'ocf_margin',
                   'debt_to_assets', 'cash_ratio', 'price_to_sales', 'capex_intensity', 'log_mcap']
    fund_factors = fund[factor_cols].copy()

    # Merge using backward-looking (no look-ahead)
    all_merged = []
    for symbol in candidates_df['symbol'].unique():
        cdf = candidates_df[candidates_df['symbol'] == symbol].sort_values('date')
        fdf = fund_factors[fund_factors['symbol'] == symbol].sort_values('date')

        if len(fdf) == 0:
            continue

        merged = pd.merge_asof(cdf, fdf.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    print(f"  {len(df):,} observations with factors")
    return df


def analyze_factor_predictive_power(df, horizon_name):
    """Analyze which factors predict 3x winners for a given horizon."""
    return_col = f'fwd_{horizon_name}'
    is_3x_col = f'is_3x_{horizon_name}'

    clean = df.dropna(subset=[return_col, is_3x_col])

    if len(clean) < 100:
        return None

    winners = clean[clean[is_3x_col] == True]
    losers = clean[clean[is_3x_col] == False]

    print(f"\n{horizon_name} Analysis:")
    print(f"  Total observations: {len(clean):,}")
    print(f"  3x Winners: {len(winners):,} ({100*len(winners)/len(clean):.2f}%)")
    print(f"  Non-winners: {len(losers):,}")

    if len(winners) < 10:
        print("  Too few winners to analyze")
        return None

    # Compare factor distributions between winners and losers
    factor_cols = ['revenue_growth', 'gross_margin', 'roa', 'ocf_margin',
                   'debt_to_assets', 'cash_ratio', 'price_to_sales', 'capex_intensity',
                   'log_mcap', 'entry_price', 'market_cap_m']

    results = []
    print(f"\n  {'Factor':<18} {'Winner Mean':>12} {'Loser Mean':>12} {'Difference':>12} {'T-stat':>10}")
    print("  " + "-" * 70)

    for factor in factor_cols:
        winner_vals = winners[factor].dropna()
        loser_vals = losers[factor].dropna()

        if len(winner_vals) < 5 or len(loser_vals) < 5:
            continue

        winner_mean = winner_vals.mean()
        loser_mean = loser_vals.mean()
        diff = winner_mean - loser_mean

        # Simple t-stat (Welch's)
        from scipy import stats
        try:
            t_stat, p_val = stats.ttest_ind(winner_vals, loser_vals, equal_var=False)
        except:
            t_stat = np.nan

        results.append({
            'factor': factor,
            'winner_mean': winner_mean,
            'loser_mean': loser_mean,
            'difference': diff,
            't_stat': t_stat
        })

        sig = "**" if abs(t_stat) > 2 else "*" if abs(t_stat) > 1.5 else ""
        print(f"  {factor:<18} {winner_mean:>+11.2f} {loser_mean:>+11.2f} {diff:>+11.2f} {t_stat:>+9.2f}{sig}")

    return pd.DataFrame(results)


def build_moonshot_score(df, significant_factors, horizon_name):
    """Build a Moonshot Score based on significant predictive factors."""
    return_col = f'fwd_{horizon_name}'
    is_3x_col = f'is_3x_{horizon_name}'

    clean = df.dropna(subset=[return_col, is_3x_col])

    if len(significant_factors) == 0:
        print("\nNo significant factors found for Moonshot Score")
        return None

    # Use top factors with t-stat > 1.5
    top_factors = significant_factors[abs(significant_factors['t_stat']) > 1.5].copy()

    if len(top_factors) == 0:
        print("\nNo factors with t-stat > 1.5")
        return None

    print(f"\n{'='*70}")
    print(f"MOONSHOT SCORE CONSTRUCTION ({horizon_name})")
    print(f"{'='*70}")
    print(f"\nUsing {len(top_factors)} significant factors:")

    # Determine direction (positive or negative relationship)
    for _, row in top_factors.iterrows():
        direction = "+" if row['difference'] > 0 else "-"
        print(f"  {direction} {row['factor']}: diff={row['difference']:+.2f}, t={row['t_stat']:+.2f}")

    # Z-score each factor and combine
    for factor in top_factors['factor']:
        if factor in clean.columns:
            mean = clean[factor].mean()
            std = clean[factor].std()
            clean[f'{factor}_z'] = ((clean[factor] - mean) / std).clip(-3, 3).fillna(0)

    # Compute score (equal weight for now)
    clean['moonshot_score'] = 0
    for _, row in top_factors.iterrows():
        factor = row['factor']
        z_col = f'{factor}_z'
        if z_col in clean.columns:
            # Flip sign if difference is negative
            sign = 1 if row['difference'] > 0 else -1
            clean['moonshot_score'] += sign * clean[z_col] / len(top_factors)

    # Evaluate
    clean['moonshot_quintile'] = pd.qcut(clean['moonshot_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')

    print(f"\nMoonshot Score Quintile Performance:")
    print(f"{'Quintile':<10} {'Avg Return':>12} {'% 3x Winners':>15} {'Count':>10}")
    print("-" * 50)

    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = clean[clean['moonshot_quintile'] == q]
        avg_ret = q_data[return_col].mean()
        pct_3x = 100 * q_data[is_3x_col].mean() if len(q_data) > 0 else 0
        count = len(q_data)
        print(f"{q:<10} {avg_ret:>+11.2f}% {pct_3x:>14.2f}% {count:>10,}")

    q5_3x = clean[clean['moonshot_quintile'] == 'Q5'][is_3x_col].mean() * 100
    q1_3x = clean[clean['moonshot_quintile'] == 'Q1'][is_3x_col].mean() * 100
    spread = q5_3x - q1_3x

    print(f"\nQ5 3x Rate: {q5_3x:.2f}%")
    print(f"Q1 3x Rate: {q1_3x:.2f}%")
    print(f"Spread: {spread:+.2f} percentage points")

    return clean


def main():
    print("=" * 70)
    print("MOONSHOT SCORE ANALYSIS")
    print(f"Finding predictors of 3x returns in small stocks")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nEntry Criteria:")
    print(f"  Stock price: < ${MAX_PRICE}")
    print(f"  Market cap: < ${MAX_MARKET_CAP}M")
    print(f"\nTarget: {TARGET_RETURN}% returns (3x)")
    print(f"Horizons: {', '.join(HORIZONS.keys())}")
    print(f"Analysis period: {ANALYSIS_START} to {ANALYSIS_END}")

    # Load data
    prices, fund = load_data()

    # Find candidates
    candidates = find_moonshot_candidates(prices, fund)

    # Compute forward returns
    candidates_with_returns = compute_forward_returns(prices, candidates, HORIZONS)

    # Count winners
    print("\n" + "=" * 70)
    print("3x WINNER COUNTS")
    print("=" * 70)

    for horizon_name in HORIZONS.keys():
        col = f'is_3x_{horizon_name}'
        clean = candidates_with_returns.dropna(subset=[col])
        n_winners = clean[col].sum()
        total = len(clean)
        pct = 100 * n_winners / total if total > 0 else 0
        print(f"  {horizon_name}: {int(n_winners):,} winners out of {total:,} ({pct:.2f}%)")

    # Compute predictive factors
    df = compute_predictive_factors(fund, candidates_with_returns)

    # Analyze each horizon
    all_results = {}
    for horizon_name in HORIZONS.keys():
        print("\n" + "=" * 70)
        print(f"FACTOR ANALYSIS: {horizon_name}")
        print("=" * 70)

        results = analyze_factor_predictive_power(df, horizon_name)
        if results is not None:
            all_results[horizon_name] = results

            # Build Moonshot Score
            scored_df = build_moonshot_score(df, results, horizon_name)

    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH PAPER")
    print("=" * 70)

    print("\n| Factor | 3-Year Impact | 5-Year Impact | Interpretation |")
    print("|--------|---------------|---------------|----------------|")

    factor_names = {
        'revenue_growth': 'Revenue Growth',
        'gross_margin': 'Gross Margin',
        'roa': 'ROA',
        'ocf_margin': 'OCF Margin',
        'debt_to_assets': 'Debt/Assets',
        'cash_ratio': 'Cash Ratio',
        'price_to_sales': 'P/S Ratio',
        'capex_intensity': 'Capex/Revenue',
        'log_mcap': 'Log Market Cap',
        'entry_price': 'Entry Price',
        'market_cap_m': 'Market Cap ($M)'
    }

    for factor in factor_names.keys():
        impact_3y = ""
        impact_5y = ""

        if '3-Year' in all_results:
            row = all_results['3-Year'][all_results['3-Year']['factor'] == factor]
            if len(row) > 0:
                t = row.iloc[0]['t_stat']
                impact_3y = f"t={t:+.2f}" if not np.isnan(t) else "-"

        if '5-Year' in all_results:
            row = all_results['5-Year'][all_results['5-Year']['factor'] == factor]
            if len(row) > 0:
                t = row.iloc[0]['t_stat']
                impact_5y = f"t={t:+.2f}" if not np.isnan(t) else "-"

        print(f"| {factor_names[factor]:<14} | {impact_3y:>13} | {impact_5y:>13} | |")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

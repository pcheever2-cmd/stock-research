#!/usr/bin/env python3
"""
Fast Factor Testing Framework
=============================
Uses vectorized pandas operations for speed.
Tests all factors in ~5-10 minutes on full dataset.
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


def load_and_compute_price_factors(sample_symbols: int = None) -> pd.DataFrame:
    """Load prices and compute all price-based factors using vectorized ops."""
    print("Loading price data...")
    conn = sqlite3.connect(BACKTEST_DB)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)

    print(f"  Loaded {len(prices):,} price records")

    # Sample if requested
    if sample_symbols:
        symbols = prices['symbol'].unique()
        np.random.seed(42)
        keep_symbols = np.random.choice(symbols, size=min(sample_symbols, len(symbols)), replace=False)
        prices = prices[prices['symbol'].isin(keep_symbols)]
        print(f"  Sampled {len(keep_symbols)} symbols")

    # Convert to datetime
    prices['date'] = pd.to_datetime(prices['date'])

    print("\nComputing price-based factors (vectorized)...")

    # Group by symbol and compute rolling metrics
    def compute_factors(group):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)

        if n < 300:
            return pd.DataFrame()

        close = group['close'].values
        volume = group['volume'].values

        # Pre-allocate arrays
        results = []

        # Sample monthly (every 21 trading days)
        for j in range(252, n - 63, 21):
            # Forward returns
            fwd_1m = ((close[j + 21] / close[j]) - 1) * 100 if j + 21 < n else np.nan
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100 if j + 63 < n else np.nan

            # Momentum factors
            mom_1m = ((close[j] / close[j - 21]) - 1) * 100 if close[j - 21] > 0 else np.nan
            mom_3m = ((close[j] / close[j - 63]) - 1) * 100 if close[j - 63] > 0 else np.nan
            mom_6m = ((close[j] / close[j - 126]) - 1) * 100 if j >= 126 and close[j - 126] > 0 else np.nan
            mom_12m = ((close[j] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # 52-week high
            high_52w = np.max(close[max(0, j - 252):j])
            pct_from_high = ((close[j] / high_52w) - 1) * 100 if high_52w > 0 else np.nan

            # Volatility
            returns_20d = np.diff(close[j - 21:j + 1]) / close[j - 21:j]
            vol_20d = np.std(returns_20d) * np.sqrt(252) * 100 if len(returns_20d) > 5 else np.nan

            returns_60d = np.diff(close[j - 63:j + 1]) / close[j - 63:j]
            vol_60d = np.std(returns_60d) * np.sqrt(252) * 100 if len(returns_60d) > 20 else np.nan

            # Volume factors
            avg_vol_20d = np.mean(volume[j - 21:j])
            avg_vol_60d = np.mean(volume[j - 63:j])
            vol_ratio = avg_vol_20d / avg_vol_60d if avg_vol_60d > 0 else np.nan
            dollar_vol = close[j] * avg_vol_20d / 1e6

            # Trend
            sma_50 = np.mean(close[j - 50:j]) if j >= 50 else np.nan
            sma_200 = np.mean(close[j - 200:j]) if j >= 200 else np.nan

            price_vs_sma50 = ((close[j] / sma_50) - 1) * 100 if sma_50 > 0 else np.nan
            price_vs_sma200 = ((close[j] / sma_200) - 1) * 100 if sma_200 > 0 else np.nan
            trend_strength = 1 if (sma_50 and sma_200 and sma_50 > sma_200 and close[j] > sma_50) else 0

            # RSI
            gains, losses = [], []
            for k in range(j - 14, j):
                change = close[k + 1] - close[k]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rsi = 100 - (100 / (1 + avg_gain / avg_loss))

            results.append({
                'symbol': group['symbol'].iloc[0],
                'date': group['date'].iloc[j],
                'close': close[j],
                'fwd_1m': fwd_1m,
                'fwd_3m': fwd_3m,
                'mom_1m': mom_1m,
                'mom_3m': mom_3m,
                'mom_6m': mom_6m,
                'mom_12m': mom_12m,
                'mom_12_1': mom_12_1,
                'pct_from_high': pct_from_high,
                'vol_20d': vol_20d,
                'vol_60d': vol_60d,
                'vol_ratio': vol_ratio,
                'dollar_vol': dollar_vol,
                'price_vs_sma50': price_vs_sma50,
                'price_vs_sma200': price_vs_sma200,
                'trend_strength': trend_strength,
                'rsi': rsi,
            })

        return pd.DataFrame(results)

    # Process in chunks for memory efficiency
    symbols = prices['symbol'].unique()
    all_results = []

    for i in range(0, len(symbols), 100):
        batch_symbols = symbols[i:i + 100]
        batch = prices[prices['symbol'].isin(batch_symbols)]

        batch_results = batch.groupby('symbol', group_keys=False).apply(compute_factors)
        all_results.append(batch_results)

        if (i + 100) % 500 == 0:
            print(f"  Processed {min(i + 100, len(symbols))}/{len(symbols)} symbols...")

    price_factors = pd.concat(all_results, ignore_index=True)
    conn.close()

    print(f"  Generated {len(price_factors):,} observations")
    return price_factors


def load_fundamentals() -> pd.DataFrame:
    """Load and merge all fundamental data."""
    print("\nLoading fundamental data...")
    conn = sqlite3.connect(BACKTEST_DB)

    income = pd.read_sql_query("""
        SELECT symbol, date, revenue, gross_profit, operating_income,
               net_income, ebitda, eps, weighted_avg_shares_diluted
        FROM historical_income_statements
    """, conn)

    balance = pd.read_sql_query("""
        SELECT symbol, date, total_assets, total_liabilities, total_equity,
               total_debt, cash_and_equivalents
        FROM historical_balance_sheets
    """, conn)

    cashflow = pd.read_sql_query("""
        SELECT symbol, date, operating_cash_flow, free_cash_flow
        FROM historical_cash_flows
    """, conn)

    metrics = pd.read_sql_query("""
        SELECT symbol, date, ev_to_ebitda, market_cap, pe_ratio, pb_ratio,
               debt_to_equity, roe
        FROM historical_key_metrics
    """, conn)

    conn.close()

    # Merge all
    fund = income.merge(balance, on=['symbol', 'date'], how='outer')
    fund = fund.merge(cashflow, on=['symbol', 'date'], how='outer')
    fund = fund.merge(metrics, on=['symbol', 'date'], how='outer')

    # Convert date
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'date'])

    # Compute derived factors
    print("  Computing derived fundamental factors...")

    # Profitability
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['op_margin'] = fund['operating_income'] / fund['revenue']
    fund['net_margin'] = fund['net_income'] / fund['revenue']

    # Cash flow quality
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['accruals'] = (fund['net_income'] - fund['operating_cash_flow']) / fund['total_assets']

    # FCF yield
    fund['fcf_yield'] = fund['free_cash_flow'] / fund['market_cap']

    # Leverage
    fund['debt_to_assets'] = fund['total_debt'] / fund['total_assets']

    # Growth (YoY)
    fund['rev_growth'] = fund.groupby('symbol')['revenue'].pct_change(4) * 100
    fund['eps_growth'] = fund.groupby('symbol')['eps'].pct_change(4) * 100
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4) * 100

    # Valuation
    fund['earnings_yield'] = 1 / fund['pe_ratio'] * 100

    print(f"  Loaded {len(fund):,} fundamental records")
    return fund


def merge_factors(price_factors: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Merge price factors with most recent fundamentals."""
    print("\nMerging price and fundamental factors...")

    # For each price observation, get the most recent fundamentals
    price_factors = price_factors.sort_values(['symbol', 'date']).reset_index(drop=True)
    fundamentals = fundamentals.sort_values(['symbol', 'date']).reset_index(drop=True)

    # Merge symbol by symbol to avoid sorting issues
    all_merged = []

    symbols = price_factors['symbol'].unique()
    for i, symbol in enumerate(symbols):
        if i % 100 == 0:
            print(f"  Merging {i}/{len(symbols)} symbols...")

        pf = price_factors[price_factors['symbol'] == symbol].sort_values('date')
        fund = fundamentals[fundamentals['symbol'] == symbol].sort_values('date')

        if len(fund) == 0:
            all_merged.append(pf)
            continue

        # Use merge_asof per symbol
        merged_sym = pd.merge_asof(
            pf,
            fund.drop(columns=['symbol']),
            on='date',
            direction='backward',
            tolerance=pd.Timedelta('365 days')
        )
        all_merged.append(merged_sym)

    merged = pd.concat(all_merged, ignore_index=True)

    print(f"  Merged dataset: {len(merged):,} observations")
    return merged


def load_analyst_factors(price_factors: pd.DataFrame) -> pd.DataFrame:
    """Add analyst-related factors."""
    print("\nLoading analyst factors...")
    conn = sqlite3.connect(BACKTEST_DB)

    grades = pd.read_sql_query("""
        SELECT symbol, date, action
        FROM historical_grades
        WHERE action IN ('upgrade', 'downgrade')
    """, conn)
    conn.close()

    grades['date'] = pd.to_datetime(grades['date'])

    # Compute analyst momentum for each observation
    analyst_data = []

    for symbol in price_factors['symbol'].unique():
        sym_prices = price_factors[price_factors['symbol'] == symbol]
        sym_grades = grades[grades['symbol'] == symbol]

        for _, row in sym_prices.iterrows():
            obs_date = row['date']
            lookback = obs_date - pd.Timedelta(days=90)

            recent = sym_grades[(sym_grades['date'] >= lookback) & (sym_grades['date'] < obs_date)]

            upgrades = (recent['action'] == 'upgrade').sum()
            downgrades = (recent['action'] == 'downgrade').sum()
            total = upgrades + downgrades

            analyst_data.append({
                'symbol': symbol,
                'date': obs_date,
                'upgrades_90d': upgrades,
                'downgrades_90d': downgrades,
                'upgrade_ratio': upgrades / total if total > 0 else 0.5,
                'analyst_momentum': upgrades - downgrades,
            })

    analyst_df = pd.DataFrame(analyst_data)
    merged = price_factors.merge(analyst_df, on=['symbol', 'date'], how='left')

    print(f"  Added analyst factors")
    return merged


def test_all_factors(df: pd.DataFrame, return_col: str = 'fwd_3m') -> pd.DataFrame:
    """Test correlation of each factor with forward returns."""
    print(f"\n{'='*70}")
    print(f"FACTOR PREDICTIVE POWER ANALYSIS (vs {return_col})")
    print('='*70)

    # All potential factors
    factor_columns = [
        # Momentum
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1', 'pct_from_high',
        # Volatility
        'vol_20d', 'vol_60d', 'vol_ratio',
        # Volume
        'dollar_vol',
        # Trend
        'price_vs_sma50', 'price_vs_sma200', 'trend_strength', 'rsi',
        # Profitability
        'gp_assets', 'roe', 'roa', 'op_margin', 'net_margin',
        # Cash flow
        'ocf_assets', 'fcf_assets', 'accruals', 'fcf_yield',
        # Valuation
        'ev_to_ebitda', 'pe_ratio', 'pb_ratio', 'earnings_yield',
        # Leverage
        'debt_to_equity', 'debt_to_assets',
        # Growth
        'rev_growth', 'eps_growth', 'asset_growth',
        # Size
        'market_cap',
        # Analyst
        'upgrades_90d', 'downgrades_90d', 'upgrade_ratio', 'analyst_momentum',
    ]

    results = []

    for factor in factor_columns:
        if factor not in df.columns:
            continue

        clean = df.dropna(subset=[factor, return_col]).copy()

        if len(clean) < 1000:
            continue

        # Winsorize extreme values
        lower = clean[factor].quantile(0.01)
        upper = clean[factor].quantile(0.99)
        clean[factor] = clean[factor].clip(lower, upper)

        # Correlation
        corr = clean[factor].corr(clean[return_col])

        # Q5-Q1 spread
        try:
            q5_thresh = clean[factor].quantile(0.8)
            q1_thresh = clean[factor].quantile(0.2)
            q5_ret = clean[clean[factor] >= q5_thresh][return_col].mean()
            q1_ret = clean[clean[factor] <= q1_thresh][return_col].mean()
            spread = q5_ret - q1_ret
        except:
            spread = np.nan

        # Monotonicity score (are quintile returns in order?)
        try:
            quintiles = pd.qcut(clean[factor], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
            quintile_means = clean.groupby(quintiles, observed=False)[return_col].mean()
            monotonic = (quintile_means.diff().dropna() > 0).sum() / 4  # 0 to 1 score
        except:
            monotonic = np.nan

        # T-statistic
        n = len(clean)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0

        results.append({
            'factor': factor,
            'correlation': corr,
            'q5_q1_spread': spread,
            'monotonicity': monotonic,
            't_statistic': t_stat,
            'n_obs': n,
        })

    results_df = pd.DataFrame(results)
    results_df['abs_corr'] = results_df['correlation'].abs()
    results_df = results_df.sort_values('abs_corr', ascending=False)

    # Print results
    print(f"\n{'Factor':<22} {'Corr':>10} {'Spread':>10} {'Mono':>8} {'T-Stat':>10} {'N':>12}")
    print('-' * 78)

    for _, row in results_df.iterrows():
        mono_str = f"{row['monotonicity']:.2f}" if not pd.isna(row['monotonicity']) else 'N/A'
        sig = '***' if abs(row['t_statistic']) > 3 else '**' if abs(row['t_statistic']) > 2 else '*' if abs(row['t_statistic']) > 1.5 else ''
        print(f"{row['factor']:<22} {row['correlation']:>+9.4f} {row['q5_q1_spread']:>+9.2f}% {mono_str:>8} {row['t_statistic']:>+9.1f} {row['n_obs']:>10,} {sig}")

    return results_df


def test_factor_interactions(df: pd.DataFrame, return_col: str = 'fwd_3m') -> pd.DataFrame:
    """Test interactions between top factors."""
    print(f"\n{'='*70}")
    print("FACTOR INTERACTION ANALYSIS")
    print('='*70)

    # Key factor pairs to test
    interactions = [
        ('mom_12_1', 'gp_assets', 'Momentum x Profitability'),
        ('mom_12_1', 'vol_20d', 'Momentum x Volatility'),
        ('mom_12_1', 'ev_to_ebitda', 'Momentum x Valuation'),
        ('gp_assets', 'ev_to_ebitda', 'Profitability x Valuation'),
        ('gp_assets', 'vol_20d', 'Profitability x Volatility'),
        ('mom_12_1', 'accruals', 'Momentum x Earnings Quality'),
        ('roa', 'asset_growth', 'ROA x Asset Growth'),
        ('fcf_yield', 'mom_12_1', 'FCF Yield x Momentum'),
    ]

    results = []

    for f1, f2, name in interactions:
        if f1 not in df.columns or f2 not in df.columns:
            continue

        clean = df.dropna(subset=[f1, f2, return_col]).copy()

        if len(clean) < 1000:
            continue

        # Create high/low groups
        f1_median = clean[f1].median()
        f2_median = clean[f2].median()

        clean['f1_high'] = clean[f1] > f1_median
        clean['f2_high'] = clean[f2] > f2_median

        # Calculate returns for each quadrant
        both_high = clean[(clean['f1_high']) & (clean['f2_high'])][return_col].mean()
        f1_only = clean[(clean['f1_high']) & (~clean['f2_high'])][return_col].mean()
        f2_only = clean[(~clean['f1_high']) & (clean['f2_high'])][return_col].mean()
        both_low = clean[(~clean['f1_high']) & (~clean['f2_high'])][return_col].mean()

        # Interaction effect
        interaction = (both_high - f1_only) - (f2_only - both_low)

        results.append({
            'interaction': name,
            'both_high': both_high,
            'f1_only': f1_only,
            'f2_only': f2_only,
            'both_low': both_low,
            'interaction_effect': interaction,
            'best_quadrant': max([('Both High', both_high), ('F1 Only', f1_only),
                                  ('F2 Only', f2_only), ('Both Low', both_low)],
                                 key=lambda x: x[1] if not pd.isna(x[1]) else -999)[0],
        })

    results_df = pd.DataFrame(results)

    print(f"\n{'Interaction':<30} {'Both High':>10} {'F1 Only':>10} {'F2 Only':>10} {'Both Low':>10} {'Effect':>10}")
    print('-' * 95)

    for _, row in results_df.iterrows():
        print(f"{row['interaction']:<30} {row['both_high']:>+9.2f}% {row['f1_only']:>+9.2f}% "
              f"{row['f2_only']:>+9.2f}% {row['both_low']:>+9.2f}% {row['interaction_effect']:>+9.2f}%")

    return results_df


def run_fast_test(sample: int = None):
    """Run the fast factor test."""
    print("=" * 70)
    print("FAST COMPREHENSIVE FACTOR TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load and compute factors
    price_factors = load_and_compute_price_factors(sample_symbols=sample)
    fundamentals = load_fundamentals()
    merged = merge_factors(price_factors, fundamentals)

    # Add analyst factors (skip if too slow)
    # merged = load_analyst_factors(merged)

    # Winsorize returns
    merged['fwd_3m'] = merged['fwd_3m'].clip(-100, 100)

    # Test factors
    factor_results = test_all_factors(merged, 'fwd_3m')

    # Test interactions
    interaction_results = test_factor_interactions(merged, 'fwd_3m')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: TOP PREDICTIVE FACTORS")
    print('='*70)

    print("\n🟢 STRONGEST POSITIVE PREDICTORS (buy high, sell low):")
    top_positive = factor_results[factor_results['correlation'] > 0].head(10)
    for _, row in top_positive.iterrows():
        print(f"   {row['factor']:<20} r={row['correlation']:+.4f}  spread={row['q5_q1_spread']:+.2f}%")

    print("\n🔴 STRONGEST NEGATIVE PREDICTORS (buy low, sell high):")
    top_negative = factor_results[factor_results['correlation'] < 0].sort_values('correlation').head(10)
    for _, row in top_negative.iterrows():
        print(f"   {row['factor']:<20} r={row['correlation']:+.4f}  spread={row['q5_q1_spread']:+.2f}%")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return merged, factor_results, interaction_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None,
                        help='Number of symbols to sample (default: all)')
    args = parser.parse_args()

    merged, factor_results, interaction_results = run_fast_test(sample=args.sample)

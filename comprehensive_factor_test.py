#!/usr/bin/env python3
"""
Comprehensive Factor Testing Framework
=======================================
Systematically tests ALL available data attributes for predictive power.
Computes derived factors and tests combinations.

Goal: Find the best predictors for V4 scoring system.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')


def load_all_data(sample_size: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load price data and all fundamental data.
    Returns: (prices_df, fundamentals_df, grades_df)
    """
    conn = sqlite3.connect(BACKTEST_DB)

    print("Loading price data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close, volume
        FROM historical_prices
        WHERE adjusted_close > 1  -- Filter penny stocks
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(prices):,} price records")

    # Sample symbols if requested
    if sample_size:
        symbols = prices['symbol'].unique()
        np.random.seed(42)
        sample_symbols = np.random.choice(symbols, size=min(sample_size, len(symbols)), replace=False)
        prices = prices[prices['symbol'].isin(sample_symbols)]
        print(f"  Sampled {len(sample_symbols)} symbols")

    print("\nLoading income statements...")
    income = pd.read_sql_query("""
        SELECT symbol, date, fiscal_year, revenue, gross_profit, operating_income,
               net_income, ebitda, eps, eps_diluted, weighted_avg_shares_diluted
        FROM historical_income_statements
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(income):,} income statement records")

    print("\nLoading balance sheets...")
    balance = pd.read_sql_query("""
        SELECT symbol, date, fiscal_year, total_assets, total_liabilities,
               total_equity, total_debt, net_debt, cash_and_equivalents
        FROM historical_balance_sheets
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(balance):,} balance sheet records")

    print("\nLoading cash flows...")
    cashflow = pd.read_sql_query("""
        SELECT symbol, date, fiscal_year, operating_cash_flow,
               capital_expenditure, free_cash_flow
        FROM historical_cash_flows
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(cashflow):,} cash flow records")

    print("\nLoading key metrics...")
    metrics = pd.read_sql_query("""
        SELECT symbol, date, fiscal_year, enterprise_value, ev_to_ebitda,
               market_cap, pe_ratio, pb_ratio, debt_to_equity, roe,
               revenue_per_share, net_income_per_share, operating_cash_flow_per_share
        FROM historical_key_metrics
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(metrics):,} key metrics records")

    print("\nLoading analyst grades...")
    grades = pd.read_sql_query("""
        SELECT symbol, date, grading_company, action, new_grade
        FROM historical_grades
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(grades):,} analyst grade records")

    conn.close()

    # Merge fundamentals
    print("\nMerging fundamental data...")
    fundamentals = income.merge(balance, on=['symbol', 'date', 'fiscal_year'], how='outer')
    fundamentals = fundamentals.merge(cashflow, on=['symbol', 'date', 'fiscal_year'], how='outer')
    fundamentals = fundamentals.merge(metrics, on=['symbol', 'date', 'fiscal_year'], how='outer')
    print(f"  Merged fundamentals: {len(fundamentals):,} records")

    return prices, fundamentals, grades


def compute_price_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price-based factors for each stock-date.
    """
    print("\nComputing price-based factors...")

    all_factors = []
    symbols = prices['symbol'].unique()

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').copy()

        if len(sym_prices) < 300:
            continue

        closes = sym_prices['adjusted_close'].values
        volumes = sym_prices['volume'].values
        dates = sym_prices['date'].values
        n = len(closes)

        # Pre-compute rolling statistics
        for j in range(252, n - 63, 21):  # Monthly sampling
            # Forward returns (target variable)
            fwd_1m = ((closes[j + 21] / closes[j]) - 1) * 100 if j + 21 < n else np.nan
            fwd_3m = ((closes[j + 63] / closes[j]) - 1) * 100 if j + 63 < n else np.nan

            # === MOMENTUM FACTORS ===
            mom_1m = ((closes[j] / closes[j - 21]) - 1) * 100 if closes[j - 21] > 0 else np.nan
            mom_3m = ((closes[j] / closes[j - 63]) - 1) * 100 if closes[j - 63] > 0 else np.nan
            mom_6m = ((closes[j] / closes[j - 126]) - 1) * 100 if j >= 126 and closes[j - 126] > 0 else np.nan
            mom_12m = ((closes[j] / closes[j - 252]) - 1) * 100 if closes[j - 252] > 0 else np.nan
            mom_12_1 = ((closes[j - 21] / closes[j - 252]) - 1) * 100 if closes[j - 252] > 0 else np.nan

            # === 52-WEEK HIGH ===
            high_52w = np.max(closes[max(0, j - 252):j])
            pct_from_high = ((closes[j] / high_52w) - 1) * 100 if high_52w > 0 else np.nan

            # === VOLATILITY FACTORS ===
            # 20-day realized volatility
            returns_20d = np.diff(closes[j - 21:j + 1]) / closes[j - 21:j]
            vol_20d = np.std(returns_20d) * np.sqrt(252) * 100 if len(returns_20d) > 5 else np.nan

            # 60-day realized volatility
            returns_60d = np.diff(closes[j - 63:j + 1]) / closes[j - 63:j]
            vol_60d = np.std(returns_60d) * np.sqrt(252) * 100 if len(returns_60d) > 20 else np.nan

            # Volatility change (vol expansion/contraction)
            vol_20d_old = np.std(np.diff(closes[j - 42:j - 21]) / closes[j - 42:j - 22]) * np.sqrt(252) * 100 if j >= 42 else np.nan
            vol_change = (vol_20d - vol_20d_old) if vol_20d and vol_20d_old else np.nan

            # === VOLUME FACTORS ===
            avg_vol_20d = np.mean(volumes[j - 21:j]) if j >= 21 else np.nan
            avg_vol_60d = np.mean(volumes[j - 63:j]) if j >= 63 else np.nan
            vol_ratio = (avg_vol_20d / avg_vol_60d) if avg_vol_60d and avg_vol_60d > 0 else np.nan

            # Dollar volume (liquidity proxy)
            dollar_vol = closes[j] * avg_vol_20d / 1e6 if avg_vol_20d else np.nan

            # === TREND FACTORS ===
            sma_50 = np.mean(closes[j - 50:j]) if j >= 50 else np.nan
            sma_200 = np.mean(closes[j - 200:j]) if j >= 200 else np.nan

            price_vs_sma50 = ((closes[j] / sma_50) - 1) * 100 if sma_50 and sma_50 > 0 else np.nan
            price_vs_sma200 = ((closes[j] / sma_200) - 1) * 100 if sma_200 and sma_200 > 0 else np.nan
            sma50_vs_sma200 = ((sma_50 / sma_200) - 1) * 100 if sma_50 and sma_200 and sma_200 > 0 else np.nan

            # Golden/death cross signal
            trend_strength = 1 if (sma_50 and sma_200 and sma_50 > sma_200 and closes[j] > sma_50) else 0

            # === MEAN REVERSION FACTORS ===
            # RSI-like factor
            gains = []
            losses = []
            for k in range(j - 14, j):
                change = closes[k + 1] - closes[k]
                if change > 0:
                    gains.append(change)
                else:
                    losses.append(abs(change))
            avg_gain = np.mean(gains) if gains else 0
            avg_loss = np.mean(losses) if losses else 0.0001
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Distance from 20-day mean (mean reversion signal)
            sma_20 = np.mean(closes[j - 20:j])
            dist_from_mean = ((closes[j] / sma_20) - 1) * 100 if sma_20 > 0 else np.nan

            all_factors.append({
                'symbol': symbol,
                'date': dates[j],
                'price': closes[j],
                # Forward returns
                'fwd_1m': fwd_1m,
                'fwd_3m': fwd_3m,
                # Momentum
                'mom_1m': mom_1m,
                'mom_3m': mom_3m,
                'mom_6m': mom_6m,
                'mom_12m': mom_12m,
                'mom_12_1': mom_12_1,
                'pct_from_high': pct_from_high,
                # Volatility
                'vol_20d': vol_20d,
                'vol_60d': vol_60d,
                'vol_change': vol_change,
                # Volume
                'vol_ratio': vol_ratio,
                'dollar_vol': dollar_vol,
                # Trend
                'price_vs_sma50': price_vs_sma50,
                'price_vs_sma200': price_vs_sma200,
                'sma50_vs_sma200': sma50_vs_sma200,
                'trend_strength': trend_strength,
                # Mean reversion
                'rsi': rsi,
                'dist_from_mean': dist_from_mean,
            })

    df = pd.DataFrame(all_factors)
    print(f"  Computed {len(df):,} price-factor observations")
    return df


def compute_fundamental_factors(prices_factors: pd.DataFrame,
                                fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Merge fundamental data and compute derived factors.
    """
    print("\nComputing fundamental factors...")

    # Prepare fundamentals - get most recent prior to each date
    fundamentals['date'] = pd.to_datetime(fundamentals['date'])

    # Create lookup dict for each symbol
    fund_by_symbol = {}
    for symbol in fundamentals['symbol'].unique():
        sym_fund = fundamentals[fundamentals['symbol'] == symbol].sort_values('date')
        fund_by_symbol[symbol] = sym_fund

    # Compute derived fundamental factors
    results = []

    for i, row in prices_factors.iterrows():
        if i % 50000 == 0:
            print(f"  Processing {i}/{len(prices_factors)} observations...")

        symbol = row['symbol']
        obs_date = pd.to_datetime(row['date'])

        # Get most recent fundamentals before this date
        if symbol not in fund_by_symbol:
            results.append({})
            continue

        sym_fund = fund_by_symbol[symbol]
        prior_fund = sym_fund[sym_fund['date'] < obs_date]

        if len(prior_fund) < 2:
            results.append({})
            continue

        current = prior_fund.iloc[-1]
        prior = prior_fund.iloc[-5] if len(prior_fund) >= 5 else prior_fund.iloc[0]  # ~1 year ago

        # === PROFITABILITY FACTORS ===
        # Gross profitability (Novy-Marx)
        gp_assets = current['gross_profit'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # ROE (from key metrics or computed)
        roe = current['roe']
        if pd.isna(roe) and current['total_equity'] and current['total_equity'] > 0:
            roe = current['net_income'] / current['total_equity']

        # ROA
        roa = current['net_income'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # Operating margin
        op_margin = current['operating_income'] / current['revenue'] if current['revenue'] and current['revenue'] > 0 else np.nan

        # Net margin
        net_margin = current['net_income'] / current['revenue'] if current['revenue'] and current['revenue'] > 0 else np.nan

        # EBITDA margin
        ebitda_margin = current['ebitda'] / current['revenue'] if current['revenue'] and current['revenue'] > 0 else np.nan

        # === GROWTH FACTORS ===
        # Revenue growth
        rev_growth = ((current['revenue'] / prior['revenue']) - 1) * 100 if prior['revenue'] and prior['revenue'] > 0 else np.nan

        # EPS growth
        eps_growth = ((current['eps'] / prior['eps']) - 1) * 100 if prior['eps'] and prior['eps'] > 0 else np.nan

        # Asset growth (inverse is predictive per literature)
        asset_growth = ((current['total_assets'] / prior['total_assets']) - 1) * 100 if prior['total_assets'] and prior['total_assets'] > 0 else np.nan

        # Operating income growth
        op_income_growth = ((current['operating_income'] / prior['operating_income']) - 1) * 100 if prior['operating_income'] and prior['operating_income'] > 0 else np.nan

        # === VALUATION FACTORS ===
        ev_ebitda = current['ev_to_ebitda']
        pe_ratio = current['pe_ratio']
        pb_ratio = current['pb_ratio']

        # Earnings yield (inverse of P/E)
        earnings_yield = (1 / pe_ratio) * 100 if pe_ratio and pe_ratio > 0 else np.nan

        # FCF yield
        fcf_yield = (current['free_cash_flow'] / current['market_cap']) * 100 if current['market_cap'] and current['market_cap'] > 0 else np.nan

        # Enterprise value to sales
        ev_sales = current['enterprise_value'] / current['revenue'] if current['revenue'] and current['revenue'] > 0 else np.nan

        # === LEVERAGE/QUALITY FACTORS ===
        debt_to_equity = current['debt_to_equity']

        # Debt to assets
        debt_to_assets = current['total_debt'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # Net debt to EBITDA
        net_debt_ebitda = current['net_debt'] / current['ebitda'] if current['ebitda'] and current['ebitda'] > 0 else np.nan

        # Current ratio proxy (cash / short-term obligations approximation)
        cash_to_debt = current['cash_and_equivalents'] / current['total_debt'] if current['total_debt'] and current['total_debt'] > 0 else np.nan

        # === CASH FLOW FACTORS ===
        # Operating cash flow to assets
        ocf_assets = current['operating_cash_flow'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # FCF to assets
        fcf_assets = current['free_cash_flow'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # Accruals (earnings quality) = Net Income - OCF
        accruals = (current['net_income'] - current['operating_cash_flow']) / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else np.nan

        # === SIZE FACTORS ===
        market_cap = current['market_cap']
        log_market_cap = np.log(market_cap) if market_cap and market_cap > 0 else np.nan

        # === PIOTROSKI F-SCORE COMPONENTS ===
        # Simplified F-Score (0-9 points)
        f_score = 0
        # 1. Positive ROA
        if roa and roa > 0:
            f_score += 1
        # 2. Positive OCF
        if current['operating_cash_flow'] and current['operating_cash_flow'] > 0:
            f_score += 1
        # 3. ROA improvement
        prior_roa = prior['net_income'] / prior['total_assets'] if prior['total_assets'] and prior['total_assets'] > 0 else None
        if roa and prior_roa and roa > prior_roa:
            f_score += 1
        # 4. Accruals (OCF > Net Income = better quality)
        if current['operating_cash_flow'] and current['net_income'] and current['operating_cash_flow'] > current['net_income']:
            f_score += 1
        # 5. Leverage decrease
        prior_debt_assets = prior['total_debt'] / prior['total_assets'] if prior['total_assets'] and prior['total_assets'] > 0 else None
        if debt_to_assets and prior_debt_assets and debt_to_assets < prior_debt_assets:
            f_score += 1
        # 6. Liquidity improvement (cash ratio)
        prior_cash_debt = prior['cash_and_equivalents'] / prior['total_debt'] if prior['total_debt'] and prior['total_debt'] > 0 else None
        if cash_to_debt and prior_cash_debt and cash_to_debt > prior_cash_debt:
            f_score += 1
        # 7. No dilution (shares didn't increase)
        if current['weighted_avg_shares_diluted'] and prior['weighted_avg_shares_diluted']:
            if current['weighted_avg_shares_diluted'] <= prior['weighted_avg_shares_diluted']:
                f_score += 1
        # 8. Gross margin improvement
        prior_gp_margin = prior['gross_profit'] / prior['revenue'] if prior['revenue'] and prior['revenue'] > 0 else None
        curr_gp_margin = current['gross_profit'] / current['revenue'] if current['revenue'] and current['revenue'] > 0 else None
        if curr_gp_margin and prior_gp_margin and curr_gp_margin > prior_gp_margin:
            f_score += 1
        # 9. Asset turnover improvement
        prior_turnover = prior['revenue'] / prior['total_assets'] if prior['total_assets'] and prior['total_assets'] > 0 else None
        curr_turnover = current['revenue'] / current['total_assets'] if current['total_assets'] and current['total_assets'] > 0 else None
        if curr_turnover and prior_turnover and curr_turnover > prior_turnover:
            f_score += 1

        results.append({
            # Profitability
            'gp_assets': gp_assets,
            'roe': roe,
            'roa': roa,
            'op_margin': op_margin,
            'net_margin': net_margin,
            'ebitda_margin': ebitda_margin,
            # Growth
            'rev_growth': rev_growth,
            'eps_growth': eps_growth,
            'asset_growth': asset_growth,
            'op_income_growth': op_income_growth,
            # Valuation
            'ev_ebitda': ev_ebitda,
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'earnings_yield': earnings_yield,
            'fcf_yield': fcf_yield,
            'ev_sales': ev_sales,
            # Leverage/Quality
            'debt_to_equity': debt_to_equity,
            'debt_to_assets': debt_to_assets,
            'net_debt_ebitda': net_debt_ebitda,
            'cash_to_debt': cash_to_debt,
            # Cash flow
            'ocf_assets': ocf_assets,
            'fcf_assets': fcf_assets,
            'accruals': accruals,
            # Size
            'market_cap': market_cap,
            'log_market_cap': log_market_cap,
            # Quality composite
            'f_score': f_score,
        })

    fund_df = pd.DataFrame(results)

    # Merge with price factors
    merged = pd.concat([prices_factors.reset_index(drop=True), fund_df], axis=1)
    print(f"  Merged {len(merged):,} observations with fundamental factors")

    return merged


def compute_analyst_factors(merged_df: pd.DataFrame, grades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute analyst-related factors (earnings surprise proxies, revision ratios).
    """
    print("\nComputing analyst factors...")

    # Create lookup for analyst grades by symbol
    grades['date'] = pd.to_datetime(grades['date'])

    analyst_factors = []

    for i, row in merged_df.iterrows():
        if i % 50000 == 0:
            print(f"  Processing {i}/{len(merged_df)} observations...")

        symbol = row['symbol']
        obs_date = pd.to_datetime(row['date'])

        # Look back 90 days for analyst activity
        lookback_start = obs_date - pd.Timedelta(days=90)

        sym_grades = grades[(grades['symbol'] == symbol) &
                           (grades['date'] >= lookback_start) &
                           (grades['date'] < obs_date)]

        if len(sym_grades) == 0:
            analyst_factors.append({
                'upgrades_90d': 0,
                'downgrades_90d': 0,
                'upgrade_ratio': 0.5,
                'analyst_momentum': 0,
                'recent_upgrade': 0,
                'recent_downgrade': 0,
            })
            continue

        upgrades = len(sym_grades[sym_grades['action'] == 'upgrade'])
        downgrades = len(sym_grades[sym_grades['action'] == 'downgrade'])
        total = upgrades + downgrades

        upgrade_ratio = upgrades / total if total > 0 else 0.5
        analyst_momentum = upgrades - downgrades

        # Recent activity (last 30 days gets more weight)
        recent_start = obs_date - pd.Timedelta(days=30)
        recent_grades = sym_grades[sym_grades['date'] >= recent_start]
        recent_upgrade = 1 if 'upgrade' in recent_grades['action'].values else 0
        recent_downgrade = 1 if 'downgrade' in recent_grades['action'].values else 0

        analyst_factors.append({
            'upgrades_90d': upgrades,
            'downgrades_90d': downgrades,
            'upgrade_ratio': upgrade_ratio,
            'analyst_momentum': analyst_momentum,
            'recent_upgrade': recent_upgrade,
            'recent_downgrade': recent_downgrade,
        })

    analyst_df = pd.DataFrame(analyst_factors)
    merged = pd.concat([merged_df.reset_index(drop=True), analyst_df], axis=1)
    print(f"  Added analyst factors to {len(merged):,} observations")

    return merged


def test_factor_correlations(df: pd.DataFrame, return_col: str = 'fwd_3m') -> pd.DataFrame:
    """
    Test correlation of each factor with forward returns.
    """
    print(f"\n{'='*70}")
    print(f"FACTOR CORRELATION ANALYSIS (vs {return_col})")
    print('='*70)

    # List of factors to test
    factors = [
        # Momentum
        'mom_1m', 'mom_3m', 'mom_6m', 'mom_12m', 'mom_12_1', 'pct_from_high',
        # Volatility
        'vol_20d', 'vol_60d', 'vol_change',
        # Volume
        'vol_ratio', 'dollar_vol',
        # Trend
        'price_vs_sma50', 'price_vs_sma200', 'sma50_vs_sma200', 'trend_strength',
        # Mean reversion
        'rsi', 'dist_from_mean',
        # Profitability
        'gp_assets', 'roe', 'roa', 'op_margin', 'net_margin', 'ebitda_margin',
        # Growth
        'rev_growth', 'eps_growth', 'asset_growth', 'op_income_growth',
        # Valuation
        'ev_ebitda', 'pe_ratio', 'pb_ratio', 'earnings_yield', 'fcf_yield', 'ev_sales',
        # Leverage
        'debt_to_equity', 'debt_to_assets', 'net_debt_ebitda', 'cash_to_debt',
        # Cash flow
        'ocf_assets', 'fcf_assets', 'accruals',
        # Size
        'log_market_cap',
        # Quality
        'f_score',
        # Analyst
        'upgrades_90d', 'downgrades_90d', 'upgrade_ratio', 'analyst_momentum',
        'recent_upgrade', 'recent_downgrade',
    ]

    results = []

    for factor in factors:
        if factor not in df.columns:
            continue

        clean = df.dropna(subset=[factor, return_col])

        if len(clean) < 1000:
            continue

        # Correlation
        corr = clean[factor].corr(clean[return_col])

        # Q5-Q1 spread
        try:
            q5 = clean[clean[factor] > clean[factor].quantile(0.8)][return_col].mean()
            q1 = clean[clean[factor] < clean[factor].quantile(0.2)][return_col].mean()
            spread = q5 - q1
        except:
            spread = np.nan

        # T-statistic approximation
        n = len(clean)
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0

        results.append({
            'factor': factor,
            'correlation': corr,
            'q5_q1_spread': spread,
            't_statistic': t_stat,
            'n_obs': n,
            'significant': abs(t_stat) > 2.0,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('correlation', ascending=False)

    # Print results
    print(f"\n{'Factor':<25} {'Corr':>10} {'Spread':>10} {'T-Stat':>10} {'N':>12} {'Sig'}")
    print('-' * 80)

    for _, row in results_df.iterrows():
        sig = '***' if abs(row['t_statistic']) > 3 else '**' if abs(row['t_statistic']) > 2 else '*' if abs(row['t_statistic']) > 1.5 else ''
        print(f"{row['factor']:<25} {row['correlation']:>+9.4f} {row['q5_q1_spread']:>+9.2f}% {row['t_statistic']:>+9.2f} {row['n_obs']:>12,} {sig:>3}")

    return results_df


def test_factor_combinations(df: pd.DataFrame, top_factors: List[str],
                            return_col: str = 'fwd_3m') -> pd.DataFrame:
    """
    Test combinations of top factors using simple scoring.
    """
    print(f"\n{'='*70}")
    print("FACTOR COMBINATION ANALYSIS")
    print('='*70)

    from itertools import combinations

    results = []

    # Test single factors
    for factor in top_factors:
        if factor not in df.columns:
            continue

        clean = df.dropna(subset=[factor, return_col])
        if len(clean) < 1000:
            continue

        # Normalize factor to 0-100
        pct_rank = clean[factor].rank(pct=True) * 100

        corr = pct_rank.corr(clean[return_col])
        q5 = clean[pct_rank > 80][return_col].mean()
        q1 = clean[pct_rank < 20][return_col].mean()

        results.append({
            'combination': factor,
            'n_factors': 1,
            'correlation': corr,
            'q5_q1_spread': q5 - q1,
            'n_obs': len(clean),
        })

    # Test pairs
    print("\nTesting factor pairs...")
    for f1, f2 in combinations(top_factors[:10], 2):
        if f1 not in df.columns or f2 not in df.columns:
            continue

        clean = df.dropna(subset=[f1, f2, return_col])
        if len(clean) < 1000:
            continue

        # Simple equal-weighted combination
        score = (clean[f1].rank(pct=True) + clean[f2].rank(pct=True)) / 2 * 100

        corr = score.corr(clean[return_col])
        q5 = clean[score > 80][return_col].mean()
        q1 = clean[score < 20][return_col].mean()

        results.append({
            'combination': f"{f1} + {f2}",
            'n_factors': 2,
            'correlation': corr,
            'q5_q1_spread': q5 - q1,
            'n_obs': len(clean),
        })

    # Test triples of best pairs
    print("Testing factor triples...")
    for f1, f2, f3 in combinations(top_factors[:8], 3):
        if f1 not in df.columns or f2 not in df.columns or f3 not in df.columns:
            continue

        clean = df.dropna(subset=[f1, f2, f3, return_col])
        if len(clean) < 1000:
            continue

        score = (clean[f1].rank(pct=True) + clean[f2].rank(pct=True) + clean[f3].rank(pct=True)) / 3 * 100

        corr = score.corr(clean[return_col])
        q5 = clean[score > 80][return_col].mean()
        q1 = clean[score < 20][return_col].mean()

        results.append({
            'combination': f"{f1} + {f2} + {f3}",
            'n_factors': 3,
            'correlation': corr,
            'q5_q1_spread': q5 - q1,
            'n_obs': len(clean),
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('q5_q1_spread', ascending=False)

    # Print top combinations
    print(f"\n{'Combination':<50} {'Corr':>8} {'Spread':>10} {'N':>10}")
    print('-' * 85)

    for _, row in results_df.head(30).iterrows():
        print(f"{row['combination']:<50} {row['correlation']:>+7.4f} {row['q5_q1_spread']:>+9.2f}% {row['n_obs']:>10,}")

    return results_df


def run_comprehensive_test(sample_size: int = 1000):
    """
    Run the full comprehensive factor testing pipeline.
    """
    print("=" * 70)
    print("COMPREHENSIVE FACTOR TESTING FRAMEWORK")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    prices, fundamentals, grades = load_all_data(sample_size=sample_size)

    # Compute all factors
    price_factors = compute_price_factors(prices)
    merged = compute_fundamental_factors(price_factors, fundamentals)
    full_df = compute_analyst_factors(merged, grades)

    # Winsorize returns
    for col in ['fwd_1m', 'fwd_3m']:
        if col in full_df.columns:
            full_df[col] = full_df[col].clip(-100, 100)

    # Test correlations
    corr_results = test_factor_correlations(full_df, 'fwd_3m')

    # Get top factors (positive correlation, significant)
    top_positive = corr_results[
        (corr_results['correlation'] > 0) &
        (corr_results['significant'] == True)
    ]['factor'].tolist()

    # Get factors with highest absolute correlation
    top_factors = corr_results.head(15)['factor'].tolist()

    # Test combinations
    combo_results = test_factor_combinations(full_df, top_factors, 'fwd_3m')

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: BEST INDIVIDUAL FACTORS")
    print('='*70)

    print("\nTop 10 Positively Correlated Factors:")
    for _, row in corr_results[corr_results['correlation'] > 0].head(10).iterrows():
        print(f"  {row['factor']:<25} r={row['correlation']:+.4f}  spread={row['q5_q1_spread']:+.2f}%")

    print("\nTop 10 Negatively Correlated Factors (INVERSE is predictive):")
    for _, row in corr_results[corr_results['correlation'] < 0].tail(10).iterrows():
        print(f"  {row['factor']:<25} r={row['correlation']:+.4f}  spread={row['q5_q1_spread']:+.2f}%")

    print(f"\n{'='*70}")
    print("BEST FACTOR COMBINATIONS")
    print('='*70)
    print("\nTop 10 Combinations by Q5-Q1 Spread:")
    for _, row in combo_results.head(10).iterrows():
        print(f"  {row['combination']:<50} spread={row['q5_q1_spread']:+.2f}%")

    print(f"\n\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return full_df, corr_results, combo_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Comprehensive factor testing')
    parser.add_argument('--sample', type=int, default=1000,
                        help='Number of symbols to sample (0 for all)')
    parser.add_argument('--full', action='store_true',
                        help='Run on full dataset (slow)')
    args = parser.parse_args()

    sample_size = None if args.full else args.sample

    full_df, corr_results, combo_results = run_comprehensive_test(sample_size=sample_size)

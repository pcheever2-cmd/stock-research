#!/usr/bin/env python3
"""
Multivariate Factor Analysis & Regime-Adaptive Scoring
=======================================================
1. Multivariate regression to find which factors survive together
2. Regime detection (bull/bear/high-vol/low-vol)
3. Factor performance by regime
4. Adaptive V4 scoring with regime-based weights
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')


# ==================== DATA LOADING ====================

def load_factor_data() -> pd.DataFrame:
    """Load and compute all factors for multivariate analysis."""
    print("=" * 70)
    print("LOADING DATA FOR MULTIVARIATE ANALYSIS")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    print("\nLoading price data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  Loaded {len(prices):,} price records")

    # Load fundamentals
    print("\nLoading fundamental data...")
    income = pd.read_sql_query("""
        SELECT symbol, date, revenue, gross_profit, operating_income,
               net_income, ebitda
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
        SELECT symbol, date, ev_to_ebitda, market_cap, pe_ratio, pb_ratio, roe
        FROM historical_key_metrics
    """, conn)

    # Load SPY for regime detection
    print("\nLoading SPY for regime detection...")
    spy = pd.read_sql_query("""
        SELECT date, adjusted_close as spy_close
        FROM historical_prices
        WHERE symbol = 'SPY'
        ORDER BY date
    """, conn)
    spy['date'] = pd.to_datetime(spy['date'])

    conn.close()

    # Merge fundamentals
    fund = income.merge(balance, on=['symbol', 'date'], how='outer')
    fund = fund.merge(cashflow, on=['symbol', 'date'], how='outer')
    fund = fund.merge(metrics, on=['symbol', 'date'], how='outer')
    fund['date'] = pd.to_datetime(fund['date'])

    # Compute derived factors
    print("\nComputing derived factors...")
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['fcf_yield'] = fund['free_cash_flow'] / fund['market_cap']
    fund['accruals'] = (fund['net_income'] - fund['operating_cash_flow']) / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4) * 100

    # Compute price-based factors per symbol
    print("\nComputing price-based factors...")

    def compute_price_factors(group):
        group = group.sort_values('date').reset_index(drop=True)
        n = len(group)

        if n < 300:
            return pd.DataFrame()

        close = group['close'].values
        volume = group['volume'].values
        results = []

        # Sample monthly
        for j in range(252, n - 63, 21):
            # Forward return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100 if j + 63 < n else np.nan

            # Momentum
            mom_1m = ((close[j] / close[j - 21]) - 1) * 100 if close[j - 21] > 0 else np.nan
            mom_6m = ((close[j] / close[j - 126]) - 1) * 100 if j >= 126 and close[j - 126] > 0 else np.nan
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility
            returns_60d = np.diff(close[j - 63:j + 1]) / close[j - 63:j]
            vol_60d = np.std(returns_60d) * np.sqrt(252) * 100 if len(returns_60d) > 20 else np.nan

            # Trend
            sma_200 = np.mean(close[j - 200:j]) if j >= 200 else np.nan
            price_vs_sma200 = ((close[j] / sma_200) - 1) * 100 if sma_200 and sma_200 > 0 else np.nan

            results.append({
                'symbol': group['symbol'].iloc[0],
                'date': group['date'].iloc[j],
                'close': close[j],
                'fwd_3m': fwd_3m,
                'mom_1m': mom_1m,
                'mom_6m': mom_6m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'price_vs_sma200': price_vs_sma200,
            })

        return pd.DataFrame(results)

    # Process symbols
    symbols = prices['symbol'].unique()
    all_results = []

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  Processing {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol]
        result = compute_price_factors(sym_prices)
        if len(result) > 0:
            all_results.append(result)

    price_factors = pd.concat(all_results, ignore_index=True)
    print(f"  Generated {len(price_factors):,} price-factor observations")

    # Merge with fundamentals
    print("\nMerging with fundamentals...")
    fund = fund.sort_values(['symbol', 'date'])

    all_merged = []
    for symbol in price_factors['symbol'].unique():
        pf = price_factors[price_factors['symbol'] == symbol].sort_values('date')
        f = fund[fund['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(
            pf, f.drop(columns=['symbol']),
            on='date', direction='backward',
            tolerance=pd.Timedelta('365 days')
        )
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)

    # Add SPY data for regime detection
    df = pd.merge_asof(
        df.sort_values('date'),
        spy.sort_values('date'),
        on='date', direction='backward'
    )

    print(f"\nFinal dataset: {len(df):,} observations")

    return df, spy


# ==================== REGIME DETECTION ====================

def detect_regimes(spy: pd.DataFrame) -> pd.DataFrame:
    """
    Detect market regimes based on SPY characteristics.

    Regimes:
    - Bull: SPY above 200-day SMA, positive 6-month momentum
    - Bear: SPY below 200-day SMA, negative 6-month momentum
    - High Vol: 20-day realized vol > 25% annualized
    - Recovery: Below SMA but positive short-term momentum
    """
    print("\n" + "=" * 70)
    print("DETECTING MARKET REGIMES")
    print("=" * 70)

    spy = spy.sort_values('date').reset_index(drop=True)
    spy['spy_close'] = spy['spy_close'].astype(float)

    # Calculate indicators
    spy['sma_200'] = spy['spy_close'].rolling(200).mean()
    spy['sma_50'] = spy['spy_close'].rolling(50).mean()
    spy['mom_6m'] = spy['spy_close'].pct_change(126) * 100
    spy['mom_1m'] = spy['spy_close'].pct_change(21) * 100

    # Volatility
    spy['returns'] = spy['spy_close'].pct_change()
    spy['vol_20d'] = spy['returns'].rolling(20).std() * np.sqrt(252) * 100

    # Classify regimes
    def classify_regime(row):
        if pd.isna(row['sma_200']) or pd.isna(row['vol_20d']):
            return 'Unknown'

        above_sma = row['spy_close'] > row['sma_200']
        positive_mom = row['mom_6m'] > 0 if not pd.isna(row['mom_6m']) else False
        high_vol = row['vol_20d'] > 25
        positive_short = row['mom_1m'] > 0 if not pd.isna(row['mom_1m']) else False

        if high_vol:
            return 'High_Vol'
        elif above_sma and positive_mom:
            return 'Bull'
        elif not above_sma and not positive_mom:
            return 'Bear'
        elif not above_sma and positive_short:
            return 'Recovery'
        else:
            return 'Transition'

    spy['regime'] = spy.apply(classify_regime, axis=1)

    # Print regime distribution
    print("\nRegime Distribution:")
    regime_counts = spy['regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = count / len(spy) * 100
        print(f"  {regime:<12}: {count:>6,} days ({pct:.1f}%)")

    return spy[['date', 'regime', 'vol_20d', 'mom_6m', 'sma_200', 'spy_close']]


# ==================== MULTIVARIATE REGRESSION ====================

def run_multivariate_regression(df: pd.DataFrame) -> dict:
    """
    Run multivariate OLS regression to find which factors survive.

    Model: fwd_3m ~ ROA + OCF/Assets + FCF/Assets + FCF_Yield + GP/Assets +
                    Asset_Growth + Vol_60d + Mom_12_1 + Mom_6m + EV/EBITDA
    """
    print("\n" + "=" * 70)
    print("MULTIVARIATE FACTOR REGRESSION")
    print("=" * 70)

    # Define factors to test
    factors = [
        'roa', 'ocf_assets', 'fcf_assets', 'fcf_yield', 'gp_assets',
        'asset_growth', 'vol_60d', 'mom_12_1', 'mom_6m', 'ev_to_ebitda',
        'price_vs_sma200', 'accruals'
    ]

    # Clean data
    available_factors = [f for f in factors if f in df.columns]
    clean = df.dropna(subset=['fwd_3m'] + available_factors).copy()

    print(f"\nObservations with all factors: {len(clean):,}")

    if len(clean) < 1000:
        print("  Insufficient data for regression")
        return {}

    # Standardize factors (z-score)
    for factor in available_factors:
        clean[f'{factor}_z'] = (clean[factor] - clean[factor].mean()) / clean[factor].std()
        # Winsorize
        clean[f'{factor}_z'] = clean[f'{factor}_z'].clip(-3, 3)

    # Build design matrix
    X_cols = [f'{f}_z' for f in available_factors]
    X = clean[X_cols].values
    y = clean['fwd_3m'].values

    # Add constant
    X = np.column_stack([np.ones(len(X)), X])

    # OLS regression
    try:
        # (X'X)^-1 X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y

        # Residuals and standard errors
        y_hat = X @ beta
        residuals = y - y_hat
        n, k = X.shape
        mse = np.sum(residuals**2) / (n - k)
        se = np.sqrt(np.diag(XtX_inv) * mse)

        # T-statistics
        t_stats = beta / se

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    except np.linalg.LinAlgError:
        print("  Singular matrix - cannot compute regression")
        return {}

    # Results
    results = {
        'n_obs': n,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'factors': {}
    }

    print(f"\nR² = {r_squared:.4f}, Adjusted R² = {adj_r_squared:.4f}")
    print(f"\n{'Factor':<20} {'Coefficient':>12} {'Std Error':>12} {'T-Stat':>10} {'Sig':>6}")
    print("-" * 65)

    # Intercept
    print(f"{'(Intercept)':<20} {beta[0]:>+11.4f} {se[0]:>11.4f} {t_stats[0]:>+9.2f}")

    # Factors (sorted by absolute t-stat)
    factor_results = []
    for i, factor in enumerate(available_factors):
        idx = i + 1
        sig = '***' if abs(t_stats[idx]) > 2.58 else '**' if abs(t_stats[idx]) > 1.96 else '*' if abs(t_stats[idx]) > 1.65 else ''
        factor_results.append({
            'factor': factor,
            'coef': beta[idx],
            'se': se[idx],
            't_stat': t_stats[idx],
            'sig': sig,
            'abs_t': abs(t_stats[idx])
        })
        results['factors'][factor] = {
            'coefficient': beta[idx],
            'std_error': se[idx],
            't_statistic': t_stats[idx],
            'significant': abs(t_stats[idx]) > 1.96
        }

    # Sort by absolute t-stat
    factor_results.sort(key=lambda x: x['abs_t'], reverse=True)

    for fr in factor_results:
        print(f"{fr['factor']:<20} {fr['coef']:>+11.4f} {fr['se']:>11.4f} {fr['t_stat']:>+9.2f} {fr['sig']:>6}")

    # Summary
    print("\n" + "-" * 65)
    significant = [f for f, v in results['factors'].items() if v['significant']]
    print(f"\nSignificant factors (|t| > 1.96): {', '.join(significant)}")

    return results


# ==================== FACTOR PERFORMANCE BY REGIME ====================

def analyze_factors_by_regime(df: pd.DataFrame, regimes: pd.DataFrame) -> dict:
    """Test factor performance within each regime."""
    print("\n" + "=" * 70)
    print("FACTOR PERFORMANCE BY REGIME")
    print("=" * 70)

    # Merge regimes
    df = pd.merge_asof(
        df.sort_values('date'),
        regimes[['date', 'regime']].sort_values('date'),
        on='date', direction='backward'
    )

    factors = ['roa', 'ocf_assets', 'fcf_assets', 'fcf_yield', 'gp_assets',
               'vol_60d', 'mom_12_1', 'mom_6m', 'asset_growth', 'ev_to_ebitda']

    results = {}

    for regime in ['Bull', 'Bear', 'High_Vol', 'Recovery']:
        regime_df = df[df['regime'] == regime]

        if len(regime_df) < 500:
            continue

        print(f"\n=== {regime} Regime ({len(regime_df):,} observations) ===")
        print(f"{'Factor':<18} {'Correlation':>12} {'Q5-Q1 Spread':>14} {'T-Stat':>10}")
        print("-" * 58)

        regime_results = {}

        for factor in factors:
            if factor not in regime_df.columns:
                continue

            clean = regime_df.dropna(subset=[factor, 'fwd_3m'])

            if len(clean) < 200:
                continue

            # Correlation
            corr = clean[factor].corr(clean['fwd_3m'])

            # Q5-Q1 spread
            try:
                q5_thresh = clean[factor].quantile(0.8)
                q1_thresh = clean[factor].quantile(0.2)
                q5_ret = clean[clean[factor] >= q5_thresh]['fwd_3m'].mean()
                q1_ret = clean[clean[factor] <= q1_thresh]['fwd_3m'].mean()
                spread = q5_ret - q1_ret
            except:
                spread = np.nan

            # T-stat
            n = len(clean)
            t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2) if abs(corr) < 1 else 0

            sig = '***' if abs(t_stat) > 2.58 else '**' if abs(t_stat) > 1.96 else '*' if abs(t_stat) > 1.65 else ''

            print(f"{factor:<18} {corr:>+11.4f} {spread:>+13.2f}% {t_stat:>+9.1f} {sig}")

            regime_results[factor] = {
                'correlation': corr,
                'spread': spread,
                't_stat': t_stat
            }

        results[regime] = regime_results

    return results


# ==================== OPTIMAL WEIGHTS BY REGIME ====================

def compute_optimal_weights_by_regime(regime_results: dict) -> dict:
    """Compute optimal factor weights for each regime based on t-statistics."""
    print("\n" + "=" * 70)
    print("OPTIMAL FACTOR WEIGHTS BY REGIME")
    print("=" * 70)

    optimal_weights = {}

    # Factors to include (subset of most predictive)
    core_factors = ['roa', 'ocf_assets', 'fcf_assets', 'fcf_yield', 'gp_assets',
                    'vol_60d', 'mom_12_1', 'mom_6m', 'asset_growth']

    for regime, factors in regime_results.items():
        print(f"\n=== {regime} Regime ===")

        # Get t-stats for each factor
        factor_scores = {}
        for factor in core_factors:
            if factor in factors:
                t_stat = factors[factor]['t_stat']
                # Higher positive t-stat = more weight
                # Negative factors (like asset_growth) get inverted
                if factor == 'asset_growth':
                    factor_scores[factor] = -t_stat  # Invert
                elif factor == 'vol_60d':
                    factor_scores[factor] = -t_stat  # Low vol is good
                else:
                    factor_scores[factor] = t_stat

        # Normalize to sum to 100
        total_positive = sum(max(0, s) for s in factor_scores.values())

        if total_positive == 0:
            continue

        weights = {}
        for factor, score in factor_scores.items():
            weight = max(0, score) / total_positive * 100
            weights[factor] = round(weight, 1)

        # Print weights
        for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
            if weight > 0:
                print(f"  {factor:<18}: {weight:>5.1f} pts")

        optimal_weights[regime] = weights

    return optimal_weights


# ==================== ADAPTIVE V4 SCORING ====================

def create_adaptive_v4_weights() -> dict:
    """
    Create V4 weight specifications for each regime.
    Based on factor analysis results.
    """

    # Default V4 (cash-flow focused, based on multivariate analysis)
    v4_default = {
        'cash_flow_composite': 30,  # Average of ROA + OCF + FCF
        'fcf_yield': 15,
        'inverse_asset_growth': 10,
        'inverse_volatility': 10,
        'gross_profitability': 10,
        'trend': 10,
        'valuation': 10,
        'momentum_12_1': 5,
    }

    # Bull market: Add momentum tilt
    v4_bull = {
        'cash_flow_composite': 25,
        'fcf_yield': 10,
        'inverse_asset_growth': 5,
        'inverse_volatility': 5,
        'gross_profitability': 10,
        'trend': 15,
        'valuation': 10,
        'momentum_12_1': 15,  # Momentum works in bull
        'momentum_6m': 5,
    }

    # Bear market: Defensive, quality focus
    v4_bear = {
        'cash_flow_composite': 35,
        'fcf_yield': 15,
        'inverse_asset_growth': 15,
        'inverse_volatility': 15,  # Low vol very important
        'gross_profitability': 10,
        'trend': 5,
        'valuation': 5,
        'momentum_12_1': 0,  # Momentum fails in bear
    }

    # High volatility: Maximum defensive
    v4_high_vol = {
        'cash_flow_composite': 30,
        'fcf_yield': 15,
        'inverse_asset_growth': 10,
        'inverse_volatility': 20,  # Very high weight on low vol
        'gross_profitability': 15,
        'trend': 5,
        'valuation': 5,
        'momentum_12_1': 0,
    }

    # Recovery: Balance quality with momentum
    v4_recovery = {
        'cash_flow_composite': 25,
        'fcf_yield': 10,
        'inverse_asset_growth': 5,
        'inverse_volatility': 10,
        'gross_profitability': 10,
        'trend': 15,
        'valuation': 10,
        'momentum_12_1': 10,
        'momentum_6m': 5,
    }

    return {
        'Default': v4_default,
        'Bull': v4_bull,
        'Bear': v4_bear,
        'High_Vol': v4_high_vol,
        'Recovery': v4_recovery,
    }


def print_v4_weight_comparison():
    """Print V4 weights for each regime."""
    print("\n" + "=" * 70)
    print("PROPOSED V4 ADAPTIVE WEIGHTS")
    print("=" * 70)

    weights = create_adaptive_v4_weights()

    # Get all factors
    all_factors = set()
    for regime_weights in weights.values():
        all_factors.update(regime_weights.keys())
    all_factors = sorted(all_factors)

    # Print header
    regimes = list(weights.keys())
    header = f"{'Factor':<25}" + "".join(f"{r:>12}" for r in regimes)
    print(f"\n{header}")
    print("-" * (25 + 12 * len(regimes)))

    # Print each factor
    for factor in all_factors:
        row = f"{factor:<25}"
        for regime in regimes:
            val = weights[regime].get(factor, 0)
            row += f"{val:>11.0f}pt" if val > 0 else f"{'—':>12}"
        print(row)

    # Print totals
    print("-" * (25 + 12 * len(regimes)))
    row = f"{'TOTAL':<25}"
    for regime in regimes:
        total = sum(weights[regime].values())
        row += f"{total:>11.0f}pt"
    print(row)


# ==================== BACKTEST ADAPTIVE VS STATIC ====================

def backtest_adaptive_strategy(df: pd.DataFrame, regimes: pd.DataFrame) -> dict:
    """
    Compare adaptive strategy vs static V4 strategy.
    """
    print("\n" + "=" * 70)
    print("BACKTESTING: ADAPTIVE VS STATIC STRATEGY")
    print("=" * 70)

    # Merge regimes
    df = pd.merge_asof(
        df.sort_values('date'),
        regimes[['date', 'regime']].sort_values('date'),
        on='date', direction='backward'
    )

    # Compute z-scores for factors
    factors = ['roa', 'ocf_assets', 'fcf_assets', 'fcf_yield', 'gp_assets',
               'vol_60d', 'mom_12_1', 'asset_growth']

    for factor in factors:
        if factor in df.columns:
            df[f'{factor}_z'] = (df[factor] - df[factor].mean()) / df[factor].std()
            df[f'{factor}_z'] = df[f'{factor}_z'].clip(-3, 3)

    # Static V4 score (cash-flow focused)
    df['static_score'] = (
        df['roa_z'].fillna(0) * 0.12 +
        df['ocf_assets_z'].fillna(0) * 0.12 +
        df['fcf_assets_z'].fillna(0) * 0.12 +
        df['fcf_yield_z'].fillna(0) * 0.15 +
        df['gp_assets_z'].fillna(0) * 0.10 +
        (-df['vol_60d_z'].fillna(0)) * 0.10 +
        (-df['asset_growth_z'].fillna(0)) * 0.10 +
        df['mom_12_1_z'].fillna(0) * 0.05
    )

    # Adaptive V4 score (regime-dependent)
    def get_adaptive_score(row):
        regime = row.get('regime', 'Default')

        if regime == 'Bull':
            return (
                row.get('roa_z', 0) * 0.10 +
                row.get('ocf_assets_z', 0) * 0.10 +
                row.get('fcf_assets_z', 0) * 0.10 +
                row.get('fcf_yield_z', 0) * 0.10 +
                row.get('gp_assets_z', 0) * 0.10 +
                (-row.get('vol_60d_z', 0)) * 0.05 +
                (-row.get('asset_growth_z', 0)) * 0.05 +
                row.get('mom_12_1_z', 0) * 0.15  # Higher momentum weight
            )
        elif regime == 'Bear':
            return (
                row.get('roa_z', 0) * 0.15 +
                row.get('ocf_assets_z', 0) * 0.15 +
                row.get('fcf_assets_z', 0) * 0.15 +
                row.get('fcf_yield_z', 0) * 0.15 +
                row.get('gp_assets_z', 0) * 0.10 +
                (-row.get('vol_60d_z', 0)) * 0.15 +  # High weight on low vol
                (-row.get('asset_growth_z', 0)) * 0.15 +
                row.get('mom_12_1_z', 0) * 0.00  # Zero momentum
            )
        elif regime == 'High_Vol':
            return (
                row.get('roa_z', 0) * 0.12 +
                row.get('ocf_assets_z', 0) * 0.12 +
                row.get('fcf_assets_z', 0) * 0.12 +
                row.get('fcf_yield_z', 0) * 0.15 +
                row.get('gp_assets_z', 0) * 0.10 +
                (-row.get('vol_60d_z', 0)) * 0.20 +  # Very high weight on low vol
                (-row.get('asset_growth_z', 0)) * 0.10 +
                row.get('mom_12_1_z', 0) * 0.00
            )
        else:  # Recovery or Default
            return (
                row.get('roa_z', 0) * 0.12 +
                row.get('ocf_assets_z', 0) * 0.12 +
                row.get('fcf_assets_z', 0) * 0.12 +
                row.get('fcf_yield_z', 0) * 0.12 +
                row.get('gp_assets_z', 0) * 0.10 +
                (-row.get('vol_60d_z', 0)) * 0.10 +
                (-row.get('asset_growth_z', 0)) * 0.08 +
                row.get('mom_12_1_z', 0) * 0.10
            )

    df['adaptive_score'] = df.apply(get_adaptive_score, axis=1)

    # Compute Q5-Q1 spreads for each strategy
    clean = df.dropna(subset=['fwd_3m', 'static_score', 'adaptive_score', 'regime'])

    print(f"\nBacktest observations: {len(clean):,}")

    # Overall performance
    results = {}

    for strategy in ['static_score', 'adaptive_score']:
        corr = clean[strategy].corr(clean['fwd_3m'])

        q5 = clean[clean[strategy] > clean[strategy].quantile(0.8)]['fwd_3m'].mean()
        q1 = clean[clean[strategy] < clean[strategy].quantile(0.2)]['fwd_3m'].mean()
        spread = q5 - q1

        results[strategy] = {'correlation': corr, 'spread': spread, 'q5_return': q5, 'q1_return': q1}

    print("\n=== Overall Performance ===")
    print(f"{'Strategy':<20} {'Correlation':>12} {'Q5-Q1 Spread':>14} {'Q5 Ret':>10} {'Q1 Ret':>10}")
    print("-" * 70)

    for strategy, stats in results.items():
        name = 'Static V4' if 'static' in strategy else 'Adaptive V4'
        print(f"{name:<20} {stats['correlation']:>+11.4f} {stats['spread']:>+13.2f}% "
              f"{stats['q5_return']:>+9.2f}% {stats['q1_return']:>+9.2f}%")

    # Performance by regime
    print("\n=== Performance by Regime ===")

    for regime in ['Bull', 'Bear', 'High_Vol', 'Recovery']:
        regime_df = clean[clean['regime'] == regime]

        if len(regime_df) < 200:
            continue

        print(f"\n--- {regime} ({len(regime_df):,} obs) ---")

        for strategy in ['static_score', 'adaptive_score']:
            corr = regime_df[strategy].corr(regime_df['fwd_3m'])

            q5 = regime_df[regime_df[strategy] > regime_df[strategy].quantile(0.8)]['fwd_3m'].mean()
            q1 = regime_df[regime_df[strategy] < regime_df[strategy].quantile(0.2)]['fwd_3m'].mean()
            spread = q5 - q1

            name = 'Static V4' if 'static' in strategy else 'Adaptive V4'
            print(f"  {name:<18} r={corr:>+.4f}  spread={spread:>+.2f}%")

    return results


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("MULTIVARIATE FACTOR ANALYSIS & REGIME-ADAPTIVE SCORING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    df, spy = load_factor_data()

    # Detect regimes
    regimes = detect_regimes(spy)

    # Multivariate regression
    regression_results = run_multivariate_regression(df)

    # Factor performance by regime
    regime_results = analyze_factors_by_regime(df, regimes)

    # Compute optimal weights by regime
    optimal_weights = compute_optimal_weights_by_regime(regime_results)

    # Print proposed V4 weights
    print_v4_weight_comparison()

    # Backtest adaptive vs static
    backtest_results = backtest_adaptive_strategy(df, regimes)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. MULTIVARIATE SURVIVING FACTORS:")
    if regression_results and 'factors' in regression_results:
        significant = [(f, v['t_statistic']) for f, v in regression_results['factors'].items()
                       if v['significant']]
        significant.sort(key=lambda x: abs(x[1]), reverse=True)
        for factor, t in significant:
            print(f"   {factor:<20} t={t:>+.2f}")

    print("\n2. REGIME-SPECIFIC INSIGHTS:")
    print("   - Bull markets: Momentum adds value (+15pt weight)")
    print("   - Bear markets: Low volatility critical (+15pt weight), zero momentum")
    print("   - High vol: Maximum defensive posture, low vol +20pt")
    print("   - Recovery: Balance quality with momentum")

    print("\n3. ADAPTIVE STRATEGY BENEFIT:")
    if backtest_results:
        static_spread = backtest_results.get('static_score', {}).get('spread', 0)
        adaptive_spread = backtest_results.get('adaptive_score', {}).get('spread', 0)
        improvement = adaptive_spread - static_spread
        print(f"   Static V4 spread:   {static_spread:>+.2f}%")
        print(f"   Adaptive V4 spread: {adaptive_spread:>+.2f}%")
        print(f"   Improvement:        {improvement:>+.2f}%")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Blue Chip Score Analysis
========================
Identify factors that predict strong 2+ year returns for quality compounders.
These are Apple/Microsoft/JNJ-type stocks you hold for years and let compound.

Focus on DURABILITY and CONSISTENCY:
- Margin stability (not just high margins, but stable margins)
- Revenue consistency (steady growth, not volatile)
- Strong balance sheet (low debt, high cash)
- High and stable ROA/ROIC
- GP/Assets (proven important at longer horizons)
- Valuation discipline (don't overpay for quality)

Target Horizon: 2-3 years (Compass Score handles <1.5 years well)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'

# Horizons for Blue Chip (long-term focus)
HORIZONS = {
    '2-Year': 504,
    '3-Year': 756
}


def log(msg):
    """Print with flush for real-time output."""
    print(msg, flush=True)


def load_data():
    """Load price and fundamental data."""
    log("Loading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    # Prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    log(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    # Income statements (for margin stability)
    income = pd.read_sql_query("""
        SELECT symbol, date, revenue, gross_profit, operating_income, net_income
        FROM historical_income_statements
        WHERE revenue > 0
    """, conn)
    income['date'] = pd.to_datetime(income['date'])
    log(f"  {len(income):,} income statement records")

    # Balance sheets (for balance sheet strength)
    balance = pd.read_sql_query("""
        SELECT symbol, date, total_assets, total_liabilities,
               total_debt, cash_and_equivalents, total_equity
        FROM historical_balance_sheets
        WHERE total_assets > 0
    """, conn)
    balance['date'] = pd.to_datetime(balance['date'])
    log(f"  {len(balance):,} balance sheet records")

    # Cash flows
    cash = pd.read_sql_query("""
        SELECT symbol, date, operating_cash_flow, free_cash_flow
        FROM historical_cash_flows
    """, conn)
    cash['date'] = pd.to_datetime(cash['date'])
    log(f"  {len(cash):,} cash flow records")

    conn.close()
    return prices, income, balance, cash


def compute_stability_metrics(income, balance, cash):
    """Compute stability and consistency metrics using vectorized operations."""
    log("\nComputing stability metrics...")

    # Merge all fundamentals
    fund = income.merge(balance, on=['symbol', 'date'], how='inner')
    fund = fund.merge(cash, on=['symbol', 'date'], how='inner')
    fund = fund.sort_values(['symbol', 'date']).reset_index(drop=True)
    log(f"  {len(fund):,} merged fundamental records")

    # Point-in-time metrics
    fund['gross_margin'] = fund['gross_profit'] / fund['revenue']
    fund['operating_margin'] = fund['operating_income'] / fund['revenue']
    fund['net_margin'] = fund['net_income'] / fund['revenue']
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['roe'] = fund['net_income'] / fund['total_equity']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']

    # Balance sheet strength
    fund['debt_to_equity'] = fund['total_debt'] / fund['total_equity']
    fund['debt_to_assets'] = fund['total_debt'] / fund['total_assets']
    fund['cash_to_assets'] = fund['cash_and_equivalents'] / fund['total_assets']
    fund['equity_ratio'] = fund['total_equity'] / fund['total_assets']

    # Revenue growth (YoY)
    fund['revenue_growth'] = fund.groupby('symbol')['revenue'].pct_change(4)

    # Asset growth
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities
    numeric_cols = fund.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # STABILITY METRICS using rolling windows (vectorized)
    log("  Computing rolling stability metrics...")

    # Rolling 8-quarter (2-year) standard deviations
    fund['gross_margin_std'] = fund.groupby('symbol')['gross_margin'].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    fund['operating_margin_std'] = fund.groupby('symbol')['operating_margin'].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    fund['roa_std'] = fund.groupby('symbol')['roa'].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    fund['revenue_growth_std'] = fund.groupby('symbol')['revenue_growth'].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )

    # Coefficient of variation
    fund['gross_margin_cv'] = fund['gross_margin_std'] / fund['gross_margin'].abs()
    fund['roa_cv'] = fund['roa_std'] / fund['roa'].abs()

    # Trend (current vs rolling mean)
    fund['gross_margin_mean'] = fund.groupby('symbol')['gross_margin'].transform(
        lambda x: x.rolling(8, min_periods=4).mean()
    )
    fund['gross_margin_trend'] = fund['gross_margin'] - fund['gross_margin_mean']

    fund['roa_mean'] = fund.groupby('symbol')['roa'].transform(
        lambda x: x.rolling(8, min_periods=4).mean()
    )
    fund['roa_trend'] = fund['roa'] - fund['roa_mean']

    # Drop rows without stability metrics
    fund = fund.dropna(subset=['gross_margin_std', 'roa_std'])
    log(f"  {len(fund):,} records with stability metrics")

    return fund


def compute_price_factors(prices, symbols, horizons):
    """Compute volatility and forward returns."""
    log(f"\nComputing price factors for {len(symbols)} symbols...")

    max_horizon = max(horizons.values())
    results = []

    symbols_list = list(symbols)
    for i, symbol in enumerate(symbols_list):
        if i % 500 == 0:
            log(f"  {i}/{len(symbols_list)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300 + max_horizon:
            continue

        close = sym_prices['close'].values
        dates = sym_prices['date'].values

        # Sample quarterly
        for j in range(252, n - max_horizon, 63):
            date = dates[j]
            price = close[j]

            # Volatility (60-day trailing)
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Long-term volatility (252-day trailing)
            if j >= 252:
                rets_1y = np.diff(close[j-252:j+1]) / close[j-252:j]
                vol_1y = np.std(rets_1y) * np.sqrt(252) * 100
            else:
                vol_1y = np.nan

            # 12-month momentum
            if j >= 252 and close[j-252] > 0:
                mom_12m = ((close[j] / close[j-252]) - 1) * 100
            else:
                mom_12m = np.nan

            row = {
                'symbol': symbol,
                'date': pd.Timestamp(date),
                'price': price,
                'vol_60d': vol_60d,
                'vol_1y': vol_1y,
                'mom_12m': np.clip(mom_12m, -90, 500) if mom_12m is not None else np.nan,
            }

            # Forward returns
            for horizon_name, days in horizons.items():
                if j + days < n:
                    fwd_ret = ((close[j + days] / close[j]) - 1) * 100
                    row[f'fwd_{horizon_name}'] = np.clip(fwd_ret, -100, 1000)
                else:
                    row[f'fwd_{horizon_name}'] = np.nan

            results.append(row)

    df = pd.DataFrame(results)
    log(f"  {len(df):,} price observations")
    return df


def merge_data(price_df, stability_df):
    """Merge price and stability data using backward-looking fundamentals."""
    log("\nMerging price and stability data...")

    all_merged = []
    symbols = price_df['symbol'].unique()

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            log(f"  {i}/{len(symbols)} symbols...")

        pf = price_df[price_df['symbol'] == symbol].sort_values('date')
        sf = stability_df[stability_df['symbol'] == symbol].sort_values('date')

        if len(sf) == 0:
            continue

        merged = pd.merge_asof(pf, sf.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('400 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    log(f"  {len(df):,} merged observations")

    # Diagnostic: how many have fundamental data?
    has_fundamentals = df.dropna(subset=['gp_assets', 'roa']).shape[0]
    log(f"  {has_fundamentals:,} observations with fundamental data ({100*has_fundamentals/len(df):.1f}%)")

    return df


def univariate_analysis(df, factor, horizon_name, return_col):
    """Analyze a single factor's predictive power."""
    clean = df.dropna(subset=[factor, return_col])

    if len(clean) < 500:
        return None

    # Correlation
    corr = clean[factor].corr(clean[return_col])

    # Quintile analysis
    try:
        clean['quintile'] = pd.qcut(clean[factor], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    except ValueError:
        clean['quintile'] = pd.cut(clean[factor].rank(pct=True),
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quintile_returns = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = clean[clean['quintile'] == q]
        quintile_returns[q] = q_data[return_col].mean()

    spread = quintile_returns['Q5'] - quintile_returns['Q1']

    # T-test for spread significance
    q5_returns = clean[clean['quintile'] == 'Q5'][return_col]
    q1_returns = clean[clean['quintile'] == 'Q1'][return_col]
    t_stat, p_val = stats.ttest_ind(q5_returns, q1_returns, equal_var=False)

    return {
        'factor': factor,
        'n_obs': len(clean),
        'correlation': corr,
        'Q1': quintile_returns['Q1'],
        'Q5': quintile_returns['Q5'],
        'spread': spread,
        't_stat': t_stat,
        'p_val': p_val
    }


def multivariate_regression(is_df, factors, return_col):
    """Run multivariate regression to find optimal weights."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    clean = is_df.dropna(subset=factors + [return_col])

    if len(clean) < 500:
        return None, None

    X = clean[factors].values
    y = clean[return_col].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit
    model = LinearRegression()
    model.fit(X_scaled, y)

    # Coefficients
    coefs = dict(zip(factors, model.coef_))

    # Normalize to sum to 1 (absolute values)
    total = sum(abs(c) for c in coefs.values())
    weights = {f: c / total for f, c in coefs.items()}

    return coefs, weights


def compute_blue_chip_score(df, weights, is_df):
    """Compute Blue Chip Score using provided weights."""
    log("\nComputing Blue Chip Score...")

    factors = list(weights.keys())

    # Winsorize using in-sample bounds
    for col in factors:
        if col in df.columns:
            lower = is_df[col].quantile(0.01)
            upper = is_df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)

    # Z-score using in-sample stats
    for col in factors:
        if col in df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
            df[f'{col}_z'] = ((df[col] - mean) / std).clip(-3, 3).fillna(0)

    # Compute score
    df['blue_chip_score'] = 0
    for factor, weight in weights.items():
        if f'{factor}_z' in df.columns:
            df['blue_chip_score'] += df[f'{factor}_z'] * weight

    return df


def validate_score(df, horizon_name, return_col):
    """Validate score with quintile analysis."""
    clean = df.dropna(subset=['blue_chip_score', return_col])

    if len(clean) < 500:
        return None

    # Quintiles
    try:
        clean['quintile'] = pd.qcut(clean['blue_chip_score'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
    except ValueError:
        clean['quintile'] = pd.cut(clean['blue_chip_score'].rank(pct=True),
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    results = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = clean[clean['quintile'] == q]
        results[q] = {
            'n': len(q_data),
            'mean_return': q_data[return_col].mean(),
            'median_return': q_data[return_col].median(),
            'std': q_data[return_col].std()
        }

    spread = results['Q5']['mean_return'] - results['Q1']['mean_return']
    correlation = clean['blue_chip_score'].corr(clean[return_col])

    # Monotonicity check
    returns_list = [results[q]['mean_return'] for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']]
    monotonic = all(returns_list[i] <= returns_list[i+1] for i in range(4))

    return {
        'horizon': horizon_name,
        'n_obs': len(clean),
        'correlation': correlation,
        'quintiles': results,
        'spread': spread,
        'monotonic': monotonic
    }


def main():
    log("=" * 70)
    log("BLUE CHIP SCORE ANALYSIS")
    log("Finding factors that predict 2+ year returns for quality compounders")
    log("=" * 70)
    log(f"Started: {datetime.now()}")
    log(f"\nHorizons: {', '.join(HORIZONS.keys())}")
    log(f"In-Sample: ≤ {IN_SAMPLE_END}")
    log(f"Out-of-Sample: ≥ {OOS_START}")

    # Load data
    prices, income, balance, cash = load_data()

    # Compute stability metrics (vectorized)
    stability_df = compute_stability_metrics(income, balance, cash)

    # Get symbols with stability data
    symbols = stability_df['symbol'].unique()
    log(f"\n{len(symbols)} symbols with stability metrics")

    # Sample for speed if needed
    np.random.seed(42)
    if len(symbols) > 2000:
        symbols = np.random.choice(symbols, size=2000, replace=False)
        log(f"Sampled to {len(symbols)} symbols for speed")

    # Compute price factors
    price_df = compute_price_factors(prices, symbols, HORIZONS)

    # Merge
    df = merge_data(price_df, stability_df)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Split IS/OOS
    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()

    log(f"\nIn-Sample:  {len(is_df):,} observations")
    log(f"Out-of-Sample: {len(oos_df):,} observations")

    # ==========================================
    # UNIVARIATE ANALYSIS
    # ==========================================

    # Factors to test
    factors_to_test = [
        # Point-in-time quality
        'roa', 'roe', 'gp_assets', 'ocf_assets', 'fcf_assets',
        'gross_margin', 'operating_margin', 'net_margin',

        # Balance sheet strength
        'debt_to_equity', 'debt_to_assets', 'cash_to_assets', 'equity_ratio',

        # Growth
        'revenue_growth', 'asset_growth',

        # STABILITY (lower std = more stable = better)
        'gross_margin_std', 'operating_margin_std', 'roa_std', 'revenue_growth_std',

        # Consistency (coefficient of variation)
        'gross_margin_cv', 'roa_cv',

        # Trend
        'gross_margin_trend', 'roa_trend',

        # Price-based
        'vol_60d', 'vol_1y', 'mom_12m'
    ]

    for horizon_name, days in HORIZONS.items():
        return_col = f'fwd_{horizon_name}'

        log("\n" + "=" * 70)
        log(f"UNIVARIATE ANALYSIS: {horizon_name}")
        log("=" * 70)

        results = []
        for factor in factors_to_test:
            if factor not in is_df.columns:
                continue
            result = univariate_analysis(is_df, factor, horizon_name, return_col)
            if result:
                results.append(result)

        # Sort by absolute spread
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('spread', key=abs, ascending=False)

        log(f"\n{'Factor':<22} {'N':>10} {'Corr':>8} {'Q1':>10} {'Q5':>10} {'Spread':>10} {'T-stat':>8}")
        log("-" * 90)

        for _, row in results_df.iterrows():
            sig = "**" if abs(row['t_stat']) > 2 else "*" if abs(row['t_stat']) > 1.5 else ""
            log(f"{row['factor']:<22} {row['n_obs']:>10,} {row['correlation']:>+7.3f} "
                  f"{row['Q1']:>+9.2f}% {row['Q5']:>+9.2f}% {row['spread']:>+9.2f}% {row['t_stat']:>+7.2f}{sig}")

    # ==========================================
    # MULTIVARIATE REGRESSION (2-Year Focus)
    # ==========================================

    log("\n" + "=" * 70)
    log("MULTIVARIATE REGRESSION (2-Year Horizon)")
    log("=" * 70)

    # Select promising factors
    candidate_factors = [
        'gp_assets',           # Quality
        'roa',                 # Quality
        'fcf_assets',          # Quality
        'gross_margin_std',    # Stability
        'roa_std',             # Stability
        'debt_to_assets',      # Balance sheet
        'equity_ratio',        # Balance sheet
        'revenue_growth',      # Growth
        'vol_1y',              # Risk
    ]

    # Filter to available factors
    available_factors = [f for f in candidate_factors if f in is_df.columns]

    return_col = 'fwd_2-Year'
    coefs, weights = multivariate_regression(is_df, available_factors, return_col)

    if coefs:
        log(f"\nRegression Coefficients (standardized):")
        log("-" * 50)
        for factor, coef in sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = "Higher → Better" if coef > 0 else "Lower → Better"
            log(f"  {factor:<22}: {coef:>+8.4f}  ({direction})")

    # ==========================================
    # CREATE BLUE CHIP SCORE
    # ==========================================

    log("\n" + "=" * 70)
    log("BLUE CHIP SCORE CONSTRUCTION")
    log("=" * 70)

    # Manual weights based on univariate + multivariate analysis
    blue_chip_weights = {
        'gp_assets': 0.20,           # Gross profit / assets (quality)
        'roa': 0.15,                 # Return on assets
        'fcf_assets': 0.15,          # Free cash flow / assets
        'gross_margin_std': -0.15,   # NEGATIVE - lower std is better (stability)
        'roa_std': -0.10,            # NEGATIVE - lower std is better (stability)
        'debt_to_assets': -0.10,     # NEGATIVE - lower debt is better
        'equity_ratio': 0.10,        # Higher equity ratio is better
        'vol_1y': -0.05,             # NEGATIVE - lower vol is better
    }

    log("\nBlue Chip Score Weights:")
    log("-" * 50)
    for factor, weight in blue_chip_weights.items():
        direction = "Higher → Better" if weight > 0 else "Lower → Better"
        log(f"  {factor:<22}: {weight:>+6.0%}  ({direction})")

    # Compute score
    is_df = compute_blue_chip_score(is_df, blue_chip_weights, is_df)
    oos_df = compute_blue_chip_score(oos_df, blue_chip_weights, is_df)  # Use IS stats

    # ==========================================
    # VALIDATION
    # ==========================================

    log("\n" + "=" * 70)
    log("OUT-OF-SAMPLE VALIDATION (2020+)")
    log("=" * 70)

    for horizon_name, days in HORIZONS.items():
        return_col = f'fwd_{horizon_name}'
        result = validate_score(oos_df, horizon_name, return_col)

        if result:
            log(f"\n{horizon_name} Results:")
            log("-" * 50)
            log(f"  N Observations: {result['n_obs']:,}")
            log(f"  Correlation: {result['correlation']:+.4f}")
            log(f"  Monotonic: {'YES' if result['monotonic'] else 'NO'}")
            log(f"\n  {'Quintile':<10} {'N':>8} {'Mean Return':>14} {'Std':>10}")
            log("  " + "-" * 45)
            for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                qr = result['quintiles'][q]
                log(f"  {q:<10} {qr['n']:>8,} {qr['mean_return']:>+13.2f}% {qr['std']:>9.2f}%")
            log(f"\n  Q5-Q1 Spread: {result['spread']:+.2f}%")

    # ==========================================
    # IN-SAMPLE COMPARISON
    # ==========================================

    log("\n" + "=" * 70)
    log("IN-SAMPLE VALIDATION (Reference)")
    log("=" * 70)

    for horizon_name, days in HORIZONS.items():
        return_col = f'fwd_{horizon_name}'
        result = validate_score(is_df, horizon_name, return_col)

        if result:
            log(f"\n{horizon_name}: Q5-Q1 Spread = {result['spread']:+.2f}%, "
                  f"Corr = {result['correlation']:+.4f}, "
                  f"Monotonic = {'YES' if result['monotonic'] else 'NO'}")

    # ==========================================
    # COMPARISON WITH COMPASS SCORE
    # ==========================================

    log("\n" + "=" * 70)
    log("COMPARISON: BLUE CHIP vs COMPASS SCORE APPROACH")
    log("=" * 70)

    # Also compute Compass Score for comparison
    compass_weights = {
        'roa': 0.20,
        'ocf_assets': 0.15,
        'fcf_assets': 0.15,
        'gp_assets': 0.20,
        'vol_60d': -0.15,
        'asset_growth': -0.15,
    }

    # Filter to common factors
    compass_available = {k: v for k, v in compass_weights.items() if k in oos_df.columns}

    # Z-score compass factors
    for col in compass_available.keys():
        mean = is_df[col].mean()
        std = is_df[col].std()
        oos_df[f'{col}_compass_z'] = ((oos_df[col] - mean) / std).clip(-3, 3).fillna(0)

    # Compute compass score
    oos_df['compass_score'] = 0
    for factor, weight in compass_available.items():
        oos_df['compass_score'] += oos_df[f'{factor}_compass_z'] * weight

    log(f"\n{'Horizon':<12} {'Score':<15} {'Q5-Q1 Spread':>14} {'Corr':>10} {'Mono':>8}")
    log("-" * 65)

    for horizon_name in HORIZONS.keys():
        return_col = f'fwd_{horizon_name}'

        # Blue Chip
        bc_result = validate_score(oos_df, horizon_name, return_col)

        # Compass (using blue_chip_score column temporarily)
        oos_temp = oos_df.copy()
        oos_temp['blue_chip_score'] = oos_temp['compass_score']
        compass_result = validate_score(oos_temp, horizon_name, return_col)

        if bc_result and compass_result:
            log(f"{horizon_name:<12} {'Blue Chip':<15} {bc_result['spread']:>+13.2f}% "
                  f"{bc_result['correlation']:>+9.4f} {'YES' if bc_result['monotonic'] else 'NO':>8}")
            log(f"{'':<12} {'Compass':<15} {compass_result['spread']:>+13.2f}% "
                  f"{compass_result['correlation']:>+9.4f} {'YES' if compass_result['monotonic'] else 'NO':>8}")

    log(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

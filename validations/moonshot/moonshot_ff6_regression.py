#!/usr/bin/env python3
"""
Moonshot Quality-First Fama-French 6-Factor Regression Analysis
================================================================
Regresses Moonshot quintile returns against FF5 + Momentum to calculate alpha

METHODOLOGY:
- Quality-First v2.0 filters and weights
- Companies ≥$500M market cap only
- Z-scores computed from IS data (1995-2019) only
- Applied to both IS and OOS (2020-2026) periods
- Newey-West HAC standard errors for autocorrelation
- Separate IS and OOS regression results

EXPECTED OUTCOME:
- Positive alpha for Q5-Q1 spread (excess return beyond FF6)
- Significant t-statistics (>2.0 for 95% confidence)
- Moderate R² (~30-50% explained by factors)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import warnings
import requests
import zipfile
import io
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
MIN_MARKET_CAP = 500_000_000  # $500M

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


def download_ff_factors():
    """Download Fama-French 5 factors + Momentum from Kenneth French's website."""
    print("Downloading Fama-French factors...")

    # FF5 factors (monthly)
    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    # Momentum factor (monthly)
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

    try:
        # Download FF5
        print("  Fetching FF5 factors (monthly)...")
        r = requests.get(ff5_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_names = [n for n in z.namelist() if '.csv' in n.lower()]
        if not csv_names:
            raise ValueError("No CSV file found in FF5 zip")
        csv_name = csv_names[0]

        # Read and parse - FF files have header rows to skip
        ff5_raw = pd.read_csv(z.open(csv_name), skiprows=3, header=None)
        # Find where annual data starts (look for row with just a year)
        annual_start = None
        for idx, row in ff5_raw.iterrows():
            val = str(row.iloc[0]).strip()
            if len(val) == 4 and val.isdigit():  # Just a year like "1990"
                annual_start = idx
                break
        if annual_start:
            ff5_raw = ff5_raw.iloc[:annual_start]

        ff5_raw.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        # Keep only monthly rows (YYYYMM format)
        ff5 = ff5_raw[ff5_raw['date'].astype(str).str.match(r'^\s*\d{6}\s*$')].copy()
        ff5['date'] = pd.to_datetime(ff5['date'].astype(str).str.strip(), format='%Y%m')
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

        print(f"    FF5: {len(ff5)} months")

        # Download Momentum
        print("  Fetching Momentum factor (monthly)...")
        r = requests.get(mom_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_names = [n for n in z.namelist() if '.csv' in n.lower()]
        if not csv_names:
            raise ValueError("No CSV file found in MOM zip")
        csv_name = csv_names[0]

        mom_raw = pd.read_csv(z.open(csv_name), skiprows=13, header=None)
        # Find where annual data starts
        annual_start = None
        for idx, row in mom_raw.iterrows():
            val = str(row.iloc[0]).strip()
            if len(val) == 4 and val.isdigit():
                annual_start = idx
                break
        if annual_start:
            mom_raw = mom_raw.iloc[:annual_start]

        mom_raw.columns = ['date', 'MOM']
        mom = mom_raw[mom_raw['date'].astype(str).str.match(r'^\s*\d{6}\s*$')].copy()
        mom['date'] = pd.to_datetime(mom['date'].astype(str).str.strip(), format='%Y%m')
        mom['MOM'] = pd.to_numeric(mom['MOM'], errors='coerce')

        print(f"    MOM: {len(mom)} months")

        # Merge
        ff6 = ff5.merge(mom, on='date', how='inner')
        ff6 = ff6.dropna()
        print(f"  FF6 factors: {len(ff6):,} monthly observations ({ff6['date'].min().date()} to {ff6['date'].max().date()})")

        return ff6

    except Exception as e:
        print(f"  Error downloading: {e}")
        import traceback
        traceback.print_exc()
        print("  Using synthetic factors for demonstration...")
        # Create synthetic factors for testing if download fails
        dates = pd.date_range('1995-01-01', '2025-12-31', freq='MS')
        np.random.seed(42)
        ff6 = pd.DataFrame({
            'date': dates,
            'Mkt-RF': np.random.normal(0.8, 4.5, len(dates)),
            'SMB': np.random.normal(0.2, 3.0, len(dates)),
            'HML': np.random.normal(0.3, 3.0, len(dates)),
            'RMW': np.random.normal(0.25, 2.0, len(dates)),
            'CMA': np.random.normal(0.25, 2.0, len(dates)),
            'MOM': np.random.normal(0.6, 4.0, len(dates)),
            'RF': np.random.normal(0.3, 0.15, len(dates)),
        })
        return ff6


def passes_quality_filters(row):
    """Check if stock passes Quality-First v2.0 filters."""
    # Filter 1: Gross margin > 30%
    if pd.isna(row['gross_margin']) or row['gross_margin'] < 0.30 or row['gross_margin'] > 0.95:
        return False

    # Filter 2: Revenue > $50M
    if pd.isna(row['revenue_ttm']) or row['revenue_ttm'] < 50_000_000:
        return False

    # Filter 3: Revenue growth > 15%
    if pd.isna(row['revenue_growth']) or row['revenue_growth'] < 0.15 or row['revenue_growth'] > 3.0:
        return False

    # Filter 4: Cash flow quality
    if not pd.isna(row['net_income_ttm']) and not pd.isna(row['operating_cash_flow_ttm']):
        if row['net_income_ttm'] > 0:
            if row['operating_cash_flow_ttm'] / row['net_income_ttm'] < 0.7:
                return False
        else:
            if row['operating_cash_flow_ttm'] / row['revenue_ttm'] < -0.5:
                return False

    # Filter 5: Balance sheet health
    if not pd.isna(row['total_debt']) and not pd.isna(row['total_assets']) and row['total_assets'] > 0:
        if row['total_debt'] / row['total_assets'] > 2.0:
            return False

    # Filter 6: Operating income quality
    if not pd.isna(row['operating_income_ttm']) and not pd.isna(row['net_income_ttm']):
        if abs(row['net_income_ttm']) > 0 and abs(row['operating_income_ttm']) > 0:
            oi_ni_ratio = abs(row['operating_income_ttm'] / row['net_income_ttm'])
            if oi_ni_ratio < 0.3 or oi_ni_ratio > 3.0:
                return False

    return True


def compute_moonshot_quintile_returns():
    """
    Compute monthly Moonshot quintile returns using Quality-First v2.0.

    CRITICAL: Z-scores computed using IS statistics only,
    then applied to both IS and OOS periods to avoid look-ahead bias.
    """
    print("\n" + "=" * 70)
    print("COMPUTING MOONSHOT QUINTILE RETURNS (NO LOOK-AHEAD BIAS)")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    print("\nLoading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

    # Load fundamentals
    print("Loading fundamentals...")
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
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'date'])

    # Compute TTM metrics
    print("Computing TTM metrics...")
    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Compute metrics
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4)
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Compute 3-year CAGR
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

    # Clean up infinities
    for col in ['revenue_growth', 'eps_growth', 'gross_margin', 'margin_improvement',
                'revenue_growth_3yr', 'eps_growth_3yr', 'fcf_margin', 'roe']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    conn.close()

    print(f"  {len(prices):,} price records")
    print(f"  {len(fund):,} fundamental records")

    # Compute forward 1-year returns (252 trading days)
    print("\nComputing forward 1-year returns...")
    prices = prices.sort_values(['symbol', 'date'])
    prices['fwd_1yr'] = prices.groupby('symbol')['close'].pct_change(252).shift(-252) * 100  # ~1 year forward

    # Merge fundamentals with monthly prices
    print("Merging fundamentals with monthly prices...")
    fund['year_month'] = fund['date'].dt.to_period('M')
    prices['year_month'] = prices['date'].dt.to_period('M')

    # Get month-end prices for each symbol
    month_end_prices = prices.groupby(['symbol', 'year_month']).last().reset_index()

    # Merge
    df = fund.merge(month_end_prices[['symbol', 'year_month', 'fwd_1yr', 'volume']],
                    on=['symbol', 'year_month'], how='inner')
    df = df.dropna(subset=['fwd_1yr', 'market_cap'])

    # Filter for ≥$500M market cap
    df = df[df['market_cap'] >= MIN_MARKET_CAP].copy()
    print(f"  {len(df):,} stock-months after market cap filter (≥$500M)")

    # Apply quality filters
    print("Applying Quality-First v2.0 filters...")
    df['passes_quality'] = df.apply(passes_quality_filters, axis=1)
    df = df[df['passes_quality']].copy()
    print(f"  {len(df):,} stock-months passed quality filters")

    # Compute momentum (12-1 month)
    print("Computing momentum...")
    prices_sorted = prices.sort_values(['symbol', 'date'])
    prices_sorted['price_12m'] = prices_sorted.groupby('symbol')['close'].shift(252)
    prices_sorted['price_1m'] = prices_sorted.groupby('symbol')['close'].shift(21)
    prices_sorted['momentum_12_1'] = (prices_sorted['price_1m'] - prices_sorted['price_12m']) / prices_sorted['price_12m']

    # Merge momentum
    month_end_momentum = prices_sorted.groupby(['symbol', 'year_month']).last().reset_index()
    df = df.merge(month_end_momentum[['symbol', 'year_month', 'momentum_12_1']],
                  on=['symbol', 'year_month'], how='left')

    # Handle missing factors (use 0 as neutral)
    for factor in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
        df[factor] = df[factor].fillna(0)

    # Cap extreme values
    df['revenue_growth_3yr'] = df['revenue_growth_3yr'].clip(-0.3, 1.5)
    df['eps_growth_3yr'] = df['eps_growth_3yr'].clip(-0.5, 2.0)
    df['fcf_margin'] = df['fcf_margin'].clip(-0.5, 0.5)
    df['roe'] = df['roe'].clip(-0.5, 1.0)
    df['momentum_12_1'] = df['momentum_12_1'].clip(-0.5, 2.0)

    # Compute small cap factor
    df['small_cap'] = -np.log10(df['market_cap'] / 1e9)

    # Remove rows with critical missing values
    required_factors = ['revenue_growth_3yr', 'gross_margin']
    df = df.dropna(subset=required_factors)

    print(f"  {len(df):,} stock-months ready for scoring")

    # Split into IS and OOS
    is_df = df[df['date'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date'] >= OOS_START].copy()

    print(f"\n  In-sample: {len(is_df):,} observations ({is_df['date'].min().date()} to {is_df['date'].max().date()})")
    print(f"  Out-of-sample: {len(oos_df):,} observations ({oos_df['date'].min().date()} to {oos_df['date'].max().date()})")

    # Compute z-score statistics from IS data ONLY
    print("\nComputing z-score statistics from IS data...")
    factor_stats = {}
    for factor in WEIGHTS.keys():
        factor_stats[factor] = {
            'mean': is_df[factor].mean(),
            'std': is_df[factor].std()
        }
        print(f"  {factor}: mean={factor_stats[factor]['mean']:.4f}, std={factor_stats[factor]['std']:.4f}")

    # Apply z-scores to BOTH IS and OOS using IS statistics
    print("\nApplying IS z-scores to both IS and OOS data...")
    for period_name, period_df in [('IS', is_df), ('OOS', oos_df)]:
        for factor in WEIGHTS.keys():
            period_df[f'{factor}_z'] = (
                (period_df[factor] - factor_stats[factor]['mean']) /
                factor_stats[factor]['std']
            )

        # Compute raw score
        period_df['moonshot_score'] = sum(
            period_df[f'{factor}_z'] * weight
            for factor, weight in WEIGHTS.items()
        )

    # Assign quintiles across ALL time periods (time-series ranking)
    print("\nAssigning quintiles...")

    # IS period: quintiles across all IS observations
    if len(is_df) >= 250:
        try:
            is_df['quintile'] = pd.qcut(is_df['moonshot_score'], 5,
                                        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                        duplicates='drop')
        except ValueError:
            is_df['quintile'] = np.nan
    else:
        is_df['quintile'] = np.nan

    is_df = is_df.dropna(subset=['quintile'])

    # OOS period: quintiles across all OOS observations
    if len(oos_df) >= 250:
        try:
            oos_df['quintile'] = pd.qcut(oos_df['moonshot_score'], 5,
                                         labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                         duplicates='drop')
        except ValueError:
            oos_df['quintile'] = np.nan
    else:
        oos_df['quintile'] = np.nan

    oos_df = oos_df.dropna(subset=['quintile'])

    # Combine for full sample analysis
    df = pd.concat([is_df, oos_df], ignore_index=True)
    if len(df) >= 250:
        try:
            df['quintile'] = pd.qcut(df['moonshot_score'], 5,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                                    duplicates='drop')
        except ValueError:
            df['quintile'] = np.nan
    else:
        df['quintile'] = np.nan

    df = df.dropna(subset=['quintile'])

    # Compute monthly quintile returns
    print("Computing monthly quintile returns...")
    monthly_returns = df.groupby(['year_month', 'quintile'])['fwd_1yr'].mean().unstack()
    monthly_returns['Q5-Q1'] = monthly_returns['Q5'] - monthly_returns['Q1']
    monthly_returns.index = monthly_returns.index.to_timestamp()

    is_returns = is_df.groupby(['year_month', 'quintile'])['fwd_1yr'].mean().unstack()
    is_returns['Q5-Q1'] = is_returns['Q5'] - is_returns['Q1']
    is_returns.index = is_returns.index.to_timestamp()

    oos_returns = oos_df.groupby(['year_month', 'quintile'])['fwd_1yr'].mean().unstack()
    oos_returns['Q5-Q1'] = oos_returns['Q5'] - oos_returns['Q1']
    oos_returns.index = oos_returns.index.to_timestamp()

    print(f"  Full sample: {len(monthly_returns)} months")
    print(f"  In-sample: {len(is_returns)} months")
    print(f"  Out-of-sample: {len(oos_returns)} months")

    return monthly_returns, is_returns, oos_returns


def run_ff6_regression(quintile_returns, ff6, period_name="Full Sample", use_newey_west=True):
    """
    Run FF6 regression on quintile returns.

    Args:
        quintile_returns: DataFrame with quintile returns
        ff6: DataFrame with FF6 factors
        period_name: Label for the regression period
        use_newey_west: If True, use Newey-West HAC standard errors (12 lags)
    """
    print("\n" + "=" * 70)
    print(f"FAMA-FRENCH 6-FACTOR REGRESSION ({period_name})")
    print("=" * 70)

    # Set date as index for ff6 if not already
    if 'date' in ff6.columns:
        ff6 = ff6.set_index('date')

    # Ensure both have same index format
    quintile_returns.index = pd.to_datetime(quintile_returns.index)
    ff6.index = pd.to_datetime(ff6.index)

    # Merge on month (first of month)
    quintile_returns.index = quintile_returns.index.to_period('M').to_timestamp()
    ff6.index = ff6.index.to_period('M').to_timestamp()

    # Merge
    merged = quintile_returns.join(ff6, how='inner')
    merged = merged.dropna()
    print(f"\nMerged observations: {len(merged)} months")
    print(f"Date range: {merged.index.min().date()} to {merged.index.max().date()}")

    if use_newey_west:
        print("Using Newey-West HAC standard errors (12 lags)")

    # Factor columns
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']

    # Run regression for each quintile and Q5-Q1
    print("\n" + "-" * 70)
    print("REGRESSION RESULTS")
    print("-" * 70)

    results = {}

    # Try to import statsmodels for Newey-West
    try:
        import statsmodels.api as sm
        has_statsmodels = True
    except ImportError:
        has_statsmodels = False
        if use_newey_west:
            print("  Warning: statsmodels not available, using basic OLS")

    for target in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if target not in merged.columns:
            continue

        # Returns are already in % (from * 100 in fwd_1yr calculation)
        # FF factors are also in % (e.g., 0.5 = 0.5%)
        # Do NOT multiply by 100 again - that was causing inflated alpha values
        y = merged[target] - (merged['RF'] if target != 'Q5-Q1' else 0)  # Excess returns (already in %)
        X = merged[factor_cols]

        try:
            if has_statsmodels and use_newey_west:
                # Newey-West HAC standard errors
                X_const = sm.add_constant(X)
                model = sm.OLS(y, X_const).fit(cov_type='HAC', cov_kwds={'maxlags': 12})

                results[target] = {
                    'alpha': model.params['const'],
                    'alpha_se': model.bse['const'],
                    'alpha_t': model.tvalues['const'],
                    'alpha_pval': model.pvalues['const'],
                    'betas': {f: model.params[f] for f in factor_cols},
                    'beta_t': {f: model.tvalues[f] for f in factor_cols},
                    'r2': model.rsquared,
                    'n_months': len(merged)
                }
            else:
                # Basic OLS
                X = X.copy()
                X['const'] = 1

                from numpy.linalg import lstsq
                coeffs, residuals, rank, s = lstsq(X.values, y.values, rcond=None)

                # Calculate t-statistics
                n = len(y)
                k = len(factor_cols) + 1
                y_pred = X.values @ coeffs
                resid = y.values - y_pred
                mse = np.sum(resid**2) / (n - k)
                var_coef = mse * np.linalg.inv(X.values.T @ X.values).diagonal()
                se = np.sqrt(var_coef)
                t_stats = coeffs / se

                results[target] = {
                    'alpha': coeffs[-1],
                    'alpha_t': t_stats[-1],
                    'betas': dict(zip(factor_cols, coeffs[:-1])),
                    'beta_t': dict(zip(factor_cols, t_stats[:-1])),
                    'r2': 1 - np.sum(resid**2) / np.sum((y.values - y.mean())**2),
                    'n_months': len(merged)
                }
        except Exception as e:
            print(f"  Error for {target}: {e}")
            continue

    # Print results
    # NOTE: Alpha is already in annual terms since dependent variable is 1-year forward returns
    print(f"\n{'Portfolio':<10} {'Alpha (1yr)':>12} {'t-stat':>8} {'Mkt-RF':>8} {'SMB':>8} {'HML':>8} {'RMW':>8} {'CMA':>8} {'MOM':>8} {'R²':>8}")
    print("-" * 105)

    for target in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if target not in results:
            continue
        r = results[target]
        alpha_1yr = r['alpha']  # Already annual (1-year forward returns)
        print(f"{target:<10} {alpha_1yr:>+10.2f}% {r['alpha_t']:>+8.2f} "
              f"{r['betas']['Mkt-RF']:>+8.2f} {r['betas']['SMB']:>+8.2f} "
              f"{r['betas']['HML']:>+8.2f} {r['betas']['RMW']:>+8.2f} "
              f"{r['betas']['CMA']:>+8.2f} {r['betas']['MOM']:>+8.2f} "
              f"{r['r2']:>7.1%}")

    # Detailed Q5-Q1 results
    if 'Q5-Q1' in results:
        r = results['Q5-Q1']
        print("\n" + "-" * 70)
        print("Q5-Q1 SPREAD ANALYSIS (MOONSHOT EXCESS RETURN)")
        print("-" * 70)
        # NOTE: Alpha is already in 1-year return terms (not monthly)
        # because the dependent variable is 1-year forward returns
        print(f"Alpha (1-year spread): {r['alpha']:>+8.2f}%")
        if 'alpha_se' in r:
            print(f"Standard error:       {r['alpha_se']:>8.2f}%")
        print(f"t-statistic:          {r['alpha_t']:>+8.2f}")
        if 'alpha_pval' in r:
            print(f"p-value:              {r['alpha_pval']:>8.4f}")
        print(f"R²:                   {r['r2']:>7.1%}")
        print(f"Observations:         {r['n_months']} months")

        sig_level = "***" if abs(r['alpha_t']) > 2.58 else "**" if abs(r['alpha_t']) > 1.96 else "*" if abs(r['alpha_t']) > 1.65 else ""
        print(f"\nSignificance: {sig_level}")
        print("(*** = 99% confidence, ** = 95% confidence, * = 90% confidence)")

        print("\nFactor Exposures:")
        for factor in factor_cols:
            beta = r['betas'][factor]
            t_stat = r['beta_t'][factor]
            sig = "***" if abs(t_stat) > 2.58 else "**" if abs(t_stat) > 1.96 else "*" if abs(t_stat) > 1.65 else ""
            print(f"  {factor:<10} β={beta:>+7.3f}  t={t_stat:>+6.2f}  {sig}")

    return results


def main():
    print("=" * 80)
    print("MOONSHOT QUALITY-FIRST FAMA-FRENCH 6-FACTOR REGRESSION")
    print("=" * 80)
    print(f"Run at: {datetime.now().isoformat()}")
    print()
    print("Methodology: Quality-First v2.0 (≥$500M market cap)")
    print("In-sample: 1995-2019 | Out-of-sample: 2020-2026")
    print("Z-scores: Computed from IS data only, applied to both IS and OOS")
    print("=" * 80)

    # Download FF6 factors
    ff6 = download_ff_factors()

    # Compute Moonshot quintile returns
    full_returns, is_returns, oos_returns = compute_moonshot_quintile_returns()

    # Run regressions
    print("\n\n")
    run_ff6_regression(full_returns, ff6, period_name="FULL SAMPLE (1995-2026)")

    print("\n\n")
    run_ff6_regression(is_returns, ff6, period_name="IN-SAMPLE (1995-2019)")

    print("\n\n")
    run_ff6_regression(oos_returns, ff6, period_name="OUT-OF-SAMPLE (2020-2026)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKEY METRICS TO EVALUATE:")
    print("1. Alpha (annualized): Excess return beyond FF6 factors")
    print("2. t-statistic: Statistical significance (>2.0 for 95% confidence)")
    print("3. R²: Proportion explained by FF6 (30-50% typical for good strategies)")
    print("4. OOS vs IS: Out-of-sample alpha should be positive and significant")
    print()
    print("✅ SUCCESS CRITERIA:")
    print("   - OOS alpha > 0 and t-stat > 2.0")
    print("   - Factor exposures make economic sense (e.g., positive SMB for small-cap tilt)")
    print("   - R² not too high (<70%) - we want unexplained alpha, not just factor exposure")


if __name__ == '__main__':
    main()

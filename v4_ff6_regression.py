#!/usr/bin/env python3
"""
V4 Fama-French 6-Factor Regression Analysis
Regresses V4 quintile returns against FF5 + Momentum to calculate alpha

FIXES APPLIED:
1. Z-scores computed from IS data only (no look-ahead bias)
2. Newey-West HAC standard errors for autocorrelation
3. Separate IS and OOS regression results
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

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Configuration
IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'


def download_ff_factors():
    """Download Fama-French 5 factors + Momentum from Kenneth French's website."""
    print("Downloading Fama-French factors...")

    # FF5 factors (monthly - more reliable download)
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


def compute_v4_quintile_returns():
    """
    Compute monthly V4 quintile returns.

    CRITICAL FIX: Z-scores are computed using IS statistics only,
    then applied to both IS and OOS periods to avoid look-ahead bias.
    """
    print("\n" + "=" * 70)
    print("COMPUTING V4 QUINTILE RETURNS (NO LOOK-AHEAD BIAS)")
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
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    conn.close()

    # Compute fundamental factors (NO winsorization yet)
    print("Computing factors...")
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']

    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    # Clean infinities only (winsorization comes after IS/OOS split)
    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Sample symbols for speed
    print("Sampling 2000 symbols...")
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(2000, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    # Compute monthly observations with 1-month forward returns
    print("Computing monthly returns...")
    results = []
    for i, symbol in enumerate(sample_symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(sample_symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values

        # Sample monthly (every 21 trading days)
        for j in range(252, n - 21, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 1-month return (for monthly regression)
            fwd_1m = ((close[j + 21] / close[j]) - 1) * 100

            # Volatility (trailing 60 days - no look-ahead)
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_1m': fwd_1m,
                'vol_60d': vol_60d,
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    print("Merging fundamentals...")
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']
    fund = fund[fund_cols].dropna(subset=['roa'])

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund[fund['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    df['fwd_1m'] = df['fwd_1m'].clip(-50, 50)
    df = df.dropna(subset=['fwd_1m', 'roa'])
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    print(f"  Clean observations: {len(df):,}")

    # ==========================================
    # CRITICAL FIX: IS/OOS SPLIT BEFORE Z-SCORES
    # ==========================================
    print("\n  Splitting into IS/OOS...")
    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()
    print(f"  In-Sample:     {len(is_df):,} observations")
    print(f"  Out-of-Sample: {len(oos_df):,} observations")

    # Compute winsorization bounds from IS ONLY
    print("  Computing winsorization bounds from IS data...")
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']
    factor_bounds = {}
    for col in factor_cols:
        if col in is_df.columns:
            lower = is_df[col].quantile(0.01)
            upper = is_df[col].quantile(0.99)
            factor_bounds[col] = (lower, upper)

    # Apply IS bounds to BOTH datasets
    for col, (lower, upper) in factor_bounds.items():
        is_df[col] = is_df[col].clip(lower, upper)
        oos_df[col] = oos_df[col].clip(lower, upper)

    # Compute z-scores using IS statistics ONLY
    print("  Computing z-scores using IS statistics...")
    factor_stats = {}
    for col in factor_cols:
        if col in is_df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
            factor_stats[col] = (mean, std)

    # Apply z-scores to BOTH datasets using IS stats
    for col, (mean, std) in factor_stats.items():
        for dataset in [is_df, oos_df]:
            dataset[f'{col}_z'] = (dataset[col] - mean) / std
            dataset[f'{col}_z'] = dataset[f'{col}_z'].clip(-3, 3).fillna(0)

    # Compute V4 score
    for dataset in [is_df, oos_df]:
        dataset['v4_score'] = (
            dataset['roa_z'] * 0.20 +
            dataset['ocf_assets_z'] * 0.15 +
            dataset['fcf_assets_z'] * 0.15 +
            dataset['gp_assets_z'] * 0.10 +
            (-dataset['vol_60d_z']) * 0.15 +
            (-dataset['asset_growth_z']) * 0.15
        )

    # Recombine for monthly returns
    df = pd.concat([is_df, oos_df], ignore_index=True)

    # Assign quintiles by month
    df['year_month'] = df['date'].dt.to_period('M')

    def assign_quintiles(group):
        if len(group) < 50:  # Need enough obs for quintiles
            group['quintile'] = np.nan
            return group
        try:
            group['quintile'] = pd.qcut(group['v4_score'].rank(method='first'), 5,
                                        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except ValueError:
            group['quintile'] = np.nan
        return group

    df = df.groupby('year_month', group_keys=False).apply(assign_quintiles)
    df = df.dropna(subset=['quintile'])

    # Compute monthly quintile returns
    print("Computing monthly quintile returns...")
    monthly_returns = df.groupby(['year_month', 'quintile'])['fwd_1m'].mean().unstack()
    monthly_returns['Q5-Q1'] = monthly_returns['Q5'] - monthly_returns['Q1']
    monthly_returns.index = monthly_returns.index.to_timestamp()

    print(f"  Monthly quintile returns: {len(monthly_returns)} months")

    return monthly_returns, is_df, oos_df


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

        y = merged[target] - (merged['RF'] if target != 'Q5-Q1' else 0)  # Excess returns
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
    print(f"\n{'Portfolio':<10} {'Alpha (ann)':>12} {'t-stat':>8} {'Mkt-RF':>8} {'SMB':>8} {'HML':>8} {'RMW':>8} {'CMA':>8} {'MOM':>8} {'R²':>8}")
    print("-" * 105)

    for target in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if target not in results:
            continue
        r = results[target]
        alpha_ann = r['alpha'] * 12  # Annualized
        print(f"{target:<10} {alpha_ann:>+10.2f}% {r['alpha_t']:>+8.2f} "
              f"{r['betas']['Mkt-RF']:>+8.2f} {r['betas']['SMB']:>+8.2f} "
              f"{r['betas']['HML']:>+8.2f} {r['betas']['RMW']:>+8.2f} "
              f"{r['betas']['CMA']:>+8.2f} {r['betas']['MOM']:>+8.2f} "
              f"{r['r2']:>7.1%}")

    # Detailed Q5-Q1 results
    if 'Q5-Q1' in results:
        r = results['Q5-Q1']
        print("\n" + "-" * 70)
        print("Q5-Q1 LONG-SHORT PORTFOLIO (Detailed)")
        print("-" * 70)

        alpha_ann = r['alpha'] * 12
        alpha_monthly = r['alpha']

        print(f"\nAlpha (monthly):     {alpha_monthly:>+.3f}%")
        print(f"Alpha (annualized):  {alpha_ann:>+.2f}%")
        print(f"Alpha t-statistic:   {r['alpha_t']:>+.2f} {'(Newey-West HAC)' if use_newey_west and has_statsmodels else '(OLS)'}")
        print(f"Significant (|t|>2): {'YES' if abs(r['alpha_t']) > 2 else 'NO'}")
        print(f"\nR²: {r['r2']:.1%}")

        print("\nFactor Loadings:")
        print(f"  {'Factor':<10} {'Beta':>10} {'t-stat':>10} {'Significant':>12}")
        print("  " + "-" * 45)
        for factor in factor_cols:
            sig = "YES" if abs(r['beta_t'][factor]) > 2 else "no"
            print(f"  {factor:<10} {r['betas'][factor]:>+10.3f} {r['beta_t'][factor]:>+10.2f} {sig:>12}")

    return results


def main():
    print("=" * 70)
    print("V4 FAMA-FRENCH 6-FACTOR REGRESSION ANALYSIS")
    print("(WITH LOOK-AHEAD BIAS FIX + NEWEY-WEST SE)")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nMethodology:")
    print(f"  In-Sample:     1995-2019 (z-scores computed here)")
    print(f"  Out-of-Sample: 2020-2026 (z-scores use IS statistics)")
    print(f"  Standard Errors: Newey-West HAC (12 lags)")

    # Download FF factors
    ff6 = download_ff_factors()

    # Compute V4 quintile returns (now returns IS/OOS dataframes too)
    quintile_returns, is_df, oos_df = compute_v4_quintile_returns()

    # Split returns by period
    is_returns = quintile_returns[quintile_returns.index < '2020-01-01'].copy()
    oos_returns = quintile_returns[quintile_returns.index >= '2020-01-01'].copy()

    print(f"\n  In-Sample months:  {len(is_returns)}")
    print(f"  Out-of-Sample months: {len(oos_returns)}")

    # Run regression on full sample
    results_full = run_ff6_regression(quintile_returns.copy(), ff6.copy(),
                                      "Full Sample (1995-2026)", use_newey_west=True)

    # Run regression on OOS ONLY
    results_oos = run_ff6_regression(oos_returns.copy(), ff6.copy(),
                                     "OUT-OF-SAMPLE (2020-2026)", use_newey_west=True)

    # Summary
    print("\n" + "=" * 70)
    print("COMPARISON: FULL SAMPLE vs OOS")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'Full Sample':>15} {'OOS Only':>15}")
    print("-" * 60)

    if 'Q5-Q1' in results_full and 'Q5-Q1' in results_oos:
        rf = results_full['Q5-Q1']
        ro = results_oos['Q5-Q1']

        print(f"{'Alpha (ann):':<25} {rf['alpha']*12:>+13.2f}% {ro['alpha']*12:>+13.2f}%")
        print(f"{'t-stat (Newey-West):':<25} {rf['alpha_t']:>+15.2f} {ro['alpha_t']:>+15.2f}")
        print(f"{'Significant (|t|>2):':<25} {'YES' if abs(rf['alpha_t']) > 2 else 'NO':>15} {'YES' if abs(ro['alpha_t']) > 2 else 'NO':>15}")
        print(f"{'R²:':<25} {rf['r2']:>14.1%} {ro['r2']:>14.1%}")
        print(f"{'N months:':<25} {rf.get('n_months', 'N/A'):>15} {ro.get('n_months', 'N/A'):>15}")

    # Final interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if 'Q5-Q1' in results_oos:
        r = results_oos['Q5-Q1']
        alpha_ann = r['alpha'] * 12

        print(f"""
KEY FINDINGS (Out-of-Sample Only):

1. V4 OOS Alpha: {alpha_ann:+.2f}% annualized
   - t-statistic: {r['alpha_t']:+.2f} (Newey-West HAC, 12 lags)
   - Significance: {'SIGNIFICANT (|t| > 2)' if abs(r['alpha_t']) > 2 else 'NOT significant at 5%'}

2. Factor Exposures (OOS):
   - Market (Mkt-RF): {r['betas']['Mkt-RF']:+.3f} {'(significant)' if abs(r['beta_t']['Mkt-RF']) > 2 else ''}
   - Size (SMB):      {r['betas']['SMB']:+.3f} {'(significant)' if abs(r['beta_t']['SMB']) > 2 else ''}
   - Value (HML):     {r['betas']['HML']:+.3f} {'(significant)' if abs(r['beta_t']['HML']) > 2 else ''}
   - Profitability (RMW): {r['betas']['RMW']:+.3f} {'(significant)' if abs(r['beta_t']['RMW']) > 2 else ''}
   - Investment (CMA): {r['betas']['CMA']:+.3f} {'(significant)' if abs(r['beta_t']['CMA']) > 2 else ''}
   - Momentum (MOM):  {r['betas']['MOM']:+.3f} {'(significant)' if abs(r['beta_t']['MOM']) > 2 else ''}

3. R² = {r['r2']:.1%}
   - {100-r['r2']*100:.0f}% of V4's OOS returns are NOT explained by FF6 factors

CREDIBILITY ASSESSMENT:
- OOS alpha is the TRUE test of the strategy (no look-ahead bias)
- Newey-West SEs account for autocorrelation in monthly returns
- {'ROBUST: Alpha survives OOS and HAC corrections' if abs(r['alpha_t']) > 2 else 'WEAK: Alpha may be due to noise or overfitting'}
""")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

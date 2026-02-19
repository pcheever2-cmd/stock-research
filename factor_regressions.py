#!/usr/bin/env python3
"""
Fama-French Factor Regressions
==============================
Test whether V3 scoring generates true alpha after controlling for known factors:
- Fama-French 5 factors: MKT-RF, SMB, HML, RMW, CMA
- Momentum factor (MOM)

Downloads factor returns from Kenneth French's website and regresses
V3 quintile portfolio returns against them.

Usage:
    python factor_regressions.py              # Run full analysis
    python factor_regressions.py --download   # Just download/update factor data
"""

import sqlite3
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import urllib.request
import zipfile
import io

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME

# Kenneth French data library URLs
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip"

CACHE_DIR = Path(__file__).parent / "cache"
FF5_CACHE = CACHE_DIR / "ff5_daily.csv"
MOM_CACHE = CACHE_DIR / "mom_daily.csv"


def download_french_factors(force: bool = False) -> tuple:
    """
    Download Fama-French 5 factors + Momentum from Ken French's website.
    Returns (ff5_df, mom_df) with daily factor returns.
    """
    CACHE_DIR.mkdir(exist_ok=True)

    # Check cache freshness (refresh if older than 30 days)
    cache_fresh = (
        FF5_CACHE.exists() and
        MOM_CACHE.exists() and
        (datetime.now().timestamp() - FF5_CACHE.stat().st_mtime) < 30 * 24 * 3600
    )

    if cache_fresh and not force:
        print("Loading cached factor data...")
        ff5 = pd.read_csv(FF5_CACHE, parse_dates=['date'])
        mom = pd.read_csv(MOM_CACHE, parse_dates=['date'])
        return ff5, mom

    print("Downloading Fama-French 5 factors from Ken French's website...")

    # Download FF5
    try:
        with urllib.request.urlopen(FF5_URL, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                # Find the CSV file inside
                csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    # Skip header rows until we find the data
                    lines = f.read().decode('utf-8').split('\n')

                    # Find where daily data starts
                    data_start = None
                    for i, line in enumerate(lines):
                        if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')[0]) == 8:
                            data_start = i
                            break

                    if data_start is None:
                        raise ValueError("Could not find data start in FF5 file")

                    # Parse the data
                    data_lines = []
                    for line in lines[data_start:]:
                        if not line.strip() or not line.strip()[0].isdigit():
                            break
                        data_lines.append(line)

                    ff5_data = pd.read_csv(
                        io.StringIO('\n'.join(data_lines)),
                        names=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
                        header=None
                    )
                    ff5_data['date'] = pd.to_datetime(ff5_data['date'], format='%Y%m%d')

                    # Convert from percent to decimal
                    for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
                        ff5_data[col] = ff5_data[col].astype(float) / 100

                    ff5_data.to_csv(FF5_CACHE, index=False)
                    print(f"  FF5: {len(ff5_data)} daily observations")

    except Exception as e:
        print(f"  Error downloading FF5: {e}")
        if FF5_CACHE.exists():
            print("  Using cached FF5 data")
            ff5_data = pd.read_csv(FF5_CACHE, parse_dates=['date'])
        else:
            raise

    # Download Momentum
    print("Downloading Momentum factor...")
    try:
        with urllib.request.urlopen(MOM_URL, timeout=30) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                csv_name = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')][0]
                with z.open(csv_name) as f:
                    lines = f.read().decode('utf-8').split('\n')

                    data_start = None
                    for i, line in enumerate(lines):
                        if line.strip() and line.strip()[0].isdigit() and len(line.strip().split(',')[0]) == 8:
                            data_start = i
                            break

                    if data_start is None:
                        raise ValueError("Could not find data start in MOM file")

                    data_lines = []
                    for line in lines[data_start:]:
                        if not line.strip() or not line.strip()[0].isdigit():
                            break
                        data_lines.append(line)

                    mom_data = pd.read_csv(
                        io.StringIO('\n'.join(data_lines)),
                        names=['date', 'Mom'],
                        header=None
                    )
                    mom_data['date'] = pd.to_datetime(mom_data['date'], format='%Y%m%d')
                    mom_data['Mom'] = mom_data['Mom'].astype(float) / 100

                    mom_data.to_csv(MOM_CACHE, index=False)
                    print(f"  MOM: {len(mom_data)} daily observations")

    except Exception as e:
        print(f"  Error downloading MOM: {e}")
        if MOM_CACHE.exists():
            print("  Using cached MOM data")
            mom_data = pd.read_csv(MOM_CACHE, parse_dates=['date'])
        else:
            raise

    return ff5_data, mom_data


def load_quintile_returns(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Load V3 quintile portfolio returns from backtest data.
    Returns monthly portfolio returns for Q1-Q5.

    Calculates V3 scores on-the-fly using:
    - Momentum 12-1 (from price history)
    - 52-week high proximity
    - Trend, Fundamentals, Valuation (from backtest_daily_scores)
    """
    print("\nLoading and scoring data for quintile returns...")

    # Load daily scores with prices
    df = pd.read_sql_query("""
        SELECT d.symbol, d.date, d.close, d.lt_score, d.value_score_v2,
               d.trend_score, d.fundamentals_score, d.valuation_score
        FROM backtest_daily_scores d
        WHERE d.date >= '2020-01-01'
        ORDER BY d.symbol, d.date
    """, conn)

    print(f"  Loaded {len(df):,} daily records")

    if df.empty:
        return pd.DataFrame()

    # Calculate momentum and 52w high for each stock-date
    print("  Calculating momentum factors...")
    df['date'] = pd.to_datetime(df['date'])

    # Group by symbol and calculate momentum/52w-high
    def calc_momentum_factors(group):
        group = group.sort_values('date').reset_index(drop=True)
        closes = group['close'].values
        n = len(closes)

        mom_12_1 = np.full(n, np.nan)
        high52w_pct = np.full(n, np.nan)

        for i in range(252, n):
            # 12-1 momentum
            if closes[i-252] > 0 and closes[i-21] > 0:
                mom_12_1[i] = ((closes[i-21] / closes[i-252]) - 1) * 100

            # 52w high
            high_52w = np.max(closes[max(0, i-252):i])
            if high_52w > 0:
                high52w_pct[i] = ((closes[i] / high_52w) - 1) * 100

        group['mom_12_1'] = mom_12_1
        group['high52w_pct'] = high52w_pct
        return group

    # Sample for speed (500 symbols)
    symbols = df['symbol'].unique()
    np.random.seed(42)
    if len(symbols) > 500:
        sample_symbols = np.random.choice(symbols, size=500, replace=False)
        df = df[df['symbol'].isin(sample_symbols)]
        print(f"  Sampled {len(sample_symbols)} symbols for analysis")

    df = df.groupby('symbol', group_keys=False).apply(calc_momentum_factors)

    # Calculate V3-like score
    # Momentum score (0-25)
    def mom_score(m):
        if pd.isna(m): return 12
        if m > 40: return 25
        if m > 10: return 18
        if m > -10: return 12
        if m > -30: return 6
        return 0

    # 52w high score (0-10)
    def high52w_score(h):
        if pd.isna(h): return 5
        if h > -5: return 10
        if h > -15: return 7
        if h > -30: return 4
        return 0

    df['mom_score'] = df['mom_12_1'].apply(mom_score)
    df['high52w_score'] = df['high52w_pct'].apply(high52w_score)

    # V3 = momentum(25) + high52w(10) + trend(15) + fund(15) + val(5) + ...
    # Simplified V3 using available components
    df['value_score_v3'] = (
        df['mom_score'] +                    # 0-25
        df['high52w_score'] +                # 0-10
        (df['trend_score'] * 15 / 25).fillna(7) +        # 0-15 (scaled)
        (df['fundamentals_score'] * 15 / 25).fillna(7) + # 0-15 (scaled)
        (df['valuation_score'] * 5 / 15).fillna(2)       # 0-5 (scaled)
    ).clip(0, 100)

    # Drop rows without V3 score
    df = df.dropna(subset=['value_score_v3'])

    # Convert to monthly
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')

    # Get month-end scores and calculate returns properly
    print("  Building monthly quintile portfolios...")
    monthly_data = []

    months = sorted(df['month'].unique())

    for i, month in enumerate(months[:-1]):  # Skip last month (no forward return)
        next_month = months[i + 1]

        month_df = df[df['month'] == month]
        next_month_df = df[df['month'] == next_month]

        # Get end-of-month scores
        eom = month_df.groupby('symbol').agg({
            'value_score_v3': 'last',
            'close': 'last',
            'date': 'last'
        }).reset_index()

        # Get beginning of next month prices
        bom = next_month_df.groupby('symbol').agg({
            'close': ['first', 'last']
        }).reset_index()
        bom.columns = ['symbol', 'bom_price', 'eom_price']

        # Merge to get monthly returns
        merged = eom.merge(bom, on='symbol', how='inner')
        merged['monthly_return'] = (merged['eom_price'] / merged['bom_price']) - 1

        # Filter out extreme returns (data errors)
        merged = merged[(merged['monthly_return'] > -0.9) & (merged['monthly_return'] < 5.0)]

        if len(merged) < 50:
            continue

        # Assign quintiles based on V3 score at end of previous month
        try:
            merged['quintile'] = pd.qcut(merged['value_score_v3'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except ValueError:
            continue

        # Calculate equal-weighted return for each quintile
        for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            q_data = merged[merged['quintile'] == quintile]

            if len(q_data) < 10:
                continue

            monthly_data.append({
                'month': next_month.to_timestamp(),
                'quintile': quintile,
                'return': q_data['monthly_return'].mean(),
                'n_stocks': len(q_data),
            })

    result = pd.DataFrame(monthly_data)
    print(f"  Calculated {len(result)} monthly quintile returns")
    print(f"  Date range: {result['month'].min()} to {result['month'].max()}")

    return result


def run_factor_regression(portfolio_returns: pd.Series, factors: pd.DataFrame) -> dict:
    """
    Run OLS regression: R_p - R_f = α + β₁(MKT-RF) + β₂(SMB) + β₃(HML) + β₄(RMW) + β₅(CMA) + β₆(MOM) + ε

    Returns regression results including alpha, betas, t-stats, R².
    """
    # Convert series to dataframe with proper column names
    port_df = portfolio_returns.reset_index()
    port_df.columns = ['date', 'return']
    port_df['date'] = pd.to_datetime(port_df['date'])

    # Align dates
    merged = pd.merge(
        port_df,
        factors,
        on='date',
        how='inner'
    )

    if len(merged) < 12:
        return None

    # Dependent variable: portfolio excess return
    y = merged['return'] - merged['RF']

    # Independent variables
    X = merged[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Mom']].values
    X = np.column_stack([np.ones(len(X)), X])  # Add intercept

    # OLS regression
    try:
        # β = (X'X)^(-1) X'y
        XtX_inv = np.linalg.inv(X.T @ X)
        betas = XtX_inv @ X.T @ y.values

        # Residuals and R²
        y_hat = X @ betas
        residuals = y.values - y_hat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y.values - y.values.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot

        # Standard errors and t-stats
        n, k = X.shape
        mse = ss_res / (n - k)
        se = np.sqrt(np.diag(mse * XtX_inv))
        t_stats = betas / se

        # Annualized alpha (monthly to annual)
        alpha_monthly = betas[0]
        alpha_annual = (1 + alpha_monthly) ** 12 - 1

        return {
            'alpha_monthly': alpha_monthly,
            'alpha_annual': alpha_annual,
            'alpha_t_stat': t_stats[0],
            'beta_mkt': betas[1],
            'beta_smb': betas[2],
            'beta_hml': betas[3],
            'beta_rmw': betas[4],
            'beta_cma': betas[5],
            'beta_mom': betas[6],
            't_mkt': t_stats[1],
            't_smb': t_stats[2],
            't_hml': t_stats[3],
            't_rmw': t_stats[4],
            't_cma': t_stats[5],
            't_mom': t_stats[6],
            'r_squared': r_squared,
            'n_obs': n,
        }
    except np.linalg.LinAlgError:
        return None


def run_analysis():
    """Run the full factor regression analysis."""
    print("=" * 70)
    print("FAMA-FRENCH FACTOR REGRESSION ANALYSIS")
    print("Testing V3 for true alpha vs known factor exposures")
    print("=" * 70)

    # Download/load factor data
    ff5, mom = download_french_factors()

    # Merge factors
    factors = pd.merge(ff5, mom, on='date', how='inner')
    print(f"\nFactor data: {factors['date'].min()} to {factors['date'].max()}")

    # Convert to monthly
    factors['month'] = factors['date'].dt.to_period('M')
    factors_monthly = factors.groupby('month').agg({
        'Mkt-RF': lambda x: (1 + x).prod() - 1,
        'SMB': lambda x: (1 + x).prod() - 1,
        'HML': lambda x: (1 + x).prod() - 1,
        'RMW': lambda x: (1 + x).prod() - 1,
        'CMA': lambda x: (1 + x).prod() - 1,
        'Mom': lambda x: (1 + x).prod() - 1,
        'RF': lambda x: (1 + x).prod() - 1,
    }).reset_index()
    factors_monthly['date'] = factors_monthly['month'].dt.to_timestamp()

    # Load quintile returns
    conn = sqlite3.connect(BACKTEST_DB)
    quintile_returns = load_quintile_returns(conn)
    conn.close()

    if quintile_returns.empty:
        print("\nNo quintile return data available. Run backtest first.")
        return

    # Pivot to wide format
    quintile_returns['month'] = pd.to_datetime(quintile_returns['month'])
    quintile_wide = quintile_returns.pivot(index='month', columns='quintile', values='return')

    # Q5-Q1 spread (long-short)
    if 'Q5' in quintile_wide.columns and 'Q1' in quintile_wide.columns:
        quintile_wide['Q5-Q1'] = quintile_wide['Q5'] - quintile_wide['Q1']

    print("\n" + "=" * 70)
    print("REGRESSION RESULTS")
    print("=" * 70)

    print("\nModel: R_p - R_f = α + β_mkt(MKT-RF) + β_smb(SMB) + β_hml(HML)")
    print("                   + β_rmw(RMW) + β_cma(CMA) + β_mom(MOM) + ε")

    results = {}

    for portfolio in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if portfolio not in quintile_wide.columns:
            continue

        port_returns = quintile_wide[portfolio].dropna()
        port_returns.index = pd.to_datetime(port_returns.index)

        reg_result = run_factor_regression(port_returns, factors_monthly)

        if reg_result:
            results[portfolio] = reg_result

    # Display results table
    print(f"\n{'Portfolio':<10} {'Alpha(Ann)':>12} {'t(α)':>8} {'β_mkt':>8} {'β_smb':>8} {'β_hml':>8} {'β_mom':>8} {'R²':>8}")
    print("-" * 85)

    for portfolio in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q5-Q1']:
        if portfolio not in results:
            continue

        r = results[portfolio]
        sig = "**" if abs(r['alpha_t_stat']) > 2 else "*" if abs(r['alpha_t_stat']) > 1.65 else ""

        print(f"{portfolio:<10} {r['alpha_annual']:>+11.2%}{sig} {r['alpha_t_stat']:>+7.2f} "
              f"{r['beta_mkt']:>+7.2f} {r['beta_smb']:>+7.2f} {r['beta_hml']:>+7.2f} "
              f"{r['beta_mom']:>+7.2f} {r['r_squared']:>7.1%}")

    print("\n* p<0.10, ** p<0.05")

    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if 'Q5-Q1' in results:
        r = results['Q5-Q1']
        alpha = r['alpha_annual']
        t_stat = r['alpha_t_stat']

        print(f"\nQ5-Q1 Long-Short Strategy:")
        print(f"  Annualized Alpha: {alpha:+.2%}")
        print(f"  T-statistic: {t_stat:+.2f}")

        if abs(t_stat) > 2:
            print("\n  ✅ SIGNIFICANT ALPHA: V3 generates returns not explained by FF5+Mom")
            print("     The scoring system provides genuine stock-picking value.")
        elif abs(t_stat) > 1.65:
            print("\n  ⚠️  MARGINAL ALPHA: V3 shows some unexplained returns (p<0.10)")
            print("     Results are suggestive but not statistically conclusive.")
        else:
            print("\n  ❌ NO SIGNIFICANT ALPHA: V3 returns explained by known factors")
            print("     The scoring system may be repackaging existing factor exposures.")

        print(f"\nFactor Exposures of Q5-Q1:")
        print(f"  Market Beta: {r['beta_mkt']:+.2f} (t={r['t_mkt']:+.2f})")
        print(f"  Size (SMB):  {r['beta_smb']:+.2f} (t={r['t_smb']:+.2f})")
        print(f"  Value (HML): {r['beta_hml']:+.2f} (t={r['t_hml']:+.2f})")
        print(f"  Prof (RMW):  {r['beta_rmw']:+.2f} (t={r['t_rmw']:+.2f})")
        print(f"  Inv (CMA):   {r['beta_cma']:+.2f} (t={r['t_cma']:+.2f})")
        print(f"  Mom (MOM):   {r['beta_mom']:+.2f} (t={r['t_mom']:+.2f})")

        # Identify significant exposures
        sig_exposures = []
        if abs(r['t_mkt']) > 2: sig_exposures.append('Market')
        if abs(r['t_smb']) > 2: sig_exposures.append('Size')
        if abs(r['t_hml']) > 2: sig_exposures.append('Value')
        if abs(r['t_rmw']) > 2: sig_exposures.append('Profitability')
        if abs(r['t_cma']) > 2: sig_exposures.append('Investment')
        if abs(r['t_mom']) > 2: sig_exposures.append('Momentum')

        if sig_exposures:
            print(f"\n  Significant factor exposures: {', '.join(sig_exposures)}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fama-French factor regression analysis')
    parser.add_argument('--download', action='store_true', help='Just download/update factor data')
    args = parser.parse_args()

    if args.download:
        download_french_factors(force=True)
    else:
        run_analysis()

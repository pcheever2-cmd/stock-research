#!/usr/bin/env python3
"""
Comprehensive Robustness Tests for Compass Score
=================================================
Addresses critical review concerns:

1. HAC/Newey-West Standard Errors - Re-run FF6 regression with proper HAC
2. Non-Overlapping Returns - Test with quarterly samples to eliminate autocorrelation
3. Capacity/Liquidity Modeling - Estimate realistic position sizes given volume constraints

Output: Results for inclusion in RESEARCH_PAPER.md
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

    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

    try:
        # Download FF5
        r = requests.get(ff5_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_names = [n for n in z.namelist() if '.csv' in n.lower()]
        csv_name = csv_names[0]

        ff5_raw = pd.read_csv(z.open(csv_name), skiprows=3, header=None)
        annual_start = None
        for idx, row in ff5_raw.iterrows():
            val = str(row.iloc[0]).strip()
            if len(val) == 4 and val.isdigit():
                annual_start = idx
                break
        if annual_start:
            ff5_raw = ff5_raw.iloc[:annual_start]

        ff5_raw.columns = ['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        ff5 = ff5_raw[ff5_raw['date'].astype(str).str.match(r'^\s*\d{6}\s*$')].copy()
        ff5['date'] = pd.to_datetime(ff5['date'].astype(str).str.strip(), format='%Y%m')
        for col in ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']:
            ff5[col] = pd.to_numeric(ff5[col], errors='coerce')

        # Download Momentum
        r = requests.get(mom_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_names = [n for n in z.namelist() if '.csv' in n.lower()]
        csv_name = csv_names[0]

        mom_raw = pd.read_csv(z.open(csv_name), skiprows=13, header=None)
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

        ff6 = ff5.merge(mom, on='date', how='inner').dropna()
        print(f"  FF6 factors: {len(ff6):,} months")
        return ff6

    except Exception as e:
        print(f"  Error: {e}")
        return None


def load_and_prepare_data(sample_size=2000):
    """Load data and prepare for analysis."""
    print("\nLoading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Load fundamentals with market cap
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.gross_profit, i.net_income,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow,
               m.market_cap
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")
    conn.close()

    # Compute factors
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund = fund.sort_values(['symbol', 'date'])
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

    for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Sample symbols
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(sample_size, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]
    print(f"  Sampled {len(sample_symbols)} symbols")

    return prices, fund, sample_symbols


def compute_observations(prices, fund, sample_symbols, return_horizon=63, sample_freq=21):
    """
    Compute observations with forward returns.

    Args:
        return_horizon: Forward return period in trading days (63 = 3 months)
        sample_freq: Sampling frequency in trading days (21 = monthly, 63 = quarterly)
    """
    print(f"\nComputing observations (horizon={return_horizon}, freq={sample_freq} days)...")

    results = []
    for i, symbol in enumerate(sample_symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(sample_symbols)}...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values

        for j in range(252, n - return_horizon, sample_freq):
            date = sym_prices['date'].iloc[j]

            # Forward return
            fwd_ret = ((close[j + return_horizon] / close[j]) - 1) * 100

            # 1-month return for monthly FF6 regression
            if j + 21 < n:
                fwd_1m = ((close[j + 21] / close[j]) - 1) * 100
            else:
                fwd_1m = np.nan

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Average daily dollar volume (trailing 63 days)
            avg_dollar_vol = np.mean(close[j-63:j] * volume[j-63:j]) if j >= 63 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_ret': fwd_ret,
                'fwd_1m': fwd_1m,
                'vol_60d': vol_60d,
                'avg_dollar_vol': avg_dollar_vol,
            })

    df = pd.DataFrame(results)
    print(f"  Generated {len(df):,} observations")

    # Merge fundamentals
    fund_cols = ['symbol', 'date', 'roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'market_cap']
    fund_clean = fund[fund_cols].dropna(subset=['roa'])

    all_merged = []
    for symbol in df['symbol'].unique():
        pf = df[df['symbol'] == symbol].sort_values('date')
        f = fund_clean[fund_clean['symbol'] == symbol].sort_values('date')

        if len(f) == 0:
            continue

        merged = pd.merge_asof(pf, f.drop(columns=['symbol']),
                               on='date', direction='backward',
                               tolerance=pd.Timedelta('365 days'))
        all_merged.append(merged)

    df = pd.concat(all_merged, ignore_index=True)
    df['fwd_ret'] = df['fwd_ret'].clip(-100, 200)
    df['fwd_1m'] = df['fwd_1m'].clip(-50, 50)
    df = df.dropna(subset=['fwd_ret', 'roa'])
    print(f"  Clean observations: {len(df):,}")

    return df


def apply_is_oos_split_and_scores(df):
    """Apply IS/OOS split and compute z-scores using IS stats only."""
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()

    print(f"\n  IS: {len(is_df):,}  |  OOS: {len(oos_df):,}")

    # Winsorize using IS bounds
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']
    factor_bounds = {}
    for col in factor_cols:
        if col in is_df.columns:
            lower = is_df[col].quantile(0.01)
            upper = is_df[col].quantile(0.99)
            factor_bounds[col] = (lower, upper)

    for col, (lower, upper) in factor_bounds.items():
        is_df[col] = is_df[col].clip(lower, upper)
        oos_df[col] = oos_df[col].clip(lower, upper)

    # Z-scores using IS stats
    factor_stats = {}
    for col in factor_cols:
        if col in is_df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
            factor_stats[col] = (mean, std)

    for col, (mean, std) in factor_stats.items():
        for dataset in [is_df, oos_df]:
            dataset[f'{col}_z'] = (dataset[col] - mean) / std
            dataset[f'{col}_z'] = dataset[f'{col}_z'].clip(-3, 3).fillna(0)

    # Compute Compass Score
    for dataset in [is_df, oos_df]:
        dataset['compass_score'] = (
            dataset['roa_z'] * 0.20 +
            dataset['ocf_assets_z'] * 0.15 +
            dataset['fcf_assets_z'] * 0.15 +
            dataset['gp_assets_z'] * 0.20 +
            (-dataset['vol_60d_z']) * 0.15 +
            (-dataset['asset_growth_z']) * 0.15
        )

    return is_df, oos_df


# ============================================================================
# TEST 1: HAC/Newey-West Standard Errors
# ============================================================================

def test_hac_regression(oos_df, ff6):
    """Run FF6 regression with Newey-West HAC standard errors."""
    print("\n" + "=" * 70)
    print("TEST 1: FF6 REGRESSION WITH NEWEY-WEST HAC STANDARD ERRORS")
    print("=" * 70)

    try:
        import statsmodels.api as sm
    except ImportError:
        print("ERROR: statsmodels not installed")
        return None

    # Compute monthly quintile returns
    oos_df['year_month'] = oos_df['date'].dt.to_period('M')

    def assign_quintiles(group):
        if len(group) < 50:
            group['quintile'] = np.nan
            return group
        try:
            group['quintile'] = pd.qcut(group['compass_score'].rank(method='first'), 5,
                                        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except ValueError:
            group['quintile'] = np.nan
        return group

    oos_df = oos_df.groupby('year_month', group_keys=False).apply(assign_quintiles)
    oos_df = oos_df.dropna(subset=['quintile', 'fwd_1m'])

    # Monthly quintile returns
    monthly_returns = oos_df.groupby(['year_month', 'quintile'])['fwd_1m'].mean().unstack()
    monthly_returns['Q5-Q1'] = monthly_returns['Q5'] - monthly_returns['Q1']
    monthly_returns.index = monthly_returns.index.to_timestamp()

    print(f"\nOOS months: {len(monthly_returns)}")

    # Merge with FF6
    ff6_indexed = ff6.set_index('date').copy()
    ff6_indexed.index = ff6_indexed.index.to_period('M').to_timestamp()
    monthly_returns.index = monthly_returns.index.to_period('M').to_timestamp()

    merged = monthly_returns.join(ff6_indexed, how='inner').dropna()
    print(f"Merged months: {len(merged)}")

    # Run regressions with different lag specifications
    factor_cols = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    y = merged['Q5-Q1']
    X = sm.add_constant(merged[factor_cols])

    results = {}

    # OLS (no HAC)
    model_ols = sm.OLS(y, X).fit()
    results['OLS'] = {
        'alpha': model_ols.params['const'] * 12,
        't_stat': model_ols.tvalues['const'],
        'r2': model_ols.rsquared
    }

    # HAC with different lag lengths
    for lags in [6, 12, 18]:
        model_hac = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        results[f'HAC-{lags}'] = {
            'alpha': model_hac.params['const'] * 12,
            't_stat': model_hac.tvalues['const'],
            'r2': model_hac.rsquared
        }

    # Print results
    print("\n" + "-" * 60)
    print(f"{'Method':<15} {'Alpha (ann)':>15} {'t-statistic':>15} {'% Reduction':>15}")
    print("-" * 60)

    ols_t = results['OLS']['t_stat']
    for method, r in results.items():
        reduction = ((ols_t - r['t_stat']) / ols_t * 100) if method != 'OLS' else 0
        print(f"{method:<15} {r['alpha']:>+14.2f}% {r['t_stat']:>+15.2f} {reduction:>14.1f}%")

    # Return detailed results for the standard HAC-12
    r = results['HAC-12']
    print(f"\n  HAC-12 alpha:    {r['alpha']:+.2f}% annualized")
    print(f"  HAC-12 t-stat:   {r['t_stat']:+.2f}")
    print(f"  Significant?:    {'YES' if abs(r['t_stat']) > 2 else 'NO'}")
    print(f"  t-stat reduction: {((ols_t - r['t_stat']) / ols_t * 100):.1f}% vs OLS")

    return results


# ============================================================================
# TEST 2: Non-Overlapping Quarterly Returns
# ============================================================================

def test_non_overlapping_returns(prices, fund, sample_symbols):
    """Test with non-overlapping quarterly observations."""
    print("\n" + "=" * 70)
    print("TEST 2: NON-OVERLAPPING QUARTERLY RETURNS")
    print("=" * 70)
    print("Using 63-day forward returns sampled every 63 days (no overlap)")

    # Compute quarterly observations (sample every 63 days = 3 months)
    df = compute_observations(prices, fund, sample_symbols,
                              return_horizon=63, sample_freq=63)

    is_df, oos_df = apply_is_oos_split_and_scores(df)

    # Assign quintiles
    for dataset, name in [(is_df, 'IS'), (oos_df, 'OOS')]:
        def assign_quintiles(group):
            if len(group) < 30:
                group['quintile'] = np.nan
                return group
            try:
                group['quintile'] = pd.qcut(group['compass_score'].rank(method='first'), 5,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            except ValueError:
                group['quintile'] = np.nan
            return group

        dataset['quarter'] = dataset['date'].dt.to_period('Q')
        dataset = dataset.groupby('quarter', group_keys=False).apply(assign_quintiles)

    # Re-apply quintiles (the above doesn't modify in place properly)
    is_df['quarter'] = is_df['date'].dt.to_period('Q')
    oos_df['quarter'] = oos_df['date'].dt.to_period('Q')

    def assign_q(g):
        if len(g) < 30:
            g['quintile'] = np.nan
            return g
        try:
            g['quintile'] = pd.qcut(g['compass_score'].rank(method='first'), 5,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except:
            g['quintile'] = np.nan
        return g

    is_df = is_df.groupby('quarter', group_keys=False).apply(assign_q)
    oos_df = oos_df.groupby('quarter', group_keys=False).apply(assign_q)

    # Compute quintile returns
    print("\nQuintile Performance (Non-Overlapping):")
    print("-" * 60)
    print(f"{'Period':<10} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Q5':>10} {'Q5-Q1':>10}")
    print("-" * 60)

    results = {}
    for dataset, name in [(is_df, 'IS'), (oos_df, 'OOS')]:
        clean = dataset.dropna(subset=['quintile', 'fwd_ret'])

        quintile_returns = {}
        for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
            ret = clean[clean['quintile'] == q]['fwd_ret'].mean()
            quintile_returns[q] = ret

        spread = quintile_returns['Q5'] - quintile_returns['Q1']

        print(f"{name:<10} {quintile_returns['Q1']:>+9.2f}% {quintile_returns['Q2']:>+9.2f}% "
              f"{quintile_returns['Q3']:>+9.2f}% {quintile_returns['Q4']:>+9.2f}% "
              f"{quintile_returns['Q5']:>+9.2f}% {spread:>+9.2f}%")

        results[name] = {
            'quintile_returns': quintile_returns,
            'spread': spread,
            'n_obs': len(clean),
            'n_quarters': clean['quarter'].nunique()
        }

    print(f"\nOOS observations: {results['OOS']['n_obs']:,}")
    print(f"OOS quarters:     {results['OOS']['n_quarters']}")
    print(f"\nOOS Q5-Q1 Spread: {results['OOS']['spread']:+.2f}%")

    # Check monotonicity
    oos_rets = results['OOS']['quintile_returns']
    is_monotonic = all(oos_rets[f'Q{i}'] < oos_rets[f'Q{i+1}'] for i in range(1, 5))
    print(f"Strictly Monotonic: {'YES' if is_monotonic else 'NO'}")

    return results


# ============================================================================
# TEST 3: Capacity/Liquidity Modeling
# ============================================================================

def test_capacity_modeling(oos_df):
    """Estimate realistic capacity given liquidity constraints."""
    print("\n" + "=" * 70)
    print("TEST 3: CAPACITY/LIQUIDITY MODELING")
    print("=" * 70)

    # Assign quintiles
    oos_df['quarter'] = oos_df['date'].dt.to_period('Q')

    def assign_q(g):
        if len(g) < 50:
            g['quintile'] = np.nan
            return g
        try:
            g['quintile'] = pd.qcut(g['compass_score'].rank(method='first'), 5,
                                    labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
        except:
            g['quintile'] = np.nan
        return g

    oos_df = oos_df.groupby('quarter', group_keys=False).apply(assign_q)

    # Filter to Q5 (stocks we'd want to buy)
    q5_df = oos_df[oos_df['quintile'] == 'Q5'].dropna(subset=['avg_dollar_vol', 'market_cap'])

    print(f"\nQ5 stocks in OOS: {len(q5_df):,} observations")

    # Capacity analysis
    print("\nCapacity Analysis by Market Cap:")
    print("-" * 70)

    # Define market cap buckets
    cap_bins = [0, 300e6, 2e9, 10e9, np.inf]
    cap_labels = ['Micro (<$300M)', 'Small ($300M-2B)', 'Mid ($2B-10B)', 'Large (>$10B)']

    q5_df['cap_bucket'] = pd.cut(q5_df['market_cap'], bins=cap_bins, labels=cap_labels)

    # Assume we can trade 5% of daily volume without impact
    participation_rate = 0.05

    print(f"\n{'Cap Segment':<20} {'Avg Daily Vol':>15} {'Position Size':>15} {'5% ADV':>15} {'Stocks':>10}")
    print("-" * 70)

    capacity_results = {}
    total_capacity = 0

    for bucket in cap_labels:
        bucket_df = q5_df[q5_df['cap_bucket'] == bucket]

        if len(bucket_df) == 0:
            continue

        avg_adv = bucket_df['avg_dollar_vol'].median()
        max_position = avg_adv * participation_rate * 21  # 21 days to build position
        n_stocks = bucket_df['symbol'].nunique()

        # Total capacity for this bucket (assume hold all Q5 stocks equally)
        bucket_capacity = max_position * n_stocks
        total_capacity += bucket_capacity

        print(f"{bucket:<20} ${avg_adv/1e6:>13.1f}M ${max_position/1e6:>13.1f}M "
              f"${avg_adv * participation_rate/1e6:>13.2f}M {n_stocks:>10,}")

        capacity_results[bucket] = {
            'avg_adv': avg_adv,
            'max_position': max_position,
            'n_stocks': n_stocks,
            'bucket_capacity': bucket_capacity
        }

    print("-" * 70)
    print(f"\nTotal Estimated Capacity: ${total_capacity/1e9:.2f}B")
    print(f"  (assuming 5% ADV participation, 21-day entry)")

    # More realistic: large-cap only
    large_cap_capacity = capacity_results.get('Large (>$10B)', {}).get('bucket_capacity', 0)
    mid_cap_capacity = capacity_results.get('Mid ($2B-10B)', {}).get('bucket_capacity', 0)

    print(f"\nInstitutional-Grade Capacity:")
    print(f"  Large-cap only: ${large_cap_capacity/1e9:.2f}B")
    print(f"  Large + Mid:    ${(large_cap_capacity + mid_cap_capacity)/1e9:.2f}B")

    # Compute spread by cap bucket
    print("\nQ5-Q1 Spread by Market Cap (to assess capacity vs. alpha tradeoff):")
    print("-" * 50)

    # Need to add cap_bucket to full oos_df
    oos_df['cap_bucket'] = pd.cut(oos_df['market_cap'], bins=cap_bins, labels=cap_labels)

    spread_by_cap = {}
    for bucket in cap_labels:
        bucket_df = oos_df[oos_df['cap_bucket'] == bucket].dropna(subset=['quintile', 'fwd_ret'])

        if len(bucket_df) < 100:
            continue

        q5_ret = bucket_df[bucket_df['quintile'] == 'Q5']['fwd_ret'].mean()
        q1_ret = bucket_df[bucket_df['quintile'] == 'Q1']['fwd_ret'].mean()
        spread = q5_ret - q1_ret

        print(f"  {bucket:<20}: {spread:+.2f}%")
        spread_by_cap[bucket] = spread

    return {
        'total_capacity': total_capacity,
        'large_cap_capacity': large_cap_capacity,
        'capacity_by_bucket': capacity_results,
        'spread_by_cap': spread_by_cap
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE ROBUSTNESS TESTS FOR COMPASS SCORE")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Load data
    prices, fund, sample_symbols = load_and_prepare_data(sample_size=2500)

    # Compute observations with monthly sampling (for HAC test)
    df_monthly = compute_observations(prices, fund, sample_symbols,
                                       return_horizon=63, sample_freq=21)
    is_df, oos_df = apply_is_oos_split_and_scores(df_monthly)

    # Download FF factors
    ff6 = download_ff_factors()

    results = {}

    # TEST 1: HAC Regression
    if ff6 is not None:
        results['hac'] = test_hac_regression(oos_df.copy(), ff6)

    # TEST 2: Non-Overlapping Returns
    results['non_overlapping'] = test_non_overlapping_returns(prices, fund, sample_symbols)

    # TEST 3: Capacity Modeling
    results['capacity'] = test_capacity_modeling(oos_df.copy())

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH PAPER")
    print("=" * 70)

    print("""
## Updated Results for RESEARCH_PAPER.md

### 1. HAC-Corrected FF6 Regression (Section 4.2)
""")

    if results.get('hac'):
        hac = results['hac']
        ols_t = hac['OLS']['t_stat']
        hac_t = hac['HAC-12']['t_stat']
        reduction = (ols_t - hac_t) / ols_t * 100

        print(f"| Method | Alpha (ann) | t-statistic |")
        print(f"|--------|-------------|-------------|")
        print(f"| OLS | {hac['OLS']['alpha']:+.2f}% | {ols_t:+.2f} |")
        print(f"| HAC-12 (Newey-West) | {hac['HAC-12']['alpha']:+.2f}% | {hac_t:+.2f} |")
        print(f"\nt-stat reduction with HAC: {reduction:.1f}%")
        print(f"Still significant at 5%: {'YES' if abs(hac_t) > 2 else 'NO'}")

    print("""
### 2. Non-Overlapping Returns (New Section 5.6)
""")

    if results.get('non_overlapping'):
        no = results['non_overlapping']
        print(f"OOS Q5-Q1 Spread (non-overlapping): {no['OOS']['spread']:+.2f}%")
        print(f"OOS quarters: {no['OOS']['n_quarters']}")
        print(f"OOS observations: {no['OOS']['n_obs']:,}")

    print("""
### 3. Capacity Analysis (New Section 5.4)
""")

    if results.get('capacity'):
        cap = results['capacity']
        print(f"Total estimated capacity: ${cap['total_capacity']/1e9:.2f}B")
        print(f"Large-cap only capacity: ${cap['large_cap_capacity']/1e9:.2f}B")
        print(f"\nSpread by market cap:")
        for bucket, spread in cap['spread_by_cap'].items():
            print(f"  {bucket}: {spread:+.2f}%")

    print(f"\nCompleted: {datetime.now()}")

    return results


if __name__ == '__main__':
    main()

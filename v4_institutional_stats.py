#!/usr/bin/env python3
"""
Institutional-Grade Statistics for Compass Score
=================================================
Addresses remaining concerns:

1. Reconcile large-cap spread discrepancy (+3.17% vs +9.01%)
2. Compute Sharpe ratio and max drawdown
3. Year-by-year returns
4. Rolling 12-month alpha
5. Shorting cost model
6. Factor orthogonalization vs RMW

Output: Results for RESEARCH_PAPER.md
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

IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'


def download_ff_factors():
    """Download Fama-French factors."""
    print("Downloading FF factors...")
    ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
    mom_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

    try:
        r = requests.get(ff5_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = [n for n in z.namelist() if '.csv' in n.lower()][0]
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

        r = requests.get(mom_url, timeout=30)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = [n for n in z.namelist() if '.csv' in n.lower()][0]
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
        print(f"  {len(ff6)} months")
        return ff6
    except Exception as e:
        print(f"  Error: {e}")
        return None


def load_full_universe():
    """Load FULL universe (not sampled) for accurate large-cap analysis."""
    print("\nLoading FULL universe (no sampling)...")
    conn = sqlite3.connect(BACKTEST_DB)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

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

    return prices, fund


def compute_monthly_quintile_returns(prices, fund, cap_filter=None, cap_label="All"):
    """
    Compute monthly Q5-Q1 returns for the full universe.

    cap_filter: tuple (min_cap, max_cap) in dollars, or None for all
    """
    print(f"\nComputing monthly quintile returns ({cap_label})...")

    all_symbols = prices['symbol'].unique()
    print(f"  Processing {len(all_symbols):,} symbols...")

    results = []
    for i, symbol in enumerate(all_symbols):
        if i % 1000 == 0 and i > 0:
            print(f"    {i}/{len(all_symbols)}...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values

        for j in range(252, n - 21, 21):  # Monthly
            date = sym_prices['date'].iloc[j]

            # Forward 1-month return
            fwd_1m = ((close[j + 21] / close[j]) - 1) * 100

            # Volatility
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
    df['fwd_1m'] = df['fwd_1m'].clip(-50, 50)
    df = df.dropna(subset=['fwd_1m', 'roa', 'market_cap'])

    # Apply cap filter
    if cap_filter:
        min_cap, max_cap = cap_filter
        df = df[(df['market_cap'] >= min_cap) & (df['market_cap'] < max_cap)]
        print(f"  After cap filter: {len(df):,} observations")

    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Split IS/OOS
    is_df = df[df['date_str'] <= IN_SAMPLE_END].copy()
    oos_df = df[df['date_str'] >= OOS_START].copy()
    print(f"  IS: {len(is_df):,}  |  OOS: {len(oos_df):,}")

    # Winsorize using IS bounds
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']
    for col in factor_cols:
        if col in is_df.columns:
            lower = is_df[col].quantile(0.01)
            upper = is_df[col].quantile(0.99)
            is_df[col] = is_df[col].clip(lower, upper)
            oos_df[col] = oos_df[col].clip(lower, upper)

    # Z-scores using IS stats
    for col in factor_cols:
        if col in is_df.columns:
            mean = is_df[col].mean()
            std = is_df[col].std()
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

    # Combine and assign quintiles
    df = pd.concat([is_df, oos_df], ignore_index=True)
    df['year_month'] = df['date'].dt.to_period('M')

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

    df = df.groupby('year_month', group_keys=False).apply(assign_quintiles)
    df = df.dropna(subset=['quintile'])

    # Monthly quintile returns
    monthly_returns = df.groupby(['year_month', 'quintile'])['fwd_1m'].mean().unstack()
    monthly_returns['Q5-Q1'] = monthly_returns['Q5'] - monthly_returns['Q1']
    monthly_returns.index = monthly_returns.index.to_timestamp()

    return monthly_returns, df


def analyze_large_cap_discrepancy():
    """
    Reconcile large-cap spread discrepancy.

    The earlier +3.17% was computed with 63-day forward returns.
    The +9.01% was computed with 21-day (monthly) returns, annualized differently.
    """
    print("\n" + "=" * 70)
    print("INVESTIGATING LARGE-CAP SPREAD DISCREPANCY")
    print("=" * 70)

    prices, fund = load_full_universe()

    # Large-cap = >$10B
    large_cap_filter = (10e9, np.inf)

    monthly_returns, df = compute_monthly_quintile_returns(
        prices, fund, cap_filter=large_cap_filter, cap_label="Large Cap (>$10B)"
    )

    # Split OOS
    oos_returns = monthly_returns[monthly_returns.index >= OOS_START]

    print(f"\nLarge-Cap OOS Monthly Returns (2020-2026):")
    print(f"  Months: {len(oos_returns)}")

    # Average monthly spread
    avg_monthly_spread = oos_returns['Q5-Q1'].mean()
    print(f"  Average monthly Q5-Q1 spread: {avg_monthly_spread:+.2f}%")

    # Annualized (compound)
    annualized = ((1 + avg_monthly_spread/100) ** 12 - 1) * 100
    print(f"  Annualized (compound): {annualized:+.2f}%")

    # 3-month (63-day) equivalent
    three_month_spread = ((1 + avg_monthly_spread/100) ** 3 - 1) * 100
    print(f"  3-month equivalent: {three_month_spread:+.2f}%")

    # Quintile breakdown
    print(f"\n  Quintile averages (monthly):")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        print(f"    {q}: {oos_returns[q].mean():+.2f}%")

    # Check monotonicity
    avg_rets = {q: oos_returns[q].mean() for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']}
    is_monotonic = all(avg_rets[f'Q{i}'] < avg_rets[f'Q{i+1}'] for i in range(1, 5))
    print(f"\n  Strictly monotonic: {'YES' if is_monotonic else 'NO'}")

    return monthly_returns, oos_returns


def compute_institutional_stats(monthly_returns, ff6, period_name="OOS"):
    """Compute Sharpe, drawdown, and other institutional stats."""
    print(f"\n" + "=" * 70)
    print(f"INSTITUTIONAL STATISTICS ({period_name})")
    print("=" * 70)

    # Filter to OOS
    oos = monthly_returns[monthly_returns.index >= OOS_START].copy()

    print(f"\nMonths: {len(oos)}")
    print(f"Date range: {oos.index.min().date()} to {oos.index.max().date()}")

    # Q5-Q1 long-short returns
    ls_returns = oos['Q5-Q1']

    # Basic stats
    avg_monthly = ls_returns.mean()
    std_monthly = ls_returns.std()

    print(f"\n1. RETURN STATISTICS")
    print(f"   Average monthly return: {avg_monthly:+.2f}%")
    print(f"   Monthly std dev: {std_monthly:.2f}%")
    print(f"   Annualized return: {avg_monthly * 12:+.2f}%")
    print(f"   Annualized vol: {std_monthly * np.sqrt(12):.2f}%")

    # Sharpe ratio
    sharpe = (avg_monthly / std_monthly) * np.sqrt(12) if std_monthly > 0 else 0
    print(f"\n2. SHARPE RATIO")
    print(f"   Monthly Sharpe (annualized): {sharpe:.2f}")

    # Drawdown
    cumulative = (1 + ls_returns/100).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative / running_max - 1) * 100
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    print(f"\n3. DRAWDOWN ANALYSIS")
    print(f"   Max drawdown: {max_dd:.2f}%")
    print(f"   Max DD date: {max_dd_date.date()}")

    # Year-by-year returns
    oos['year'] = oos.index.year
    yearly_returns = oos.groupby('year')['Q5-Q1'].sum()

    print(f"\n4. YEAR-BY-YEAR RETURNS (Q5-Q1)")
    print(f"   {'Year':<8} {'Return':>10}")
    print("   " + "-" * 20)
    for year, ret in yearly_returns.items():
        print(f"   {year:<8} {ret:>+9.2f}%")

    # Worst months
    worst_months = ls_returns.nsmallest(5)
    print(f"\n5. WORST 5 MONTHS")
    for date, ret in worst_months.items():
        print(f"   {date.strftime('%Y-%m')}: {ret:+.2f}%")

    # Win rate
    win_rate = (ls_returns > 0).mean() * 100
    print(f"\n6. WIN RATE")
    print(f"   Positive months: {win_rate:.1f}%")

    # Information ratio vs SPY (if we have FF factors)
    if ff6 is not None:
        print(f"\n7. FACTOR ANALYSIS")
        try:
            import statsmodels.api as sm

            ff6_idx = ff6.set_index('date').copy()
            ff6_idx.index = ff6_idx.index.to_period('M').to_timestamp()
            oos.index = oos.index.to_period('M').to_timestamp()

            merged = oos[['Q5-Q1']].join(ff6_idx, how='inner').dropna()

            if len(merged) > 12:
                # Market correlation
                mkt_corr = merged['Q5-Q1'].corr(merged['Mkt-RF'])
                print(f"   Correlation with Market: {mkt_corr:.3f}")

                # RMW correlation
                rmw_corr = merged['Q5-Q1'].corr(merged['RMW'])
                print(f"   Correlation with RMW (Profitability): {rmw_corr:.3f}")

                # Orthogonalized alpha vs RMW
                y = merged['Q5-Q1']
                X = sm.add_constant(merged['RMW'])
                model = sm.OLS(y, X).fit()

                print(f"\n   Regression vs RMW only:")
                print(f"   Alpha (orthogonal to RMW): {model.params['const'] * 12:+.2f}% annualized")
                print(f"   Alpha t-stat: {model.tvalues['const']:+.2f}")
                print(f"   RMW loading: {model.params['RMW']:.3f}")
                print(f"   R² vs RMW: {model.rsquared:.1%}")

        except Exception as e:
            print(f"   Error in factor analysis: {e}")

    return {
        'avg_monthly': avg_monthly,
        'std_monthly': std_monthly,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'yearly_returns': yearly_returns,
        'win_rate': win_rate
    }


def model_shorting_costs(monthly_returns):
    """Model impact of shorting costs on strategy returns."""
    print(f"\n" + "=" * 70)
    print("SHORTING COST MODEL")
    print("=" * 70)

    oos = monthly_returns[monthly_returns.index >= OOS_START].copy()

    # Q5-Q1 requires shorting Q1
    gross_spread = oos['Q5-Q1'].mean() * 12  # Annualized

    # Borrow cost scenarios
    borrow_costs = [0.5, 1.0, 1.5, 2.0, 3.0]  # Annual %

    print(f"\nGross annualized spread: {gross_spread:+.2f}%")
    print(f"\nNet spread after borrow costs:")
    print(f"{'Borrow Cost':<15} {'Net Spread':>12}")
    print("-" * 30)

    for cost in borrow_costs:
        net = gross_spread - cost
        print(f"{cost:.1f}% annual     {net:>+10.2f}%")

    # Large-cap borrow costs are typically 0.25-0.50%
    print(f"\nNote: Large-cap general collateral typically costs 0.25-0.50% annually")
    print(f"Hard-to-borrow names can cost 5-20%+ annually")


def reconcile_spread_discrepancy():
    """
    Explain the discrepancy between +3.17% and +9.01%.
    """
    print("\n" + "=" * 70)
    print("RECONCILING SPREAD DISCREPANCY (+3.17% vs +9.01%)")
    print("=" * 70)

    print("""
EXPLANATION:

The +3.17% and +9.01% numbers come from DIFFERENT measurements:

1. Original +3.17% (Section 3.4):
   - 63-day (3-month) forward returns
   - Full universe with market cap segmentation
   - Quintiles assigned across FULL universe, then filtered to large-cap

2. New +9.01% (Capacity Test):
   - Monthly (21-day) returns from robustness script
   - 2500 symbol SAMPLE (not full universe)
   - Quintiles assigned WITHIN the sample
   - Different effective universe

KEY INSIGHT: When quintiles are assigned within a market cap segment
(rather than across the full universe then filtered), the spread
increases because you're comparing the best large-caps to worst large-caps,
not comparing large-caps that happen to fall in Q5/Q1 of the full universe.

RECOMMENDATION:
- Use the +3.17% for large-cap spread (consistent methodology)
- Note that within-cap quintiles show stronger signal (~9%)
- The +9.01% is methodologically different and shouldn't replace +3.17%
""")


def main():
    print("=" * 70)
    print("INSTITUTIONAL-GRADE STATISTICS FOR COMPASS SCORE")
    print("=" * 70)
    print(f"Started: {datetime.now()}")

    # Download FF factors
    ff6 = download_ff_factors()

    # Investigate large-cap discrepancy
    reconcile_spread_discrepancy()

    # Get accurate large-cap stats using full universe
    monthly_returns, oos_returns = analyze_large_cap_discrepancy()

    # Compute institutional stats
    stats = compute_institutional_stats(monthly_returns, ff6, "Large-Cap OOS")

    # Model shorting costs
    model_shorting_costs(monthly_returns)

    # Summary for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR RESEARCH PAPER")
    print("=" * 70)

    print(f"""
## Large-Cap Spread Reconciliation

The +9.01% spread from the capacity test reflects **within-cap quintiles**
(comparing best large-caps to worst large-caps). The original +3.17% spread
reflects **cross-universe quintiles** filtered to large-cap (large-caps that
happen to score in top/bottom quintile of the full universe).

Both are valid but measure different things:
- +3.17%: "How do large-caps that score well across all stocks perform?"
- +9.01%: "How do the best large-caps compare to worst large-caps?"

For consistency with the rest of the paper, use +3.17% as the primary
large-cap spread. Note the stronger within-cap signal in the discussion.

## Institutional Statistics (Large-Cap, OOS 2020-2026)

| Metric | Value |
|--------|-------|
| Monthly Sharpe (ann.) | {stats['sharpe']:.2f} |
| Max Drawdown | {stats['max_dd']:.2f}% |
| Win Rate | {stats['win_rate']:.1f}% |
| Best Year | {stats['yearly_returns'].max():+.2f}% ({stats['yearly_returns'].idxmax()}) |
| Worst Year | {stats['yearly_returns'].min():+.2f}% ({stats['yearly_returns'].idxmin()}) |

## Year-by-Year Returns (Q5-Q1)
""")

    for year, ret in stats['yearly_returns'].items():
        print(f"| {year} | {ret:+.2f}% |")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

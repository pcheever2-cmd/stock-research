#!/usr/bin/env python3
"""
V4 Regime Analysis
==================
Identify market conditions where quality factors underperform,
and potential indicators for regime switching.
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


def load_and_prepare_data():
    """Load data and compute V4 scores with proper IS normalization."""
    print("Loading data...")
    conn = sqlite3.connect(BACKTEST_DB)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])

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


def compute_monthly_spreads(prices, fund, sample_size=2000):
    """Compute monthly V4 quintile spreads with market conditions."""
    print("\nComputing monthly spreads with market conditions...")

    # Sample symbols
    np.random.seed(42)
    symbols = prices['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(sample_size, len(symbols)), replace=False)
    prices = prices[prices['symbol'].isin(sample_symbols)]

    # Compute price factors
    results = []
    for i, symbol in enumerate(sample_symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(sample_symbols)} symbols...", flush=True)

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values

        for j in range(252, n - 63, 21):
            date = sym_prices['date'].iloc[j]

            # Forward 3m return
            fwd_3m = ((close[j + 63] / close[j]) - 1) * 100

            # Momentum 12-1
            mom_12_1 = ((close[j - 21] / close[j - 252]) - 1) * 100 if close[j - 252] > 0 else np.nan

            # Volatility
            rets = np.diff(close[j-60:j+1]) / close[j-60:j]
            vol_60d = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 20 else np.nan

            # Past 3m return (for market regime)
            past_3m = ((close[j] / close[j - 63]) - 1) * 100 if j >= 63 else np.nan

            results.append({
                'symbol': symbol,
                'date': date,
                'fwd_3m': fwd_3m,
                'mom_12_1': mom_12_1,
                'vol_60d': vol_60d,
                'past_3m': past_3m,
            })

    df = pd.DataFrame(results)

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
    df['fwd_3m'] = df['fwd_3m'].clip(-100, 100)
    df = df.dropna(subset=['fwd_3m', 'roa'])
    df['year_month'] = df['date'].dt.to_period('M')

    return df


def analyze_regimes(df):
    """Analyze V4 performance by market regime."""
    print("\n" + "=" * 70)
    print("REGIME ANALYSIS")
    print("=" * 70)

    # Compute z-scores per month (rolling, no look-ahead)
    factor_cols = ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']

    # For simplicity, use expanding window for z-scores
    monthly_results = []

    for ym in sorted(df['year_month'].unique()):
        month_df = df[df['year_month'] == ym].copy()

        if len(month_df) < 100:
            continue

        # Use past data for z-scores (expanding window)
        past_df = df[df['year_month'] < ym]
        if len(past_df) < 500:
            continue

        # Compute z-scores using past data
        for col in factor_cols:
            if col in month_df.columns and col in past_df.columns:
                mean = past_df[col].mean()
                std = past_df[col].std()
                if std > 0:
                    month_df[f'{col}_z'] = (month_df[col] - mean) / std
                    month_df[f'{col}_z'] = month_df[f'{col}_z'].clip(-3, 3).fillna(0)

        # Compute V4 score
        month_df['v4_score'] = (
            month_df['roa_z'] * 0.20 +
            month_df['ocf_assets_z'] * 0.15 +
            month_df['fcf_assets_z'] * 0.15 +
            month_df['gp_assets_z'] * 0.10 +
            (-month_df['vol_60d_z']) * 0.15 +
            (-month_df['asset_growth_z']) * 0.15
        )

        # Compute quintile spread
        try:
            month_df['quintile'] = pd.qcut(month_df['v4_score'].rank(method='first'), 5,
                                           labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            q5_ret = month_df[month_df['quintile'] == 'Q5']['fwd_3m'].mean()
            q1_ret = month_df[month_df['quintile'] == 'Q1']['fwd_3m'].mean()
            spread = q5_ret - q1_ret
        except:
            continue

        # Market conditions
        avg_vol = month_df['vol_60d'].mean()
        avg_mom = month_df['mom_12_1'].mean()
        avg_past_3m = month_df['past_3m'].mean()
        mkt_return = month_df['fwd_3m'].mean()

        # Momentum spread (high mom - low mom)
        try:
            month_df['mom_quintile'] = pd.qcut(month_df['mom_12_1'].rank(method='first'), 5,
                                               labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
            mom_spread = (month_df[month_df['mom_quintile'] == 'Q5']['fwd_3m'].mean() -
                         month_df[month_df['mom_quintile'] == 'Q1']['fwd_3m'].mean())
        except:
            mom_spread = np.nan

        monthly_results.append({
            'year_month': ym,
            'date': ym.to_timestamp(),
            'v4_spread': spread,
            'q5_ret': q5_ret,
            'q1_ret': q1_ret,
            'avg_vol': avg_vol,
            'avg_mom': avg_mom,
            'avg_past_3m': avg_past_3m,
            'mkt_return': mkt_return,
            'mom_spread': mom_spread,
            'n_stocks': len(month_df),
        })

    results_df = pd.DataFrame(monthly_results)
    results_df['year'] = results_df['date'].dt.year

    print(f"\nAnalyzed {len(results_df)} months")

    return results_df


def identify_underperformance_patterns(results_df):
    """Identify patterns when V4 underperforms."""
    print("\n" + "=" * 70)
    print("UNDERPERFORMANCE ANALYSIS")
    print("=" * 70)

    # Define underperformance (negative spread)
    results_df['underperform'] = results_df['v4_spread'] < 0

    underperform_months = results_df[results_df['underperform']]
    outperform_months = results_df[~results_df['underperform']]

    print(f"\nTotal months: {len(results_df)}")
    print(f"Underperforming months (spread < 0): {len(underperform_months)} ({len(underperform_months)/len(results_df)*100:.1f}%)")
    print(f"Outperforming months (spread >= 0): {len(outperform_months)} ({len(outperform_months)/len(results_df)*100:.1f}%)")

    # Compare characteristics
    print("\n" + "-" * 70)
    print("MARKET CONDITIONS: Underperform vs Outperform")
    print("-" * 70)

    metrics = ['avg_vol', 'avg_mom', 'avg_past_3m', 'mkt_return', 'mom_spread']
    labels = ['Avg Volatility', 'Avg Momentum 12-1', 'Past 3M Return', 'Fwd Market Return', 'Momentum Spread']

    print(f"\n{'Metric':<25} {'Underperform':>15} {'Outperform':>15} {'Difference':>15}")
    print("-" * 70)

    for metric, label in zip(metrics, labels):
        under_val = underperform_months[metric].mean()
        out_val = outperform_months[metric].mean()
        diff = under_val - out_val
        print(f"{label:<25} {under_val:>+14.2f}% {out_val:>+14.2f}% {diff:>+14.2f}%")

    # By year
    print("\n" + "-" * 70)
    print("V4 SPREAD BY YEAR")
    print("-" * 70)

    yearly = results_df.groupby('year').agg({
        'v4_spread': ['mean', 'std', 'count'],
        'underperform': 'sum',
        'mom_spread': 'mean',
        'avg_vol': 'mean',
    })
    yearly.columns = ['avg_spread', 'std_spread', 'months', 'underperform_months', 'mom_spread', 'avg_vol']
    yearly['underperform_pct'] = yearly['underperform_months'] / yearly['months'] * 100

    print(f"\n{'Year':>6} {'Avg Spread':>12} {'Std':>8} {'Under%':>10} {'Mom Spread':>12} {'Avg Vol':>10}")
    print("-" * 60)
    for year, row in yearly.iterrows():
        print(f"{year:>6} {row['avg_spread']:>+10.2f}% {row['std_spread']:>8.2f} "
              f"{row['underperform_pct']:>9.1f}% {row['mom_spread']:>+10.2f}% {row['avg_vol']:>10.1f}")

    return results_df, underperform_months, outperform_months


def find_regime_indicators(results_df, underperform_months, outperform_months):
    """Find potential indicators for regime switching."""
    print("\n" + "=" * 70)
    print("REGIME INDICATORS")
    print("=" * 70)

    # Correlations with spread
    print("\nCorrelation of market conditions with V4 spread:")
    print("-" * 50)

    correlations = {}
    for col in ['avg_vol', 'avg_mom', 'avg_past_3m', 'mkt_return', 'mom_spread']:
        corr = results_df['v4_spread'].corr(results_df[col])
        correlations[col] = corr
        print(f"  {col:<20}: {corr:>+.3f}")

    # Best predictor
    best_predictor = max(correlations.keys(), key=lambda k: abs(correlations[k]))
    print(f"\n  Best predictor: {best_predictor} (r = {correlations[best_predictor]:+.3f})")

    # Test threshold-based rules
    print("\n" + "-" * 70)
    print("THRESHOLD-BASED REGIME RULES")
    print("-" * 70)

    # Test: When momentum spread is positive, does quality underperform?
    results_df['mom_positive'] = results_df['mom_spread'] > 0

    mom_pos = results_df[results_df['mom_positive']]
    mom_neg = results_df[~results_df['mom_positive']]

    print(f"\nWhen MOMENTUM is working (mom_spread > 0):")
    print(f"  V4 avg spread: {mom_pos['v4_spread'].mean():+.2f}%")
    print(f"  Months: {len(mom_pos)}")

    print(f"\nWhen MOMENTUM is NOT working (mom_spread <= 0):")
    print(f"  V4 avg spread: {mom_neg['v4_spread'].mean():+.2f}%")
    print(f"  Months: {len(mom_neg)}")

    # Test: High volatility regime
    vol_median = results_df['avg_vol'].median()
    high_vol = results_df[results_df['avg_vol'] > vol_median]
    low_vol = results_df[results_df['avg_vol'] <= vol_median]

    print(f"\nHigh volatility regime (vol > {vol_median:.1f}%):")
    print(f"  V4 avg spread: {high_vol['v4_spread'].mean():+.2f}%")

    print(f"\nLow volatility regime (vol <= {vol_median:.1f}%):")
    print(f"  V4 avg spread: {low_vol['v4_spread'].mean():+.2f}%")

    # Test: Market direction
    up_market = results_df[results_df['avg_past_3m'] > 0]
    down_market = results_df[results_df['avg_past_3m'] <= 0]

    print(f"\nUp market (past 3M > 0):")
    print(f"  V4 avg spread: {up_market['v4_spread'].mean():+.2f}%")
    print(f"  Mom spread:    {up_market['mom_spread'].mean():+.2f}%")

    print(f"\nDown market (past 3M <= 0):")
    print(f"  V4 avg spread: {down_market['v4_spread'].mean():+.2f}%")
    print(f"  Mom spread:    {down_market['mom_spread'].mean():+.2f}%")

    # Combined rule
    print("\n" + "-" * 70)
    print("COMBINED REGIME RULE")
    print("-" * 70)

    # V4 struggles when: momentum is working AND market is up (speculative regime)
    speculative = results_df[(results_df['mom_spread'] > 5) & (results_df['avg_past_3m'] > 5)]
    not_speculative = results_df[~((results_df['mom_spread'] > 5) & (results_df['avg_past_3m'] > 5))]

    print(f"\nSPECULATIVE REGIME (mom_spread > 5% AND past_3m > 5%):")
    print(f"  V4 avg spread: {speculative['v4_spread'].mean():+.2f}%")
    print(f"  Months: {len(speculative)} ({len(speculative)/len(results_df)*100:.1f}%)")

    print(f"\nNORMAL/RISK-OFF REGIME:")
    print(f"  V4 avg spread: {not_speculative['v4_spread'].mean():+.2f}%")
    print(f"  Months: {len(not_speculative)} ({len(not_speculative)/len(results_df)*100:.1f}%)")

    return correlations


def suggest_alternatives(results_df):
    """Suggest alternative strategies for underperforming regimes."""
    print("\n" + "=" * 70)
    print("ALTERNATIVE STRATEGIES FOR UNDERPERFORMING REGIMES")
    print("=" * 70)

    # When momentum is working, use momentum instead of quality
    print("""
RECOMMENDATION:

1. **Regime Detection Signal**: Monitor MOMENTUM SPREAD
   - When momentum quintile spread > 5%: Momentum is working
   - When momentum quintile spread <= 5%: Quality factors preferred

2. **Alternative Strategy in Speculative Regimes**:
   - Instead of V4 (quality), switch to momentum-based strategy
   - Or reduce position sizes when in speculative regime
   - Or blend: V4 weight = max(0.5, 1 - momentum_spread/20)

3. **Implementation**:
   ```
   if momentum_spread > 5% AND past_3m > 5%:
       # Speculative regime - momentum works better
       use_momentum_strategy() OR reduce_v4_exposure()
   else:
       # Normal regime - quality works better
       use_v4_strategy()
   ```

4. **Historical Performance of Switching Rule**:
""")

    # Simulate switching rule
    speculative = (results_df['mom_spread'] > 5) & (results_df['avg_past_3m'] > 5)

    # In speculative: use momentum; otherwise: use V4
    results_df['switched_spread'] = np.where(
        speculative,
        results_df['mom_spread'],  # Use momentum in speculative
        results_df['v4_spread']    # Use V4 otherwise
    )

    print(f"   Pure V4 avg spread:     {results_df['v4_spread'].mean():+.2f}%")
    print(f"   Pure Momentum spread:   {results_df['mom_spread'].mean():+.2f}%")
    print(f"   Switched strategy:      {results_df['switched_spread'].mean():+.2f}%")

    improvement = results_df['switched_spread'].mean() - results_df['v4_spread'].mean()
    print(f"   Improvement:            {improvement:+.2f}%")


def main():
    print("=" * 70)
    print("V4 REGIME ANALYSIS")
    print("What conditions cause quality factors to underperform?")
    print("=" * 70)
    print(f"Started: {datetime.now()}\n")

    # Load data
    prices, fund = load_and_prepare_data()

    # Compute monthly spreads with market conditions
    df = compute_monthly_spreads(prices, fund)

    # Analyze regimes
    results_df = analyze_regimes(df)

    # Identify underperformance patterns
    results_df, underperform, outperform = identify_underperformance_patterns(results_df)

    # Find regime indicators
    correlations = find_regime_indicators(results_df, underperform, outperform)

    # Suggest alternatives
    suggest_alternatives(results_df)

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

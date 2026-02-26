#!/usr/bin/env python3
"""
Moonshot Score Analysis (Price-Only)
====================================
Identify factors that predict 3x returns using ONLY price/volume data.
This avoids the fundamental data bottleneck.

Entry Criteria:
- Stock price < $20
- No market cap filter (use price as proxy)

Factors to Test:
- Entry price (lower = more potential?)
- 12-month momentum (winners have momentum?)
- Volatility (higher vol = higher upside?)
- Volume trend (accumulation signal?)
- Distance from 52-week high
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

# Entry criteria
MAX_PRICE = 20.0

# Target return
TARGET_RETURN = 300  # 3x = 300% gain

# Horizons (trading days)
HORIZONS = {
    '3-Year': 756,
    '5-Year': 1260
}

# Analysis period
ANALYSIS_START = '2005-01-01'
ANALYSIS_END = '2018-01-01'  # Need 5 years forward


def load_prices():
    """Load price data."""
    print("Loading price data...")
    conn = sqlite3.connect(BACKTEST_DB)

    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 0.5  -- Filter penny stocks
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records, {prices['symbol'].nunique():,} symbols")

    conn.close()
    return prices


def find_candidates_and_compute_factors(prices):
    """Find candidates under $20 and compute price-based factors."""
    print(f"\nFinding candidates (price < ${MAX_PRICE})...")

    results = []
    symbols = prices['symbol'].unique()
    max_horizon = max(HORIZONS.values())

    for i, symbol in enumerate(symbols):
        if i % 500 == 0:
            print(f"  {i}/{len(symbols)} symbols...")

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').reset_index(drop=True)
        n = len(sym_prices)

        if n < 300 + max_horizon:
            continue

        close = sym_prices['close'].values
        volume = sym_prices['volume'].values
        dates = pd.to_datetime(sym_prices['date'].values)

        # Sample quarterly (every 63 days)
        for j in range(252, n - max_horizon, 63):
            entry_price = close[j]
            date = dates[j]

            # Filter to analysis period
            if date < pd.Timestamp(ANALYSIS_START) or date > pd.Timestamp(ANALYSIS_END):
                continue

            # Entry criteria
            if entry_price >= MAX_PRICE or entry_price < 0.5:
                continue

            # Compute price-based factors

            # 1. Entry price (log)
            log_price = np.log10(entry_price)

            # 2. 12-month momentum
            if j >= 252 and close[j-252] > 0:
                mom_12m = ((close[j] / close[j-252]) - 1) * 100
            else:
                mom_12m = np.nan

            # 3. 60-day volatility
            if j >= 60:
                rets = np.diff(close[j-60:j+1]) / close[j-60:j]
                vol_60d = np.std(rets) * np.sqrt(252) * 100
            else:
                vol_60d = np.nan

            # 4. Volume trend (ratio of recent 20d avg to 60d avg)
            if j >= 60:
                vol_20d = np.mean(volume[j-20:j])
                vol_60d_avg = np.mean(volume[j-60:j])
                vol_ratio = vol_20d / vol_60d_avg if vol_60d_avg > 0 else np.nan
            else:
                vol_ratio = np.nan

            # 5. Distance from 52-week high
            if j >= 252:
                high_52w = np.max(close[j-252:j])
                pct_from_high = ((entry_price / high_52w) - 1) * 100
            else:
                pct_from_high = np.nan

            # 6. Recent trend (price vs 50-day MA)
            if j >= 50:
                ma_50 = np.mean(close[j-50:j])
                above_ma50 = 1 if entry_price > ma_50 else 0
            else:
                above_ma50 = np.nan

            row = {
                'symbol': symbol,
                'date': date,
                'entry_price': entry_price,
                'log_price': log_price,
                'mom_12m': np.clip(mom_12m, -90, 500) if not np.isnan(mom_12m) else np.nan,
                'vol_60d': vol_60d,
                'vol_ratio': vol_ratio,
                'pct_from_high': pct_from_high,
                'above_ma50': above_ma50,
            }

            # Compute forward returns
            for horizon_name, days in HORIZONS.items():
                if j + days < n:
                    exit_price = close[j + days]
                    fwd_ret = ((exit_price / entry_price) - 1) * 100
                    row[f'fwd_{horizon_name}'] = fwd_ret
                    row[f'is_3x_{horizon_name}'] = fwd_ret >= TARGET_RETURN
                else:
                    row[f'fwd_{horizon_name}'] = np.nan
                    row[f'is_3x_{horizon_name}'] = np.nan

            results.append(row)

    df = pd.DataFrame(results)
    print(f"  {len(df):,} candidate observations")
    return df


def analyze_factor(df, factor, horizon_name):
    """Analyze a single factor's predictive power for 3x returns."""
    return_col = f'fwd_{horizon_name}'
    is_3x_col = f'is_3x_{horizon_name}'

    clean = df.dropna(subset=[factor, return_col, is_3x_col])

    if len(clean) < 100:
        return None

    winners = clean[clean[is_3x_col] == True]
    losers = clean[clean[is_3x_col] == False]

    if len(winners) < 10:
        return None

    winner_mean = winners[factor].mean()
    loser_mean = losers[factor].mean()
    diff = winner_mean - loser_mean

    from scipy import stats
    try:
        t_stat, p_val = stats.ttest_ind(winners[factor], losers[factor], equal_var=False)
    except:
        t_stat = np.nan

    return {
        'factor': factor,
        'n_winners': len(winners),
        'n_losers': len(losers),
        'winner_mean': winner_mean,
        'loser_mean': loser_mean,
        'difference': diff,
        't_stat': t_stat,
    }


def main():
    print("=" * 70)
    print("MOONSHOT ANALYSIS (PRICE-ONLY)")
    print("Finding predictors of 3x returns using price/volume data")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"\nEntry: price < ${MAX_PRICE}")
    print(f"Target: {TARGET_RETURN}% returns (3x)")
    print(f"Analysis period: {ANALYSIS_START} to {ANALYSIS_END}")

    # Load data
    prices = load_prices()

    # Find candidates and compute factors
    df = find_candidates_and_compute_factors(prices)

    # Count winners
    print("\n" + "=" * 70)
    print("3x WINNER COUNTS")
    print("=" * 70)

    for horizon_name in HORIZONS.keys():
        col = f'is_3x_{horizon_name}'
        clean = df.dropna(subset=[col])
        n_winners = clean[col].sum()
        total = len(clean)
        pct = 100 * n_winners / total if total > 0 else 0
        print(f"  {horizon_name}: {int(n_winners):,} winners out of {total:,} ({pct:.2f}%)")

    # Analyze factors
    factors = ['log_price', 'mom_12m', 'vol_60d', 'vol_ratio', 'pct_from_high', 'above_ma50']

    for horizon_name in HORIZONS.keys():
        print("\n" + "=" * 70)
        print(f"FACTOR ANALYSIS: {horizon_name}")
        print("=" * 70)

        return_col = f'fwd_{horizon_name}'
        is_3x_col = f'is_3x_{horizon_name}'
        clean = df.dropna(subset=[is_3x_col])
        n_winners = int(clean[is_3x_col].sum())
        n_total = len(clean)

        print(f"\n  Total observations: {n_total:,}")
        print(f"  3x Winners: {n_winners:,} ({100*n_winners/n_total:.2f}%)")

        if n_winners < 10:
            print("  Too few winners to analyze")
            continue

        print(f"\n  {'Factor':<18} {'Winner Mean':>12} {'Loser Mean':>12} {'Difference':>12} {'T-stat':>10}")
        print("  " + "-" * 70)

        results = []
        for factor in factors:
            result = analyze_factor(df, factor, horizon_name)
            if result:
                results.append(result)
                sig = "**" if abs(result['t_stat']) > 2 else "*" if abs(result['t_stat']) > 1.5 else ""
                print(f"  {result['factor']:<18} {result['winner_mean']:>+11.2f} {result['loser_mean']:>+11.2f} "
                      f"{result['difference']:>+11.2f} {result['t_stat']:>+9.2f}{sig}")

        # Show strongest predictors
        if results:
            results_df = pd.DataFrame(results)
            significant = results_df[abs(results_df['t_stat']) > 1.5]
            if len(significant) > 0:
                print(f"\n  SIGNIFICANT PREDICTORS (|t| > 1.5):")
                for _, row in significant.iterrows():
                    direction = "Higher" if row['difference'] > 0 else "Lower"
                    print(f"    {row['factor']}: {direction} values predict 3x winners (t={row['t_stat']:+.2f})")

    print(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()

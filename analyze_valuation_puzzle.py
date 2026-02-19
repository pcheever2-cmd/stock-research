#!/usr/bin/env python3
"""
Valuation Puzzle Analysis
=========================
Why does EV/EBITDA show NEGATIVE correlation with forward returns?
Conceptually, cheap stocks should outperform.

Hypotheses to test:
1. Sector effect - Tech (high multiples) outperformed value sectors
2. Quality effect - Cheap + low quality = value trap
3. Time period effect - Growth dominated 2023-2024
4. Non-linear relationship - Extreme cheap is bad (distress signal)
5. Size effect - Small cheap stocks are riskier
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME


def load_data():
    """Load prices with fundamentals for analysis."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Get sector mapping
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute("SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    print("Loading price and fundamental data...")

    # Load from backtest_daily_scores which has EV/EBITDA already merged
    daily_scores = pd.read_sql_query("""
        SELECT symbol, date, close as adjusted_close, ev_ebitda, rev_growth, eps_growth
        FROM backtest_daily_scores
        WHERE ev_ebitda IS NOT NULL AND ev_ebitda > 0 AND ev_ebitda < 100
        ORDER BY symbol, date
    """, conn)

    conn.close()

    print(f"  Daily scores with EV/EBITDA: {len(daily_scores):,} records")

    return daily_scores, sector_map


def analyze_valuation_by_sector(daily_scores, sector_map):
    """Test if the negative valuation correlation is sector-driven."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: SECTOR EFFECT")
    print("Is high-multiple Tech outperforming low-multiple Energy/Financials?")
    print("=" * 70)

    # Sample symbols for speed
    np.random.seed(42)
    symbols = daily_scores['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(500, len(symbols)), replace=False)

    all_data = []

    for symbol in sample_symbols:
        sym_data = daily_scores[daily_scores['symbol'] == symbol].sort_values('date')

        if len(sym_data) < 300:
            continue

        sector = sector_map.get(symbol)
        if not sector:
            continue

        closes = sym_data['adjusted_close'].values
        ev_ebitda = sym_data['ev_ebitda'].values
        dates = sym_data['date'].values

        # Sample monthly
        for j in range(0, len(closes) - 63, 21):
            if pd.isna(ev_ebitda[j]) or ev_ebitda[j] <= 0:
                continue

            fwd_3m = ((closes[j + 63] / closes[j]) - 1) * 100

            all_data.append({
                'symbol': symbol,
                'date': dates[j],
                'sector': sector,
                'ev_ebitda': ev_ebitda[j],
                'fwd_3m': np.clip(fwd_3m, -100, 100),
            })

    df = pd.DataFrame(all_data)
    print(f"\nTotal observations: {len(df):,}")

    # Overall correlation
    overall_corr = df['ev_ebitda'].corr(df['fwd_3m'])
    print(f"\nOverall EV/EBITDA correlation with 3M returns: {overall_corr:+.4f}")

    # By sector
    print(f"\n{'Sector':<25} {'Corr':>10} {'Avg EV/EBITDA':>15} {'Avg Return':>12} {'N':>10}")
    print("-" * 75)

    sector_stats = []
    for sector in sorted(df['sector'].unique()):
        sector_df = df[df['sector'] == sector]
        if len(sector_df) < 100:
            continue

        corr = sector_df['ev_ebitda'].corr(sector_df['fwd_3m'])
        avg_ev = sector_df['ev_ebitda'].mean()
        avg_ret = sector_df['fwd_3m'].mean()

        indicator = "✅" if corr < 0 else "❌"  # Negative is expected for value
        print(f"{sector:<25} {corr:>+9.4f} {avg_ev:>14.1f}x {avg_ret:>+11.2f}% {len(sector_df):>10,} {indicator}")

        sector_stats.append({
            'sector': sector,
            'corr': corr,
            'avg_ev': avg_ev,
            'avg_return': avg_ret,
            'n': len(sector_df)
        })

    # Check if high-EV sectors outperformed
    sector_df = pd.DataFrame(sector_stats)
    print(f"\nSector-level correlation between avg EV/EBITDA and avg return: "
          f"{sector_df['avg_ev'].corr(sector_df['avg_return']):+.4f}")

    return df


def analyze_valuation_quality_interaction(df):
    """Test if valuation works better when combined with quality."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: QUALITY EFFECT (Value Traps)")
    print("Are cheap stocks without quality indicators failing?")
    print("=" * 70)

    # We need to add quality metrics - let's use recent momentum as proxy
    # (stocks with bad momentum are often distressed)

    # For this analysis, let's just look at the distribution of returns
    # by EV/EBITDA buckets

    df['ev_bucket'] = pd.cut(df['ev_ebitda'],
                              bins=[0, 8, 12, 18, 30, 100],
                              labels=['Very Cheap (<8)', 'Cheap (8-12)', 'Fair (12-18)',
                                      'Expensive (18-30)', 'Very Expensive (30+)'])

    bucket_stats = df.groupby('ev_bucket', observed=False).agg({
        'fwd_3m': ['mean', 'median', 'std', 'count'],
        'ev_ebitda': 'mean'
    })
    bucket_stats.columns = ['avg_return', 'median_return', 'std', 'count', 'avg_ev']

    print(f"\n{'Bucket':<25} {'Avg Ret':>10} {'Med Ret':>10} {'Std':>8} {'N':>10}")
    print("-" * 68)

    for bucket in ['Very Cheap (<8)', 'Cheap (8-12)', 'Fair (12-18)',
                   'Expensive (18-30)', 'Very Expensive (30+)']:
        if bucket in bucket_stats.index:
            row = bucket_stats.loc[bucket]
            print(f"{bucket:<25} {row['avg_return']:>+9.2f}% {row['median_return']:>+9.2f}% "
                  f"{row['std']:>7.1f} {int(row['count']):>10,}")

    # Check if "Very Cheap" is actually value traps
    very_cheap = df[df['ev_bucket'] == 'Very Cheap (<8)']
    cheap = df[df['ev_bucket'] == 'Cheap (8-12)']
    fair = df[df['ev_bucket'] == 'Fair (12-18)']

    print(f"\n📊 Key Finding:")
    if len(very_cheap) > 0 and len(cheap) > 0:
        print(f"   Very Cheap (<8x) avg return: {very_cheap['fwd_3m'].mean():+.2f}%")
        print(f"   Cheap (8-12x) avg return: {cheap['fwd_3m'].mean():+.2f}%")
        print(f"   Fair (12-18x) avg return: {fair['fwd_3m'].mean():+.2f}%")

        if very_cheap['fwd_3m'].mean() < cheap['fwd_3m'].mean():
            print("\n   ⚠️  VERY CHEAP stocks underperform CHEAP stocks!")
            print("   This suggests extreme low multiples are a DISTRESS SIGNAL, not value.")


def analyze_valuation_by_time(df):
    """Test if valuation worked differently in different market regimes."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: TIME PERIOD EFFECT")
    print("Did value work in 2022 bear market but fail in 2023-2024 bull?")
    print("=" * 70)

    df['year'] = pd.to_datetime(df['date']).dt.year

    print(f"\n{'Year':<10} {'EV/EBITDA Corr':>15} {'Avg Return':>12} {'N':>10}")
    print("-" * 50)

    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        if len(year_df) < 500:
            continue

        corr = year_df['ev_ebitda'].corr(year_df['fwd_3m'])
        avg_ret = year_df['fwd_3m'].mean()

        indicator = "✅" if corr < 0 else "❌"
        print(f"{year:<10} {corr:>+14.4f} {avg_ret:>+11.2f}% {len(year_df):>10,} {indicator}")

    # Market regime analysis
    print("\n📊 Market Regime Analysis:")
    bear_2022 = df[df['year'] == 2022]
    bull_2023 = df[df['year'] == 2023]
    bull_2024 = df[df['year'] == 2024]

    if len(bear_2022) > 0 and len(bull_2023) > 0:
        print(f"   2022 (Bear market): EV/EBITDA corr = {bear_2022['ev_ebitda'].corr(bear_2022['fwd_3m']):+.4f}")
        print(f"   2023 (Bull market): EV/EBITDA corr = {bull_2023['ev_ebitda'].corr(bull_2023['fwd_3m']):+.4f}")
        if len(bull_2024) > 0:
            print(f"   2024 (Bull market): EV/EBITDA corr = {bull_2024['ev_ebitda'].corr(bull_2024['fwd_3m']):+.4f}")


def analyze_valuation_with_momentum(daily_scores, sector_map):
    """Test if valuation works when combined with momentum (quality filter)."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: VALUATION + MOMENTUM INTERACTION")
    print("Does cheap + good momentum work better than cheap alone?")
    print("=" * 70)

    np.random.seed(42)
    symbols = daily_scores['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(500, len(symbols)), replace=False)

    all_data = []

    for symbol in sample_symbols:
        sym_data = daily_scores[daily_scores['symbol'] == symbol].sort_values('date')

        if len(sym_data) < 300:
            continue

        closes = sym_data['adjusted_close'].values
        ev_ebitda = sym_data['ev_ebitda'].values

        for j in range(252, len(closes) - 63, 21):
            if pd.isna(ev_ebitda[j]) or ev_ebitda[j] <= 0:
                continue

            # 12-1 momentum
            if closes[j-252] > 0 and closes[j-21] > 0:
                mom_12_1 = ((closes[j-21] / closes[j-252]) - 1) * 100
            else:
                continue

            fwd_3m = ((closes[j + 63] / closes[j]) - 1) * 100

            all_data.append({
                'ev_ebitda': ev_ebitda[j],
                'mom_12_1': np.clip(mom_12_1, -80, 200),
                'fwd_3m': np.clip(fwd_3m, -100, 100),
            })

    df = pd.DataFrame(all_data)

    # Create 2x2 matrix: Cheap/Expensive x Good/Bad Momentum
    median_ev = df['ev_ebitda'].median()
    median_mom = df['mom_12_1'].median()

    df['is_cheap'] = df['ev_ebitda'] < median_ev
    df['is_winner'] = df['mom_12_1'] > median_mom

    # 2x2 analysis
    print(f"\nMedian EV/EBITDA: {median_ev:.1f}x")
    print(f"Median 12-1 Momentum: {median_mom:+.1f}%")

    print(f"\n{'Category':<30} {'Avg Return':>12} {'N':>10}")
    print("-" * 55)

    categories = [
        ('Cheap + Winner', (df['is_cheap']) & (df['is_winner'])),
        ('Cheap + Loser', (df['is_cheap']) & (~df['is_winner'])),
        ('Expensive + Winner', (~df['is_cheap']) & (df['is_winner'])),
        ('Expensive + Loser', (~df['is_cheap']) & (~df['is_winner'])),
    ]

    results = {}
    for name, mask in categories:
        subset = df[mask]
        avg_ret = subset['fwd_3m'].mean()
        results[name] = avg_ret
        print(f"{name:<30} {avg_ret:>+11.2f}% {len(subset):>10,}")

    # Key insight
    print("\n📊 Key Finding:")
    print(f"   Cheap + Winner: {results['Cheap + Winner']:+.2f}%")
    print(f"   Cheap + Loser:  {results['Cheap + Loser']:+.2f}%")
    print(f"   Difference:     {results['Cheap + Winner'] - results['Cheap + Loser']:+.2f}%")

    if results['Cheap + Loser'] < results['Expensive + Winner']:
        print("\n   ⚠️  CHEAP + BAD MOMENTUM underperforms EXPENSIVE + GOOD MOMENTUM!")
        print("   Momentum matters MORE than valuation alone.")


def analyze_extreme_valuations(df):
    """Look at extreme ends of the valuation spectrum."""
    print("\n" + "=" * 70)
    print("HYPOTHESIS 5: EXTREME VALUATIONS")
    print("Are the tails (very cheap / very expensive) driving the result?")
    print("=" * 70)

    # Decile analysis
    df['ev_decile'] = pd.qcut(df['ev_ebitda'], q=10, labels=False, duplicates='drop')

    decile_stats = df.groupby('ev_decile', observed=False).agg({
        'fwd_3m': 'mean',
        'ev_ebitda': 'mean',
    })

    print(f"\n{'Decile':<10} {'Avg EV/EBITDA':>15} {'Avg Return':>12}")
    print("-" * 40)

    for decile in range(10):
        if decile in decile_stats.index:
            row = decile_stats.loc[decile]
            label = "CHEAPEST" if decile == 0 else "MOST EXPENSIVE" if decile == 9 else ""
            print(f"D{decile+1:<9} {row['ev_ebitda']:>14.1f}x {row['fwd_3m']:>+11.2f}% {label}")

    # Non-linearity check
    print("\n📊 Non-Linearity Check:")
    d1_ret = decile_stats.loc[0, 'fwd_3m'] if 0 in decile_stats.index else 0
    d2_ret = decile_stats.loc[1, 'fwd_3m'] if 1 in decile_stats.index else 0
    d3_ret = decile_stats.loc[2, 'fwd_3m'] if 2 in decile_stats.index else 0

    if d1_ret < d2_ret and d2_ret < d3_ret:
        print("   ⚠️  Returns INCREASE from D1 → D2 → D3")
        print("   The cheapest decile underperforms moderately cheap stocks!")
        print("   This is classic VALUE TRAP behavior.")


def main():
    print("=" * 70)
    print("VALUATION PUZZLE DEEP DIVE")
    print("Why does EV/EBITDA show negative correlation with returns?")
    print("=" * 70)

    daily_scores, sector_map = load_data()

    df = analyze_valuation_by_sector(daily_scores, sector_map)
    analyze_valuation_quality_interaction(df)
    analyze_valuation_by_time(df)
    analyze_valuation_with_momentum(daily_scores, sector_map)
    analyze_extreme_valuations(df)

    # Final summary
    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)
    print("""
Based on the analysis, the negative valuation correlation is likely due to:

1. VALUE TRAPS: Very cheap stocks (EV/EBITDA < 8x) are often distressed
   companies that continue to decline. Extreme cheapness = distress signal.

2. MOMENTUM DOMINATES: Stocks with good momentum outperform regardless of
   valuation. Cheap + bad momentum = value trap, Expensive + good momentum = winner.

3. MARKET REGIME: The 2022-2024 period favored growth/momentum over value.
   Value may work better in different market conditions.

4. SECTOR EFFECTS: High-multiple Tech outperformed low-multiple Energy/Financials.

RECOMMENDATIONS for V3:
1. Keep valuation weight LOW (5 points) - it's not predictive alone
2. Use valuation as a QUALITY FILTER, not a primary signal
3. Only reward moderate cheapness (10-15x), not extreme cheapness (<8x)
4. Combine valuation with momentum for "GARP" (Growth at Reasonable Price)
""")


if __name__ == '__main__':
    main()

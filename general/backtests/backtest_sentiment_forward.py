#!/usr/bin/env python3
"""
Forward-Looking Sentiment Backtest
===================================
Uses historical sentiment data to test if sentiment at time T
predicts returns from T to T+30/60/90 days.

This is a proper backtest - no look-ahead bias.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')


def load_sentiment_data() -> pd.DataFrame:
    """Load all historical sentiment data."""
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query('''
        SELECT symbol, snapshot_date, sentiment_score, article_count,
               positive_pct, negative_pct
        FROM historical_sentiment
        ORDER BY snapshot_date, symbol
    ''', conn)
    conn.close()
    return df


def get_forward_returns(symbols: list, start_date: str, days: int) -> dict:
    """Get forward returns from start_date to start_date + days."""
    conn = sqlite3.connect(BACKTEST_DB)

    end_date = (datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=days)).strftime('%Y-%m-%d')

    returns = {}
    for symbol in symbols:
        # Get price on start_date (or closest available)
        start_price = conn.execute('''
            SELECT close FROM historical_prices
            WHERE symbol = ? AND date >= ?
            ORDER BY date LIMIT 1
        ''', (symbol, start_date)).fetchone()

        # Get price on end_date (or closest available before)
        end_price = conn.execute('''
            SELECT close FROM historical_prices
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        ''', (symbol, end_date)).fetchone()

        if start_price and end_price and start_price[0] > 0:
            returns[symbol] = ((end_price[0] - start_price[0]) / start_price[0]) * 100

    conn.close()
    return returns


def get_sector_mapping() -> dict:
    """Get symbol to sector mapping."""
    conn = sqlite3.connect(str(PROJECT_ROOT / 'nasdaq_stocks.db'))
    mapping = dict(conn.execute('SELECT symbol, sector FROM stock_consensus').fetchall())
    conn.close()
    return mapping


def run_backtest():
    """Run the forward-looking backtest."""
    print("=" * 70)
    print("FORWARD-LOOKING SENTIMENT BACKTEST")
    print("=" * 70)

    # Load data
    sentiment_df = load_sentiment_data()
    sector_map = get_sector_mapping()

    snapshot_dates = sorted(sentiment_df['snapshot_date'].unique())
    print(f"Sentiment data: {len(snapshot_dates)} monthly snapshots")
    print(f"Date range: {snapshot_dates[0]} to {snapshot_dates[-1]}")
    print(f"Total records: {len(sentiment_df)}")

    # Add sector info
    sentiment_df['sector'] = sentiment_df['symbol'].map(sector_map)

    # Test different forward periods
    periods = [30, 60, 90]

    results = {}

    for days in periods:
        print(f"\n{'=' * 70}")
        print(f"TESTING {days}-DAY FORWARD RETURNS")
        print("=" * 70)

        all_data = []

        for snapshot_date in snapshot_dates:
            # Get sentiment at this date
            snapshot_data = sentiment_df[sentiment_df['snapshot_date'] == snapshot_date].copy()

            # Get forward returns from this date
            symbols = snapshot_data['symbol'].tolist()
            forward_rets = get_forward_returns(symbols, snapshot_date, days)

            snapshot_data['forward_return'] = snapshot_data['symbol'].map(forward_rets)
            snapshot_data = snapshot_data.dropna(subset=['forward_return'])

            if len(snapshot_data) > 0:
                all_data.append(snapshot_data)

        if not all_data:
            print(f"No data available for {days}-day returns")
            continue

        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total observations: {len(combined)}")

        # Individual stock sentiment correlation
        corr_individual = combined['sentiment_score'].corr(combined['forward_return'])
        print(f"\nIndividual Sentiment vs Forward Return: r = {corr_individual:+.4f}")

        # Sector sentiment correlation
        sector_avg = combined.groupby(['snapshot_date', 'sector'])['sentiment_score'].transform('mean')
        corr_sector = sector_avg.corr(combined['forward_return'])
        print(f"Sector Sentiment vs Forward Return:     r = {corr_sector:+.4f}")

        # Quintile analysis
        combined['quintile'] = pd.qcut(
            combined['sentiment_score'],
            q=5,
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
            duplicates='drop'
        )

        print(f"\n{'Quintile':<12} {'Avg Sent':>10} {'Fwd Ret%':>10} {'Obs':>10}")
        print("-" * 45)

        quintile_stats = combined.groupby('quintile', observed=False).agg({
            'sentiment_score': 'mean',
            'forward_return': 'mean',
            'symbol': 'count'
        })

        for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
            if q in quintile_stats.index:
                row = quintile_stats.loc[q]
                print(f"{q:<12} {row['sentiment_score']:>10.1f} {row['forward_return']:>+9.2f}% {int(row['symbol']):>10}")

        # Q5-Q1 spread
        q5_ret = combined[combined['quintile'] == 'Q5 (High)']['forward_return'].mean()
        q1_ret = combined[combined['quintile'] == 'Q1 (Low)']['forward_return'].mean()
        spread = q5_ret - q1_ret

        print("-" * 45)
        print(f"{'Q5-Q1 Spread':<12} {'':>10} {spread:>+9.2f}%")

        # Sector analysis
        print(f"\n--- Sector Sentiment Analysis ---")
        sector_stats = combined.groupby('sector').agg({
            'sentiment_score': 'mean',
            'forward_return': 'mean',
            'symbol': 'count'
        }).rename(columns={'symbol': 'count'})

        sector_stats = sector_stats.sort_values('sentiment_score', ascending=False)

        print(f"{'Sector':<25} {'Avg Sent':>10} {'Fwd Ret%':>10} {'Obs':>8}")
        print("-" * 55)
        for sector, row in sector_stats.iterrows():
            if pd.notna(sector) and row['count'] >= 50:
                print(f"{str(sector)[:25]:<25} {row['sentiment_score']:>10.1f} {row['forward_return']:>+9.2f}% {int(row['count']):>8}")

        # High vs low sentiment sectors
        median_sect = sector_stats['sentiment_score'].median()
        high_sect = sector_stats[sector_stats['sentiment_score'] > median_sect].index.tolist()
        low_sect = sector_stats[sector_stats['sentiment_score'] <= median_sect].index.tolist()

        high_ret = combined[combined['sector'].isin(high_sect)]['forward_return'].mean()
        low_ret = combined[combined['sector'].isin(low_sect)]['forward_return'].mean()
        sector_spread = high_ret - low_ret

        print(f"\nHigh-sentiment sectors avg return: {high_ret:+.2f}%")
        print(f"Low-sentiment sectors avg return:  {low_ret:+.2f}%")
        print(f"Sector sentiment spread:           {sector_spread:+.2f}%")

        results[days] = {
            'corr_individual': corr_individual,
            'corr_sector': corr_sector,
            'quintile_spread': spread,
            'sector_spread': sector_spread,
            'observations': len(combined)
        }

    # Summary
    print("\n" + "=" * 70)
    print("BACKTEST SUMMARY")
    print("=" * 70)

    print(f"\n{'Period':>10} {'Indiv Corr':>12} {'Sect Corr':>12} {'Q Spread':>10} {'Sect Spread':>12}")
    print("-" * 60)

    for days, r in results.items():
        print(f"{days:>7} d {r['corr_individual']:>+12.4f} {r['corr_sector']:>+12.4f} {r['quintile_spread']:>+9.2f}% {r['sector_spread']:>+11.2f}%")

    # Final recommendation
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Find best period
    best_period = max(results.keys(), key=lambda k: results[k]['quintile_spread'])
    best_spread = results[best_period]['quintile_spread']

    if best_spread > 1:
        print(f"""
✅ SENTIMENT IS PREDICTIVE

Best period: {best_period} days (Q5-Q1 spread: {best_spread:+.2f}%)

Recommendation:
- Use sentiment as a confirming indicator
- High-sentiment stocks outperform low-sentiment by {best_spread:.1f}% on average
- Sector sentiment provides additional edge of {results[best_period]['sector_spread']:+.2f}%
""")
    elif best_spread > 0:
        print(f"""
➖ SENTIMENT HAS WEAK PREDICTIVE POWER

Best period: {best_period} days (Q5-Q1 spread: {best_spread:+.2f}%)

The spread is positive but small. Consider:
- Using sector sentiment as a filter only
- Combining with other factors for better signal
""")
    else:
        print(f"""
❌ SENTIMENT IS NOT PREDICTIVE

All periods show negative or zero spreads.
Past sentiment does not reliably predict future returns.

Consider:
- Sentiment may be a contrarian indicator
- Or market has already priced in sentiment
""")


if __name__ == '__main__':
    run_backtest()

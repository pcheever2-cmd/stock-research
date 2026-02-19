#!/usr/bin/env python3
"""
Factor Exploration
==================
Test additional factors for predictive power.

Based on V3 backtest findings:
- momentum_12_1_score: +0.0611 correlation (BEST)
- trend_score: +0.0554 correlation
- fundamentals_score: +0.0514 correlation
- valuation_score: -0.0711 correlation (HURTING model!)
- momentum_score (RSI/ADX): -0.0037 (noise)

Factors to test:
1. Size (market cap) - small caps often outperform
2. Volatility - low vol stocks often outperform
3. 52-week high proximity - momentum continuation
4. Short-term reversal - 1-month contrarian
5. Volume trend - increasing volume bullish?
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME


def analyze_factor(df: pd.DataFrame, factor_col: str, return_col: str = 'fwd_3m',
                   name: str = None, higher_is_better: bool = True) -> dict:
    """Analyze a single factor's predictive power."""
    clean = df.dropna(subset=[factor_col, return_col]).copy()

    if len(clean) < 1000:
        return None

    # Correlation
    corr = clean[factor_col].corr(clean[return_col])

    # Quintile analysis
    try:
        clean['quintile'] = pd.qcut(clean[factor_col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except ValueError:
        clean['quintile'] = pd.cut(clean[factor_col].rank(pct=True),
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quintile_returns = clean.groupby('quintile', observed=False)[return_col].mean()

    q5_ret = quintile_returns.get('Q5', 0)
    q1_ret = quintile_returns.get('Q1', 0)
    spread = q5_ret - q1_ret

    # If lower is better, flip the interpretation
    if not higher_is_better:
        spread = -spread

    return {
        'name': name or factor_col,
        'correlation': corr,
        'spread': spread,
        'q1_return': q1_ret,
        'q5_return': q5_ret,
        'n_obs': len(clean),
    }


def main():
    print("=" * 70)
    print("FACTOR EXPLORATION")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load prices with 3-month forward returns
    print("\nLoading price data...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close, volume
        FROM historical_prices
        ORDER BY symbol, date
    """, conn)

    print(f"  Loaded {len(prices):,} price records")

    # Calculate factors for each observation
    print("\nCalculating factors...")
    all_data = []

    symbols = prices['symbol'].unique()
    # Sample 500 symbols for faster processing
    np.random.seed(42)
    sample_symbols = np.random.choice(symbols, size=min(500, len(symbols)), replace=False)
    print(f"  Sampling {len(sample_symbols)} symbols for faster analysis...")

    for i, symbol in enumerate(sample_symbols):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(sample_symbols)} symbols...", flush=True)

        sym_prices = prices[prices['symbol'] == symbol].sort_values('date').copy()

        if len(sym_prices) < 300:
            continue

        closes = sym_prices['adjusted_close'].values
        volumes = sym_prices['volume'].values
        dates = sym_prices['date'].values

        # Sample monthly observations (every 21 days) for faster processing
        for j in range(252, len(closes) - 63, 21):  # Need 1 year history and 3m forward
            # Forward return
            fwd_3m = ((closes[j + 63] / closes[j]) - 1) * 100

            # Factor 1: 12-1 Momentum (already validated)
            mom_12_1 = ((closes[j - 21] / closes[j - 252]) - 1) * 100 if closes[j - 252] > 0 else np.nan

            # Factor 2: Size (use price * average volume as rough proxy)
            avg_vol = np.mean(volumes[j-21:j])
            market_cap_proxy = closes[j] * avg_vol / 1e6  # In millions

            # Factor 3: Volatility (20-day standard deviation of returns)
            prices_21d = closes[j-21:j+1]
            returns_20d = np.diff(prices_21d) / prices_21d[:-1]
            volatility = np.std(returns_20d) * np.sqrt(252) * 100 if len(returns_20d) > 5 else np.nan

            # Factor 4: 52-week high proximity
            high_52w = np.max(closes[j-252:j])
            pct_from_high = ((closes[j] / high_52w) - 1) * 100 if high_52w > 0 else np.nan

            # Factor 5: Short-term reversal (1-month return)
            ret_1m = ((closes[j] / closes[j-21]) - 1) * 100 if closes[j-21] > 0 else np.nan

            # Factor 6: Volume trend (current vs 3-month average)
            avg_vol_3m = np.mean(volumes[j-63:j])
            avg_vol_1w = np.mean(volumes[j-5:j])
            vol_trend = (avg_vol_1w / avg_vol_3m) if avg_vol_3m > 0 else np.nan

            # Factor 7: Price level (absolute price - often penny stocks underperform)
            price_level = closes[j]

            # Factor 8: 6-month momentum
            mom_6m = ((closes[j-21] / closes[j-126]) - 1) * 100 if closes[j-126] > 0 else np.nan

            # Factor 9: RSI (14-day)
            deltas = np.diff(closes[j-15:j])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))

            all_data.append({
                'symbol': symbol,
                'date': dates[j],
                'fwd_3m': np.clip(fwd_3m, -100, 100),
                'mom_12_1': np.clip(mom_12_1, -100, 200) if not np.isnan(mom_12_1) else np.nan,
                'mom_6m': np.clip(mom_6m, -100, 200) if not np.isnan(mom_6m) else np.nan,
                'ret_1m': np.clip(ret_1m, -50, 50) if not np.isnan(ret_1m) else np.nan,
                'size_proxy': market_cap_proxy,
                'volatility': volatility,
                'pct_from_high': pct_from_high,
                'vol_trend': vol_trend,
                'price_level': price_level,
                'rsi': rsi,
            })

    conn.close()

    df = pd.DataFrame(all_data)
    print(f"\nTotal observations: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Test each factor
    print("\n" + "=" * 70)
    print("FACTOR PREDICTIVE POWER (3-Month Forward Returns)")
    print("=" * 70)

    factors = [
        ('mom_12_1', '12-1 Momentum', True),
        ('mom_6m', '6-Month Momentum', True),
        ('ret_1m', '1-Month Return (Reversal?)', True),
        ('pct_from_high', '% From 52-Week High', True),
        ('size_proxy', 'Size (Market Cap Proxy)', False),  # Small = better?
        ('volatility', 'Volatility', False),  # Low vol = better?
        ('vol_trend', 'Volume Trend', True),
        ('price_level', 'Price Level', True),
        ('rsi', 'RSI (14-day)', True),
    ]

    results = []
    for factor_col, name, higher_is_better in factors:
        result = analyze_factor(df, factor_col, 'fwd_3m', name, higher_is_better)
        if result:
            results.append(result)

    # Sort by absolute spread
    results.sort(key=lambda x: abs(x['spread']), reverse=True)

    print(f"\n{'Factor':<30} {'Corr':>10} {'Spread':>10} {'Q1 Ret':>10} {'Q5 Ret':>10} {'N':>12}")
    print("-" * 85)

    for r in results:
        indicator = "✅" if r['spread'] > 0 and r['correlation'] > 0 else "⚠️" if r['spread'] > 0 else "❌"
        print(f"{r['name']:<30} {r['correlation']:>+9.4f} {r['spread']:>+9.2f}% "
              f"{r['q1_return']:>+9.2f}% {r['q5_return']:>+9.2f}% {r['n_obs']:>11,} {indicator}")

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    strong_factors = [r for r in results if r['spread'] > 2.0 and r['correlation'] > 0.02]
    weak_factors = [r for r in results if r['spread'] < -1.0 or r['correlation'] < -0.02]

    if strong_factors:
        print("\n🌟 STRONG FACTORS (include in scoring):")
        for r in strong_factors:
            print(f"   - {r['name']}: {r['spread']:+.2f}% spread, {r['correlation']:+.4f} corr")

    if weak_factors:
        print("\n⚠️  NEGATIVE/WEAK FACTORS (consider removing or inverting):")
        for r in weak_factors:
            print(f"   - {r['name']}: {r['spread']:+.2f}% spread, {r['correlation']:+.4f} corr")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Explore Additional Factors for V3
=================================
Now that we have a strong V3 baseline (+0.081 correlation), let's see
if we can find additional factors to improve it further.

Current V3 factors:
- 12-1 Momentum: +0.061 corr (strongest)
- 52-Week High: +0.036 corr
- Trend: +0.055 corr
- Fundamentals: +0.051 corr
- Analyst Signal: +0.024 corr
- Valuation: -0.071 corr (reduced weight)

Potential new factors to test:
1. Earnings Revisions - analyst estimate changes
2. Short Interest - contrarian signal?
3. Insider Buying - skin in the game
4. Sector Momentum - rising tide lifts all boats
5. Beta/Volatility - low vol premium
6. Price Target Upside - analyst conviction
7. DCF Upside - intrinsic value gap
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB, DATABASE_NAME


def test_factor(df, factor_col, return_col='fwd_3m', name=None, higher_is_better=True):
    """Test a single factor's predictive power."""
    clean = df.dropna(subset=[factor_col, return_col]).copy()

    if len(clean) < 500:
        return None

    corr = clean[factor_col].corr(clean[return_col])

    try:
        clean['quintile'] = pd.qcut(clean[factor_col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except ValueError:
        clean['quintile'] = pd.cut(clean[factor_col].rank(pct=True),
                                   bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    quintile_returns = clean.groupby('quintile', observed=False)[return_col].mean()

    q5_ret = quintile_returns.get('Q5', 0)
    q1_ret = quintile_returns.get('Q1', 0)
    spread = q5_ret - q1_ret if higher_is_better else q1_ret - q5_ret

    return {
        'name': name or factor_col,
        'correlation': corr if higher_is_better else -corr,
        'spread': spread,
        'q1_return': q1_ret,
        'q5_return': q5_ret,
        'n_obs': len(clean),
        'higher_is_better': higher_is_better,
    }


def main():
    print("=" * 70)
    print("EXPLORING ADDITIONAL FACTORS FOR V3")
    print("=" * 70)

    conn = sqlite3.connect(BACKTEST_DB)

    # Check what data we have available
    print("\nChecking available data sources...")

    # 1. Price Target Data
    print("\n--- PRICE TARGET UPSIDE ---")
    try:
        pt_data = pd.read_sql_query("""
            SELECT symbol, last_year_avg as price_target_avg
            FROM price_target_summary
            WHERE last_year_count >= 3
        """, conn)
        print(f"  Found {len(pt_data)} stocks with price targets")
    except Exception as e:
        print(f"  Error: {e}")
        pt_data = pd.DataFrame()

    # 2. DCF Valuations
    print("\n--- DCF UPSIDE ---")
    try:
        dcf_data = pd.read_sql_query("""
            SELECT symbol, dcf_upside_pct
            FROM dcf_valuations
            WHERE dcf_upside_pct IS NOT NULL
            AND dcf_upside_pct > -90 AND dcf_upside_pct < 500
        """, conn)
        print(f"  Found {len(dcf_data)} stocks with DCF valuations")
    except Exception as e:
        print(f"  Error: {e}")
        dcf_data = pd.DataFrame()

    # 3. Analyst Estimates (for earnings revisions)
    print("\n--- ANALYST ESTIMATES ---")
    try:
        estimates = pd.read_sql_query("""
            SELECT symbol, revenue_avg, ebitda_avg, eps_avg
            FROM analyst_estimates_snapshot
            WHERE fiscal_year = 2025
        """, conn)
        print(f"  Found {len(estimates)} stocks with 2025 estimates")
    except Exception as e:
        print(f"  Error: {e}")
        estimates = pd.DataFrame()

    # 4. Load daily scores with forward returns for testing
    print("\n--- LOADING BACKTEST DATA ---")
    daily = pd.read_sql_query("""
        SELECT symbol, date, close, ev_ebitda, rev_growth, eps_growth,
               lt_score, value_score
        FROM backtest_daily_scores
        WHERE date >= '2022-01-01'
        ORDER BY symbol, date
    """, conn)
    print(f"  Loaded {len(daily):,} daily score records")

    # Get sector mapping
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute("SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    # Calculate forward returns and additional factors
    print("\n--- CALCULATING FACTORS ---")

    np.random.seed(42)
    symbols = daily['symbol'].unique()
    sample_symbols = np.random.choice(symbols, size=min(500, len(symbols)), replace=False)

    all_data = []

    for symbol in sample_symbols:
        sym_data = daily[daily['symbol'] == symbol].sort_values('date')

        if len(sym_data) < 300:
            continue

        sector = sector_map.get(symbol)
        closes = sym_data['close'].values
        dates = sym_data['date'].values
        ev_ebitda = sym_data['ev_ebitda'].values
        rev_growth = sym_data['rev_growth'].values

        # Get additional data for this symbol
        pt_row = pt_data[pt_data['symbol'] == symbol] if not pt_data.empty else pd.DataFrame()
        dcf_row = dcf_data[dcf_data['symbol'] == symbol] if not dcf_data.empty else pd.DataFrame()

        for j in range(252, len(closes) - 63, 21):
            if closes[j] <= 0:
                continue

            fwd_3m = ((closes[j + 63] / closes[j]) - 1) * 100

            # 12-1 Momentum (baseline)
            mom_12_1 = ((closes[j-21] / closes[j-252]) - 1) * 100 if closes[j-252] > 0 else np.nan

            # Price Target Upside
            pt_upside = None
            if not pt_row.empty:
                pt_avg = pt_row.iloc[0]['price_target_avg']
                if pt_avg and pt_avg > 0:
                    pt_upside = ((pt_avg / closes[j]) - 1) * 100

            # DCF Upside
            dcf_upside = dcf_row.iloc[0]['dcf_upside_pct'] if not dcf_row.empty else None

            # Revenue Growth (already have this)
            rev_g = rev_growth[j] if not pd.isna(rev_growth[j]) else None

            # EV/EBITDA (for testing non-linear relationship)
            ev = ev_ebitda[j] if not pd.isna(ev_ebitda[j]) and ev_ebitda[j] > 0 else None

            # Volatility (20-day)
            if j >= 21:
                prices_21d = closes[j-20:j+1]
                returns = np.diff(prices_21d) / prices_21d[:-1]
                volatility = np.std(returns) * np.sqrt(252) * 100
            else:
                volatility = None

            # 52-week high
            high_52w = np.max(closes[max(0,j-252):j])
            pct_from_high = ((closes[j] / high_52w) - 1) * 100 if high_52w > 0 else None

            # 6-month momentum
            mom_6m = ((closes[j-21] / closes[j-126]) - 1) * 100 if j >= 126 and closes[j-126] > 0 else np.nan

            # 3-month momentum
            mom_3m = ((closes[j-21] / closes[j-63]) - 1) * 100 if j >= 63 and closes[j-63] > 0 else np.nan

            # Sector
            sector_val = sector

            all_data.append({
                'symbol': symbol,
                'date': dates[j],
                'sector': sector_val,
                'fwd_3m': np.clip(fwd_3m, -100, 100),
                'mom_12_1': np.clip(mom_12_1, -80, 200) if not np.isnan(mom_12_1) else None,
                'mom_6m': np.clip(mom_6m, -80, 200) if not np.isnan(mom_6m) else None,
                'mom_3m': np.clip(mom_3m, -80, 200) if not np.isnan(mom_3m) else None,
                'pct_from_high': pct_from_high,
                'volatility': volatility,
                'ev_ebitda': ev,
                'rev_growth': rev_g,
                'pt_upside': np.clip(pt_upside, -50, 200) if pt_upside else None,
                'dcf_upside': np.clip(dcf_upside, -50, 200) if dcf_upside else None,
            })

    conn.close()

    df = pd.DataFrame(all_data)
    print(f"\nTotal observations: {len(df):,}")

    # Test each factor
    print("\n" + "=" * 70)
    print("FACTOR PREDICTIVE POWER (3-Month Forward Returns)")
    print("=" * 70)

    factors_to_test = [
        ('mom_12_1', '12-1 Momentum (baseline)', True),
        ('mom_6m', '6-Month Momentum', True),
        ('mom_3m', '3-Month Momentum', True),
        ('pct_from_high', '52-Week High Proximity', True),
        ('volatility', 'Volatility (low=better)', False),
        ('rev_growth', 'Revenue Growth', True),
        ('pt_upside', 'Price Target Upside', True),
        ('dcf_upside', 'DCF Upside', True),
    ]

    results = []
    for col, name, higher_is_better in factors_to_test:
        result = test_factor(df, col, 'fwd_3m', name, higher_is_better)
        if result:
            results.append(result)

    results.sort(key=lambda x: x['correlation'], reverse=True)

    print(f"\n{'Factor':<30} {'Corr':>10} {'Spread':>10} {'Q1 Ret':>10} {'Q5 Ret':>10} {'N':>10}")
    print("-" * 85)

    for r in results:
        indicator = "✅" if r['correlation'] > 0.02 and r['spread'] > 1.0 else "⚠️" if r['correlation'] > 0 else "❌"
        print(f"{r['name']:<30} {r['correlation']:>+9.4f} {r['spread']:>+9.2f}% "
              f"{r['q1_return']:>+9.2f}% {r['q5_return']:>+9.2f}% {r['n_obs']:>10,} {indicator}")

    # Test sector momentum
    print("\n" + "=" * 70)
    print("SECTOR MOMENTUM TEST")
    print("=" * 70)

    # Calculate sector average momentum
    sector_mom = df.groupby(['date', 'sector'])['mom_12_1'].mean().reset_index()
    sector_mom.columns = ['date', 'sector', 'sector_mom']
    df_with_sector = df.merge(sector_mom, on=['date', 'sector'], how='left')

    result = test_factor(df_with_sector, 'sector_mom', 'fwd_3m', 'Sector 12-1 Momentum', True)
    if result:
        print(f"\nSector Momentum: corr={result['correlation']:+.4f}, spread={result['spread']:+.2f}%")

    # Combination test: momentum + low volatility
    print("\n" + "=" * 70)
    print("COMBINATION TESTS")
    print("=" * 70)

    df_clean = df.dropna(subset=['mom_12_1', 'volatility', 'fwd_3m'])
    median_mom = df_clean['mom_12_1'].median()
    median_vol = df_clean['volatility'].median()

    df_clean['high_mom'] = df_clean['mom_12_1'] > median_mom
    df_clean['low_vol'] = df_clean['volatility'] < median_vol

    categories = [
        ('High Mom + Low Vol', (df_clean['high_mom']) & (df_clean['low_vol'])),
        ('High Mom + High Vol', (df_clean['high_mom']) & (~df_clean['low_vol'])),
        ('Low Mom + Low Vol', (~df_clean['high_mom']) & (df_clean['low_vol'])),
        ('Low Mom + High Vol', (~df_clean['high_mom']) & (~df_clean['low_vol'])),
    ]

    print(f"\n{'Category':<25} {'Avg Return':>12} {'N':>10}")
    print("-" * 50)

    for name, mask in categories:
        subset = df_clean[mask]
        avg_ret = subset['fwd_3m'].mean()
        print(f"{name:<25} {avg_ret:>+11.2f}% {len(subset):>10,}")

    # Summary
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    strong_new = [r for r in results if r['correlation'] > 0.03 and r['name'] not in ['12-1 Momentum (baseline)', '52-Week High Proximity']]

    if strong_new:
        print("\n🌟 PROMISING NEW FACTORS to add to V3:")
        for r in strong_new:
            print(f"   - {r['name']}: corr={r['correlation']:+.4f}, spread={r['spread']:+.2f}%")
    else:
        print("\n⚠️  No strong new factors found beyond what's already in V3.")
        print("   Current V3 factors capture most of the predictive signal.")


if __name__ == '__main__':
    main()

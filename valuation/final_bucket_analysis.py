#!/usr/bin/env python3
"""
Final Bucket Analysis
======================
Layers DCF valuation + earnings surprise + all prior analysis to determine
the 4 best purchase buckets with 1Y and 2Y holdout validation.

Sections:
1. DCF Valuation Impact — Does DCF upside predict forward returns?
2. Earnings Surprise Impact — Does beating estimates predict returns?
3. Combined Signal Analysis — Layer DCF + surprise + elite combos
4. Four Best Buckets — The final 4 actionable purchase categories
5. 2-Year Performance Analysis — Extended holdout validation
6. Current Actionable Stocks — What to buy right now from each bucket

Runs against backtest.db + nasdaq_stocks.db — no new data collection needed.
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# ==================== DATA LOADING ====================

def load_signals_with_context():
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("""
        SELECT
            s.symbol, s.date, s.signal_type,
            s.lt_score, s.value_score, s.value_score_v2, s.close_price,
            s.return_1w, s.return_1m, s.return_3m, s.return_6m, s.return_1y,
            d.rsi, d.adx, d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth,
            d.sma50, d.sma200, d.close,
            d.trend_score, d.fundamentals_score, d.valuation_score,
            d.momentum_score, d.market_risk_score
        FROM backtest_signals s
        LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
        WHERE s.return_3m IS NOT NULL
    """, conn)
    conn.close()
    df['market_bullish'] = (df['market_risk_score'] == 10).astype(int)
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['winner_3m'] = (df['return_3m'] > 0).astype(int)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    return df


def load_supplementary(df):
    conn = sqlite3.connect(BACKTEST_DB)
    mcaps = pd.read_sql_query("""
        SELECT symbol, market_cap FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol HAVING date = MAX(date)
    """, conn).set_index('symbol')['market_cap']

    cf = pd.read_sql_query("""
        SELECT symbol, date, free_cash_flow FROM historical_cash_flows
        WHERE free_cash_flow IS NOT NULL ORDER BY symbol, date
    """, conn)
    if not cf.empty:
        def get_yoy(g):
            if len(g) < 5:
                return pd.Series({'fcf_latest': g.iloc[-1]['free_cash_flow'], 'fcf_yago': None})
            return pd.Series({'fcf_latest': g.iloc[-1]['free_cash_flow'],
                              'fcf_yago': g.iloc[-5]['free_cash_flow']})
        fcf = cf.groupby('symbol').apply(get_yoy)
        fcf_positive = ((fcf['fcf_latest'] > 0) & (fcf['fcf_yago'].notna()) &
                        (fcf['fcf_latest'] > fcf['fcf_yago'])).astype(int)
    else:
        fcf_positive = pd.Series(dtype=int)

    acov = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot GROUP BY symbol
    """, conn).set_index('symbol')['analyst_count']
    conn.close()

    conn2 = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query(
        "SELECT symbol, sector, industry FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    df['market_cap'] = df['symbol'].map(mcaps)
    df['fcf_positive'] = df['symbol'].map(fcf_positive)
    df['analyst_count'] = df['symbol'].map(acov)
    df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
    df['industry'] = df['symbol'].map(sectors['industry']) if 'industry' in sectors.columns else None
    return df


def enrich_dcf(df):
    """Add DCF upside from dcf_valuations table."""
    conn = sqlite3.connect(BACKTEST_DB)
    try:
        dcf = pd.read_sql_query("""
            SELECT symbol, dcf_value, stock_price, dcf_upside_pct
            FROM dcf_valuations WHERE dcf_upside_pct IS NOT NULL
        """, conn)
        conn.close()
        dcf_map = dcf.set_index('symbol')['dcf_upside_pct']
        df['dcf_upside'] = df['symbol'].map(dcf_map)
        return df
    except Exception:
        conn.close()
        df['dcf_upside'] = np.nan
        return df


def enrich_earnings_surprise(df):
    """Add earnings surprise (actual vs consensus EPS for matching fiscal years)."""
    conn = sqlite3.connect(BACKTEST_DB)
    income = pd.read_sql_query("""
        SELECT symbol, fiscal_year, SUM(eps_diluted) as actual_eps, COUNT(*) as quarters
        FROM historical_income_statements
        WHERE eps_diluted IS NOT NULL AND fiscal_year IS NOT NULL
        GROUP BY symbol, fiscal_year
        HAVING quarters >= 4
    """, conn)
    estimates = pd.read_sql_query("""
        SELECT symbol, fiscal_year, eps_avg
        FROM analyst_estimates_snapshot
        WHERE eps_avg IS NOT NULL AND fiscal_year IS NOT NULL
    """, conn)
    conn.close()

    if income.empty or estimates.empty:
        df['earnings_surprise_pct'] = np.nan
        return df

    merged = income.merge(estimates, on=['symbol', 'fiscal_year'], how='inner')
    if merged.empty:
        df['earnings_surprise_pct'] = np.nan
        return df

    merged = merged.sort_values('fiscal_year').groupby('symbol').last().reset_index()
    merged['earnings_surprise_pct'] = np.where(
        merged['eps_avg'].abs() > 0.01,
        (merged['actual_eps'] - merged['eps_avg']) / merged['eps_avg'].abs() * 100,
        np.nan
    )
    surprise_map = merged.set_index('symbol')['earnings_surprise_pct'].clip(-200, 500)
    df['earnings_surprise_pct'] = df['symbol'].map(surprise_map)
    return df


def enrich_ensemble_voting(df):
    vote_counts = df.groupby(['symbol', 'date']).size().reset_index(name='concurrent_signals')
    df = df.merge(vote_counts, on=['symbol', 'date'], how='left')
    return df


# ==================== HELPERS ====================

def stats_row(name, sub, min_count=30):
    n = len(sub)
    if n < min_count:
        return None
    ret = sub['return_3m'].clip(-90, 500)
    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    win_3m = (ret > 0).mean() * 100
    avg_3m = ret.mean()
    med_3m = ret.median()
    avg_6m = sub['return_6m'].clip(-90, 1000).mean() if sub['return_6m'].notna().sum() > 10 else np.nan
    avg_1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    pf = avg_win / avg_loss if avg_loss > 0 else np.inf
    return {
        'name': name, 'count': n, 'win_3m': win_3m,
        'avg_3m': avg_3m, 'med_3m': med_3m,
        'avg_6m': avg_6m, 'avg_1y': avg_1y,
        'profit_factor': pf,
    }


def bootstrap_ci(returns, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    boot_wins = []
    for _ in range(n_boot):
        s = rng.choice(returns, len(returns), replace=True)
        boot_wins.append((s > 0).mean() * 100)
    return {
        'win_lo': np.percentile(boot_wins, 2.5),
        'win_hi': np.percentile(boot_wins, 97.5),
    }


# ==================== SECTION 1: DCF VALUATION IMPACT ====================

def run_dcf_analysis(df):
    print("\n" + "#" * 115)
    print("  SECTION 1: DCF VALUATION IMPACT ON FORWARD RETURNS")
    print("#" * 115)

    dcf_coverage = df['dcf_upside'].notna().sum()
    print(f"\n  DCF data available for {dcf_coverage:,}/{len(df):,} signals "
          f"({dcf_coverage/len(df)*100:.1f}%)")

    if dcf_coverage < 1000:
        print("  Insufficient DCF data. Skipping.")
        return

    df_dcf = df[df['dcf_upside'].notna()].copy()

    # --- 1A: DCF upside quintile analysis ---
    print(f"\n  --- 1A: Performance by DCF Upside Quintile ---")
    print(f"  {'Quintile':<18s} {'Range':<24s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Avg6M':>8s} {'Avg1Y':>8s} {'PF':>5s}")
    print(f"  {'-'*90}")

    try:
        quintiles = pd.qcut(df_dcf['dcf_upside'], q=5,
                            labels=['Q1(overvalued)', 'Q2', 'Q3', 'Q4', 'Q5(undervalued)'],
                            duplicates='drop')
        for q_label in ['Q1(overvalued)', 'Q2', 'Q3', 'Q4', 'Q5(undervalued)']:
            sub = df_dcf[quintiles == q_label]
            if len(sub) < 30:
                continue
            ret = sub['return_3m'].clip(-90, 500)
            w = (ret > 0).mean() * 100
            a3 = ret.mean()
            a6 = sub['return_6m'].clip(-90, 1000).mean() if sub['return_6m'].notna().sum() > 10 else np.nan
            a1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
            wins = ret[ret > 0]
            losses = ret[ret <= 0]
            pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
            rng = quintiles[quintiles == q_label].cat.categories
            a6s = f"{a6:+.1f}%" if pd.notna(a6) else "    N/A"
            a1ys = f"{a1y:+.1f}%" if pd.notna(a1y) else "    N/A"
            print(f"  {q_label:<18s} {'':24s} {len(sub):>6,} {w:>6.1f}% {a3:>+7.1f}% {a6s:>7s} {a1ys:>7s} {pf:>4.2f}")
    except Exception as e:
        print(f"  Error in quintile analysis: {e}")

    # --- 1B: DCF as binary filter on signals ---
    print(f"\n\n  --- 1B: DCF Undervalued Filter (DCF upside > 0%) ---")
    print(f"  {'Signal':<35s} │ {'No Filter':^22s} │ {'DCF>0% Only':^22s} │ {'Δ Win%':>7s}")
    print(f"  {'':<35s} │ {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} │ {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} │")
    print(f"  {'-'*98}")

    for sig in ['Tier 1', 'Tier 3', 'Regime Buy Bear', 'Buy_A', 'Regime Buy Bull']:
        base = df_dcf[df_dcf['signal_type'] == sig]
        dcf_pos = base[base['dcf_upside'] > 0]
        if len(base) < 20:
            continue
        bw = (base['return_3m'] > 0).mean() * 100
        ba = base['return_3m'].clip(-90, 500).mean()
        if len(dcf_pos) >= 10:
            dw = (dcf_pos['return_3m'] > 0).mean() * 100
            da = dcf_pos['return_3m'].clip(-90, 500).mean()
            delta = dw - bw
            print(f"  {sig:<35s} │ {len(base):>5,} {bw:>5.1f}% {ba:>+6.1f}% │ "
                  f"{len(dcf_pos):>5,} {dw:>5.1f}% {da:>+6.1f}% │ {delta:>+6.1f}%")
        else:
            print(f"  {sig:<35s} │ {len(base):>5,} {bw:>5.1f}% {ba:>+6.1f}% │ "
                  f"{'N/A':>20s} │    N/A")

    # --- 1C: DCF + V1 Elite Combos ---
    print(f"\n\n  --- 1C: DCF Layering on Elite Combos ---")
    print(f"  {'Combo':<42s} │ {'Base Win%':>9s} {'N':>6s} │ {'DCF>0 Win%':>10s} {'N':>5s} {'Δ':>6s}")
    print(f"  {'-'*90}")

    combos = [
        ("V2≥60+EV5-15+RevG10-30",
         lambda d: d[(d['value_score_v2'] >= 60) & (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 15) &
                     (d['rev_growth'] >= 10) & (d['rev_growth'] <= 30)]),
        ("Bear+RSI<40+Fund≥15+V2≥40",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                     (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40)]),
        ("Cap>$2B+Analysts≥6+FCF+LT≥45",
         lambda d: d[(d['market_cap'] > 2e9) & (d['analyst_count'] >= 6) &
                     (d['fcf_positive'] == 1) & (d['lt_score'] >= 45)]),
        ("LT≥55+Fund≥18+RSI35-55+Analysts≥8",
         lambda d: d[(d['lt_score'] >= 55) & (d['fundamentals_score'] >= 18) &
                     (d['rsi'] >= 35) & (d['rsi'] <= 55) & (d['analyst_count'] >= 8)]),
        ("EV12-27+RSI<43+EPSG35+EBITDAG33",
         lambda d: d[(d['ev_ebitda_clean'] >= 12) & (d['ev_ebitda_clean'] <= 27) &
                     (d['rsi'] < 43) & (d['eps_growth'] >= 35) & (d['ebitda_growth'] >= 33)]),
    ]

    for name, filt in combos:
        base = filt(df_dcf)
        dcf_pos = base[base['dcf_upside'] > 0]
        if len(base) < 20:
            continue
        bw = (base['return_3m'] > 0).mean() * 100
        if len(dcf_pos) >= 10:
            dw = (dcf_pos['return_3m'] > 0).mean() * 100
            delta = dw - bw
            print(f"  {name:<42s} │ {bw:>8.1f}% {len(base):>5,} │ {dw:>9.1f}% {len(dcf_pos):>4,} {delta:>+5.1f}%")
        else:
            print(f"  {name:<42s} │ {bw:>8.1f}% {len(base):>5,} │ {'insufficient':>15s}")


# ==================== SECTION 2: EARNINGS SURPRISE IMPACT ====================

def run_earnings_surprise_analysis(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 2: EARNINGS SURPRISE IMPACT ON FORWARD RETURNS")
    print("#" * 115)

    es_coverage = df['earnings_surprise_pct'].notna().sum()
    print(f"\n  Earnings surprise data: {es_coverage:,}/{len(df):,} signals "
          f"({es_coverage/len(df)*100:.1f}%)")

    if es_coverage < 1000:
        print("  Insufficient earnings surprise data. Skipping.")
        return

    df_es = df[df['earnings_surprise_pct'].notna()].copy()
    df_es['surprise_cat'] = pd.cut(df_es['earnings_surprise_pct'],
                                    bins=[-np.inf, -10, 0, 10, np.inf],
                                    labels=['Big Miss', 'Miss', 'Beat', 'Big Beat'])

    # --- 2A: Surprise category performance ---
    print(f"\n  --- 2A: Performance by Earnings Surprise Category ---")
    print(f"  {'Category':<14s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Avg6M':>8s} {'Avg1Y':>8s} {'PF':>5s}")
    print(f"  {'-'*60}")

    for cat in ['Big Miss', 'Miss', 'Beat', 'Big Beat']:
        sub = df_es[df_es['surprise_cat'] == cat]
        if len(sub) < 30:
            continue
        ret = sub['return_3m'].clip(-90, 500)
        w = (ret > 0).mean() * 100
        a3 = ret.mean()
        a6 = sub['return_6m'].clip(-90, 1000).mean() if sub['return_6m'].notna().sum() > 10 else np.nan
        a1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
        a6s = f"{a6:+.1f}%" if pd.notna(a6) else "    N/A"
        a1ys = f"{a1y:+.1f}%" if pd.notna(a1y) else "    N/A"
        print(f"  {cat:<14s} {len(sub):>6,} {w:>6.1f}% {a3:>+7.1f}% {a6s:>7s} {a1ys:>7s} {pf:>4.2f}")

    # --- 2B: Surprise as filter on elite combos ---
    print(f"\n\n  --- 2B: Earnings Beat Filter on Elite Combos ---")
    print(f"  {'Combo':<42s} │ {'Base Win%':>9s} {'N':>6s} │ {'Beat Win%':>9s} {'N':>5s} {'Δ':>6s}")
    print(f"  {'-'*85}")

    combos = [
        ("V2≥60+EV5-15+RevG10-30",
         lambda d: d[(d['value_score_v2'] >= 60) & (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 15) &
                     (d['rev_growth'] >= 10) & (d['rev_growth'] <= 30)]),
        ("Bear+RSI<40+Fund≥15+V2≥40",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                     (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40)]),
        ("Cap>$2B+Analysts≥6+FCF+LT≥45",
         lambda d: d[(d['market_cap'] > 2e9) & (d['analyst_count'] >= 6) &
                     (d['fcf_positive'] == 1) & (d['lt_score'] >= 45)]),
        ("LT≥55+Fund≥18+RSI35-55",
         lambda d: d[(d['lt_score'] >= 55) & (d['fundamentals_score'] >= 18) &
                     (d['rsi'] >= 35) & (d['rsi'] <= 55)]),
    ]

    for name, filt in combos:
        base = filt(df_es)
        beat = base[base['earnings_surprise_pct'] > 0]
        if len(base) < 20:
            continue
        bw = (base['return_3m'] > 0).mean() * 100
        if len(beat) >= 10:
            dw = (beat['return_3m'] > 0).mean() * 100
            delta = dw - bw
            print(f"  {name:<42s} │ {bw:>8.1f}% {len(base):>5,} │ {dw:>8.1f}% {len(beat):>4,} {delta:>+5.1f}%")
        else:
            print(f"  {name:<42s} │ {bw:>8.1f}% {len(base):>5,} │ {'insufficient':>15s}")


# ==================== SECTION 3: COMBINED SIGNAL ANALYSIS ====================

def run_combined_analysis(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 3: COMBINED SIGNAL ANALYSIS (DCF + SURPRISE + SCORES)")
    print("#" * 115)

    # Triple-layer filters
    has_dcf = df['dcf_upside'].notna()
    has_es = df['earnings_surprise_pct'].notna()
    both = has_dcf & has_es
    print(f"\n  Signals with DCF: {has_dcf.sum():,} | Earnings surprise: {has_es.sum():,} | Both: {both.sum():,}")

    df_both = df[both].copy()
    if len(df_both) < 500:
        print("  Insufficient combined data. Running with available signals.")
        df_both = df[has_dcf | has_es].copy()

    # Test combined filters
    print(f"\n  --- Combined Filter Combinations ---")
    print(f"  {'Filter':<55s} {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} {'Avg6M':>7s} {'Avg1Y':>7s} {'PF':>5s}")
    print(f"  {'-'*95}")

    filters = [
        ("Baseline (all signals)", df),
        ("DCF>0% (undervalued by DCF)", df[df['dcf_upside'] > 0]),
        ("DCF>20% (significantly undervalued)", df[df['dcf_upside'] > 20]),
        ("EarningsBeat (surprise>0%)", df[df['earnings_surprise_pct'] > 0]),
        ("EarningsBigBeat (surprise>10%)", df[df['earnings_surprise_pct'] > 10]),
        ("DCF>0% + EarningsBeat", df[(df['dcf_upside'] > 0) & (df['earnings_surprise_pct'] > 0)]),
        ("DCF>0% + V2≥50", df[(df['dcf_upside'] > 0) & (df['value_score_v2'] >= 50)]),
        ("DCF>0% + Fund≥15", df[(df['dcf_upside'] > 0) & (df['fundamentals_score'] >= 15)]),
        ("DCF>0% + V2≥50 + Fund≥15", df[(df['dcf_upside'] > 0) & (df['value_score_v2'] >= 50) &
                                          (df['fundamentals_score'] >= 15)]),
        ("DCF>0% + V2≥50 + EarningsBeat", df[(df['dcf_upside'] > 0) & (df['value_score_v2'] >= 50) &
                                               (df['earnings_surprise_pct'] > 0)]),
        ("DCF>20% + V2≥55 + Fund≥18", df[(df['dcf_upside'] > 20) & (df['value_score_v2'] >= 55) &
                                           (df['fundamentals_score'] >= 18)]),
        ("DCF>0% + Bear + RSI<40", df[(df['dcf_upside'] > 0) & (df['market_bearish'] == 1) &
                                       (df['rsi'] < 40)]),
        ("DCF>0% + EV5-15 + RevG>10 + V2≥55", df[(df['dcf_upside'] > 0) &
                                                    (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 15) &
                                                    (df['rev_growth'] > 10) & (df['value_score_v2'] >= 55)]),
    ]

    for name, sub in filters:
        sub = sub[sub['return_3m'].notna()]
        if len(sub) < 30:
            continue
        ret = sub['return_3m'].clip(-90, 500)
        w = (ret > 0).mean() * 100
        a3 = ret.mean()
        a6 = sub['return_6m'].clip(-90, 1000).mean() if sub['return_6m'].notna().sum() > 10 else np.nan
        a1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
        a6s = f"{a6:+.1f}%" if pd.notna(a6) else "   N/A"
        a1ys = f"{a1y:+.1f}%" if pd.notna(a1y) else "   N/A"
        print(f"  {name:<55s} {len(sub):>5,} {w:>5.1f}% {a3:>+6.1f}% {a6s:>6s} {a1ys:>6s} {pf:>4.2f}")


# ==================== SECTION 4: THE FOUR BEST BUCKETS ====================

def run_four_buckets(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 4: THE FOUR BEST PURCHASE BUCKETS")
    print("#" * 115)
    print(f"\n  Selection criteria: Highest win rate + walk-forward stability + reasonable N + risk-adjusted returns")

    # Define the 4 candidate buckets based on all analysis
    buckets = {
        "BUCKET 1: Quality Growth Compounder": {
            'desc': "V2≥55 + Fund≥18 + EV5-20 + RevG>10 + Cap>$2B + Analysts≥6",
            'filter': lambda d: d[(d['value_score_v2'] >= 55) & (d['fundamentals_score'] >= 18) &
                                  (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20) &
                                  (d['rev_growth'] > 10) & (d['market_cap'] > 2e9) &
                                  (d['analyst_count'] >= 6)],
        },
        "BUCKET 2: Bear Market Dip Buy": {
            'desc': "Bear regime + RSI<40 + Fund≥15 + V2≥40 + Cap>$1B",
            'filter': lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                                  (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40) &
                                  (d['market_cap'] > 1e9)],
        },
        "BUCKET 3: High-Growth Momentum": {
            'desc': "EPSG≥35 + EBITDAG≥33 + EV12-27 + RSI<43 (auto-gen top combo)",
            'filter': lambda d: d[(d['eps_growth'] >= 35) & (d['ebitda_growth'] >= 33) &
                                  (d['ev_ebitda_clean'] >= 12) & (d['ev_ebitda_clean'] <= 27) &
                                  (d['rsi'] < 43)],
        },
        "BUCKET 4: DCF Value + Quality": {
            'desc': "DCF upside>0% + V2≥50 + Fund≥15 + Cap>$1B",
            'filter': lambda d: d[(d['dcf_upside'] > 0) & (d['value_score_v2'] >= 50) &
                                  (d['fundamentals_score'] >= 15) & (d['market_cap'] > 1e9)],
        },
    }

    # Walk-forward windows
    windows = [
        (2021, 2022, 2023, 2023),
        (2021, 2023, 2024, 2024),
        (2022, 2024, 2025, 2026),
    ]

    for bucket_name, config in buckets.items():
        print(f"\n\n  {'='*110}")
        print(f"  {bucket_name}")
        print(f"  {'='*110}")
        print(f"  Criteria: {config['desc']}")

        sub = config['filter'](df)
        sub_valid = sub[sub['return_3m'].notna()]
        if len(sub_valid) < 20:
            print(f"  Insufficient signals: {len(sub_valid)}")
            continue

        ret = sub_valid['return_3m'].clip(-90, 500)
        wins = ret[ret > 0]
        losses = ret[ret <= 0]

        # Core stats
        win_rate = (ret > 0).mean() * 100
        avg_3m = ret.mean()
        med_3m = ret.median()
        avg_6m = sub_valid['return_6m'].clip(-90, 1000).mean() if sub_valid['return_6m'].notna().sum() > 10 else np.nan
        avg_1y = sub_valid['return_1y'].clip(-90, 1000).mean() if sub_valid['return_1y'].notna().sum() > 10 else np.nan
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf

        print(f"\n  Performance Summary:")
        print(f"    Signals: {len(sub_valid):,}")
        print(f"    Win Rate (3M): {win_rate:.1f}%")
        print(f"    Avg Return (3M): {avg_3m:+.1f}%")
        print(f"    Median Return (3M): {med_3m:+.1f}%")
        if pd.notna(avg_6m):
            print(f"    Avg Return (6M): {avg_6m:+.1f}%")
        if pd.notna(avg_1y):
            print(f"    Avg Return (1Y): {avg_1y:+.1f}%")
        print(f"    Profit Factor: {pf:.2f}")
        print(f"    Avg Win: {wins.mean():+.1f}% | Avg Loss: {losses.mean():+.1f}%")

        # Walk-forward stability
        print(f"\n  Walk-Forward Stability:")
        all_pass = True
        for ts, te, vs, ve in windows:
            test = sub_valid[(sub_valid['year'] >= vs) & (sub_valid['year'] <= ve)]
            if len(test) >= 5:
                tw = (test['return_3m'] > 0).mean() * 100
                ta = test['return_3m'].clip(-90, 500).mean()
                status = "PASS" if tw >= 50 else "FAIL"
                if tw < 50:
                    all_pass = False
                print(f"    Test {vs}-{ve}: N={len(test):>4,} | Win%={tw:>5.1f}% | Avg={ta:>+6.1f}% | {status}")
            else:
                print(f"    Test {vs}-{ve}: N={len(test):>4,} | Insufficient data")

        # Bootstrap CI
        ci = bootstrap_ci(ret.values)
        print(f"\n  Bootstrap 95% CI: [{ci['win_lo']:.1f}%, {ci['win_hi']:.1f}%]")

        # Sector breakdown
        top_sectors = sub_valid['sector'].value_counts().head(5)
        if not top_sectors.empty:
            print(f"\n  Top Sectors:")
            for sector, count in top_sectors.items():
                sec_sub = sub_valid[sub_valid['sector'] == sector]
                sw = (sec_sub['return_3m'] > 0).mean() * 100
                print(f"    {sector:<25s}: N={count:>4,} | Win%={sw:.1f}%")

        # Yearly performance
        print(f"\n  Yearly Performance:")
        for year in sorted(sub_valid['year'].unique()):
            yr = sub_valid[sub_valid['year'] == year]
            if len(yr) < 5:
                continue
            yw = (yr['return_3m'] > 0).mean() * 100
            ya = yr['return_3m'].clip(-90, 500).mean()
            print(f"    {year}: N={len(yr):>4,} | Win%={yw:>5.1f}% | Avg={ya:>+6.1f}%")


# ==================== SECTION 5: 2-YEAR PERFORMANCE ANALYSIS ====================

def run_2year_analysis(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 5: EXTENDED HOLDOUT — 1Y AND 2Y FORWARD RETURNS")
    print("#" * 115)

    # Check for 2-year return data availability
    # We can approximate 2Y return from signals that are old enough
    # return_1y exists in data, but 2Y requires price data
    conn = sqlite3.connect(BACKTEST_DB)
    # Get prices for 2-year return calculation
    prices = pd.read_sql_query("""
        SELECT symbol, date, close FROM historical_prices
        WHERE date >= '2019-01-01' ORDER BY symbol, date
    """, conn)
    conn.close()

    prices['date'] = pd.to_datetime(prices['date'])
    price_dict = {}
    for sym, grp in prices.groupby('symbol'):
        price_dict[sym] = dict(zip(grp['date'], grp['close']))

    def get_price(sym, dt, window=5):
        if sym not in price_dict:
            return None
        for off in range(window + 1):
            d = dt + pd.Timedelta(days=off)
            if d in price_dict[sym]:
                return price_dict[sym][d]
            if off > 0:
                d = dt - pd.Timedelta(days=off)
                if d in price_dict[sym]:
                    return price_dict[sym][d]
        return None

    # Compute 2-year returns for signals old enough (before 2024)
    print(f"\n  Computing 2-year forward returns for signals before 2024...")
    df_2y = df[df['year'] <= 2023].copy()
    returns_2y = []
    count = 0
    for idx, row in df_2y.iterrows():
        count += 1
        if count % 20000 == 0:
            print(f"    Processing {count:,}/{len(df_2y):,}...")
        p0 = row['close_price'] or row['close']
        if p0 is None or p0 <= 0:
            returns_2y.append(np.nan)
            continue
        p2y = get_price(row['symbol'], row['date_dt'] + pd.Timedelta(days=730))
        if p2y and p0 > 0:
            returns_2y.append((p2y / p0 - 1) * 100)
        else:
            returns_2y.append(np.nan)

    df_2y['return_2y'] = returns_2y
    valid_2y = df_2y['return_2y'].notna().sum()
    print(f"  2-year returns computed: {valid_2y:,}/{len(df_2y):,}")

    # The 4 buckets with 1Y and 2Y returns
    buckets = [
        ("Quality Growth Compounder",
         lambda d: d[(d['value_score_v2'] >= 55) & (d['fundamentals_score'] >= 18) &
                     (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20) &
                     (d['rev_growth'] > 10) & (d['market_cap'] > 2e9) & (d['analyst_count'] >= 6)]),
        ("Bear Market Dip Buy",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                     (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40) &
                     (d['market_cap'] > 1e9)]),
        ("High-Growth Momentum",
         lambda d: d[(d['eps_growth'] >= 35) & (d['ebitda_growth'] >= 33) &
                     (d['ev_ebitda_clean'] >= 12) & (d['ev_ebitda_clean'] <= 27) &
                     (d['rsi'] < 43)]),
        ("DCF Value + Quality",
         lambda d: d[(d['dcf_upside'] > 0) & (d['value_score_v2'] >= 50) &
                     (d['fundamentals_score'] >= 15) & (d['market_cap'] > 1e9)]),
    ]

    # Also include key signal types
    signals = ['Tier 1', 'Tier 3', 'Regime Buy Bear', 'Regime Buy Bull']

    print(f"\n  --- 1Y and 2Y Performance Comparison ---")
    print(f"  {'Bucket/Signal':<38s} │ {'3M':^20s} │ {'1Y':^20s} │ {'2Y':^20s}")
    print(f"  {'':<38s} │ {'N':>6s} {'Win%':>6s} {'Avg':>6s} │ {'N':>6s} {'Win%':>6s} {'Avg':>6s} │ {'N':>6s} {'Win%':>6s} {'Avg':>6s}")
    print(f"  {'-'*110}")

    for name, filt in buckets:
        sub = filt(df_2y)
        sub_3m = sub[sub['return_3m'].notna()]
        sub_1y = sub[sub['return_1y'].notna()]
        sub_2y = sub[sub['return_2y'].notna()]

        def _stats(s, col):
            if len(s) < 10:
                return "   -  ", "   - ", "   - "
            ret = s[col].clip(-90, 1000 if '1y' in col or '2y' in col else 500)
            w = (ret > 0).mean() * 100
            a = ret.mean()
            return f"{len(s):>5,}", f"{w:>5.1f}%", f"{a:>+5.1f}%"

        n3, w3, a3 = _stats(sub_3m, 'return_3m')
        n1, w1, a1 = _stats(sub_1y, 'return_1y')
        n2, w2, a2 = _stats(sub_2y, 'return_2y')
        print(f"  {name:<38s} │ {n3} {w3} {a3} │ {n1} {w1} {a1} │ {n2} {w2} {a2}")

    for sig in signals:
        sub = df_2y[df_2y['signal_type'] == sig]
        sub_3m = sub[sub['return_3m'].notna()]
        sub_1y = sub[sub['return_1y'].notna()]
        sub_2y = sub[sub['return_2y'].notna()]

        def _stats(s, col):
            if len(s) < 10:
                return "   -  ", "   - ", "   - "
            ret = s[col].clip(-90, 1000 if '1y' in col or '2y' in col else 500)
            w = (ret > 0).mean() * 100
            a = ret.mean()
            return f"{len(s):>5,}", f"{w:>5.1f}%", f"{a:>+5.1f}%"

        n3, w3, a3 = _stats(sub_3m, 'return_3m')
        n1, w1, a1 = _stats(sub_1y, 'return_1y')
        n2, w2, a2 = _stats(sub_2y, 'return_2y')
        print(f"  {sig:<38s} │ {n3} {w3} {a3} │ {n1} {w1} {a1} │ {n2} {w2} {a2}")


# ==================== SECTION 6: CURRENT ACTIONABLE STOCKS ====================

def run_current_stocks(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 6: CURRENT ACTIONABLE STOCKS (MOST RECENT SIGNALS)")
    print("#" * 115)

    # Get most recent signals (last 30 days of data)
    max_date = df['date_dt'].max()
    recent = df[df['date_dt'] >= max_date - pd.Timedelta(days=30)].copy()
    print(f"\n  Recent signals (last 30 days): {len(recent):,} from {recent['date_dt'].min().date()} to {max_date.date()}")

    buckets = {
        "BUCKET 1: Quality Growth Compounder": {
            'filter': lambda d: d[(d['value_score_v2'] >= 55) & (d['fundamentals_score'] >= 18) &
                                  (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20) &
                                  (d['rev_growth'] > 10) & (d['market_cap'] > 2e9) &
                                  (d['analyst_count'] >= 6)],
        },
        "BUCKET 2: Bear Market Dip Buy": {
            'filter': lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                                  (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40) &
                                  (d['market_cap'] > 1e9)],
        },
        "BUCKET 3: High-Growth Momentum": {
            'filter': lambda d: d[(d['eps_growth'] >= 35) & (d['ebitda_growth'] >= 33) &
                                  (d['ev_ebitda_clean'] >= 12) & (d['ev_ebitda_clean'] <= 27) &
                                  (d['rsi'] < 43)],
        },
        "BUCKET 4: DCF Value + Quality": {
            'filter': lambda d: d[(d['dcf_upside'] > 0) & (d['value_score_v2'] >= 50) &
                                  (d['fundamentals_score'] >= 15) & (d['market_cap'] > 1e9)],
        },
    }

    for bucket_name, config in buckets.items():
        sub = config['filter'](recent)
        # Deduplicate: keep most recent signal per symbol
        if len(sub) > 0:
            sub = sub.sort_values('date_dt', ascending=False).drop_duplicates('symbol', keep='first')

        print(f"\n  {bucket_name}")
        print(f"  {'─'*80}")

        if len(sub) == 0:
            print(f"    No qualifying stocks in the last 30 days")
            continue

        print(f"  {len(sub)} qualifying stocks:")
        print(f"  {'Symbol':<8s} {'Date':<12s} {'Signal':<22s} {'V2':>5s} {'LT':>5s} {'Fund':>5s} "
              f"{'RSI':>5s} {'EV/EB':>6s} {'RevG':>5s} {'Sector':<20s}")
        print(f"  {'-'*100}")

        sub_sorted = sub.sort_values('value_score_v2', ascending=False)
        for _, row in sub_sorted.head(20).iterrows():
            sector = str(row.get('sector', ''))[:20] if pd.notna(row.get('sector')) else ''
            print(f"  {row['symbol']:<8s} {str(row['date'])[:10]:<12s} {str(row['signal_type']):<22s} "
                  f"{row['value_score_v2']:>5.0f} {row['lt_score']:>5.0f} "
                  f"{row['fundamentals_score']:>5.0f} {row['rsi']:>5.1f} "
                  f"{row['ev_ebitda']:>6.1f} {row['rev_growth']:>5.1f} {sector:<20s}")


# ==================== MAIN ====================

def main():
    start = datetime.now()

    print("=" * 115)
    print("  FINAL BUCKET ANALYSIS — DCF + EARNINGS SURPRISE + 2-YEAR HOLDOUT")
    print(f"  Databases: {BACKTEST_DB}, {MAIN_DB}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 115)

    print("\n  Loading signals...")
    df = load_signals_with_context()
    print(f"  Loaded {len(df):,} signals across {df['symbol'].nunique():,} stocks")

    print("\n  Enriching with supplementary data...")
    df = load_supplementary(df)

    print("\n  Adding DCF valuations...")
    df = enrich_dcf(df)
    dcf_n = df['dcf_upside'].notna().sum()
    print(f"  DCF matched: {dcf_n:,}")

    print("\n  Computing earnings surprise...")
    df = enrich_earnings_surprise(df)
    es_n = df['earnings_surprise_pct'].notna().sum()
    print(f"  Earnings surprise matched: {es_n:,}")

    print("\n  Computing ensemble voting...")
    df = enrich_ensemble_voting(df)

    # Run all sections
    run_dcf_analysis(df)
    run_earnings_surprise_analysis(df)
    run_combined_analysis(df)
    run_four_buckets(df)
    run_2year_analysis(df)
    run_current_stocks(df)

    elapsed = datetime.now() - start
    print(f"\n\n{'='*115}")
    print(f"  Final Bucket Analysis complete! Runtime: {elapsed}")
    print(f"{'='*115}")


if __name__ == "__main__":
    main()

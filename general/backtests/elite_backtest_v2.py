#!/usr/bin/env python3
"""
Elite Backtest V2 Analysis Suite
==================================
Building on v1 results with 6 enhanced analysis sections:

1. Analyst Firm Accuracy Layering on Elite Combos
2. Automated Combo Generation with Enhanced Gates
3. ML Ensemble + Regime-Specific Models
4. Regime-Adaptive Sector/Cap Thresholds
5. Enhanced Stress Tests with Slippage Simulation
6. Earnings Surprise Filter + Ensemble Signal Voting

Runs against existing backtest.db + nasdaq_stocks.db — no new data collection needed.
"""

import sqlite3
import warnings
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# ==================== DATA LOADING (reused from v1) ====================

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
    df['quarter'] = df['date_dt'].dt.to_period('Q')
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


# ==================== NEW ENRICHMENT FUNCTIONS ====================

def compute_top_firms(n_firms=10):
    """Compute top-N firms by upgrade directional accuracy (3M forward return)."""
    conn = sqlite3.connect(BACKTEST_DB)
    grades = pd.read_sql_query("""
        SELECT g.symbol, g.date, g.grading_company, g.action
        FROM historical_grades g WHERE g.date >= '2019-01-01'
    """, conn)

    prices = pd.read_sql_query("""
        SELECT symbol, date, close FROM historical_prices
        WHERE date >= '2019-01-01' ORDER BY symbol, date
    """, conn)
    conn.close()

    prices['date'] = pd.to_datetime(prices['date'])
    grades['date'] = pd.to_datetime(grades['date'])

    # Build price lookup
    price_dict = {}
    for symbol, group in prices.groupby('symbol'):
        price_dict[symbol] = dict(zip(group['date'], group['close']))

    def get_nearest_price(symbol, target_date, window_days=5):
        if symbol not in price_dict:
            return None
        pd_dict = price_dict[symbol]
        for offset in range(window_days + 1):
            d = target_date + pd.Timedelta(days=offset)
            if d in pd_dict:
                return pd_dict[d]
            if offset > 0:
                d = target_date - pd.Timedelta(days=offset)
                if d in pd_dict:
                    return pd_dict[d]
        return None

    # Focus on firms with ≥500 grades
    firm_counts = grades['grading_company'].value_counts()
    major_firms = firm_counts[firm_counts >= 500].index.tolist()

    firm_scores = []
    for firm in major_firms:
        fd = grades[grades['grading_company'] == firm]
        upgrades = fd[fd['action'] == 'upgrade']

        # Sample up to 300 upgrades for speed
        if len(upgrades) > 300:
            upgrades = upgrades.sample(300, random_state=42)

        correct = 0
        total = 0
        for _, row in upgrades.iterrows():
            p0 = get_nearest_price(row['symbol'], row['date'])
            p3m = get_nearest_price(row['symbol'], row['date'] + pd.Timedelta(days=91))
            if p0 and p3m and p0 > 0:
                total += 1
                if p3m > p0:
                    correct += 1

        if total >= 50:
            acc = correct / total * 100
            firm_scores.append({'firm': firm, 'upgrade_acc': acc, 'n_tested': total})

    fdf = pd.DataFrame(firm_scores).sort_values('upgrade_acc', ascending=False)
    top_firms = fdf.head(n_firms)['firm'].tolist()
    return top_firms, fdf


def enrich_firm_grades(df, top_firms):
    """Add top_firm_upgrade/downgrade flags within ±7 days of each signal."""
    conn = sqlite3.connect(BACKTEST_DB)
    # Only load grades from top firms
    placeholders = ','.join(['?'] * len(top_firms))
    grades = pd.read_sql_query(f"""
        SELECT symbol, date, grading_company, action
        FROM historical_grades
        WHERE grading_company IN ({placeholders}) AND date >= '2019-01-01'
    """, conn, params=top_firms)
    conn.close()

    grades['date'] = pd.to_datetime(grades['date'])

    # For each signal, check if a top-firm grade happened within ±7 days
    # Efficient approach: merge on symbol, then filter by date proximity
    grade_up = grades[grades['action'] == 'upgrade'][['symbol', 'date', 'grading_company']].copy()
    grade_dn = grades[grades['action'] == 'downgrade'][['symbol', 'date', 'grading_company']].copy()

    df['top_firm_upgrade'] = 0
    df['top_firm_downgrade'] = 0
    df['top_firm_name'] = ''

    # Process in chunks by symbol for memory efficiency
    symbols_with_grades = set(grades['symbol'].unique())
    for sym in symbols_with_grades:
        sym_mask = df['symbol'] == sym
        sym_signals = df.loc[sym_mask, 'date_dt']
        if sym_signals.empty:
            continue

        sym_up = grade_up[grade_up['symbol'] == sym]
        sym_dn = grade_dn[grade_dn['symbol'] == sym]

        for idx in sym_signals.index:
            sig_date = sym_signals[idx]

            # Check upgrades within ±7 days
            if not sym_up.empty:
                diffs = (sym_up['date'] - sig_date).abs()
                close_up = diffs <= pd.Timedelta(days=7)
                if close_up.any():
                    df.at[idx, 'top_firm_upgrade'] = 1
                    df.at[idx, 'top_firm_name'] = sym_up.loc[close_up.idxmax(), 'grading_company']

            # Check downgrades within ±7 days
            if not sym_dn.empty:
                diffs = (sym_dn['date'] - sig_date).abs()
                close_dn = diffs <= pd.Timedelta(days=7)
                if close_dn.any():
                    df.at[idx, 'top_firm_downgrade'] = 1
                    if df.at[idx, 'top_firm_name'] == '':
                        df.at[idx, 'top_firm_name'] = sym_dn.loc[close_dn.idxmax(), 'grading_company']

    return df


def enrich_earnings_surprise(df):
    """Add earnings_surprise_pct: actual TTM EPS vs analyst consensus for matching fiscal years."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Get actual EPS by fiscal year (sum quarterly eps_diluted per fiscal year, require 4 full quarters)
    income = pd.read_sql_query("""
        SELECT symbol, fiscal_year, SUM(eps_diluted) as actual_eps, COUNT(*) as quarters
        FROM historical_income_statements
        WHERE eps_diluted IS NOT NULL AND fiscal_year IS NOT NULL
        GROUP BY symbol, fiscal_year
        HAVING quarters >= 4
    """, conn)

    # Get analyst consensus EPS by fiscal year
    estimates = pd.read_sql_query("""
        SELECT symbol, fiscal_year, eps_avg
        FROM analyst_estimates_snapshot
        WHERE eps_avg IS NOT NULL AND fiscal_year IS NOT NULL
    """, conn)
    conn.close()

    if income.empty or estimates.empty:
        df['earnings_surprise_pct'] = np.nan
        return df

    # Join on BOTH symbol AND fiscal_year (inner join to only get matches)
    merged = income.merge(estimates, on=['symbol', 'fiscal_year'], how='inner')
    if merged.empty:
        df['earnings_surprise_pct'] = np.nan
        return df

    # For each symbol, take the most recent fiscal year that has BOTH actual + estimate
    merged = merged.sort_values('fiscal_year').groupby('symbol').last().reset_index()

    # Compute surprise: (actual - estimate) / |estimate| * 100
    merged['earnings_surprise_pct'] = np.where(
        merged['eps_avg'].abs() > 0.01,
        (merged['actual_eps'] - merged['eps_avg']) / merged['eps_avg'].abs() * 100,
        np.nan
    )
    surprise_map = merged.set_index('symbol')['earnings_surprise_pct'].clip(-200, 500)

    df['earnings_surprise_pct'] = df['symbol'].map(surprise_map)
    matched = df['earnings_surprise_pct'].notna().sum()
    print(f"    Earnings surprise: {matched:,}/{len(df):,} signals matched "
          f"({len(surprise_map):,} symbols with data)")
    return df


def enrich_ensemble_voting(df):
    """Count how many signals fired on the same (symbol, date)."""
    vote_counts = df.groupby(['symbol', 'date']).size().reset_index(name='concurrent_signals')
    df = df.merge(vote_counts, on=['symbol', 'date'], how='left')
    return df


# ==================== HELPER FUNCTIONS ====================

def stats_row(name, sub, min_count=50):
    n = len(sub)
    if n < min_count:
        return None
    ret = sub['return_3m'].clip(-90, 500)
    wins = ret[ret > 0]
    losses = ret[ret <= 0]
    win_3m = (ret > 0).mean() * 100
    avg_3m = ret.mean()
    med_3m = ret.median()
    avg_1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 20 else np.nan
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf
    return {
        'name': name, 'count': n, 'win_3m': win_3m,
        'avg_3m': avg_3m, 'med_3m': med_3m, 'avg_1y': avg_1y,
        'avg_win': avg_win, 'avg_loss': -abs(losses.mean()) if len(losses) > 0 else 0,
        'profit_factor': profit_factor,
    }


def print_stats_table(rows, title, min_count=50, show_risk=False):
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"\n  No combos with >= {min_count} signals")
        return
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")
    if show_risk:
        print(f"  {'Filter':<50s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s} {'AvgWin':>7s} {'AvgLoss':>8s} {'PF':>5s}")
        print(f"  {'-'*108}")
    else:
        print(f"  {'Filter':<50s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s}")
        print(f"  {'-'*90}")
    rows.sort(key=lambda r: r['win_3m'], reverse=True)
    for r in rows:
        y = f"{r['avg_1y']:+.1f}%" if pd.notna(r['avg_1y']) else "    N/A"
        if show_risk:
            print(f"  {r['name']:<50s} {r['count']:>6,} {r['win_3m']:>6.1f}% "
                  f"{r['avg_3m']:>+7.1f}% {r['med_3m']:>+7.1f}% {y:>7s} "
                  f"{r['avg_win']:>+6.1f}% {r['avg_loss']:>+7.1f}% {r['profit_factor']:>4.2f}")
        else:
            print(f"  {r['name']:<50s} {r['count']:>6,} {r['win_3m']:>6.1f}% "
                  f"{r['avg_3m']:>+7.1f}% {r['med_3m']:>+7.1f}% {y:>7s}")


def bootstrap_ci(returns, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    boot_wins = []
    boot_means = []
    for _ in range(n_boot):
        s = rng.choice(returns, len(returns), replace=True)
        boot_wins.append((s > 0).mean() * 100)
        boot_means.append(s.mean())
    return {
        'win_lo': np.percentile(boot_wins, 2.5),
        'win_hi': np.percentile(boot_wins, 97.5),
        'mean_lo': np.percentile(boot_means, 2.5),
        'mean_hi': np.percentile(boot_means, 97.5),
    }


# V1 elite combo definitions (for reference/reuse)
ELITE_COMBOS_V1 = [
    ("Bear+RSI<35+EV8-18+EPSG>0+Analysts≥6",
     lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 35) &
                 (d['ev_ebitda'] >= 8) & (d['ev_ebitda'] <= 18) &
                 (d['eps_growth'] > 0) & (d['analyst_count'] >= 6)]),
    ("V2≥60+EV5-15+RevG10-30+Analysts≥6",
     lambda d: d[(d['value_score_v2'] >= 60) &
                 (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 15) &
                 (d['rev_growth'] >= 10) & (d['rev_growth'] <= 30) &
                 (d['analyst_count'] >= 6)]),
    ("Bear+RSI<40+Fund≥15+V2≥40",
     lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                 (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40)]),
    ("Tier3+Fund≥18+EV5-20",
     lambda d: d[(d['signal_type'] == 'Tier 3') & (d['fundamentals_score'] >= 18) &
                 (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20)]),
    ("Cap>$2B+Analysts≥6+FCF+LT≥45",
     lambda d: d[(d['market_cap'] > 2e9) & (d['analyst_count'] >= 6) &
                 (d['fcf_positive'] == 1) & (d['lt_score'] >= 45)]),
    ("Bear+RSI<35+Fund≥15+EV5-18+Cap>$1B",
     lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 35) &
                 (d['fundamentals_score'] >= 15) &
                 (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 18) &
                 (d['market_cap'] > 1e9)]),
    ("LT≥55+Fund≥18+RSI35-55+Analysts≥8",
     lambda d: d[(d['lt_score'] >= 55) & (d['fundamentals_score'] >= 18) &
                 (d['rsi'] >= 35) & (d['rsi'] <= 55) &
                 (d['analyst_count'] >= 8)]),
    ("EV5-12+RevG>15+EPSG>10+Cap>$2B",
     lambda d: d[(d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 12) &
                 (d['rev_growth'] > 15) & (d['eps_growth'] > 10) &
                 (d['market_cap'] > 2e9)]),
]


# ==================== SECTION 1: ANALYST FIRM LAYERING ====================

def run_firm_layering(df, top_firms, firm_df):
    print("\n" + "#" * 115)
    print("  SECTION 1: ANALYST FIRM ACCURACY LAYERING ON ELITE COMBOS")
    print("#" * 115)

    # Show top firms used
    print(f"\n  Top-10 Firms by Upgrade Accuracy:")
    print(f"  {'Firm':<30s} {'Upgrade Acc':>11s} {'N Tested':>9s}")
    print(f"  {'-'*52}")
    for _, r in firm_df.head(10).iterrows():
        print(f"  {r['firm']:<30s} {r['upgrade_acc']:>10.1f}% {r['n_tested']:>8,}")

    n_up = (df['top_firm_upgrade'] == 1).sum()
    n_dn = (df['top_firm_downgrade'] == 1).sum()
    print(f"\n  Signals with top-firm upgrade within ±7d:   {n_up:,}")
    print(f"  Signals with top-firm downgrade within ±7d: {n_dn:,}")

    # Walk-forward windows
    windows = [
        (2021, 2022, 2023, 2023),
        (2021, 2023, 2024, 2024),
        (2022, 2024, 2025, 2026),
    ]

    print(f"\n  --- Firm Layering Impact on V1 Elite Combos ---")
    print(f"  {'Combo':<42s} │ {'Baseline':>8s} {'N':>6s} │ {'+FirmUp':>8s} {'N':>5s} {'Δ':>6s} │ {'+FirmDn':>8s} {'N':>5s} │ {'WF?':>4s}")
    print(f"  {'-'*110}")

    layered_elites = []
    for name, filt in ELITE_COMBOS_V1:
        sub = filt(df)
        if len(sub) < 20:
            continue

        # Baseline
        base_win = (sub['return_3m'] > 0).mean() * 100

        # + top firm upgrade
        sub_up = sub[sub['top_firm_upgrade'] == 1]
        up_win = (sub_up['return_3m'] > 0).mean() * 100 if len(sub_up) >= 10 else np.nan

        # + top firm downgrade (expect worse — confirms signal validity)
        sub_dn = sub[sub['top_firm_downgrade'] == 1]
        dn_win = (sub_dn['return_3m'] > 0).mean() * 100 if len(sub_dn) >= 10 else np.nan

        # Walk-forward on firm-layered version
        wf_pass = True
        if len(sub_up) >= 10:
            for ts, te, vs, ve in windows:
                test = sub_up[(sub_up['year'] >= vs) & (sub_up['year'] <= ve)]
                if len(test) >= 5:
                    w = (test['return_3m'] > 0).mean() * 100
                    if w < 52:
                        wf_pass = False
        else:
            wf_pass = False

        delta = (up_win - base_win) if pd.notna(up_win) else np.nan
        delta_s = f"{delta:>+5.1f}%" if pd.notna(delta) else "   N/A"
        up_s = f"{up_win:>7.1f}%" if pd.notna(up_win) else "    N/A"
        dn_s = f"{dn_win:>7.1f}%" if pd.notna(dn_win) else "    N/A"
        wf_s = "PASS" if wf_pass else "fail"
        n_up_combo = len(sub_up)
        n_dn_combo = len(sub_dn)

        print(f"  {name:<42s} │ {base_win:>7.1f}% {len(sub):>5,} │ {up_s} {n_up_combo:>4,} {delta_s} │ {dn_s} {n_dn_combo:>4,} │ {wf_s:>4s}")

        if pd.notna(up_win) and up_win > base_win and n_up_combo >= 15:
            layered_elites.append((name + '+FirmUp', lambda d, f=filt: f(d)[f(d)['top_firm_upgrade'] == 1],
                                   up_win, n_up_combo))

    # Bootstrap CI on promising layered combos
    if layered_elites:
        print(f"\n\n  --- Bootstrap CI for Firm-Layered Combos ---")
        print(f"  {'Combo':<52s} {'N':>6s} {'Win%':>6s} {'CI Low':>7s} {'CI Hi':>7s} {'Verdict':>8s}")
        print(f"  {'-'*90}")
        for name, filt, win_rate, n in layered_elites:
            sub = filt(df)
            if len(sub) < 15:
                continue
            returns = sub['return_3m'].clip(-90, 500).values
            ci = bootstrap_ci(returns)
            verdict = "ELITE" if ci['win_lo'] >= 55 else ("STRONG" if ci['win_lo'] >= 50 else "weak")
            print(f"  {name:<52s} {len(returns):>5,} {win_rate:>5.1f}% {ci['win_lo']:>6.1f}% "
                  f"{ci['win_hi']:>6.1f}% {verdict:>7s}")


# ==================== SECTION 2: AUTOMATED COMBO GENERATION ====================

def run_auto_combo_generation(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 2: AUTOMATED COMBO GENERATION WITH ENHANCED GATES")
    print("#" * 115)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("\n  scikit-learn not installed. Skipping.")
        return

    features = ['lt_score', 'value_score_v2', 'trend_score', 'fundamentals_score',
                'valuation_score', 'momentum_score', 'rsi', 'ev_ebitda_clean',
                'rev_growth', 'eps_growth', 'ebitda_growth', 'adx']

    ml_df = df[features + ['return_3m', 'year']].dropna(subset=features + ['return_3m'])
    X = ml_df[features].values
    y = (ml_df['return_3m'] > 0).astype(int).values

    # Step 1: Get feature importances from Random Forest
    print(f"\n  Training Random Forest for feature importance (n={len(ml_df):,})...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=50,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    top_features = imp.head(8).index.tolist()

    print(f"\n  Top-8 Features by Importance:")
    for feat in top_features:
        bar = '█' * int(imp[feat] * 200)
        print(f"    {feat:<22s} {imp[feat]:.4f}  {bar}")

    # Step 2: Find optimal quintile ranges for each top feature
    print(f"\n  Computing optimal ranges per feature...")
    feature_ranges = {}
    for feat in top_features:
        feat_data = df[[feat, 'return_3m']].dropna()
        if len(feat_data) < 500:
            continue
        try:
            bins = pd.qcut(feat_data[feat], q=5, duplicates='drop')
            grouped = feat_data.groupby(bins)['return_3m'].agg(
                count='count', win_rate=lambda x: (x > 0).mean() * 100
            )
            best = grouped.loc[grouped['win_rate'].idxmax()]
            interval = grouped['win_rate'].idxmax()
            feature_ranges[feat] = (interval.left, interval.right, best['win_rate'], int(best['count']))
            print(f"    {feat:<22s} Best range: [{interval.left:.1f}, {interval.right:.1f}] → "
                  f"{best['win_rate']:.1f}% win (n={int(best['count']):,})")
        except Exception:
            continue

    # Step 3: Generate combos (2-4 features at their optimal ranges)
    print(f"\n  Generating combos from {len(feature_ranges)} features...")
    combo_results = []
    combo_count = 0

    for r in range(2, 5):  # 2, 3, 4 feature combos
        for combo_feats in itertools.combinations(feature_ranges.keys(), r):
            combo_count += 1
            # Build filter
            mask = pd.Series(True, index=df.index)
            name_parts = []
            for feat in combo_feats:
                lo, hi, _, _ = feature_ranges[feat]
                mask = mask & (df[feat] >= lo) & (df[feat] <= hi)
                name_parts.append(f"{feat}[{lo:.0f}-{hi:.0f}]")

            sub = df[mask]
            if len(sub) < 100:
                continue

            ret = sub['return_3m'].clip(-90, 500)
            win = (ret > 0).mean() * 100
            if win < 55:
                continue

            # Profit factor
            wins = ret[ret > 0]
            losses = ret[ret <= 0]
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            pf = avg_win / avg_loss if avg_loss > 0 else np.inf

            combo_results.append({
                'name': '+'.join(name_parts),
                'features': combo_feats,
                'n': len(sub),
                'win': win,
                'avg_3m': ret.mean(),
                'pf': pf,
                'mask': mask,
            })

    print(f"  Generated {combo_count} combos → {len(combo_results)} pass Gate 1 (N≥100) + Gate 2 (Win≥55%)")

    # Gate 3: Walk-forward stability
    windows = [(2021, 2022, 2023, 2023), (2021, 2023, 2024, 2024), (2022, 2024, 2025, 2026)]
    wf_pass = []
    for c in combo_results:
        sub = df[c['mask']]
        stable = True
        for ts, te, vs, ve in windows:
            test = sub[(sub['year'] >= vs) & (sub['year'] <= ve)]
            if len(test) >= 10:
                w = (test['return_3m'] > 0).mean() * 100
                if w < 52:
                    stable = False
                    break
        if stable:
            wf_pass.append(c)

    print(f"  Gate 3 (Walk-forward ≥52%): {len(wf_pass)} survive")

    # Gate 4: Bootstrap CI
    ci_pass = []
    for c in wf_pass:
        sub = df[c['mask']]
        returns = sub['return_3m'].clip(-90, 500).values
        ci = bootstrap_ci(returns, n_boot=2000)
        c['ci_lo'] = ci['win_lo']
        c['ci_hi'] = ci['win_hi']
        if ci['win_lo'] >= 55:
            ci_pass.append(c)

    print(f"  Gate 4 (CI lower ≥55%): {len(ci_pass)} survive")

    # Gate 5: Profit Factor
    pf_pass = [c for c in ci_pass if c['pf'] >= 1.4]
    print(f"  Gate 5 (PF ≥1.4): {len(pf_pass)} survive")

    # Rank by composite score
    if pf_pass:
        max_n = max(c['n'] for c in pf_pass)
        for c in pf_pass:
            c['composite'] = (0.4 * c['win'] + 0.3 * c['ci_lo'] +
                              0.2 * min(c['pf'], 5) * 20 + 0.1 * (c['n'] / max_n) * 100)

        pf_pass.sort(key=lambda c: c['composite'], reverse=True)

        print(f"\n  {'='*115}")
        print(f"  TOP AUTO-GENERATED COMBOS (All 5 Gates Passed)")
        print(f"  {'='*115}")
        print(f"  {'Combo':<60s} {'N':>6s} {'Win%':>6s} {'CI Lo':>6s} {'CI Hi':>6s} {'PF':>5s} {'Avg3M':>7s} {'Score':>6s}")
        print(f"  {'-'*108}")

        for c in pf_pass[:15]:
            print(f"  {c['name']:<60s} {c['n']:>5,} {c['win']:>5.1f}% {c['ci_lo']:>5.1f}% "
                  f"{c['ci_hi']:>5.1f}% {c['pf']:>4.2f} {c['avg_3m']:>+6.1f}% {c['composite']:>5.1f}")
    else:
        print(f"\n  No combos survived all 5 gates. Showing best from Gate 3:")
        if wf_pass:
            wf_pass.sort(key=lambda c: c['win'], reverse=True)
            print(f"  {'Combo':<60s} {'N':>6s} {'Win%':>6s} {'PF':>5s} {'Avg3M':>7s}")
            print(f"  {'-'*85}")
            for c in wf_pass[:10]:
                print(f"  {c['name']:<60s} {c['n']:>5,} {c['win']:>5.1f}% {c['pf']:>4.2f} {c['avg_3m']:>+6.1f}%")

    return pf_pass if pf_pass else wf_pass


# ==================== SECTION 3: ML ENSEMBLE + REGIME-SPECIFIC MODELS ====================

def run_ml_ensemble(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 3: ML ENSEMBLE + REGIME-SPECIFIC MODELS")
    print("#" * 115)

    try:
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("\n  scikit-learn not installed. Skipping.")
        return

    # Expanded feature set
    base_features = ['lt_score', 'value_score_v2', 'trend_score', 'fundamentals_score',
                     'valuation_score', 'momentum_score', 'market_risk_score',
                     'rsi', 'adx', 'ev_ebitda_clean', 'rev_growth', 'eps_growth',
                     'ebitda_growth']
    new_features = ['top_firm_upgrade', 'top_firm_downgrade', 'earnings_surprise_pct',
                    'concurrent_signals', 'fcf_positive', 'analyst_count']

    # Add sector dummies for top 5 sectors
    top_5_sectors = df['sector'].value_counts().head(5).index.tolist()
    for s in top_5_sectors:
        col = f"sector_{s.replace(' ', '_')[:15]}"
        df[col] = (df['sector'] == s).astype(int)
        new_features.append(col)

    all_features = base_features + new_features
    # Deduplicate column list (market_risk_score is already in base_features)
    extra_cols = [c for c in ['return_3m', 'year', 'market_risk_score'] if c not in all_features]
    ml_df = df[all_features + extra_cols].copy()
    ml_df = ml_df.dropna(subset=base_features + ['return_3m'])
    # Fill NaN in new features with 0
    for f in new_features:
        ml_df[f] = ml_df[f].fillna(0)

    X = ml_df[all_features].values
    y = (ml_df['return_3m'] > 0).astype(int).values

    print(f"\n  Dataset: {len(ml_df):,} signals | {y.mean()*100:.1f}% positive | {len(all_features)} features")

    # Train/test split
    train_mask = ml_df['year'] <= 2024
    test_mask = ml_df['year'] >= 2025
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} (2021-2024) | Test: {len(X_test):,} (2025+)")

    # --- 3A: XGBoost ---
    print(f"\n  --- 3A: Training XGBoost ---")
    xgb = GradientBoostingClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50,
                                      learning_rate=0.05, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_train = (xgb.predict(X_train) == y_train).mean() * 100
    xgb_test = (xgb.predict(X_test) == y_test).mean() * 100
    print(f"  XGBoost Train: {xgb_train:.1f}% | Test: {xgb_test:.1f}%")

    # --- 3B: Random Forest ---
    print(f"\n  --- 3B: Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=50,
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train = (rf.predict(X_train) == y_train).mean() * 100
    rf_test = (rf.predict(X_test) == y_test).mean() * 100
    print(f"  Random Forest Train: {rf_train:.1f}% | Test: {rf_test:.1f}%")

    # Feature importance comparison
    xgb_imp = pd.Series(xgb.feature_importances_, index=all_features)
    rf_imp = pd.Series(rf.feature_importances_, index=all_features)

    print(f"\n  Feature Importance Comparison (Top 12):")
    print(f"  {'Feature':<25s} {'XGBoost':>8s} {'RF':>8s} {'Avg':>8s}")
    print(f"  {'-'*52}")
    avg_imp = ((xgb_imp + rf_imp) / 2).sort_values(ascending=False)
    for feat in avg_imp.head(12).index:
        print(f"  {feat:<25s} {xgb_imp[feat]:>7.4f} {rf_imp[feat]:>7.4f} {avg_imp[feat]:>7.4f}")

    # --- 3C: Calibrated Ensemble ---
    print(f"\n  --- 3C: Calibrated Ensemble (XGB + RF averaged) ---")
    cal_xgb = CalibratedClassifierCV(xgb, cv=3, method='isotonic')
    cal_xgb.fit(X_train, y_train)
    xgb_probs = cal_xgb.predict_proba(X_test)[:, 1]

    cal_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    cal_rf.fit(X_train, y_train)
    rf_probs = cal_rf.predict_proba(X_test)[:, 1]

    ensemble_probs = (xgb_probs + rf_probs) / 2

    test_df = ml_df[test_mask].copy()
    test_df['xgb_prob'] = xgb_probs
    test_df['rf_prob'] = rf_probs
    test_df['ensemble_prob'] = ensemble_probs

    # Decile analysis
    print(f"\n  Ensemble Probability Decile Analysis (OOS 2025+):")
    print(f"  {'Decile':<8s} {'Prob Range':<18s} {'Count':>6s} {'Actual Win%':>12s} {'Avg3M':>8s} {'Med3M':>8s}")
    print(f"  {'-'*62}")

    try:
        deciles = pd.qcut(test_df['ensemble_prob'], q=10, duplicates='drop')
        for i, dec in enumerate(sorted(test_df.groupby(deciles).groups.keys())):
            sub = test_df[deciles == dec]
            w = (sub['return_3m'] > 0).mean() * 100
            a = sub['return_3m'].clip(-90, 500).mean()
            m = sub['return_3m'].clip(-90, 500).median()
            print(f"  D{i+1:<6d} {str(dec):<18s} {len(sub):>5,} {w:>11.1f}% {a:>+7.1f}% {m:>+7.1f}%")
    except Exception:
        buckets = pd.cut(test_df['ensemble_prob'], bins=5, duplicates='drop')
        for bucket in sorted(test_df.groupby(buckets).groups.keys()):
            sub = test_df[buckets == bucket]
            if len(sub) < 10:
                continue
            w = (sub['return_3m'] > 0).mean() * 100
            a = sub['return_3m'].clip(-90, 500).mean()
            print(f"  {str(bucket):<26s} {len(sub):>5,} {w:>11.1f}% {a:>+7.1f}%")

    # ML Elite v2 trigger
    top_prob = test_df['ensemble_prob'].quantile(0.9)
    ml_elite = test_df[test_df['ensemble_prob'] >= top_prob]
    if len(ml_elite) >= 10:
        w = (ml_elite['return_3m'] > 0).mean() * 100
        a = ml_elite['return_3m'].clip(-90, 500).mean()
        print(f"\n  ** ML Elite v2 Trigger (top 10%, prob ≥ {top_prob:.3f}):")
        print(f"     Count: {len(ml_elite)} | Win%: {w:.1f}% | Avg 3M: {a:+.1f}%")

    # --- 3D: Regime-Specific Models ---
    print(f"\n\n  --- 3D: Regime-Specific Models ---")
    for regime_name, regime_val in [('Bull', 10), ('Bear', 0)]:
        r_train_mask = train_mask & (ml_df['market_risk_score'] == regime_val)
        r_test_mask = test_mask & (ml_df['market_risk_score'] == regime_val)

        if r_train_mask.sum() < 500 or r_test_mask.sum() < 50:
            print(f"\n  {regime_name}: Insufficient data (train={r_train_mask.sum()}, test={r_test_mask.sum()})")
            continue

        rX_train, ry_train = X[r_train_mask], y[r_train_mask]
        rX_test, ry_test = X[r_test_mask], y[r_test_mask]

        r_gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, min_samples_leaf=30,
                                           learning_rate=0.05, random_state=42)
        r_gb.fit(rX_train, ry_train)
        r_train_acc = (r_gb.predict(rX_train) == ry_train).mean() * 100
        r_test_acc = (r_gb.predict(rX_test) == ry_test).mean() * 100

        cal_r = CalibratedClassifierCV(r_gb, cv=3, method='isotonic')
        cal_r.fit(rX_train, ry_train)
        r_probs = cal_r.predict_proba(rX_test)[:, 1]

        r_test_df = ml_df[r_test_mask].copy()
        r_test_df['prob'] = r_probs

        top_r = r_test_df['prob'].quantile(0.9)
        r_elite = r_test_df[r_test_df['prob'] >= top_r]

        print(f"\n  {regime_name} Model: Train {r_train_acc:.1f}% | Test {r_test_acc:.1f}% "
              f"(train={len(rX_train):,}, test={len(rX_test):,})")

        if len(r_elite) >= 5:
            r_w = (r_elite['return_3m'] > 0).mean() * 100
            r_a = r_elite['return_3m'].clip(-90, 500).mean()
            print(f"    {regime_name} Elite (top 10%, prob ≥ {top_r:.3f}): "
                  f"N={len(r_elite)} | Win%: {r_w:.1f}% | Avg 3M: {r_a:+.1f}%")

            # Top 3 features for this regime
            r_imp = pd.Series(r_gb.feature_importances_, index=all_features).sort_values(ascending=False)
            top3 = r_imp.head(3)
            print(f"    Top features: {', '.join(f'{f} ({v:.3f})' for f, v in top3.items())}")


# ==================== SECTION 4: REGIME-ADAPTIVE SECTOR/CAP THRESHOLDS ====================

def run_regime_sector_thresholds(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 4: REGIME-ADAPTIVE SECTOR/CAP THRESHOLDS")
    print("#" * 115)

    lt_thresholds = [40, 45, 50, 55]
    v2_thresholds = [40, 45, 50, 55, 60]
    fund_thresholds = [12, 15, 18]

    # --- 4A: Sector × Regime ---
    print(f"\n  --- 4A: Optimal Thresholds per Sector × Regime ---")
    top_sectors = df['sector'].value_counts().head(6).index.tolist()

    print(f"\n  {'Regime':<6s} {'Sector':<22s} │ {'Best LT':>7s} {'Best V2':>7s} {'Best Fund':>9s} │ "
          f"{'Win%':>6s} {'N':>6s} {'Avg3M':>7s}")
    print(f"  {'-'*85}")

    sector_thresholds = {}
    for regime_name, regime_val in [('Bull', 10), ('Bear', 0)]:
        regime_df = df[df['market_risk_score'] == regime_val]
        for sector in top_sectors:
            sec_df = regime_df[regime_df['sector'] == sector]
            if len(sec_df) < 100:
                continue

            best = {'win': 0, 'lt': 40, 'v2': 40, 'fund': 12, 'n': 0, 'avg': 0}
            for lt in lt_thresholds:
                for v2 in v2_thresholds:
                    for fund in fund_thresholds:
                        sub = sec_df[(sec_df['lt_score'] >= lt) & (sec_df['value_score_v2'] >= v2) &
                                     (sec_df['fundamentals_score'] >= fund)]
                        if len(sub) < 30:
                            continue
                        w = (sub['return_3m'] > 0).mean() * 100
                        if w > best['win']:
                            best = {'win': w, 'lt': lt, 'v2': v2, 'fund': fund,
                                    'n': len(sub), 'avg': sub['return_3m'].clip(-90, 500).mean()}

            if best['n'] >= 30:
                print(f"  {regime_name:<6s} {sector:<22s} │ LT≥{best['lt']:<3d} V2≥{best['v2']:<3d} "
                      f"Fund≥{best['fund']:<3d} │ {best['win']:>5.1f}% {best['n']:>5,} {best['avg']:>+6.1f}%")
                sector_thresholds[(regime_name, sector)] = best

    # --- 4B: Cap × Regime ---
    print(f"\n\n  --- 4B: Optimal Thresholds per Cap Bucket × Regime ---")
    cap_buckets = [
        ('Micro (<$300M)', 0, 3e8),
        ('Small ($300M-$2B)', 3e8, 2e9),
        ('Mid ($2B-$10B)', 2e9, 10e9),
        ('Large ($10B+)', 10e9, 1e15),
    ]

    print(f"\n  {'Regime':<6s} {'Cap Bucket':<20s} │ {'Best LT':>7s} {'Best V2':>7s} {'Best Fund':>9s} │ "
          f"{'Win%':>6s} {'N':>6s} {'Avg3M':>7s}")
    print(f"  {'-'*85}")

    cap_thresholds = {}
    for regime_name, regime_val in [('Bull', 10), ('Bear', 0)]:
        regime_df = df[df['market_risk_score'] == regime_val]
        for cap_name, cap_lo, cap_hi in cap_buckets:
            cap_df = regime_df[(regime_df['market_cap'] >= cap_lo) & (regime_df['market_cap'] < cap_hi)]
            if len(cap_df) < 100:
                continue

            best = {'win': 0, 'lt': 40, 'v2': 40, 'fund': 12, 'n': 0, 'avg': 0}
            for lt in lt_thresholds:
                for v2 in v2_thresholds:
                    for fund in fund_thresholds:
                        sub = cap_df[(cap_df['lt_score'] >= lt) & (cap_df['value_score_v2'] >= v2) &
                                     (cap_df['fundamentals_score'] >= fund)]
                        if len(sub) < 30:
                            continue
                        w = (sub['return_3m'] > 0).mean() * 100
                        if w > best['win']:
                            best = {'win': w, 'lt': lt, 'v2': v2, 'fund': fund,
                                    'n': len(sub), 'avg': sub['return_3m'].clip(-90, 500).mean()}

            if best['n'] >= 30:
                print(f"  {regime_name:<6s} {cap_name:<20s} │ LT≥{best['lt']:<3d} V2≥{best['v2']:<3d} "
                      f"Fund≥{best['fund']:<3d} │ {best['win']:>5.1f}% {best['n']:>5,} {best['avg']:>+6.1f}%")
                cap_thresholds[(regime_name, cap_name)] = best

    # --- 4C: Composite regime-adaptive strategy ---
    print(f"\n\n  --- 4C: Regime-Adaptive Composite vs Flat Threshold ---")

    # Flat baseline: LT≥50, V2≥50, Fund≥15 everywhere
    flat = df[(df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) & (df['fundamentals_score'] >= 15)]

    # Adaptive: use sector-specific thresholds
    adaptive_mask = pd.Series(False, index=df.index)
    for (regime_name, sector), best in sector_thresholds.items():
        regime_val = 10 if regime_name == 'Bull' else 0
        mask = ((df['market_risk_score'] == regime_val) & (df['sector'] == sector) &
                (df['lt_score'] >= best['lt']) & (df['value_score_v2'] >= best['v2']) &
                (df['fundamentals_score'] >= best['fund']))
        adaptive_mask = adaptive_mask | mask

    adaptive = df[adaptive_mask]

    rows = [
        stats_row("Flat (LT≥50, V2≥50, Fund≥15)", flat, 50),
        stats_row("Regime-Adaptive Composite", adaptive, 50),
    ]
    print_stats_table(rows, "Flat vs Adaptive Strategy Comparison", min_count=50, show_risk=True)

    # Sector recommendations
    print(f"\n  Sector Boost/Penalty Recommendations:")
    for (regime_name, sector), best in sorted(sector_thresholds.items()):
        baseline_lt, baseline_v2 = 50, 50
        lt_diff = best['lt'] - baseline_lt
        v2_diff = best['v2'] - baseline_v2
        if abs(lt_diff) >= 5 or abs(v2_diff) >= 5:
            lt_s = f"LT {lt_diff:+d}" if lt_diff != 0 else "LT flat"
            v2_s = f"V2 {v2_diff:+d}" if v2_diff != 0 else "V2 flat"
            print(f"    {regime_name:>4s} {sector:<22s}: {lt_s}, {v2_s} (optimal: {best['win']:.1f}% win on {best['n']:,})")


# ==================== SECTION 5: ENHANCED STRESS TESTS WITH SLIPPAGE ====================

def run_enhanced_stress_tests(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 5: ENHANCED STRESS TESTS WITH SLIPPAGE SIMULATION")
    print("#" * 115)

    SLIPPAGE_RATE = 0.002  # 0.2% round-trip (0.1% each way)
    MAX_POSITIONS = 10
    STARTING_CAPITAL = 100000

    strategies = [
        ("Tier 1", df[df['signal_type'] == 'Tier 1']),
        ("Tier 3", df[df['signal_type'] == 'Tier 3']),
        ("Regime Buy Bear", df[df['signal_type'] == 'Regime Buy Bear']),
        ("Bear+RSI<40+Fund≥15+V2≥40",
         df[(df['market_bearish'] == 1) & (df['rsi'] < 40) &
            (df['fundamentals_score'] >= 15) & (df['value_score_v2'] >= 40)]),
        ("V2≥60+EV5-15+RevG10-30",
         df[(df['value_score_v2'] >= 60) & (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 15) &
            (df['rev_growth'] >= 10) & (df['rev_growth'] <= 30)]),
    ]

    # --- 5A: Full period simulation with slippage ---
    print(f"\n  --- 5A: Portfolio Simulation (${STARTING_CAPITAL/1000:.0f}K, max {MAX_POSITIONS} positions, "
          f"{SLIPPAGE_RATE*100:.1f}% slippage) ---")
    print(f"  Period: Full dataset | Position sizing: equal-weight across max {MAX_POSITIONS}")
    print(f"\n  {'Strategy':<38s} {'Trades':>6s} {'Win%':>6s} {'End$':>10s} {'TotRet':>8s} {'Sharpe*':>8s} "
          f"{'MaxDD':>7s} {'Calmar':>7s} {'PF':>5s}")
    print(f"  {'-'*105}")
    print(f"  * Sharpe = annualized mean/std of trade returns (simplified)")

    for strat_name, strat_df in strategies:
        signals = strat_df[strat_df['return_3m'].notna()].sort_values('date_dt')
        if len(signals) < 20:
            continue

        capital = STARTING_CAPITAL
        equity_curve = [capital]
        trade_returns = []
        active_positions = 0

        for _, row in signals.iterrows():
            if active_positions >= MAX_POSITIONS:
                continue

            pos_size = capital / MAX_POSITIONS
            raw_ret = row['return_3m'] / 100
            net_ret = raw_ret - SLIPPAGE_RATE  # Apply slippage
            pnl = pos_size * net_ret
            capital += pnl
            equity_curve.append(capital)
            trade_returns.append(net_ret * 100)

            # Approximate position cycling (every ~3 months a position frees up)
            active_positions = min(active_positions + 1, MAX_POSITIONS)
            if len(trade_returns) % 3 == 0:
                active_positions = max(0, active_positions - 1)

        if not trade_returns:
            continue

        tr = np.array(trade_returns)
        wins = (tr > 0).mean() * 100
        total_ret = (capital - STARTING_CAPITAL) / STARTING_CAPITAL * 100
        avg_ret = tr.mean()
        std_ret = tr.std() if tr.std() > 0 else 1
        sharpe = (avg_ret / std_ret) * np.sqrt(4)  # ~4 quarters/year

        # Max drawdown from equity curve
        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak * 100
        max_dd = dd.min()
        calmar = total_ret / abs(max_dd) if abs(max_dd) > 0 else 0

        # Profit factor
        wins_arr = tr[tr > 0]
        losses_arr = tr[tr <= 0]
        pf = (wins_arr.mean() / abs(losses_arr.mean())) if len(losses_arr) > 0 and abs(losses_arr.mean()) > 0 else np.inf

        print(f"  {strat_name:<38s} {len(tr):>5,} {wins:>5.1f}% ${capital:>9,.0f} {total_ret:>+7.1f}% "
              f"{sharpe:>7.3f} {max_dd:>+6.1f}% {calmar:>6.3f} {pf:>4.2f}")

    # --- 5B: 2022 Bear Stress Test ---
    print(f"\n\n  --- 5B: 2022 Bear Market Stress Test ---")
    print(f"  {'Strategy':<38s} {'Trades':>6s} {'Win%':>6s} {'Avg3M':>7s} {'MaxLoss':>8s} {'PF':>5s}")
    print(f"  {'-'*75}")

    for strat_name, strat_df in strategies:
        bear_2022 = strat_df[(strat_df['year'] == 2022) & (strat_df['return_3m'].notna())]
        if len(bear_2022) < 10:
            continue
        ret = bear_2022['return_3m'].clip(-90, 500)
        w = (ret > 0).mean() * 100
        a = ret.mean()
        max_loss = ret.min()
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
        print(f"  {strat_name:<38s} {len(ret):>5,} {w:>5.1f}% {a:>+6.1f}% {max_loss:>+7.1f}% {pf:>4.2f}")

    # --- 5C: ADX Volatility Filter ---
    print(f"\n\n  --- 5C: ADX Volatility Filter Impact ---")
    print(f"  {'Strategy':<38s} │ {'No ADX Filter':^22s} │ {'ADX<25 Only':^22s} │ {'Δ Win%':>7s}")
    print(f"  {'':<38s} │ {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} │ {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} │")
    print(f"  {'-'*98}")

    for strat_name, strat_df in strategies:
        base = strat_df[strat_df['return_3m'].notna()]
        adx_filt = base[base['adx'] < 25]
        if len(base) < 20 or len(adx_filt) < 10:
            continue
        bw = (base['return_3m'] > 0).mean() * 100
        ba = base['return_3m'].clip(-90, 500).mean()
        aw = (adx_filt['return_3m'] > 0).mean() * 100
        aa = adx_filt['return_3m'].clip(-90, 500).mean()
        delta = aw - bw
        print(f"  {strat_name:<38s} │ {len(base):>5,} {bw:>5.1f}% {ba:>+6.1f}% │ "
              f"{len(adx_filt):>5,} {aw:>5.1f}% {aa:>+6.1f}% │ {delta:>+6.1f}%")

    # --- 5D: Position sizing comparison ---
    print(f"\n\n  --- 5D: Position Sizing Comparison (OOS 2025+) ---")
    print(f"  {'Strategy':<38s} │ {'2% Fixed ($2K)':^20s} │ {'EqWt/10 (~$10K)':^20s}")
    print(f"  {'':<38s} │ {'P&L':>9s} {'Return':>9s} │ {'P&L':>9s} {'Return':>9s}")
    print(f"  {'-'*90}")

    for strat_name, strat_df in strategies:
        oos = strat_df[(strat_df['year'] >= 2025) & (strat_df['return_3m'].notna())]
        if len(oos) < 10:
            continue
        # 2% fixed
        fixed_pnl = sum(2000 * (r / 100) for r in oos['return_3m'].clip(-90, 500))
        fixed_ret = fixed_pnl / STARTING_CAPITAL * 100

        # Equal weight / 10 positions
        pos_size = STARTING_CAPITAL / MAX_POSITIONS
        eq_pnl = sum(pos_size * (r / 100 - SLIPPAGE_RATE) for r in oos['return_3m'].clip(-90, 500))
        eq_ret = eq_pnl / STARTING_CAPITAL * 100

        print(f"  {strat_name:<38s} │ ${fixed_pnl:>8,.0f} {fixed_ret:>+8.1f}% │ ${eq_pnl:>8,.0f} {eq_ret:>+8.1f}%")


# ==================== SECTION 6: EARNINGS SURPRISE + ENSEMBLE VOTING ====================

def run_earnings_and_voting(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 6: EARNINGS SURPRISE FILTER + ENSEMBLE SIGNAL VOTING")
    print("#" * 115)

    # --- 6A: Earnings Surprise Impact ---
    print(f"\n  --- 6A: Earnings Surprise Impact on Elite Combos ---")

    surprise_coverage = df['earnings_surprise_pct'].notna().sum()
    print(f"  Earnings surprise data available for {surprise_coverage:,}/{len(df):,} signals "
          f"({surprise_coverage/len(df)*100:.1f}%)")

    if surprise_coverage < 1000:
        print(f"  Insufficient earnings surprise data. Skipping detailed analysis.")
    else:
        # Surprise categories
        df_with_es = df[df['earnings_surprise_pct'].notna()].copy()
        df_with_es['surprise_cat'] = pd.cut(df_with_es['earnings_surprise_pct'],
                                             bins=[-np.inf, -10, 0, 10, np.inf],
                                             labels=['Big Miss', 'Miss', 'Beat', 'Big Beat'])

        print(f"\n  Surprise Distribution:")
        for cat in ['Big Miss', 'Miss', 'Beat', 'Big Beat']:
            sub = df_with_es[df_with_es['surprise_cat'] == cat]
            w = (sub['return_3m'] > 0).mean() * 100 if len(sub) >= 20 else np.nan
            ws = f"{w:.1f}%" if pd.notna(w) else "N/A"
            print(f"    {cat:<12s}: {len(sub):>6,} signals | Win%: {ws}")

        # Test surprise overlay on elite combos
        print(f"\n  {'Combo':<42s} │ {'Baseline':>8s} {'N':>6s} │ {'+Beat':>8s} {'N':>5s} {'Δ':>6s} │ "
              f"{'+BigBeat':>8s} {'N':>5s}")
        print(f"  {'-'*100}")

        for name, filt in ELITE_COMBOS_V1[:5]:  # Top 5 combos
            sub = filt(df_with_es)
            if len(sub) < 20:
                continue
            base_w = (sub['return_3m'] > 0).mean() * 100

            beat = sub[sub['earnings_surprise_pct'] > 0]
            beat_w = (beat['return_3m'] > 0).mean() * 100 if len(beat) >= 10 else np.nan

            big_beat = sub[sub['earnings_surprise_pct'] > 10]
            big_beat_w = (big_beat['return_3m'] > 0).mean() * 100 if len(big_beat) >= 10 else np.nan

            delta = (beat_w - base_w) if pd.notna(beat_w) else np.nan
            delta_s = f"{delta:>+5.1f}%" if pd.notna(delta) else "   N/A"
            beat_s = f"{beat_w:>7.1f}%" if pd.notna(beat_w) else "    N/A"
            bb_s = f"{big_beat_w:>7.1f}%" if pd.notna(big_beat_w) else "    N/A"

            print(f"  {name:<42s} │ {base_w:>7.1f}% {len(sub):>5,} │ {beat_s} {len(beat):>4,} "
                  f"{delta_s} │ {bb_s} {len(big_beat):>4,}")

    # --- 6B: Ensemble Signal Voting ---
    print(f"\n\n  --- 6B: Ensemble Signal Voting (Multiple Signals on Same Symbol+Date) ---")

    vote_dist = df['concurrent_signals'].value_counts().sort_index()
    print(f"\n  Vote Distribution:")
    for votes, count in vote_dist.items():
        print(f"    {int(votes)} signal(s): {count:>6,}")

    print(f"\n  Performance by Concurrent Signal Count:")
    print(f"  {'Signals':>8s} {'Count':>7s} {'Win%':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s} {'PF':>5s}")
    print(f"  {'-'*55}")

    for vote_count in sorted(df['concurrent_signals'].unique()):
        if vote_count > 5:
            break
        sub = df[df['concurrent_signals'] == vote_count]
        if len(sub) < 30:
            continue
        ret = sub['return_3m'].clip(-90, 500)
        w = (ret > 0).mean() * 100
        a = ret.mean()
        m = ret.median()
        y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
        ys = f"{y:+.1f}%" if pd.notna(y) else "   N/A"
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
        print(f"  {int(vote_count):>8d} {len(sub):>6,} {w:>6.1f}% {a:>+7.1f}% {m:>+7.1f}% {ys:>7s} {pf:>4.2f}")

    # 2+ votes as a filter on elite combos
    print(f"\n  2+ Votes Overlay on Elite Combos:")
    print(f"  {'Combo':<42s} │ {'Baseline':>8s} {'N':>6s} │ {'2+ Votes':>8s} {'N':>5s} {'Δ':>6s}")
    print(f"  {'-'*85}")

    for name, filt in ELITE_COMBOS_V1[:5]:
        sub = filt(df)
        if len(sub) < 20:
            continue
        base_w = (sub['return_3m'] > 0).mean() * 100

        multi = sub[sub['concurrent_signals'] >= 2]
        multi_w = (multi['return_3m'] > 0).mean() * 100 if len(multi) >= 10 else np.nan
        delta = (multi_w - base_w) if pd.notna(multi_w) else np.nan
        delta_s = f"{delta:>+5.1f}%" if pd.notna(delta) else "   N/A"
        multi_s = f"{multi_w:>7.1f}%" if pd.notna(multi_w) else "    N/A"

        print(f"  {name:<42s} │ {base_w:>7.1f}% {len(sub):>5,} │ {multi_s} {len(multi):>4,} {delta_s}")

    # --- 6C: Strict Holdout (2024-2025 only) ---
    print(f"\n\n  --- 6C: Strict Holdout Validation (2024-2025 Only) ---")
    print(f"  Comparing full-period results to strict out-of-sample holdout")

    holdout = df[(df['year'] >= 2024) & (df['return_3m'].notna())]
    print(f"  Holdout period: 2024-2025 | {len(holdout):,} signals")

    print(f"\n  {'Combo':<42s} │ {'Full Period':^15s} │ {'Holdout 24-25':^15s} │ {'Δ':>6s} {'Verdict':>8s}")
    print(f"  {'':<42s} │ {'Win%':>7s} {'N':>6s} │ {'Win%':>7s} {'N':>6s} │")
    print(f"  {'-'*100}")

    for name, filt in ELITE_COMBOS_V1:
        full = filt(df)
        hold = filt(holdout)
        if len(full) < 30 or len(hold) < 10:
            continue

        full_w = (full['return_3m'] > 0).mean() * 100
        hold_w = (hold['return_3m'] > 0).mean() * 100
        delta = hold_w - full_w
        verdict = "STABLE" if abs(delta) < 5 else ("OK" if abs(delta) < 10 else "DECAY")

        print(f"  {name:<42s} │ {full_w:>6.1f}% {len(full):>5,} │ {hold_w:>6.1f}% {len(hold):>5,} │ "
              f"{delta:>+5.1f}% {verdict:>7s}")

    # Also test key signal types
    print(f"\n  Key Signal Types:")
    print(f"  {'Signal':<42s} │ {'Full Period':^15s} │ {'Holdout 24-25':^15s} │ {'Δ':>6s} {'Verdict':>8s}")
    print(f"  {'':<42s} │ {'Win%':>7s} {'N':>6s} │ {'Win%':>7s} {'N':>6s} │")
    print(f"  {'-'*100}")

    for sig in ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear', 'Regime Buy Bull',
                'Buy_A', 'Analyst Downgrade Recovery']:
        full = df[(df['signal_type'] == sig) & (df['return_3m'].notna())]
        hold = holdout[holdout['signal_type'] == sig]
        if len(full) < 30 or len(hold) < 10:
            continue
        full_w = (full['return_3m'] > 0).mean() * 100
        hold_w = (hold['return_3m'] > 0).mean() * 100
        delta = hold_w - full_w
        verdict = "STABLE" if abs(delta) < 5 else ("OK" if abs(delta) < 10 else "DECAY")
        print(f"  {sig:<42s} │ {full_w:>6.1f}% {len(full):>5,} │ {hold_w:>6.1f}% {len(hold):>5,} │ "
              f"{delta:>+5.1f}% {verdict:>7s}")


# ==================== MAIN ====================

def main():
    start = datetime.now()

    print("=" * 115)
    print("  ELITE BACKTEST V2 ANALYSIS SUITE")
    print(f"  Databases: {BACKTEST_DB}, {MAIN_DB}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 115)

    # Phase 1: Load base data
    print("\n  Loading signals with full indicator context...")
    df = load_signals_with_context()
    print(f"  Loaded {len(df):,} signals across {df['symbol'].nunique():,} stocks")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print("\n  Enriching with market cap, FCF, analyst coverage, sector data...")
    df = load_supplementary(df)
    print(f"  Market cap matched: {df['market_cap'].notna().sum():,} | Sector matched: {df['sector'].notna().sum():,}")

    # Phase 2: Compute top analyst firms
    print("\n  Computing top-10 analyst firms by upgrade accuracy...")
    top_firms, firm_df = compute_top_firms(n_firms=10)
    print(f"  Top firms: {', '.join(top_firms[:5])}...")

    # Phase 3: Enrich with firm grades
    print("\n  Enriching signals with top-firm grade proximity (±7 days)...")
    df = enrich_firm_grades(df, top_firms)
    n_up = (df['top_firm_upgrade'] == 1).sum()
    n_dn = (df['top_firm_downgrade'] == 1).sum()
    print(f"  Firm upgrades matched: {n_up:,} | Firm downgrades matched: {n_dn:,}")

    # Phase 4: Enrich with earnings surprise
    print("\n  Computing earnings surprise (TTM actual vs consensus)...")
    df = enrich_earnings_surprise(df)
    n_es = df['earnings_surprise_pct'].notna().sum()
    print(f"  Earnings surprise matched: {n_es:,}/{len(df):,}")

    # Phase 5: Enrich with ensemble voting
    print("\n  Computing ensemble signal voting...")
    df = enrich_ensemble_voting(df)
    multi = (df['concurrent_signals'] >= 2).sum()
    print(f"  Signals with 2+ concurrent: {multi:,}/{len(df):,}")

    # ========== RUN ALL 6 SECTIONS ==========

    print("\n\n  Running Section 1: Analyst Firm Accuracy Layering...")
    run_firm_layering(df, top_firms, firm_df)

    print("\n\n  Running Section 2: Automated Combo Generation...")
    run_auto_combo_generation(df)

    print("\n\n  Running Section 3: ML Ensemble + Regime Models...")
    run_ml_ensemble(df)

    print("\n\n  Running Section 4: Regime-Adaptive Sector/Cap Thresholds...")
    run_regime_sector_thresholds(df)

    print("\n\n  Running Section 5: Enhanced Stress Tests with Slippage...")
    run_enhanced_stress_tests(df)

    print("\n\n  Running Section 6: Earnings Surprise + Ensemble Voting...")
    run_earnings_and_voting(df)

    elapsed = datetime.now() - start
    print(f"\n\n{'='*115}")
    print(f"  Elite Backtest V2 Analysis complete! Runtime: {elapsed}")
    print(f"{'='*115}")


if __name__ == "__main__":
    main()

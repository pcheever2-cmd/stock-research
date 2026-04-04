#!/usr/bin/env python3
"""
Elite Backtest Analysis Suite
==============================
1. Elite Multi-Factor Combos with Stability Gates
2. ML-Weighted Scoring + XGBoost Win Probability Predictor
3. Regime-Adaptive Filters with Sector/Cap Breakdowns
4. Stress Tests, Risk-Adjusted Metrics, Out-of-Sample Holdout
5. Analyst Firm Accuracy Scoring with Sector Breakdown

Runs against existing backtest.db + nasdaq_stocks.db — no new data collection needed.
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
    """Load all signals joined with daily scores for full indicator context."""
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
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q')
    return df


def load_supplementary(df):
    """Enrich signals with market cap, FCF growth, analyst coverage, sector."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Market cap
    mcaps = pd.read_sql_query("""
        SELECT symbol, market_cap FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol HAVING date = MAX(date)
    """, conn).set_index('symbol')['market_cap']

    # FCF growth
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

    # Analyst coverage
    acov = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot GROUP BY symbol
    """, conn).set_index('symbol')['analyst_count']
    conn.close()

    # Sector from main DB
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


# ==================== HELPER FUNCTIONS ====================

def stats_row(name, sub, min_count=50):
    """Compute stats for a filter subset."""
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
    """Print a formatted stats table."""
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"\n  No combos with >= {min_count} signals")
        return
    print(f"\n{'='*110}")
    print(f"  {title}")
    print(f"{'='*110}")
    if show_risk:
        print(f"  {'Filter':<48s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s} {'AvgWin':>7s} {'AvgLoss':>8s} {'PF':>5s}")
        print(f"  {'-'*105}")
    else:
        print(f"  {'Filter':<50s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s}")
        print(f"  {'-'*90}")
    rows.sort(key=lambda r: r['win_3m'], reverse=True)
    for r in rows:
        y = f"{r['avg_1y']:+.1f}%" if pd.notna(r['avg_1y']) else "    N/A"
        if show_risk:
            print(f"  {r['name']:<48s} {r['count']:>6,} {r['win_3m']:>6.1f}% "
                  f"{r['avg_3m']:>+7.1f}% {r['med_3m']:>+7.1f}% {y:>7s} "
                  f"{r['avg_win']:>+6.1f}% {r['avg_loss']:>+7.1f}% {r['profit_factor']:>4.2f}")
        else:
            print(f"  {r['name']:<50s} {r['count']:>6,} {r['win_3m']:>6.1f}% "
                  f"{r['avg_3m']:>+7.1f}% {r['med_3m']:>+7.1f}% {y:>7s}")


def bootstrap_ci(returns, n_boot=3000, seed=42):
    """Compute 95% bootstrap CI for win rate and mean return."""
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


# ==================== 1. ELITE MULTI-FACTOR COMBOS WITH STABILITY GATES ====================

def run_elite_combos(df):
    print("\n" + "#" * 110)
    print("  SECTION 1: ELITE MULTI-FACTOR COMBOS — Stability-Gated")
    print("#" * 110)

    # Define combo filters as (name, filter_func) pairs
    combos = [
        ("Bear+RSI<35+EV8-18+EPSGrowth>0+Analysts≥6",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 35) &
                     (d['ev_ebitda'] >= 8) & (d['ev_ebitda'] <= 18) &
                     (d['eps_growth'] > 0) & (d['analyst_count'] >= 6)]),
        ("Tier3+Cap>$2B+FCFGrowth+Fund≥18",
         lambda d: d[(d['signal_type'] == 'Tier 3') & (d['market_cap'] > 2e9) &
                     (d['fcf_positive'] == 1) & (d['fundamentals_score'] >= 18)]),
        ("LowTrend<10+Fund≥18+EV5-12+RevG4-30",
         lambda d: d[(d['trend_score'] < 10) & (d['fundamentals_score'] >= 18) &
                     (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 12) &
                     (d['rev_growth'] >= 4) & (d['rev_growth'] <= 30)]),
        ("DowngradeRecov+Bear+RSI<40+V2≥45",
         lambda d: d[(d['signal_type'] == 'Analyst Downgrade Recovery') &
                     (d['market_bearish'] == 1) & (d['rsi'] < 40) &
                     (d['value_score_v2'] >= 45)]),
        ("Tier1+Cap>$10B+EBITDAGrowth>10",
         lambda d: d[(d['signal_type'] == 'Tier 1') & (d['market_cap'] > 10e9) &
                     (d['ebitda_growth'] > 10)]),
        ("Bear+RSI<40+Fund≥15+V2≥40 (v1 proven)",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                     (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40)]),
        ("Tier3+Fund≥18+EV5-20 (v1 proven)",
         lambda d: d[(d['signal_type'] == 'Tier 3') & (d['fundamentals_score'] >= 18) &
                     (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20)]),
        ("Cap>$2B+Analysts≥6+FCF+LT≥45 (v1 proven)",
         lambda d: d[(d['market_cap'] > 2e9) & (d['analyst_count'] >= 6) &
                     (d['fcf_positive'] == 1) & (d['lt_score'] >= 45)]),
        ("Bear+RSI<35+Fund≥15+EV5-18+Cap>$1B",
         lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 35) &
                     (d['fundamentals_score'] >= 15) &
                     (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 18) &
                     (d['market_cap'] > 1e9)]),
        ("V2≥60+EV5-15+RevG10-30+Analysts≥6",
         lambda d: d[(d['value_score_v2'] >= 60) &
                     (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 15) &
                     (d['rev_growth'] >= 10) & (d['rev_growth'] <= 30) &
                     (d['analyst_count'] >= 6)]),
        ("EV5-12+RevG>15+EPSG>10+Cap>$2B",
         lambda d: d[(d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 12) &
                     (d['rev_growth'] > 15) & (d['eps_growth'] > 10) &
                     (d['market_cap'] > 2e9)]),
        ("Tier3+RSI<35+Cap>$1B+Analysts≥6",
         lambda d: d[(d['signal_type'] == 'Tier 3') & (d['rsi'] < 35) &
                     (d['market_cap'] > 1e9) & (d['analyst_count'] >= 6)]),
        ("Analysts≥8+DowngradeRecov+EPSGrowth>0 (v1)",
         lambda d: d[(d['signal_type'] == 'Analyst Downgrade Recovery') &
                     (d['eps_growth'] > 0) & (d['analyst_count'] >= 8)]),
        ("Bull+V2≥60+EV<18+RevG>10+Cap>$2B",
         lambda d: d[(d['market_bullish'] == 1) & (d['value_score_v2'] >= 60) &
                     (d['ev_ebitda'] > 0) & (d['ev_ebitda'] < 18) &
                     (d['rev_growth'] > 10) & (d['market_cap'] > 2e9)]),
        ("LT≥55+Fund≥18+RSI35-55+Analysts≥8",
         lambda d: d[(d['lt_score'] >= 55) & (d['fundamentals_score'] >= 18) &
                     (d['rsi'] >= 35) & (d['rsi'] <= 55) &
                     (d['analyst_count'] >= 8)]),
    ]

    # Walk-forward windows
    windows = [
        (2021, 2022, 2023, 2023),
        (2021, 2023, 2024, 2024),
        (2022, 2024, 2025, 2026),
    ]

    print("\n  --- Phase 1: Raw Performance ---")
    raw_rows = [stats_row("Baseline", df, 50)]
    for name, filt in combos:
        raw_rows.append(stats_row(name, filt(df), 20))
    print_stats_table(raw_rows, "All Elite Combos — Raw Performance", min_count=20, show_risk=True)

    # Phase 2: Walk-forward stability gate
    print("\n\n  --- Phase 2: Walk-Forward Stability Gate ---")
    print(f"  Requirement: Win% ≥ 52% in ALL 3 test windows (or N<20 exempted)")
    print(f"\n  {'Combo':<48s} {'Test 2023':>10s} {'Test 2024':>10s} {'Test 2025':>10s} {'Stable?':>8s}")
    print(f"  {'-'*90}")

    stable_combos = []
    for name, filt in combos:
        sub = filt(df)
        if len(sub) < 20:
            continue
        wins_by_window = []
        for ts, te, vs, ve in windows:
            test = sub[(sub['year'] >= vs) & (sub['year'] <= ve)]
            if len(test) < 10:
                wins_by_window.append(('N/A', True))  # exempt small windows
            else:
                w = (test['return_3m'] > 0).mean() * 100
                wins_by_window.append((f"{w:.1f}%", w >= 52))

        all_stable = all(ok for _, ok in wins_by_window)
        label = "PASS" if all_stable else "FAIL"
        vals = [v for v, _ in wins_by_window]
        print(f"  {name:<48s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {label:>7s}")

        if all_stable:
            stable_combos.append((name, filt))

    # Phase 3: Bootstrap CI gate on stable combos
    print(f"\n\n  --- Phase 3: Bootstrap CI Gate (95% CI lower ≥ 55%, N ≥ 200) ---")
    print(f"\n  {'Combo':<48s} {'N':>6s} {'Win%':>6s} {'CI Low':>7s} {'CI Hi':>7s} {'AvgRet':>7s} {'ELITE?':>7s}")
    print(f"  {'-'*90}")

    elite_combos = []
    for name, filt in stable_combos:
        sub = filt(df)
        returns = sub['return_3m'].clip(-90, 500).values
        if len(returns) < 30:
            continue
        win = (returns > 0).mean() * 100
        ci = bootstrap_ci(returns)
        is_elite = ci['win_lo'] >= 55 and len(returns) >= 200
        is_strong = ci['win_lo'] >= 52 and len(returns) >= 100
        label = "ELITE" if is_elite else ("STRONG" if is_strong else "no")
        print(f"  {name:<48s} {len(returns):>5,} {win:>5.1f}% {ci['win_lo']:>6.1f}% "
              f"{ci['win_hi']:>6.1f}% {returns.mean():>+6.1f}% {label:>6s}")
        if is_elite or is_strong:
            elite_combos.append((name, filt, label))

    # Phase 4: Rolling quarterly win rates for elite combos
    if elite_combos:
        print(f"\n\n  --- Phase 4: Quarterly Win Rates for Elite Combos ---")
        quarters = sorted(df['quarter'].unique())
        for name, filt, label in elite_combos:
            sub = filt(df)
            print(f"\n  {name} [{label}]:")
            print(f"  {'Quarter':<10s} {'Count':>6s} {'Win%':>7s} {'Avg3M':>8s} {'Med3M':>8s}")
            print(f"  {'-'*40}")
            for q in quarters:
                qd = sub[sub['quarter'] == q]
                if len(qd) < 5:
                    continue
                w = (qd['return_3m'] > 0).mean() * 100
                a = qd['return_3m'].clip(-90, 500).mean()
                m = qd['return_3m'].clip(-90, 500).median()
                print(f"  {str(q):<10s} {len(qd):>5,} {w:>6.1f}% {a:>+7.1f}% {m:>+7.1f}%")

    # Phase 5: Cap bucket breakdown for elite combos
    if elite_combos:
        print(f"\n\n  --- Phase 5: Performance by Market Cap Bucket ---")
        cap_buckets = [
            ('Micro (<$300M)', 0, 3e8),
            ('Small ($300M-$2B)', 3e8, 2e9),
            ('Mid ($2B-$10B)', 2e9, 10e9),
            ('Large ($10B-$100B)', 10e9, 100e9),
            ('Mega (>$100B)', 100e9, 1e15),
        ]
        for name, filt, label in elite_combos:
            sub = filt(df)
            if sub['market_cap'].isna().all():
                continue
            print(f"\n  {name} [{label}]:")
            print(f"  {'Cap Bucket':<25s} {'Count':>7s} {'Win%':>7s} {'Avg3M':>8s} {'Avg1Y':>8s}")
            print(f"  {'-'*55}")
            for bname, lo, hi in cap_buckets:
                bucket = sub[(sub['market_cap'] >= lo) & (sub['market_cap'] < hi)]
                if len(bucket) < 10:
                    continue
                w = (bucket['return_3m'] > 0).mean() * 100
                a = bucket['return_3m'].clip(-90, 500).mean()
                y = bucket['return_1y'].clip(-90, 1000).mean() if bucket['return_1y'].notna().sum() > 5 else np.nan
                ys = f"{y:+.1f}%" if pd.notna(y) else "N/A"
                print(f"  {bname:<25s} {len(bucket):>6,} {w:>6.1f}% {a:>+7.1f}% {ys:>7s}")

    return elite_combos


# ==================== 2. ML-WEIGHTED SCORING + XGBOOST WIN PROBABILITY ====================

def run_ml_scoring(df):
    print("\n\n" + "#" * 110)
    print("  SECTION 2: ML-WEIGHTED SCORING + XGBOOST WIN PROBABILITY")
    print("#" * 110)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.calibration import CalibratedClassifierCV
    except ImportError:
        print("\n  scikit-learn not installed. Skipping.")
        return

    features = ['lt_score', 'value_score_v2', 'trend_score', 'fundamentals_score',
                'valuation_score', 'momentum_score', 'market_risk_score',
                'rsi', 'adx', 'ev_ebitda_clean', 'rev_growth', 'eps_growth',
                'ebitda_growth']

    ml_df = df[features + ['return_3m', 'return_1y', 'year']].dropna(subset=features + ['return_3m'])
    X = ml_df[features].values
    y = (ml_df['return_3m'] > 0).astype(int).values

    print(f"\n  Dataset: {len(ml_df):,} signals | {y.mean()*100:.1f}% positive")

    # --- 2A: ML-Importance-Weighted Score ---
    print("\n  --- 2A: ML-Importance-Weighted Score Test ---")

    # Train GB to get importances
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=4, min_samples_leaf=50,
                                     learning_rate=0.05, random_state=42)
    gb.fit(X, y)
    imp = pd.Series(gb.feature_importances_, index=features)

    # Current V2 scoring: roughly equal sub-score weights
    # Proposed: weight by ML importance (normalized to sum=100)
    imp_norm = (imp / imp.sum() * 100).round(1)
    print(f"\n  ML-Derived Feature Weights (sum=100):")
    for feat, wt in imp_norm.sort_values(ascending=False).items():
        bar = '█' * int(wt / 2)
        print(f"    {feat:<25s} {wt:>5.1f}  {bar}")

    # Create ML-weighted composite score
    imp_weights = imp / imp.sum()
    # Normalize each feature to 0-1 range, then weight
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    ml_composite = (X_scaled * imp_weights.values).sum(axis=1) * 100

    ml_df = ml_df.copy()
    ml_df['ml_score'] = ml_composite

    # Compare: ML score buckets vs V2 score buckets
    print(f"\n  ML Score vs V2 Score — Quintile Comparison:")
    print(f"  {'Score':<12s} {'Quintile':<12s} {'Count':>7s} {'Win%':>7s} {'Avg3M':>8s} {'Med3M':>8s}")
    print(f"  {'-'*55}")
    for score_name, score_col in [('V2 Score', 'value_score_v2'), ('ML Score', 'ml_score')]:
        try:
            quintiles = pd.qcut(ml_df[score_col], q=5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                duplicates='drop')
        except ValueError:
            continue
        for q_label in ['Q1(low)', 'Q3', 'Q5(high)']:
            mask = quintiles == q_label
            sub = ml_df[mask]
            if len(sub) < 20:
                continue
            w = (sub['return_3m'] > 0).mean() * 100
            a = sub['return_3m'].clip(-90, 500).mean()
            m = sub['return_3m'].clip(-90, 500).median()
            print(f"  {score_name:<12s} {q_label:<12s} {len(sub):>6,} {w:>6.1f}% {a:>+7.1f}% {m:>+7.1f}%")

    # --- 2B: Regime-Dependent Optimal Thresholds ---
    print(f"\n\n  --- 2B: Regime-Dependent Bucket Optimization ---")
    for regime_name, regime_mask in [('Bull', ml_df['market_risk_score'] == 10),
                                      ('Bear', ml_df['market_risk_score'] == 0)]:
        regime = ml_df[regime_mask]
        if len(regime) < 200:
            continue
        print(f"\n  {regime_name} Market ({len(regime):,} signals):")
        for feat in ['ev_ebitda_clean', 'rev_growth', 'rsi', 'eps_growth']:
            feat_data = regime[[feat, 'return_3m']].dropna()
            if len(feat_data) < 100:
                continue
            try:
                bins = pd.qcut(feat_data[feat], q=5, duplicates='drop')
                grouped = feat_data.groupby(bins)['return_3m'].agg(
                    count='count', win_rate=lambda x: (x > 0).mean() * 100,
                    avg_ret=lambda x: x.clip(-90, 500).mean()
                )
                best_bucket = grouped.loc[grouped['win_rate'].idxmax()]
                print(f"    {feat:<20s} Best: {grouped['win_rate'].idxmax()} → "
                      f"{best_bucket['win_rate']:.1f}% win, {best_bucket['avg_ret']:+.1f}% avg "
                      f"(n={int(best_bucket['count'])})")
            except Exception:
                continue

    # --- 2C: XGBoost Calibrated Win Probability ---
    print(f"\n\n  --- 2C: XGBoost Calibrated Win Probability Predictor ---")

    # Train/test split: 2021-2024 train, 2025+ test
    train_mask = ml_df['year'] <= 2024
    test_mask = ml_df['year'] >= 2025

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train):,} (2021-2024) | Test: {len(X_test):,} (2025+)")

    if len(X_test) < 100:
        print("  Insufficient test data, using full CV instead")
        scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
        print(f"  5-Fold CV Accuracy: {scores.mean()*100:.1f}% (±{scores.std()*100:.1f}%)")

        # Calibrate on full data
        cal = CalibratedClassifierCV(gb, cv=3, method='isotonic')
        cal.fit(X, y)
        probs = cal.predict_proba(X)[:, 1]
    else:
        gb_oos = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                             min_samples_leaf=50, learning_rate=0.05,
                                             random_state=42)
        gb_oos.fit(X_train, y_train)
        train_acc = (gb_oos.predict(X_train) == y_train).mean() * 100
        test_acc = (gb_oos.predict(X_test) == y_test).mean() * 100
        print(f"  Train Accuracy: {train_acc:.1f}% | Test Accuracy: {test_acc:.1f}%")

        # Calibrate probabilities
        cal = CalibratedClassifierCV(gb_oos, cv=3, method='isotonic')
        cal.fit(X_train, y_train)
        probs = cal.predict_proba(X_test)[:, 1]

        # Decile analysis on test set
        test_df = ml_df[test_mask].copy()
        test_df['win_prob'] = probs

        print(f"\n  Win Probability Decile Analysis (Out-of-Sample 2025+):")
        print(f"  {'Decile':<12s} {'Prob Range':<16s} {'Count':>6s} {'Actual Win%':>12s} {'Avg3M':>8s} {'Med3M':>8s}")
        print(f"  {'-'*65}")
        try:
            deciles = pd.qcut(test_df['win_prob'], q=10, duplicates='drop')
            for dec in sorted(test_df.groupby(deciles).groups.keys()):
                sub = test_df[deciles == dec]
                w = (sub['return_3m'] > 0).mean() * 100
                a = sub['return_3m'].clip(-90, 500).mean()
                m = sub['return_3m'].clip(-90, 500).median()
                print(f"  {'D'+str(list(sorted(test_df.groupby(deciles).groups.keys())).index(dec)+1):<12s} "
                      f"{str(dec):<16s} {len(sub):>5,} {w:>11.1f}% {a:>+7.1f}% {m:>+7.1f}%")
        except Exception as e:
            # Simpler approach
            test_df['prob_bucket'] = pd.cut(test_df['win_prob'], bins=5, duplicates='drop')
            for bucket in sorted(test_df['prob_bucket'].dropna().unique()):
                sub = test_df[test_df['prob_bucket'] == bucket]
                if len(sub) < 10:
                    continue
                w = (sub['return_3m'] > 0).mean() * 100
                a = sub['return_3m'].clip(-90, 500).mean()
                m = sub['return_3m'].clip(-90, 500).median()
                print(f"  {str(bucket):<28s} {len(sub):>5,} {w:>11.1f}% {a:>+7.1f}% {m:>+7.1f}%")

        # "ML Elite" trigger: top decile
        top_prob = test_df['win_prob'].quantile(0.9)
        ml_elite = test_df[test_df['win_prob'] >= top_prob]
        if len(ml_elite) >= 10:
            w = (ml_elite['return_3m'] > 0).mean() * 100
            a = ml_elite['return_3m'].clip(-90, 500).mean()
            print(f"\n  ** ML Elite Trigger (top 10%, prob ≥ {top_prob:.3f}):")
            print(f"     Count: {len(ml_elite)} | Win%: {w:.1f}% | Avg 3M: {a:+.1f}%")

            # Compare to Tier 1, Tier 3 in same period
            for sig in ['Tier 1', 'Tier 3', 'Regime Buy Bear']:
                sig_test = ml_df[test_mask & (df.loc[ml_df.index, 'signal_type'] == sig)]
                if len(sig_test) >= 10:
                    sw = (sig_test['return_3m'] > 0).mean() * 100
                    sa = sig_test['return_3m'].clip(-90, 500).mean()
                    print(f"     vs {sig}: {len(sig_test)} signals | Win%: {sw:.1f}% | Avg 3M: {sa:+.1f}%")


# ==================== 3. REGIME-ADAPTIVE + SECTOR/CAP BREAKDOWNS ====================

def run_regime_sector_analysis(df):
    print("\n\n" + "#" * 110)
    print("  SECTION 3: REGIME-ADAPTIVE FILTERS + SECTOR/CAP BREAKDOWNS")
    print("#" * 110)

    # --- 3A: Sector breakdown for key signals ---
    print("\n  --- 3A: Signal Performance by Sector ---")
    key_signals = ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear', 'Regime Buy Bull']
    top_sectors = df['sector'].value_counts().head(8).index.tolist()

    for sig in key_signals:
        sig_data = df[df['signal_type'] == sig]
        if len(sig_data) < 50:
            continue
        print(f"\n  {sig} ({len(sig_data):,} total):")
        print(f"  {'Sector':<25s} {'Count':>6s} {'Win%':>6s} {'Avg3M':>7s} {'Avg1Y':>7s}")
        print(f"  {'-'*55}")

        sector_stats = []
        for sector in top_sectors:
            sub = sig_data[sig_data['sector'] == sector]
            if len(sub) < 15:
                continue
            w = (sub['return_3m'] > 0).mean() * 100
            a = sub['return_3m'].clip(-90, 500).mean()
            y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 5 else np.nan
            sector_stats.append((sector, len(sub), w, a, y))

        sector_stats.sort(key=lambda x: x[2], reverse=True)
        for sector, n, w, a, y_val in sector_stats:
            ys = f"{y_val:+.1f}%" if pd.notna(y_val) else "  N/A"
            print(f"  {sector:<25s} {n:>5,} {w:>5.1f}% {a:>+6.1f}% {ys:>6s}")

    # --- 3B: Adaptive regime thresholds with quality gate ---
    print(f"\n\n  --- 3B: Regime-Adaptive Strategy with Quality Gate ---")
    print(f"  Quality gate: Cap>$2B + Analysts≥6 + FCF Growth")

    quality = df[(df['market_cap'] > 2e9) & (df['analyst_count'] >= 6) & (df['fcf_positive'] == 1)]
    bull_q = quality[quality['market_bullish'] == 1]
    bear_q = quality[quality['market_bearish'] == 1]

    print(f"\n  Quality-gated: {len(quality):,} signals (Bull: {len(bull_q):,}, Bear: {len(bear_q):,})")

    # Bull adaptive
    bull_strategies = [
        ("Bull+Quality (baseline)", bull_q),
        ("Bull+Quality+V2≥60", bull_q[bull_q['value_score_v2'] >= 60]),
        ("Bull+Quality+V2≥60+EV<18", bull_q[(bull_q['value_score_v2'] >= 60) &
                                             (bull_q['ev_ebitda'] > 0) & (bull_q['ev_ebitda'] < 18)]),
        ("Bull+Quality+LT≥55+Fund≥18", bull_q[(bull_q['lt_score'] >= 55) &
                                                (bull_q['fundamentals_score'] >= 18)]),
        ("Bull+Quality+RevG15-30+EV<12", bull_q[(bull_q['rev_growth'] >= 15) & (bull_q['rev_growth'] <= 30) &
                                                  (bull_q['ev_ebitda'] > 0) & (bull_q['ev_ebitda'] < 12)]),
    ]

    bear_strategies = [
        ("Bear+Quality (baseline)", bear_q),
        ("Bear+Quality+RSI<40", bear_q[bear_q['rsi'] < 40]),
        ("Bear+Quality+RSI<35+Fund≥15", bear_q[(bear_q['rsi'] < 35) & (bear_q['fundamentals_score'] >= 15)]),
        ("Bear+Quality+V2≥40+RSI<40", bear_q[(bear_q['value_score_v2'] >= 40) & (bear_q['rsi'] < 40)]),
        ("Bear+Quality+LT≥45+EPSG>8+RSI<40", bear_q[(bear_q['lt_score'] >= 45) & (bear_q['eps_growth'] > 8) &
                                                       (bear_q['rsi'] < 40)]),
    ]

    for label, strats in [("BULL", bull_strategies), ("BEAR", bear_strategies)]:
        rows = []
        for name, sub in strats:
            rows.append(stats_row(name, sub, 10))
        print_stats_table(rows, f"{label} Market — Quality-Gated Strategies", min_count=10, show_risk=True)

    # --- 3C: ADX volatility avoidance ---
    print(f"\n\n  --- 3C: ADX (Volatility) Filter Impact ---")
    for sig in ['Tier 1', 'Tier 3', 'Buy_A']:
        sig_data = df[df['signal_type'] == sig]
        if len(sig_data) < 100:
            continue
        rows = []
        for adx_label, lo, hi in [('ADX < 20 (calm)', 0, 20), ('ADX 20-30', 20, 30),
                                    ('ADX 30-40 (trending)', 30, 40), ('ADX > 40 (volatile)', 40, 200)]:
            sub = sig_data[(sig_data['adx'] >= lo) & (sig_data['adx'] < hi)]
            rows.append(stats_row(f"{sig} + {adx_label}", sub, 15))
        print_stats_table(rows, f"{sig} — Split by ADX Volatility", min_count=15)


# ==================== 4. STRESS TESTS, RISK-ADJUSTED METRICS, OOS HOLDOUT ====================

def run_stress_tests(df):
    print("\n\n" + "#" * 110)
    print("  SECTION 4: STRESS TESTS, RISK-ADJUSTED METRICS, OOS HOLDOUT")
    print("#" * 110)

    # --- 4A: Stress test — 2022 Bear vs 2023 Bull ---
    print("\n  --- 4A: Stress Tests (2022 Bear Market vs 2023 Bull Recovery) ---")
    bear_2022 = df[(df['year'] == 2022)]
    bull_2023 = df[(df['year'] == 2023)]

    key_sigs = ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear', 'Buy_A', 'Buy_B',
                'Analyst Downgrade Recovery', 'High Coverage Buy']

    print(f"\n  {'Signal':<30s} │ {'2022 Bear':^30s} │ {'2023 Bull':^30s}")
    print(f"  {'':<30s} │ {'N':>6s} {'Win%':>6s} {'Avg':>7s} {'MaxDD':>7s} │ "
          f"{'N':>6s} {'Win%':>6s} {'Avg':>7s} {'MaxDD':>7s}")
    print(f"  {'-'*95}")

    for sig in key_sigs:
        b22 = bear_2022[bear_2022['signal_type'] == sig]
        b23 = bull_2023[bull_2023['signal_type'] == sig]
        if len(b22) < 10 and len(b23) < 10:
            continue

        def _stats(sub):
            if len(sub) < 5:
                return "   -  ", "   - ", "    - ", "    - "
            ret = sub['return_3m'].clip(-90, 500)
            w = (ret > 0).mean() * 100
            a = ret.mean()
            max_dd = ret.min()
            return f"{len(sub):>5,}", f"{w:>5.1f}%", f"{a:>+6.1f}%", f"{max_dd:>+6.1f}%"

        n22, w22, a22, d22 = _stats(b22)
        n23, w23, a23, d23 = _stats(b23)
        print(f"  {sig:<30s} │ {n22} {w22} {a22} {d22} │ {n23} {w23} {a23} {d23}")

    # --- 4B: Risk-adjusted metrics for all key signals ---
    print(f"\n\n  --- 4B: Risk-Adjusted Metrics (Full Dataset) ---")
    print(f"  {'Signal':<30s} {'N':>6s} {'Win%':>6s} {'AvgWin':>7s} {'AvgLoss':>8s} {'PF':>5s} "
          f"{'Sharpe*':>7s} {'MaxDD':>7s} {'CalmarR':>8s}")
    print(f"  {'-'*90}")
    print(f"  * Sharpe approximation: mean/std of 3M returns")

    all_sigs = ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear', 'Regime Buy Bull',
                'Buy_A', 'Buy_B', 'Analyst Downgrade Recovery', 'High Coverage Buy']

    for sig in all_sigs:
        sub = df[(df['signal_type'] == sig) & (df['return_3m'].notna())]
        if len(sub) < 30:
            continue
        ret = sub['return_3m'].clip(-90, 500)
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        w = (ret > 0).mean() * 100
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        pf = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else np.inf
        sharpe = ret.mean() / ret.std() if ret.std() > 0 else 0
        max_dd = ret.min()
        calmar = ret.mean() / abs(max_dd) if abs(max_dd) > 0 else 0
        print(f"  {sig:<30s} {len(sub):>5,} {w:>5.1f}% {avg_win:>+6.1f}% {avg_loss:>+7.1f}% "
              f"{pf:>4.2f} {sharpe:>6.3f} {max_dd:>+6.1f}% {calmar:>7.3f}")

    # --- 4C: Out-of-sample holdout (train 2021-2024, test 2025) ---
    print(f"\n\n  --- 4C: Out-of-Sample Holdout (Train: 2021-2024, Test: 2025+) ---")
    train = df[df['year'] <= 2024]
    test = df[df['year'] >= 2025]
    print(f"  Train: {len(train):,} signals | Test: {len(test):,} signals")

    if len(test) >= 100:
        print(f"\n  {'Signal':<30s} {'Train N':>8s} {'Train Win%':>10s} {'Test N':>8s} "
              f"{'Test Win%':>10s} {'Δ':>6s} {'Test Avg3M':>10s} {'Verdict':>8s}")
        print(f"  {'-'*95}")

        for sig in all_sigs:
            tr = train[train['signal_type'] == sig]
            te = test[test['signal_type'] == sig]
            if len(tr) < 20 or len(te) < 10:
                continue
            tr_w = (tr['return_3m'] > 0).mean() * 100
            te_w = (te['return_3m'] > 0).mean() * 100
            delta = te_w - tr_w
            te_avg = te['return_3m'].clip(-90, 500).mean()
            verdict = "STABLE" if abs(delta) < 5 else ("OK" if abs(delta) < 10 else "DECAY")
            print(f"  {sig:<30s} {len(tr):>7,} {tr_w:>9.1f}% {len(te):>7,} "
                  f"{te_w:>9.1f}% {delta:>+5.1f}% {te_avg:>+9.1f}% {verdict:>7s}")

    # --- 4D: Simple position sizing simulation ---
    print(f"\n\n  --- 4D: Position Sizing Simulation (2% base, scale by win probability) ---")
    print(f"  Scenario: $100K portfolio, 2% position ($2K per trade)")
    print(f"  Test period: 2025+ only (true out-of-sample)")

    if len(test) >= 50:
        for sig in ['Tier 3', 'Regime Buy Bear', 'Tier 1']:
            sig_test = test[test['signal_type'] == sig]
            if len(sig_test) < 10:
                continue
            portfolio = 100000
            position_size = 2000  # 2% of 100K
            total_pnl = 0
            trades = 0
            wins = 0
            for _, row in sig_test.iterrows():
                ret = row['return_3m']
                if pd.notna(ret):
                    pnl = position_size * (ret / 100)
                    total_pnl += pnl
                    trades += 1
                    if ret > 0:
                        wins += 1

            if trades > 0:
                win_rate = wins / trades * 100
                total_return = total_pnl / portfolio * 100
                avg_pnl = total_pnl / trades
                print(f"\n  {sig}:")
                print(f"    Trades: {trades} | Win Rate: {win_rate:.1f}%")
                print(f"    Total P&L: ${total_pnl:,.0f} ({total_return:+.1f}% on portfolio)")
                print(f"    Avg P&L/trade: ${avg_pnl:,.0f}")


# ==================== 5. ANALYST FIRM ACCURACY SCORING ====================

def run_analyst_accuracy(df_signals=None):
    print("\n\n" + "#" * 110)
    print("  SECTION 5: ANALYST FIRM ACCURACY SCORING")
    print("#" * 110)

    conn = sqlite3.connect(BACKTEST_DB)

    # Load grades
    grades = pd.read_sql_query("""
        SELECT g.symbol, g.date, g.grading_company, g.previous_grade, g.new_grade, g.action
        FROM historical_grades g
        WHERE g.date >= '2019-01-01'
    """, conn)

    # Load daily prices for return computation
    print(f"\n  Loading price data for forward return computation...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, close
        FROM historical_prices
        WHERE date >= '2019-01-01'
        ORDER BY symbol, date
    """, conn)
    conn.close()

    # Load sectors
    conn2 = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query(
        "SELECT symbol, sector, industry FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    print(f"  Loaded {len(grades):,} grades and {len(prices):,} price records")

    # Pivot prices to get quick lookup
    prices['date'] = pd.to_datetime(prices['date'])
    prices = prices.sort_values(['symbol', 'date'])

    # For each symbol, create a price series indexed by date
    # More efficient: merge grades with prices at grade date, +1m, +3m, +6m, +1y
    grades['date'] = pd.to_datetime(grades['date'])

    # Get price at grade date and forward dates
    # Create date offset columns
    from dateutil.relativedelta import relativedelta

    print(f"  Computing forward returns for each grade action...")

    # Build price lookup: for each symbol, dict of date -> close
    price_dict = {}
    for symbol, group in prices.groupby('symbol'):
        price_dict[symbol] = dict(zip(group['date'], group['close']))

    def get_nearest_price(symbol, target_date, window_days=5):
        """Get price on or near target date."""
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

    # Compute forward returns for grades
    results = []
    total = len(grades)
    for idx, row in grades.iterrows():
        if idx % 50000 == 0 and idx > 0:
            print(f"    Processing grade {idx:,}/{total:,}...")

        sym = row['symbol']
        dt = row['date']
        p0 = get_nearest_price(sym, dt)
        if p0 is None or p0 <= 0:
            continue

        p_1m = get_nearest_price(sym, dt + pd.Timedelta(days=30))
        p_3m = get_nearest_price(sym, dt + pd.Timedelta(days=91))
        p_6m = get_nearest_price(sym, dt + pd.Timedelta(days=182))
        p_1y = get_nearest_price(sym, dt + pd.Timedelta(days=365))

        r_1m = ((p_1m / p0) - 1) * 100 if p_1m else None
        r_3m = ((p_3m / p0) - 1) * 100 if p_3m else None
        r_6m = ((p_6m / p0) - 1) * 100 if p_6m else None
        r_1y = ((p_1y / p0) - 1) * 100 if p_1y else None

        results.append({
            'symbol': sym, 'date': dt,
            'company': row['grading_company'],
            'action': row['action'],
            'new_grade': row['new_grade'],
            'r_1m': r_1m, 'r_3m': r_3m, 'r_6m': r_6m, 'r_1y': r_1y,
        })

    gdf = pd.DataFrame(results)
    gdf['sector'] = gdf['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None

    # Map grades to directional signals
    bullish_grades = {'Buy', 'Strong Buy', 'Overweight', 'Outperform', 'Market Outperform',
                      'Sector Outperform', 'Positive', 'Strong Buy'}
    bearish_grades = {'Sell', 'Underperform', 'Underweight', 'Reduce'}
    neutral_grades = {'Neutral', 'Equal Weight', 'Hold', 'Market Perform', 'Sector Perform',
                      'Perform', 'In Line', 'Peer Perform', 'Sector Weight'}

    def grade_direction(grade):
        if grade in bullish_grades:
            return 'Bullish'
        elif grade in bearish_grades:
            return 'Bearish'
        elif grade in neutral_grades:
            return 'Neutral'
        return 'Other'

    gdf['direction'] = gdf['new_grade'].map(grade_direction)

    print(f"\n  Matched {len(gdf):,} grades with price data")
    print(f"  Actions: {dict(gdf['action'].value_counts())}")
    print(f"  Directions: {dict(gdf['direction'].value_counts())}")

    # --- 5A: Overall firm accuracy (upgrades should go up, downgrades should go down) ---
    print(f"\n\n  --- 5A: Top 25 Firms — Directional Accuracy (Upgrades & Downgrades) ---")

    # Focus on firms with enough data
    firm_counts = gdf['company'].value_counts()
    major_firms = firm_counts[firm_counts >= 500].index.tolist()

    print(f"\n  Scoring method:")
    print(f"    Upgrade accuracy  = % of upgrades where stock rose 3M later")
    print(f"    Downgrade accuracy = % of downgrades where stock fell 3M later")
    print(f"    Composite score   = (upgrade_acc + downgrade_acc) / 2")

    firm_scores = []
    for firm in major_firms:
        firm_data = gdf[gdf['company'] == firm]

        # Upgrades
        upgrades = firm_data[(firm_data['action'] == 'upgrade') & (firm_data['r_3m'].notna())]
        up_acc = (upgrades['r_3m'] > 0).mean() * 100 if len(upgrades) >= 20 else np.nan
        up_avg = upgrades['r_3m'].clip(-90, 500).mean() if len(upgrades) >= 20 else np.nan

        # Downgrades
        downgrades = firm_data[(firm_data['action'] == 'downgrade') & (firm_data['r_3m'].notna())]
        dn_acc = (downgrades['r_3m'] < 0).mean() * 100 if len(downgrades) >= 20 else np.nan
        dn_avg = downgrades['r_3m'].clip(-90, 500).mean() if len(downgrades) >= 20 else np.nan

        # Bullish calls
        bullish = firm_data[(firm_data['direction'] == 'Bullish') & (firm_data['r_3m'].notna())]
        bull_acc = (bullish['r_3m'] > 0).mean() * 100 if len(bullish) >= 30 else np.nan

        # Bearish calls
        bearish_calls = firm_data[(firm_data['direction'] == 'Bearish') & (firm_data['r_3m'].notna())]
        bear_acc = (bearish_calls['r_3m'] < 0).mean() * 100 if len(bearish_calls) >= 10 else np.nan

        composite = np.nanmean([up_acc, dn_acc]) if pd.notna(up_acc) and pd.notna(dn_acc) else np.nan

        firm_scores.append({
            'firm': firm,
            'total_grades': len(firm_data),
            'upgrades': len(upgrades), 'up_acc': up_acc, 'up_avg_3m': up_avg,
            'downgrades': len(downgrades), 'dn_acc': dn_acc, 'dn_avg_3m': dn_avg,
            'bullish_n': len(bullish), 'bull_acc': bull_acc,
            'bearish_n': len(bearish_calls), 'bear_acc': bear_acc,
            'composite': composite,
        })

    fdf = pd.DataFrame(firm_scores).sort_values('composite', ascending=False)

    print(f"\n  {'Firm':<28s} {'Total':>6s} │ {'Up N':>5s} {'Up Acc':>7s} {'Up Avg':>7s} │ "
          f"{'Dn N':>5s} {'Dn Acc':>7s} {'Dn Avg':>7s} │ {'Score':>6s}")
    print(f"  {'-'*105}")

    for _, r in fdf.head(25).iterrows():
        up_a = f"{r['up_acc']:.1f}%" if pd.notna(r['up_acc']) else "  N/A"
        up_r = f"{r['up_avg_3m']:+.1f}%" if pd.notna(r['up_avg_3m']) else "   N/A"
        dn_a = f"{r['dn_acc']:.1f}%" if pd.notna(r['dn_acc']) else "  N/A"
        dn_r = f"{r['dn_avg_3m']:+.1f}%" if pd.notna(r['dn_avg_3m']) else "   N/A"
        comp = f"{r['composite']:.1f}%" if pd.notna(r['composite']) else "  N/A"
        print(f"  {r['firm']:<28s} {r['total_grades']:>5,} │ {r['upgrades']:>4,} {up_a:>7s} {up_r:>7s} │ "
              f"{r['downgrades']:>4,} {dn_a:>7s} {dn_r:>7s} │ {comp:>6s}")

    # --- 5B: Bullish rating accuracy (when they say Buy, does it go up?) ---
    print(f"\n\n  --- 5B: Bullish Rating Accuracy (All Buy/Overweight/Outperform calls) ---")
    print(f"  {'Firm':<28s} {'Bullish N':>9s} {'3M Win%':>8s} {'Avg 3M':>7s} {'6M Win%':>8s} {'1Y Win%':>8s}")
    print(f"  {'-'*70}")

    bull_scores = []
    for firm in major_firms:
        fd = gdf[(gdf['company'] == firm) & (gdf['direction'] == 'Bullish')]
        if len(fd) < 30:
            continue
        r3 = fd['r_3m'].dropna().clip(-90, 500)
        r6 = fd['r_6m'].dropna().clip(-90, 500)
        r1y = fd['r_1y'].dropna().clip(-90, 1000)
        w3 = (r3 > 0).mean() * 100 if len(r3) >= 20 else np.nan
        w6 = (r6 > 0).mean() * 100 if len(r6) >= 20 else np.nan
        w1y = (r1y > 0).mean() * 100 if len(r1y) >= 20 else np.nan
        avg3 = r3.mean() if len(r3) >= 20 else np.nan
        bull_scores.append({'firm': firm, 'n': len(fd), 'w3': w3, 'avg3': avg3, 'w6': w6, 'w1y': w1y})

    bdf = pd.DataFrame(bull_scores).sort_values('w3', ascending=False)
    for _, r in bdf.head(25).iterrows():
        w3s = f"{r['w3']:.1f}%" if pd.notna(r['w3']) else "  N/A"
        a3s = f"{r['avg3']:+.1f}%" if pd.notna(r['avg3']) else "   N/A"
        w6s = f"{r['w6']:.1f}%" if pd.notna(r['w6']) else "  N/A"
        w1ys = f"{r['w1y']:.1f}%" if pd.notna(r['w1y']) else "  N/A"
        print(f"  {r['firm']:<28s} {r['n']:>8,} {w3s:>8s} {a3s:>7s} {w6s:>8s} {w1ys:>8s}")

    # --- 5C: Sector expertise — which firms are best at which sectors ---
    print(f"\n\n  --- 5C: Sector Expertise — Best Firm per Sector ---")
    print(f"  (Based on 3M directional accuracy of upgrade calls)")

    top_sectors = gdf['sector'].value_counts().head(10).index.tolist()
    # Top 15 firms by volume
    top_firms = fdf.head(15)['firm'].tolist()

    print(f"\n  {'Sector':<22s} │ {'Best Firm':<28s} {'Up Acc':>7s} {'N':>5s} │ "
          f"{'2nd Best':<28s} {'Up Acc':>7s} {'N':>5s}")
    print(f"  {'-'*110}")

    sector_expertise = {}
    for sector in top_sectors:
        sector_data = gdf[(gdf['sector'] == sector) & (gdf['action'] == 'upgrade') &
                          (gdf['r_3m'].notna())]
        firm_sector_scores = []
        for firm in top_firms:
            fs = sector_data[sector_data['company'] == firm]
            if len(fs) < 10:
                continue
            acc = (fs['r_3m'] > 0).mean() * 100
            firm_sector_scores.append((firm, acc, len(fs)))

        firm_sector_scores.sort(key=lambda x: x[1], reverse=True)
        if len(firm_sector_scores) >= 2:
            f1, a1, n1 = firm_sector_scores[0]
            f2, a2, n2 = firm_sector_scores[1]
            print(f"  {sector:<22s} │ {f1:<28s} {a1:>6.1f}% {n1:>4,} │ "
                  f"{f2:<28s} {a2:>6.1f}% {n2:>4,}")
            sector_expertise[sector] = firm_sector_scores
        elif len(firm_sector_scores) == 1:
            f1, a1, n1 = firm_sector_scores[0]
            print(f"  {sector:<22s} │ {f1:<28s} {a1:>6.1f}% {n1:>4,} │ {'(no 2nd)':<28s}")

    # --- 5D: Firm accuracy over time (are they getting better or worse?) ---
    print(f"\n\n  --- 5D: Firm Accuracy Trend (Yearly Upgrade 3M Win Rate) ---")
    top_5_firms = fdf.head(5)['firm'].tolist()

    gdf['grade_year'] = gdf['date'].dt.year
    years = sorted(gdf['grade_year'].unique())
    years = [y for y in years if y >= 2020]

    print(f"\n  {'Firm':<28s}", end='')
    for y in years:
        print(f" {y:>6d}", end='')
    print(f" {'Trend':>8s}")
    print(f"  {'-'*(28 + 7*len(years) + 8)}")

    for firm in top_5_firms:
        fd = gdf[(gdf['company'] == firm) & (gdf['action'] == 'upgrade') & (gdf['r_3m'].notna())]
        print(f"  {firm:<28s}", end='')
        yearly_accs = []
        for y in years:
            fy = fd[fd['grade_year'] == y]
            if len(fy) >= 10:
                acc = (fy['r_3m'] > 0).mean() * 100
                print(f" {acc:>5.1f}%", end='')
                yearly_accs.append(acc)
            else:
                print(f"    N/A", end='')
        # Trend
        if len(yearly_accs) >= 3:
            recent = np.mean(yearly_accs[-2:])
            early = np.mean(yearly_accs[:2])
            trend = "↑ Better" if recent > early + 3 else ("↓ Worse" if recent < early - 3 else "→ Stable")
        else:
            trend = "N/A"
        print(f" {trend:>8s}")

    # --- 5E: Full heatmap — Firms x Sectors ---
    print(f"\n\n  --- 5E: Upgrade Accuracy Heatmap (Top 10 Firms × Top 8 Sectors) ---")
    print(f"  (Upgrade 3M Win Rate | blank = insufficient data)")

    top_10_firms = fdf.head(10)['firm'].tolist()
    top_8_sectors = gdf['sector'].value_counts().head(8).index.tolist()

    # Header
    print(f"\n  {'Firm':<24s}", end='')
    for s in top_8_sectors:
        abbrev = s[:10]
        print(f" {abbrev:>10s}", end='')
    print(f" {'Overall':>8s}")
    print(f"  {'-'*(24 + 11*len(top_8_sectors) + 8)}")

    for firm in top_10_firms:
        fd = gdf[(gdf['company'] == firm) & (gdf['action'] == 'upgrade') & (gdf['r_3m'].notna())]
        overall = (fd['r_3m'] > 0).mean() * 100 if len(fd) >= 20 else np.nan
        print(f"  {firm:<24s}", end='')
        for s in top_8_sectors:
            fs = fd[fd['sector'] == s]
            if len(fs) >= 10:
                acc = (fs['r_3m'] > 0).mean() * 100
                # Color coding (text markers)
                marker = '★' if acc >= 60 else ('●' if acc >= 55 else '○')
                print(f" {acc:>5.1f}%{marker:>3s}", end='')
            else:
                print(f"       -  ", end='')
        ov = f"{overall:.1f}%" if pd.notna(overall) else "N/A"
        print(f" {ov:>7s}")

    print(f"\n  Legend: ★ = ≥60% accuracy (excellent), ● = ≥55% (good), ○ = <55%")

    return fdf


# ==================== MAIN ====================

def main():
    start = datetime.now()

    print("=" * 110)
    print("  ELITE BACKTEST ANALYSIS SUITE")
    print(f"  Databases: {BACKTEST_DB}, {MAIN_DB}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 110)

    print("\n  Loading signals with full indicator context...")
    df = load_signals_with_context()
    print(f"  Loaded {len(df):,} signals across {df['symbol'].nunique():,} stocks")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print("\n  Enriching with market cap, FCF, analyst coverage, sector data...")
    df = load_supplementary(df)
    enriched = df['market_cap'].notna().sum()
    sector_matched = df['sector'].notna().sum()
    print(f"  Market cap matched: {enriched:,} | Sector matched: {sector_matched:,}")

    print("\n  Running Section 1: Elite Multi-Factor Combos...")
    elite = run_elite_combos(df)

    print("\n  Running Section 2: ML Scoring + Win Probability...")
    run_ml_scoring(df)

    print("\n  Running Section 3: Regime-Adaptive + Sector Breakdowns...")
    run_regime_sector_analysis(df)

    print("\n  Running Section 4: Stress Tests + Risk-Adjusted Metrics...")
    run_stress_tests(df)

    print("\n  Running Section 5: Analyst Firm Accuracy Scoring...")
    run_analyst_accuracy(df)

    elapsed = datetime.now() - start
    print(f"\n\n{'='*110}")
    print(f"  Analysis complete! Runtime: {elapsed}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()

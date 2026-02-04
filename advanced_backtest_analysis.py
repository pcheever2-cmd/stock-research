#!/usr/bin/env python3
"""
Advanced Backtest Analysis Suite
================================
1. Multi-Factor Interaction / Conditional Analysis
2. Walk-Forward Rolling Validation
3. Bootstrap Confidence Intervals
4. Feature Importance (Random Forest / XGBoost)
5. New High-Conviction Filter Tests
6. Regime-Adaptive Threshold Analysis

Runs against existing backtest.db — no new data collection needed.
"""

import sqlite3
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

BACKTEST_DB = str(Path(__file__).parent / 'backtest.db')

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

    # Derived features
    df['market_bullish'] = (df['market_risk_score'] == 10).astype(int)
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['winner_3m'] = (df['return_3m'] > 0).astype(int)
    df['winner_1y'] = (df['return_1y'] > 0).astype(int) if 'return_1y' in df.columns else np.nan

    # Clean EV/EBITDA for analysis (cap outliers)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)

    return df


def load_daily_scores():
    """Load all daily scores for walk-forward analysis."""
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("""
        SELECT * FROM backtest_daily_scores
        WHERE lt_score IS NOT NULL
        ORDER BY date
    """, conn)
    conn.close()
    df['market_bullish'] = (df['market_risk_score'] == 10).astype(int)
    return df


def load_market_caps():
    """Load latest market cap per symbol from key metrics."""
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("""
        SELECT symbol, market_cap
        FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol
        HAVING date = MAX(date)
    """, conn)
    conn.close()
    return df.set_index('symbol')['market_cap']


def load_fcf_growth():
    """Compute FCF growth from historical cash flows (YoY quarterly)."""
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("""
        SELECT symbol, date, free_cash_flow
        FROM historical_cash_flows
        WHERE free_cash_flow IS NOT NULL
        ORDER BY symbol, date
    """, conn)
    conn.close()
    if df.empty:
        return pd.Series(dtype=float)

    # Latest vs year-ago FCF per symbol
    latest = df.groupby('symbol').last().rename(columns={'free_cash_flow': 'fcf_latest'})
    # Get row that's ~4 quarters back
    def get_yoy(group):
        if len(group) < 5:
            return pd.Series({'fcf_latest': group.iloc[-1]['free_cash_flow'],
                              'fcf_yago': None})
        return pd.Series({'fcf_latest': group.iloc[-1]['free_cash_flow'],
                          'fcf_yago': group.iloc[-5]['free_cash_flow']})

    fcf = df.groupby('symbol').apply(get_yoy)
    fcf['fcf_positive_growth'] = ((fcf['fcf_latest'] > 0) &
                                   (fcf['fcf_yago'].notna()) &
                                   (fcf['fcf_latest'] > fcf['fcf_yago'])).astype(int)
    return fcf['fcf_positive_growth']


def load_analyst_coverage():
    """Get analyst coverage count per symbol from estimates snapshot."""
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot
        GROUP BY symbol
    """, conn)
    conn.close()
    return df.set_index('symbol')['analyst_count']


# ==================== HELPER FUNCTIONS ====================

def stats_row(name, sub, min_count=50):
    """Compute stats for a filter subset."""
    n = len(sub)
    if n < min_count:
        return None
    win_3m = (sub['return_3m'] > 0).mean() * 100
    avg_3m = sub['return_3m'].clip(-90, 500).mean()
    med_3m = sub['return_3m'].clip(-90, 500).median()
    avg_1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 20 else np.nan
    return {
        'name': name, 'count': n, 'win_3m': win_3m,
        'avg_3m': avg_3m, 'med_3m': med_3m, 'avg_1y': avg_1y,
    }


def print_stats_table(rows, title, min_count=50):
    """Print a formatted stats table."""
    rows = [r for r in rows if r is not None]
    if not rows:
        print(f"\n  No combos with >= {min_count} signals")
        return

    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"{'='*95}")
    print(f"  {'Filter':<50s} {'Count':>7s} {'Win%3M':>7s} {'Avg3M':>8s} {'Med3M':>8s} {'Avg1Y':>8s}")
    print(f"  {'-'*90}")

    # Sort by win rate descending
    rows.sort(key=lambda r: r['win_3m'], reverse=True)
    for r in rows:
        y = f"{r['avg_1y']:+.1f}%" if pd.notna(r['avg_1y']) else "    N/A"
        print(f"  {r['name']:<50s} {r['count']:>6,} {r['win_3m']:>6.1f}% "
              f"{r['avg_3m']:>+7.1f}% {r['med_3m']:>+7.1f}% {y:>7s}")


# ==================== 1. MULTI-FACTOR INTERACTION ANALYSIS ====================

def run_multi_factor_analysis(df):
    print("\n" + "#" * 95)
    print("  SECTION 1: MULTI-FACTOR INTERACTION ANALYSIS")
    print("#" * 95)

    rows = []

    # Baseline
    rows.append(stats_row("Baseline (all signals with 3M return)", df, 50))

    # --- Requested combos ---
    rows.append(stats_row(
        "Bear + RSI<35 + Fund≥18 + EV/EBITDA 8-18",
        df[(df['market_bearish'] == 1) & (df['rsi'] < 35) &
           (df['fundamentals_score'] >= 18) &
           (df['ev_ebitda'] >= 8) & (df['ev_ebitda'] <= 18)], 20))

    rows.append(stats_row(
        "Low Trend(<10) + High Fund(≥18) + RSI<40",
        df[(df['trend_score'] < 10) & (df['fundamentals_score'] >= 18) &
           (df['rsi'] < 40)], 20))

    rows.append(stats_row(
        "Analysts≥8 + Downgrade Recovery + EPS Growth>0",
        df[(df['signal_type'] == 'Analyst Downgrade Recovery') &
           (df['eps_growth'] > 0)], 20))

    rows.append(stats_row(
        "Tier 1 + ADX<30",
        df[(df['signal_type'] == 'Tier 1') & (df['adx'] < 30)], 20))

    # --- Additional high-value combos ---
    rows.append(stats_row(
        "Bear + RSI<40 + Fund≥15 + V2≥40",
        df[(df['market_bearish'] == 1) & (df['rsi'] < 40) &
           (df['fundamentals_score'] >= 15) & (df['value_score_v2'] >= 40)], 20))

    rows.append(stats_row(
        "Bull + LT≥55 + V2≥55 + EV/EBITDA 0-18",
        df[(df['market_bullish'] == 1) & (df['lt_score'] >= 55) &
           (df['value_score_v2'] >= 55) &
           (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 18)], 20))

    rows.append(stats_row(
        "LT≥50 + V2≥50 + Fund≥15 + RSI 35-55",
        df[(df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) &
           (df['fundamentals_score'] >= 15) &
           (df['rsi'] >= 35) & (df['rsi'] <= 55)], 20))

    rows.append(stats_row(
        "V2≥60 + EV/EBITDA 0-15 + RevGrowth>10",
        df[(df['value_score_v2'] >= 60) &
           (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 15) &
           (df['rev_growth'] > 10)], 20))

    rows.append(stats_row(
        "Tier 3 + Fund≥18 + EV/EBITDA 5-20",
        df[(df['signal_type'] == 'Tier 3') &
           (df['fundamentals_score'] >= 18) &
           (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 20)], 20))

    rows.append(stats_row(
        "RSI Recovery + Fund≥15 + LT≥45",
        df[(df['signal_type'] == 'RSI Recovery') &
           (df['fundamentals_score'] >= 15) & (df['lt_score'] >= 45)], 20))

    rows.append(stats_row(
        "Bear + V2≥45 + RSI<35 + EPSGrowth>8",
        df[(df['market_bearish'] == 1) & (df['value_score_v2'] >= 45) &
           (df['rsi'] < 35) & (df['eps_growth'] > 8)], 20))

    rows.append(stats_row(
        "LT≥55 + Fund≥20 + Val≥10 + RSI 35-60",
        df[(df['lt_score'] >= 55) & (df['fundamentals_score'] >= 20) &
           (df['valuation_score'] >= 10) &
           (df['rsi'] >= 35) & (df['rsi'] <= 60)], 20))

    rows.append(stats_row(
        "V2≥50 + Trend≥15 + Fund≥15",
        df[(df['value_score_v2'] >= 50) & (df['trend_score'] >= 15) &
           (df['fundamentals_score'] >= 15)], 20))

    rows.append(stats_row(
        "EV/EBITDA 5-12 + RevGrowth>15 + EPSGrowth>10",
        df[(df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 12) &
           (df['rev_growth'] > 15) & (df['eps_growth'] > 10)], 20))

    rows.append(stats_row(
        "LT≥60 + V2≥55 + Fund≥18 + EV/EBITDA 0-22",
        df[(df['lt_score'] >= 60) & (df['value_score_v2'] >= 55) &
           (df['fundamentals_score'] >= 18) &
           (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 22)], 20))

    # Extreme value plays
    rows.append(stats_row(
        "EV/EBITDA < 8 + Fund≥15 + LT≥45",
        df[(df['ev_ebitda'] > 0) & (df['ev_ebitda'] < 8) &
           (df['fundamentals_score'] >= 15) & (df['lt_score'] >= 45)], 20))

    rows.append(stats_row(
        "RevGrowth>25 + EV/EBITDA<20 + RSI 35-65",
        df[(df['rev_growth'] > 25) &
           (df['ev_ebitda'] > 0) & (df['ev_ebitda'] < 20) &
           (df['rsi'] >= 35) & (df['rsi'] <= 65)], 20))

    print_stats_table(rows, "Multi-Factor Combinations (sorted by Win% 3M)", min_count=20)

    # Highlight the best
    valid = [r for r in rows if r is not None and r['count'] >= 100]
    if valid:
        best = max(valid, key=lambda r: r['win_3m'])
        print(f"\n  ** Best combo (≥100 signals): {best['name']}")
        print(f"     Count: {best['count']:,} | Win%: {best['win_3m']:.1f}% | "
              f"Avg 3M: {best['avg_3m']:+.1f}% | Avg 1Y: "
              f"{best['avg_1y']:+.1f}%" if pd.notna(best.get('avg_1y')) else "N/A")


# ==================== 2. WALK-FORWARD ROLLING VALIDATION ====================

def run_walk_forward(df):
    print("\n\n" + "#" * 95)
    print("  SECTION 2: WALK-FORWARD ROLLING VALIDATION")
    print("#" * 95)

    df['year'] = pd.to_datetime(df['date']).dt.year

    # Define test windows
    windows = [
        ("Train 2021-2022 → Test 2023", 2021, 2022, 2023, 2023),
        ("Train 2021-2023 → Test 2024", 2021, 2023, 2024, 2024),
        ("Train 2022-2024 → Test 2025-26", 2022, 2024, 2025, 2026),
    ]

    # Key signals to track
    key_signals = ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear',
                   'Regime Buy Bull', 'Buy_A', 'Buy_B', 'Buy_D',
                   'Analyst Downgrade Recovery', 'High Coverage Buy']

    for window_name, train_start, train_end, test_start, test_end in windows:
        print(f"\n{'='*80}")
        print(f"  {window_name}")
        print(f"{'='*80}")

        train = df[(df['year'] >= train_start) & (df['year'] <= train_end) &
                   (df['return_3m'].notna())]
        test = df[(df['year'] >= test_start) & (df['year'] <= test_end) &
                  (df['return_3m'].notna())]

        if test.empty:
            print("  No test data for this window")
            continue

        print(f"  Train: {len(train):,} signals ({train_start}-{train_end})")
        print(f"  Test:  {len(test):,} signals ({test_start}-{test_end})")

        print(f"\n  {'Signal':<30s} {'Train N':>8s} {'Train Win%':>10s} "
              f"{'Test N':>8s} {'Test Win%':>10s} {'Δ Win%':>8s} {'Stable?':>8s}")
        print(f"  {'-'*85}")

        for sig in key_signals:
            tr = train[train['signal_type'] == sig]
            te = test[test['signal_type'] == sig]
            if len(tr) < 30 or len(te) < 10:
                continue
            tr_win = (tr['return_3m'] > 0).mean() * 100
            te_win = (te['return_3m'] > 0).mean() * 100
            delta = te_win - tr_win
            stable = "YES" if abs(delta) < 8 else ("WARN" if abs(delta) < 15 else "NO")
            print(f"  {sig:<30s} {len(tr):>7,} {tr_win:>9.1f}% "
                  f"{len(te):>7,} {te_win:>9.1f}% {delta:>+7.1f}% {stable:>7s}")

    # Rolling 12-month win rates for top signals
    print(f"\n{'='*80}")
    print("  ROLLING 12-MONTH WIN RATES (Quarterly Buckets)")
    print(f"{'='*80}")

    df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q')
    quarters = sorted(df['quarter'].unique())

    for sig in ['Tier 1', 'Tier 3', 'Regime Buy Bear', 'Buy_B']:
        sig_data = df[(df['signal_type'] == sig) & (df['return_3m'].notna())]
        if len(sig_data) < 50:
            continue

        print(f"\n  {sig}:")
        print(f"  {'Quarter':<12s} {'Count':>7s} {'Win%3M':>8s} {'Avg3M':>8s}")
        print(f"  {'-'*38}")
        for q in quarters:
            qd = sig_data[sig_data['quarter'] == q]
            if len(qd) < 10:
                continue
            win = (qd['return_3m'] > 0).mean() * 100
            avg = qd['return_3m'].clip(-90, 500).mean()
            print(f"  {str(q):<12s} {len(qd):>6,} {win:>7.1f}% {avg:>+7.1f}%")


# ==================== 3. BOOTSTRAP CONFIDENCE INTERVALS ====================

def run_bootstrap_ci(df, n_bootstrap=5000):
    print("\n\n" + "#" * 95)
    print("  SECTION 3: BOOTSTRAP CONFIDENCE INTERVALS (95% CI)")
    print("#" * 95)

    signals_to_test = ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear',
                       'Regime Buy Bull', 'Buy_A', 'Buy_B', 'Buy_C', 'Buy_D',
                       'RSI Recovery', 'Analyst Downgrade Recovery',
                       'High Coverage Buy', 'Analyst Upgrade Cluster',
                       'Strong Buy', 'Buy', 'Golden Cross', 'Bullish Alignment']

    print(f"\n  {'Signal':<28s} {'N':>7s} {'Win%':>7s} {'95% CI':>16s} {'Avg3M':>8s} "
          f"{'3M CI':>18s} {'Trustworthy':>12s}")
    print(f"  {'-'*100}")

    results = []
    for sig in signals_to_test:
        sub = df[(df['signal_type'] == sig) & (df['return_3m'].notna())]
        returns = sub['return_3m'].clip(-90, 500).values
        if len(returns) < 30:
            continue

        win_rate = (returns > 0).mean() * 100
        avg_3m = returns.mean()

        # Bootstrap
        rng = np.random.default_rng(42)
        boot_wins = []
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(returns, size=len(returns), replace=True)
            boot_wins.append((sample > 0).mean() * 100)
            boot_means.append(sample.mean())

        win_lo = np.percentile(boot_wins, 2.5)
        win_hi = np.percentile(boot_wins, 97.5)
        mean_lo = np.percentile(boot_means, 2.5)
        mean_hi = np.percentile(boot_means, 97.5)

        # Trustworthy: lower bound of win CI >= 55% AND count >= 500
        trustworthy = "STRONG" if (win_lo >= 55 and len(returns) >= 500) else \
                      "YES" if (win_lo >= 52 and len(returns) >= 200) else \
                      "MAYBE" if (win_lo >= 50 and len(returns) >= 100) else "NO"

        print(f"  {sig:<28s} {len(returns):>6,} {win_rate:>6.1f}% "
              f"[{win_lo:>5.1f}%-{win_hi:>5.1f}%] {avg_3m:>+7.1f}% "
              f"[{mean_lo:>+6.1f}%,{mean_hi:>+6.1f}%] {trustworthy:>11s}")

        results.append({
            'signal': sig, 'count': len(returns),
            'win_rate': win_rate, 'win_ci_lo': win_lo, 'win_ci_hi': win_hi,
            'avg_3m': avg_3m, 'trustworthy': trustworthy
        })

    # Summary
    strong = [r for r in results if r['trustworthy'] == 'STRONG']
    yes = [r for r in results if r['trustworthy'] == 'YES']
    print(f"\n  STRONG signals (CI lower bound ≥55%, N≥500): "
          f"{', '.join(r['signal'] for r in strong) or 'None'}")
    print(f"  Reliable signals (CI lower bound ≥52%, N≥200): "
          f"{', '.join(r['signal'] for r in yes) or 'None'}")

    # Also test key multi-factor combos
    print(f"\n  --- Multi-Factor Combo CIs ---")
    combos = [
        ("Bear+RSI<40+Fund≥15+V2≥40",
         df[(df['market_bearish'] == 1) & (df['rsi'] < 40) &
            (df['fundamentals_score'] >= 15) & (df['value_score_v2'] >= 40)]),
        ("LT≥50+V2≥50+Fund≥15+RSI35-55",
         df[(df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) &
            (df['fundamentals_score'] >= 15) &
            (df['rsi'] >= 35) & (df['rsi'] <= 55)]),
        ("EV/EBITDA 5-12+RevG>15+EPSG>10",
         df[(df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 12) &
            (df['rev_growth'] > 15) & (df['eps_growth'] > 10)]),
        ("Tier3+Fund≥18+EV5-20",
         df[(df['signal_type'] == 'Tier 3') &
            (df['fundamentals_score'] >= 18) &
            (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 20)]),
    ]

    print(f"\n  {'Combo':<40s} {'N':>7s} {'Win%':>7s} {'95% CI':>16s} {'Trust':>8s}")
    print(f"  {'-'*80}")
    for name, sub in combos:
        returns = sub['return_3m'].clip(-90, 500).dropna().values
        if len(returns) < 20:
            continue
        win_rate = (returns > 0).mean() * 100
        rng = np.random.default_rng(42)
        boot_wins = [(rng.choice(returns, len(returns), replace=True) > 0).mean() * 100
                     for _ in range(n_bootstrap)]
        win_lo = np.percentile(boot_wins, 2.5)
        win_hi = np.percentile(boot_wins, 97.5)
        trust = "YES" if win_lo >= 52 else "MAYBE" if win_lo >= 48 else "NO"
        print(f"  {name:<40s} {len(returns):>6,} {win_rate:>6.1f}% "
              f"[{win_lo:>5.1f}%-{win_hi:>5.1f}%] {trust:>7s}")


# ==================== 4. FEATURE IMPORTANCE (ML) ====================

def run_feature_importance(df):
    print("\n\n" + "#" * 95)
    print("  SECTION 4: FEATURE IMPORTANCE (Random Forest + Gradient Boosting)")
    print("#" * 95)

    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("\n  scikit-learn not installed. Install with: pip install scikit-learn")
        print("  Skipping ML analysis.")
        return

    # Prepare features
    features = ['lt_score', 'value_score_v2', 'trend_score', 'fundamentals_score',
                'valuation_score', 'momentum_score', 'market_risk_score',
                'rsi', 'adx', 'ev_ebitda_clean', 'rev_growth', 'eps_growth',
                'ebitda_growth']

    ml_df = df[features + ['return_3m']].dropna()
    if len(ml_df) < 1000:
        print("  Insufficient data for ML analysis")
        return

    X = ml_df[features].values
    y = (ml_df['return_3m'] > 0).astype(int).values

    print(f"\n  Dataset: {len(ml_df):,} signals | {y.mean()*100:.1f}% positive")

    # Random Forest
    print("\n  --- Random Forest (500 trees) ---")
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=50,
                                random_state=42, n_jobs=-1)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {rf_scores.mean()*100:.1f}% (±{rf_scores.std()*100:.1f}%)")

    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

    print(f"\n  {'Feature':<25s} {'Importance':>12s} {'Rank':>6s}")
    print(f"  {'-'*45}")
    for i, (feat, imp) in enumerate(importances.items(), 1):
        bar = '█' * int(imp * 100)
        print(f"  {feat:<25s} {imp:>11.4f}  #{i:<4d} {bar}")

    # Gradient Boosting
    print("\n  --- Gradient Boosting (500 trees) ---")
    gb = GradientBoostingClassifier(n_estimators=500, max_depth=4, min_samples_leaf=50,
                                     learning_rate=0.05, random_state=42)
    gb_scores = cross_val_score(gb, X, y, cv=5, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {gb_scores.mean()*100:.1f}% (±{gb_scores.std()*100:.1f}%)")

    gb.fit(X, y)
    gb_imp = pd.Series(gb.feature_importances_, index=features).sort_values(ascending=False)

    print(f"\n  {'Feature':<25s} {'RF Rank':>8s} {'GB Rank':>8s} {'RF Imp':>8s} {'GB Imp':>8s}")
    print(f"  {'-'*55}")
    rf_ranks = importances.rank(ascending=False).astype(int)
    gb_ranks = gb_imp.rank(ascending=False).astype(int)
    for feat in importances.index:
        print(f"  {feat:<25s} #{rf_ranks[feat]:<6d} #{gb_ranks[feat]:<6d} "
              f"{importances[feat]:>7.4f} {gb_imp[feat]:>7.4f}")

    # Partial dependence for top features (simplified)
    print("\n  --- Top Feature Thresholds (Win Rate by Bucket) ---")
    top_features = importances.head(5).index.tolist()
    for feat in top_features:
        feat_data = ml_df[[feat, 'return_3m']].dropna()
        if len(feat_data) < 100:
            continue
        try:
            bins = pd.qcut(feat_data[feat], q=5, duplicates='drop')
            grouped = feat_data.groupby(bins)['return_3m'].agg(
                count='count',
                win_rate=lambda x: (x > 0).mean() * 100,
                avg_return=lambda x: x.clip(-90, 500).mean()
            )
            print(f"\n  {feat}:")
            print(f"  {'Bucket':<25s} {'Count':>7s} {'Win%':>7s} {'Avg3M':>8s}")
            print(f"  {'-'*50}")
            for bucket, row in grouped.iterrows():
                print(f"  {str(bucket):<25s} {int(row['count']):>6,} "
                      f"{row['win_rate']:>6.1f}% {row['avg_return']:>+7.1f}%")
        except Exception:
            continue


# ==================== 5. NEW HIGH-CONVICTION FILTER TESTS ====================

def run_new_filter_tests(df):
    print("\n\n" + "#" * 95)
    print("  SECTION 5: NEW HIGH-CONVICTION FILTER TESTS")
    print("#" * 95)

    # Load supplementary data
    print("  Loading supplementary data (market cap, FCF, analyst coverage)...")
    mcaps = load_market_caps()
    fcf_growth = load_fcf_growth()
    analyst_cov = load_analyst_coverage()

    # Merge into signals df
    df_enriched = df.copy()
    df_enriched['market_cap'] = df_enriched['symbol'].map(mcaps)
    df_enriched['fcf_positive'] = df_enriched['symbol'].map(fcf_growth)
    df_enriched['analyst_count'] = df_enriched['symbol'].map(analyst_cov)

    rows = []
    rows.append(stats_row("Baseline (all signals)", df_enriched, 50))

    # --- Analyst coverage filters ---
    rows.append(stats_row(
        "Analyst Coverage ≥ 6",
        df_enriched[df_enriched['analyst_count'] >= 6], 50))
    rows.append(stats_row(
        "Analyst Coverage ≥ 10",
        df_enriched[df_enriched['analyst_count'] >= 10], 50))
    rows.append(stats_row(
        "Analyst Coverage ≥ 10 + LT≥50",
        df_enriched[(df_enriched['analyst_count'] >= 10) &
                    (df_enriched['lt_score'] >= 50)], 20))

    # --- Market cap / liquidity filters ---
    rows.append(stats_row(
        "Market Cap > $1B",
        df_enriched[df_enriched['market_cap'] > 1e9], 50))
    rows.append(stats_row(
        "Market Cap > $2B",
        df_enriched[df_enriched['market_cap'] > 2e9], 50))
    rows.append(stats_row(
        "Market Cap > $2B + LT≥50 + V2≥50",
        df_enriched[(df_enriched['market_cap'] > 2e9) &
                    (df_enriched['lt_score'] >= 50) &
                    (df_enriched['value_score_v2'] >= 50)], 20))
    rows.append(stats_row(
        "Market Cap > $10B (Large Cap only)",
        df_enriched[df_enriched['market_cap'] > 10e9], 50))

    # --- FCF growth filter ---
    rows.append(stats_row(
        "Positive FCF Growth (YoY)",
        df_enriched[df_enriched['fcf_positive'] == 1], 50))
    rows.append(stats_row(
        "FCF Growth + LT≥50 + V2≥45",
        df_enriched[(df_enriched['fcf_positive'] == 1) &
                    (df_enriched['lt_score'] >= 50) &
                    (df_enriched['value_score_v2'] >= 45)], 20))

    # --- Combined high-conviction ---
    rows.append(stats_row(
        "Cap>$2B + Analysts≥6 + FCF Growth + LT≥45",
        df_enriched[(df_enriched['market_cap'] > 2e9) &
                    (df_enriched['analyst_count'] >= 6) &
                    (df_enriched['fcf_positive'] == 1) &
                    (df_enriched['lt_score'] >= 45)], 20))

    rows.append(stats_row(
        "Cap>$1B + Analysts≥6 + EV/EBITDA 5-18 + Fund≥15",
        df_enriched[(df_enriched['market_cap'] > 1e9) &
                    (df_enriched['analyst_count'] >= 6) &
                    (df_enriched['ev_ebitda'] >= 5) & (df_enriched['ev_ebitda'] <= 18) &
                    (df_enriched['fundamentals_score'] >= 15)], 20))

    rows.append(stats_row(
        "Cap>$2B + V2≥55 + RSI 35-60 + Analysts≥8",
        df_enriched[(df_enriched['market_cap'] > 2e9) &
                    (df_enriched['value_score_v2'] >= 55) &
                    (df_enriched['rsi'] >= 35) & (df_enriched['rsi'] <= 60) &
                    (df_enriched['analyst_count'] >= 8)], 20))

    # --- EPS momentum (strong recent growth) ---
    rows.append(stats_row(
        "EPS Growth > 20% + Rev Growth > 15%",
        df_enriched[(df_enriched['eps_growth'] > 20) &
                    (df_enriched['rev_growth'] > 15)], 50))
    rows.append(stats_row(
        "EPS>20 + Rev>15 + EV/EBITDA<20 + LT≥45",
        df_enriched[(df_enriched['eps_growth'] > 20) & (df_enriched['rev_growth'] > 15) &
                    (df_enriched['ev_ebitda'] > 0) & (df_enriched['ev_ebitda'] < 20) &
                    (df_enriched['lt_score'] >= 45)], 20))

    print_stats_table(rows, "New High-Conviction Filters (sorted by Win% 3M)", min_count=20)


# ==================== 6. REGIME-ADAPTIVE THRESHOLDS ====================

def run_regime_analysis(df):
    print("\n\n" + "#" * 95)
    print("  SECTION 6: REGIME-ADAPTIVE THRESHOLD ANALYSIS")
    print("#" * 95)

    bull = df[df['market_bullish'] == 1]
    bear = df[df['market_bearish'] == 1]

    print(f"\n  Bull market signals: {len(bull):,}")
    print(f"  Bear market signals: {len(bear):,}")

    # --- Bull market optimal thresholds ---
    print(f"\n{'='*80}")
    print("  BULL MARKET THRESHOLDS")
    print(f"{'='*80}")

    bull_rows = []
    bull_rows.append(stats_row("Bull Baseline", bull, 50))
    bull_rows.append(stats_row("Bull + LT≥55", bull[bull['lt_score'] >= 55], 50))
    bull_rows.append(stats_row("Bull + LT≥55 + EV/EBITDA<16", bull[(bull['lt_score'] >= 55) & (bull['ev_ebitda'] > 0) & (bull['ev_ebitda'] < 16)], 20))
    bull_rows.append(stats_row("Bull + LT≥55 + Fund≥18", bull[(bull['lt_score'] >= 55) & (bull['fundamentals_score'] >= 18)], 20))
    bull_rows.append(stats_row("Bull + LT≥50 + V2≥55", bull[(bull['lt_score'] >= 50) & (bull['value_score_v2'] >= 55)], 20))
    bull_rows.append(stats_row("Bull + V2≥60", bull[bull['value_score_v2'] >= 60], 20))
    bull_rows.append(stats_row("Bull + V2≥60 + EV/EBITDA<18", bull[(bull['value_score_v2'] >= 60) & (bull['ev_ebitda'] > 0) & (bull['ev_ebitda'] < 18)], 20))
    bull_rows.append(stats_row("Bull + LT≥55 + Fund≥18 + RSI 40-60", bull[(bull['lt_score'] >= 55) & (bull['fundamentals_score'] >= 18) & (bull['rsi'] >= 40) & (bull['rsi'] <= 60)], 20))

    print_stats_table(bull_rows, "Bull Market Optimal Thresholds", min_count=20)

    # --- Bear market optimal thresholds ---
    print(f"\n{'='*80}")
    print("  BEAR MARKET THRESHOLDS")
    print(f"{'='*80}")

    bear_rows = []
    bear_rows.append(stats_row("Bear Baseline", bear, 50))
    bear_rows.append(stats_row("Bear + LT≥48", bear[bear['lt_score'] >= 48], 20))
    bear_rows.append(stats_row("Bear + LT≥45 + RSI<35", bear[(bear['lt_score'] >= 45) & (bear['rsi'] < 35)], 20))
    bear_rows.append(stats_row("Bear + RSI<35 + Fund≥15", bear[(bear['rsi'] < 35) & (bear['fundamentals_score'] >= 15)], 20))
    bear_rows.append(stats_row("Bear + RSI<35 + Fund≥18", bear[(bear['rsi'] < 35) & (bear['fundamentals_score'] >= 18)], 20))
    bear_rows.append(stats_row("Bear + V2≥40 + RSI<40", bear[(bear['value_score_v2'] >= 40) & (bear['rsi'] < 40)], 20))
    bear_rows.append(stats_row("Bear + V2≥45 + Fund≥15 + RSI<40", bear[(bear['value_score_v2'] >= 45) & (bear['fundamentals_score'] >= 15) & (bear['rsi'] < 40)], 20))
    bear_rows.append(stats_row("Bear + EV/EBITDA 5-15 + Fund≥18 + RSI<40", bear[(bear['ev_ebitda'] >= 5) & (bear['ev_ebitda'] <= 15) & (bear['fundamentals_score'] >= 18) & (bear['rsi'] < 40)], 20))
    bear_rows.append(stats_row("Bear + LT≥45 + EPSGrowth>8 + RSI<40", bear[(bear['lt_score'] >= 45) & (bear['eps_growth'] > 8) & (bear['rsi'] < 40)], 20))

    print_stats_table(bear_rows, "Bear Market Optimal Thresholds", min_count=20)

    # --- Regime comparison for each signal type ---
    print(f"\n{'='*80}")
    print("  SIGNAL PERFORMANCE BY REGIME")
    print(f"{'='*80}")

    print(f"\n  {'Signal':<28s} {'Bull N':>7s} {'Bull Win%':>9s} {'Bull 3M':>8s} "
          f"{'Bear N':>7s} {'Bear Win%':>9s} {'Bear 3M':>8s} {'Better In':>10s}")
    print(f"  {'-'*90}")

    for sig in ['Tier 1', 'Tier 2', 'Tier 3', 'Regime Buy Bear', 'Regime Buy Bull',
                'Buy_A', 'Buy_B', 'Buy_D', 'RSI Recovery',
                'Analyst Downgrade Recovery', 'High Coverage Buy']:
        b = bull[(bull['signal_type'] == sig) & (bull['return_3m'].notna())]
        br = bear[(bear['signal_type'] == sig) & (bear['return_3m'].notna())]
        if len(b) < 20 and len(br) < 20:
            continue
        b_win = (b['return_3m'] > 0).mean() * 100 if len(b) > 0 else 0
        br_win = (br['return_3m'] > 0).mean() * 100 if len(br) > 0 else 0
        b_avg = b['return_3m'].clip(-90, 500).mean() if len(b) > 0 else 0
        br_avg = br['return_3m'].clip(-90, 500).mean() if len(br) > 0 else 0
        better = "BEAR" if br_win > b_win else "BULL"
        print(f"  {sig:<28s} {len(b):>6,} {b_win:>8.1f}% {b_avg:>+7.1f}% "
              f"{len(br):>6,} {br_win:>8.1f}% {br_avg:>+7.1f}% {better:>9s}")


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Advanced Backtest Analysis')
    parser.add_argument('--section', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Run specific section only (1-6)')
    parser.add_argument('--bootstrap-n', type=int, default=5000,
                        help='Number of bootstrap iterations (default: 5000)')
    args = parser.parse_args()

    start = datetime.now()

    print("=" * 95)
    print("  ADVANCED BACKTEST ANALYSIS SUITE")
    print(f"  Database: {BACKTEST_DB}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 95)

    print("\n  Loading signals with full indicator context...")
    df = load_signals_with_context()
    print(f"  Loaded {len(df):,} signals across {df['symbol'].nunique():,} stocks")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Signals with 3M return: {df['return_3m'].notna().sum():,}")
    print(f"  Signals with 1Y return: {df['return_1y'].notna().sum():,}")

    sections = {
        1: ("Multi-Factor Interactions", lambda: run_multi_factor_analysis(df)),
        2: ("Walk-Forward Validation", lambda: run_walk_forward(df)),
        3: ("Bootstrap Confidence Intervals", lambda: run_bootstrap_ci(df, args.bootstrap_n)),
        4: ("Feature Importance (ML)", lambda: run_feature_importance(df)),
        5: ("New High-Conviction Filters", lambda: run_new_filter_tests(df)),
        6: ("Regime-Adaptive Thresholds", lambda: run_regime_analysis(df)),
    }

    if args.section:
        name, fn = sections[args.section]
        print(f"\n  Running Section {args.section}: {name}")
        fn()
    else:
        for num, (name, fn) in sections.items():
            print(f"\n  Running Section {num}: {name}...")
            fn()

    elapsed = datetime.now() - start
    print(f"\n\n{'='*95}")
    print(f"  Analysis complete! Runtime: {elapsed}")
    print(f"{'='*95}")


if __name__ == "__main__":
    main()

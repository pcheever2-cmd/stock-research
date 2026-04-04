#!/usr/bin/env python3
"""
Trading Strategy & Final Validation
=====================================
1. Earnings Surprise Overlay on 3 Final Buckets (drop Bucket 4 / DCF)
2. Strict 2025 OOS Validation
3. Exact Trading Rule Cards (thresholds, regime logic, alerts)
4. Portfolio Simulation (position sizing, stop-loss, equity curve)
5. Current Actionable Stocks with all overlays + badges

Runs against backtest.db + nasdaq_stocks.db (read-only).
"""

import sqlite3, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (same as final_bucket_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Load and enrich all signal data."""
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

    # Supplementary
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

    # DCF
    try:
        dcf = pd.read_sql_query("""
            SELECT symbol, dcf_value, stock_price, dcf_upside_pct
            FROM dcf_valuations WHERE dcf_upside_pct IS NOT NULL
        """, conn).set_index('symbol')['dcf_upside_pct']
    except Exception:
        dcf = pd.Series(dtype=float)

    # Earnings surprise
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

    conn2 = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query(
        "SELECT symbol, sector, industry FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    # Enrich
    df['market_cap'] = df['symbol'].map(mcaps)
    df['fcf_positive'] = df['symbol'].map(fcf_positive)
    df['analyst_count'] = df['symbol'].map(acov)
    df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
    df['industry'] = df['symbol'].map(sectors['industry']) if 'industry' in sectors.columns else None
    df['market_bullish'] = (df['market_risk_score'] == 10).astype(int)
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    df['dcf_upside'] = df['symbol'].map(dcf)

    # Earnings surprise
    if not income.empty and not estimates.empty:
        merged = income.merge(estimates, on=['symbol', 'fiscal_year'], how='inner')
        if not merged.empty:
            merged = merged.sort_values('fiscal_year').groupby('symbol').last().reset_index()
            merged['earnings_surprise_pct'] = np.where(
                merged['eps_avg'].abs() > 0.01,
                (merged['actual_eps'] - merged['eps_avg']) / merged['eps_avg'].abs() * 100,
                np.nan
            )
            surprise_map = merged.set_index('symbol')['earnings_surprise_pct'].clip(-200, 500)
            df['earnings_surprise_pct'] = df['symbol'].map(surprise_map)
        else:
            df['earnings_surprise_pct'] = np.nan
    else:
        df['earnings_surprise_pct'] = np.nan

    # Ensemble voting
    vote_counts = df.groupby(['symbol', 'date']).size().reset_index(name='concurrent_signals')
    df = df.merge(vote_counts, on=['symbol', 'date'], how='left')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET DEFINITIONS (3 final buckets)
# ══════════════════════════════════════════════════════════════════════════════

BUCKETS = {
    "Quality Growth Compounder": {
        'id': 1,
        'criteria': "V2≥55 + Fund≥18 + EV[5-20] + RevG>10 + Cap>$2B + Analysts≥6",
        'filter': lambda d: d[(d['value_score_v2'] >= 55) & (d['fundamentals_score'] >= 18) &
                              (d['ev_ebitda'] >= 5) & (d['ev_ebitda'] <= 20) &
                              (d['rev_growth'] > 10) & (d['market_cap'] > 2e9) &
                              (d['analyst_count'] >= 6)],
        'hold_months': 6,
        'position_pct': 3.0,     # % of portfolio per position
        'max_positions': 8,
        'stop_loss_pct': -15.0,  # exit if down this much
        'regime': 'all',         # works in all regimes
    },
    "Bear Market Dip Buy": {
        'id': 2,
        'criteria': "Bear regime + RSI<40 + Fund≥15 + V2≥40 + Cap>$1B",
        'filter': lambda d: d[(d['market_bearish'] == 1) & (d['rsi'] < 40) &
                              (d['fundamentals_score'] >= 15) & (d['value_score_v2'] >= 40) &
                              (d['market_cap'] > 1e9)],
        'hold_months': 6,
        'position_pct': 4.0,     # larger sizing — high conviction
        'max_positions': 6,
        'stop_loss_pct': -20.0,  # wider stop — volatile entries
        'regime': 'bear',
    },
    "High-Growth Momentum": {
        'id': 3,
        'criteria': "EPSG≥35 + EBITDAG≥33 + EV[12-27] + RSI<43",
        'filter': lambda d: d[(d['eps_growth'] >= 35) & (d['ebitda_growth'] >= 33) &
                              (d['ev_ebitda_clean'] >= 12) & (d['ev_ebitda_clean'] <= 27) &
                              (d['rsi'] < 43)],
        'hold_months': 3,
        'position_pct': 2.5,
        'max_positions': 10,
        'stop_loss_pct': -12.0,  # tighter stop — momentum reversal
        'regime': 'all',
    },
}


def bootstrap_ci(returns, n_boot=3000, seed=42):
    rng = np.random.default_rng(seed)
    wins = [(rng.choice(returns, len(returns), replace=True) > 0).mean() * 100 for _ in range(n_boot)]
    return np.percentile(wins, 2.5), np.percentile(wins, 97.5)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EARNINGS SURPRISE OVERLAY ON 3 BUCKETS
# ══════════════════════════════════════════════════════════════════════════════

def section_1_earnings_overlay(df):
    print("\n" + "#" * 115)
    print("  SECTION 1: EARNINGS SURPRISE OVERLAY ON 3 FINAL BUCKETS")
    print("#" * 115)

    es_n = df['earnings_surprise_pct'].notna().sum()
    print(f"\n  Earnings surprise coverage: {es_n:,}/{len(df):,} ({es_n/len(df)*100:.1f}%)")

    df_es = df[df['earnings_surprise_pct'].notna()].copy()

    overlays = [
        ("No filter", lambda d: d),
        ("Earnings Beat (>0%)", lambda d: d[d['earnings_surprise_pct'] > 0]),
        ("Earnings Big Beat (>10%)", lambda d: d[d['earnings_surprise_pct'] > 10]),
        ("Earnings Beat + 2+ signals", lambda d: d[(d['earnings_surprise_pct'] > 0) &
                                                     (d['concurrent_signals'] >= 2)]),
    ]

    print(f"\n  {'Bucket':<30s} {'Overlay':<28s} │ {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} {'Avg6M':>7s} "
          f"{'Avg1Y':>7s} {'PF':>5s} {'CI_lo':>6s} {'CI_hi':>6s} │ {'ΔWin':>6s}")
    print(f"  {'─'*130}")

    results = {}
    for bname, bconf in BUCKETS.items():
        base_all = bconf['filter'](df_es)
        base_all = base_all[base_all['return_3m'].notna()]
        if len(base_all) < 30:
            print(f"  {bname:<30s} {'(insufficient data)'}")
            continue

        base_win = (base_all['return_3m'] > 0).mean() * 100

        for oname, ofilt in overlays:
            sub = ofilt(base_all)
            if len(sub) < 20:
                print(f"  {bname:<30s} {oname:<28s} │ {len(sub):>5,}   (N too small)")
                continue

            ret = sub['return_3m'].clip(-90, 500)
            w = (ret > 0).mean() * 100
            a3 = ret.mean()
            a6 = sub['return_6m'].clip(-90, 1000).mean() if sub['return_6m'].notna().sum() > 10 else np.nan
            a1y = sub['return_1y'].clip(-90, 1000).mean() if sub['return_1y'].notna().sum() > 10 else np.nan
            wins = ret[ret > 0]
            losses = ret[ret <= 0]
            pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
            lo, hi = bootstrap_ci(ret.values)
            delta = w - base_win

            a6s = f"{a6:+.1f}%" if pd.notna(a6) else "   N/A"
            a1ys = f"{a1y:+.1f}%" if pd.notna(a1y) else "   N/A"
            print(f"  {bname:<30s} {oname:<28s} │ {len(sub):>5,} {w:>5.1f}% {a3:>+6.1f}% "
                  f"{a6s:>6s} {a1ys:>6s} {pf:>4.2f} {lo:>5.1f}% {hi:>5.1f}% │ {delta:>+5.1f}%")

            key = f"{bname}|{oname}"
            results[key] = {'win': w, 'n': len(sub), 'delta': delta, 'pf': pf, 'ci_lo': lo}

    # Summary recommendation
    print(f"\n  ── Overlay Recommendations ──")
    for bname in BUCKETS:
        base_key = f"{bname}|No filter"
        beat_key = f"{bname}|Earnings Beat (>0%)"
        if base_key in results and beat_key in results:
            d = results[beat_key]['delta']
            n_ratio = results[beat_key]['n'] / results[base_key]['n']
            if d >= 3.0 and n_ratio >= 0.25:
                print(f"  ✓ {bname}: Earnings Beat overlay RECOMMENDED (+{d:.1f}% win, "
                      f"retains {n_ratio*100:.0f}% of signals)")
            elif d >= 1.0:
                print(f"  ~ {bname}: Earnings Beat overlay OPTIONAL (+{d:.1f}% win, "
                      f"retains {n_ratio*100:.0f}% of signals)")
            else:
                print(f"  ✗ {bname}: Earnings Beat overlay NOT NEEDED ({d:+.1f}% win)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: STRICT 2025 OUT-OF-SAMPLE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def section_2_oos_validation(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 2: STRICT 2025 OUT-OF-SAMPLE VALIDATION")
    print("#" * 115)
    print(f"\n  Methodology: Buckets defined on 2021-2024 data. Test ONLY on 2025+ signals.")
    print(f"  This is the most conservative test — no peeking at 2025 data during bucket design.")

    df_oos = df[df['year'] >= 2025].copy()
    df_train = df[df['year'] < 2025].copy()
    print(f"  Training signals (pre-2025): {len(df_train):,}")
    print(f"  OOS signals (2025+): {len(df_oos):,}")

    print(f"\n  {'Bucket':<30s} │ {'Train':^30s} │ {'OOS (2025+)':^30s} │ {'Δ Win':>6s}")
    print(f"  {'':<30s} │ {'N':>6s} {'Win%':>6s} {'Avg':>7s} {'PF':>5s} {'CI_lo':>6s} │ "
          f"{'N':>6s} {'Win%':>6s} {'Avg':>7s} {'PF':>5s} {'CI_lo':>6s} │")
    print(f"  {'─'*120}")

    for bname, bconf in BUCKETS.items():
        train_sub = bconf['filter'](df_train)
        train_sub = train_sub[train_sub['return_3m'].notna()]
        oos_sub = bconf['filter'](df_oos)
        oos_sub = oos_sub[oos_sub['return_3m'].notna()]

        def _stats(s):
            if len(s) < 10:
                return len(s), None, None, None, None
            ret = s['return_3m'].clip(-90, 500)
            w = (ret > 0).mean() * 100
            a = ret.mean()
            wins = ret[ret > 0]
            losses = ret[ret <= 0]
            pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
            lo, _ = bootstrap_ci(ret.values)
            return len(s), w, a, pf, lo

        tn, tw, ta, tpf, tlo = _stats(train_sub)
        on, ow, oa, opf, olo = _stats(oos_sub)

        if tw is not None and ow is not None:
            delta = ow - tw
            print(f"  {bname:<30s} │ {tn:>5,} {tw:>5.1f}% {ta:>+6.1f}% {tpf:>4.2f} {tlo:>5.1f}% │ "
                  f"{on:>5,} {ow:>5.1f}% {oa:>+6.1f}% {opf:>4.2f} {olo:>5.1f}% │ {delta:>+5.1f}%")
        elif tw is not None:
            print(f"  {bname:<30s} │ {tn:>5,} {tw:>5.1f}% {ta:>+6.1f}% {tpf:>4.2f} {tlo:>5.1f}% │ "
                  f"{'No OOS signals':>30s} │    N/A")
        else:
            print(f"  {bname:<30s} │ {'Insufficient data':>30s} │ {'Insufficient data':>30s} │    N/A")

    # Overlay test: earnings beat on OOS
    print(f"\n  --- With Earnings Beat Overlay (OOS 2025+) ---")
    df_oos_es = df_oos[df_oos['earnings_surprise_pct'].notna()]
    print(f"  OOS signals with earnings data: {len(df_oos_es):,}")

    for bname, bconf in BUCKETS.items():
        base = bconf['filter'](df_oos_es)
        base = base[base['return_3m'].notna()]
        beat = base[base['earnings_surprise_pct'] > 0]

        if len(base) < 10:
            print(f"  {bname:<30s} │ Base: N={len(base):>4} (insufficient)")
            continue

        bw = (base['return_3m'] > 0).mean() * 100
        ba = base['return_3m'].clip(-90, 500).mean()

        if len(beat) >= 10:
            ew = (beat['return_3m'] > 0).mean() * 100
            ea = beat['return_3m'].clip(-90, 500).mean()
            d = ew - bw
            print(f"  {bname:<30s} │ Base: N={len(base):>4} Win={bw:>5.1f}% Avg={ba:>+5.1f}% │ "
                  f"+Beat: N={len(beat):>4} Win={ew:>5.1f}% Avg={ea:>+5.1f}% │ Δ={d:>+5.1f}%")
        else:
            print(f"  {bname:<30s} │ Base: N={len(base):>4} Win={bw:>5.1f}% │ +Beat: N={len(beat):>4} (insufficient)")

    # Year-by-year for 2025 signals
    print(f"\n  --- Monthly Breakdown (2025 OOS) ---")
    df_2025 = df_oos[df_oos['year'] == 2025].copy()
    df_2025['month'] = df_2025['date_dt'].dt.month

    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df_2025)
        sub = sub[sub['return_3m'].notna()]
        if len(sub) < 10:
            continue
        print(f"\n  {bname}:")
        months = sorted(sub['month'].unique())
        for m in months:
            ms = sub[sub['month'] == m]
            if len(ms) < 5:
                continue
            w = (ms['return_3m'] > 0).mean() * 100
            a = ms['return_3m'].clip(-90, 500).mean()
            print(f"    2025-{m:02d}: N={len(ms):>4} Win={w:>5.1f}% Avg={a:>+5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRADING RULE CARDS
# ══════════════════════════════════════════════════════════════════════════════

def section_3_rule_cards(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 3: EXACT TRADING RULE CARDS")
    print("#" * 115)

    # Determine current regime
    latest = df.sort_values('date_dt').tail(100)
    bull_pct = latest['market_bullish'].mean() * 100
    bear_pct = latest['market_bearish'].mean() * 100
    regime = "BULL" if bull_pct > 50 else ("BEAR" if bear_pct > 50 else "NEUTRAL")

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  CURRENT REGIME: {regime:>8s}                 │")
    print(f"  │  Bull signals: {bull_pct:>5.1f}%                    │")
    print(f"  │  Bear signals: {bear_pct:>5.1f}%                    │")
    print(f"  └─────────────────────────────────────────┘")

    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna()]
        ret = sub['return_3m'].clip(-90, 500)
        w = (ret > 0).mean() * 100 if len(sub) > 0 else 0
        wins = ret[ret > 0]
        losses = ret[ret <= 0]
        pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else 0

        active = True
        if bconf['regime'] == 'bear' and regime != 'BEAR':
            active = False

        status = "ACTIVE" if active else "STANDBY (waiting for bear regime)"

        print(f"\n  ╔{'═'*110}╗")
        print(f"  ║  BUCKET {bconf['id']}: {bname.upper():<70s} [{status}] ║")
        print(f"  ╠{'═'*110}╣")
        print(f"  ║  Criteria: {bconf['criteria']:<97s} ║")
        print(f"  ║{'─'*110}║")
        print(f"  ║  ENTRY RULES:                                                                                          ║")

        # Parse criteria into individual rules
        rules = bconf['criteria'].replace(' + ', '|').split('|')
        for r in rules:
            print(f"  ║    ► {r.strip():<102s} ║")

        print(f"  ║{'─'*110}║")
        print(f"  ║  POSITION MANAGEMENT:                                                                                   ║")
        print(f"  ║    Position Size:    {bconf['position_pct']:.1f}% of portfolio per trade"
              f"{'':>60s} ║")
        print(f"  ║    Max Positions:    {bconf['max_positions']} concurrent"
              f"{'':>72s} ║")
        print(f"  ║    Max Exposure:     {bconf['position_pct'] * bconf['max_positions']:.0f}% of portfolio"
              f"{'':>67s} ║")
        print(f"  ║    Hold Period:      {bconf['hold_months']} months (target)"
              f"{'':>69s} ║")
        print(f"  ║    Stop Loss:        {bconf['stop_loss_pct']:.0f}%"
              f"{'':>79s} ║")
        print(f"  ║{'─'*110}║")
        print(f"  ║  HISTORICAL PERFORMANCE (all data):                                                                     ║")
        print(f"  ║    Signals:          {len(sub):>6,}{'':>80s} ║")
        print(f"  ║    Win Rate (3M):    {w:>5.1f}%{'':>80s} ║")
        print(f"  ║    Profit Factor:    {pf:>5.2f}{'':>80s} ║")

        # Earnings beat boost
        sub_es = sub[sub['earnings_surprise_pct'] > 0]
        if len(sub_es) >= 20:
            ew = (sub_es['return_3m'] > 0).mean() * 100
            boost = ew - w
            badge = "EARNINGS BEAT CONFIRMED" if boost >= 2 else ""
            print(f"  ║    + Earnings Beat:  {ew:>5.1f}% win ({boost:+.1f}% boost)  "
                  f"{badge:<60s} ║")

        print(f"  ║{'─'*110}║")
        print(f"  ║  REGIME:             {bconf['regime'].upper():<86s} ║")

        if bconf['regime'] == 'bear':
            print(f"  ║  ⚠  This bucket only fires during BEAR regimes.{'':>62s} ║")
            print(f"  ║     When bear is detected, this becomes your HIGHEST CONVICTION play.{'':>39s} ║")
        elif bconf['regime'] == 'all':
            print(f"  ║  Works in all market conditions.{'':>76s} ║")

        print(f"  ╚{'═'*110}╝")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: PORTFOLIO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def section_4_portfolio_sim(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 4: PORTFOLIO SIMULATION WITH POSITION SIZING & STOP-LOSSES")
    print("#" * 115)

    STARTING_CAPITAL = 100_000
    SLIPPAGE = 0.001  # 0.1% per leg, 0.2% round trip

    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna()].copy()
        sub = sub.sort_values('date_dt')

        if len(sub) < 20:
            print(f"\n  {bname}: Insufficient signals ({len(sub)})")
            continue

        pos_pct = bconf['position_pct'] / 100
        max_pos = bconf['max_positions']
        stop_pct = bconf['stop_loss_pct'] / 100
        hold_months = bconf['hold_months']

        # Simulate
        capital = STARTING_CAPITAL
        positions = []  # list of (entry_date, exit_date, return_pct)
        active_positions = []  # (entry_date, symbol, capital_allocated)
        equity_curve = [(sub['date_dt'].iloc[0], capital)]
        max_equity = capital
        max_drawdown = 0
        total_trades = 0
        wins = 0
        losses_list = []
        wins_list = []

        for _, row in sub.iterrows():
            signal_date = row['date_dt']
            raw_return = row['return_3m'] / 100  # convert percentage to decimal

            # Clean out expired positions
            active_positions = [p for p in active_positions
                                if (signal_date - p[0]).days < hold_months * 30]

            # Check if we can take a new position
            if len(active_positions) >= max_pos:
                continue

            # Check if already holding this symbol
            held_symbols = {p[1] for p in active_positions}
            if row['symbol'] in held_symbols:
                continue

            # Apply slippage
            net_return = raw_return - 2 * SLIPPAGE

            # Apply stop loss
            if net_return < stop_pct:
                net_return = stop_pct - SLIPPAGE  # stopped out with slippage

            # Calculate P&L
            alloc = capital * pos_pct
            pnl = alloc * net_return
            capital += pnl
            total_trades += 1

            if net_return > 0:
                wins += 1
                wins_list.append(net_return * 100)
            else:
                losses_list.append(net_return * 100)

            active_positions.append((signal_date, row['symbol'], alloc))
            equity_curve.append((signal_date, capital))

            # Track drawdown
            if capital > max_equity:
                max_equity = capital
            dd = (capital - max_equity) / max_equity * 100
            if dd < max_drawdown:
                max_drawdown = dd

        # Compute metrics
        years = max(1, (sub['date_dt'].max() - sub['date_dt'].min()).days / 365.25)
        cagr = ((capital / STARTING_CAPITAL) ** (1 / years) - 1) * 100 if capital > 0 else -100
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        avg_win = np.mean(wins_list) if wins_list else 0
        avg_loss = np.mean(losses_list) if losses_list else 0
        pf = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf

        # Sharpe approximation (monthly returns from equity curve)
        eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        eq_df = eq_df.set_index('date').resample('M').last().dropna()
        if len(eq_df) > 2:
            monthly_rets = eq_df['equity'].pct_change().dropna()
            sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0
        else:
            sharpe = 0

        calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else np.inf

        print(f"\n  ╔{'═'*90}╗")
        print(f"  ║  {bname.upper():<86s}  ║")
        print(f"  ╠{'═'*90}╣")
        print(f"  ║  Starting Capital:   ${STARTING_CAPITAL:>12,.0f}{'':>57s}║")
        print(f"  ║  Ending Capital:     ${capital:>12,.0f}{'':>57s}║")
        print(f"  ║  Total Return:       {(capital/STARTING_CAPITAL - 1)*100:>+11.1f}%{'':>57s}║")
        print(f"  ║  CAGR:               {cagr:>+11.1f}%{'':>57s}║")
        print(f"  ║  Period:             {years:>11.1f} years{'':>53s}║")
        print(f"  ║{'─'*90}║")
        print(f"  ║  Total Trades:       {total_trades:>11,}{'':>58s}║")
        print(f"  ║  Win Rate:           {win_rate:>10.1f}%{'':>58s}║")
        print(f"  ║  Avg Win:            {avg_win:>+10.1f}%{'':>58s}║")
        print(f"  ║  Avg Loss:           {avg_loss:>+10.1f}%{'':>58s}║")
        print(f"  ║  Profit Factor:      {pf:>10.2f}{'':>59s}║")
        print(f"  ║{'─'*90}║")
        print(f"  ║  Max Drawdown:       {max_drawdown:>+10.1f}%{'':>58s}║")
        print(f"  ║  Sharpe Ratio:       {sharpe:>10.2f}{'':>59s}║")
        print(f"  ║  Calmar Ratio:       {calmar:>10.2f}{'':>59s}║")

        # Risk flags
        flags = []
        if sharpe < 0.5:
            flags.append("LOW SHARPE (<0.5)")
        if calmar < 0.3:
            flags.append("LOW CALMAR (<0.3)")
        if max_drawdown < -30:
            flags.append("SEVERE DRAWDOWN (>30%)")

        if flags:
            print(f"  ║{'─'*90}║")
            print(f"  ║  ⚠ RISK FLAGS:{'':>74s}║")
            for f in flags:
                print(f"  ║    • {f:<83s}║")
        else:
            print(f"  ║  ✓ All risk metrics within acceptable ranges{'':>43s}║")

        print(f"  ╚{'═'*90}╝")

    # === COMBINED PORTFOLIO ===
    print(f"\n  {'═'*90}")
    print(f"  COMBINED MULTI-BUCKET PORTFOLIO SIMULATION")
    print(f"  {'═'*90}")
    print(f"  Strategy: Allocate across all 3 buckets simultaneously")
    print(f"  - Quality Growth: 40% allocation (stable)")
    print(f"  - Bear Market Dip: 30% allocation (when active)")
    print(f"  - High-Growth Momentum: 30% allocation")
    print(f"  When Bear bucket is inactive, split its 30% → 15% each to other two")

    # Combined sim
    capital = STARTING_CAPITAL
    all_signals = []
    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna()].copy()
        sub['bucket'] = bname
        all_signals.append(sub)

    combined = pd.concat(all_signals).sort_values('date_dt')
    combined = combined.drop_duplicates(subset=['symbol', 'date'], keep='first')

    # Bucket allocations
    alloc_map = {
        "Quality Growth Compounder": 0.40,
        "Bear Market Dip Buy": 0.30,
        "High-Growth Momentum": 0.30,
    }

    total_trades = 0
    total_wins = 0
    max_equity = capital
    max_dd = 0
    eq_points = [(combined['date_dt'].iloc[0], capital)]
    pnl_list = []

    active_by_bucket = defaultdict(list)

    for _, row in combined.iterrows():
        bucket = row['bucket']
        bconf = BUCKETS[bucket]
        sig_date = row['date_dt']

        # Clean expired
        for b in active_by_bucket:
            active_by_bucket[b] = [p for p in active_by_bucket[b]
                                    if (sig_date - p[0]).days < BUCKETS[b]['hold_months'] * 30]

        # Check limits
        if len(active_by_bucket[bucket]) >= bconf['max_positions']:
            continue

        held_all = set()
        for b in active_by_bucket:
            for p in active_by_bucket[b]:
                held_all.add(p[1])
        if row['symbol'] in held_all:
            continue

        # Determine allocation
        bear_active = any(len(active_by_bucket["Bear Market Dip Buy"]) > 0
                         for _ in [1])  # simplified check
        if bucket == "Bear Market Dip Buy":
            bucket_alloc = alloc_map[bucket]
        elif not row['market_bearish']:
            # Bear bucket inactive, redistribute
            bucket_alloc = alloc_map[bucket] + 0.15
        else:
            bucket_alloc = alloc_map[bucket]

        pos_size = capital * bucket_alloc * (bconf['position_pct'] / 100)
        raw_ret = row['return_3m'] / 100
        net_ret = raw_ret - 0.002  # round-trip slippage

        if net_ret < bconf['stop_loss_pct'] / 100:
            net_ret = bconf['stop_loss_pct'] / 100 - 0.001

        pnl = pos_size * net_ret
        capital += pnl
        total_trades += 1
        pnl_list.append(net_ret * 100)
        if net_ret > 0:
            total_wins += 1

        active_by_bucket[bucket].append((sig_date, row['symbol']))
        eq_points.append((sig_date, capital))

        if capital > max_equity:
            max_equity = capital
        dd = (capital - max_equity) / max_equity * 100
        if dd < max_dd:
            max_dd = dd

    years = max(1, (combined['date_dt'].max() - combined['date_dt'].min()).days / 365.25)
    cagr = ((capital / STARTING_CAPITAL) ** (1 / years) - 1) * 100 if capital > 0 else -100
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    eq_df = pd.DataFrame(eq_points, columns=['date', 'equity'])
    eq_df = eq_df.set_index('date').resample('M').last().dropna()
    if len(eq_df) > 2:
        monthly_rets = eq_df['equity'].pct_change().dropna()
        sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0
    else:
        sharpe = 0

    print(f"\n  Starting Capital:   ${STARTING_CAPITAL:>12,.0f}")
    print(f"  Ending Capital:     ${capital:>12,.0f}")
    print(f"  Total Return:       {(capital/STARTING_CAPITAL - 1)*100:>+.1f}%")
    print(f"  CAGR:               {cagr:>+.1f}%")
    print(f"  Total Trades:       {total_trades:,}")
    print(f"  Win Rate:           {win_rate:.1f}%")
    print(f"  Max Drawdown:       {max_dd:+.1f}%")
    print(f"  Sharpe Ratio:       {sharpe:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: CURRENT ACTIONABLE STOCKS WITH BADGES
# ══════════════════════════════════════════════════════════════════════════════

def section_5_actionable(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 5: CURRENT ACTIONABLE STOCKS WITH FULL OVERLAYS")
    print("#" * 115)

    max_date = df['date_dt'].max()
    recent = df[df['date_dt'] >= max_date - pd.Timedelta(days=30)].copy()
    print(f"\n  Signal window: {(max_date - pd.Timedelta(days=30)).date()} to {max_date.date()}")
    print(f"  Total recent signals: {len(recent):,}")

    # Current regime
    latest = df.sort_values('date_dt').tail(100)
    bull_pct = latest['market_bullish'].mean() * 100
    bear_pct = latest['market_bearish'].mean() * 100
    regime = "BULL" if bull_pct > 50 else ("BEAR" if bear_pct > 50 else "NEUTRAL")

    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  REGIME: {regime:>8s}  │  Bull: {bull_pct:>5.1f}%  │  Bear: {bear_pct:>5.1f}%  │")
    print(f"  └────────────────────────────────────────────────────────┘")

    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](recent)
        if len(sub) > 0:
            sub = sub.sort_values('date_dt', ascending=False).drop_duplicates('symbol', keep='first')

        active = bconf['regime'] == 'all' or (bconf['regime'] == 'bear' and regime == 'BEAR')

        print(f"\n  ╔{'═'*108}╗")
        status = "ACTIVE" if active else "STANDBY"
        print(f"  ║  BUCKET {bconf['id']}: {bname.upper():<60s} [{status:>8s}] ║")
        print(f"  ║  {bconf['criteria']:<106s} ║")
        print(f"  ║  Position: {bconf['position_pct']:.1f}% | Hold: {bconf['hold_months']}mo | "
              f"Stop: {bconf['stop_loss_pct']:.0f}% | Max: {bconf['max_positions']} positions"
              f"{'':>40s} ║")
        print(f"  ╠{'═'*108}╣")

        if not active:
            print(f"  ║  ⏸  Bucket is on STANDBY — waiting for {bconf['regime'].upper()} regime"
                  f"{'':>48s} ║")
            print(f"  ╚{'═'*108}╝")
            continue

        if len(sub) == 0:
            print(f"  ║  No qualifying stocks in the last 30 days{'':>63s} ║")
            print(f"  ╚{'═'*108}╝")
            continue

        print(f"  ║  {'Symbol':<8s} {'Date':<11s} {'Signal':<18s} {'V2':>4s} {'LT':>4s} {'Fnd':>4s} "
              f"{'RSI':>5s} {'EV/EB':>6s} {'RevG':>5s} {'Badges':<30s} ║")
        print(f"  ║  {'─'*106} ║")

        sub_sorted = sub.sort_values('value_score_v2', ascending=False)
        for _, row in sub_sorted.head(15).iterrows():
            # Build badges
            badges = []
            if pd.notna(row.get('earnings_surprise_pct')) and row['earnings_surprise_pct'] > 0:
                if row['earnings_surprise_pct'] > 10:
                    badges.append("BIG-BEAT")
                else:
                    badges.append("BEAT")
            if pd.notna(row.get('earnings_surprise_pct')) and row['earnings_surprise_pct'] < -10:
                badges.append("BIG-MISS")

            if pd.notna(row.get('dcf_upside')) and row['dcf_upside'] > 20:
                badges.append("DCF+")
            elif pd.notna(row.get('dcf_upside')) and row['dcf_upside'] < -20:
                badges.append("DCF-")

            if row.get('concurrent_signals', 1) >= 3:
                badges.append(f"{int(row['concurrent_signals'])}SIG")

            if pd.notna(row.get('fcf_positive')) and row['fcf_positive'] == 1:
                badges.append("FCF+")

            badge_str = " ".join(badges)[:30]

            ev = row['ev_ebitda'] if pd.notna(row['ev_ebitda']) else 0
            rg = row['rev_growth'] if pd.notna(row['rev_growth']) else 0

            print(f"  ║  {row['symbol']:<8s} {str(row['date'])[:10]:<11s} "
                  f"{str(row['signal_type'])[:18]:<18s} "
                  f"{row['value_score_v2']:>4.0f} {row['lt_score']:>4.0f} "
                  f"{row['fundamentals_score']:>4.0f} {row['rsi']:>5.1f} "
                  f"{ev:>6.1f} {rg:>5.1f} {badge_str:<30s} ║")

        print(f"  ╚{'═'*108}╝")

    # === Cross-bucket overlap ===
    print(f"\n  ── Cross-Bucket Overlap (stocks appearing in multiple buckets) ──")
    all_symbols = {}
    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](recent)
        if len(sub) > 0:
            sub = sub.drop_duplicates('symbol')
            for sym in sub['symbol'].unique():
                if sym not in all_symbols:
                    all_symbols[sym] = []
                all_symbols[sym].append(bname)

    multi = {s: b for s, b in all_symbols.items() if len(b) > 1}
    if multi:
        for sym, buckets_list in sorted(multi.items(), key=lambda x: -len(x[1])):
            bucket_ids = [str(BUCKETS[b]['id']) for b in buckets_list]
            print(f"  {sym:<8s} → Buckets {', '.join(bucket_ids)} "
                  f"({', '.join([b.split()[0] for b in buckets_list])})")
    else:
        print(f"  No overlap found (each stock qualifies for only one bucket)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SECTOR ROTATION & STOP-LOSS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def section_6_advanced_trading(df):
    print("\n\n" + "#" * 115)
    print("  SECTION 6: SECTOR ROTATION + STOP-LOSS OPTIMIZATION")
    print("#" * 115)

    # 6A: Best sectors per bucket
    print(f"\n  --- 6A: Sector Performance by Bucket (ranked by win rate) ---")
    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna() & sub['sector'].notna()]
        if len(sub) < 30:
            continue

        print(f"\n  {bname}:")
        print(f"    {'Sector':<25s} {'N':>6s} {'Win%':>6s} {'Avg3M':>7s} {'PF':>5s} {'Verdict':>10s}")
        print(f"    {'─'*65}")

        sectors = sub.groupby('sector').agg(
            n=('return_3m', 'count'),
            win=('return_3m', lambda x: (x > 0).mean() * 100),
            avg=('return_3m', lambda x: x.clip(-90, 500).mean()),
        ).sort_values('win', ascending=False)

        for sec, row in sectors.iterrows():
            if row['n'] < 20:
                continue
            sec_sub = sub[sub['sector'] == sec]['return_3m'].clip(-90, 500)
            w = sec_sub[sec_sub > 0]
            l = sec_sub[sec_sub <= 0]
            pf = (w.mean() / abs(l.mean())) if len(l) > 0 and abs(l.mean()) > 0 else np.inf

            if row['win'] >= 60:
                verdict = "STRONG"
            elif row['win'] >= 52:
                verdict = "OK"
            else:
                verdict = "AVOID"

            print(f"    {sec[:25]:<25s} {row['n']:>5.0f} {row['win']:>5.1f}% {row['avg']:>+6.1f}% "
                  f"{pf:>4.2f} {verdict:>10s}")

    # 6B: Stop-loss optimization
    print(f"\n\n  --- 6B: Stop-Loss Sensitivity Analysis ---")
    print(f"  Testing different stop-loss levels to find optimal risk/reward balance")

    stop_levels = [-8, -10, -12, -15, -20, -25, None]  # None = no stop

    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna()].copy()
        if len(sub) < 30:
            continue

        print(f"\n  {bname}:")
        print(f"    {'Stop Loss':>10s} {'Win%':>6s} {'Avg Ret':>8s} {'PF':>5s} {'Stopped':>8s} {'Net Effect':>12s}")
        print(f"    {'─'*55}")

        raw_ret = sub['return_3m'].clip(-90, 500)
        base_avg = raw_ret.mean()

        for sl in stop_levels:
            if sl is None:
                adj = raw_ret.copy()
                label = "None"
                stopped = 0
            else:
                adj = raw_ret.copy()
                mask = adj < sl
                stopped = mask.sum()
                adj[mask] = sl
                label = f"{sl}%"

            w = (adj > 0).mean() * 100
            a = adj.mean()
            wins = adj[adj > 0]
            losses = adj[adj <= 0]
            pf = (wins.mean() / abs(losses.mean())) if len(losses) > 0 and abs(losses.mean()) > 0 else np.inf
            net = a - base_avg

            print(f"    {label:>10s} {w:>5.1f}% {a:>+7.1f}% {pf:>4.2f} {stopped:>7,} {net:>+11.1f}%")

    # 6C: Holding period analysis
    print(f"\n\n  --- 6C: Optimal Holding Period Analysis ---")
    for bname, bconf in BUCKETS.items():
        sub = bconf['filter'](df)
        sub = sub[sub['return_3m'].notna()].copy()
        if len(sub) < 30:
            continue

        print(f"\n  {bname}:")
        print(f"    {'Period':>10s} {'N':>6s} {'Win%':>6s} {'Avg':>8s} {'Med':>8s}")
        print(f"    {'─'*45}")

        for col, label in [('return_1w', '1 Week'), ('return_1m', '1 Month'),
                           ('return_3m', '3 Months'), ('return_6m', '6 Months'),
                           ('return_1y', '1 Year')]:
            vals = sub[col].dropna().clip(-90, 1000 if '1y' in col or '6m' in col else 500)
            if len(vals) < 20:
                continue
            w = (vals > 0).mean() * 100
            a = vals.mean()
            m = vals.median()
            print(f"    {label:>10s} {len(vals):>5,} {w:>5.1f}% {a:>+7.1f}% {m:>+7.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = datetime.now()
    print("=" * 115)
    print("  TRADING STRATEGY & FINAL VALIDATION")
    print(f"  Databases: {BACKTEST_DB}, {MAIN_DB}")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 115)

    print("\n  Loading and enriching data...")
    df = load_data()
    print(f"  Loaded {len(df):,} signals across {df['symbol'].nunique():,} stocks")
    print(f"  DCF coverage: {df['dcf_upside'].notna().sum():,}")
    print(f"  Earnings surprise coverage: {df['earnings_surprise_pct'].notna().sum():,}")
    print(f"  Date range: {df['date_dt'].min().date()} to {df['date_dt'].max().date()}")

    section_1_earnings_overlay(df)
    section_2_oos_validation(df)
    section_3_rule_cards(df)
    section_4_portfolio_sim(df)
    section_5_actionable(df)
    section_6_advanced_trading(df)

    elapsed = datetime.now() - start
    print(f"\n\n{'='*115}")
    print(f"  Trading Strategy analysis complete! Runtime: {elapsed}")
    print(f"{'='*115}")


if __name__ == "__main__":
    main()

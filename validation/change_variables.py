#!/usr/bin/env python3
"""
CHANGE / CATALYST variables for the big-winners search (Phase 1).

Hypothesis: big 1-year winners share a *change/expectation* signature a year earlier
(accelerating fundamentals, margin/return inflection, deleveraging, sector-relative
breakout, rating upgrades, earnings beats) — not static value/quality/size/vol levels,
which the prior phase proved are a vol/beta mirage vs matched controls.

This module BOLTS ONTO the cached moonshot panel (read-only) and the sealed loaders are
NOT modified. It owns:
  - its own quarterly loader (the sealed `load_moonshot_plus_fundamentals` exposes only
    TTM aggregates, not the per-quarter level series the slope/inflection vars need);
  - the new variable computations, with no-look-ahead merges:
      * fundamental-inflection (A): filing-date `asof`, merge_asof backward (filing<=date);
      * price/technical (B): contemporaneous within-date cross-section (uses panel momentum,
        already shift(21) lagged — no leakage);
      * catalyst (C): EVENT-dated grades/earnings, merged STRICTLY BEFORE the formation date.
  - its own r_<var> within-date percentile ranks (jump_screen.prep ranks only a closed set).

Signs are FROZEN in CHANGE_LIBRARY below, before any test (anti-overfit hygiene).

IMPORTANT: attach() returns a COPY and must NEVER re-pickle the cached panel
(/tmp/moonshot_search_panel.pkl) — that would poison the moonshot search cache.
"""
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import BACKTEST_DB
from validation.jump_screen import _rank   # within-date percentile rank (pct=True)


# ----------------------------------------------------------------------------
# Frozen library — name -> (group, sign, is_bool, rationale). +1 = higher -> more
# likely big winner. Signs/flags are PRE-DECLARED; no test may change them.
# ----------------------------------------------------------------------------
CHANGE_LIBRARY = {
    # A. fundamental-inflection (quarterly frame, filing-date asof)
    'rev_growth_accel_2d':  ('accel',     +1, False, '2nd diff of YoY TTM revenue growth (acceleration)'),
    'eps_growth_accel_2d':  ('accel',     +1, False, '2nd diff of YoY TTM EPS growth'),
    'op_margin_slope_8q':   ('inflect',   +1, False, 'operating-margin trend up over 8q (op leverage)'),
    'gross_margin_slope_8q':('inflect',   +1, False, 'gross-margin trend up over 8q (pricing/mix)'),
    'roa_slope_8q':         ('inflect',   +1, False, 'ROA trend up over 8q (capital efficiency)'),
    'ni_first_profitable':  ('turnaround',+1, True,  'TTM NI crossed <=0 -> >0 within trailing 4q'),
    'fcf_first_positive':   ('turnaround',+1, True,  'TTM FCF crossed <=0 -> >0 within trailing 4q'),
    'delev_slope_8q':       ('delever',   +1, False, 'debt/EBITDA falling over 8q (de-risking)'),
    'debt_assets_slope_4q': ('delever',   +1, False, 'debt/assets falling over 4q'),
    # B. price/technical change (panel cross-section)
    'rel_strength_sector_6m':('price',    +1, False, '6m momentum minus sector median (idiosyncratic strength)'),
    'breakout_from_base':   ('price',     +1, True,  'near 52w-high after a low-vol consolidation base'),
    # C. catalyst (event-dated, strictly-before)
    'net_upgrades_180d':    ('catalyst',  +1, False, 'net analyst upgrades-downgrades, trailing 180d'),
    'net_upgrades_180d_norm':('catalyst', +1, False, 'net upgrades / directional actions, trailing 180d'),
    'eps_surprise_pct':     ('catalyst',  +1, False, 'most-recent EPS surprise % (PEAD)'),
    'beat_streak':          ('catalyst',  +1, False, 'consecutive positive EPS surprises'),
}
CHANGE_VARS = list(CHANGE_LIBRARY)
BOOL_VARS = [k for k, v in CHANGE_LIBRARY.items() if v[2]]


# ----------------------------------------------------------------------------
# rolling OLS slope with a fixed integer x-ramp (0..k-1). For a full window the
# design matrix is constant, so the slope is a fixed linear filter w·y — fast and
# exact. min_periods=k (full window only) keeps the fixed weights valid.
# ----------------------------------------------------------------------------
def _slope_weights(k):
    x = np.arange(k, dtype=float)
    xc = x - x.mean()
    return xc / (xc ** 2).sum()


def _roll_slope(s, k):
    w = _slope_weights(k)
    return s.rolling(k, min_periods=k).apply(lambda y: float(np.dot(w, y)), raw=True)


def _first_cross_pos(ttm, lookback=4):
    """1.0 where TTM>0 now AND <=0 in ANY of the trailing `lookback` quarters (per symbol)."""
    now_pos = ttm > 0
    was_neg = pd.Series(False, index=ttm.index)
    for k in range(1, lookback + 1):
        was_neg = was_neg | (ttm.shift(k) <= 0)
    return (now_pos & was_neg).astype(float)


# ----------------------------------------------------------------------------
# A. Fundamental-inflection — own quarterly loader (mirrors the sealed loader's
# SQL/TTM/asof, then computes change-specific columns the sealed loader doesn't expose).
# ----------------------------------------------------------------------------
def build_change_fundamentals(db=BACKTEST_DB):
    conn = sqlite3.connect(str(db), timeout=60)
    f = pd.read_sql_query("""
        SELECT i.symbol, i.date AS period_end, i.filing_date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income, i.ebitda,
               i.eps_diluted AS eps,
               b.total_assets, b.total_debt,
               c.free_cash_flow
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets b ON i.symbol=b.symbol AND i.date=b.date
        LEFT JOIN historical_cash_flows c     ON i.symbol=c.symbol AND i.date=c.date
    """, conn)
    conn.close()

    pe = pd.to_datetime(f['period_end'], errors='coerce')
    asof = pd.to_datetime(f['filing_date'], errors='coerce')
    f['asof'] = asof.fillna(pe + pd.Timedelta(days=75))      # same no-look-ahead rule as the sealed loader
    f = f.dropna(subset=['asof']).sort_values(['symbol', 'period_end'])
    g = f.groupby('symbol')

    # TTM rolling-4 sums for flow items
    for col in ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda', 'eps', 'free_cash_flow']:
        f[f'{col}_ttm'] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum())

    rev = f['revenue_ttm']
    # acceleration = 2nd difference of YoY TTM growth (growth(t) - growth(t-4))
    def _yoy(s):
        base = s.shift(4)
        return (s / base.where(base > 0)) - 1
    g_rev = g['revenue_ttm'].transform(_yoy)
    g_eps = g['eps_ttm'].transform(_yoy)
    f['rev_growth_accel_2d'] = _accel(f, g_rev)
    f['eps_growth_accel_2d'] = _accel(f, g_eps)

    # margin / return slopes (use TTM-based ratios for stability)
    op_margin = f['operating_income_ttm'] / rev
    gross_margin = f['gross_profit_ttm'] / rev
    roa = f['net_income_ttm'] / f['total_assets']
    debt_ebitda = np.where(f['ebitda_ttm'] > 0, f['total_debt'] / f['ebitda_ttm'], np.nan)
    debt_assets = f['total_debt'] / f['total_assets']
    f['_op_margin'] = op_margin.replace([np.inf, -np.inf], np.nan)
    f['_gross_margin'] = gross_margin.replace([np.inf, -np.inf], np.nan)
    f['_roa'] = roa.replace([np.inf, -np.inf], np.nan)
    f['_debt_ebitda'] = pd.Series(debt_ebitda, index=f.index).replace([np.inf, -np.inf], np.nan)
    f['_debt_assets'] = debt_assets.replace([np.inf, -np.inf], np.nan)

    f['op_margin_slope_8q'] = g_apply_slope(f, '_op_margin', 8)
    f['gross_margin_slope_8q'] = g_apply_slope(f, '_gross_margin', 8)
    f['roa_slope_8q'] = g_apply_slope(f, '_roa', 8)
    f['delev_slope_8q'] = -g_apply_slope(f, '_debt_ebitda', 8)      # falling leverage -> positive
    f['debt_assets_slope_4q'] = -g_apply_slope(f, '_debt_assets', 4)

    # inflection crossings on TTM NI / FCF
    f['ni_first_profitable'] = g['net_income_ttm'].transform(lambda s: _first_cross_pos(s, 4))
    f['fcf_first_positive'] = g['free_cash_flow_ttm'].transform(lambda s: _first_cross_pos(s, 4))

    a_vars = ['rev_growth_accel_2d', 'eps_growth_accel_2d', 'op_margin_slope_8q',
              'gross_margin_slope_8q', 'roa_slope_8q', 'ni_first_profitable',
              'fcf_first_positive', 'delev_slope_8q', 'debt_assets_slope_4q']
    for c in a_vars:
        f[c] = f[c].replace([np.inf, -np.inf], np.nan)
    return f[['symbol', 'asof'] + a_vars]


def _accel(f, growth_series):
    """2nd difference: growth(t) - growth(t-4), per symbol."""
    return growth_series - growth_series.groupby(f['symbol']).shift(4)


def g_apply_slope(f, col, k):
    return f.groupby('symbol')[col].transform(lambda s: _roll_slope(s, k))


# ----------------------------------------------------------------------------
# B. Price/technical change — on the panel (contemporaneous within-date cross-section).
# ----------------------------------------------------------------------------
def attach_price_change(panel):
    p = panel.copy()
    if 'momentum_6m' in p.columns:
        # sector-relative 6m strength; fall back to per-date median if sector missing
        if 'sector' in p.columns:
            sec_med = p.groupby(['date', 'sector'])['momentum_6m'].transform('median')
            date_med = p.groupby('date')['momentum_6m'].transform('median')
            sec_med = sec_med.fillna(date_med)
        else:
            sec_med = p.groupby('date')['momentum_6m'].transform('median')
        p['rel_strength_sector_6m'] = p['momentum_6m'] - sec_med
    else:
        p['rel_strength_sector_6m'] = np.nan
    # breakout: near 52w high AND in the calm (bottom-40%) of vol this date
    if {'prox_52w_high', 'vol_60d'}.issubset(p.columns):
        vol_rank = _rank(p, 'vol_60d')
        p['breakout_from_base'] = ((p['prox_52w_high'] >= 0.95) & (vol_rank <= 0.40)).astype(float)
        p.loc[p['prox_52w_high'].isna() | vol_rank.isna(), 'breakout_from_base'] = np.nan
    else:
        p['breakout_from_base'] = np.nan
    return p


# ----------------------------------------------------------------------------
# C. Catalyst — EVENT-dated, merged STRICTLY BEFORE the formation date.
# ----------------------------------------------------------------------------
def attach_grades(panel, db=BACKTEST_DB, window_days=180):
    """net upgrades-downgrades over [date-window, date) per symbol, via per-symbol
    searchsorted (right edge strict '<' = look-ahead safe)."""
    p = panel.reset_index(drop=True).copy()
    grades = pd.read_sql_query(
        "SELECT symbol, date, action FROM historical_grades WHERE action IN ('upgrade','downgrade')",
        sqlite3.connect(str(db)))
    grades['d'] = pd.to_datetime(grades['date'], errors='coerce')
    grades = grades.dropna(subset=['d']).sort_values(['symbol', 'd'])

    net = np.full(len(p), np.nan)
    norm = np.full(len(p), np.nan)
    # prefix sums per symbol
    pref = {}
    for sym, sub in grades.groupby('symbol'):
        d = sub['d'].values.astype('datetime64[D]').astype(np.int64)
        up = (sub['action'].values == 'upgrade').astype(np.int64)
        dn = (sub['action'].values == 'downgrade').astype(np.int64)
        pref[sym] = (d, np.concatenate([[0], np.cumsum(up)]),
                     np.concatenate([[0], np.cumsum(dn)]))
    pdays = p['date'].values.astype('datetime64[D]').astype(np.int64)
    for sym, pos in p.groupby('symbol').groups.items():
        if sym not in pref:
            continue
        d, Pup, Pdn = pref[sym]
        idx = np.asarray(pos)
        fd = pdays[idx]
        r = np.searchsorted(d, fd, 'left')              # events strictly before formation date
        l = np.searchsorted(d, fd - window_days, 'left')
        nup = Pup[r] - Pup[l]
        ndn = Pdn[r] - Pdn[l]
        ndir = nup + ndn
        net[idx] = nup - ndn
        with np.errstate(invalid='ignore', divide='ignore'):
            norm[idx] = np.where(ndir > 0, (nup - ndn) / ndir, np.nan)
    p['net_upgrades_180d'] = net
    p['net_upgrades_180d_norm'] = norm
    return p


def attach_earnings(panel, db=BACKTEST_DB):
    """most-recent EPS surprise % and consecutive-beat streak as of STRICTLY BEFORE the
    formation date, via merge_asof(backward, allow_exact_matches=False)."""
    try:
        es = pd.read_sql_query(
            "SELECT symbol, date, eps_surprise_pct FROM earnings_surprises", sqlite3.connect(str(db)))
    except Exception:
        p = panel.copy()
        p['eps_surprise_pct'] = np.nan
        p['beat_streak'] = np.nan
        return p
    es['d'] = pd.to_datetime(es['date'], errors='coerce')
    es = es.dropna(subset=['d', 'eps_surprise_pct']).sort_values(['symbol', 'd'])
    beat = (es['eps_surprise_pct'] > 0).astype(int)
    es['beat_streak'] = beat.groupby(es['symbol']).transform(
        lambda x: x.groupby((x == 0).cumsum()).cumsum())
    ev = es[['symbol', 'd', 'eps_surprise_pct', 'beat_streak']].rename(columns={'d': 'asof_e'})

    p = panel.sort_values('date').copy()
    ev = ev.sort_values('asof_e')
    merged = pd.merge_asof(p, ev, left_on='date', right_on='asof_e', by='symbol',
                           direction='backward', allow_exact_matches=False,
                           tolerance=pd.Timedelta('400 days'))
    merged = merged.drop(columns=['asof_e'], errors='ignore')
    return merged


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
def attach(panel, db=BACKTEST_DB):
    """Return panel + all Phase-1 change/catalyst columns and their r_<var> ranks.
    No look-ahead; returns a copy; NEVER re-pickles the cached panel."""
    p = panel.copy()

    # A: fundamental-inflection via filing-date asof merge (filing <= date)
    cf = build_change_fundamentals(db)
    p = p.sort_values('date')
    cf = cf.sort_values('asof')
    p = pd.merge_asof(p, cf, left_on='date', right_on='asof', by='symbol',
                      direction='backward', tolerance=pd.Timedelta('365 days'))
    p = p.drop(columns=['asof'], errors='ignore')

    # B: price/technical change
    p = attach_price_change(p)
    # C: catalysts (strictly-before)
    p = attach_grades(p, db)
    p = attach_earnings(p, db)

    # own within-date percentile ranks for every non-bool change var
    for c in CHANGE_VARS:
        if c in BOOL_VARS:
            continue
        if c in p.columns:
            p[f'r_{c}'] = _rank(p, c)
    return p


if __name__ == '__main__':
    import time
    from validation import search_moonshot as SM
    t = time.time()
    panel = SM.build_search_panel()
    p = attach(panel)
    print(f"attached in {time.time()-t:.0f}s | panel {len(p):,} rows")
    print("\nchange-variable coverage (non-null frac):")
    for c in CHANGE_VARS:
        if c in p.columns:
            frac = p[c].notna().mean()
            extra = f"  mean={p[c].mean():+.3f}" if c not in BOOL_VARS else f"  rate={p[c].mean():.3f}"
            print(f"  {c:24s} {frac:.2f}{extra}")

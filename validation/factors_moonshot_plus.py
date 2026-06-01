#!/usr/bin/env python3
"""
Theory-first factor loader for the Moonshot-with-Quality search (Stage 0).

Superset of harness.load_moonshot_fundamentals — same TTM/asof machinery, but a
broad, CITED factor library partitioned into QUALITY (conditioning core), UPSIDE
(fundamental growth/momentum/size engine), and VALUE/GARP. Signs are pre-declared
from the literature (see FACTOR_LIBRARY) and frozen before any screen.

This module produces RAW factor columns only. All z-scaling, the quality composite,
the quality x upside interactions, and the tercile re-rank live in search_moonshot.py
so their moments can be fit on TRAIN-only dates (leakage-safe). EV/EBITDA and FCF
yield need market_cap, so they are finished in build_panel's want_momentum path via
add_market_factors() (called by the search after build_panel).

Citations: Novy-Marx (2013) gross profitability; Fama-French (2015) RMW/CMA; Asness-
Frazzini-Pedersen (2019) QMJ (profitability, safety, low uncertainty-of-profits);
Sloan (1996) accruals/cash-conversion; Jegadeesh-Titman (1993) momentum; Asness et al.
(2018) "Size Matters If You Control Your Junk"; Frazzini-Pedersen (2014) BAB / low-vol.
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKTEST_DB


# ----------------------------------------------------------------------------
# Factor library — name -> (group, sign, one-line theory). Signs are FROZEN here
# (pre-declared); the IC screen may not change them. +1 means higher value -> Q5.
# ----------------------------------------------------------------------------
FACTOR_LIBRARY = {
    # QUALITY (conditioning core)
    'gp_assets':         ('quality', +1, 'Novy-Marx gross profitability'),
    'operating_margin':  ('quality', +1, 'operating profitability'),
    'net_margin':        ('quality', +1, 'net profitability'),
    'roa':               ('quality', +1, 'return on assets'),
    'roe':               ('quality', +1, 'return on equity'),
    'cash_conversion':   ('quality', +1, 'FCF/NI earnings quality (Sloan accruals)'),
    'profit_stability':  ('quality', +1, 'QMJ low uncertainty-of-profits (neg std)'),
    'debt_assets':       ('quality', -1, 'QMJ safety: low leverage'),
    'debt_ebitda':       ('quality', -1, 'QMJ safety: low leverage/EBITDA'),
    'asset_growth':      ('quality', -1, 'FF5 CMA conservative investment'),
    'vol_60d':           ('quality', -1, 'BAB/low-vol safety (from build_panel)'),
    # UPSIDE engine (fundamental, NOT lottery)
    'revenue_growth_2q': ('upside',  +1, '6-month TTM revenue growth'),
    'revenue_growth_1y': ('upside',  +1, '1-year revenue growth'),
    'revenue_growth_3yr':('upside',  +1, '3-year revenue CAGR'),
    'ebitda_growth_1y':  ('upside',  +1, '1-year EBITDA growth'),
    'ebitda_growth_3yr': ('upside',  +1, '3-year EBITDA CAGR'),
    'ni_growth_1y':      ('upside',  +1, '1-year net-income growth'),
    'ni_growth_3yr':     ('upside',  +1, '3-year net-income CAGR'),
    'eps_growth_3yr':    ('upside',  +1, '3-year EPS CAGR'),
    'growth_acceleration':('upside', +1, 'rev growth 1y minus 3y (inflection)'),
    'momentum_12_1':     ('upside',  +1, 'Jegadeesh-Titman 12-1 momentum'),
    'momentum_6m':       ('upside',  +1, '6-1 momentum'),
    'momentum_3m':       ('upside',  +1, '3-1 momentum'),
    'prox_52w_high':     ('upside',  +1, '52-week-high proximity'),
    'small_cap':         ('upside',  +1, 'size (only after controlling junk)'),
    # VALUE / GARP (finished in add_market_factors with market_cap)
    'ev_ebitda':         ('value',   -1, 'GARP: cheap EV/EBITDA'),
    'fcf_yield':         ('value',   +1, 'GARP: high FCF yield'),
}

# Columns the loader must carry through build_panel's merge so market factors + the
# gate-free junk floor can read them off the panel.
PASSTHROUGH = ['revenue_ttm', 'net_income_ttm', 'ebitda_ttm', 'free_cash_flow_ttm',
               'net_debt', 'total_assets', 'total_debt']

# Factor columns produced directly by this loader (fundamentals-only). Price factors
# (momentum_*, prox_52w_high, small_cap, vol_60d) come from build_panel; ev_ebitda /
# fcf_yield from add_market_factors.
_LOADER_FACTORS = ['gp_assets', 'operating_margin', 'net_margin', 'roa', 'roe',
                   'cash_conversion', 'profit_stability', 'debt_assets', 'debt_ebitda',
                   'asset_growth', 'revenue_growth_2q', 'revenue_growth_1y',
                   'revenue_growth_3yr', 'ebitda_growth_1y', 'ebitda_growth_3yr',
                   'ni_growth_1y', 'ni_growth_3yr', 'eps_growth_3yr', 'growth_acceleration']


def _cagr3(s):
    """3yr CAGR (12 quarters back), per-row, no look-ahead. NaN if base<=0 or numerator<0."""
    base = s.shift(12)
    with np.errstate(invalid='ignore'):
        out = (s / base.where(base > 0)) ** (1 / 3) - 1
    return out


def _growth_1y(s):
    """1yr TTM growth, NaN if base<=0 (avoid sign-flip artifacts)."""
    base = s.shift(4)
    return (s / base.where(base > 0)) - 1


def load_moonshot_plus_fundamentals(db=BACKTEST_DB, date_basis='filing'):
    """Broad theory-first factor panel. date_basis='filing' (no look-ahead) sets asof."""
    conn = sqlite3.connect(str(db), timeout=60)
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date AS period_end, i.filing_date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income, i.ebitda,
               i.eps_diluted AS eps, i.weighted_avg_shares_diluted AS shares,
               b.total_assets, b.total_debt, b.total_equity, b.net_debt, b.cash_and_equivalents,
               c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets b ON i.symbol=b.symbol AND i.date=b.date
        LEFT JOIN historical_cash_flows c     ON i.symbol=c.symbol AND i.date=c.date
    """, conn)
    conn.close()

    fund['shares'] = fund['shares'].where(fund['shares'] > 0, np.nan)
    pe = pd.to_datetime(fund['period_end'], errors='coerce')
    if date_basis == 'filing':
        asof = pd.to_datetime(fund['filing_date'], errors='coerce')
        fund['asof'] = asof.fillna(pe + pd.Timedelta(days=75))
    else:
        fund['asof'] = pe
    fund = fund.dropna(subset=['asof']).sort_values(['symbol', 'period_end'])
    g = fund.groupby('symbol')

    # TTM rolling-4 sums for flow items.
    for col in ['revenue', 'gross_profit', 'operating_income', 'net_income', 'ebitda',
                'eps', 'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum())

    rev, ta, te = fund['revenue_ttm'], fund['total_assets'], fund['total_equity']
    # QUALITY
    fund['gp_assets'] = fund['gross_profit_ttm'] / ta
    fund['operating_margin'] = fund['operating_income_ttm'] / rev
    fund['net_margin'] = fund['net_income_ttm'] / rev
    fund['roa'] = fund['net_income_ttm'] / ta
    fund['roe'] = np.where(te > 0, fund['net_income_ttm'] / te, np.nan)
    fund['cash_conversion'] = np.where(fund['net_income_ttm'] > 0,
                                       fund['free_cash_flow_ttm'] / fund['net_income_ttm'], np.nan)
    # profit_stability = -std of per-quarter net margin over trailing 8 quarters
    # (QMJ "low uncertainty of profits"); higher (less negative) = more stable.
    fund['_q_net_margin'] = (fund['net_income'] / fund['revenue']).replace([np.inf, -np.inf], np.nan)
    fund['profit_stability'] = -fund.groupby('symbol')['_q_net_margin'].transform(
        lambda s: s.rolling(8, min_periods=6).std())
    fund['debt_assets'] = fund['total_debt'] / ta
    fund['debt_ebitda'] = np.where(fund['ebitda_ttm'] > 0,
                                   fund['total_debt'] / fund['ebitda_ttm'], np.nan)
    fund['asset_growth'] = g['total_assets'].transform(lambda s: s.pct_change(4, fill_method=None))
    # UPSIDE (fundamental)
    fund['revenue_growth_2q'] = g['revenue_ttm'].transform(lambda s: (s / s.shift(2).where(s.shift(2) > 0)) - 1)
    fund['revenue_growth_1y'] = g['revenue_ttm'].transform(_growth_1y)
    fund['revenue_growth_3yr'] = g['revenue_ttm'].transform(_cagr3)
    fund['ebitda_growth_1y'] = g['ebitda_ttm'].transform(_growth_1y)
    fund['ebitda_growth_3yr'] = g['ebitda_ttm'].transform(_cagr3)
    fund['ni_growth_1y'] = g['net_income_ttm'].transform(_growth_1y)
    fund['ni_growth_3yr'] = g['net_income_ttm'].transform(_cagr3)
    fund['eps_growth_3yr'] = g['eps_ttm'].transform(_cagr3)
    fund['growth_acceleration'] = fund['revenue_growth_1y'] - fund['revenue_growth_3yr']

    for col in _LOADER_FACTORS:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Keep rows with the basic fundamentals present (don't drop on individual factors —
    # each factor's IC uses its own non-null subset; add_score neutral-fills the rest).
    fund = fund.dropna(subset=['total_assets', 'revenue_ttm'])
    keep = (['symbol', 'period_end', 'filing_date', 'asof', 'shares']
            + _LOADER_FACTORS + PASSTHROUGH)
    keep = list(dict.fromkeys([c for c in keep if c in fund.columns]))
    return fund[keep]


def add_market_factors(panel):
    """Finish the cap-dependent VALUE factors AFTER build_panel finalizes market_cap.
    ev_ebitda = (market_cap + net_debt) / ebitda_ttm (mask ebitda_ttm>0);
    fcf_yield  = free_cash_flow_ttm / market_cap. Returns the panel with both columns."""
    panel = panel.copy()
    mc = panel['market_cap']
    if 'ebitda_ttm' in panel and 'net_debt' in panel:
        ev = mc + panel['net_debt'].fillna(0)
        panel['ev_ebitda'] = np.where(panel['ebitda_ttm'] > 0, ev / panel['ebitda_ttm'], np.nan)
    if 'free_cash_flow_ttm' in panel:
        panel['fcf_yield'] = np.where(mc > 0, panel['free_cash_flow_ttm'] / mc, np.nan)
    for c in ['ev_ebitda', 'fcf_yield']:
        if c in panel:
            panel[c] = panel[c].replace([np.inf, -np.inf], np.nan)
    return panel


if __name__ == '__main__':
    import time
    t = time.time()
    f = load_moonshot_plus_fundamentals()
    print(f"loaded {len(f):,} rows in {time.time()-t:.0f}s")
    print("factor coverage (non-null frac):")
    for c in _LOADER_FACTORS:
        print(f"  {c:22s} {f[c].notna().mean():.2f}")

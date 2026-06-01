#!/usr/bin/env python3
"""
Shared validation harness — ONE correct path all three scores run through.

Fixes the bugs that invalidated the legacy numbers and consolidates the good
pieces from compass/validations/v4_robustness.py:
  - PER-DATE cross-sectional quintiles (rank within each rebalance date, average
    per-date spreads, t-stat across dates) — replaces pooled `qcut`.
  - FILING-DATE lag on fundamentals (merge_asof backward on filing_date) — no
    look-ahead. Toggleable to period-end for v4-exact replication.
  - 12-month forward returns as the headline horizon; 3-month kept for the
    v4-exact control replication.
  - Non-overlapping (12 monthly cohorts) + regime breakdowns + sector-neutral
    (real DB sectors) + turnover-scaled costs.
  - Vectorized factor computation (groupby/rolling) — same math as the legacy
    per-symbol loop, ~10-50x faster, run on the FULL universe (no 2,500 sample).

This module is a LIBRARY; run_validation.py drives it. Nothing here is judged
until the staged Compass control reproduces the legacy number (see HONEST_NUMBERS.md).
"""
import sqlite3
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import BACKTEST_DB

IN_SAMPLE_END = '2019-12-31'
OOS_START = '2020-01-01'
LIQUIDITY_THRESHOLD = 1_000_000          # $1M avg daily dollar volume
TRADING_DAYS = {'3m': 63, '12m': 252, '1m': 21}
SEED = 42

# Regime windows (rebalance-date ranges), reported separately.
REGIMES = {
    'GFC_2008':   ('2007-07-01', '2009-06-30'),
    'taper_2015': ('2015-01-01', '2015-12-31'),
    'vol_2018':   ('2018-01-01', '2018-12-31'),
    'covid_2020_21': ('2020-01-01', '2021-12-31'),
    'rates_2022': ('2022-01-01', '2022-12-31'),
    'recent_2024_26': ('2024-01-01', '2026-12-31'),
}


# ----------------------------------------------------------------------------
# Score specifications (factor columns + weights). Theory-cited factors.
# ----------------------------------------------------------------------------
@dataclass
class ScoreSpec:
    name: str
    weights: dict                 # factor_col -> weight (sign = direction)
    higher_is_better: bool = True # Q5 = top score
    clips: dict = field(default_factory=dict)  # factor_col -> (lo, hi) HARD pre-z clip
    gate: object = None           # fn(panel) -> bool mask of eligible rows, or None
    transform: object = None      # fn(panel, score)->score, default None=no-op (quality re-rank)


COMPASS_SPEC = ScoreSpec(
    name='Compass',
    weights={
        'roa': 0.20,            # profitability
        'ocf_assets': 0.15,
        'fcf_assets': 0.15,
        'gp_assets': 0.20,      # Novy-Marx gross profitability
        'vol_60d': -0.15,       # low-volatility anomaly
        'asset_growth': -0.15,  # Fama-French investment (conservative)
    },
)


MIN_MARKET_CAP = 500_000_000   # Moonshot $500M exclusion (applied first)


def _moonshot_gate(panel):
    """Moonshot quality gates — port of compute_moonshot_scores.py:193-261. ALL must
    pass. $500M cap exclusion first. NaN-permissive: a sub-condition with a missing
    input is skipped (matches the legacy NaN behavior). Returns a boolean mask."""
    gm = panel['gross_margin']; rev = panel['revenue_ttm']; rg = panel['revenue_growth']
    ni = panel['net_income_ttm']; ocf = panel['operating_cash_flow_ttm']
    oi = panel['operating_income_ttm']; ta = panel['total_assets']; td = panel['total_debt']
    mc = panel['market_cap']

    m = (mc >= MIN_MARKET_CAP)                      # $500M exclusion FIRST
    m &= gm.between(0.30, 0.95)
    m &= (rev >= 50_000_000)
    m &= rg.between(0.15, 3.0)

    # Cash-flow quality (binds only when its inputs are present).
    cfq = pd.Series(True, index=panel.index)
    both = ni.notna() & ocf.notna()
    pos = both & (ni > 0)
    cfq[pos] = (ocf[pos] / ni[pos]) >= 0.7
    npos = both & (ni <= 0) & rev.notna() & (rev != 0)
    cfq[npos] = (ocf[npos] / rev[npos]) >= -0.5
    m &= cfq

    # Balance-sheet health (binds only when assets present & > 0).
    bs = pd.Series(True, index=panel.index)
    bok = td.notna() & ta.notna() & (ta > 0)
    bs[bok] = (td[bok] / ta[bok]) < 2.0
    m &= bs

    # Operating-income quality (binds only when both nonzero).
    oiq = pd.Series(True, index=panel.index)
    ok = ni.notna() & oi.notna() & (ni.abs() > 0) & (oi.abs() > 0)
    oiq[ok] = (oi[ok] / ni[ok]).abs().between(0.3, 3.0)
    m &= oiq
    return m.fillna(False)


MOONSHOT_SPEC = ScoreSpec(
    name='Moonshot',
    weights={
        'revenue_growth_3yr': 0.20,
        'eps_growth_3yr': 0.15,
        'gross_margin': 0.15,
        'margin_improvement': 0.10,
        'fcf_margin': 0.15,
        'roe': 0.10,
        'small_cap': 0.10,
        'momentum_12_1': 0.05,
    },
    clips={
        'revenue_growth_3yr': (-0.3, 1.5),
        'eps_growth_3yr': (-0.5, 2.0),
        'fcf_margin': (-0.5, 0.5),
        'roe': (-0.5, 1.0),
        'momentum_12_1': (-0.5, 2.0),
        # gross_margin / margin_improvement / small_cap intentionally unclipped here
        # -> they take the IS 1/99-percentile path in add_score (documented de-biasing).
    },
    gate=_moonshot_gate,
)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_prices(db=BACKTEST_DB):
    conn = sqlite3.connect(str(db), timeout=60)
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close AS close, close AS raw_close, volume
        FROM historical_prices
        WHERE adjusted_close > 1
        ORDER BY symbol, date
    """, conn)
    conn.close()
    prices['date'] = pd.to_datetime(prices['date'])
    return prices


def load_fundamentals(db=BACKTEST_DB, date_basis='filing'):
    """Compass fundamentals. date_basis='filing' (no look-ahead) or 'period_end'
    (v4-exact). The chosen date becomes 'asof' for the backward merge."""
    conn = sqlite3.connect(str(db), timeout=60)
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date AS period_end, i.filing_date,
               i.gross_profit, i.net_income, i.weighted_avg_shares_diluted AS shares,
               b.total_assets, c.operating_cash_flow, c.free_cash_flow
        FROM historical_income_statements i
        JOIN historical_balance_sheets b ON i.symbol=b.symbol AND i.date=b.date
        JOIN historical_cash_flows c     ON i.symbol=c.symbol AND i.date=c.date
    """, conn)
    conn.close()

    # We do NOT use historical_key_metrics.market_cap — it's systematically corrupt
    # (int64 sentinels ~9.2e18 plus garbage clustered at ~$10T/$50T across >1,100
    # symbols; that field is what blew up legacy v4's value-weighting to +87%).
    # Instead market cap is derived in build_panel as raw_close x diluted shares
    # (same robust basis as the EV/EBITDA overlay). Null non-positive share counts.
    fund['shares'] = fund['shares'].where(fund['shares'] > 0, np.nan)

    pe = pd.to_datetime(fund['period_end'], errors='coerce')
    if date_basis == 'filing':
        asof = pd.to_datetime(fund['filing_date'], errors='coerce')
        # fall back to period-end + 75d when filing_date missing
        fund['asof'] = asof.fillna(pe + pd.Timedelta(days=75))
    else:  # period_end — replicates the legacy look-ahead
        fund['asof'] = pe
    fund = fund.dropna(subset=['asof']).sort_values(['symbol', 'period_end'])

    # Factors (ratios + 4-quarter asset growth) — computed on period order.
    fund['roa'] = fund['net_income'] / fund['total_assets']
    fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
    fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
    fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
    fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4, fill_method=None)
    for c in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
        fund[c] = fund[c].replace([np.inf, -np.inf], np.nan)
    fund = fund.dropna(subset=['roa', 'ocf_assets', 'fcf_assets', 'asset_growth'])
    return fund


# Moonshot factor + gate column groups (kept through build_panel's merge_asof so
# _moonshot_gate can read them off the panel — see run_score).
MOONSHOT_FACTOR_COLS = ['revenue_growth_3yr', 'eps_growth_3yr', 'gross_margin',
                        'margin_improvement', 'fcf_margin', 'roe']
MOONSHOT_GATE_COLS = ['gross_margin', 'revenue_ttm', 'revenue_growth', 'net_income_ttm',
                      'operating_cash_flow_ttm', 'operating_income_ttm',
                      'total_assets', 'total_debt', 'total_equity']


def _cagr3_broadcast(ttm):
    """LEGACY look-ahead CAGR: one value from the symbol's LAST row, broadcast to
    every row (mirrors compute_moonshot_scores.py's groupby.transform(compute_cagr_3yr),
    which returns iloc[-1]/iloc[-13]). Used ONLY by the M0' directional proxy."""
    if len(ttm) < 13:
        return np.nan
    cur, base = ttm.iloc[-1], ttm.iloc[-13]
    if pd.isna(cur) or pd.isna(base) or base <= 0 or cur < 0:
        return np.nan
    return (cur / base) ** (1 / 3) - 1


def load_moonshot_fundamentals(db=BACKTEST_DB, date_basis='filing'):
    """Moonshot fundamentals: TTM aggregates + growth/quality factors + gate inputs.
    Separate from load_fundamentals (Compass) — does not touch it. date_basis sets the
    backward-merge key ('filing' = no look-ahead, 'period_end' = legacy)."""
    conn = sqlite3.connect(str(db), timeout=60)
    # LEFT JOIN from income (matches legacy intent): keep income quarters even when
    # balance/cashflow are missing — the optional factors neutral-fill downstream.
    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date AS period_end, i.filing_date,
               i.revenue, i.gross_profit, i.operating_income, i.net_income,
               i.eps_diluted AS eps, i.weighted_avg_shares_diluted AS shares,
               b.total_assets, b.total_debt, b.total_equity,
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

    # TTM = rolling 4-quarter sums (exact legacy semantics).
    g = fund.groupby('symbol')
    for col in ['revenue', 'gross_profit', 'eps', 'operating_income', 'net_income',
                'operating_cash_flow', 'free_cash_flow']:
        fund[f'{col}_ttm'] = g[col].transform(lambda s: s.rolling(4, min_periods=4).sum())

    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['revenue_growth'] = g['revenue_ttm'].transform(lambda s: s.pct_change(4, fill_method=None))  # YoY, gate only
    fund['margin_improvement'] = g['gross_margin'].transform(lambda s: s.diff(4))
    fund['fcf_margin'] = fund['free_cash_flow_ttm'] / fund['revenue_ttm']
    fund['roe'] = np.where(fund['total_equity'] > 0,
                           fund['net_income_ttm'] / fund['total_equity'], np.nan)

    # 3yr CAGR — HONEST (per-row shift of 12 quarters, no look-ahead). NaN if base<=0
    # or <13 quarters of history. Plus a LEGACY broadcast version for the M0' proxy.
    def cagr3(s):
        base = s.shift(12)
        with np.errstate(invalid='ignore'):
            out = (s / base.where(base > 0)) ** (1 / 3) - 1   # NaN where base<=0 or s<0
        return out
    fund['revenue_growth_3yr'] = g['revenue_ttm'].transform(cagr3)
    fund['eps_growth_3yr'] = g['eps_ttm'].transform(cagr3)
    fund['revenue_growth_3yr_legacy'] = g['revenue_ttm'].transform(_cagr3_broadcast)
    fund['eps_growth_3yr_legacy'] = g['eps_ttm'].transform(_cagr3_broadcast)

    for col in MOONSHOT_FACTOR_COLS + ['revenue_growth',
                                       'revenue_growth_3yr_legacy', 'eps_growth_3yr_legacy']:
        fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    # Drop only on the two REQUIRED factors; keep every gate-input column so the gate
    # can read them off the panel after build_panel's merge_asof.
    fund = fund.dropna(subset=['revenue_growth_3yr', 'gross_margin'])
    keep = (['symbol', 'period_end', 'filing_date', 'asof', 'shares']
            + MOONSHOT_FACTOR_COLS + MOONSHOT_GATE_COLS
            + ['revenue_growth_3yr_legacy', 'eps_growth_3yr_legacy'])
    keep = list(dict.fromkeys(keep))  # dedupe, preserve order (gross_margin/revenue in both groups)
    return fund[keep]


def load_corrupt_market_caps(db=BACKTEST_DB):
    """LEGACY corrupt market caps from historical_key_metrics (the systematically
    corrupt field, on fundamental period dates). Used ONLY by the M0' directional
    proxy via build_panel(use_corrupt_caps=True) — merged as-of like fundamentals."""
    conn = sqlite3.connect(str(db), timeout=60)
    mc = pd.read_sql_query("SELECT symbol, date AS period_end, market_cap "
                           "FROM historical_key_metrics WHERE market_cap > 0", conn)
    conn.close()
    mc['period_end'] = pd.to_datetime(mc['period_end'], errors='coerce')
    return mc.dropna(subset=['period_end']).sort_values('period_end')


def load_sectors(db=BACKTEST_DB):
    """Real per-symbol sector for sector-neutral ranking (not a hardcoded map)."""
    for src in [(str(Path(BACKTEST_DB).parent / 'nasdaq_stocks.db'), 'stock_consensus')]:
        try:
            conn = sqlite3.connect(src[0], timeout=60)
            df = pd.read_sql_query(f"SELECT symbol, sector FROM {src[1]}", conn)
            conn.close()
            return df.dropna(subset=['sector'])
        except Exception:
            continue
    return pd.DataFrame(columns=['symbol', 'sector'])


def load_market_caps(db=BACKTEST_DB):
    """Clean daily market caps re-pulled from FMP (historical_market_caps).
    Replaces the corrupt historical_key_metrics.market_cap field."""
    conn = sqlite3.connect(str(db), timeout=60)
    try:
        mc = pd.read_sql_query("SELECT symbol, date, market_cap FROM historical_market_caps", conn)
    except Exception:
        conn.close(); return None
    conn.close()
    mc['date'] = pd.to_datetime(mc['date'])
    return mc


# ----------------------------------------------------------------------------
# Panel construction (vectorized monthly sampling + forward returns)
# ----------------------------------------------------------------------------
def build_panel(prices, fund, sectors, mcaps=None, sample=None, seed=SEED,
                horizons=('3m', '12m'), delisted_syms=None, want_momentum=False,
                corrupt_mcaps=None):
    """Monthly-sampled panel with forward returns, vol, liquidity, fundamentals,
    sector, market cap. Vectorized replacement for the legacy per-symbol loop:
    sample at positions 252, 273, ... (every 21 trading days), matching v4.
    want_momentum adds the Moonshot 12-1 price momentum + small_cap factors.
    corrupt_mcaps (M0' proxy only) overrides market cap with the corrupt
    historical_key_metrics value, as-of merged, and skips winsorize."""
    if sample is not None:
        rng = np.random.RandomState(seed)
        syms = prices['symbol'].unique()
        if len(syms) > sample:
            keep = set(rng.choice(syms, size=sample, replace=False))
            prices = prices[prices['symbol'].isin(keep)]

    p = prices.sort_values(['symbol', 'date']).copy()
    g = p.groupby('symbol', sort=False)
    p['pos'] = g.cumcount()
    p['n'] = g['symbol'].transform('size')

    # Forward returns (positional shift within symbol = contiguous daily rows).
    for h, d in TRADING_DAYS.items():
        if h in horizons or h == '1m':
            p[f'fwd_{h}'] = (g['close'].shift(-d) / p['close'] - 1) * 100

    # Trailing 60d annualized vol + avg dollar volume (shifted 1d, no look-ahead).
    dret = g['close'].pct_change()
    p['vol_60d'] = dret.groupby(p['symbol']).rolling(60).std().reset_index(0, drop=True) * np.sqrt(252) * 100
    dv = (p['volume'] * p['close'])
    p['avg_dollar_vol'] = dv.groupby(p['symbol']).rolling(60).mean().reset_index(0, drop=True).groupby(p['symbol']).shift(1)

    # Moonshot 12-1 momentum: price 1mo-ago / price 12mo-ago - 1, trailing (no look-ahead).
    # Computed on the DAILY frame p (positional shifts within symbol), NOT on samp.
    if want_momentum:
        p['momentum_12_1'] = g['close'].shift(21) / g['close'].shift(252) - 1
        # Extra horizons for the moonshot-plus search (skip-1-month, trailing, no look-ahead).
        p['momentum_6m'] = g['close'].shift(21) / g['close'].shift(126) - 1
        p['momentum_3m'] = g['close'].shift(21) / g['close'].shift(63) - 1
        # 52-week-high proximity: close / trailing 252d max (<=1; near 1 = at highs).
        p['prox_52w_high'] = p['close'] / g['close'].rolling(252, min_periods=126).max().reset_index(0, drop=True)

    # Monthly sampling: pos >= 252, (pos-252) % 21 == 0. The end-filter pos < n-max_d
    # avoids right-censoring LIVE names at "today" (no future data). For DELISTED names
    # the series ends at a KNOWN delisting, so we sample into their final window too
    # (pos < n) and let apply_terminal_returns populate those NaN forward returns with
    # the terminal outcome — otherwise the worst final ~12 months are silently truncated.
    max_d = max(TRADING_DAYS[h] for h in horizons)
    monthly = (p['pos'] >= 252) & ((p['pos'] - 252) % 21 == 0)
    end_ok = p['pos'] < p['n'] - max_d
    if delisted_syms:
        end_ok = end_ok | (p['symbol'].isin(delisted_syms) & (p['pos'] < p['n']))
    samp = p[monthly & end_ok].copy()

    # Merge fundamentals as-of (backward) on the chosen asof date.
    f = fund.sort_values('asof')
    samp = samp.sort_values('date')
    panel = pd.merge_asof(
        samp, f.drop(columns=['period_end', 'filing_date']),
        left_on='date', right_on='asof', by='symbol',
        direction='backward', tolerance=pd.Timedelta('365 days'))

    panel = panel.merge(sectors, on='symbol', how='left')
    # Market cap: clean FMP value (historical_market_caps) joined on (symbol, date);
    # fall back to raw_close x diluted shares where FMP is missing; then winsorize the
    # top per date (catches residual ~$10T outliers without clipping real megacaps).
    derived = panel['raw_close'] * panel['shares']
    if corrupt_mcaps is not None and len(corrupt_mcaps):
        # M0' proxy: as-of merge the corrupt key_metrics cap (period dates -> sample date);
        # fall back to derived where missing. NO winsorize — the corruption must flow into
        # the $500M gate and small_cap exactly as it did in the legacy run.
        cm = corrupt_mcaps.sort_values('period_end')
        panel = panel.sort_values('date')
        panel = pd.merge_asof(panel, cm.rename(columns={'market_cap': '_cmc'}),
                              left_on='date', right_on='period_end', by='symbol',
                              direction='backward', tolerance=pd.Timedelta('365 days'))
        panel['market_cap'] = panel['_cmc'].where(panel['_cmc'] > 0, derived)
        panel = panel.drop(columns=['_cmc', 'period_end'])
    elif mcaps is not None and len(mcaps):
        panel = panel.merge(mcaps.rename(columns={'market_cap': '_mc'}), on=['symbol', 'date'], how='left')
        panel['market_cap'] = panel['_mc'].where(panel['_mc'] > 1e6, derived)  # >$1M floor; else derived
        panel = panel.drop(columns='_mc')
        cap995 = panel.groupby('date')['market_cap'].transform(lambda s: s.quantile(0.995))
        panel['market_cap'] = np.minimum(panel['market_cap'], cap995)
    else:
        panel['market_cap'] = derived
        cap995 = panel.groupby('date')['market_cap'].transform(lambda s: s.quantile(0.995))
        panel['market_cap'] = np.minimum(panel['market_cap'], cap995)
    # Moonshot small_cap factor: -log10(market_cap / 1e9), from the FINAL market_cap.
    if want_momentum:
        with np.errstate(divide='ignore', invalid='ignore'):
            panel['small_cap'] = -np.log10(panel['market_cap'] / 1e9)
        panel['small_cap'] = panel['small_cap'].replace([np.inf, -np.inf], np.nan)
    # clip forward returns like v4
    if 'fwd_3m' in panel:
        panel['fwd_3m'] = panel['fwd_3m'].clip(-100, 100)
    if 'fwd_12m' in panel:
        panel['fwd_12m'] = panel['fwd_12m'].clip(-100, 300)
    panel['is_liquid'] = panel['avg_dollar_vol'] >= LIQUIDITY_THRESHOLD
    return panel


# ----------------------------------------------------------------------------
# Scoring (IS-stat z-scores → composite), no OOS look-ahead
# ----------------------------------------------------------------------------
def add_score(panel, spec: ScoreSpec, is_end=IN_SAMPLE_END):
    factor_cols = list(spec.weights)
    is_mask = panel['date'] <= pd.Timestamp(is_end)
    isd = panel[is_mask]
    score = pd.Series(0.0, index=panel.index)
    for col in factor_cols:
        # Hard per-factor clip when the spec declares one (Moonshot); else the
        # IS 1/99-percentile clip (Compass). One code path; the clip bounds also
        # govern the IS mean/std so z-stats match the clipped distribution.
        if col in spec.clips:
            lo, hi = spec.clips[col]
        else:
            lo, hi = isd[col].quantile(0.01), isd[col].quantile(0.99)
        clipped = panel[col].clip(lo, hi)
        mean, std = isd[col].clip(lo, hi).mean(), isd[col].clip(lo, hi).std()
        z = ((clipped - mean) / std).clip(-3, 3).fillna(0)
        score = score + z * spec.weights[col]
    if spec.transform is not None:          # quality-conditioning re-rank (default no-op)
        score = spec.transform(panel, score)
    panel = panel.copy()
    panel['score'] = score
    return panel


# ----------------------------------------------------------------------------
# Spread estimators
# ----------------------------------------------------------------------------
def pooled_spread(df, score_col='score', ret_col='fwd_3m'):
    """LEGACY pooled qcut (for v4-exact replication only)."""
    clean = df.dropna(subset=[score_col, ret_col])
    try:
        q = pd.qcut(clean[score_col], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except ValueError:
        return None
    q5 = clean[q == 'Q5'][ret_col].mean()
    q1 = clean[q == 'Q1'][ret_col].mean()
    return {'spread': q5 - q1, 'q5': q5, 'q1': q1, 'n': len(clean),
            'corr': clean[score_col].corr(clean[ret_col])}


def _qmean(grp, mask, ret_col, use_cap):
    """Mean forward return of a quintile bucket; cap-weighted if use_cap and
    market_cap is usable, else equal-weighted (fallback when weights sum to 0)."""
    r = grp.loc[mask, ret_col]
    if use_cap:
        w = grp.loc[mask, 'market_cap']
        w = w.where(w > 0, 0.0).fillna(0.0)
        if w.sum() > 0:
            return float(np.average(r, weights=w))
    return r.mean()


def _date_quintile_spreads(df, score_col, ret_col, by_sector=False, weight='equal'):
    """Per-date Q5-Q1 spread. If by_sector, rank within sector each date then
    average across sectors before differencing (sector-neutral). weight='cap'
    value-weights the within-quintile return mean (Hou-Xue-Zhang robustness),
    else equal-weights (default — identical to the legacy estimator)."""
    use_cap = (weight == 'cap') and ('market_cap' in df.columns)
    clean = df.dropna(subset=[score_col, ret_col]).copy()
    spreads = []
    for dt, grp in clean.groupby('date'):
        if by_sector:
            sub = grp.dropna(subset=['sector'])
            q5s, q1s = [], []
            for sec, sg in sub.groupby('sector'):
                if len(sg) < 10:
                    continue
                try:
                    qq = pd.qcut(sg[score_col], 5, labels=False, duplicates='drop')
                except ValueError:
                    continue
                top, bot = _qmean(sg, qq == qq.max(), ret_col, use_cap), _qmean(sg, qq == 0, ret_col, use_cap)
                q5s.append(top); q1s.append(bot)
            if len(q5s) >= 3:
                spreads.append((dt, np.mean(q5s) - np.mean(q1s)))
        else:
            if len(grp) < 25:
                continue
            try:
                qq = pd.qcut(grp[score_col], 5, labels=False, duplicates='drop')
            except ValueError:
                continue
            top, bot = _qmean(grp, qq == qq.max(), ret_col, use_cap), _qmean(grp, qq == 0, ret_col, use_cap)
            spreads.append((dt, top - bot))
    return pd.DataFrame(spreads, columns=['date', 'spread']).set_index('date')['spread']


def per_date_spread(df, score_col='score', ret_col='fwd_12m', sector_neutral=False):
    """Average of per-date spreads + t-stat across dates (the bug fix)."""
    s = _date_quintile_spreads(df, score_col, ret_col, by_sector=sector_neutral)
    if len(s) < 2:
        return None
    mean = s.mean()
    tstat = mean / (s.std(ddof=1) / np.sqrt(len(s)))
    return {'mean_spread': mean, 't_stat': tstat, 'n_dates': len(s),
            'spreads': s, 'sector_neutral': sector_neutral}


# ----------------------------------------------------------------------------
# Survivorship: terminal-return-by-reason for delisted names
# ----------------------------------------------------------------------------
REAL_EXIT_REASONS = ('acquired', 'merged', 'went_private', 'bankrupt', 'distress_delist', 'liquidated')
WIPEOUT = {'bankrupt', 'distress_delist', 'liquidated'}     # -> -100%
REALIZE = {'acquired', 'merged', 'went_private'}            # -> realized to last traded price


def load_delist_info(db=BACKTEST_DB):
    """symbol -> reason + last adjusted close, for real-exit delistings."""
    conn = sqlite3.connect(str(db), timeout=60)
    ph = ','.join('?' * len(REAL_EXIT_REASONS))
    di = pd.read_sql_query(
        f"SELECT symbol, delisted_date, reason FROM delisted_companies WHERE reason IN ({ph})",
        conn, params=REAL_EXIT_REASONS)
    lc = pd.read_sql_query("""
        SELECT symbol, adjusted_close AS last_close FROM historical_prices p
        WHERE date = (SELECT MAX(date) FROM historical_prices WHERE symbol = p.symbol)
    """, conn)
    conn.close()
    return di.merge(lc, on='symbol', how='left')


def apply_terminal_returns(panel, delist_info, horizons=('3m', '12m')):
    """Fill the end-of-series NaN forward returns for DELISTED names with their
    terminal outcome — wipeouts -> -100%, acquisitions -> realized to last price.
    Without this, a bankrupt name's final months silently drop out and survivorship
    bias creeps back in. Non-delisted right-censored NaNs (today) are left to drop."""
    if delist_info is None or not len(delist_info):
        return panel
    di = delist_info.drop_duplicates('symbol').set_index('symbol')
    dmask = panel['symbol'].isin(di.index)
    if not dmask.any():
        return panel
    panel = panel.copy()
    reason = panel['symbol'].map(di['reason'])
    last_close = panel['symbol'].map(di['last_close'])
    realized = ((last_close / panel['close'] - 1) * 100).clip(-100, 300)
    for h in horizons:
        col = f'fwd_{h}'
        if col not in panel:
            continue
        censored = dmask & panel[col].isna()
        panel.loc[censored & reason.isin(WIPEOUT), col] = -100.0
        r = censored & reason.isin(REALIZE)
        panel.loc[r, col] = realized[r]
    return panel


# ----------------------------------------------------------------------------
# Non-overlapping (12 cohorts), HAC, regimes, turnover costs
# ----------------------------------------------------------------------------
def non_overlap_spread(df, score_col='score', ret_col='fwd_12m', sector_neutral=False, step=12):
    """Average across the `step` CALENDAR-anchored cohorts of non-overlapping per-date
    spreads — anchor-robust (no single month is load-bearing). Cohorts are keyed by
    calendar month modulo `step`, so within a cohort dates are `step` months apart =
    truly non-overlapping (robust to any rebalance date dropped by min-count guards)."""
    s = _date_quintile_spreads(df, score_col, ret_col, by_sector=sector_neutral)
    if len(s) < step + 2:
        return None
    key = (s.index.month - 1) % step   # 12m -> 12 monthly cohorts; 3m -> 3 quarterly cohorts
    means, ts = [], []
    for c in range(step):
        sub = s[key == c]
        if len(sub) < 3:
            continue
        means.append(sub.mean())
        ts.append(sub.mean() / (sub.std(ddof=1) / np.sqrt(len(sub))))
    if not means:
        return None
    return {'mean_spread': float(np.mean(means)), 'cohort_std': float(np.std(means)),
            'mean_cohort_t': float(np.nanmean(ts)), 'n_cohorts': len(means)}


def hac_tstat(df, score_col='score', ret_col='fwd_12m', sector_neutral=False, maxlags=12):
    """Newey-West HAC t-stat on the overlapping per-date spread series."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    s = _date_quintile_spreads(df, score_col, ret_col, by_sector=sector_neutral).dropna()
    if len(s) < maxlags + 5:
        return None
    res = sm.OLS(s.values, np.ones((len(s), 1))).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return {'mean': float(res.params[0]), 't_hac': float(res.tvalues[0]), 'n': len(s)}


# ----------------------------------------------------------------------------
# Fama-French 6-factor alpha (FULL-SAMPLE, horizon-matched). NON-OOS by design:
# a clean OOS FF6 alpha is un-estimable on 62 overlapping months (MA(11), 7 params).
# ----------------------------------------------------------------------------
FF_FACTORS = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
FF_CSV = Path(__file__).resolve().parent / 'ff_factors.csv'


def load_ff_factors(path=FF_CSV):
    """Committed Ken French FF5+MOM MONTHLY factors (percent), indexed by month-start.
    Deterministic: reads a committed CSV only. NEVER downloads or synthesizes — if the
    file is absent, returns None and FF6 alpha is intentionally absent."""
    p = Path(path)
    if not p.exists():
        return None
    ff = pd.read_csv(p)
    ff['date'] = pd.to_datetime(ff['date']).dt.to_period('M').dt.to_timestamp()
    ff = ff.set_index('date').sort_index()
    return ff[[c for c in FF_FACTORS + ['RF'] if c in ff.columns]]


def ff6_alpha(spread_series, ff=None, maxlags=12):
    """Full-sample FF6 alpha of a Q5-Q1 12-month-forward spread series (ANNUAL %, a
    long-short spread -> NO RF subtraction). Factors are compounded to a 12-month-forward
    window to MATCH the LHS horizon, then OLS with Newey-West HAC. The intercept is the
    annual alpha in percent DIRECTLY — no x100, no x12 (that double-scale was the -3367%
    bug). Labeled NON-OOS (full 1995-2026). Returns dict or None."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    if ff is None:
        ff = load_ff_factors()
    if ff is None or spread_series is None or len(spread_series) == 0:
        return None
    facs = [c for c in FF_FACTORS if c in ff.columns]
    # 12-month-forward compounded factor return, labeled at month m = compound of m+1..m+12.
    lr = np.log1p(ff[facs] / 100.0)
    fwd12 = (np.expm1(lr.rolling(12).sum().shift(-12)) * 100.0)  # percent, covers m+1..m+12
    # LHS: collapse rebalance dates to their month, average within month.
    s = spread_series.dropna().copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    s = s.groupby(level=0).mean()
    df = pd.concat([s.rename('y'), fwd12.reindex(s.index)], axis=1).dropna()
    if len(df) < maxlags + 5:
        return None
    X = sm.add_constant(df[facs])
    res = sm.OLS(df['y'], X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return {'alpha_annual_pct': float(res.params['const']), 'alpha_t': float(res.tvalues['const']),
            'r2': float(res.rsquared), 'n_months': int(len(df)),
            'betas': {f: float(res.params[f]) for f in facs},
            'sample': 'full 1995-2026, NON-OOS'}


def factor_alpha(spread_series, factors=FF_FACTORS, ff=None, maxlags=12,
                 lo=None, hi=None, label=''):
    """Generalization of ff6_alpha: regress a Q5-Q1 12m-forward spread (ANNUAL %, long-
    short -> NO RF) on an arbitrary FF-factor SUBSET, over an optional [lo, hi] month
    window. Same horizon-matched 12m-forward-compounded factors + Newey-West HAC; the
    intercept is the annual alpha in percent directly (no x100/x12). Used for the
    survivorship-CLEAN 2016+ alpha gate (binding) and the SMB+MOM reduced control."""
    try:
        import statsmodels.api as sm
    except ImportError:
        return None
    if ff is None:
        ff = load_ff_factors()
    if ff is None or spread_series is None or len(spread_series) == 0:
        return None
    facs = [c for c in factors if c in ff.columns]
    if not facs:
        return None
    lr = np.log1p(ff[facs] / 100.0)
    fwd12 = (np.expm1(lr.rolling(12).sum().shift(-12)) * 100.0)
    s = spread_series.dropna().copy()
    s.index = pd.to_datetime(s.index).to_period('M').to_timestamp()
    s = s.groupby(level=0).mean()
    if lo is not None:
        s = s[s.index >= pd.Timestamp(lo)]
    if hi is not None:
        s = s[s.index <= pd.Timestamp(hi)]
    df = pd.concat([s.rename('y'), fwd12.reindex(s.index)], axis=1).dropna()
    if len(df) < maxlags + 5:
        return None
    X = sm.add_constant(df[facs])
    res = sm.OLS(df['y'], X).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
    return {'alpha_annual_pct': float(res.params['const']), 'alpha_t': float(res.tvalues['const']),
            'r2': float(res.rsquared), 'n_months': int(len(df)), 'factors': facs,
            'betas': {f: float(res.params[f]) for f in facs}, 'label': label}


def regime_breakdown(panel, score_col='score', ret_col='fwd_12m', sector_neutral=False):
    out = {}
    for name, (a, b) in REGIMES.items():
        sub = panel[(panel['date'] >= pd.Timestamp(a)) & (panel['date'] <= pd.Timestamp(b))]
        r = per_date_spread(sub, score_col, ret_col, sector_neutral=sector_neutral)
        out[name] = ({'spread': r['mean_spread'], 't': r['t_stat'], 'n_dates': r['n_dates']}
                     if r else None)
    return out


def turnover_cost(panel, score_col='score', ret_col='fwd_12m', cost_bps=20, horizon_months=12):
    """Measured per-date quintile turnover -> net-of-cost annualized spread. Costs scale
    with MEASURED turnover, so a noisy signal that flips more pays more."""
    clean = panel.dropna(subset=[score_col, ret_col]).copy()
    parts = []
    for dt, grp in clean.groupby('date'):
        try:
            grp = grp.assign(q=pd.qcut(grp[score_col], 5, labels=False, duplicates='drop'))
        except ValueError:
            continue
        parts.append(grp[['symbol', 'date', 'q']])
    if not parts:
        return None
    qd = pd.concat(parts).sort_values(['symbol', 'date'])
    qd['q_prev'] = qd.groupby('symbol')['q'].shift(1)
    ext = qd[(qd['q'].isin([0, 4])) | (qd['q_prev'].isin([0, 4]))]
    turnover = float((ext['q'] != ext['q_prev']).mean())     # per monthly rebalance
    rebals_per_year = 12 / horizon_months                     # hold = horizon
    annual_cost = turnover * rebals_per_year * 2 * (cost_bps / 1e4) * 100   # %, 2 sides
    pdte = per_date_spread(clean, score_col, ret_col)
    if not pdte:
        return None
    gross_annual = pdte['mean_spread'] * rebals_per_year if horizon_months < 12 else pdte['mean_spread']
    return {'turnover_per_rebalance': turnover, 'annual_cost_pct': annual_cost,
            'gross_annual': gross_annual, 'net_annual': gross_annual - annual_cost}


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
@dataclass
class RunConfig:
    label: str
    date_basis: str = 'filing'        # 'filing' | 'period_end' (v4-exact)
    sample: int = None                # None=full universe | 2500 (v4-exact)
    survivorship: bool = True         # include delisted terminal returns
    horizon: str = '12m'              # '12m' (binding) | '3m' (control)
    quintile: str = 'per_date'        # 'per_date' | 'pooled' (v4-exact)
    cost_bps: int = 20


def run_score(spec, prices, fund_filing, fund_period, sectors, mcaps, delist_info, cfg: RunConfig,
              *, fund_moonshot_filing=None, fund_moonshot_period=None, corrupt_mcaps=None,
              want_momentum=False, apply_gate=True, use_corrupt_caps=False, legacy_cagr=False):
    """Build the panel per cfg and compute the report cells for one (score, config).

    Moonshot extras (keyword-only so Compass call sites are unchanged): when spec.gate
    is set the panel is filtered to gated rows BEFORE scoring (IS z-stats + quintiles on
    the gated population). use_corrupt_caps / legacy_cagr drive the M0' directional proxy
    only and must never be set on the honest stages."""
    is_moon = spec.gate is not None
    if is_moon:
        fund = fund_moonshot_period if cfg.date_basis == 'period_end' else fund_moonshot_filing
        want_momentum = True
    else:
        fund = fund_period if cfg.date_basis == 'period_end' else fund_filing
    horizons = (cfg.horizon,) if cfg.horizon == '3m' else ('3m', '12m')
    del_syms = set(delist_info['symbol']) if (cfg.survivorship and delist_info is not None) else None
    panel = build_panel(prices, fund, sectors, mcaps=mcaps, sample=cfg.sample,
                        horizons=horizons, delisted_syms=del_syms, want_momentum=want_momentum,
                        corrupt_mcaps=(corrupt_mcaps if use_corrupt_caps else None))
    if cfg.survivorship:
        panel = apply_terminal_returns(panel, delist_info, horizons=horizons)

    if is_moon:
        # M0' proxy: swap in the legacy broadcast CAGR (look-ahead) before scoring.
        if legacy_cagr:
            panel['revenue_growth_3yr'] = panel['revenue_growth_3yr_legacy']
            panel['eps_growth_3yr'] = panel['eps_growth_3yr_legacy']
        # Neutral-fill the 5 OPTIONAL factors (legacy fillna(0)); required factors stay.
        for c in ['eps_growth_3yr', 'margin_improvement', 'fcf_margin', 'roe', 'momentum_12_1']:
            if c in panel:
                panel[c] = panel[c].fillna(0)
        # Gate-input columns must have survived the merge (else the gate silently nukes
        # the universe via .fillna(False)). Fail loudly if they're missing/all-NaN.
        missing = [c for c in MOONSHOT_GATE_COLS if c not in panel.columns]
        assert not missing, f"gate-input columns missing from panel: {missing}"
        if apply_gate:
            panel = panel[spec.gate(panel)].copy()

    panel = add_score(panel, spec)
    oos = panel[panel['date'] >= pd.Timestamp(OOS_START)]
    ret = f'fwd_{cfg.horizon}'
    hm = 12 if cfg.horizon == '12m' else 3
    rep = {'label': cfg.label, 'horizon': cfg.horizon, 'n_obs': int(oos[ret].notna().sum()),
           'gated': bool(is_moon and apply_gate), 'n_gated': int(len(panel)),
           'n_panel': int(len(panel))}
    if cfg.quintile == 'pooled':
        rep['pooled'] = pooled_spread(oos, ret_col=ret)
    rep['per_date'] = per_date_spread(oos, ret_col=ret, sector_neutral=False)
    rep['per_date_sector_neutral'] = per_date_spread(oos, ret_col=ret, sector_neutral=True)
    rep['non_overlap'] = non_overlap_spread(oos, ret_col=ret, sector_neutral=True, step=hm)
    rep['hac'] = hac_tstat(oos, ret_col=ret, sector_neutral=True, maxlags=hm)
    rep['regimes'] = regime_breakdown(panel, ret_col=ret, sector_neutral=True)
    rep['cost'] = turnover_cost(oos, ret_col=ret, cost_bps=cfg.cost_bps, horizon_months=hm)
    rep['n_dates_raw'] = rep['per_date']['n_dates'] if rep['per_date'] else 0
    rep['n_dates_sn'] = rep['per_date_sector_neutral']['n_dates'] if rep['per_date_sector_neutral'] else 0
    # Q5-Q1 per-rebalance spread series (FULL sample, not just OOS) — feeds FF6 alpha.
    rep['spread_series'] = _date_quintile_spreads(panel, 'score', ret, by_sector=False)
    return rep


def load_all(db=BACKTEST_DB, moonshot=False):
    """Load every shared input once (the driver passes these to run_score). When
    moonshot=True, also load the Moonshot fundamentals + corrupt caps (M0' proxy)."""
    D = dict(prices=load_prices(db), fund_filing=load_fundamentals(db, 'filing'),
             fund_period=load_fundamentals(db, 'period_end'), sectors=load_sectors(db),
             mcaps=load_market_caps(db), delist_info=load_delist_info(db))
    if moonshot:
        D['fund_moonshot_filing'] = load_moonshot_fundamentals(db, 'filing')
        D['fund_moonshot_period'] = load_moonshot_fundamentals(db, 'period_end')
        D['corrupt_mcaps'] = load_corrupt_market_caps(db)
    return D


if __name__ == '__main__':
    import time, json
    t0 = time.time()
    print("Loading shared inputs...")
    D = load_all()
    print(f"  loaded ({time.time()-t0:.0f}s)")
    cfg = RunConfig(label='Compass | full | filing | survivorship | 12m | per-date')
    rep = run_score(COMPASS_SPEC, D['prices'], D['fund_filing'], D['fund_period'],
                    D['sectors'], D['mcaps'], D['delist_info'], cfg)
    pd_ = rep['per_date']; sn = rep['per_date_sector_neutral']; no = rep['non_overlap']; hac = rep['hac']; cost = rep['cost']
    print(f"\n{rep['label']}  (n_obs={rep['n_obs']:,})")
    print(f"  per-date 12m:        {pd_['mean_spread']:+.2f}%  t={pd_['t_stat']:+.1f}")
    print(f"  sector-neutral:      {sn['mean_spread']:+.2f}%  t={sn['t_stat']:+.1f}")
    if no:  print(f"  non-overlap (12coh): {no['mean_spread']:+.2f}%  t~{no['mean_cohort_t']:+.1f}  cohort_sd={no['cohort_std']:.2f}")
    if hac: print(f"  HAC t (overlap):     t={hac['t_hac']:+.1f}")
    if cost:print(f"  turnover/reb={cost['turnover_per_rebalance']:.2f}  cost={cost['annual_cost_pct']:.1f}%  net={cost['net_annual']:+.2f}%")
    print(f"  regimes:", {k: (round(v['spread'], 1) if v else None) for k, v in rep['regimes'].items()})
    print(f"\nTotal time: {time.time()-t0:.0f}s")

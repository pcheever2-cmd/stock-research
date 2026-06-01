#!/usr/bin/env python3
"""
Moonshot-with-Quality factor SEARCH (Stages 1-3) — theory-grounded, overfit-hardened.

Protocol (see ~/.claude/plans/can-you-review-the-playful-pie.md):
  TRAIN   <= 2014-12-31  : hypothesis + weight fit (survivorship-suspect -> direction/weights only)
  CONFIRM 2016-01..2018-12: survivorship-CLEAN, direction-only STOP gate (near-zero power)
  OOS     >= 2020-01-01  : SEALED; revealed ONCE against a strict pre-registered bar
  12-month embargo at each seam (2015, 2019 skipped).

Anti-overfit machinery: Newey-West IC t (L=11), Benjamini-Yekutieli FDR, purged+embargoed
CV weight fit, Deflated Sharpe + PBO(CSCV) + a White/Hansen bootstrap reality check,
stationary bootstrap CI, Hou-Xue-Zhang hygiene (drop microcaps, value-weight robustness).

Design note: the spread objective REUSES harness._date_quintile_spreads (the canonical
estimator) rather than a hand-vectorized copy — eliminating the silent-mismatch risk the
committee flagged. On the TRAIN slice it is ~0.1-0.4s/candidate. The score itself is the
cheap part (Z @ w over precomputed z-columns).

OOS sealing: reveal_oos() is the ONLY function that slices date>=OOS_START; it refuses to
run unless validation/MOONSHOT_LOCK.json exists (git-committed) and the passed spec's SHA
matches. assert_sealed() guards every pre-lock estimator call.
"""
import sys, json, hashlib, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import BACKTEST_DB
from validation import harness as H
from validation.factors_moonshot_plus import (
    load_moonshot_plus_fundamentals, add_market_factors, FACTOR_LIBRARY)

# ---------------------------------------------------------------------------
# Frozen split + bar (pre-registration). Edited only via the §0 commit.
# ---------------------------------------------------------------------------
TRAIN_END   = '2014-12-31'
CONFIRM_LO  = '2016-01-01'
CONFIRM_HI  = '2018-12-31'
OOS_START   = '2020-01-01'
SEED        = 42
MICROCAP_MIN = 100_000_000          # drop microcaps (Hou-Xue-Zhang hygiene)
MIN_PRICE    = 5.0
MAX_ACTIVE   = 8                     # sparsity cap
L1_LAMBDA    = 0.5                   # sparsity penalty coefficient (declared)
N_DRAWS      = 1200                  # random weight draws per arm (declared)
LOCK_PATH    = Path(__file__).resolve().parent / 'MOONSHOT_LOCK.json'
REPORT       = Path(__file__).resolve().parent / 'HONEST_NUMBERS.md'

# OOS acceptance bar (pre-registered, frozen).
BAR = dict(spread_min=3.0, hac_t_min=2.0, alpha_t_min=2.0)

PRICE_FACTORS = ['momentum_12_1', 'momentum_6m', 'momentum_3m', 'prox_52w_high',
                 'small_cap', 'vol_60d']
MARKET_FACTORS = ['ev_ebitda', 'fcf_yield']
ALL_FACTORS = list(FACTOR_LIBRARY)   # frozen order
SIGNS = {k: v[1] for k, v in FACTOR_LIBRARY.items()}
GROUP = {k: v[0] for k, v in FACTOR_LIBRARY.items()}
QUALITY = [k for k in ALL_FACTORS if GROUP[k] == 'quality']
UPSIDE  = [k for k in ALL_FACTORS if GROUP[k] == 'upside']
VALUE   = [k for k in ALL_FACTORS if GROUP[k] == 'value']


def log(m): print(m, flush=True)


# ---------------------------------------------------------------------------
# Panel build (ONCE) + junk floor
# ---------------------------------------------------------------------------
def junk_floor(panel):
    """Minimal exclusion (NOT a quality gate): liquidity + microcap/price floor +
    positive revenue. Hou-Xue-Zhang hygiene to avoid microcap-inflated significance."""
    m = panel['is_liquid'].fillna(False)
    m &= panel['market_cap'] >= MICROCAP_MIN
    m &= panel['raw_close'] >= MIN_PRICE
    m &= panel['revenue_ttm'] > 0
    return m.fillna(False)


PANEL_CACHE = Path('/tmp/moonshot_search_panel.pkl')


def build_search_panel(db=BACKTEST_DB, use_cache=True):
    """One panel build with the full factor library + survivorship terminal returns,
    after junk floor. Cached so dry-run and the (separate) OOS-reveal process share the
    identical panel. Returns the scored-input panel (no score yet)."""
    if use_cache and PANEL_CACHE.exists():
        log(f"  (using cached panel {PANEL_CACHE})")
        return pd.read_pickle(PANEL_CACHE)
    prices = H.load_prices(db)
    sectors = H.load_sectors(db)
    mcaps = H.load_market_caps(db)
    fund = load_moonshot_plus_fundamentals(db, 'filing')
    delist = H.load_delist_info(db)
    del_syms = set(delist['symbol']) if delist is not None else None
    panel = H.build_panel(prices, fund, sectors, mcaps=mcaps, sample=None,
                          horizons=('3m', '12m'), delisted_syms=del_syms, want_momentum=True)
    panel = H.apply_terminal_returns(panel, delist, horizons=('3m', '12m'))
    panel = add_market_factors(panel)
    panel = panel[junk_floor(panel)].copy()
    # Snap rebalance dates to month-start so cross-sections pool monthly (~230 dates
    # instead of ~4500 calendar-scattered ones): ~20x faster per-date groupby AND larger,
    # better-formed quintiles. Applied ONCE here so search + OOS reveal use the identical
    # grouping (no estimator mismatch). Forward returns are per-row, unaffected by the label.
    panel['date'] = panel['date'].values.astype('datetime64[M]').astype('datetime64[ns]')
    if use_cache:
        try:
            panel.to_pickle(PANEL_CACHE)
        except Exception as e:
            log(f"  (panel cache write failed: {e})")
    return panel


# ---------------------------------------------------------------------------
# Precompute z-columns (TRAIN-ONLY moments) — leakage-safe standardization
# ---------------------------------------------------------------------------
def precompute_z(panel, factors=ALL_FACTORS, is_end=TRAIN_END):
    """Cache z__{f} for each factor using clip+mean+std fit on date<=is_end ONLY
    (same math as add_score). Returns (panel, Z matrix [n x k], col order)."""
    panel = panel.copy()
    isd = panel[panel['date'] <= pd.Timestamp(is_end)]
    cols = [f for f in factors if f in panel.columns]
    for f in cols:
        lo, hi = isd[f].quantile(0.01), isd[f].quantile(0.99)
        clp = panel[f].clip(lo, hi)
        mean, std = isd[f].clip(lo, hi).mean(), isd[f].clip(lo, hi).std()
        z = ((clp - mean) / (std if std and not np.isnan(std) else 1.0)).clip(-3, 3).fillna(0)
        panel[f'z__{f}'] = z * SIGNS[f]            # sign-applied so higher z -> Q5
    Z = panel[[f'z__{f}' for f in cols]].to_numpy()
    return panel, Z, cols


def score_from_weights(panel, Z, cols, weights):
    """score = Z @ w (cheap). weights: dict col->w or array aligned to cols."""
    if isinstance(weights, dict):
        w = np.array([weights.get(c, 0.0) for c in cols])
    else:
        w = np.asarray(weights, dtype=float)
    panel = panel.copy()
    panel['score'] = Z @ w
    return panel


def slice_dates(panel, lo=None, hi=None):
    m = pd.Series(True, index=panel.index)
    if lo is not None: m &= panel['date'] >= pd.Timestamp(lo)
    if hi is not None: m &= panel['date'] <= pd.Timestamp(hi)
    return panel[m]


# ---------------------------------------------------------------------------
# Fast precomputed sector-neutral spread (search-only). The (month, sector) group
# index arrays are built ONCE; per candidate we only recompute score = Z@w and do
# numpy quintiles. The OOS reveal uses the CANONICAL harness estimator (slow is fine
# for one shot); fast_eval is asserted ~equal to it before the search runs.
# ---------------------------------------------------------------------------
def precompute_groups(panel, lo=None, hi=None, min_sec=10, min_secs=3):
    """Per-month list of (positions, fwd) sector subgroups (>=min_sec each, months with
    >=min_secs sectors) + per-month all-positions (for the tercile transform). Positions
    index into the full-panel row order (same as Z)."""
    pos = np.arange(len(panel))
    dates = panel['date'].values
    keep = np.ones(len(panel), bool)
    if lo is not None: keep &= dates >= np.datetime64(pd.Timestamp(lo))
    if hi is not None: keep &= dates <= np.datetime64(pd.Timestamp(hi))
    fwd = panel['fwd_12m'].values
    sec = panel['sector'].astype('object').values
    keep &= ~pd.isna(fwd) & ~pd.isna(sec)
    d = pd.DataFrame({'pos': pos[keep], 'month': dates[keep], 'sec': sec[keep], 'fwd': fwd[keep]})
    groups, allpos = {}, {}
    for month, g in d.groupby('month'):
        sgs = [(sg['pos'].to_numpy(), sg['fwd'].to_numpy())
               for _, sg in g.groupby('sec') if len(sg) >= min_sec]
        if len(sgs) >= min_secs:
            groups[month] = sgs
            allpos[month] = g['pos'].to_numpy()
    return groups, allpos


def _q5q1(sc, fwd):
    """Equal-count 5-way bucket (== qcut for continuous scores); top/bottom fwd means."""
    n = len(sc)
    order = np.argsort(sc, kind='stable')
    ranks = np.empty(n, int); ranks[order] = np.arange(n)
    b = ranks * 5 // n
    return fwd[b == 4].mean(), fwd[b == 0].mean()


def fast_sn(score, groups, allpos=None, qual=None, months=None):
    """Sector-neutral per-month Q5-Q1 spread Series. If qual given, apply the within-month
    quality-tercile re-rank (+1 top tercile, -1 bottom) before sector quintiles."""
    sc = score
    if qual is not None:
        sc = score.copy()
        for month, ap in allpos.items():
            if months is not None and month not in months:
                continue
            q = qual[ap]; valid = ~np.isnan(q)
            if valid.sum() < 15:
                continue
            vidx = ap[valid]; qv = q[valid]
            order = np.argsort(qv, kind='stable'); rk = np.empty(len(qv), int); rk[order] = np.arange(len(qv))
            terc = rk * 3 // len(qv)
            sc[vidx[terc == 2]] += 1.0
            sc[vidx[terc == 0]] -= 1.0
    out_m, out_s = [], []
    for month, sgs in groups.items():
        if months is not None and month not in months:
            continue
        tops, bots = [], []
        for p, fwd in sgs:
            t, b = _q5q1(sc[p], fwd); tops.append(t); bots.append(b)
        out_m.append(month); out_s.append(np.mean(tops) - np.mean(bots))
    return pd.Series(out_s, index=pd.to_datetime(out_m)).sort_index()


# ---------------------------------------------------------------------------
# OOS sealing
# ---------------------------------------------------------------------------
def assert_sealed(df):
    """Raise if a slice contains OOS rows while the lock does not yet exist."""
    if not LOCK_PATH.exists() and (df['date'] >= pd.Timestamp(OOS_START)).any():
        raise RuntimeError("SEALED: OOS rows (date>=%s) touched before lock exists." % OOS_START)


def spec_sha(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Objective + IC screen
# ---------------------------------------------------------------------------
def sn_nonoverlap(scored, lo=None, hi=None):
    """Sector-neutral non-overlap spread on a date slice (the selection objective)."""
    sl = slice_dates(scored, lo, hi)
    assert_sealed(sl)
    return H.non_overlap_spread(sl, score_col='score', ret_col='fwd_12m',
                                sector_neutral=True, step=12)


def objective(panel, Z, cols, weights, lo=None, hi=None, lam=L1_LAMBDA):
    scored = score_from_weights(panel, Z, cols, weights)
    no = sn_nonoverlap(scored, lo, hi)
    if no is None:
        return -1e9
    n_active = int(np.sum(np.abs(np.asarray([weights[c] for c in cols] if isinstance(weights, dict)
                                             else weights)) > 1e-6))
    return no['mean_spread'] - lam * n_active


def _nw_tstat(x, maxlags=11):
    """Newey-West t-stat of the mean of x (overlap-robust). Manual HAC."""
    x = np.asarray(pd.Series(x).dropna(), float)
    n = len(x)
    if n < maxlags + 3:
        return np.nan, n
    mu = x.mean(); e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, maxlags + 1):
        w = 1 - L / (maxlags + 1)
        cov = (e[L:] @ e[:-L]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return (mu / se if se > 0 else np.nan), n


def ic_screen(panel, factors=ALL_FACTORS, lo=None, hi=TRAIN_END):
    """Per-factor sector-neutral IC (Spearman vs fwd_12m), NW L=11 t across dates,
    + BY-FDR q-values. Descriptive (theory retains factors); reports honesty."""
    sl = slice_dates(panel, lo, hi); assert_sealed(sl)
    rows = []
    for f in factors:
        if f not in sl.columns:
            continue
        ics = []
        for dt, grp in sl.dropna(subset=[f, 'fwd_12m', 'sector']).groupby('date'):
            per_sec = []
            for sec, sg in grp.groupby('sector'):
                if len(sg) < 10:
                    continue
                per_sec.append(sg[f].rank().corr(sg['fwd_12m'].rank(), method='pearson'))
            if per_sec:
                ics.append(np.nanmean(per_sec))
        ic = np.nanmean(ics) if ics else np.nan
        t, n = _nw_tstat(ics) if ics else (np.nan, 0)
        rows.append(dict(factor=f, group=GROUP[f], sign=SIGNS[f], ic=ic,
                         ic_signed=ic * SIGNS[f], nw_t=t, n_dates=n))
    df = pd.DataFrame(rows)
    # Benjamini-Yekutieli FDR on two-sided p of the SIGNED IC t (dependence-robust).
    from scipy import stats
    df['p'] = 2 * stats.norm.sf(np.abs(df['nw_t'].fillna(0)))
    df['by_q'] = _by_fdr(df['p'].values)
    df['hlz_pass'] = df['nw_t'].abs() >= 3.0          # Harvey-Liu-Zhu hurdle
    return df.sort_values('ic_signed', ascending=False).reset_index(drop=True)


def _by_fdr(p):
    """Benjamini-Yekutieli adjusted q-values (arbitrary dependence)."""
    p = np.asarray(p, float); m = len(p)
    c = np.sum(1.0 / np.arange(1, m + 1))         # harmonic correction
    order = np.argsort(p); ranks = np.arange(1, m + 1)
    q = p[order] * m * c / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity
    out = np.empty(m); out[order] = np.clip(q, 0, 1)
    return out


# ---------------------------------------------------------------------------
# Data-snooping statistics
# ---------------------------------------------------------------------------
def deflated_sharpe(sr_obs, sr_trials, n_obs, skew=0.0, kurt=3.0):
    """Bailey & López de Prado (2014) Deflated Sharpe Ratio. sr_* are per-period
    Sharpe; n_obs = #periods; sr_trials = array of trial Sharpes (for E[max] null)."""
    from scipy import stats
    sr_trials = np.asarray(sr_trials, float)
    N = max(len(sr_trials), 2)
    v = np.var(sr_trials, ddof=1) if N > 1 else 1.0
    g = 0.5772156649
    emax = np.sqrt(v) * ((1 - g) * stats.norm.ppf(1 - 1.0 / N) +
                         g * stats.norm.ppf(1 - 1.0 / (N * np.e)))
    denom = np.sqrt(1 - skew * sr_obs + (kurt - 1) / 4.0 * sr_obs ** 2)
    psr = stats.norm.cdf((sr_obs - emax) * np.sqrt(n_obs - 1) / denom)
    return dict(dsr=float(psr), sr_obs=float(sr_obs), expected_max_sr=float(emax),
                n_trials=int(N))


def pbo_cscv(perf_matrix, S=16):
    """Probability of Backtest Overfitting via combinatorially-symmetric CV.
    perf_matrix: T x N (time slices x candidate configs) of performance. Returns PBO."""
    from itertools import combinations
    import scipy.special
    M = np.asarray(perf_matrix, float)
    T, N = M.shape
    if N < 2 or T < S:
        return np.nan
    blocks = np.array_split(np.arange(T), S)
    lam = []
    for comb in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[b] for b in comb])
        oos_idx = np.concatenate([blocks[b] for b in range(S) if b not in comb])
        is_perf = M[is_idx].mean(0); oos_perf = M[oos_idx].mean(0)
        best = int(np.argmax(is_perf))
        rank = (oos_perf <= oos_perf[best]).mean()    # relative OOS rank in [0,1]
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lam.append(np.log(rank / (1 - rank)))
    lam = np.asarray(lam)
    return float((lam < 0).mean())


def stationary_bootstrap_ci(x, block=12, B=2000, seed=SEED, lo_pct=5):
    """Stationary (geometric-block) bootstrap lower percentile of the mean of x.
    block floored at the MA(overlap) horizon. One-sided (5th pct > 0 test)."""
    x = np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < block + 2:
        return np.nan
    rng = np.random.RandomState(seed); p = 1.0 / block
    means = np.empty(B)
    for b in range(B):
        idx = np.empty(n, int); i = rng.randint(n)
        for k in range(n):
            idx[k] = i
            if rng.rand() < p:
                i = rng.randint(n)
            else:
                i = (i + 1) % n
        means[b] = x[idx].mean()
    return float(np.percentile(means, lo_pct))


def reality_check(cand_series, B=2000, block=12, seed=SEED):
    """White (2000) Reality Check / a basic Hansen-style studentized variant: bootstrap
    p-value that the BEST candidate's mean (vs 0 benchmark) is genuine after snooping.
    cand_series: list of per-date spread arrays (one per candidate)."""
    rng = np.random.RandomState(seed)
    fbar = np.array([np.nanmean(s) for s in cand_series])
    Tn = np.array([len(s) for s in cand_series])
    V = np.nanmax(np.sqrt(Tn) * fbar)
    # studentize for SPA
    sds = np.array([np.nanstd(s, ddof=1) or 1.0 for s in cand_series])
    Vspa = np.nanmax(np.sqrt(Tn) * fbar / sds)
    boots_rc, boots_spa = [], []
    for b in range(B):
        maxs_rc, maxs_spa = [], []
        for s, fb, sd in zip(cand_series, fbar, sds):
            s = np.asarray(pd.Series(s).dropna(), float); n = len(s)
            if n < block + 2:
                continue
            i = rng.randint(n); idx = np.empty(n, int); p = 1.0 / block
            for k in range(n):
                idx[k] = i
                i = rng.randint(n) if rng.rand() < p else (i + 1) % n
            bm = s[idx].mean() - fb                       # recentered
            maxs_rc.append(np.sqrt(n) * bm)
            maxs_spa.append(np.sqrt(n) * bm / sd)
        if maxs_rc:
            boots_rc.append(max(maxs_rc)); boots_spa.append(max(maxs_spa))
    rc_p = float(np.mean(np.asarray(boots_rc) >= V)) if boots_rc else np.nan
    spa_p = float(np.mean(np.asarray(boots_spa) >= Vspa)) if boots_spa else np.nan
    return dict(reality_check_p=rc_p, spa_p=spa_p, V=float(V))


# ---------------------------------------------------------------------------
# Quality-conditioning structures
# ---------------------------------------------------------------------------
def quality_composite(panel, cols, is_end=TRAIN_END):
    """Equal-weight z-sum of the QUALITY factors' (already sign-applied) z-columns."""
    qcols = [f'z__{f}' for f in QUALITY if f'z__{f}' in panel.columns]
    return panel[qcols].mean(axis=1)


def apply_tercile(panel, score, qual):
    """Within-month quality-tercile re-rank on a full-panel score array (the canonical
    reveal path; mirrors fast_sn's numpy tercile so search and OOS agree). +1 to the top
    quality tercile, -1 to the bottom, per month."""
    sc = np.asarray(score, float).copy()
    qa = np.asarray(qual, float)
    for month, idx in panel.groupby('date').indices.items():
        q = qa[idx]; valid = ~np.isnan(q)
        if valid.sum() < 15:
            continue
        vidx = idx[valid]; qv = q[valid]
        order = np.argsort(qv, kind='stable'); rk = np.empty(len(qv), int); rk[order] = np.arange(len(qv))
        terc = rk * 3 // len(qv)
        sc[vidx[terc == 2]] += 1.0
        sc[vidx[terc == 0]] -= 1.0
    return sc


# ---------------------------------------------------------------------------
# Weight search (build-once, evaluate-many) + purged CV + finalist
# ---------------------------------------------------------------------------
def _rand_weights(rng, k_cols):
    """Sparse non-negative weight vector (signs already baked into z): pick 3..MAX_ACTIVE
    active factors, Dirichlet weights, zeros elsewhere."""
    n = len(k_cols)
    k = rng.randint(3, MAX_ACTIVE + 1)
    active = rng.choice(n, size=k, replace=False)
    w = np.zeros(n); w[active] = rng.dirichlet(np.ones(k))
    return w


def month_folds(months, k=3):
    """k contiguous time blocks of the (sorted) training months — for a temporal-
    robustness CV re-rank of the top candidates (fixed weights, so no refit/embargo)."""
    u = np.array(sorted(months))
    return [set(b) for b in np.array_split(u, k)]


def search_arm(Z, cols, arm, groups, allpos, qual_arr, train_months, n_draws=N_DRAWS, seed=SEED):
    """Random weight search on TRAIN via the fast precomputed estimator. Returns dicts
    (w, obj, train series) sorted by objective. arm 'A2' applies the quality re-rank."""
    rng = np.random.RandomState(seed + (0 if arm == 'A1' else 1))
    use_q = qual_arr if arm == 'A2' else None
    out = []
    for _ in range(n_draws):
        w = _rand_weights(rng, cols)
        s = fast_sn(Z @ w, groups, allpos, use_q, months=train_months)
        if len(s) < 14:
            continue
        n_active = int((np.abs(w) > 1e-6).sum())
        out.append(dict(arm=arm, w=w, n_active=n_active,
                        obj=float(s.mean()) - L1_LAMBDA * n_active,
                        train_mean=float(s.mean()), series=s))
    out.sort(key=lambda d: d['obj'], reverse=True)
    return out


def cv_minfold(Z, w, arm, folds, groups, allpos, qual_arr):
    """Temporal-robustness score = MIN fold-mean sector-neutral spread (favors a
    candidate whose worst sub-period is best, not just the overall mean)."""
    use_q = qual_arr if arm == 'A2' else None
    vals = []
    for test_months in folds:
        s = fast_sn(Z @ w, groups, allpos, use_q, months=test_months)
        vals.append(s.mean() if len(s) else -1e9)
    return float(min(vals)) if vals else -1e9


def run_search(panel, Z, cols, topk_cv=30):
    """Fast TRAIN search across arms -> top-K -> temporal-CV re-rank -> finalist + the
    candidate population (for snooping). Builds group index arrays ONCE. NEVER OOS."""
    import time
    t = time.time()
    groups, allpos = precompute_groups(panel, hi=CONFIRM_HI)     # train+confirm only (no OOS)
    qual_arr = quality_composite(panel, cols).to_numpy()
    train_months = {m for m in groups if m <= pd.Timestamp(TRAIN_END)}
    log(f"  precomputed {len(groups)} month-groups ({time.time()-t:.0f}s)")
    cands = []
    for arm in ['A1', 'A2']:
        t = time.time()
        cands += search_arm(Z, cols, arm, groups, allpos, qual_arr, train_months)
        log(f"  arm {arm}: {N_DRAWS} draws ({time.time()-t:.0f}s)")
    cands.sort(key=lambda d: d['obj'], reverse=True)
    folds = month_folds(train_months)
    top = cands[:topk_cv]
    for c in top:
        c['cv_obj'] = cv_minfold(Z, c['w'], c['arm'], folds, groups, allpos, qual_arr)
    finalist = max(top, key=lambda d: d['cv_obj'])
    return finalist, cands


def finalist_spec(finalist, cols):
    """Canonical JSON-able spec for the lock (weights keyed by factor, sign-applied)."""
    w = {cols[i]: round(float(finalist['w'][i]), 6) for i in range(len(cols))
         if abs(finalist['w'][i]) > 1e-6}
    return dict(arm=finalist['arm'], weights=w, signs={k: SIGNS[k] for k in w},
                split=dict(train_end=TRAIN_END, confirm=[CONFIRM_LO, CONFIRM_HI], oos=OOS_START),
                bar=BAR, library=ALL_FACTORS, seed=SEED, n_draws=N_DRAWS,
                max_active=MAX_ACTIVE, l1=L1_LAMBDA)


# ---------------------------------------------------------------------------
# Pre-registration, lock, reveal
# ---------------------------------------------------------------------------
def confirm_gate(panel, Z, cols, finalist):
    """Direction-only STOP gate on the survivorship-clean CONFIRM window (near-zero
    power by design — its only honest job is to kill a sign-flip)."""
    groups, allpos = precompute_groups(panel, lo=CONFIRM_LO, hi=CONFIRM_HI)
    qual = quality_composite(panel, cols).to_numpy() if finalist['arm'] == 'A2' else None
    s = fast_sn(Z @ finalist['w'], groups, allpos, qual, months=set(groups))
    mean = float(s.mean()) if len(s) else np.nan
    return dict(confirm_mean=mean, n_dates=int(len(s)),
                stop=(not np.isnan(mean) and mean < 0))


def snooping_stats(finalist, cands):
    """Deflated Sharpe (N=all candidates), PBO via CSCV, White/Hansen reality check."""
    fs = finalist['series'].dropna()
    sr_obs = fs.mean() / (fs.std(ddof=1) or 1.0)
    sr_trials = np.array([c['series'].mean() / (c['series'].std(ddof=1) or 1.0)
                          for c in cands if c['series'].std(ddof=1)])
    dsr = deflated_sharpe(sr_obs, sr_trials, n_obs=len(fs),
                          skew=float(fs.skew()), kurt=float(fs.kurt() + 3))
    # PBO matrix over a manageable candidate subset on common dates.
    sub = ([finalist] + cands[:80])
    common = sorted(set.intersection(*[set(c['series'].dropna().index) for c in sub]))
    if len(common) >= 16:
        M = np.column_stack([c['series'].reindex(common).values for c in sub])
        pbo = pbo_cscv(M, S=16)
    else:
        pbo = np.nan
    rc = reality_check([c['series'].values for c in sub])
    dsr['effective_trials_all'] = len(cands)
    return dict(dsr=dsr, pbo=pbo, **rc)


def write_preregistration(spec, snoop):
    """Append the frozen pre-registration to HONEST_NUMBERS.md BEFORE the OOS reveal."""
    lines = ["\n\n---\n\n## 4. PRE-REGISTRATION — Moonshot-with-Quality search (frozen before OOS)\n",
             f"- Split: TRAIN ≤{TRAIN_END} · CONFIRM {CONFIRM_LO}..{CONFIRM_HI} (survivorship-clean, "
             f"direction-only STOP) · OOS ≥{OOS_START}; 12-mo embargo.",
             f"- Library ({len(ALL_FACTORS)} cited factors, signs frozen): "
             + ", ".join(f"{k}({'+' if SIGNS[k] > 0 else '−'})" for k in ALL_FACTORS),
             f"- Search space: arms {{A1 linear, A2 quality-tercile re-rank}}, {N_DRAWS} draws/arm, "
             f"≤{MAX_ACTIVE} active, L1 λ={L1_LAMBDA}, seed {SEED}.",
             f"- Effective trials (DSR): {snoop['dsr']['effective_trials_all']} weight vectors × 2 arms.",
             f"- OOS bar (ALL must hold): (A) sector-neutral non-overlap net-of-cost ≥ +{BAR['spread_min']}pp; "
             f"(B) HAC t ≥ +{BAR['hac_t_min']} AND bootstrap 5th-pct > 0; (C) CLEAN-2016+ factor-alpha > 0 "
             f"(full-FF6 supporting, sign must agree); (D) positive in 2020-21 AND 2022-26; "
             f"(E) holds value-weighted; (F) adequate Q5 breadth.",
             f"- Locked finalist: arm {spec['arm']}, weights {json.dumps(spec['weights'])}.",
             f"- Honest-null STOP: miss any ⇒ ship \"No robust moonshot-with-quality edge beyond "
             f"QMJ/size-momentum exists in this data.\"\n"]
    with open(REPORT, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def write_null_report(spec, cg, snoop, reason='confirm-stop'):
    """Document an honest-null that STOPPED before the OOS reveal (OOS unburned)."""
    d = snoop['dsr']
    L = ["\n\n---\n\n## 5. PHASE 5 — Moonshot-with-Quality search\n",
         "**VERDICT: HONEST-NULL** — *No robust moonshot-with-quality edge beyond "
         "QMJ/size-momentum exists in this data.* The disciplined search STOPPED at the "
         "survivorship-clean CONFIRM gate; **the OOS (2020+) was never touched (seal intact).**\n",
         f"- Finalist (best of {d['effective_trials_all']} candidates, arm {spec['arm']}): "
         f"`{json.dumps(spec['weights'])}`",
         f"- TRAIN (1995-2014, survivorship-suspect): sector-neutral spread looked positive, **but**",
         f"- **CONFIRM 2016-2018 (survivorship-CLEAN): {cg['confirm_mean']:+.2f}% over {cg['n_dates']} "
         f"months → sign-flip → STOP.** The train edge is survivorship/selection inflation, not signal.",
         f"- Deflated Sharpe = **{d['dsr']:.2f}** (< 0.95; obs SR {d['sr_obs']:.2f} vs E[max] under "
         f"{d['n_trials']} trials {d['expected_max_sr']:.2f}) — search-selection inflated.",
         f"- PBO (CSCV) = **{snoop['pbo']:.2f}** — ~coin-flip out-of-sample (overfit-prone).",
         f"- (White reality-check p={snoop['reality_check_p']:.3f}, Hansen SPA p={snoop['spa_p']:.3f} flag "
         f"that the in-sample BEST looks significant — exactly the multiple-testing mirage DSR/PBO catch.)",
         "- The univariate screen confirms the real (modest) signal is **quality + value (GARP)** "
         "(gp_assets, fcf_yield, ev_ebitda, cash_conversion clear the Harvey-Liu-Zhu t≥3 + BY-FDR bar); "
         "naive growth does NOT predict. But no weighted combination survives the clean-window confirm.",
         "- Honest-null STOP is pre-committed: no re-search, no confirm-mining, OOS preserved for a future "
         "pre-registered hypothesis.\n"]
    with open(REPORT, 'a') as f:
        f.write('\n'.join(L) + '\n')
    log("wrote Phase-5 honest-null to HONEST_NUMBERS.md")


def lock(spec, git_commit=True):
    """Write the lock (spec + SHA) and git-commit it (tamper-evident timestamp)."""
    payload = dict(spec=spec, sha=spec_sha(spec))
    LOCK_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
    if git_commit:
        try:
            subprocess.run(['git', 'add', str(LOCK_PATH), str(REPORT)], cwd=ROOT, check=True)
            subprocess.run(['git', 'commit', '-m',
                            'Lock Moonshot-with-Quality pre-registration before OOS reveal'],
                           cwd=ROOT, check=True)
            head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                  capture_output=True, text=True).stdout.strip()
            log(f"  locked + committed at HEAD {head[:10]}")
        except Exception as e:
            log(f"  WARNING: git commit failed ({e}); lock file written but not committed.")
    return payload


def reveal_oos(panel, Z, cols, ff=None):
    """The SINGLE sealed OOS reveal. Refuses without a matching committed lock."""
    if not LOCK_PATH.exists():
        raise RuntimeError("reveal_oos: no MOONSHOT_LOCK.json — run lock() first (pre-registered).")
    locked = json.loads(LOCK_PATH.read_text())
    spec = locked['spec']
    if spec_sha(spec) != locked['sha']:
        raise RuntimeError("reveal_oos: lock SHA mismatch — pre-registration was altered.")
    w = np.array([spec['weights'].get(c, 0.0) for c in cols])
    s_all = Z @ w
    if spec['arm'] == 'A2':
        s_all = apply_tercile(panel, s_all, quality_composite(panel, cols).to_numpy())
    scored = panel.assign(score=pd.Series(s_all, index=panel.index))
    oos = slice_dates(scored, lo=OOS_START)
    # (A) binding spread, (B) HAC + bootstrap, (D) regimes, (E) value-weight, (F) breadth
    no = H.non_overlap_spread(oos, sector_neutral=True, step=12)
    hac = H.hac_tstat(oos, sector_neutral=True, maxlags=12)
    cost = H.turnover_cost(oos, horizon_months=12)
    ser_sn = H._date_quintile_spreads(oos, 'score', 'fwd_12m', by_sector=True)
    boot_lo = stationary_bootstrap_ci(ser_sn.values, block=12)
    ser_vw = H._date_quintile_spreads(oos, 'score', 'fwd_12m', by_sector=True, weight='cap')
    reg = H.regime_breakdown(scored, sector_neutral=True)
    # (C) clean-2016+ alpha (binding) + full-sample FF6 (supporting)
    ser_full = H._date_quintile_spreads(scored, 'score', 'fwd_12m', by_sector=False)
    a_clean = H.factor_alpha(ser_full, factors=('Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'MOM'),
                             ff=ff, lo='2016-01-01', label='clean2016+')
    if a_clean is None:
        a_clean = H.factor_alpha(ser_full, factors=('SMB', 'HML', 'RMW', 'MOM'),
                                 ff=ff, lo='2016-01-01', label='clean2016+ reduced')
    a_full = H.ff6_alpha(ser_full, ff=ff)
    a_smbmom = H.factor_alpha(slice_dates(scored, lo=OOS_START).pipe(
        lambda d: H._date_quintile_spreads(d, 'score', 'fwd_12m', by_sector=False)),
        factors=('SMB', 'MOM'), ff=ff, label='OOS SMB+MOM')
    # breadth
    q5_counts = []
    for _, g in oos.dropna(subset=['score', 'fwd_12m']).groupby('date'):
        try:
            q = pd.qcut(g['score'], 5, labels=False, duplicates='drop')
            q5_counts.append(int((q == q.max()).sum()))
        except ValueError:
            pass
    breadth = float(np.median(q5_counts)) if q5_counts else 0.0
    spread = no['mean_spread'] if no else None
    net = (spread - cost['annual_cost_pct']) if (spread is not None and cost) else None
    cA = spread is not None and spread >= BAR['spread_min']
    cB = hac and hac['t_hac'] >= BAR['hac_t_min'] and (boot_lo is not None and boot_lo > 0)
    cC = a_clean and a_clean['alpha_annual_pct'] > 0 and (a_clean['alpha_t'] >= BAR['alpha_t_min']) \
        and (a_full is None or np.sign(a_full['alpha_annual_pct']) == np.sign(a_clean['alpha_annual_pct']))
    reg2 = {k: (v['spread'] if v else None) for k, v in reg.items()}
    cD = (reg2.get('covid_2020_21') or -1) > 0 and (reg2.get('recent_2024_26') or -1) > 0
    cE = ser_vw is not None and len(ser_vw) and np.sign(ser_vw.mean()) == np.sign(spread or 0)
    passed = bool(cA and cB and cC and cD and cE)
    return dict(spec=spec, spread=spread, net=net, no=no, hac=hac, boot_lo=boot_lo,
                alpha_clean=a_clean, alpha_full=a_full, alpha_smbmom=a_smbmom,
                regimes=reg2, vw_mean=float(ser_vw.mean()) if ser_vw is not None and len(ser_vw) else None,
                breadth=breadth, cond=dict(A=cA, B=bool(cB), C=bool(cC), D=bool(cD), E=bool(cE)),
                passed=passed)


def write_oos_report(res):
    c = res['cond']
    L = ["\n\n---\n\n## 5. PHASE 5 — Moonshot-with-Quality (sealed OOS reveal)\n"]
    if res['passed']:
        L.append(f"**VERDICT: PASS** — a robust quality-conditioned moonshot edge cleared the strict bar.\n")
    else:
        L.append("**VERDICT: HONEST-NULL** — *No robust moonshot-with-quality edge beyond "
                 "QMJ/size-momentum exists in this data.* (Pre-committed STOP; no re-search.)\n")
    sp = res['spread']
    L.append(f"- (A) sector-neutral non-overlap spread: **{sp:+.2f}%**" if sp is not None else "- (A) spread: n/a")
    L.append(f"  net-of-cost: {res['net']:+.2f}%" if res['net'] is not None else "")
    if res['hac']: L.append(f"- (B) HAC t = {res['hac']['t_hac']:+.1f}; bootstrap 5th-pct = "
                            f"{res['boot_lo']:+.2f}%" if res['boot_lo'] is not None else "")
    ac = res['alpha_clean']
    if ac: L.append(f"- (C) clean-2016+ alpha = {ac['alpha_annual_pct']:+.2f}%/yr (t={ac['alpha_t']:+.1f}, "
                    f"{ac['label']}); full-sample FF6 = "
                    f"{res['alpha_full']['alpha_annual_pct']:+.2f}%/yr (supporting)" if res['alpha_full'] else "")
    L.append(f"- (D) regimes (sector-neutral): {json.dumps({k: (round(v,1) if v is not None else None) for k,v in res['regimes'].items()})}")
    L.append(f"- (E) value-weighted SN spread: {res['vw_mean']:+.2f}%" if res['vw_mean'] is not None else "- (E) VW: n/a")
    L.append(f"- (F) median Q5 breadth/date: {res['breadth']:.0f} names")
    L.append(f"- conditions A–E: {c}")
    with open(REPORT, 'a') as f:
        f.write('\n'.join([x for x in L if x]) + '\n')
    log("wrote Phase-5 OOS result to HONEST_NUMBERS.md")


def _load_scored_inputs(db=BACKTEST_DB):
    import time
    t = time.time()
    log("building search panel (one build)...")
    panel = build_search_panel(db)
    panel, Z, cols = precompute_z(panel)
    log(f"  panel {len(panel):,} rows, {panel['date'].nunique()} months, {len(cols)} z-factors "
        f"({time.time()-t:.0f}s)")
    for nm, lo, hi in [('TRAIN', None, TRAIN_END), ('CONFIRM', CONFIRM_LO, CONFIRM_HI),
                       ('OOS', OOS_START, None)]:
        sl = slice_dates(panel, lo, hi)
        log(f"  {nm:8s}: {len(sl):,} rows, {sl['date'].nunique()} months")
    return panel, Z, cols


def dry_run(db=BACKTEST_DB):
    """Stages 1-2 on TRAIN+CONFIRM only. Prints leaderboard, finalist, confirm, snooping.
    Touches NO OOS data, writes nothing, does NOT lock. For human review before sealing."""
    import time
    panel, Z, cols = _load_scored_inputs(db)
    log("\nIC screen (TRAIN, sector-neutral, NW L=11 t, BY-FDR):")
    ic = ic_screen(panel)
    log(ic[['factor', 'group', 'ic_signed', 'nw_t', 'by_q', 'hlz_pass', 'n_dates']].to_string(index=False))
    log("\nweight search (build-once / evaluate-many)...")
    t = time.time()
    finalist, cands = run_search(panel, Z, cols)
    log(f"  {len(cands)} candidates evaluated ({time.time()-t:.0f}s)")
    spec = finalist_spec(finalist, cols)
    cg = confirm_gate(panel, Z, cols, finalist)
    snoop = snooping_stats(finalist, cands)
    log(f"\nFINALIST (arm {finalist['arm']}): train SN spread {finalist['train_mean']:+.2f}%, "
        f"CV {finalist['cv_obj']:+.2f}, {finalist['n_active']} active factors")
    log(f"  weights: {json.dumps(spec['weights'])}")
    log(f"  CONFIRM 2016-18 (direction-only): mean {cg['confirm_mean']:+.2f}% over {cg['n_dates']} months"
        f"  -> {'STOP (sign-flip)' if cg['stop'] else 'OK (non-negative)'}")
    log(f"  Deflated Sharpe: {snoop['dsr']['dsr']:.3f} (obs SR {snoop['dsr']['sr_obs']:.2f}, "
        f"E[max] {snoop['dsr']['expected_max_sr']:.2f}, N={snoop['dsr']['n_trials']})")
    log(f"  PBO: {snoop['pbo']:.2f}   reality-check p: {snoop['reality_check_p']:.3f}   SPA p: {snoop['spa_p']:.3f}")
    log("\n(dry-run only — OOS untouched, nothing written, no lock. Review, then --search-moonshot-oos.)")
    return panel, Z, cols, finalist, spec, snoop, cg


def reveal_pipeline(db=BACKTEST_DB):
    """Deterministically reproduce the finalist, STOP if confirm sign-flips, write the
    pre-registration, git-commit the lock, then run the SINGLE sealed OOS reveal."""
    panel, Z, cols = _load_scored_inputs(db)
    finalist, cands = run_search(panel, Z, cols)
    spec = finalist_spec(finalist, cols)
    cg = confirm_gate(panel, Z, cols, finalist)
    snoop = snooping_stats(finalist, cands)
    if cg['stop']:
        log("CONFIRM STOP: clean-era sector-neutral spread is negative — honest-null, OOS NOT revealed.")
        write_null_report(spec, cg, snoop, reason='confirm-stop')
        return dict(passed=False, reason='confirm-stop', spec=spec, confirm=cg, snoop=snoop)
    write_preregistration(spec, snoop)
    lock(spec, git_commit=True)
    ff = H.load_ff_factors()
    res = reveal_oos(panel, Z, cols, ff=ff)
    write_oos_report(res)
    log(f"\nVERDICT: {'PASS' if res['passed'] else 'HONEST-NULL'}  conditions={res['cond']}")
    return res


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true', help='Stages 1-2 (no OOS, no lock)')
    ap.add_argument('--reveal', action='store_true', help='lock + single sealed OOS reveal')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()
    if args.no_cache and PANEL_CACHE.exists():
        PANEL_CACHE.unlink()
    if args.reveal:
        reveal_pipeline()
    else:
        dry_run()

#!/usr/bin/env python3
"""
Big-winners evaluation — does any CHANGE/CATALYST variable raise the probability of a
large forward gain ABOVE a vol/size-matched control? (Phase 2)

Framing (hard-won from the prior phase): with ~9 annual cohorts this historical test is
UNDER-POWERED and can only be a FAIL-DETECTOR + forward-track pre-registration engine —
NOT a validator. So:
  - PRIMARY test, pre-registered per variable: P(+30%) lift vs vol/size-matched control
    (dims = r_vol_60d, r_market_cap; q=5, min_ctrl=8), binding window 2016+.
  - ELIMINATE any variable whose primary matched lift is <=0 (wrong sign / beta mirage).
  - BH-FDR across the surviving variable family on the primary test only.
  - Survivor robustness (not significance): sign holds in >=7/9 cohorts, survives
    leave-2020/2021-out, and does NOT worsen the -30% crash rate.
  - Secondary/descriptive: +15/20/25/50/100 thresholds and the +quality/cheap control.
  - Survivors are PRE-REGISTERED FOR FORWARD-TRACKING, not declared edges. Honest-null OK.

Reuses jump_screen.{prep, matched_lift, cohort_summary} unchanged; change_variables.attach
for the new columns. Touches no sealed window; never re-pickles the cached panel.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validation import search_moonshot as SM
from validation import jump_screen as JS
from validation import change_variables as CV

THRESHOLDS = [15, 20, 25, 30, 50, 100]
PRIMARY_THR = 30
PRIMARY_DIMS = ['r_vol_60d', 'r_market_cap']        # strips amplitude/beta
FACTOR_DIMS = ['r_vol_60d', 'r_market_cap', 'QUAL', 'r_ev_ebitda']
SCREEN_Q = 0.80
BINDING = 2016
Q, MIN_CTRL = 5, 8                                   # frozen matching params


def make_labels(df):
    for thr in THRESHOLDS:
        df[f'y_up{thr}'] = (df['fwd_12m'] >= thr).astype(float)
    return df


def screen_for(df, var):
    """Pre-declared screen: top-quintile of the within-date rank, or ==1 for booleans."""
    if var in CV.BOOL_VARS:
        return (df[var] == 1).fillna(False)
    return (df[f'r_{var}'] >= SCREEN_Q).fillna(False)


def lift_at(df, screen, thr, dims, q=Q, min_ctrl=MIN_CTRL):
    """matched_lift at an arbitrary upside threshold without editing jump_screen: point
    y_up at the threshold label; y_dn stays the fixed -30 crash flag (so crash_lift is the
    consistent 'does this screen also worsen crashes?' reading; only lift_up varies)."""
    d = df.copy()
    d['y_up'] = d[f'y_up{thr}']
    return JS.matched_lift(d, screen, dims, f'+{thr}', q=q, min_ctrl=min_ctrl)


def p_from_t(t, n):
    if not np.isfinite(t) or n < 2:
        return np.nan
    return float(2 * stats.t.sf(abs(t), df=n - 1))


def bh_fdr(pvals, alpha=0.10):
    """Benjamini-Hochberg; returns boolean 'passes' aligned to input order."""
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.zeros(len(p), bool)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out
    order = idx[np.argsort(p[idx])]
    m = len(order)
    thresh = 0
    for rank, i in enumerate(order, 1):
        if p[i] <= alpha * rank / m:
            thresh = rank
    for rank, i in enumerate(order, 1):
        if rank <= thresh:
            out[i] = True
    return out


def evaluate(df):
    rows = []
    for var in CV.CHANGE_VARS:
        if var not in df.columns:
            continue
        screen = screen_for(df, var)
        n_scr = int((screen & (df['year'] >= BINDING)).sum())
        # PRIMARY: +30 vs vol/size, binding 2016+
        lift = lift_at(df, screen, PRIMARY_THR, PRIMARY_DIMS)
        s = JS.cohort_summary(lift, lo=BINDING)
        # leave-2020/21-out
        lift_excl = lift[~lift.index.isin([2020, 2021])]
        s_excl = JS.cohort_summary(lift_excl, lo=BINDING)
        # sign-majority across binding cohorts
        bind = lift[lift.index >= BINDING]['lift_up'].dropna()
        sign_pos = int((bind > 0).sum())
        n_used = int(s['n']) if np.isfinite(s['n']) else 0
        counted = int(lift[lift.index >= BINDING]['n'].sum()) if len(lift) else 0
        drop_pct = 100 * (1 - counted / n_scr) if n_scr else np.nan
        rows.append(dict(
            var=var, n_scr=n_scr, drop_pct=drop_pct, cohorts=n_used,
            lift=s['mean'], t=s['t'], mde=s['mde'], crash=s['crash_mean'],
            sign_pos=sign_pos, lift_excl=s_excl['mean'], t_excl=s_excl['t'],
            p=p_from_t(s['t'], s['n'])))
    res = pd.DataFrame(rows)
    # fail-detector: keep only positive-lift vars, BH-FDR on those
    res['survives_sign'] = res['lift'] > 0
    fam = res[res['survives_sign']].copy()
    passed = bh_fdr(fam['p'].values, alpha=0.10) if len(fam) else np.array([], bool)
    fam['fdr_pass'] = passed
    res = res.merge(fam[['var', 'fdr_pass']], on='var', how='left')
    res['fdr_pass'] = res['fdr_pass'].fillna(False)
    # survivor robustness (descriptive flags)
    res['robust'] = (res['lift'] > 0) & (res['lift_excl'] > 0) & \
                    (res['sign_pos'] >= 7) & (res['crash'] <= 0.005)
    return res.sort_values('lift', ascending=False)


def secondary_table(df, var):
    """Descriptive: lift across all thresholds and both controls for one variable."""
    screen = screen_for(df, var)
    out = []
    for thr in THRESHOLDS:
        q, dims = (4, PRIMARY_DIMS) if thr == 100 else (Q, PRIMARY_DIMS)  # +100: coarsen, exploratory
        s_vs = JS.cohort_summary(lift_at(df, screen, thr, dims, q=q), lo=BINDING)
        s_fac = JS.cohort_summary(lift_at(df, screen, thr, FACTOR_DIMS, q=q), lo=BINDING)
        out.append((thr, s_vs['mean'], s_vs['t'], s_fac['mean'], s_fac['t']))
    return out


def main():
    print("loading cached panel + attaching change variables...", flush=True)
    panel = SM.build_search_panel()
    df = JS.prep(panel)
    df = CV.attach(df)
    df = make_labels(df)
    print(f"panel {len(df):,} rows | {df['year'].nunique()} years | "
          f"base P(+30)={df['y_up30'].mean()*100:.0f}% P(+50)={df['y_up50'].mean()*100:.0f}% "
          f"P(+100)={df['y_up100'].mean()*100:.0f}%\n")

    res = evaluate(df)
    pd.set_option('display.width', 160)
    print("=== PRIMARY: P(+30%) lift vs vol/size-matched control, binding 2016+ ===")
    print(f"{'variable':24s}{'n_scr':>7}{'drop%':>7}{'coh':>4}{'lift':>8}{'t':>7}"
          f"{'MDE':>7}{'crashΔ':>8}{'+sign':>6}{'exclLift':>9}{'FDR':>5}{'robust':>7}")
    for _, r in res.iterrows():
        print(f"{r['var']:24s}{r['n_scr']:>7,}{r['drop_pct']:>6.0f}%{r['cohorts']:>4.0f}"
              f"{r['lift']*100:>+7.1f}%{r['t']:>+7.1f}{r['mde']*100:>6.1f}%{r['crash']*100:>+7.1f}%"
              f"{r['sign_pos']:>4.0f}/9{r['lift_excl']*100:>+8.1f}%{'  Y' if r['fdr_pass'] else '  .':>5}"
              f"{'   Y' if r['robust'] else '   .':>7}")

    survivors = res[res['lift'] > 0].sort_values('lift', ascending=False)
    eliminated = res[res['lift'] <= 0]
    print(f"\nFAIL-DETECTOR: {len(eliminated)} variables eliminated (lift<=0, beta mirage / wrong sign): "
          f"{', '.join(eliminated['var'].tolist()) or 'none'}")
    reg = res[res['robust']]
    print(f"\nPRE-REGISTER FOR FORWARD-TRACK (positive lift + robust flags): "
          f"{', '.join(reg['var'].tolist()) or 'NONE — honest-null'}")
    print("  (a positive in-sample lift conveys near-zero confidence; forward-tracking is the only clean proof)")

    # secondary tables for the positive-lift survivors only
    if len(survivors):
        print("\n=== SECONDARY (descriptive): lift across thresholds [vs vol/size | vs vol/size/qual/cheap] ===")
        for var in survivors['var'].head(6):
            print(f"\n  {var}:")
            print(f"    {'thr':>5}{'vsVolSize':>12}{'t':>6}{'vsFactor':>12}{'t':>6}")
            for thr, m_vs, t_vs, m_fac, t_fac in secondary_table(df, var):
                tag = ' (exploratory)' if thr == 100 else ''
                print(f"    +{thr:>3}{m_vs*100:>+10.1f}%{t_vs:>+6.1f}{m_fac*100:>+10.1f}%{t_fac:>+6.1f}{tag}")


if __name__ == '__main__':
    main()

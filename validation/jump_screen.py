#!/usr/bin/env python3
"""
Jump Screen — find stocks with a higher PROBABILITY of a big absolute jump (+30%/yr),
with controlled downside. See ~/.claude/plans/can-you-review-the-playful-pie.md.

This is NOT an alpha search (that returned honest-null). The target is hit-rate /
asymmetry, judged against matched benchmarks so we don't mislabel factor beta as skill.

S1 (this stage) is the empirical NO-GO tripwire:
  - measure the in-sample lift of the hand-screen vs TWO matched controls
      (vol/size  -> strips amplitude/beta; vol/size/quality/cheap -> strips factor premia)
  - measure breadth (names/cohort) and the real binding-cohort count + MDE
  - STOP if the lift is clearly <=0 (it's just beta) or breadth is too thin.
A POSITIVE in-sample lift conveys NO confidence (it's the hand-tuned number on its own
data) — only a fail saves work. Real validation is the forward-track (years out).

Reuses the cached factor panel + frozen factor library from the moonshot search. Touches
no sealed window; S1 is descriptive only.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validation import search_moonshot as SM   # cached panel + frozen factor library

JUMP_UP = 30.0          # +30% in 12m = "jump"
CRASH_DN = -30.0        # -30% in 12m = "crash"
BINDING_FROM = '2016-01-01'   # survivorship-clean window

# Quality composite (sign-applied, higher = better quality) — same factors as the
# moonshot QUALITY group, averaged as within-month percentile ranks.
QUAL_FACTORS = ['gp_assets', 'roe', 'profit_stability', 'cash_conversion', 'net_margin']


def _rank(df, col):
    """Within-month cross-sectional percentile rank (0..1) of a raw factor column."""
    return df.groupby('date')[col].rank(pct=True)


def prep(panel):
    """Attach the percentile-rank columns, the quality composite, jump/crash labels,
    and the calendar year (cohort). Returns a copy."""
    df = panel.dropna(subset=['fwd_12m']).copy()
    df['year'] = df['date'].dt.year
    for c in set(QUAL_FACTORS + ['vol_60d', 'market_cap', 'small_cap', 'ev_ebitda',
                                 'prox_52w_high', 'asset_growth', 'cash_conversion',
                                 'profit_stability', 'debt_assets']):
        if c in df.columns:
            df[f'r_{c}'] = _rank(df, c)
    df['QUAL'] = df[[f'r_{c}' for c in QUAL_FACTORS if f'r_{c}' in df.columns]].mean(axis=1)
    df['y_up'] = (df['fwd_12m'] >= JUMP_UP).astype(float)
    df['y_dn'] = (df['fwd_12m'] <= CRASH_DN).astype(float)
    return df


def hand_screen(df):
    """The exploratory 'jump screen' hypothesis (the thing S1 stress-tests):
    high-vol + quality + smaller + cheap, de-risked. Returns a boolean mask.
    NOTE: this is a hand-tuned, hindsight-contaminated structure — S1 only asks
    whether its in-sample edge survives the matched controls; a pass means nothing."""
    m = (df['r_vol_60d'] >= 0.70)          # high volatility (amplitude)
    m &= (df['QUAL'] >= 0.60)              # quality composite
    m &= (df['r_small_cap'] >= 0.50)       # smaller
    m &= (df['r_ev_ebitda'] <= 0.50)       # cheap (low EV/EBITDA)
    # de-risk filters
    m &= (df['r_profit_stability'] >= 0.40)   # stable profits (not unstable)
    m &= (df['r_prox_52w_high'] >= 0.35)      # not a falling knife
    m &= (df['r_cash_conversion'] >= 0.35)    # real cash backing (not cash-burn)
    m &= (df['r_asset_growth'] <= 0.75)       # not over-expanding
    m &= (df['r_vol_60d'] <= 0.90)            # not the most extreme vol (blow-up)
    m &= (df['r_ev_ebitda'] >= 0.12)          # not the deepest value-trap
    return m.fillna(False)


def _decile(s, q=10):
    """Within-group decile 0..q-1 from a [0,1] percentile rank. Kept as float so NaN
    cells stay NaN and drop out of the groupby matching."""
    return np.minimum(np.floor(s.astype(float) * q), q - 1)


def matched_lift(df, screen, dims, label, q=5, min_ctrl=8):
    """Decile/quantile-cell matched control, aggregated at the (year, cell) level for
    robustness with multi-dim matches. Bin the universe by `dims` (q-quantiles of each
    within-month percentile rank); the control for a screened name = the NON-screen
    +30%/crash rate in its (year, cell). Returns a per-cohort (year) paired-lift frame."""
    d = df.copy()
    d['_scr'] = screen.values
    for c in dims:
        d[f'_d_{c}'] = _decile(d[c], q)
    cellcols = ['year'] + [f'_d_{c}' for c in dims]
    nonscr = d[~d['_scr']].dropna(subset=[f'_d_{c}' for c in dims])
    g = nonscr.groupby(cellcols)
    cell = pd.DataFrame({'ctrl_up': g['y_up'].mean(), 'ctrl_dn': g['y_dn'].mean(),
                         'ctrl_n': g['y_up'].size()})
    scr = d[d['_scr']].dropna(subset=[f'_d_{c}' for c in dims]).merge(
        cell, on=cellcols, how='left')
    scr = scr[scr['ctrl_n'] >= min_ctrl]
    rows = []
    for yr, gg in scr.groupby('year'):
        if len(gg) < 5:
            continue
        rows.append(dict(year=int(yr), n=len(gg),
                         scr_up=gg['y_up'].mean(), ctrl_up=gg['ctrl_up'].mean(),
                         lift_up=gg['y_up'].mean() - gg['ctrl_up'].mean(),
                         scr_dn=gg['y_dn'].mean(), ctrl_dn=gg['ctrl_dn'].mean(),
                         crash_lift=gg['y_dn'].mean() - gg['ctrl_dn'].mean()))
    cols = ['year', 'n', 'scr_up', 'ctrl_up', 'lift_up', 'scr_dn', 'ctrl_dn', 'crash_lift']
    out = (pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)).set_index('year')
    out.attrs['label'] = label
    return out


def cohort_summary(lift_df, lo=None):
    """Cohort-clustered mean lift + t + MDE over the (optionally binding) cohorts."""
    d = lift_df if lo is None else lift_df[lift_df.index >= lo]
    L = d['lift_up'].dropna().values
    n = len(L)
    if n < 2:
        return dict(n=n, mean=np.nan, t=np.nan, sd=np.nan, mde=np.nan,
                    crash_mean=float(d['crash_lift'].mean()) if len(d) else np.nan)
    mean, sd = L.mean(), L.std(ddof=1)
    t = mean / (sd / np.sqrt(n)) if sd > 0 else np.nan
    mde = 2.8 * sd / np.sqrt(n)            # ~MDE at 80% power, two-sided 0.05
    return dict(n=n, mean=float(mean), t=float(t), sd=float(sd), mde=float(mde),
                crash_mean=float(d['crash_lift'].mean()))


def s1_report():
    """S1 NO-GO tripwire: in-sample lift vs both controls + breadth + cohort count/MDE."""
    print("loading cached panel...", flush=True)
    panel = SM.build_search_panel()
    df = prep(panel)
    screen = hand_screen(df)
    print(f"panel {len(df):,} rows, {df['year'].nunique()} years; "
          f"screen selects {int(screen.sum()):,} rows ({screen.mean()*100:.1f}%)\n")

    # breadth per cohort + per month
    by_year = df[screen].groupby('year').size()
    by_month = df[screen].groupby('date').size()
    print("BREADTH:")
    print(f"  names/year: median {by_year.median():.0f}  min {by_year.min()}  "
          f"5th-pct {np.percentile(by_year,5):.0f}")
    print(f"  names/month: median {by_month.median():.0f}  min {by_month.min()}  "
          f"5th-pct {np.percentile(by_month,5):.0f}\n")

    controls = [
        (['r_vol_60d', 'r_market_cap'], 'vs vol/size  (strips amplitude/beta)'),
        (['r_vol_60d', 'r_market_cap', 'QUAL', 'r_ev_ebitda'],
         'vs vol/size/quality/cheap  (strips factor premia)'),
    ]
    print(f"{'CONTROL':48s}{'cohorts':>8}{'lift':>8}{'t':>7}{'MDE':>7}{'crashΔ':>8}")
    summaries = {}
    for dims, lab in controls:
        lift = matched_lift(df, screen, dims, lab)
        s_all = cohort_summary(lift)
        s_bind = cohort_summary(lift, lo=2016)
        summaries[lab] = s_bind
        print(f"{lab:48s}{s_bind['n']:>8}{s_bind['mean']*100:>+7.1f}%{s_bind['t']:>+7.1f}"
              f"{s_bind['mde']*100:>6.1f}%{s_bind['crash_mean']*100:>+7.1f}%   "
              f"[full-sample lift {s_all['mean']*100:+.1f}%]")

    # vs the raw universe base rate too (supporting only)
    base = df['y_up'].mean(); scr_up = df[screen]['y_up'].mean()
    print(f"\n  (supporting) raw screen P(+30%)={scr_up*100:.0f}% vs universe {base*100:.0f}%"
          f" = {(scr_up-base)*100:+.0f}pp; crash {df[screen]['y_dn'].mean()*100:.0f}% vs {df['y_dn'].mean()*100:.0f}%")

    # NO-GO decision
    volsize = summaries['vs vol/size  (strips amplitude/beta)']
    factor = summaries['vs vol/size/quality/cheap  (strips factor premia)']
    thin = np.percentile(by_month, 5) < 10
    print("\n=== S1 NO-GO TRIPWIRE ===")
    print(f"  binding cohorts (2016+): {volsize['n']}  | MDE ~{volsize['mde']*100:.1f}pp "
          f"vs ~+2pp target -> walk-forward is {'UNDER-POWERED (FAIL-detector only)' if volsize['mde']>0.03 else 'borderline'}")
    if volsize['mean'] <= 0:
        print("  -> NO-GO: lift vs vol/size-matched is <=0. The screen is just a beta/amplitude bet. STOP.")
    elif thin:
        print(f"  -> NO-GO: breadth too thin (5th-pct month {np.percentile(by_month,5):.0f} < 10). STOP.")
    else:
        verdict = ("FACTOR TILT (named exposure, not skill)" if factor['mean'] <= 0
                   else "survives both controls in-sample (hypothesis only — NOT validated)")
        print(f"  -> proceed to S2/S3 (hypothesis buildable). Nature of edge: {verdict}.")
        print("     NOTE: a positive in-sample lift conveys NO confidence; real proof is the forward-track.")


if __name__ == '__main__':
    s1_report()

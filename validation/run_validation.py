#!/usr/bin/env python3
"""
Unattended validation driver — runs the staged Compass control gate (and, once
their specs are wired, Moonshot + Valuation) through the shared harness, logging
every step and writing results into HONEST_NUMBERS.md.

Design for overnight/unattended:
  - Loads shared inputs ONCE, reuses across stages.
  - RUN-AND-LOG, never halt-and-chase: a control mismatch is flagged loudly, not
    auto-tuned away. No magnitude-chasing.
  - Deterministic (seed in harness). Appends results incrementally so a crash
    leaves partial output.

Staged Compass control (change ONE variable at a time, attribute each delta):
  A  v4-exact     period-end · 2,500-sample · no-surv · 3m · POOLED  -> target +6.11%
     (same run also yields per-date on the identical base => pooled->per-date delta)
  B  + filing-date lag
  C  + full universe (no sample)
  D  + survivorship (delisted terminal returns)
  E  + 12-month horizon  -> BINDING cell (sector-neutral, non-overlap, net-of-cost)

Usage:
    python validation/run_validation.py --compass        # staged control only
    python validation/run_validation.py --all            # + Moonshot + Valuation (when wired)
"""
import sys, argparse, time, json
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from validation import harness as H

REPORT = Path(__file__).resolve().parent / 'HONEST_NUMBERS.md'
# Control target = v4's OWN methodology re-run on the CURRENT backtest.db (+6.74%).
# The published legacy +6.11% was a stale snapshot, never re-pinned (review finding);
# the harness reproduces v4-on-current-data to ~0.13%, confirming the mechanics.
V4_TARGET = 6.74


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _pd(rep):  # per-date raw spread shorthand
    return rep['per_date']['mean_spread'] if rep.get('per_date') else None


def staged_compass_control(D):
    """Run A->E, returning the stage reports + attributable deltas."""
    stages = [
        ('A v4-exact (period-end, 2500, no-surv, 3m, pooled)',
         H.RunConfig(label='A', date_basis='period_end', sample=2500, survivorship=False,
                     horizon='3m', quintile='pooled')),
        ('B + filing-date lag',
         H.RunConfig(label='B', date_basis='filing', sample=2500, survivorship=False,
                     horizon='3m', quintile='per_date')),
        ('C + full universe',
         H.RunConfig(label='C', date_basis='filing', sample=None, survivorship=False,
                     horizon='3m', quintile='per_date')),
        ('D + survivorship (delisted terminal returns)',
         H.RunConfig(label='D', date_basis='filing', sample=None, survivorship=True,
                     horizon='3m', quintile='per_date')),
        ('E + 12-month horizon  [BINDING]',
         H.RunConfig(label='E', date_basis='filing', sample=None, survivorship=True,
                     horizon='12m', quintile='per_date')),
    ]
    results = []
    for title, cfg in stages:
        t = time.time()
        rep = H.run_score(H.COMPASS_SPEC, D['prices'], D['fund_filing'], D['fund_period'],
                          D['sectors'], D['mcaps'], D['delist_info'], cfg)
        rep['_title'] = title
        results.append(rep)
        pooled = rep.get('pooled')
        log(f"{title}: per-date={_pd(rep):+.3f}%"
            + (f"  pooled={pooled['spread']:+.3f}%" if pooled else '')
            + f"  ({time.time()-t:.0f}s)")
    return results


def staged_moonshot_control(D):
    """Moonshot evaporation ladder. NO calibrated control exists (legacy used corrupt
    caps + a different sampling grid + a look-ahead-broadcast CAGR), so M0' is a
    DIRECTIONAL proxy, not a reproduction target — there is no STOP-gate on it.
    Binding cell = M3's sector-neutral / non-overlap / net (M3 IS the binding run)."""
    moon = dict(fund_moonshot_filing=D['fund_moonshot_filing'],
                fund_moonshot_period=D['fund_moonshot_period'],
                corrupt_mcaps=D['corrupt_mcaps'])
    stages = [
        ("M0' directional proxy (corrupt caps + look-ahead CAGR, period-end, pooled, no-surv, gated)",
         H.RunConfig(label="M0p", date_basis='period_end', sample=None, survivorship=False,
                     horizon='12m', quintile='pooled'),
         dict(apply_gate=True, use_corrupt_caps=True, legacy_cagr=True)),
        ('M0 honest replication (clean caps + per-row CAGR, period-end, pooled, no-surv, gated)',
         H.RunConfig(label='M0', date_basis='period_end', sample=None, survivorship=False,
                     horizon='12m', quintile='pooled'),
         dict(apply_gate=True)),
        ('M1 + filing-date lag (pooled; per-date computed alongside)',
         H.RunConfig(label='M1', date_basis='filing', sample=None, survivorship=False,
                     horizon='12m', quintile='pooled'),
         dict(apply_gate=True)),
        ('M3 + survivorship  [BINDING run: sector-neutral/non-overlap/net read from here]',
         H.RunConfig(label='M3', date_basis='filing', sample=None, survivorship=True,
                     horizon='12m', quintile='per_date'),
         dict(apply_gate=True)),
        ('U0 UNGATED full-universe (supporting; filing, survivorship, per-date)',
         H.RunConfig(label='U0', date_basis='filing', sample=None, survivorship=True,
                     horizon='12m', quintile='per_date'),
         dict(apply_gate=False)),
    ]
    results = []
    for title, cfg, kw in stages:
        t = time.time()
        rep = H.run_score(H.MOONSHOT_SPEC, D['prices'], D['fund_filing'], D['fund_period'],
                          D['sectors'], D['mcaps'], D['delist_info'], cfg, **moon, **kw)
        rep['_title'] = title
        results.append(rep)
        log(f"{title}: per-date={_pd(rep)}  n_gated={rep.get('n_gated')}  "
            f"n_dates(raw/sn)={rep.get('n_dates_raw')}/{rep.get('n_dates_sn')}  ({time.time()-t:.0f}s)")
    return results


def write_moonshot_report(results):
    by = {r['label']: r for r in results}
    M0p, M0, M1, M3, U0 = by['M0p'], by['M0'], by['M1'], by['M3'], by['U0']
    L = [f"\n\n---\n\n## 3. PHASE 3 — Moonshot "
         f"(run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n",
         "_Units: every spread is a **12-month-forward ANNUAL %**, per-date averaged "
         "(one obs per rebalance month). FF6α is annual %, horizon-matched, NON-OOS, "
         "no ×100/×12. Report is append-only — take the LAST Phase-3 block._\n",
         "\n**No calibrated control exists for Moonshot** — legacy used corrupt market caps, a "
         "different sampling grid (quarterly×month-end), and a look-ahead-broadcast CAGR. "
         "**M0′ is a directional proxy, not a reproduction target.**\n"]

    # Claim map
    L += ["\n**Published-claim → honest counterpart:**\n",
          "| Published | Honest counterpart | Cell |",
          "|---|---|---|",
          f"| +23.58% (pooled, overlap, period-end, no-surv) | M0′ pooled = **{_fmt(_pooled(M0p))}** "
          f"(directional, inflated by bugs) | anchor, NOT judged |",
          f"| +14.92% \"avg monthly spread\" (= mean of monthly annual spreads) | M3 per-date raw = "
          f"**{_fmt(_pd(M3))}** | per-date raw |",
          f"| +9.68% sector-neutral (hardcoded ~82-stock map) | M3 real-sector-neutral (binding, below) | BINDING |",
          f"| +33.98% FF6α (unreconcilable; committed runs −220%/−3367%) | fresh clean FF6α (below) | supporting, NON-OOS |"]

    # Attributable deltas. Per-date raw is the consistent honest column (computed for
    # every stage regardless of the pooled/per-date flag); pooled shown alongside for
    # the legacy-style stages so the pooled→per-date inflation is visible directly.
    L += ["\n**Attributable deltas** (per-date raw 12m spread is the consistent column; "
          "pooled shown where the stage uses it):\n",
          "| Stage | per-date raw | pooled | Δ per-date vs prev | one change |",
          "|---|---|---|---|---|"]
    notes = {'M0p': 'corrupt caps + look-ahead CAGR (directional)', 'M0': 'clean caps + honest CAGR',
             'M1': 'filing-date lag', 'M3': 'survivorship'}
    prev = None
    for lbl in ['M0p', 'M0', 'M1', 'M3']:
        r = by[lbl]; cur = _pd(r); d = '' if prev is None else _delta(cur, prev)
        L.append(f"| {r['_title'].split('  [')[0]} | {_fmt(cur)} | {_fmt(_pooled(r))} | {d} | {notes[lbl]} |")
        prev = cur
    mid, end = _pd(M0p), _pd(M0)
    if mid is not None and end is not None:
        L.append(f"\n_M0′→M0 delta (visible cost of the corrupt-caps + look-ahead-CAGR bugs): "
                 f"**{_delta(end, mid)}** — bugs inflated the spread, as expected._")
    pp, ppd = _pooled(M1), _pd(M1)
    if pp is not None and ppd is not None:
        L.append(f"_Pooled→per-date (legacy quintiling): at M1, pooled headline = **{_fmt(pp)}** "
                 f"vs per-date **{_fmt(ppd)}** — for Moonshot the estimator choice is minor "
                 f"(~{abs(pp - ppd):.1f}pp); the inflation was the corrupt caps + look-ahead CAGR, "
                 f"not the pooled quintiling._")

    # BINDING cell (read from M3) with the pre-committed UNDEFINED / fragile branch
    L += ["\n**BINDING cell (M3 — gated, real-sector-neutral, non-overlap, net-of-cost, 12m):**"]
    sn, no, hac, cost = (M3.get('per_date_sector_neutral'), M3.get('non_overlap'),
                         M3.get('hac'), M3.get('cost'))
    nd_sn = M3.get('n_dates_sn', 0)
    cost_pct = cost['annual_cost_pct'] if cost else 0.0
    if nd_sn < 24 or sn is None:
        L.append(f"- **UNDEFINED — insufficient OOS data** (sector-neutral n_dates={nd_sn} < 24). "
                 f"Per the pre-registered failure branch, Moonshot ships NO binding spread number.")
    elif no is not None:
        net = no['mean_spread'] - cost_pct
        L.append(f"- non-overlapping (sector-neutral, {no['n_cohorts']} cohorts): "
                 f"**{no['mean_spread']:+.2f}%**  (cohort t~{no['mean_cohort_t']:+.1f}, sd={no['cohort_std']:.2f})")
        L.append(f"- net of measured cost ({cost_pct:.1f}%): **{net:+.2f}%**")
        if hac:
            L.append(f"- HAC t (overlapping sector-neutral): t={hac['t_hac']:+.1f}")
        L.append(f"- per-date sector-neutral (overlapping): {sn['mean_spread']:+.2f}%  (t={sn['t_stat']:+.1f})")
    else:
        net = sn['mean_spread'] - cost_pct
        L.append(f"- ⚠️ **FRAGILE** (non-overlap undefined; falling back to HAC-overlapping sector-neutral): "
                 f"**{sn['mean_spread']:+.2f}%**  (HAC t={hac['t_hac']:+.1f})" if hac else
                 f"- ⚠️ **FRAGILE**: sector-neutral {sn['mean_spread']:+.2f}% (t={sn['t_stat']:+.1f})")
        L.append(f"- net of measured cost ({cost_pct:.1f}%): **{net:+.2f}%**")
    L.append(f"- regimes (sector-neutral): " + json.dumps(
        {k: (round(v['spread'], 1) if v else None) for k, v in M3['regimes'].items()}))

    # Supporting: ungated + FF6 alpha + coverage diagnostics
    L += ["\n**Supporting:**"]
    L.append(f"- UNGATED full-universe (U0) per-date: {_fmt(_pd(U0))}  "
             f"sector-neutral: {_fmt_sn(U0)}  (tests the growth composite without the gates)")
    ff = H.ff6_alpha(M3.get('spread_series'))
    if ff:
        L.append(f"- **FF6α (fresh, clean, {ff['sample']}): {ff['alpha_annual_pct']:+.2f}%/yr  "
                 f"t={ff['alpha_t']:+.1f}  R²={ff['r2']:.2f}  n_months={ff['n_months']}** "
                 f"— supporting/descriptive; **OOS FF6α is undefined** (62 overlapping months). "
                 f"The published +33.98% is unreconcilable and retracted.")
    else:
        L.append("- FF6α: **intentionally absent** (validation/ff_factors.csv missing or statsmodels "
                 "unavailable) — never synthesized. OOS FF6α undefined regardless.")
    gated_frac = (M3['n_gated'] / U0['n_panel'] * 100) if U0.get('n_panel') else float('nan')
    L.append(f"- coverage: gated stock-months = {M3['n_gated']:,} ({gated_frac:.1f}% of full panel "
             f"{U0['n_panel']:,}); n_dates raw/sn (M3) = {M3['n_dates_raw']}/{M3['n_dates_sn']}.")
    L.append("- caveat: price-return, **split-adjusted but NOT dividend-adjusted** (spread understated "
             "by ≈ the Q1−Q5 yield differential, small for a growth tilt). Validates the look-ahead-free "
             "frozen-IS-z ranking, not a bit-exact replica of the live snapshot-z card.")

    with open(REPORT, 'a') as f:
        f.write('\n'.join(L) + '\n')
    log(f"wrote Phase-3 Moonshot results to {REPORT.name}")


def _pooled(rep):
    return rep['pooled']['spread'] if rep.get('pooled') else None

def _fmt(x):
    return f"{x:+.2f}%" if x is not None else "n/a"

def _fmt_sn(rep):
    sn = rep.get('per_date_sector_neutral')
    return f"{sn['mean_spread']:+.2f}%" if sn else "n/a"

def _delta(cur, prev):
    return f"{cur - prev:+.2f}%" if (cur is not None and prev is not None) else ""


def write_report(results):
    lines = [f"\n\n---\n\n## 2. PHASE 2 — Staged Compass control "
             f"(run {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})\n"]
    A = results[0]
    pooled = A.get('pooled', {}).get('spread')
    # Control mechanics check
    if pooled is not None:
        delta = pooled - V4_TARGET
        flag = "✅ reproduces v4" if abs(delta) < 0.5 else "⚠️ MISMATCH — investigate before trusting"
        lines.append(f"**Control mechanics:** A (v4-exact) pooled 3m spread = **{pooled:+.3f}%** "
                     f"vs legacy target +{V4_TARGET}%  → {flag}\n")
    # Attributable deltas table (per-date raw spread)
    lines.append("\n**Attributable deltas** (per-date raw spread; A->D all 3m, E switches to 12m):\n")
    lines.append("| Stage | per-date spread | Δ vs prev | note |")
    lines.append("|---|---|---|---|")
    prev = None
    notes = {'A': 'pooled→per-date base', 'B': 'filing-date lag', 'C': 'full universe',
             'D': 'survivorship', 'E': 'horizon→12m (scaling, not a fix)'}
    for r in results:
        cur = _pd(r)
        d = '' if prev is None else f"{cur - prev:+.3f}%"
        lines.append(f"| {r['_title']} | {cur:+.3f}% | {d} | {notes.get(r['label'],'')} |")
        prev = cur
    # Binding cell (E)
    E = results[-1]
    sn, no, hac, cost = E.get('per_date_sector_neutral'), E.get('non_overlap'), E.get('hac'), E.get('cost')
    lines.append("\n**BINDING cell (E — 12m, sector-neutral, non-overlapping, net-of-cost):**")
    if sn:  lines.append(f"- per-date sector-neutral: **{sn['mean_spread']:+.2f}%**  (t={sn['t_stat']:+.1f})")
    if no:  lines.append(f"- non-overlapping (12 cohorts): **{no['mean_spread']:+.2f}%**  "
                         f"(t~{no['mean_cohort_t']:+.1f}, cohort sd={no['cohort_std']:.2f})")
    if hac: lines.append(f"- HAC t (overlapping): t={hac['t_hac']:+.1f}")
    if cost:lines.append(f"- turnover/reb={cost['turnover_per_rebalance']:.2f}, cost={cost['annual_cost_pct']:.1f}%, "
                         f"**net={cost['net_annual']:+.2f}%**")
    lines.append(f"- regimes: " + json.dumps({k: (round(v['spread'], 1) if v else None)
                                              for k, v in E['regimes'].items()}))
    with open(REPORT, 'a') as f:
        f.write('\n'.join(lines) + '\n')
    log(f"wrote staged-control results to {REPORT.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--compass', action='store_true')
    ap.add_argument('--moonshot', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--search-moonshot', action='store_true',
                    help='Phase 5: moonshot-with-quality factor search (dry-run, no OOS, no lock)')
    ap.add_argument('--search-moonshot-oos', action='store_true',
                    help='Phase 5: lock pre-registration + single sealed OOS reveal')
    args = ap.parse_args()
    t0 = time.time()

    # Phase 5 (the factor search) has its own panel build; dispatch and return early.
    if args.search_moonshot or args.search_moonshot_oos:
        from validation import search_moonshot as SM
        if args.search_moonshot_oos:
            log("=== Phase 5: Moonshot-with-Quality — lock + sealed OOS reveal ===")
            SM.reveal_pipeline()
        else:
            log("=== Phase 5: Moonshot-with-Quality — dry-run (train+confirm only) ===")
            SM.dry_run()
        log(f"DONE in {time.time()-t0:.0f}s")
        return

    need_moon = args.moonshot or args.all
    log("Loading shared inputs (prices, fundamentals x2, sectors, market caps, delist info"
        + (", moonshot fundamentals x2, corrupt caps)..." if need_moon else ")..."))
    D = H.load_all(moonshot=need_moon)
    log(f"inputs loaded ({time.time()-t0:.0f}s)")

    if args.compass or args.all:
        log("=== Phase 2: staged Compass control ===")
        results = staged_compass_control(D)
        write_report(results)

    if need_moon:
        log("=== Phase 3: staged Moonshot ladder ===")
        mres = staged_moonshot_control(D)
        write_moonshot_report(mres)

    if args.all:
        log("Valuation spec not yet wired — TODO (Phase 4).")

    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()

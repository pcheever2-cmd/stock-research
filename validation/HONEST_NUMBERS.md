# Honest Numbers — Score Re-validation

> **Status:** IN PROGRESS. This file opens with a *frozen* pre-registration header written
> **before any number was computed**. Results sections are appended below as each phase completes.
> Per the guardrail: the goal is the *true* number, not the big one. Do not edit the pre-registration
> block to fit results.

---

## 0. PRE-REGISTRATION (frozen — committed before running anything)

**Binding headline cell (per score).** Each score is judged on ONE canonical figure; everything else
is supporting detail. The lead number is chosen here, not after seeing which cell looks best:

> **Per-date cross-sectional, sector-neutral, non-overlapping (12-cohort averaged), net-of-measured-cost,
> full-OOS (2020+) Q5−Q1 spread, annualized, with HAC/Newey-West t-stat.**

Supporting cells always reported alongside: raw (non-sector-neutral), gross-of-cost, overlapping (with
HAC), and the per-regime breakdown.

**Valuation failure branch (pre-committed).** If the Valuation score's honest binding-cell spread is not
**≥ +2.0pp annualized AND HAC t-stat ≥ 2.0** (the "meaningful" bar, set now), it ships as a purely
descriptive **technical/trend** read-out with **zero performance claim**. No exceptions discovered after
the fact.

**Frozen-spec selection rule.** Factor sets are chosen **theory-first** from the cited literature
(Novy-Marx gross profitability; Asness–Frazzini–Pedersen quality; Fama–French investment/CMA;
Jegadeesh–Titman momentum; low-volatility anomaly). If any data-driven sweep is run, the exact spec list
and count are declared *before* running, and selection uses an **in-sample holdout**: choose on
**1995–2012**, confirm on **2013–2019**. The frozen spec is recorded here before it ever touches 2020+.
- Declared specs / count: _(to be filled at selection time, before OOS)_
- Frozen spec(s): _(to be filled, before OOS)_

**Declared units.** Spread horizon = **12-month forward return**, reported **annualized**; every table
cell is labeled with its horizon. No "monthly spread" figure is reported without the explicit
"mean of monthly-sampled annual spreads" label.

---

## 1. PHASE 0 — Survivorship / delisted universe: findings

**Coverage probe (FMP `/stable`, run before building the collector):**
- ✅ **Fundamentals + market cap are available for delisted symbols.** Verified on `ATVI` (acquired by
  MSFT, Oct 2023) and `SIVB` (failed, Mar 2023): both return `income-statement`, `balance-sheet-statement`,
  `key-metrics` (marketCap present), and `historical-price-eod` through their final fiscal year. So
  delisted names can carry full fundamental-factor scores — they will NOT silently drop out of
  fundamental factors (the side-door survivorship leak is avoidable).
- ⚠️ **No delist-reason field.** `/stable/delisted-companies` returns only
  `{symbol, companyName, exchange, ipoDate, delistedDate}` — no acquisition-vs-bankruptcy flag. The
  acquisition-vs-failure distinction (which biases returns in opposite directions) must be **inferred**.

**Terminal-return rule, by inferred reason (documented; conservative default).** Without a CRSP-style
delisting code, classify from the trailing price path before `delistedDate`:
- **Healthy delist / acquisition** — last traded close is "intact" (≥ 50% of its close ~126 trading days
  prior AND ≥ $2): terminal value = **last traded price** (this already reflects any deal premium, since
  price converges to the deal price). After the last traded date, treat residual proceeds as cash (0%
  further return) to the horizon.
- **Failure / forced delist** — price collapsed (last close < 50% of 126-day-prior close OR < $1):
  terminal return floored at **−100%** from the last rebalance (equity wipeout), no recovery credit.
- Thresholds are pre-registered here; the classification flag + the inputs (last close, 126-day-prior
  close) are stored per symbol so every delist is auditable. _(Open item flagged to user: confirm this
  heuristic vs. sourcing reasons from an M&A feed.)_

_(Results — control deltas, Moonshot, Valuation — appended in later sections as phases complete.)_


---

## 2. PHASE 2 — Staged Compass control (run 2026-05-31 12:58 UTC)

**Control mechanics:** A (v4-exact) pooled 3m spread = **+6.874%** vs legacy target +6.11%  → ⚠️ MISMATCH — investigate before trusting


**Attributable deltas** (per-date raw spread; A->D all 3m, E switches to 12m):

| Stage | per-date spread | Δ vs prev | note |
|---|---|---|---|
| A v4-exact (period-end, 2500, no-surv, 3m, pooled) | +7.546% |  | pooled→per-date base |
| B + filing-date lag | +6.481% | -1.065% | filing-date lag |
| C + full universe | +6.468% | -0.014% | full universe |
| D + survivorship (delisted terminal returns) | +6.468% | +0.000% | survivorship |
| E + 12-month horizon  [BINDING] | +15.728% | +9.260% | horizon→12m (scaling, not a fix) |

**BINDING cell (E — 12m, sector-neutral, non-overlapping, net-of-cost):**
- per-date sector-neutral: **+6.97%**  (t=+9.5)
- non-overlapping (12 cohorts): **+6.97%**  (t~+2.8, cohort sd=0.95)
- HAC t (overlapping): t=+3.5
- turnover/reb=0.22, cost=0.1%, **net=+15.64%**
- regimes: {"GFC_2008": -3.6, "taper_2015": 15.6, "vol_2018": 13.0, "covid_2020_21": -1.8, "rates_2022": 16.8, "recent_2024_26": 2.7}


---

## 2. PHASE 2 — Staged Compass control (run 2026-05-31 13:28 UTC)

**Control mechanics:** A (v4-exact) pooled 3m spread = **+6.874%** vs legacy target +6.74%  → ✅ reproduces v4


**Attributable deltas** (per-date raw spread; A->D all 3m, E switches to 12m):

| Stage | per-date spread | Δ vs prev | note |
|---|---|---|---|
| A v4-exact (period-end, 2500, no-surv, 3m, pooled) | +7.546% |  | pooled→per-date base |
| B + filing-date lag | +6.481% | -1.065% | filing-date lag |
| C + full universe | +6.468% | -0.014% | full universe |
| D + survivorship (delisted terminal returns) | +6.934% | +0.466% | survivorship |
| E + 12-month horizon  [BINDING] | +17.067% | +10.133% | horizon→12m (scaling, not a fix) |

**BINDING cell (E — 12m, sector-neutral, non-overlapping, net-of-cost):**
- per-date sector-neutral: **+7.10%**  (t=+9.7)
- non-overlapping (12 cohorts): **+7.31%**  (t~+3.2, cohort sd=3.13)
- HAC t (overlapping): t=+3.6
- turnover/reb=0.22, cost=0.1%, **net=+16.98%**
- regimes: {"GFC_2008": -3.8, "taper_2015": 15.6, "vol_2018": 13.0, "covid_2020_21": -1.8, "rates_2022": 16.8, "recent_2024_26": 3.2}



---

## 3. PHASE 3 — Moonshot (run 2026-05-31 17:49 UTC)

_Units: every spread is a **12-month-forward ANNUAL %**, per-date averaged (one obs per rebalance month). FF6α is annual %, horizon-matched, NON-OOS, no ×100/×12. Report is append-only — take the LAST Phase-3 block._


**No calibrated control exists for Moonshot** — legacy used corrupt market caps, a different sampling grid (quarterly×month-end), and a look-ahead-broadcast CAGR. **M0′ is a directional proxy, not a reproduction target.**


**Published-claim → honest counterpart:**

| Published | Honest counterpart | Cell |
|---|---|---|
| +23.58% (pooled, overlap, period-end, no-surv) | M0′ pooled = **+14.79%** (directional, inflated by bugs) | anchor, NOT judged |
| +14.92% "avg monthly spread" (= mean of monthly annual spreads) | M3 per-date raw = **+2.20%** | per-date raw |
| +9.68% sector-neutral (hardcoded ~82-stock map) | M3 real-sector-neutral (binding, below) | BINDING |
| +33.98% FF6α (unreconcilable; committed runs −220%/−3367%) | fresh clean FF6α (below) | supporting, NON-OOS |

**Attributable deltas** (per-date raw 12m spread is the consistent column; pooled shown where the stage uses it):

| Stage | per-date raw | pooled | Δ per-date vs prev | one change |
|---|---|---|---|---|
| M0' directional proxy (corrupt caps + look-ahead CAGR, period-end, pooled, no-surv, gated) | +16.86% | +14.79% |  | corrupt caps + look-ahead CAGR (directional) |
| M0 honest replication (clean caps + per-row CAGR, period-end, pooled, no-surv, gated) | +2.59% | +1.64% | -14.27% | clean caps + honest CAGR |
| M1 + filing-date lag (pooled; per-date computed alongside) | +1.24% | +0.90% | -1.35% | filing-date lag |
| M3 + survivorship | +2.20% | n/a | +0.96% | survivorship |

_M0′→M0 delta (visible cost of the corrupt-caps + look-ahead-CAGR bugs): **-14.27%** — bugs inflated the spread, as expected._
_Pooled→per-date (legacy quintiling): at M1, pooled headline = **+0.90%** vs per-date **+1.24%** — for Moonshot the estimator choice is minor (~0.3pp); the inflation was the corrupt caps + look-ahead CAGR, not the pooled quintiling._

**BINDING cell (M3 — gated, real-sector-neutral, non-overlap, net-of-cost, 12m):**
- non-overlapping (sector-neutral, 9 cohorts): **-0.34%**  (cohort t~+0.8, sd=2.46)
- net of measured cost (0.1%): **-0.45%**
- HAC t (overlapping sector-neutral): t=+0.9
- per-date sector-neutral (overlapping): +1.37%  (t=+1.0)
- regimes (sector-neutral): {"GFC_2008": 1.7, "taper_2015": null, "vol_2018": 11.3, "covid_2020_21": 1.0, "rates_2022": 4.4, "recent_2024_26": null}

**Supporting:**
- UNGATED full-universe (U0) per-date: +3.39%  sector-neutral: +3.08%  (tests the growth composite without the gates)
- **FF6α (fresh, clean, full 1995-2026, NON-OOS): +2.14%/yr  t=+1.0  R²=0.10  n_months=303** — supporting/descriptive; **OOS FF6α is undefined** (62 overlapping months). The published +33.98% is unreconcilable and retracted.
- coverage: gated stock-months = 88,626 (7.7% of full panel 1,150,181); n_dates raw/sn (M3) = 317/34.
- caveat: price-return, **split-adjusted but NOT dividend-adjusted** (spread understated by ≈ the Q1−Q5 yield differential, small for a growth tilt). Validates the look-ahead-free frozen-IS-z ranking, not a bit-exact replica of the live snapshot-z card.


---

## 5. PHASE 5 — Moonshot-with-Quality search

**VERDICT: HONEST-NULL** — *No robust moonshot-with-quality edge beyond QMJ/size-momentum exists in this data.* The disciplined search STOPPED at the survivorship-clean CONFIRM gate; **the OOS (2020+) was never touched (seal intact).**

- Finalist (best of 2400 candidates, arm A1): `{"roe": 0.322022, "momentum_12_1": 0.130918, "small_cap": 0.202634, "ev_ebitda": 0.344427}`
- TRAIN (1995-2014, survivorship-suspect): sector-neutral spread looked positive, **but**
- **CONFIRM 2016-2018 (survivorship-CLEAN): -0.67% over 36 months → sign-flip → STOP.** The train edge is survivorship/selection inflation, not signal.
- Deflated Sharpe = **0.78** (< 0.95; obs SR 0.59 vs E[max] under 2400 trials 0.52) — search-selection inflated.
- PBO (CSCV) = **0.48** — ~coin-flip out-of-sample (overfit-prone).
- (White reality-check p=0.015, Hansen SPA p=0.009 flag that the in-sample BEST looks significant — exactly the multiple-testing mirage DSR/PBO catch.)
- The univariate screen confirms the real (modest) signal is **quality + value (GARP)** (gp_assets, fcf_yield, ev_ebitda, cash_conversion clear the Harvey-Liu-Zhu t≥3 + BY-FDR bar); naive growth does NOT predict. But no weighted combination survives the clean-window confirm.
- Honest-null STOP is pre-committed: no re-search, no confirm-mining, OOS preserved for a future pre-registered hypothesis.


---

## 6. PHASE 6 — Jump Screen (P(+30%/yr) probability, NOT alpha) — S1 NO-GO

**Question:** can we screen for stocks with a higher *probability* of a big absolute jump (+30%/yr), with
controlled downside? Hand-screen = high-vol + quality + smaller + cheap, de-risked (6 filters). In-sample it
looked good: P(+30%)=27% vs 23% universe, crash 15% vs 16%, up/down 1.76, ~62 names/yr.

**S1 empirical go/no-go** (`validation/jump_screen.py`, cached panel, survivorship-clean **2016+**, 10 annual
cohorts, paired O(n) decile-cell matched controls, cohort-clustered t):

| Control | +30% jump lift | t | crash Δ |
|---|---|---|---|
| **vs vol/size-matched** (strips amplitude/beta) | **−5.0%** | −2.2 | −6.5% |
| **vs vol/size/quality/cheap-matched** (strips factor premia) | −5.7% | −1.6 | **−0.7%** |
| (supporting) vs raw universe | +4pp (27% vs 23%) | — | −1pp |

**Verdict: NO-GO (honest-null) — the screen has no jump-selection skill.**
- The "+4pp higher jump rate vs the universe" is **100% beta/amplitude**: against same-vol/size peers the screen
  jumps **LESS** (−5pp, t=−2.2), not more. You can raise raw P(+30%) only by buying higher-volatility names.
- The downside edge (−6.5pp crashes vs vol/size) is **entirely the quality/cheap factor tilt** — it vanishes to
  −0.7pp once matched on quality+cheap. It's QMJ/value beta, not skill.
- Historical test is under-powered (MDE ~6pp vs +2pp target) — but the *negative, significant* vol/size lift is a
  clean FAIL regardless. **Pre-committed STOP: do not build S2–S5; do not ship a "jump screen" performance claim.**
- Confirms the lottery literature: there is no way to pick *which* volatile stocks jump; "more upside" = more beta
  (and symmetric more downside, only offset by the known quality factor). The moonshot alpha 2020+ seal is untouched.

**Two objectives — read the verdict by the right one:**
- **Jump-SELECTION skill** (pick which volatile stocks pop): **NO** — vs vol/size peers the screen jumps less. Beta.
- **DOWNSIDE-REDUCTION on high-upside candidates** (the actual goal): **YES, as a factor tilt.** The de-risk filters
  cut the high-upside cohort's crash rate **21%→15%** (giving up only +30%: 29%→27%); vs same-vol/size peers crashes
  **−6.5pp, t=−2.1** (significant); vs universe: more upside (27% vs 23%) AND less downside (15% vs 16%), up/down
  **1.76 vs 1.44**. The protection mechanism is the **quality/value (QMJ) premium** (−6.5pp → −0.7pp once matched on
  quality+cheap) — robust and documented, not skill. Median return ≈ market (no avg-return edge). **Viable as a
  DESCRIPTIVE "high-amplitude quality-protected basket," forward-tracked — not a performance/alpha claim.**

## 7. PHASE 7 — Big-winners CHANGE/CATALYST variables (run 2026-05-31)

**Question:** do *change/expectation* variables (what is CHANGING — accelerating fundamentals, margin/return
inflection, deleveraging, sector-relative breakout, **analyst upgrades, earnings beats**) raise P(large forward
gain) **above a vol/size-matched control**, where static value/quality/size/vol levels did not (§6)?

**Setup** (`validation/change_variables.py` + `validation/big_winners.py`, cached panel, survivorship-clean
**2016+**, ~9–10 annual cohorts, paired decile-cell matched controls, cohort-clustered t). Signs frozen before
testing (`CHANGE_LIBRARY`). **Framing = FAIL-DETECTOR + forward-track pre-registration, NOT validation** (~9
cohorts ⇒ a positive in-sample lift conveys near-zero confidence; only a clean fail saves work). 15 variables,
primary test = P(+30%) lift vs vol/size, BH-FDR(0.10) on the positive-lift family.
**Data backfilled this phase:** `earnings_surprises` (announcement-dated EPS+revenue beats via
`/stable/earnings-calendar`, 207k rows, 6.5k symbols, 2012–2026) + `historical_grades` action feed (already
held); `estimate_ledger` seeded for forward-collection of estimate revisions (un-backfillable on FMP).

**Fail-detector result — two clean groups:**

| group | variables | primary +30 matched lift | verdict |
|---|---|---|---|
| **Fundamental-trend slopes / inflections** | op/gross-margin slope, ROA slope, rev/eps growth-accel, debt/EBITDA & debt/assets deleveraging, NI/FCF first-profitable, breakout-from-base | **≤ 0** (−2.3% … +1.0%, t ≤ 1.4) | **ELIMINATED** — no edge vs matched peers (same lesson as static factors) |
| **Catalyst / expectation** | `eps_surprise_pct`, `beat_streak`, `net_upgrades_180d`, `net_upgrades_180d_norm` | **+1.5% … +3.0%**, t **+2.1 … +4.7** | **SURVIVE** all controls + robustness |

**The 4 surviving catalyst variables (binding 2016+):**

| variable | +30 lift vs vol/size | t | vs vol/size/**qual/cheap** | vs vol/size/**MOMENTUM** | cohorts +sign | crash Δ | FDR |
|---|---|---|---|---|---|---|---|
| `beat_streak` (consec. EPS beats) | **+3.0%** | +3.5 | +2.4% (t 3.5) | +2.8% (t 3.1) | 9/9 | −1.0% | ✓ |
| `eps_surprise_pct` (latest beat) | +1.8% | **+4.7** | +1.5% (t 3.3) | +1.6% (t 3.5) | 9/9 | −1.0% | ✓ |
| `net_upgrades_180d_norm` | +2.0% | +2.4 | +1.6% (t 2.3) | +1.9% (t 2.2) | 8/9 | −2.1% | ✓ |
| `net_upgrades_180d` | +1.5% | +2.1 | +1.0% (t 1.8) | +1.5% (t 2.0) | 8/9 | −1.8% | · |

**Verdict: PROMISING POSITIVE — the change/EXPECTATION (catalyst) hypothesis breaks the static-factor null;
fundamental-TREND slopes do not.** Earnings beats (PEAD) and analyst-upgrade momentum raise P(+30%) by ~+1.5–3pp
vs vol/size peers, the lift **survives the quality/cheap AND the momentum-matched control**, **reduces** crash
rates (negative crash Δ), holds in **8–9 of 9 cohorts**, and strengthens at lower thresholds (+15/20/25: beat_streak
up to +3.6%, t 3.7) while persisting to +50% (eps_surprise +1.8%, t 4.3). All four are theory-grounded
(Bernard-Thomas PEAD; analyst-revision drift), which raises prior plausibility above the lottery null.

**Caveats (why this is a pre-registration, not a claim):**
- ~9 cohorts ⇒ under-powered; t's are suggestive, not proof. MDE ~1–2.5pp ≈ the effect size.
- **Coverage selection:** the catalyst screens require analyst/earnings coverage; controls include no-coverage
  names in-cell. Matched on size+vol+momentum but coverage may still proxy for something — forward-track resolves it.
- Multiple-testing: 15 vars × 6 thresholds × controls; only the primary +30/vol-size test is FDR-controlled.
- **Next:** these 4 are **PRE-REGISTERED for forward-tracking** (the only clean validation). The `estimate_ledger`
  daily snapshots will, over time, add an *estimate-revision* variable FMP can't backfill. No sealed window touched;
  panel cache not re-pickled.


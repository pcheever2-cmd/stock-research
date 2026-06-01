# Session Summary — Score Re-validation (handoff for a fresh chat)

> **Project:** re-validate the three stock scores (Compass, Moonshot, Valuation) HONESTLY,
> because the published performance numbers were invalidated by methodology bugs. This pass =
> **technical core only** — build one correct survivorship-free harness, run all three through it,
> produce the *true* numbers. Rewriting the public papers, re-anchoring the site, label fixes, and
> legal review are a **gated follow-on** (do NOT touch the live site this pass).
>
> **Guardrail (hold the line):** find the *true* number, not the big one. A defensible +5% beats a
> +40% that evaporates. When a clean run comes back smaller, that's success — do NOT tune to recover
> magnitude. Full plan: `~/.claude/plans/can-you-please-review-glittery-diffie.md`.

---

## 🟢 STATUS: Phase 0 ✅ · 1 (harness) ✅ · 2 (Compass) ✅ GREEN · 3 (Moonshot) ✅ HONEST-NULL · 5 (factor search) ✅ NULL · 6 (jump screen) ✅ NO-GO · 7 (big-winners/catalyst) ✅ SHIPPED · **Phase 4 (Valuation) = SOLE REMAINING TODO**

> **NEXT CHAT: do Phase 4 — honestly re-validate the Valuation score.** Everything else in the
> re-validation is done; Phase 4 is the last open piece. Details in the Phase 4 section below.

### How to run
```
python validation/run_validation.py --compass            # staged Compass control (~6 min)
python validation/run_validation.py --moonshot           # Moonshot honest run (~6 min)
python validation/run_validation.py --search-moonshot    # factor search dry-run (Phase 5)
python validation/big_winners.py                         # big-winners catalyst eval (Phase 7)
# Phase 4: NO --valuation flag exists yet — it must be built (see Phase 4 section).
```
Results append to `validation/HONEST_NUMBERS.md` (which opens with the FROZEN pre-registration; §3 Moonshot,
§5 factor search, §6 jump screen, §7 big-winners/catalyst are already written).

### Key files
- `validation/harness.py` — the shared harness (vectorized). Loads panel, scores, computes spreads.
- `validation/run_validation.py` — unattended driver; runs the staged control, writes the report.
- `validation/HONEST_NUMBERS.md` — **deliverable**: frozen pre-registration header + appended results.
- `collectors/collect_delisted.py` — FMP delisted list → `delisted_companies` table.
- `collectors/classify_delisted_reasons.py` — `--prep`/`--ingest` for the reason-research workflow.
- `collectors/collect_market_caps.py` — clean FMP market-cap re-pull → `historical_market_caps` table.
- DB: `backtest.db` (now ~8GB). New tables: `delisted_companies`, `historical_market_caps`.

---

## ✅ Phase 0 — Survivorship foundation (DONE)
- **Delisted universe:** 2,986 operating-co delistings classified by reason via a 150-agent web-research
  workflow. **1,867 real exits** (acquired 790 / merged 304 / went_private 296 = positive; bankrupt 156
  / distress 152 / liquidated 169 = wipeout). 1,117 excluded (not_operating 731, not_delisted 386 =
  SPAC/note/warrant/ticker-rename — correctly NOT given terminal returns). Reasons in `delisted_companies.reason`.
- **Market-cap corruption found + fixed:** `historical_key_metrics.market_cap` was systematically corrupt
  (foreign ADRs inflated 100–1000×, int64 sentinels, >1,100 symbols) — a STORAGE bug, not FMP. Re-pulled
  clean daily caps from FMP → `historical_market_caps` (28.9M rows, 8,391 symbols). Harness uses this.
- **Delisted backfill:** prices+fundamentals pulled for the real exits — 1,862/1,867 have prices, 1,799
  have fundamentals; ~1,181 land in the OOS panel.
- **Coverage caveat:** FMP's delisted list is dense only **2016+**. So the 2020+ OOS headline is
  survivorship-correctable; **pre-2016 IS selection / 2008 / 2015 regimes are NOT** — caveat them.

## ✅ Phase 1 — Harness (DONE, reviewed, fixed)
Vectorized (full ~9,500-symbol universe, not the legacy 2,500 sample). Implements: per-date
cross-sectional quintiles, filing-date lag (no look-ahead), 12-mo (binding) + 3-mo (control) horizons,
sector-neutral (real DB sectors), non-overlapping **calendar-anchored** 12-cohort + HAC/Newey-West,
turnover-scaled costs, regime breakdowns, survivorship terminal-returns-by-reason, and a v4-exact mode.

**Agent review gate caught a CRITICAL bug (now fixed):** the survivorship correction was *dead code* —
the sampling guard removed delisted names' final window before terminal returns could fill it, so
`apply_terminal_returns` filled 0 rows. Fix: `build_panel` now samples delisted names to series end
(passes `delisted_syms`) and terminal returns populate those NaNs. Also fixed: re-pinned control target
6.11→6.74, calendar-anchored cohorts (were positional), market-cap fallback floor >$1M.

## ✅ Phase 2 — Compass control GREEN
Compass is the **control** (expected to hold up). Staged control reproduces v4 and attributes each fix:

| Stage | spread | Δ | meaning |
|---|---|---|---|
| A v4-exact pooled 3m | +6.874% | — | vs re-pinned +6.74% → **reproduces v4** (Δ 0.13%) |
| filing-date lag | +6.48% | **−1.07%** | look-ahead was inflating ~1pp |
| full universe | +6.47% | −0.01% | — |
| + survivorship | +6.93% | **+0.47%** | bankruptcies' −100% widen the long-short spread |
| → 12-mo horizon | +17.07% | — | horizon scaling |

**BINDING cell (12m, sector-neutral, non-overlap, net-of-cost): +7.31%, t≈3.2** (HAC t +3.6).
Regimes: GFC −3.8, '15 +15.6, '18 +13.0, COVID −1.8, '22 +16.8, '24–26 +3.2 (NOT crash-carried).

> **⚠️ Don't misread +7.31% as "better."** It's a **12-month** number; the old "+6.79%" was 3-month
> (actually a mislabeled in-sample correlation, r=0.0679). Like-for-like the honest numbers are LOWER
> (3-mo sector-neutral ≈ +4.4%) and the **t-stat collapsed ~22→~3.2** once overlap was removed. Compass
> "holding up" tells us little about Moonshot/Valuation.

---

## ✅ Phase 3 — Moonshot (DONE, HONEST-NULL) — see HONEST_NUMBERS.md §3
Built `MOONSHOT_SPEC` (quality gates + growth weights) in the harness; `--moonshot`. **Honest finding:
Moonshot's edge is ~0.** Binding gated real-sector-neutral non-overlap = **−0.34% (net −0.45%, HAC t≈+0.9)**
vs the claimed +9.68%; the methodology bugs alone were worth **−14.27pp** (M0′ corrupt-caps+look-ahead-CAGR
+16.86% → honest M0 +2.59%). Clean full-sample FF6α +2.14%/yr t=+1.0; the +33.98% claim is unreconcilable with
any committed code → **retract**.

## ✅ Phase 5 — "Moonshot-with-quality" factor search (DONE, HONEST-NULL) — §5
`factors_moonshot_plus.py` + `search_moonshot.py` (TRAIN≤2014 / CONFIRM 2016–18 survivorship-clean direction-only
STOP / OOS 2020+ SEALED; HLZ t≥3 + BY-FDR; DSR + PBO/CSCV + Hansen SPA). Univariate signal is **quality+value
(GARP)** — gp_assets, fcf_yield, ev_ebitda, cash_conversion clear the bar; **naive growth doesn't predict.** The
CV-best finalist sign-flips on the clean CONFIRM → STOP. No robust edge beyond QMJ/size-momentum; **OOS seal
intact** (`MOONSHOT_LOCK.json`) for a future pre-registered hypothesis. Don't burn the 2020+ seal.

## ✅ Phase 6 — Jump screen P(+30%/yr) (DONE, NO-GO at S1) — §6
`jump_screen.py`. Hand-screen looked good vs the universe but **collapses vs matched controls**: vs vol/size the
jump lift is **−5.0% (t=−2.2)** (it jumps LESS than matched peers; the "+4pp" was 100% beta), and its crash
reduction vanishes to −0.7pp once matched on quality/cheap → it's the QMJ/value factor, not skill.

## ✅ Phase 7 — Big-winners change/catalyst (DONE + SHIPPED to site) — §7
`change_variables.py` + `big_winners.py`; `validation/CATALYST_SIGNAL_METHODOLOGY.md`. Fundamental-trend slopes
fail vs matched controls, but **4 CATALYST/expectation vars survive** (eps_surprise, beat_streak, net_upgrades)
— +1.5–3pp P(+30%) lift vs vol/size, holding vs momentum + quality/cheap controls, crash-reducing, 8–9/9
cohorts, theory-grounded (PEAD/revision drift). Pre-registered, forward-tracking. Backfilled `earnings_surprises`
+ seeded `estimate_ledger`. Shipped as a PREMIUM per-stock "Catalyst Signal" (exporter + daily pipeline + UI +
methodology page). See [[project_big_winners_pivot]].

---

## ⏭️ NEXT — Phase 4 (Valuation): the only remaining piece

**Goal:** honestly re-validate the Valuation score through the same harness, just like Compass/Moonshot.
The site claims **+37–44pp / 86%** for it (`compass-score-site/src/pages/valuation-methodology.astro`).

**What the live score actually is:** `compute_valuation_scores.py` = **Price-vs-SMA200 / SMA50 / 52w-position
only** — a technical/trend read-out, no fundamentals. (Separately, `value_score_v2` in
`score_long_term_OPTIMIZED.py` is a fundamentals composite — validate the SMA-based one the site claims are about;
note the distinction.)

**How to build it (mirror Phase 3):**
1. Add a `VALUATION_SPEC` in `validation/harness.py` next to `MOONSHOT_SPEC`, encoding the live SMA/52w rule
   (per-date cross-sectional ranking; these are price-derived, so use trailing/formation-date prices — no
   fundamentals, no filing-date lag needed). Wire a `--valuation` flag in `validation/run_validation.py`
   (it only has `--compass`/`--moonshot` today).
2. Run it through the binding cell (12m, sector-neutral, non-overlap, HAC/Newey-West, net-of-cost), 2016+.
3. If any parameter sweep is tempting, **IS holdout: fit 1995–2012, confirm 2013–2019, before 2020+.**

**Pre-committed failure branch (frozen):** if the honest binding spread < **+2.0pp OR HAC t < 2.0**, it ships as
a purely descriptive **technical/trend** read-out with **ZERO performance claim** — reframe, don't pit it against
fundamentals. (A trend signal beating a vol/size-matched control is the real bar; expect the lottery/beta caveat.)
Append results to HONEST_NUMBERS.md as §4 (or §8), honest-null friendly.

### Then (gated follow-on, NOT required for Phase 4 itself)
- Compile the HONEST_NUMBERS.md final report across all four scores (raw + sector-neutral + non-overlap + regime
  + HAC, units declared).
- Re-anchor the public site figures (+58/86/+37 in `valuation-methodology.astro`, +6.79 in `index.astro`, +23 in
  `moonshot-methodology.astro`) + disclaimers; relabel the live Valuation card; rewrite the 2 papers; legal review.

## Known residuals / limitations
- Pre-2016 survivorship gap (FMP coverage) → caveat IS/2008/2015.
- BBBY-style **ticker-reuse** evades the delisted list (uncorrected; a few cases).
- ~9 non-operating fund/share-class names leaked into real-exits (minor, mostly filtered).
- Residual market-cap outliers (~$10T) winsorized per-date 99.5pct; market_cap unused by Compass.
- Binding-cell pre-registration + Valuation failure threshold are FROZEN at the top of HONEST_NUMBERS.md.

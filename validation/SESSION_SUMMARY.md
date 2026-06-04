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

## 🟢 STATUS: Phase 0 ✅ · 1 (harness) ✅ · 2 (Compass) ✅ GREEN · 3 (Moonshot) ✅ HONEST-NULL · 4 (Valuation) ✅ HONEST-NEGATIVE · 5 (factor search) ✅ NULL · 6 (jump screen) ✅ NO-GO · 7 (big-winners/catalyst) ✅ SHIPPED · **ALL PHASES DONE — technical core complete**

> **The technical re-validation is COMPLETE.** All four scores have run through the one
> survivorship-clean harness. Remaining work is the GATED follow-on (rewrite papers, re-anchor the
> live site figures, relabel cards, legal) — NOT part of this pass. See "gated follow-on" below.
>
> **➡️ SINCE THEN (through 2026-06-04):** acted on the findings — shipped Golden Stocks, removed
> Moonshot, shipped the Catalyst Signal, fixed a chain of catalyst data/ops + browse-filter bugs,
> reframed the home + methodology pages. **See "📦 SHIPPED + OPS LOG" below.**
>
> **🎯 NEW WINDOW STARTS HERE:** make the **Valuation score predictive** — see "🎯 NEXT" below.

> ⚠️ **SIGNIFICANCE CORRECTION (2026-06-01, see HONEST_NUMBERS §9).** Independent review found the
> per-date spread series was effectively DAILY (symbols sampled monthly but on staggered calendar dates →
> overlapping 12-mo returns), so the published HAC/cohort t-stats were inflated ~3–4×. Fixed: estimators
> now collapse to a MONTHLY grid (`harness._to_monthly`, maxlags=12). **Corrected: NO score clears t≥2 at
> ANY window** — Compass HAC OOS +1.04 / full-sample(30y) +1.71; Value +0.78 / +1.58; Moonshot +0.88 /
> −0.10; SMA −0.47 / +1.14. **Spreads + rank-ordering UNCHANGED, only confidence.** Our crude quintile/
> sector-neutral/12m-overlap composites are too blunt to certify in-house; the value/quality basis is the
> **published literature** (FF value ~4–5%; Novy-Marx/AQR quality ~3–6% — our point estimates match in
> sign+magnitude), NOT any in-house t-stat. **Make zero in-house significance claims.**

---

## 📦 SHIPPED + OPS LOG (post-technical-core, through 2026-06-04)
The technical core (above) was the *honest re-validation*. Since then we acted on the findings and
shipped product, then chased the resulting data/ops bugs to ground. Done:

- **Golden Stocks shipped, Moonshot removed site-wide.** Golden = **top-decile Compass × top-decile
  catalyst** (premium flag `isGolden`, ~71 names live). Moonshot deleted end-to-end (score ~0); old
  `/moonshot-methodology` 301→`/golden-methodology`. See [[project_golden_stocks]].
- **Catalyst Signal shipped** as a PREMIUM per-stock signal (⚡ chip shows for everyone, *functions*
  for Plus+; ⚡+score on the card for premium). `catalystScore`/`catalystTag` premium-gated.
- **Catalyst data-staleness saga (3 distinct bugs, all fixed) — now 367 tagged / 71 Golden live:**
  1. `attach_earnings` merge_asof **dtype crash** (`<M8[ns]` vs `<M8[us]`) → forced both keys to `[ns]`.
  2. Analyst grades were **collect-once** → `net_upgrades_180d` decayed to 0 → added
     `collect_analyst_data.py --refresh-grades` (clears the 'grades' progress markers; idempotent via
     PK + INSERT OR IGNORE), wired into `daily-pipeline.yml`.
  3. Earnings history was a **14-day window** → `beat_streak` capped at 1 → made the earnings step
     **self-healing**: backfill 5y once when `earnings_surprises` is shallow, else 14d.
  - Also: **FMP retired `/stable/price-target`** (404 for all 6,678) → `price-target-news` (same fields).
  - See [[project_catalyst_grade_freshness]].
- **Browse-filter front-end bugs (the "only 8 catalyst stocks" saga):** (a) Golden+Catalyst hydration
  was gated behind the **Pro** `valuation_score` feature → Plus users got nothing → re-gated at **Plus**
  (`golden_stocks`); (b) hydration was ~85 **sequential** `/api/stocks/premium` calls (slow, showed a
  partial count) → replaced with **one** `GET /api/stocks/premium-summary` served from a single KV key
  (`summary:premium-filters`) → near-instant; (c) closed a leak (premium API now strips `valuationRating`
  for non-Pro). **Verified live KV = 367 tagged via wrangler.**
- **Home page reframe** toward the target market (find quality + understand why; removed trust-bar
  counts). **Methodology pages:** valuation rewritten honest (no perf claim), Compass wording verified,
  catalyst page linked in nav/footer.
- **DEPLOY GOTCHA (cost us a cycle):** `stockbrowse-app-pages` does **NOT** auto-deploy on git push —
  must `npx wrangler pages deploy dist --project-name=stockbrowse-app-pages --commit-dirty=true`. Premium
  data is served from **KV**, not the repo JSON. See [[reference_pages_deploy]].

---

## 🎯 NEXT (start here in the new window) — make the Valuation score PREDICTIVE
**Goal:** the current "Valuation Score" ships as *descriptive-only* (no performance claim). Build a
valuation read that actually **predicts returns** — honestly, under the same guardrails.

**What we already know (don't re-discover):**
- The **SMA-based "Valuation Score" is a dead end** for prediction: negative / regime-flipping, it's a
  short-momentum signature (§4). Do NOT try to rescue it.
- The **FUNDAMENTAL value score is the real predictive base and ALREADY WORKS** (§8): equal-weight
  earnings-yield (E/P) + fcf_yield + ebitda/EV + book_yield, sector-relative, built from TTM flows +
  CLEAN market cap. Binding sn non-overlap net 12m OOS 2020+ = **+4.85% (HAC t +2.6)**, FF6α **+4.15%/yr**.
  Run: `python3 validation/value_fundamental.py` (~4 min). Spec = `VALUE_SPEC` in `harness.py`.
- **Known failure modes to design around:** value **inverts on megacaps / analyst-covered names**
  (cheap large-caps = value traps, expensive = AI/quality compounders) (§10); naive 50/50 with Compass
  **dilutes** quality (§4b); forward P/E & PEG are **NOT backtestable** (no point-in-time data) — only
  the forward-accruing `estimate_ledger` (seeded 2026-06-01) will enable revision tests later.

**Candidate directions to test (pre-register first, theory-first):**
1. **Value-trap avoidance** — condition value on quality/size/sector so the megacap inversion doesn't
   drag it (value within quality buckets, or quality-gated value, not a 50/50 blend).
2. **Smarter quality×value integration** than the failed 50/50 (e.g., double-sort, or value only in the
   high-quality tertile).
3. **Sector/size-relative refinements** of the composite; check the value-trap/survivorship leg.
4. **Revision-based value** (forward, once `estimate_ledger` has enough history — not yet).

**Guardrails (NON-NEGOTIABLE, from §9):** monthly-grid t-stats only (`harness._to_monthly`, maxlags=12);
**zero in-house significance claims** — lean on published literature (FF value, AQR/Novy-Marx quality);
**find the true number, not the big one**; always test vs **vol/size/quality-matched controls** (value is
often a beta/HML mirage — see jump-screen §6 and big-winners §7 methodology). Don't burn the 2020+ OOS seal.

---

### How to run
```
python validation/run_validation.py --compass            # staged Compass control (~6 min)
python validation/run_validation.py --moonshot           # Moonshot honest run (~6 min)
python validation/run_validation.py --search-moonshot    # factor search dry-run (Phase 5)
python validation/big_winners.py                         # big-winners catalyst eval (Phase 7)
python validation/run_validation.py --valuation          # Valuation honest run (Phase 4, ~7 min)
```
Results append to `validation/HONEST_NUMBERS.md` (which opens with the FROZEN pre-registration; §3 Moonshot,
§4 Valuation, §5 factor search, §6 jump screen, §7 big-winners/catalyst are all written).

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

## ✅ Phase 4 — Valuation (DONE, HONEST-NEGATIVE) — see HONEST_NUMBERS.md §4
Built `VALUATION_SPEC` (price-derived: SMA200 0.50 / SMA50 0.25 / 52w-position 0.25, inverted) + `--valuation`
in the harness, mirroring Moonshot. **The live "+37pp / 86%" does NOT survive — the honest spread is NEGATIVE.**
Binding cell (real-sector-neutral, non-overlap, net, 12m, OOS 2020+) = **−3.23% (HAC t −1.5, 12 cohorts)**;
per-date raw V0 (vs-universe, no-surv) −5.50% → V1 (+survivorship) −7.37% (bankruptcies fill the undervalued
long leg, as expected). **FAILS the frozen ≥+2.0pp & t≥2.0 bar → ships as a purely descriptive technical/trend
read-out with ZERO performance claim** (no matched control needed — the conditional jump_screen control runs
only to *kill* a passing claim, never to rescue a fail). The negative spread is the **short-momentum signature**
(MOM β −0.61 OOS / −0.76 full); OOS reduced-factor α = −15%/yr (t −7.7), full-sample FF6α +2.5%/yr (t 1.0, a
factor-neutral wash). The site's +37pp was a survivorship + look-ahead + favorable-window (post-COVID rebound:
GFC +41, COVID +9; but 2022 −6, 2024–26 −7) artifact. **Validation NOT a cousin:** harness-vs-live ranking
Spearman **0.9997**, Q5 membership overlap **100%**. Sign + no-look-ahead confirmed empirically.

**Phase 4b — Valuation as a QUALITY OVERLAY (the site's actual claim): also FAILS.** The score is sold only
as a 50/50 blend with Compass ("low consistency alone → 86% hit rate combined w/ quality, maintaining
returns"). Built `validation/valuation_combined.py` to test the increment vs **Compass ALONE** (the right
benchmark) on the OOS-2020+ survivorship panel. Result: Compass-alone +7.10% sn-spread / 78% consistency /
56% hit(>median); **Combined 50/50 +6.52% / 72% / 52%** — the overlay moves EVERY metric the wrong way
(return −0.57pp, consistency −6pp, hit −4pp). The "86%" was a rising-tape base rate + the same inflation,
not an overlay gain. (Combined HAC t +6.7 > Compass +3.6 is only a variance/diversification artifact —
higher t on LOWER mean + lower consistency, not an improvement.) **Conclusion across both lenses:**
standalone it's negatively-predictive / regime-flipping; as a quality overlay it DILUTES quality. Ship as a
descriptive technical read-out only; do NOT present it as improving a quality screen. See HONEST_NUMBERS §4b.

## ✅ Phase 8 — FUNDAMENTAL VALUE score (NEW, the honest "cheap vs peers" idea) — see HONEST_NUMBERS.md §8
The SMA score isn't valuation; a fundamental sector-relative cheapness read IS. Pre-registered theory-first
`VALUE_SPEC` (equal-weight earnings_yield E/P + fcf_yield + ebitda_to_ev + book_yield B/P; built from TTM
flows + CLEAN market cap, never key_metrics' corrupt ratios) in `harness.py`; orchestrated by
`validation/value_fundamental.py`. **It WORKS (unlike SMA):** binding (sn non-overlap net 12m OOS 2020+)
**+4.85% HAC t +2.6**, full-sample FF6α **+4.15%/yr t 2.5** — a real, modest **value premium**. Caveats:
(1) lead with ~+4%, NOT the +9.99% OOS HML-controlled alpha (window-specific, leans on the 2022 rate-shock
value rebound: regimes +15 in 2022 vs −4.3 in 2020-21 growth melt-up); (2) "survives HML" mostly = our
composite is better-built value than pure-B/P HML (degraded post-2020), not proprietary alpha;
(3) regime-dependent (value droughts); (4) value-trap/survivorship effect small here (+0.25pp). **Forward
P/E / PEG: NOT backtestable** — FMP `/stable/analyst-estimates` probed directly = ~10 fiscal years
(recent+forward), no as-of field, from/to ignored; only the forward-accruing `estimate_ledger` (seeded
2026-06-01) will enable revision tests later. Product: ship a descriptive "cheap/fair/expensive vs peers"
label (forward P/E/PEG display-only); a modest, caveated value-premium note is defensible. Run:
`python3 validation/value_fundamental.py` (~4 min).

## ✅ §10 — Value × analyst-revision (descriptive exploration) — see HONEST_NUMBERS §10
Asked (descriptively, NO sig claim): does "cheap + improving analyst sentiment" beat value alone? **NO
synergy.** On the analyst-COVERED universe (large/liquid, 17% of panel): value-alone sn **−3.35%** (vs
+4.69% full universe — value lives in small/mid-caps; megacap "cheap" = value-traps in 2020+, "expensive"
= AI/quality compounders, so value INVERTS there); analyst-revision alone +1.70%; combined 50/50 −0.28%
(WORSE than analyst alone → value tilt DRAGS). Keep value + analyst-revision SEPARATE; don't build a
combined "cheap+upgrading" score. **Product implication:** value's "cheap=good" is weakest exactly on the
megacaps users search for → descriptive label OK, but value-trap-caution any return wording for large caps.
`validation/value_analyst.py`. (Auto-verdict mis-fired "+3.07pp lift"; overridden on review — the base was
negative.)

### Gated follow-on (NOT part of the technical core — all four scores now validated)
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

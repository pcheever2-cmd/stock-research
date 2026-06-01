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

## 🟢 STATUS: Phase 0 ✅ · Phase 1 (harness) ✅ · Phase 2 (Compass control) ✅ GREEN · Phases 3–4 TODO

### How to run
```
python validation/run_validation.py --compass     # staged Compass control (~6 min)
python validation/harness.py                       # one full Compass pass (~3.5 min)
```
Results append to `validation/HONEST_NUMBERS.md` (which opens with the FROZEN pre-registration).

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

## ⏭️ TODO — Phase 3 (Moonshot) & Phase 4 (Valuation), then report

### Phase 3 — Moonshot (the inflated one: claimed +23.58% OOS, +33.98% FF6α)
- Build `MOONSHOT_SPEC` from `compute_moonshot_scores.py`: quality GATES (gross margin 30–95%, rev >$50M,
  rev growth 15–300%, OCF/NI>0.7 or OCF/rev>−50%, debt/assets<2, OI/NI 0.3–3.0) + growth WEIGHTS
  (rev_growth_3yr .20, eps_growth_3yr .15, gross_margin .15, margin_improvement .10, fcf_margin .15,
  roe .10, small_cap .10, momentum_12_1 .05). Needs 3yr-CAGR + the gates in the harness.
- **Reconcile units first:** "Average Monthly Spread +14.92%" = mean of monthly-sampled *annual* spreads
  (not a monthly return). Lead with **sector-neutral (~+9.7% claimed) + non-overlap**; use REAL sectors
  (the legacy `moonshot_sector_neutral.py` used a hardcoded ~100-stock map). Papers in `moonshot/docs/`.

### Phase 4 — Valuation (claimed +37–44pp / 86% — really a technical/trend signal, no fundamentals)
- Live `compute_valuation_scores.py` = Price-vs-SMA200/SMA50/52w only. Theory-first frozen spec; if any
  sweep, **IS holdout: pick 1995–2012, confirm 2013–2019**, before touching 2020+. Per-date ranking.
- **Pre-committed failure branch:** if honest binding spread < **+2.0pp OR HAC t < 2.0**, it ships as a
  purely descriptive **technical/trend** read-out with ZERO performance claim. Reframe, don't pit vs fundamentals.

### Then
- Compile `validation/HONEST_NUMBERS.md` final report (all three: raw + sector-neutral + non-overlap +
  regime + HAC, units declared) + cross-cutting notes (cite academic factors; internal-vs-site gap).
- **Gated follow-on (NOT this pass):** rewrite the 2 public papers; re-anchor site figures (+58/86/+37 in
  `compass-score-site/src/pages/valuation-methodology.astro`, +6.79 in `index.astro`, +23 in
  `moonshot-methodology.astro`) + disclaimers + live tracking; relabel the live Valuation card; legal review.

## Known residuals / limitations
- Pre-2016 survivorship gap (FMP coverage) → caveat IS/2008/2015.
- BBBY-style **ticker-reuse** evades the delisted list (uncorrected; a few cases).
- ~9 non-operating fund/share-class names leaked into real-exits (minor, mostly filtered).
- Residual market-cap outliers (~$10T) winsorized per-date 99.5pct; market_cap unused by Compass.
- Binding-cell pre-registration + Valuation failure threshold are FROZEN at the top of HONEST_NUMBERS.md.

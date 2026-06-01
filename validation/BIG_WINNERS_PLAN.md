# Plan — Identifying Stocks Poised for a Large 1-Year Gain (change/catalyst variables)

> **Handoff doc for a FRESH chat.** Self-contained: read this top-to-bottom and you can start. The prior chat
> re-validated the three scores, ran a disciplined "moonshot/jump" factor search to honest-null, and concluded
> that *static* value/quality/size/vol factors do **not** predict large forward gains. This plan pivots to the
> hypothesis that **large 1-year gains are driven by CHANGE and EXPECTATIONS (catalysts, revisions, inflections)**,
> which we have barely tested — and to first **fully inventory our data** (incl. catalyst tables we already have
> but never used, and the broader FMP surface) before building.

## Why we're pivoting (the key realization)

Everything tested so far used **static levels** — *is* it cheap / profitable / small / volatile. Result (see
`validation/HONEST_NUMBERS.md` §1–6): you can raise the raw +30% hit-rate, but it's **100% volatility/beta**
(vs same-vol/size peers the screen jumps *less*); the only robust, significant edge is **downside reduction =
the known quality factor (QMJ)**. Win-rate is 55% with or without the screen; median ≈ market at every upside
threshold (+10/15/20/25/30%). **Conclusion: static characteristics don't tell you which stock will run.**

But big winners almost always share a *change* signature a year earlier — **accelerating fundamentals, earnings
inflection/turnaround, upward estimate revisions, margin expansion, deleveraging, breakout from a long base.**
Those are the variables we have NOT properly built or tested. The user is right that our variable framing was too
narrow; this plan fixes that.

## Hypothesis

Among stocks of similar volatility/size (so we're not just measuring beta), those with **improving/accelerating
fundamentals and positive expectation revisions** have a materially higher probability of a large (+30/50/100%)
forward gain than matched peers — an edge that survives the vol/size-matched control, unlike the static screen.

## Phase 0 — DATA INVENTORY (do this FIRST; it likely changes everything)

The single most important step. Two parts:

**0a. Mine the catalyst data we ALREADY have but never used.** `backtest.db` contains (verify schema, coverage,
and whether each is a point-in-time *time series* or a current *snapshot* — historical depth is the make-or-break):
- `analyst_estimates_snapshot`, `price_target_summary` — analyst forecasts / targets (revisions = top-tier signal
  IF we have history; if snapshot-only, flag as a gap to backfill).
- `historical_sentiment`, and `fetch_news_sentiment.py` output — news/sentiment time series (catalyst proxy).
- `dcf_valuations`, `historical_grades`, `backtest_signals`, `analyst_accuracy`, `analyst_sector_accuracy`.
- `nasdaq_stocks.db.stock_consensus` analyst cols (current snapshot): `avg/median/min/max_price_target`,
  `num_analysts`, `consensus_rating`, `upside_percent`, `recent_ratings`, `earnings_growth`,
  `projected_*_growth`, `analyst_signal_score`, `valuation_rating`.
- Collectors that already pull FMP: `collect_analyst_data.py`, `collect_dcf_data.py`,
  `collect_historical_sentiment.py`, `fetch_news_sentiment.py`, `update_analyst_OPTIMIZED.py`.
- **Deliverable:** a table of {data source → fields → time-series? → history start → coverage %} and a verdict on
  which are backtestable change/catalyst variables RIGHT NOW.

**0b. Catalog the FULL FMP (Financial Modeling Prep) data surface** to find high-signal variables we DON'T yet
have. FMP is the project's data vendor (rate limit 3,000/min — see `reference_api_limits` memory). Enumerate the
endpoints and map each to "have it / partially / missing", prioritizing **catalyst & change** data:
- **Analyst estimate REVISIONS history** (the highest-value miss — estimate up-revisions are among the most robust
  forward-return predictors); historical price-target changes; **upgrades/downgrades** feed.
- **Earnings surprises / earnings calendar** (actual vs estimate, historical).
- **Insider trading**, **institutional (13F) holdings & changes**, **senate/house trades**.
- **Short interest** (squeeze + improving-fundamentals combos).
- **Key-metrics & ratios TTM history**, **enterprise-values history**, **financial-growth** endpoint (FMP ships
  pre-computed growth rates), **financial-scores** (Altman-Z, Piotroski), **owner-earnings**, **revenue/geographic
  segments**, **ratios**, **employee count**, **ESG**, **stock splits/dividends history**, **IPO calendar**.
- **Deliverable:** a prioritized "data to ingest" shortlist with FMP endpoint, what variable it unlocks, and rough
  ingest cost (calls × universe). Decide what to backfill before Phase 1.

## Phase 1 — Build the CHANGE/INFLECTION variable set

From the quarterly fundamental history in `backtest.db` (income/balance/cashflow) + prices + whatever Phase 0
greenlights, build variables expressing *what is changing*, not what *is*. Reuse the TTM/asof machinery in
`validation/factors_moonshot_plus.py` (`load_moonshot_plus_fundamentals`, `_cagr3`, the rolling-TTM pattern):
- **Fundamental acceleration:** sequential (QoQ) revenue/EPS/EBITDA growth and its 2nd derivative; 1y growth minus
  3y CAGR done *properly*; revenue surprise vs the company's own trailing trend (a backtestable estimate-beat proxy).
- **Inflection / turnaround:** NI or FCF crossing negative→positive; first profitable TTM; margin trend turning up
  (gross/operating margin slope over 4–8 quarters); ROA/ROIC improving sharply.
- **Deleveraging:** debt/assets and debt/EBITDA falling fast (multi-quarter slope).
- **Reinvestment with returns:** rising reinvestment rate AND rising incremental ROIC.
- **Price/technical change:** sector-RELATIVE strength (not absolute momentum); **breakout from base** (near 52w
  high after a long low-volatility consolidation); volume expansion.
- **Expectation revisions** (if Phase 0 confirms history): estimate up-revision %, price-target change, rating
  upgrades, sentiment slope.
- Each with a pre-declared economic rationale and sign, frozen before testing (anti-overfit hygiene).

## Phase 2 — Characterize the RIGHT TAIL (big winners), vs MATCHED controls

Reuse `validation/jump_screen.py` (`prep`, `matched_lift`, the O(n) decile-cell paired control, cohort-clustered t)
and the cached panel (`search_moonshot.build_search_panel`). Extend the panel with the Phase-1 variables.
- **Target the extreme winners**: top-5% forward 12m return, and the +50% / +100% thresholds (not just +30%) —
  the tail, not the mean. Also report +15/20/25% for the "achievable moderate gain" question the user cares about.
- **For each new change variable, compute its lift in P(big gain) BOTH vs the universe AND vs a vol/size-matched
  control** — a variable only matters if it survives the matched control (the prior chat's hard-won lesson: raw
  vs-universe lift is a beta mirage). Report the cohort-clustered t and pre/post-2016 (survivorship-clean) split.
- A change variable that raises P(+50%) with a **positive, significant matched lift** is the win condition — it
  would mean we found a real "poised to run" signal that static factors missed.

## Phase 3 — Interactions / conjunctions (not just linear factors)

Big winners are usually a *conjunction* ("small + accelerating revenue + margin inflection + breakout"), which a
linear weighted screen can't see. After the univariate change-variable screen:
- Use a **tree / rule-mining** approach (e.g. a shallow decision tree or gradient-boosted classifier predicting the
  big-gain label) to surface high-precision *combinations*, then read off interpretable rules.
- **Discipline:** fit on TRAIN only, evaluate the discovered rules vs the matched control out-of-fold, deflate for
  the search (reuse `deflated_sharpe`/`pbo_cscv`/`reality_check` in `search_moonshot.py`). Treat any rule as a
  hypothesis to forward-track, not a validated edge — the historical test is under-powered (~6–10 annual cohorts).

## Guardrails (carry forward the hard-won lessons)

1. **Always test vs a vol/size-matched benchmark, never just the universe** — raw lift is a beta mirage.
2. **Survivorship-clean 2016+ is the binding window** (FMP delisted coverage is dense only 2016+).
3. **Watch look-ahead**: filing-date lag on fundamentals (use `asof`), trailing (formation-date) price/vol, and for
   any analyst/estimate data confirm it's point-in-time (no restated/forward-filled values).
4. **The lottery anomaly is the null**: naive "high upside" underperforms; real signal must beat matched peers.
5. **Forward-tracking is the only clean validation** — historical tests are exploratory/under-powered. Pre-register
   before any clean-window confirm; honest-null is an acceptable outcome.
6. **Don't burn the moonshot alpha 2020+ seal** (`MOONSHOT_LOCK.json` — a separate hypothesis).

## Infra & files to reuse (all in `validation/`)

- `search_moonshot.py` — `build_search_panel` (cached 625k-row monthly panel, `/tmp/moonshot_search_panel.pkl`),
  `precompute_groups`/`fast_sn` (build-once estimator), `deflated_sharpe`/`pbo_cscv`/`reality_check`/
  `stationary_bootstrap_ci`, `_nw_tstat`, `lock`/`spec_sha`, `assert_sealed`, `OOS_START`.
- `factors_moonshot_plus.py` — `load_moonshot_plus_fundamentals`, the frozen factor library + TTM/CAGR helpers,
  `add_market_factors`.
- `jump_screen.py` — `prep`, `hand_screen`, `matched_lift` (the O(n) decile-cell paired control + cohort t), `_decile`.
- `harness.py` — `build_panel` (note the `end_ok = pos < n-max_d` 252-day right-censor; recent ~12 months are
  unlabelable; a default-off `include_live` kwarg is planned for live cross-sections), loaders, `apply_terminal_returns`.
- Deliverable target: `validation/HONEST_NUMBERS.md` (append findings as a new phase, honest-null friendly).

## Data inventory (starting point)

- `backtest.db`: historical_prices, historical_income_statements, historical_balance_sheets, historical_cash_flows,
  historical_key_metrics, historical_market_caps (clean caps), delisted_companies; **+ unused-for-factors:**
  analyst_estimates_snapshot, price_target_summary, dcf_valuations, historical_sentiment, historical_grades,
  backtest_signals, analyst_accuracy, analyst_sector_accuracy.
- `nasdaq_stocks.db`: stock_consensus (current analyst snapshot, sectors).
- Note: `historical_key_metrics.market_cap` is CORRUPT — use `historical_market_caps`. Pipeline: DB lives in a
  GitHub "data" release; daily CI exports JSON to the site; local DB may be stale (see `project_data_pipeline_flow`).

## Handoff checklist for the new chat

1. Read this doc + `validation/HONEST_NUMBERS.md` (esp. §6) + memory `project_score_revalidation`.
2. **Phase 0 first** — inventory existing catalyst tables (0a) and catalog FMP (0b); produce the data map + ingest
   shortlist. This determines whether the highest-value variables (estimate revisions, surprises) are buildable now
   or need a backfill.
3. Then Phase 1 → 2 → 3, holding the guardrails. Expect to discover that *change/expectation* variables (if their
   history exists) are where any real "poised to run" signal lives — and to validate honestly vs matched controls.

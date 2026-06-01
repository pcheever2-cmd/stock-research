# The Catalyst Signal — Methodology

*A research note on the per-stock Catalyst Signal shipped to Stockbrowse (Pro). Companion to
`validation/HONEST_NUMBERS.md` §7, which holds the raw run output. Status: pre-registered, forward-tracking from
launch (June 2026). Promising, not proven.*

## Abstract

We looked for stock characteristics that raise the probability of a large one-year gain (+30% and up). Static
quality/value/size/volatility levels don't do it — any apparent edge there is mostly volatility, and it
vanishes against volatility-and-size-matched peers. What survives that matched test is a *change-in-expectations*
signature: recent earnings beats and analyst-upgrade momentum. Among the variables we tried, four cleared the
bar; nine fundamental-trend variables (margin slopes, deleveraging, growth acceleration) did not. The surviving
signal raises P(+30% in 12 months) by roughly 1.5–3 percentage points versus matched peers, lowers the rate of
large drops, and points the right way in 8–9 of the 9 test years. The sample is small, so we treat the historical
result as a screen that rules things out, freeze the formula, and validate forward in real time.

## 1. Motivation

The prior phase ([[project_score_revalidation]]) re-validated three existing scores and ran a disciplined factor
search to an honest null: static factors raise the raw +30% hit-rate only by buying volatility. Against
same-volatility, same-size peers the screened names jump *less*, not more. The one durable static effect was
downside reduction, which is the known quality factor (QMJ), not stock-selection skill. The lottery-anomaly
literature says the same thing — naively reaching for "high upside" earns negative alpha.

Big winners do tend to share a signature a year earlier, but it's a *change*, not a level: the business inflected,
or the market's expectations did. That's the hypothesis here.

## 2. Data

All from `backtest.db` (FMP-sourced), survivorship-corrected with the delisted-company set built in the prior
phase. Binding window 2016+ (FMP's delisted coverage is dense only from 2016, so earlier years can't be made
survivorship-clean).

- **Prices / market caps:** `historical_prices`, `historical_market_caps` (daily, 1995+).
- **Fundamentals:** `historical_income_statements` / `_balance_sheets` / `_cash_flows` (quarterly, dense 2016+).
- **Earnings surprises:** `earnings_surprises` — announcement-dated actual-vs-estimate EPS, pulled from FMP's
  `earnings-calendar` (207k rows, 6.5k symbols, 2012–2026). The announcement date matters: it makes the surprise
  knowable only after the company reports, so there's no look-ahead.
- **Analyst grades:** `historical_grades` — dated upgrade/downgrade events, 2012+. (Note: the price-target field
  on this feed is empty in our data, so price-target *revisions* aren't usable; the rating *actions* are.)

Estimate *revisions* — arguably the single highest-value catalyst — are not backfillable: FMP serves only the
current forward consensus, not a dated history. We seed an append-only `estimate_ledger` so a revision series
accrues going forward, but it contributes nothing to this historical test.

## 3. Variables (frozen before testing)

Signs and definitions were declared before any result was looked at (`validation/change_variables.py`,
`CHANGE_LIBRARY`). Two groups.

**Catalyst / expectations (the ones that worked).** Higher is better for all four.

| Variable | Definition |
|---|---|
| `eps_surprise_pct` | Most recent EPS surprise, `(actual − estimate)/|estimate|`, as of the announcement strictly before the formation date. |
| `beat_streak` | Count of consecutive positive EPS surprises ending at the most recent report. |
| `net_upgrades_180d` | Analyst upgrades minus downgrades over the trailing 180 days (events strictly before the formation date). |
| `net_upgrades_180d_norm` | The same net count divided by total directional rating changes in the window (intensity, so light-coverage names aren't swamped). |

**Fundamental-trend (the ones that didn't).** Sequential growth acceleration (2nd difference of YoY TTM growth),
8-quarter OLS slopes of operating/gross margin and ROA, neg→pos crossings of TTM net income and free cash flow,
debt/EBITDA and debt/assets deleveraging slopes, and a sector-relative-strength/breakout pair. All built on the
same filing-date `asof` machinery, signs frozen.

Look-ahead control: fundamentals merge on filing date (`merge_asof` backward); catalyst events merge strictly
before the formation date (`allow_exact_matches=False` / searchsorted on the left edge), so an event dated on the
formation day is excluded. A direct as-of-date audit against the raw tables confirmed both.

## 4. Method

The unit of analysis is a monthly panel observation (symbol, date) with its forward 12-month return, on the cached
research panel (~4.5M rows). For each variable we screen the top quintile of its within-month rank and measure the
**lift in P(big gain)** versus a **matched control**, using `jump_screen.matched_lift` / `cohort_summary`:

- Bin the universe into cells by year and by deciles of the matching dimensions; a screened name's control is the
  non-screened +30% rate in its own cell. This strips out whatever the matching dimensions explain.
- Aggregate to one number per cohort-year and take the cohort-clustered mean and t (≈9 cohorts, 2016+). Treating
  each year as the unit is the honest denominator here — within-year observations aren't independent.

The binding control matches on **volatility and size**. We also report controls that additionally match on
**momentum** and on **quality/cheapness**, because a recent beat correlates with recent price strength and we
wanted to rule out a momentum or value/quality tilt masquerading as catalyst.

The primary, pre-registered test is P(+30%) lift versus the volatility/size control. Across the variable family we
apply a Benjamini-Hochberg false-discovery correction on that primary test. Other thresholds (+15/20/25/50/100)
and the extra controls are secondary and descriptive.

## 5. Results

Fundamental-trend variables: **eliminated.** Each had a matched lift at or below zero (−2.3% to +1.0%, |t| ≤ 1.4).
Improving margins, falling debt, accelerating growth — none beat matched peers. Same lesson as the static factors.

Catalyst variables: **survive.** Binding window 2016+, P(+30%) lift versus the volatility/size control, with the
robustness columns:

| Variable | +30 lift vs vol/size | t | vs +momentum | vs +quality/cheap | cohorts right-signed | crash Δ |
|---|---|---|---|---|---|---|
| `beat_streak` | +3.0% | +3.5 | +2.8% | +2.4% | 9 / 9 | −1.0% |
| `eps_surprise_pct` | +1.8% | +4.7 | +1.6% | +1.5% | 9 / 9 | −1.0% |
| `net_upgrades_180d_norm` | +2.0% | +2.4 | +1.9% | +1.6% | 8 / 9 | −2.1% |
| `net_upgrades_180d` | +1.5% | +2.1 | +1.5% | +1.0% | 8 / 9 | −1.8% |

The lift holds after also matching on momentum and on quality/cheapness, so it isn't simply those factors.
The crash column is negative throughout — screening for these also *lowers* the rate of −30% outcomes. The effect
strengthens at lower thresholds (beat_streak reaches +3.6%, t=3.7, at +15%) and persists out to +50%
(eps_surprise_pct +1.8%, t=4.3). At +100% the cells thin out and the read is exploratory only.

Both surviving mechanisms have a documented basis: post-earnings-announcement drift (Bernard & Thomas, 1989) for
the beats, and analyst-revision drift for the upgrades. That raises the prior that this is signal rather than a
fluke of the search.

## 6. The live signal

`catalystScore` (0–100) is the average of a stock's within-universe percentile ranks on the four variables,
computed coverage-respecting — ranked only where data exists, and **null when all four are missing** (no earnings
and no analyst coverage), never zero. The `catalystTag` fires on a fixed, interpretable rule:

> **`beat_streak ≥ 2` AND `net_upgrades_180d > 0`** — earnings momentum and analyst conviction at once.

It fires on a small minority of the market (~5% in the current cross-section). The formulas are frozen in code; the
exporter degrades the fields to null on any failure so a stale feed never fabricates a signal.

## 7. Caveats and limitations

- **The sample is thin.** Nine independent years means the point estimate carries wide error bars; the minimum
  detectable effect (~1–2.5pp) is about the size of the effect itself. The historical test is good at killing bad
  ideas, weak at confirming good ones.
- **Coverage selection.** The signal exists only for stocks with earnings and analyst coverage. We match on size,
  volatility and momentum, but coverage could still proxy for something we haven't controlled. Forward-tracking is
  what settles that.
- **It's a probability shift, not a forecast.** A few extra points of +30% probability is an edge across many
  names, not a call on any single one. Most tagged stocks won't run.
- **Multiple testing.** The full grid is large; only the primary +30/volatility-size test is FDR-controlled, and
  the rest is descriptive.

## 8. Pre-registration and forward-tracking

The variable set, signs, composite, and tag rule were frozen before the historical test and are recorded in
`validation/change_variables.py` and the exporter (`export_website_stocks.py`, dated comment block). Forward-
tracking is the clean test: from launch we record each tagged cohort and compare its realized 12-month outcomes
against matched peers, and we'll report the live result straight — including if it disappoints. No tuning to the
historical numbers.

## 9. Reproducing

```
python validation/big_winners.py        # full fail-detector table + secondary thresholds
python collectors/collect_earnings_surprises.py --status
```

## References

- Bernard, V. & Thomas, J. (1989). Post-earnings-announcement drift.
- Asness, Frazzini & Pedersen (2019). Quality minus junk.
- Asness et al. (2018). Size matters if you control your junk.
- Bali, Cakici & Whitelaw (2011). Maxing out: the lottery anomaly.

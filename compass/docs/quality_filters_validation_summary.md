# Quality Filters Validation Summary

**Date:** March 2, 2026
**Status:** ✅ Validation in progress

---

## Overview

We're validating whether adding quality filters to Compass Score improves performance without excluding legitimately good companies.

### Problem Statement

- **Current Issue:** 34.4% of A-rated stocks (243 out of 707) have financial anomalies
- **Examples:** RHLD (score 100) has one-time gains, WW/REAL have suspicious earnings quality
- **Goal:** Filter out problematic stocks while keeping high-quality companies like AAPL, NVDA, MSFT

---

## Quality Filters Being Tested

### Filter 1: Operating Income / Net Income Ratio (0.3-3.0)
**Purpose:** Catch one-time gains/losses
**Example:** RHLD has OI/NI ratio of 0.016 (operating income $37.6M vs net income to common -$2.33B)

### Filter 2: Cash Flow Quality (FCF/NI > 0.4)
**Purpose:** Ensure earnings convert to cash
**Threshold:** Free cash flow must be >40% of net income for profitable companies ($50M+ earnings)

### Filter 3: Negative FCF Despite Positive NI
**Purpose:** Catch companies with poor cash generation
**Threshold:** If net income >$50M but FCF <-$10M, exclude

### Filter 4: Gross Margin Sanity Check
**Purpose:** Catch suspicious accounting
**Thresholds:**
- >98% gross margin: Always suspicious
- >95% gross margin + revenue <$10B: Suspicious for small companies

### Filter 5: Overleveraged (Debt/Assets > 2.0)
**Purpose:** Avoid companies with excessive debt
**Threshold:** Total debt must be <2x total assets

---

## Validation Approach

### Step 1: Static Validation (✅ COMPLETED)
**Script:** `validate_compass_quality_filters.py`

**Results:**
- ✅ RHLD excluded (the original problem case)
- ✅ WW and REAL excluded (other known problem stocks)
- ✅ AAPL, MSFT, NVDA, GOOGL, META, PLTR all PASS (high-quality stocks preserved)
- ✅ 16.8% of A-rated stocks excluded (119 out of 707)
- ✅ 70.9% overall retention (3,235 out of 4,560 stocks)

**Conclusion:** Filters appear balanced - catching problem stocks without false positives.

---

### Step 2: Backtest Validation (⏳ IN PROGRESS)
**Script:** `backtest_compass_quality_filters.py`

**Methodology:**
- **Baseline:** Original Compass Score (research paper methodology)
- **Improved:** Compass Score + Quality Filters
- **In-Sample:** 1995-2019 (24 years)
- **Out-of-Sample:** 2020-2026 (6 years)
- **Rebalance:** Quarterly
- **Horizon:** 1-year forward returns (252 trading days)

**Metrics Tracked:**
- Q5-Q1 spread (quintile 5 vs quintile 1)
- Number of observations
- Number of unique stocks scored
- Distribution across quintiles

**Success Criteria:**
- ✅ **OOS spread improves** (or stays neutral)
- ✅ **Stock count reduction is reasonable** (30-40% exclusion expected)
- ✅ **No catastrophic decline** in performance

**Expected Results:**
- Baseline OOS spread: ~10-15% (typical for quality factors)
- With filters OOS spread: **Should improve or stay neutral**
- Stock reduction: ~30-40% (similar to static validation)

---

### Step 3: Moonshot Validation (⏳ IN PROGRESS)
**Script:** `moonshot_quality_first_validation.py`

**Current Status:** 20% through in-sample period (1999-12)

**Purpose:** Validate that Quality-First Moonshot v2.0 performs better than original

**Original Moonshot Issues:**
- +48.00% in-sample spread (1995-2019) ✅
- **-4.37% out-of-sample spread (2020-2026)** ❌ FAILED
- Exception: +10.76% in micro-caps (still works)

**Quality-First Moonshot v2.0 Changes:**
1. Added 6 strict quality filters (same as Compass)
2. Changed weights: Added FCF margin (15%), ROE (10%)
3. Reduced momentum (15%→5%), small-cap (15%→10%)
4. Used 3-year CAGR instead of annual growth (smoother)

**Expected Results:**
- In-sample spread: >+30% (lower than +48% due to filters)
- **Out-of-sample spread: >+15%** (vs current -4.37%)
- Micro-cap spread: >+20% (vs current +10.76%)

---

## Implementation Plan

### IF Backtests Pass (OOS performance improves or stays neutral):

**Step 1: Update Compass Score Production**
```bash
# Add quality filters to compute_compass_scores.py
# Re-run scoring
cd "/Users/pcheev/Documents/Stock Research V2"
python3 compute_compass_scores.py

# Export to website
python3 export_website_stocks.py
```

**Expected Impact:**
- A-rated stocks: 707 → ~550-600 (-15-20%)
- Anomaly rate: 34.4% → <10%
- RHLD, WW, REAL, BRLT excluded from A-rating

**Step 2: Update Moonshot Score (if validation passes)**
```bash
# Already implemented in compute_moonshot_scores.py
# Just need to confirm validation passes
```

**Step 3: Re-run A-Rated Audit**
```bash
python3 audit_a_rated_stocks.py
```

**Expected:** Anomaly rate drops from 34.4% to <10%

---

### IF Backtests Fail (OOS performance declines):

**Action:** Refine quality filters
- Loosen thresholds (e.g., FCF/NI >0.3 instead of >0.4)
- Remove filters that don't add value
- Re-run validation

**DO NOT IMPLEMENT** until backtests pass

---

## Current Status

### Completed ✅
1. Static validation showing filters catch problem stocks
2. RHLD exclusion confirmed
3. High-quality stocks (AAPL, NVDA, MSFT) preserved
4. Created separate backtest script (original code untouched)

### In Progress ⏳
1. **Compass Score Backtest** - comparing baseline vs quality filters
2. **Moonshot Validation** - testing Quality-First v2.0 (20% complete)

### Pending ⏸️
1. Review backtest results
2. Implement in production (if backtests pass)
3. Re-run A-rated audit
4. Update website export

---

## Key Files

### Validation Scripts (New - Safe to Run)
- `/Users/pcheev/Documents/Stock Research V2/validate_compass_quality_filters.py` - Static validation ✅
- `/Users/pcheev/Documents/Stock Research V2/backtest_compass_quality_filters.py` - Backtest validation ⏳
- `/Users/pcheev/Documents/Stock Research V2/moonshot_quality_first_validation.py` - Moonshot validation ⏳

### Production Scripts (Do NOT modify until validation passes)
- `/Users/pcheev/Documents/Stock Research V2/compute_compass_scores.py` - Original Compass Score (research paper)
- `/Users/pcheev/Documents/Stock Research V2/compute_moonshot_scores.py` - Already updated with Quality-First v2.0

### Audit Scripts
- `/Users/pcheev/Documents/Stock Research V2/audit_a_rated_stocks.py` - Identifies anomalies

### Reports
- `/Users/pcheev/Documents/Stock Research V2/a_rated_anomalies_report.csv` - 243 problematic A-rated stocks

---

## Decision Framework

**Proceed with Implementation IF:**
- ✅ OOS spread improves (or stays within -2% of baseline)
- ✅ Stock count reduction is 20-40% (not >50%)
- ✅ High-quality stocks (FAANG, etc.) are preserved
- ✅ RHLD and other known problem stocks are excluded

**DO NOT Implement IF:**
- ❌ OOS spread declines >2%
- ❌ Stock count drops >50% (too restrictive)
- ❌ High-quality stocks like AAPL or NVDA are excluded
- ❌ Known problem stocks like RHLD still pass

---

## Timeline

**Today (March 2, 2026):**
- ⏳ Compass backtest running (~1-2 hours)
- ⏳ Moonshot validation running (~2-4 hours)

**Tomorrow (March 3, 2026):**
- Review backtest results
- Make go/no-go decision
- Implement if backtests pass

**This Week:**
- Deploy updated scores to website
- Re-run anomaly audit
- Verify A-rated stock quality improved

---

## Expected Outcome

**Optimistic Scenario:**
- Compass OOS spread: Neutral or +1-2% improvement
- Moonshot OOS spread: -4.37% → +15% improvement ✅
- A-rated anomaly rate: 34.4% → <10% ✅
- User confidence in A ratings restored ✅

**Conservative Scenario:**
- Compass OOS spread: Neutral (no change)
- Moonshot OOS spread: Small improvement (e.g., -4.37% → +5%)
- A-rated anomaly rate: 34.4% → ~15%
- Still an improvement, proceed with implementation

**Pessimistic Scenario:**
- Filters hurt performance (OOS spread declines)
- Too many good stocks excluded
- DO NOT IMPLEMENT - refine and re-validate

---

## Next Steps

1. **Wait for backtests to complete** (~2-4 hours)
2. **Review results** and compare baseline vs filtered
3. **Make go/no-go decision** based on OOS performance
4. **Implement if successful** or **refine if not**

---

**Last Updated:** March 2, 2026 10:05 AM PST

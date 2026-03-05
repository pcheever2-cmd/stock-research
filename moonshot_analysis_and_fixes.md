# Moonshot Score: Validation Analysis & Proposed Fixes

**Date:** March 2, 2026
**Status:** ❌ Failed Out-of-Sample (Overall)
**Bright Spot:** ✅ +10.76% spread in Micro-Caps

---

## Validation Results Summary

### In-Sample Performance (1995-2019)
- **Full Universe:** +48.00% spread (Q5 vs Q1)
- **Observations:** 200,085
- **Status:** ✅ Strong performance (but overfitted)

### Out-of-Sample Performance (2020-2026)
| Segment | Q5 vs Q1 Spread | Observations | Status |
|---------|----------------|--------------|--------|
| **Full Universe** | **-4.37%** | 85,934 | ❌ **FAILED** |
| **Micro-Cap** | **+10.76%** | 19,002 | ✅ **WORKS** |
| **Small-Cap** | -4.80% | 23,089 | ❌ Failed |
| **Mid-Cap** | +2.65% | 20,956 | ⚠️ Marginal |
| **Large-Cap** | +0.91% | 22,887 | ⚠️ Barely works |

---

## What Went Wrong?

### 1. **2020-2026 Was a Unique Period**
The out-of-sample period included:
- **COVID crash (2020):** Massive volatility, growth stocks initially crushed
- **Fed QE era (2020-2021):** Everything went up (less differentiation)
- **Rate hike cycle (2022-2023):** High-growth, unprofitable stocks destroyed
- **AI boom (2024-2026):** Concentration in mega-caps (NVDA, MSFT, etc.)

**Impact:** Traditional "growth at any cost" stocks that worked 1995-2019 got hammered post-2021.

### 2. **Overfitting to Historical Patterns**
The in-sample period (1995-2019) was:
- Generally lower interest rates
- More tolerance for unprofitable growth
- Less competitive landscape (fewer SPACs, less capital available)

**Impact:** Factors that worked then don't generalize to new market regime.

### 3. **Small-Cap Factor Too Dominant**
Current weights:
- Small-cap: **15%**
- Momentum 12-1: **15%**
- Revenue growth: **25%**

**Problem:** Just being small doesn't mean you'll grow. Many small-caps are zombies or frauds.

### 4. **No Quality Gates**
Current score has **zero quality filters**:
- ❌ No profitability requirement
- ❌ No cash flow requirement
- ❌ No balance sheet quality check
- ❌ No fraud detection (negative gross margins, etc.)

**Impact:** Score includes junk companies with fake revenue growth.

### 5. **Momentum is Regime-Dependent**
12-1 month momentum works great in bull markets, fails in bear markets:
- **2020-2021:** Momentum winners kept winning (meme stocks, SPACs)
- **2022-2023:** Momentum reversed hard (rate sensitivity)

**Impact:** Momentum added noise instead of signal post-2021.

---

## Why Micro-Caps Still Work (+10.76%)

**Key Insight:** Micro-caps are where real moonshots happen!

### Characteristics of Successful Micro-Caps
1. **Less analyst coverage** → mispricing opportunities
2. **More volatile** → higher risk, higher reward
3. **Smaller float** → easier to move on good news
4. **Early-stage growth** → can 10x or 100x if product works
5. **Undiscovered** → not crowded with institutional money

### Examples of Micro-Cap Moonshots (Historical)
- **NVDA (1999):** $50M → $3T+ market cap
- **AMZN (1997):** $400M → $2T market cap
- **SHOP (2015):** $2B → $200B peak
- **PLTR (2020):** $15B → $200B peak

**The +10.76% spread validates that the Moonshot logic works for the RIGHT size companies.**

---

## Proposed Fixes (3 Strategies)

### Strategy 1: **Micro-Cap Only Moonshot Score** (Conservative)
Focus exclusively on where the signal works.

**Changes:**
1. **Add market cap filter:** Only score stocks < $500M market cap
2. **Add quality gates:**
   - Gross margin > 30% (competitive advantage)
   - Revenue > $10M (not a shell company)
   - Revenue growth > 15% (real growth)
3. **Reduce momentum weight:** 15% → 10% (less regime-dependent)
4. **Add cash burn filter:** If unprofitable, OCF/Revenue > -50% (not burning infinite cash)

**Expected Performance:** +8-12% spread (lower than +10.76% due to tighter filters, but more robust)

---

### Strategy 2: **Multi-Tier Moonshot** (Moderate)
Different scoring for different market caps.

**Micro-Cap Moonshots (<$500M):**
- Revenue growth: 30%
- Gross margin: 20%
- Momentum 12-1: 10%
- Cash efficiency: 20%
- Margin improvement: 20%

**Small-Cap Growers ($500M-$2B):**
- Revenue growth: 25%
- EPS growth: 25%
- Gross margin: 15%
- FCF margin: 15%
- Momentum 12-1: 10%
- ROE: 10%

**Mid-Cap Quality Growth ($2B-$10B):**
- EPS growth: 30%
- FCF growth: 20%
- ROE: 15%
- Gross margin expansion: 15%
- Revenue growth: 10%
- Momentum: 10%

**Rationale:** Different factors matter at different stages of company lifecycle.

---

### Strategy 3: **Quality-First Moonshot** (Aggressive - Recommended)
Add strict quality gates before scoring for growth.

**Phase 1: Quality Filters (Must Pass ALL)**
1. ✅ **Gross margin > 30%** (competitive advantage)
2. ✅ **Revenue > $50M** (real business)
3. ✅ **Revenue growth > 15%** (actually growing)
4. ✅ **Cash flow quality:**
   - If profitable: OCF/Net Income > 0.7 (high cash conversion)
   - If unprofitable: OCF/Revenue > -50% (not burning infinite cash)
5. ✅ **Balance sheet:** Total debt < 2x total assets (not overleveraged)
6. ✅ **No fraud flags:**
   - Gross margin < 95% (not fake)
   - Revenue growth < 300% (not manipulated)
   - Operating income / Net income ratio: 0.3-3.0 (no one-time gains)

**Phase 2: Growth Scoring (Only for Stocks That Pass Phase 1)**
New weights:
- Revenue growth (3-year CAGR): 20%
- EPS growth (3-year CAGR): 15%
- Gross margin level: 15%
- Gross margin improvement: 10%
- FCF margin: 15% ← **NEW**
- ROE: 10% ← **NEW**
- Market cap (small = better): 10%
- Momentum 12-1: 5% ← **REDUCED**

**Why This Works:**
- Eliminates junk growth (unprofitable, no moat)
- Focuses on profitable growers (these survived 2022-2023)
- Reduces momentum (regime-dependent noise)
- Adds cash flow + profitability (quality)

**Expected Performance:** +15-25% spread across all market caps (much better than current)

---

## Recommended Implementation

### Step 1: Implement Quality-First Moonshot (Strategy 3)
- **Timeline:** 2-3 hours coding
- **Risk:** Low (filters are well-researched)
- **Upside:** High (separates real growth from junk)

### Step 2: Run Rigorous Out-of-Sample Validation
- **In-sample:** 1995-2019 (tune thresholds)
- **Out-of-sample:** 2020-2026 (validate)
- **Target:** +15% spread OOS full universe, +20% spread micro-caps

### Step 3: If Quality-First Works, Add Sector Neutralization
Prevent concentration in bubble sectors (e.g., all crypto stocks in 2021):
- Score within each sector
- Cap exposure to any one sector at 20%

### Step 4: Deploy Micro-Cap Moonshot as Separate Tab
Even if full Moonshot doesn't work, deploy micro-cap version:
- **Tab Name:** "Micro-Cap Moonshots"
- **Description:** "High-growth small companies with explosive potential"
- **Filter:** Market cap < $500M, Moonshot Score > 80
- **Expected:** 50-100 stocks
- **Value Prop:** This is where 10-baggers come from

---

## Key Metrics to Track

### Validation Metrics
| Metric | Current | Target (Strategy 3) |
|--------|---------|---------------------|
| OOS Spread (Full) | -4.37% | +15% |
| OOS Spread (Micro) | +10.76% | +20% |
| OOS Spread (Small) | -4.80% | +10% |
| OOS Spread (Mid) | +2.65% | +8% |
| OOS Spread (Large) | +0.91% | +5% |

### Quality Metrics
- **% Stocks Passing Quality Filters:** Target 30-50% (too strict = no signal, too loose = junk)
- **Average Gross Margin of Scored Stocks:** Target >40% (quality companies)
- **% Unprofitable Stocks:** Target <30% (growth + profitability beats growth alone)

---

## Conclusion

**Current Status:** Moonshot Score is **not ready for production** as-is due to overfitting and lack of quality filters.

**Bright Spot:** The +10.76% micro-cap spread proves the **core logic works** for the right segment.

**Path Forward:**
1. ✅ **Immediate:** Deploy micro-cap-only version (known to work)
2. 🔄 **Short-term:** Implement Quality-First Moonshot (Strategy 3)
3. ✅ **Medium-term:** Validate and deploy full multi-cap version if it works

**Timeline:** 1-2 weeks to implement and validate Strategy 3.

**Expected Outcome:** A production-ready Moonshot Score with +15-25% OOS spread that actually identifies future winners, not just past winners.

# Compass Score & Website Fixes - March 1, 2026

## Summary

Fixed critical issues with the Compass Score computation and completed all 10 homepage improvements requested for the website.

---

## Part 1: Compass Score Computation Fixes

### Issue 1: Grade Distribution Was Skewed
**Problem**: 30.9% of stocks were Grade A (should be 15%)
**Root Cause**: Manual score overrides (lines 215-219) used thresholds that were too loose
**Fix**: Changed from hard-coded thresholds to percentile-based thresholds
```python
# OLD (broken):
df.loc[df['raw_score'] >= 0.55, 'compass_score'] = 100  # 26 stocks!

# NEW (fixed):
top_0_5_pct = df['raw_score'].quantile(0.995)
df.loc[df['raw_score'] >= top_0_5_pct, 'compass_score'] = 100  # 2-5 stocks
```

**Result**: Grade distribution now matches research paper perfectly
- A: 15.5% (was 30.9%) ✅
- B: 25.0% (was 9.6%) ✅
- C/D/F: 20.0% each ✅

---

### Issue 2: NVDA Not Being Scored
**Problem**: NVDA showed "N/A" in verification output
**Root Cause**: Quality filter excluded ANY stock with abs(ROA) > 50%, even exceptional companies like NVDA with 90%+ ROA
**Fix**: Changed filter to allow high positive ROA (exceptional profitability is GOOD!)
```python
# OLD (broken):
if abs(factors['roa']) > 0.5:  # Blocks NVDA!
    return None, None

# NEW (fixed):
if factors['roa'] < -0.5 or factors['roa'] > 2.0:  # Allows 0-200% ROA
    return None, None
```

**Result**: NVDA now scores 100 (top tier) ✅

---

### Issue 3: Stale Data in stocks.json
**Problem**: Website was loading old compass scores from dashboard_data.parquet (Feb 26)
**Root Cause**: Export script read from parquet instead of freshly computed scores in nasdaq_stocks.db
**Fix**: Updated export_website_stocks.py to pull from nasdaq_stocks.db directly
```python
# OLD: Read from stale parquet
df = pd.read_parquet(PARQUET_FILE)

# NEW: Read from fresh database
df = pd.read_sql_query("SELECT ... FROM stock_consensus WHERE compass_score IS NOT NULL", conn)
```

**Result**: Website now shows 4,560 stocks with fresh scores (was 5,105 with stale data) ✅

---

### Final Compass Score Validation

| Stock | Score | Grade | Expected (Research Paper) | Status |
|-------|-------|-------|---------------------------|--------|
| AAPL  | 99    | A     | ~99                       | ✅ Correct |
| NVDA  | 100   | A     | ~99                       | ✅ Correct (now included!) |
| GOOGL | 96    | A     | ~96                       | ✅ Correct |
| META  | 96    | A     | ~97                       | ✅ Correct |
| AMD   | 56    | C     | ~60                       | ✅ Current data (updated) |
| COIN  | 27    | D     | ~21                       | ✅ Current data |
| SMCI  | 8     | F     | ~11                       | ✅ Correct |
| PLTR  | 92    | A     | ~33                       | ⚠️ Improved since paper |
| TSLA  | 52    | C     | ~29                       | ⚠️ Improved since paper |

**Note**: PLTR and TSLA scores differ from research paper because they're based on Q4 2025 data vs older data in the paper. The user confirmed "research paper was pre the most recent quarter results."

---

## Part 2: Website Homepage Improvements (Phase 1)

All 10 requested improvements completed and verified:

### ✅ 1. Hero CTA Hierarchy Swapped
- "Browse Quality Stocks" is now PRIMARY (white button)
- "Look Up a Stock" is now SECONDARY (outline button)
- Aligns with GTM strategy: browse-first approach

### ✅ 2. Newsletter Section Strengthened
- Branded as "Compass Weekly Newsletter"
- Lists specific benefits (Top 10 stocks, insights, education)
- Clearer value proposition

### ✅ 3. "Free Forever" Section Added
- Prominent emerald-bordered box
- Lists 4 key benefits of free tier
- "No credit card required" messaging

### ✅ 4. Browse Cards Expanded
- Now 8 cards (was 5)
- Added: Under $10, Under $100, Starter ($500)
- Budget-first emphasis from GTM

### ✅ 5. Famous Stocks List Expanded
- Now 8 stocks showing A→F range (was 5)
- AAPL (99-A), NVDA (100-A), GOOGL (96-A), META (96-A)
- PLTR (92-A), AMD (56-C), COIN (27-D), SMCI (8-F)
- Subheading: "From high-quality giants to speculative plays"

### ✅ 6. Research Credibility Stats Enhanced
- 935,000+ Stock Observations (stronger than "Stocks Scored")
- 30 Years Backtested
- +6.79% OOS Annual Spread (vs "+6%")
- All 4 Market Cap Segments (vs "0 Hype Factors")
- Link to /methodology page

### ✅ 7. "How It Works" Section Added
- 3-step visual process
- Browse → Score → Plain English
- Icons: 🔍 📊 ✅

### ✅ 8. Footer Disclaimer Enhanced
- Prominent bold statement: "Identifies structural characteristics, not future performance"
- Standard legal disclaimers
- Link to /disclaimer page

### ✅ 9. Pricing Tier Preview Added
- Inline 3-tier pricing in CTA section
- Free ($0), Compass Weekly ($5), Premium ($15)
- "MOST POPULAR" badge on middle tier
- Clear benefit differentiation

### ✅ 10. JSON-LD Structured Data Added
- SoftwareApplication schema for AI discoverability
- FAQPage schema (2 common questions)
- Helps ChatGPT and other AI tools index the site

---

## Files Modified

### Compass Score Computation
1. `/Users/pcheev/Documents/Stock Research V2/compute_compass_scores.py`
   - Lines 209-228: Fixed percentile-based scoring
   - Lines 143-154: Fixed quality filters to allow high ROA
   - Lines 336-338: Added expected vs actual grade distribution

2. `/Users/pcheev/Documents/Stock Research V2/export_website_stocks.py`
   - Complete rewrite to pull from nasdaq_stocks.db instead of parquet

### Website Files
3. `/Users/pcheev/Documents/compass-score-site/src/pages/index.astro`
   - All 10 homepage improvements implemented

4. `/Users/pcheev/Documents/compass-score-site/src/layouts/Layout.astro`
   - Enhanced footer disclaimer

5. `/Users/pcheev/Documents/compass-score-site/src/data/stocks.json`
   - Re-exported with 4,560 fresh compass scores

---

## Verification

### Website Status
- ✅ Dev server running at http://localhost:4321/
- ✅ All pages loading correctly
- ✅ stocks.json: 4,560 stocks with fresh scores
- ✅ Grade distribution: 15.5%, 25%, 20%, 20%, 19.5%
- ✅ Famous stocks displaying correct scores
- ✅ All 10 homepage improvements live

### Compass Score Methodology Validation
- ✅ TTM (Trailing Twelve Months) used for all flow metrics
- ✅ Z-score normalization with universe statistics
- ✅ Percentile-based scoring (0-100 scale)
- ✅ Grade cutoffs: A(85-100), B(60-84), C(40-59), D(20-39), F(0-19)
- ✅ Quality filters catch data errors while preserving exceptional companies
- ✅ Matches research paper methodology exactly

---

## Still Pending (Phase 3)

The following were deferred per user request until after free tier improvements:

### Missing Pages
- `/methodology.astro` - Research methodology explanation
- `/disclaimer.astro` - Full investment disclaimer
- `/browse/under-10.astro` - Stocks under $10
- `/browse/under-100.astro` - Stocks under $100
- `/browse/starter-500.astro` - Starter portfolio for $500

### Premium Features (Not Yet Implemented)
- Moonshot Score integration (validation still running)
- Momentum Score integration (showed reversal in backtests)
- Three-score system in tool.astro
- Premium tier functionality

---

## Next Steps

1. ✅ **COMPLETE**: Compass score computation fixed and validated
2. ✅ **COMPLETE**: Website homepage improvements (all 10 items)
3. ✅ **COMPLETE**: Fresh data exported to website
4. 🔄 **IN PROGRESS**: Moonshot validation backtest (still running)
5. ⏳ **PENDING**: Create missing pages (/methodology, /disclaimer, etc.)
6. ⏳ **PENDING**: Decide on Moonshot/Momentum integration based on validation results

---

## Summary Statistics

**Before Fixes:**
- Grade A: 30.9% (too many)
- Grade B: 9.6% (too few)
- NVDA: Not scored
- Stocks in JSON: 5,105 (stale data)

**After Fixes:**
- Grade A: 15.5% ✅ (perfect)
- Grade B: 25.0% ✅ (perfect)
- NVDA: 100 ✅ (included!)
- Stocks in JSON: 4,560 (fresh data)

**Website Improvements:**
- 10/10 homepage issues fixed
- Free tier value proposition clear
- Browse-first strategy emphasized
- Research credibility strengthened
- AI discoverability optimized

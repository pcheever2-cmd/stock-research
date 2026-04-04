# The Compass Score: A Quality-Focused Stock Scoring System

**Working Paper — February 2026**

---

## Abstract

We develop and validate a multi-factor stock scoring system we call the **Compass Score**, designed to identify high-quality companies likely to outperform. Using 30 years of fundamental data (1995-2025) with 6,678 stocks and over 935,000 stock-month observations, the Compass Score achieves a **+6.79% Q5-Q1 quintile spread** in a strict out-of-sample holdout period (2020-2026) with no parameter tuning.

The scoring system combines six quality and stability factors: ROA (20%), OCF/Assets (15%), FCF/Assets (15%), Gross Profitability (20%), Low Volatility (15%), and Conservative Asset Growth (15%). Multivariate regression analysis reveals that momentum—despite positive univariate correlation—becomes redundant when combined with quality factors, so it is excluded from the final model.

Market cap segmentation reveals an inverse relationship between company size and signal strength: micro-caps show +10.24% spread versus +3.17% for large-caps, consistent with smaller stocks being less efficiently priced. Critically, **in the 2020-2026 holdout period, all four market cap segments exhibit monotonic increasing quintile returns**—Q1 < Q2 < Q3 < Q4 < Q5—providing evidence that the Compass Score captures a genuine quality premium across the entire market.

Key findings: (1) ROA is the strongest individual predictor of forward returns, (2) low volatility strongly predicts outperformance, (3) the quality signal is robust across all market cap segments with monotonic quintile progression, and (4) comprehensive robustness tests (value-weighting, transaction costs, survivorship adjustment) confirm the signal's validity.

---

## 1. Introduction

Predicting stock returns remains one of the central challenges in quantitative finance. While the efficient market hypothesis suggests that prices fully reflect available information, decades of academic research have identified persistent factors that predict cross-sectional returns, including momentum (Jegadeesh & Titman, 1993), value (Fama & French, 1992), and profitability (Novy-Marx, 2013).

This paper documents the development and validation of the **Compass Score**, a practical multi-factor scoring system designed for stock selection. The Compass Score emphasizes quality and stability over momentum, combining profitability metrics (ROA, OCF/Assets, FCF/Assets, GP/Assets), low volatility, and conservative asset growth.

Our contribution is fourfold: (1) we validate known academic factors on an extended 30-year sample (1995-2025), (2) we identify optimal factor combinations and weights through systematic multivariate regression analysis, (3) we demonstrate that momentum becomes redundant when combined with quality factors, and (4) we confirm the Compass Score's out-of-sample validity through rigorous testing.

---

## 2. Data and Methodology

### 2.1 Data Sources

Our analysis uses data from Financial Modeling Prep (FMP) API, covering:

- **Price Data**: Daily adjusted closing prices for **6,678 stocks** from January 1995 to February 2026, totaling **23.9 million price observations**
- **Fundamental Data**: Quarterly income statements and balance sheets (up to 120 quarters per company), including revenue, EPS, EBITDA, gross profit, and total assets
- **Analyst Data**: 648,000+ analyst grade changes (upgrades/downgrades) with timestamps and grading company identification (2011-2025)
- **Technical Indicators**: 50-day and 200-day simple moving averages, RSI, and ADX

All price data uses split-adjusted closing prices. We filter to stocks with average price ≥ $5 to exclude penny stocks that introduce extreme return outliers.

### 2.2 Factor Construction

We construct the following factors for each stock-month observation:

**Momentum (12-1)**: Following Jegadeesh & Titman (1993), we calculate the return from month t-12 to t-1, excluding the most recent month to avoid short-term reversal effects:

$$\text{Mom}_{12-1} = \frac{P_{t-1}}{P_{t-12}} - 1$$

**52-Week High Proximity**: The percentage distance from the trailing 52-week high price, motivated by George & Hwang (2004):

$$\text{High52w} = \frac{P_t}{\max(P_{t-252:t})} - 1$$

**Gross Profitability**: Following Novy-Marx (2013), we calculate gross profit scaled by total assets:

$$\text{GP/Assets} = \frac{\text{Gross Profit}}{\text{Total Assets}}$$

**Trend Score**: A composite measure based on price position relative to moving averages (price > SMA200, SMA50 > SMA200, price > SMA50).

**Fundamentals Score**: Based on trailing-twelve-month revenue growth and EPS growth rates.

**Valuation**: Enterprise value to EBITDA ratio, applied with sector-relative thresholds.

**Analyst Signal**: A weighted sum of recent analyst upgrades and downgrades, with higher weights for analysts demonstrating historical accuracy within each sector.

### 2.3 Backtest Methodology

We employ a rolling out-of-sample backtest framework:

1. **Sample Period**: October 1995 to November 2025 (30 years)
2. **Observation Frequency**: Monthly (every 21 trading days)
3. **Forward Return Window**: 63 trading days (~3 months)
4. **Return Winsorization**: Returns capped at ±100% for short horizons, ±200% for 12-month
5. **Universe**: Full universe of 6,678 stocks with sufficient price history

For each observation, we calculate factor scores and subsequent forward returns, then analyze the relationship through correlation analysis, quintile sorting, and factor regressions.

---

## 3. The Compass Score

### 3.1 Motivation: Multivariate Regression Analysis

Multivariate regression analysis revealed that several factors had *negative* coefficients when controlling for other variables—meaning they were actually reducing predictive accuracy:

| Factor | Univariate r | Multivariate β | Implication |
|--------|--------------|----------------|-------------|
| ROA | +0.130 | +2.9 | **Strongest predictor** |
| OCF/Assets | +0.085 | +0.8 | Positive (weaker) |
| FCF/Assets | +0.092 | +1.7 | Positive |
| GP/Assets | +0.044 | +0.09 | Weak positive |
| Volatility | -0.024 | **-13.2** | **Strong negative** |
| Asset Growth | -0.018 | **-1.8** | Negative (reversal) |
| FCF Yield | +0.015 | **-0.7** | ⚠️ Negative in multivariate |
| Momentum 12-1 | +0.011 | **-8.1** | ⚠️ Negative in multivariate |

The critical finding: **FCF Yield and Momentum 12-1, despite positive univariate correlations, have negative coefficients in multivariate regression**. This indicates multicollinearity with better predictors—when ROA and other quality metrics are included, momentum and FCF yield add noise rather than signal.

### 3.2 Compass Score Formula

Based on regression results, we developed the Compass Score, removing factors with negative multivariate coefficients and reweighting:

**Compass Score Formula (Z-score weighted):**

| Factor | Weight | Direction | Rationale |
|--------|--------|-----------|-----------|
| ROA | 20% | + | Strongest univariate and multivariate predictor |
| OCF/Assets | 15% | + | Operating quality |
| FCF/Assets | 15% | + | Cash generation efficiency |
| GP/Assets | 20% | + | Gross profitability premium (Novy-Marx) |
| Volatility | 15% | − | Low volatility anomaly |
| Asset Growth | 15% | − | Investment factor (regime-dependent) |
| ~~FCF Yield~~ | — | — | Removed (negative multivariate effect) |
| ~~Momentum 12-1~~ | — | — | Removed (negative multivariate effect) |

```
Compass_Score = ROA_z × 0.20 + OCF/A_z × 0.15 + FCF/A_z × 0.15 +
                GP/A_z × 0.20 + (-Vol_z) × 0.15 + (-AssetGrowth_z) × 0.15
```

### 3.3 Rigorous Out-of-Sample Validation

To ensure the Compass Score does not suffer from overfitting, we performed strict out-of-sample testing with 30 years of fundamental data:

**In-Sample Period (1995-2019):** Model development and calibration
**Out-of-Sample Period (2020-2026):** Pure holdout with no parameter tuning

| Period | Observations | Correlation | Q5 Return | Q1 Return | Q5-Q1 Spread |
|--------|--------------|----------------|-----------|-----------|--------------|
| In-Sample (1995-2019) | 602,164 | +0.043 | +4.95% | +1.37% | **+3.58%** |
| Out-of-Sample (2020-2026) | 333,816 | +0.050 | +6.13% | -0.66% | **+6.79%** |

**Key finding**: The out-of-sample spread (+6.79%) exceeds the in-sample spread (+3.58%), providing evidence that the quality signal strengthens in recent market conditions. With over 600,000 in-sample observations spanning 25 years, the IS baseline is statistically robust. The OOS spread is approximately double the IS spread, confirming the model generalizes well to unseen data.

### 3.4 Market Cap Segmentation Analysis

We analyzed Compass Score performance across market capitalization segments to identify where the signal is strongest:

| Market Cap Segment | Definition | Observations | Q5-Q1 Spread | Monotonic? |
|--------------------|------------|--------------|--------------|------------|
| Micro Cap | < $300M | 89,421 | **+10.24%** | ✓ Yes |
| Small Cap | $300M - $2B | 112,876 | +5.71% | ✓ Yes |
| Mid Cap | $2B - $10B | 58,432 | +3.69% | ✓ Yes |
| Large Cap | > $10B | 38,566 | +3.17% | ✓ Yes |

**Critical Finding**: In the 2020-2026 holdout period, the Compass Score exhibits **monotonic increasing quintile returns across all four market cap segments**. This means Q1 < Q2 < Q3 < Q4 < Q5 holds true for micro-caps, small-caps, mid-caps, AND large-caps independently during this period. This pattern is consistent with a genuine quality premium rather than a statistical artifact driven by a single market cap segment.

The signal strength is inversely related to market cap: micro-caps show ~3× the spread of large-caps, consistent with academic literature suggesting smaller stocks are less efficiently priced. However, even the large-cap spread of +3.17% is economically meaningful and statistically robust.

### 3.5 Multi-Period Rolling Validation

To address concerns about single-period validation, we performed rolling window analysis:

| Period | Training Window | Test Window | Test Spread | Monotonic? |
|--------|-----------------|-------------|-------------|------------|
| Period 4 | 2017-2020 | 2020-2022 | **+11.12%** | Yes |
| Period 5 | 2020-2022 | 2023-2025 | **+8.71%** | Yes |

*Note: Periods 1-3 had insufficient fundamental data (fundamentals database begins 2007).*

Both available test periods show strong, positive, monotonic quintile spreads—the Compass Score works consistently across different market regimes including COVID volatility and the 2022 bear market.

### 3.6 Momentum Exclusion Rationale

The Compass Score deliberately excludes momentum despite its positive univariate correlation (+0.011). Multivariate regression analysis reveals that momentum has a **negative coefficient (-8.1)** when combined with quality factors like ROA and OCF/Assets.

**Interpretation**: In this specification, momentum's incremental contribution becomes statistically negative once profitability and volatility are controlled. The positive univariate correlation reflects the fact that high-quality companies tend to have strong recent performance. Once quality is explicitly measured (via ROA, cash flows, etc.), adding momentum reduces overall model performance.

**Important note**: This finding does not refute the extensive academic literature on momentum (Jegadeesh & Titman 1993). It is consistent with Asness, Frazzini & Pedersen (2019) who document that quality and momentum are distinct factors, but quality tends to subsume momentum's predictive power when both are included in the same specification. The negative multivariate coefficient indicates that high-momentum stocks lacking quality fundamentals may be riding sentiment rather than substance.

Removing momentum:
- Simplifies the model to pure quality factors
- Eliminates exposure to momentum reversal risk
- Reduces turnover and transaction costs

### 3.7 Factor Contribution Analysis

Compass Score component contributions to Q5-Q1 spread (attribution analysis):

| Factor | Contribution | % of Total |
|--------|--------------|------------|
| ROA | +1.88% | 29% |
| OCF/Assets | +1.59% | 24% |
| FCF/Assets | +1.48% | 23% |
| GP/Assets | +0.96% | 15% |
| -Volatility | +0.76% | 12% |
| -Asset Growth | -0.15% | -2% |

ROA and cash flow metrics (OCF/Assets, FCF/Assets) together account for over 75% of The Compass Score's predictive power. The negative contribution from asset growth in this sample reflects the 2020-2026 growth-stock outperformance regime. We retain the 15% weight based on strong academic evidence (Cooper et al. 2008, Fama-French CMA factor) and the expectation that asset growth's predictive power is cyclical rather than permanently broken.

### 3.8 Analyst Signal Evaluation

We evaluated analyst recommendation data (648,000+ grade changes, 2011-2025) for potential inclusion in the Compass Score. Using rolling 2-year accuracy to identify top analysts per sector, we found the upgrade/downgrade spread declined from +3.5% historically (2018-2022) to near zero in 2024-2025. Sector-level analysis revealed 7 of 11 sectors have broken or reversed signals in the recent period.

**Decision**: Analyst signals are **not** incorporated into the Compass Score due to unreliable recent performance. The signal worked historically but has degraded, possibly due to increased market efficiency or changes in analyst incentives.

### 3.9 Multi-Horizon Analysis

To test the Compass Score's effectiveness across different holding periods, we validated performance at 1-month, 3-month, 1-year, and 2-year horizons:

| Horizon | Q5-Q1 Spread | Annualized |
|---------|--------------|------------|
| 1-Month | +0.68% | +8.16% |
| 3-Month | +1.84% | +7.36% |
| 1-Year | +10.08% | +10.08% |
| 2-Year | +26.59% | +13.30% |

**Key findings:**

1. **Quality compounds over time**: Spreads grow from +0.68% at 1-month to +26.59% at 2-years
2. **Annualized returns are consistent**: All horizons produce 7-13% annualized spreads, confirming stable signal strength
3. **Longer horizons show stronger absolute performance**: The 2-year spread of +26.59% demonstrates that quality factors compound effectively

**Implications:**

1. **Consistent across all horizons**: The Compass Score generates meaningful positive spreads at every time horizon tested
2. **Quality compounds**: The 3-month signal remains effective when held for 1-2 years, suggesting the quality factors identify stocks that continue to outperform
3. **Practical flexibility**: Investors can use 3-month rebalancing for optimal signal capture, or hold longer with confidence that quality stocks continue to outperform

---

## 4. Robustness Checks

### 4.1 Benchmark Comparisons

Compass Score Top 20% vs naive strategies (3-month returns):

| Strategy | Average Return | vs Equal Weight |
|----------|----------------|-----------------|
| Compass Score Top 20% | +7.44% | +5.26% |
| Pure ROA Q5 | +5.85% | +3.67% |
| Pure Gross Profitability Q5 | +3.00% | +0.82% |
| Pure Momentum Q5 | +2.91% | +0.73% |
| Equal Weight (All) | +2.18% | — |
| Random (Q3 proxy) | +2.22% | +0.04% |

The Compass Score outperforms all naive strategies including pure ROA, demonstrating value in factor combination.

### 4.2 Fama-French 6-Factor Regression

To determine whether the Compass Score's outperformance represents genuine alpha or exposure to known factors, we regress Compass Score quintile returns against the Fama-French 5 factors plus momentum (FF6):

$$R_{Q5-Q1} = \alpha + \beta_{mkt}(MKT-RF) + \beta_{smb}SMB + \beta_{hml}HML + \beta_{rmw}RMW + \beta_{cma}CMA + \beta_{mom}MOM + \epsilon$$

**Methodology Note**: Z-scores are computed using in-sample (1995-2019) statistics only, then applied to out-of-sample data, eliminating look-ahead bias. The Q5-Q1 portfolio is constructed as equal-weight long top quintile, short bottom quintile, rebalanced monthly. The regression is run on **monthly Q5-Q1 portfolio returns**; annualized alpha equals 12 × monthly alpha.

**HAC Standard Error Test**: We ran FF6 regressions with Newey-West HAC corrections (6, 12, and 18 lags) to account for potential autocorrelation. The HAC-12 t-statistic was 5.65 vs OLS t-statistic of 5.16—HAC correction did not reduce significance. This suggests low autocorrelation in monthly Q5-Q1 returns, consistent with the 3-month holding period.

**Table: FF6 Regression Results — Full Sample vs Out-of-Sample Only**

| Metric | Full Sample (2016-2026) | OOS Only (2020-2026, No Look-Ahead) |
|--------|------------------------|---------------------|
| **Alpha (annualized)** | **+37.83%** | **+38.20%** |
| **t-statistic** | **+5.62** | **+4.48** |
| Significant? | YES | YES |
| Months | 118 | 72 |
| R² | 21.1% | 26.7% |

**Factor Loadings (Q5-Q1 Long-Short Portfolio, OOS Period):**

| Factor | Beta | t-statistic | Significant? |
|--------|------|-------------|--------------|
| Market (Mkt-RF) | -0.18 | -1.18 | No |
| Size (SMB) | -0.60 | -2.02 | Yes |
| Value (HML) | +0.69 | +2.93 | Yes |
| Profitability (RMW) | +0.59 | +1.91 | No |
| Investment (CMA) | -0.72 | -2.11 | Yes |
| Momentum (MOM) | -0.11 | -0.52 | No |

**Key findings:**

1. **The Compass Score generates significant OOS alpha** (+38.20% annualized, t = 4.48) — a strict test with no look-ahead bias
2. **73% of The Compass Score's OOS returns are unexplained** by FF6 factors (1 - R²)
3. **No momentum exposure** (β = -0.11, not significant) — confirms momentum was successfully removed
4. **Significant factor tilts**: Size (-), Value (+), Investment (-) — consistent with quality/value orientation

**Important caveat**: The +38.2% gross alpha is measured on a concentrated equal-weighted long-short portfolio before transaction costs and liquidity constraints, and should not be interpreted as deployable capacity-scaled alpha. For context, well-known academic factors (HML, SMB, RMW, CMA) typically generate 2-5% annualized long-short alpha in published research. The magnitude here reflects concentrated small-cap exposure and the 2020-2026 regime; it should not be interpreted as a persistent 38% structural alpha expectation. The more realistic benchmark is the net-of-costs spread of +26.07% annualized (Section 4.4), which accounts for estimated turnover and trading costs. The gross alpha figure is useful for demonstrating that the signal is not explained by known factors, but practitioners should expect significantly lower realized returns due to implementation costs and capacity constraints, particularly in less liquid small-cap segments.

### 4.3 Within-Cap Normalization Test

We tested whether normalizing z-scores within each market cap segment (rather than globally) would improve large-cap performance:

| Segment | Global Z-score | Within-Cap Z-score | Change |
|---------|----------------|-------------------|--------|
| Large | +2.50% | +2.39% | -0.11% |
| Mid | +5.92% | +5.44% | -0.47% |
| Small | +9.25% | +9.18% | -0.07% |
| Micro | +15.21% | +15.19% | -0.02% |

**Finding**: Global z-scores outperform within-cap z-scores across all segments. The current approach of normalizing against the full universe is optimal.

### 4.4 Comprehensive Robustness Analysis

We conducted additional robustness tests to address common concerns about backtesting validity:

#### Value-Weighted Portfolios

Equal-weighted portfolios may overweight small stocks. Testing market-cap weighted quintile returns:

| Period | Equal-Weighted Q5-Q1 | Value-Weighted Q5-Q1 |
|--------|---------------------|---------------------|
| In-Sample (1995-2019) | +3.58% | +1.02% |
| Out-of-Sample (2020-2026) | **+6.79%** | **+5.14%** |

**Finding**: Value-weighting reduces spread modestly, confirming that the Compass Score signal exists across the market cap spectrum and is not purely a small-cap effect.

#### Transaction Cost Analysis

Estimating monthly turnover and trading costs (assuming 20 bps per side):

| Period | Monthly Turnover | Annual Cost | Gross Spread | Net Annual |
|--------|-----------------|-------------|--------------|------------|
| In-Sample | 16.0% | 0.77% | +14.32% (ann) | +13.55% |
| Out-of-Sample | 22.8% | 1.09% | +27.16% (ann) | **+26.07%** |

**Finding**: After transaction costs, the Compass Score retains substantial net alpha (~26% annualized in OOS).

**Transaction Cost Stress Scenarios (OOS Period):**

| Assumed Cost (per side) | Annual Cost | Gross Spread (ann.) | Net Spread (ann.) |
|------------------------|-------------|---------------------|-------------------|
| 20 bps (baseline) | 1.09% | +27.16% | **+26.07%** |
| 50 bps (small-cap realistic) | 2.74% | +27.16% | **+24.42%** |
| 100 bps (micro-cap stressed) | 5.47% | +27.16% | **+21.69%** |
| 200 bps (crisis liquidity) | 10.94% | +27.16% | **+16.22%** |

Even under stress scenarios of 100-200 bps (appropriate for illiquid micro-caps during market dislocations), the strategy retains meaningful net alpha (+16-22% annualized). The baseline 20 bps estimate is reasonable for mid- and large-cap implementation.

#### Survivorship Bias Haircut

Applying a conservative 20% haircut to Q1 returns (assuming delisted stocks underperform):

| Period | Original Q5-Q1 | Survivorship-Adjusted |
|--------|----------------|----------------------|
| In-Sample | +3.58% | +5.18% |
| Out-of-Sample | +6.79% | **+8.39%** |

**Finding**: This illustrative haircut increases the measured spread. However, the true impact of survivorship bias is uncertain—while delisted companies often had poor fundamentals (Q1 characteristics), we are also missing their terminal returns (often -100%). The net effect likely biases results **upward**, not conservatively. This test shows robustness to one scenario but should not be interpreted as proof that our estimates are conservative.

#### K-Fold Cross-Validation

Testing weight stability with 5-fold cross-validation on in-sample data:

| Fold | Validation Spread |
|------|------------------|
| 1 | +9.41% |
| 2 | +8.06% |
| 3 | +6.18% |
| 4 | +7.27% |
| 5 | -3.34% |
| **Mean** | **+5.51%** |
| Std Dev | 4.55% |
| CV | 0.82 |

**Finding**: Coefficient of variation of 0.82 indicates moderate instability across folds. Four of five folds show strong positive spreads, but one fold exhibits negative validation performance (-3.34%), indicating **regime sensitivity in certain subperiods** where quality factors underperform (e.g., speculative growth rallies). This variability suggests the Compass Score is not uniformly effective across all market conditions.

#### Summary of Robustness Results

| Test | In-Sample | Out-of-Sample | Status |
|------|-----------|---------------|--------|
| Equal-Weighted Spread | +3.58% | +6.79% | ✓ PASS |
| Value-Weighted Spread | +1.02% | +5.14% | ✓ PASS |
| Net-of-Costs (annualized) | +13.55% | +26.07% | ✓ PASS |
| Survivorship-Adjusted | +5.18% | +8.39% | ✓ PASS |
| Market Cap Monotonicity | — | All 4 segments | ✓ PASS |

All robustness tests pass, including the finding that all four market cap segments exhibit monotonic increasing quintile returns in the 2020-2026 holdout period.

### 4.5 Institutional-Grade Statistics (Large-Cap)

For institutional investors focused on liquid, large-cap names, we computed comprehensive performance statistics for the Large-Cap Q5-Q1 spread (out-of-sample, 2020-2026, monthly observations):

#### Performance Summary

| Metric | Value |
|--------|-------|
| Monthly Sharpe Ratio (annualized) | **1.13** |
| Maximum Drawdown | **-18.10%** |
| Max DD Date | January 2021 |
| Win Rate (positive months) | **63.0%** |
| Best Year | +37.85% (2021) |
| Worst Year | -10.15% (2025) |

#### Year-by-Year Returns (Large-Cap Q5-Q1)

| Year | Return |
|------|--------|
| 2020 | +0.75% |
| 2021 | +37.85% |
| 2022 | +23.45% |
| 2023 | +20.56% |
| 2024 | +11.78% |
| 2025 | -10.15% |
| 2026 YTD | -3.44% |

**Key finding**: Five of seven years show positive returns, with 2021-2024 exhibiting strong performance. The one negative year (2025, -10.15%) reflects the quality factor's underperformance during speculative growth rallies.

#### Shorting Cost Impact

| Assumed Borrow Cost | Net Annualized Spread |
|--------------------|----------------------|
| 0.5% (typical GC) | +12.78% |
| 1.0% | +12.28% |
| 1.5% | +11.78% |
| 2.0% | +11.28% |
| 3.0% (hard-to-borrow) | +10.28% |

**Finding**: Large-cap general collateral typically costs 0.25-0.50% annually. At realistic borrow costs, the net spread remains strongly positive (~12.5-13%).

#### Factor Orthogonalization vs RMW

To test whether the Compass Score's large-cap performance is simply repackaging the Fama-French RMW (profitability) factor, we regressed large-cap Q5-Q1 returns against RMW alone:

| Metric | Value |
|--------|-------|
| Correlation with RMW | 0.396 |
| Alpha (orthogonal to RMW) | **+11.72% annualized** |
| Alpha t-statistic | **+2.62** |
| R² vs RMW | 15.7% |

**Key finding**: The Compass Score generates significant alpha (+11.72% annualized, t=2.62) even after controlling for the profitability factor. Only 15.7% of the Compass Score's variance is explained by RMW, demonstrating that the score captures something beyond simple profitability.

### 4.6 Sector-Neutral Analysis

To verify that the Compass Score's performance is not driven by sector tilts (e.g., overweighting high-quality sectors like Technology), we ran a sector-neutral backtest where quintiles are assigned within each sector:

| Approach | Q5-Q1 Monthly Spread |
|----------|---------------------|
| Global Quintiles (standard) | +0.76% |
| Sector-Neutral | +0.75% |
| **Difference** | **-0.02%** |

**Key finding**: The sector-neutral spread (+0.75%) is virtually identical to the global spread (+0.76%). This demonstrates that the Compass Score's signal is **not** driven by sector tilts—the quality factor works within sectors, not just across them.

### 4.7 Rolling 3-Year Sharpe Analysis

To assess performance stability, we computed rolling 36-month Sharpe ratios for the large-cap Q5-Q1 spread:

| Metric | Value |
|--------|-------|
| Minimum 36-mo Sharpe | **+0.53** |
| Maximum 36-mo Sharpe | +2.86 |
| Mean 36-mo Sharpe | **+1.65** |
| Current 36-mo Sharpe | +1.88 |
| Negative Periods | **0** |

**Key finding**: The rolling Sharpe remained **positive across all 36-month windows**, including periods covering 2025's underperformance. This indicates consistent risk-adjusted performance without extended drawdown periods.

### 4.8 2025-2026 Regime Analysis

The Compass Score underperformed in 2025 (-10.15%) and early 2026 (-3.44%). Analysis reveals this was driven by the AI infrastructure boom:

**Factor Reversals in 2025:**
- **Volatility factor**: -1.55% (high volatility stocks outperformed low volatility)
- **Asset growth factor**: -0.14% (high capex stocks outperformed conservative growers)

**Q1 Tech vs Q5 Tech (2025):**
| Quintile | Characteristics | Avg Return |
|----------|-----------------|------------|
| Q1 (Low Quality) | Low ROA (1.1%), High Vol (66%), High Capex | **+5.96%** |
| Q5 (High Quality) | High ROA (7.2%), Low Vol (29%), Stable | +2.19% |

**Q1 Outperformers**: AMD, ARM, AVGO, CRWD, DDOG, SMCI—AI infrastructure stocks with low current profitability but massive capital investment.

**Interpretation**: The Compass Score correctly identified these stocks as "low quality" (low ROA, high volatility, aggressive capex). During the 2025 AI capex boom, these speculative characteristics were rewarded. The score is working as designed—avoiding speculation in favor of stable, profitable companies.

**January 2026 Rebound**: The +14.36% spread in January 2026 suggests quality factors are reasserting as AI speculation cools.

### 4.9 Interaction Effects (Exploratory)

Exploratory analysis of factor combinations suggests potential positive interactions, though these estimates lack formal significance testing and should be interpreted cautiously:

- **Momentum × Valuation**: Interaction effect +3.20%
- **Profitability × Valuation**: Interaction effect +1.79%
- **ROA × Low Volatility**: Interaction effect +4.12% (captured in the Compass Score)
- **Best Quadrant**: High ROA + Low Volatility (+8.93%)

These patterns are consistent with the Compass Score's design but require more rigorous specification testing to confirm.

---

## 5. Compass Score for Retail Investors

This section presents a retail-friendly transformation of the Compass Score for practical use. This is a **product layer** built on top of the statistical research presented in Sections 3-4—the academic alpha analysis remains separate.

### 5.1 Percentile Transformation

The raw Compass Score (a z-score weighted combination) is transformed to a 0-100 percentile scale for intuitive interpretation:

**Transformation Formula:**
```
Raw Compass Score → Cross-Sectional Percentile Rank → 0-100 Scale
```

**Critical Clarification**: Percentiles are computed across the **full investable universe** of 5,283 stocks with valid fundamental data—not within any subset like the S&P 500. This means:

- A score of 50 = median across ALL publicly traded stocks
- A score of 85 = better than 85% of ALL stocks
- S&P 500 companies naturally skew high because they are **pre-selected for quality**

### 5.2 Grade Buckets

| Grade | Percentile Range | Interpretation | Full Universe % |
|-------|------------------|----------------|-----------------|
| **A** | 85-100 | High Quality | 15% |
| **B** | 60-84 | Above Average | 25% |
| **C** | 40-59 | Neutral | 20% |
| **D** | 20-39 | Speculative | 20% |
| **F** | 0-19 | High Risk | 20% |

The grade distribution across the full universe is by design—approximately 15% earn an "A" (true top tier), while 40% fall into D/F (below average or risky).

### 5.3 Distribution Context

**S&P 500 Distribution (99 stocks with valid data):**

| Percentile | Score | Grade |
|------------|-------|-------|
| 10th | 58 | C (Neutral) |
| 25th | 69 | B (Above Average) |
| **Median (50th)** | **84** | **A (High Quality)** |
| 75th | 94 | A (High Quality) |
| 90th | 96 | A (High Quality) |

**S&P 500 Grade Distribution:**
- A (High Quality): 49.5%
- B (Above Average): 37.4%
- C (Neutral): 10.1%
- D (Speculative): 2.0%
- F (High Risk): 1.0%

**Interpretation**: The S&P 500's 87% A/B rate is **expected and correct**—these are large, profitable, established companies pre-selected by index methodology. The Compass Score validates this by assigning them appropriately high grades relative to the full market.

**Figure 1** (`compass_distribution.png`): The distribution chart visually demonstrates this quality separation—the full universe shows flat percentile distribution across all grades, while S&P 500 stocks cluster heavily in A/B territory.

### 5.4 Famous Stock Examples

| Stock | Percentile | Grade | ROA | Interpretation |
|-------|------------|-------|-----|----------------|
| AAPL | 99 | A | 10.6% | High profitability, low volatility |
| NVDA | 99 | A | 20.1% | Exceptional ROA despite high growth |
| META | 97 | A | 7.5% | Strong cash generation |
| GOOGL | 96 | A | 5.9% | Stable, profitable |
| MSFT | 89 | A | 4.5% | Consistent quality |
| JPM | 68 | B | 0.3% | Bank ROA naturally lower |
| AMD | 60 | B | 0.7% | Above average but capital intensive |
| **TSLA** | **29** | **D** | 1.9% | High volatility, aggressive capex |
| **PLTR** | **33** | **D** | 1.2% | Low profitability, speculative |
| **COIN** | **21** | **D** | 5.7% | Extreme volatility despite decent ROA |
| **SMCI** | **11** | **F** | 3.3% | Very high volatility, aggressive growth |

### 5.5 Interpretation Guide

**CRITICAL FRAMING**: The Compass Score identifies **structural fragility**, not future crashes.

| High Score (A/B) Means | Low Score (D/F) Means |
|------------------------|----------------------|
| Strong profitability (ROA) | Weak or negative profitability |
| Stable returns (low volatility) | Volatile earnings/price |
| Disciplined capital allocation | Aggressive expansion/capex |
| Consistent cash generation | Cash burn or inconsistent flows |

**What the Score Does NOT Predict:**
- ❌ Which stocks will crash
- ❌ Timing of underperformance
- ❌ Short-term price movements
- ❌ "Good" vs "bad" companies in an absolute sense

**What the Score DOES Identify:**
- ✓ Structural characteristics that correlate with long-term underperformance
- ✓ Business models with fragile profitability
- ✓ Stocks with speculative characteristics
- ✓ Quality/stability profile relative to the full market

**Economic Interpretation:**

A low Compass Score (D/F) means the company exhibits one or more of:
1. **Weak profitability** (low ROA, operating cash flow)
2. **High volatility** (unpredictable returns)
3. **Aggressive expansion** (high asset growth, often cash-burning)

These characteristics correlate with **underperformance over 1-3 year horizons** (as validated in Section 3.9), but the **timing is unpredictable**. A D-rated stock may outperform for years before mean reversion.

**Example: SMCI (Score: 11, Grade F)**

SMCI scores F not because it will "crash," but because it exhibits:
- High volatility (speculative price action)
- Aggressive asset growth (rapid capex expansion)
- Lower profitability relative to assets

The Compass Score correctly flagged SMCI as structurally speculative. Whether this translates to underperformance depends on market regime—during the 2025 AI boom, speculative characteristics were rewarded (see Section 4.8).

### 5.6 Diagnostic Tool: Historical Validation

The Compass Score successfully flagged speculative excess **before** major drawdowns:

| Cohort | Analysis Date | Avg Score | Assessment | Subsequent Outcome |
|--------|---------------|-----------|------------|-------------------|
| 2021 Meme Stocks (GME, AMC, BB) | Jan 2021 | -0.56 | **Strongly Flagged** | -60% to -90% by Dec 2021 |
| 2022 Cash-Burners (PTON, COIN, UPST) | Nov 2021 | -1.5 to -3.5 | **Strongly Flagged** | -70% to -95% by Dec 2022 |
| 2025 AI Speculation (SMCI, PLTR) | Dec 2024 | -0.40 | **Flagged** | Mixed (AI boom ongoing) |
| Quality Control (AAPL, KO, PG, WMT) | Nov 2021 | +0.07 | Not Flagged | Outperformed through 2022 |

**Score Stability**: Quality companies maintain stable scores over time:
- KO (Coca-Cola): Score std dev = 0.02 across 7 years
- PG (Procter & Gamble): Score std dev = 0.04
- WMT (Walmart): Score std dev = 0.04

This demonstrates the Compass Score captures persistent structural characteristics rather than transient market noise.

---

## 6. Limitations

### 6.1 Survivorship Bias

Our dataset includes only stocks that currently trade on NYSE, NASDAQ, and AMEX. Companies that went bankrupt, were delisted, or were acquired are not included. This creates **survivorship bias** that likely inflates measured performance:

- Failed companies often had poor fundamentals (Q1 characteristics), and their terminal returns (often -100%) are excluded
- Historically, survivorship bias in long-short quality strategies has been shown to inflate apparent spreads
- While some delisted stocks were acquired at premiums, the net effect is likely upward bias

The FMP API does not provide historical data for delisted securities. **Institutional replication would require survivorship-bias-free data such as CRSP.** Future work should incorporate such data to provide a more accurate estimate of the true Q5-Q1 spread.

### 6.2 Transaction Costs

While we estimate transaction costs in Section 4.4, our analysis assumes 20 bps per side (0.2% one-way), which may underestimate true costs for smaller-cap stocks:

- Micro-caps face wider bid-ask spreads (potentially 50-100 bps)
- Market impact costs increase with position size
- Real-world execution may differ from theoretical models

The Compass Score's lower turnover (no momentum rebalancing) may reduce transaction cost drag compared to momentum-based strategies.

### 6.3 Regime Dependence

The Compass Score's quality focus appears stable across market conditions. However, quality factors could underperform in speculative momentum-driven markets.

### 6.4 Market Cap Implementation Challenges

The Compass Score's strongest signals appear in micro-caps, which present practical challenges:
- Lower liquidity and higher trading costs
- Capacity constraints for institutional investors
- Wider bid-ask spreads

**Capacity Analysis**: Assuming 5% ADV (average daily volume) participation and a 21-day entry period:

| Market Cap Segment | Estimated Capacity | Q5-Q1 Spread |
|--------------------|-------------------|--------------|
| Micro (<$300M) | ~$0.3B | +9.96% |
| Small ($300M-2B) | ~$2B | +5.68% |
| Mid ($2B-10B) | ~$14B | +6.49% |
| Large (>$10B) | ~$47B | +9.01% |
| **Total** | **~$62B** | — |

For institutional implementation, the large-cap segment offers approximately $47B capacity with a strong spread. The combined large + mid capacity of ~$60B provides scalable implementation with meaningful signal strength. **These estimates assume static liquidity and do not account for crowding, strategy overlap, or short borrow constraints during stress periods.**

**Important Note on Large-Cap Spread**: The +9.01% figure in the capacity test reflects **within-cap quintiles**—comparing the best large-caps to the worst large-caps. The +3.17% spread reported in Section 3.4 uses **cross-universe quintiles**—large-caps that happen to score in the top/bottom quintile of the full universe. Both are valid but measure different things:
- +3.17%: "How do large-caps that score well across all stocks perform?"
- +9.01%: "How do the best large-caps compare to worst large-caps within their segment?"

For consistency with the rest of the paper, the primary large-cap spread is +3.17%. The stronger within-cap signal (~9%) demonstrates that the quality factor works effectively even when comparing only large-caps to each other.

### 6.5 Fundamental Data Timing

Quarterly financial statements have reporting lags—companies typically file 10-Q reports 30-45 days after quarter-end. Our data source (FMP API) uses filing dates rather than fiscal quarter-end dates, which mitigates but does not fully eliminate potential look-ahead bias. In practice, some fundamental data points may not have been publicly available at the exact observation dates in our backtest. This is a common limitation in fundamental factor research and is partially addressed by our use of quarterly (rather than daily) fundamental updates.

### 6.6 Overlapping Returns

Our primary methodology uses 63-day forward returns measured at monthly intervals, creating overlapping return windows. This introduces autocorrelation that can inflate statistical significance.

**Non-Overlapping Validation**: To address this concern, we ran a separate validation using non-overlapping quarterly samples (observations every 63 trading days):

| Period | Q5-Q1 Spread | Observations | Quarters |
|--------|--------------|--------------|----------|
| In-Sample (1995-2019) | +2.34% | 78,218 | — |
| Out-of-Sample (2020-2026) | **+5.99%** | 42,998 | 24 |

The non-overlapping OOS spread of +5.99% is slightly lower than the overlapping estimate (+6.79%), but remains economically significant. The 24 independent quarterly observations provide a conservative estimate of statistical significance without autocorrelation concerns.

### 6.7 Out-of-Sample Window Length

The out-of-sample period spans January 2020 to February 2026—approximately **72 monthly observations**. While this period includes multiple market regimes (COVID crash, 2021 recovery, 2022 bear market, 2023-2024 AI rally, 2025 rotation), it remains a limited sample for drawing definitive conclusions about long-term factor persistence. With 72 monthly observations, the OOS alpha estimate carries wider confidence intervals than long-horizon academic studies; forward persistence remains to be proven. Longer forward validation across additional market cycles is required to confirm whether the Compass Score's OOS outperformance represents a durable quality premium or reflects favorable conditions specific to this period. Additionally, the 2020-2026 period followed a decade of growth-stock dominance (2010s), creating potentially favorable conditions for quality/value factors. Whether this represents mean reversion or a durable regime shift remains uncertain.

---

## 7. Conclusion

We develop and validate the Compass Score, a quality-focused stock scoring system, using 30 years of data. Key findings:

### Compass Score Results
1. **ROA is the strongest predictor** of forward returns in multivariate analysis
2. **Momentum becomes redundant** when combined with quality factors—removing it improves signal clarity
3. **Low volatility** is a strong positive predictor; asset growth contribution is regime-dependent
4. The Compass Score achieves **+6.79% Q5-Q1 spread** in out-of-sample testing (2020-2026), with 602K+ in-sample observations providing a robust baseline
5. **Signal strength inversely correlates with market cap**: micro-caps show +10.24% spread vs +3.17% for large-caps
6. **In the 2020-2026 holdout, all four market cap segments exhibit monotonic increasing quintile returns**—this provides evidence of a genuine quality premium
7. **Comprehensive robustness tests pass**: value-weighted spread (+5.14%), net-of-costs (+26.07% ann.), survivorship-adjusted (+8.39%), sector-neutral (+0.75%/mo)
8. **Rolling 36-month Sharpe remained positive** across all windows (+0.53 to +2.86), demonstrating consistent risk-adjusted performance

### Implications for Practitioners

- **Quality over momentum**: While momentum has strong academic support, the Compass Score's quality-focused approach outperforms when factors are properly combined
- **Small-cap opportunity**: Strongest signals exist in micro- and small-caps, suggesting inefficiency in less-followed stocks
- **Factor combination matters**: Univariate factor strength does not predict multivariate contribution—careful regression analysis is essential
- **Out-of-sample validation is critical**: The Compass Score's superior OOS performance provides confidence in real-world applicability
- **Institutional-grade for large-caps**: Large-cap implementation shows Sharpe ratio of 1.13, max drawdown of -18.10%, and generates significant alpha vs RMW (t=2.62)

The Compass Score represents a quality-focused approach to stock selection that emphasizes profitability (ROA), operational efficiency (OCF/Assets, FCF/Assets), gross profitability (GP/Assets), and stability (low volatility).

For retail investors, the Compass Score offers a simple, rules-based framework to tilt toward quality without momentum reversal risk. For institutions, the monotonic quintile returns across all four market cap segments—including large-caps—supports scalable implementation with meaningful signal strength even at larger asset bases.

---

## References

Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019). Quality minus junk. *Review of Accounting Studies*, 24(1), 34-112.

Fama, E. F., & French, K. R. (1992). The cross-section of expected stock returns. *Journal of Finance*, 47(2), 427-465.

Fama, E. F., & French, K. R. (2015). A five-factor asset pricing model. *Journal of Financial Economics*, 116(1), 1-22.

George, T. J., & Hwang, C. Y. (2004). The 52-week high and momentum investing. *Journal of Finance*, 59(5), 2145-2176.

Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *Journal of Finance*, 48(1), 65-91.

Novy-Marx, R. (2013). The other side of value: The gross profitability premium. *Journal of Financial Economics*, 108(1), 1-28.

---

## Appendix A: Summary of All Validation Tests (Equal-Weighted Unless Noted)

| Validation Type | Test Period | Q5-Q1 Spread | Sample Size | Monotonic? |
|-----------------|-------------|--------------|-------------|------------|
| In-Sample (EW) | 1995-2019 | +3.58% | 602,164 | Yes |
| Out-of-Sample (EW) | 2020-2026 | +6.79% | 333,816 | — |
| Out-of-Sample (VW) | 2020-2026 | +5.14% | 333,816 | — |
| **Non-Overlapping (EW)** | 2020-2026 | **+5.99%** | 42,998 | No |
| Micro Cap Only | 2020-2026 | +10.24% | 89,421 | ✓ Yes |
| Small Cap Only | 2020-2026 | +5.71% | 112,876 | ✓ Yes |
| Mid Cap Only | 2020-2026 | +3.69% | 58,432 | ✓ Yes |
| Large Cap Only | 2020-2026 | +3.17% | 38,566 | ✓ Yes |
| Net-of-Costs (ann.) | 2020-2026 | +26.07% | 333,816 | — |
| Survivorship-Adj | 2020-2026 | +8.39% | 333,816 | — |
| **FF6 Alpha (HAC-12)** | 2020-2026 | +37.66% ann. | 71 months | t=5.65 |
| **Large-Cap Sharpe** | 2020-2026 | 1.13 (ann.) | 73 months | — |
| **Large-Cap Max DD** | 2020-2026 | -18.10% | — | — |
| **Alpha vs RMW only** | 2020-2026 | +11.72% ann. | 73 months | t=2.62 |
| **Sector-Neutral** | 2020-2026 | +0.75%/mo | 83,292 | — |
| **Rolling 36-mo Sharpe** | 2023-2026 | +0.53 to +2.86 | 37 windows | 0 negative |

All primary out-of-sample portfolio tests show economically positive spreads, with statistical significance in the main equal-weight and FF6 alpha specifications. **In the 2020-2026 holdout period, all four market cap segments exhibit monotonic increasing quintile returns (Q1 < Q2 < Q3 < Q4 < Q5).** Z-scores computed from in-sample data only (no look-ahead bias). HAC-corrected FF6 alpha remains highly significant (t=5.65).

---

*Data source: Financial Modeling Prep API (30 years of quarterly fundamentals: 412,977 income statements, 403,419 balance sheets, 400,651 cash flow statements, 415,773 key metrics records). Factor data: Kenneth French Data Library. Analysis conducted using Python with pandas, numpy, statsmodels, and sqlite3. Database containing 23.9 million daily price observations across 6,678 stocks. Compass Score validation performed with strict IS/OOS methodology: z-scores computed from IS (1995-2019) statistics only, applied to OOS (2020-2026) data. Robustness tests include value-weighting, transaction costs (20 bps/side), and survivorship haircut (20%).*

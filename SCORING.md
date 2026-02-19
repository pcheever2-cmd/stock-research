# StockCatalog Scoring Methodology

## Overview

StockCatalog uses two primary scores to rank stocks:
1. **Value Score V2** (0-100): Focus on value and growth potential
2. **Long-Term Score** (0-100): Comprehensive multi-factor score

Both scores have been backtested and validated to show positive predictive power for future returns.

## Validation Results

Based on comprehensive backtesting with 5 years of historical data:

| Score | Q5-Q1 Spread (3M) | Correlation |
|-------|------------------|-------------|
| Value Score V2 | +2.15% | +0.031 |
| Long-Term Score | +3.43% | +0.032 |

**Interpretation**: High-scoring stocks (top 20%) outperform low-scoring stocks (bottom 20%) by 2-3% over a 3-month period.

## Score Components

### Long-Term Score (max 100)

| Component | Max Points | What it Measures |
|-----------|-----------|------------------|
| Trend Score | 25 | Price above SMA200, SMA50 > SMA200, Price > SMA50 |
| Fundamentals Score | 25 | Revenue growth, EPS growth |
| Valuation Score | 16 | EV/EBITDA relative to sector |
| Momentum Score | 10 | RSI in healthy range, ADX trend strength |
| Market Risk Score | 10 | SPY above SMA200 (bull market) |

### Value Score V2 (max 100)

| Component | Max Points | What it Measures |
|-----------|-----------|------------------|
| Valuation | 40 | EV/EBITDA percentile within sector |
| Revenue Growth | 25 | Projected revenue growth |
| EPS Growth | 20 | Projected earnings growth |
| Quality | 15 | EBITDA growth + reasonable valuation |

## Sector-Relative Valuation

**Key Innovation**: Valuation thresholds are sector-specific.

"Cheap" for Technology (EV/EBITDA < 10.7) is different from "Cheap" for Energy (EV/EBITDA < 5.5).

### Sector EV/EBITDA Thresholds

| Sector | P25 (Cheap) | P50 (Fair) | P75 (Expensive) |
|--------|-------------|------------|-----------------|
| Energy | 5.5x | 7.9x | 11.5x |
| Communication Services | 5.7x | 10.3x | 16.1x |
| Financial Services | 7.6x | 10.7x | 15.7x |
| Healthcare | 6.5x | 13.8x | 22.6x |
| Technology | 10.7x | 17.0x | 29.8x |
| Industrials | 10.1x | 14.7x | 19.8x |
| Consumer Cyclical | 8.8x | 12.0x | 18.4x |
| Consumer Defensive | 9.0x | 12.5x | 17.3x |
| Basic Materials | 9.2x | 13.0x | 20.5x |
| Real Estate | 11.0x | 15.0x | 19.3x |
| Utilities | 10.1x | 12.2x | 13.8x |

## Component Predictive Power

Based on backtesting, components ranked by correlation with 3-month returns:

| Component | Correlation | Assessment |
|-----------|-------------|------------|
| Trend Score | +0.055 | **Best predictor** |
| Fundamentals Score | +0.051 | **Strong predictor** |
| Valuation Score | -0.071* | Negative without sector adjustment |
| Momentum Score | -0.004 | Weak |
| Market Risk Score | -0.033 | Weak |

*Note: Valuation score showed negative correlation before sector adjustment was implemented. Sector-relative valuation corrects this bias.

## Backtesting Methodology

1. **No Look-Ahead Bias**: Scores are calculated using only data available at the time
2. **Trading Costs Included**: Returns are net of 0.4% round-trip costs (0.2% per side)
3. **Outlier Handling**: Returns capped at ±100% to prevent penny stock distortion
4. **Minimum Price Filter**: Stocks below $5 excluded to avoid penny stocks

## Limitations

1. **Correlations are weak**: Even the best components have r < 0.06
2. **No guarantee**: Past performance does not guarantee future results
3. **Market regime dependent**: Scores work better in some market conditions
4. **Data quality**: Dependent on FMP API data accuracy

## Usage Recommendations

1. **Use as a filter, not sole decision**: Scores identify candidates, not guarantees
2. **Combine with other research**: Check company news, filings, and sector trends
3. **Diversify**: Don't concentrate in highest-scoring stocks only
4. **Rebalance periodically**: Scores change as new data arrives

## Files

| File | Purpose |
|------|---------|
| `score_long_term_OPTIMIZED.py` | Main scoring engine |
| `validate_scoring.py` | Backtest validation script |
| `run_backtest.py` | Full historical backtest |
| `analyze_sector_valuations.py` | Sector threshold analysis |

## Updates

- **2026-02-13**: Added sector-relative valuation thresholds
- **2026-02-13**: Added trading costs to backtest (0.4% round-trip)
- **2026-02-13**: Created validation script confirming +2-3% Q5-Q1 spread

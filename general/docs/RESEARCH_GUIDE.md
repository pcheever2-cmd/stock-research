# Research Papers & Validation Documentation

This directory contains all research papers, validation studies, and backtesting documentation for the Stock Research scoring methodologies.

---

## Directory Structure

```
research_papers/
├── compass/                     # Compass Score (Quality/Value)
│   ├── RESEARCH_PAPER.md        # Main 46-page research paper
│   ├── compass_quality_filters_validation.md
│   └── quality_filters_validation_summary.md
└── moonshot/                    # Moonshot Score (Quality/Growth)
    ├── moonshot_quality_first_validation_paper.md
    └── moonshot_analysis_and_fixes.md

validations/
├── compass/                     # 32 Compass validation scripts
└── moonshot/                    # 21 Moonshot validation scripts
```

---

## Compass Score Research

### Main Paper: [RESEARCH_PAPER.md](compass/RESEARCH_PAPER.md)

**Key Results:**
- Out-of-Sample Spread: +6.79% (Q5-Q1)
- FF6 Alpha: +38.20% annualized (t-stat 4.48)
- Value-Weighted Spread: +5.14%
- Net-of-Costs: +26.07% annualized
- All 4 market cap segments show monotonic quintile returns

### Analyses in Paper → Validation Scripts

| Section | Analysis | Script(s) |
|---------|----------|-----------|
| §3.3 | Out-of-Sample Validation | `compass_oos_validation.py` |
| §3.4 | Market Cap Segmentation | `v4_market_cap_analysis.py` |
| §3.5 | Multi-Period Rolling | `v4_multiperiod_validation.py` |
| §3.7 | Factor Contribution | `v4_factor_contribution.py`, `quick_factor_contribution.py` |
| §3.8 | Analyst Signal Evaluation | `v4_analyst_by_sector.py`, `v4_analyst_hybrid.py`, `v4_analyst_rolling.py` |
| §3.9 | Multi-Horizon Analysis | `multi_horizon_validation.py` |
| §4.2 | FF6 Regression | `v4_ff6_regression.py` |
| §4.3 | Within-Cap Normalization | `v4_within_cap_analysis.py` |
| §4.4 | Comprehensive Robustness | `v4_comprehensive_robustness.py`, `v4_robustness.py` |
| §4.5 | Institutional Statistics | `v4_institutional_stats.py` |
| §4.6 | Sector-Neutral Analysis | `sector_neutral_rolling_sharpe.py` |
| §4.7 | Rolling Sharpe Analysis | `sector_neutral_rolling_sharpe.py` |
| §4.8 | Regime Analysis | `v4_regime_analysis.py` |
| — | Quality Filters Backtest | `backtest_compass_quality_filters.py`, `validate_compass_quality_filters.py` |
| — | Z-Score Capping Analysis | `backtest_compass_zscore_capped.py`, `analyze_extreme_zscores.py` |
| — | A-Rated Stock Audit | `audit_a_rated_stocks.py` |
| — | Full Validation Suite | `v4_full_validation.py`, `v4_rigorous_validation.py` |
| — | Score Diagnostics | `compass_diagnostic_analysis.py` |
| — | Original Backtest | `compass_score_backtest.py` |
| — | Long-Term Score | `long_term_score_analysis.py` |

### All Compass Validation Scripts (32 files)

```
validations/compass/
├── analyze_extreme_zscores.py        # Z-score outlier analysis
├── audit_a_rated_stocks.py           # Audit top-rated stocks for anomalies
├── backtest_compass_quality_filters.py
├── backtest_compass_quality_filters_optimized.py
├── backtest_compass_zscore_capped.py # Z-score capping impact
├── compass_diagnostic_analysis.py    # Score component diagnostics
├── compass_oos_validation.py         # Out-of-sample validation
├── compass_score_backtest.py         # Original backtest
├── long_term_score_analysis.py       # Long-horizon analysis
├── multi_horizon_validation.py       # 1mo/3mo/1yr/2yr horizons
├── quick_factor_contribution.py      # Factor attribution
├── sector_neutral_rolling_sharpe.py  # Sector-neutral + rolling Sharpe
├── test_zscore_capping.py            # Z-score capping tests
├── v4_analyst_by_sector.py           # Analyst accuracy by sector
├── v4_analyst_by_sector_detail.py    # Detailed analyst analysis
├── v4_analyst_hybrid.py              # Hybrid analyst signal
├── v4_analyst_hybrid_v2.py           # Analyst signal v2
├── v4_analyst_rolling.py             # Rolling analyst accuracy
├── v4_comprehensive_robustness.py    # Full robustness suite
├── v4_factor_contribution.py         # Factor contribution analysis
├── v4_ff6_regression.py              # Fama-French 6-factor regression
├── v4_full_validation.py             # Complete validation
├── v4_institutional_stats.py         # Institutional-grade statistics
├── v4_market_cap_analysis.py         # Market cap segmentation
├── v4_multiperiod_validation.py      # Rolling period validation
├── v4_regime_analysis.py             # Market regime analysis
├── v4_revised_validation.py          # Revised methodology test
├── v4_rigorous_validation.py         # Rigorous validation suite
├── v4_robustness.py                  # Robustness checks
├── v4_within_cap_analysis.py         # Within-cap quintile analysis
├── validate_compass_quality_filters.py # Quality filter validation
└── validate_scoring.py               # Score validation
```

---

## Moonshot Score Research

### Main Paper: [moonshot_quality_first_validation_paper.md](moonshot/moonshot_quality_first_validation_paper.md)

**Key Results:**
- Out-of-Sample Spread: +23.58% (Q5-Q1)
- FF6 Alpha: +33.98% (t-stat 4.82)
- Small-cap: +19.59%, Mid-cap: +22.32%, Large-cap: +24.35%
- Non-overlapping returns: +27.71% (t=8.67)
- 100% of rolling 3-year windows positive

### Analyses in Paper → Validation Scripts

| Section | Analysis | Script(s) |
|---------|----------|-----------|
| Main | Quality-First Validation | `moonshot_quality_first_validation.py` |
| Main | Market Cap Analysis | `moonshot_market_cap_analysis.py` |
| §Robustness | Non-Overlapping Returns | `moonshot_comprehensive_robustness.py` |
| §Robustness | Bootstrap CI | `moonshot_comprehensive_robustness.py` |
| §Robustness | Sub-Period Consistency | `moonshot_comprehensive_robustness.py` |
| §Robustness | Within-Cap Quintiles | `moonshot_within_cap_analysis.py` |
| §Robustness | Rolling Performance | `moonshot_rolling_performance.py` |
| §Sector | Sector-Neutral Analysis | `moonshot_sector_neutral.py` |
| §Sector | 25-Year Sector Evolution | `moonshot_sector_evolution.py` |
| §FF6 | Fama-French Regression | `moonshot_ff6_regression.py` |
| — | Factor Contribution | `moonshot_factor_contribution.py` |
| — | Regime Analysis | `moonshot_regime_analysis.py` |
| — | Institutional Statistics | `moonshot_institutional_stats.py` |
| — | Rigorous Validation | `moonshot_rigorous_validation.py` |
| — | Compass+Moonshot Interaction | `compass_moonshot_interaction.py` |
| — | Golden Stocks Backtest | `golden_stocks_backtest.py` |

### All Moonshot Validation Scripts (21 files)

```
validations/moonshot/
├── compass_moonshot_interaction.py   # Compass + Moonshot correlation
├── debug_scoring.py                  # Score debugging
├── golden_stocks_backtest.py         # Combined score analysis
├── moonshot_comprehensive_robustness.py  # Full robustness suite
├── moonshot_factor_contribution.py   # Factor attribution
├── moonshot_ff6_regression.py        # Fama-French 6-factor
├── moonshot_hybrid_validation.py     # Hybrid methodology test
├── moonshot_institutional_stats.py   # Institutional statistics
├── moonshot_market_cap_analysis.py   # Market cap segmentation
├── moonshot_phase2_validation.py     # Phase 2 validation
├── moonshot_price_only.py            # Price-only factors test
├── moonshot_quality_first_validation.py      # Main validation
├── moonshot_quality_first_validation_optimized.py
├── moonshot_regime_analysis.py       # Market regime analysis
├── moonshot_rigorous_validation.py   # Rigorous validation
├── moonshot_rolling_performance.py   # Rolling window analysis
├── moonshot_score_analysis.py        # Score component analysis
├── moonshot_sector_evolution.py      # 25-year sector trends
├── moonshot_sector_neutral.py        # Sector-neutral test
├── moonshot_validation_optimized.py  # Optimized validation
└── moonshot_within_cap_analysis.py   # Within-cap quintiles
```

---

## Study Methodology

### Data Period
- **Total:** 1995-2026 (31 years)
- **In-Sample:** 1995-2019 (24 years)
- **Out-of-Sample:** 2020-2026 (6 years)

### Universe
- **Compass:** All US stocks with sufficient data
- **Moonshot:** US stocks with market cap ≥ $500M

### Validation Approach
- Strict out-of-sample holdout (no parameter tuning on OOS data)
- Fama-French 6-factor regression with Newey-West HAC standard errors
- Multiple robustness tests (value-weighting, transaction costs, survivorship)
- Rolling window analysis for stability
- Sector-neutral tests to verify within-sector stock selection

---

## Quick Reference: Key Scripts

| Purpose | Compass | Moonshot |
|---------|---------|----------|
| Main Validation | `compass_oos_validation.py` | `moonshot_quality_first_validation.py` |
| FF6 Alpha | `v4_ff6_regression.py` | `moonshot_ff6_regression.py` |
| Market Cap | `v4_market_cap_analysis.py` | `moonshot_market_cap_analysis.py` |
| Robustness | `v4_comprehensive_robustness.py` | `moonshot_comprehensive_robustness.py` |
| Factor Attribution | `v4_factor_contribution.py` | `moonshot_factor_contribution.py` |
| Sector-Neutral | `sector_neutral_rolling_sharpe.py` | `moonshot_sector_neutral.py` |
| Quality Filters | `validate_compass_quality_filters.py` | (built into scoring) |

---

*Last Updated: March 2026*

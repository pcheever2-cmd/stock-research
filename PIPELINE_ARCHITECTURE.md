# Pipeline Architecture — FMP Data Workflows

## Workflow Schedule Overview

| Workflow | Schedule | Duration | What It Does |
|----------|----------|----------|--------------|
| **Daily Pipeline** | Weekdays 6 AM UTC (1 AM EST) | ~2 hrs | Full data pull: prices, analysts, scores, export, deploy |
| **Compass Score Update** | Mon/Wed/Fri 7 AM UTC (2 AM EST) | ~13 min | Recompute compass/moonshot/valuation scores, export, deploy |
| **Price Update** | Weekdays 12 PM + 4:30 PM EST | ~2 min | Prices only: bulk quotes, update JSONs, deploy |

## Data Flow

```
FMP API
  │
  ├── Daily Pipeline (1 AM EST)
  │     ├── Batch Quotes ──────────────► nasdaq_stocks.db (prices)
  │     ├── Analyst Data (9 endpoints) ► nasdaq_stocks.db (ratings, targets, estimates)
  │     ├── Financial Scores ──────────► nasdaq_stocks.db (Altman Z, Piotroski)
  │     ├── Company Profiles ──────────► nasdaq_stocks.db (name, sector, description)
  │     └── export_website_stocks.py ──► stocks.json / stocks-public.json / stocks-premium.json
  │           └── Push to compass-score-site repo ──► Cloudflare Pages deploy + KV sync
  │
  ├── Compass Scores (2 AM EST, Mon/Wed/Fri)
  │     ├── compute_compass_scores.py ─► nasdaq_stocks.db (uses backtest.db fundamentals)
  │     ├── compute_moonshot_scores.py ► nasdaq_stocks.db (uses backtest.db fundamentals)
  │     ├── compute_valuation_scores.py► nasdaq_stocks.db (uses backtest.db prices)
  │     └── export + deploy (same as above)
  │
  └── Price Update (12 PM + 4:30 PM EST)
        ├── Bulk Quotes (~5 API calls) ► Update stocks-public.json prices in-place
        └── Deploy to Cloudflare Pages + KV sync
```

## Scripts & FMP Endpoints

### Daily Pipeline Scripts (in execution order)

| # | Script | FMP Endpoints | DB Tables | Time |
|---|--------|---------------|-----------|------|
| 1 | `run_pipeline_OPTIMIZED.py` | `/stable/batch-quote` | stock_consensus (prices) | ~60 min |
| 2 | `update_analyst_OPTIMIZED.py` | `/stable/profile`, `/stable/quote`, `/stable/price-target-consensus`, `/stable/price-target`, `/stable/analyst-estimates`, `/stable/ratings-snapshot`, `/stable/grades`, `/stable/key-metrics-ttm`, `/stable/ratios-ttm` | stock_consensus (analyst data) | 5-10 min |
| 3 | `enrich_stock_data.py` | `/stable/financial-scores`, `/stable/profile` | stock_consensus (Altman Z, Piotroski, Forward P/E, PEG) | ~10 min |
| 4 | `score_long_term_OPTIMIZED.py` | None (uses backtest.db) | stock_consensus (long_term_score, trend_signal) | 3-5 min |
| 5 | `export_website_stocks.py` | None (reads DB) | outputs stocks.json, stocks-public.json, stocks-premium.json | ~2 min |

### Compass Score Scripts (no FMP calls — use backtest.db)

| Script | What It Computes | Factors |
|--------|-----------------|---------|
| `compute_compass_scores.py` | Quality score (0-100, A-F) | ROA, Gross Profit/Assets, OCF/Assets, FCF/Assets, Volatility, Asset Growth |
| `compute_moonshot_scores.py` | Growth score (0-100, A-F) | Revenue CAGR, EPS CAGR, Gross Margin, FCF Margin, ROE, Momentum |
| `compute_valuation_scores.py` | Valuation rating | Price vs SMA200, Price vs SMA50, 52W Range Position |

### Historical Collectors (manual/one-time, not scheduled)

| Script | FMP Endpoints | Purpose |
|--------|---------------|---------|
| `collectors/collect_historical_data.py` | `/stable/historical-price-eod/full`, `/stable/income-statement`, `/stable/balance-sheet-statement`, `/stable/cash-flow-statement`, `/stable/key-metrics` | Backfill 5-30 years of price + fundamental data |
| `collectors/collect_analyst_data.py` | `/stable/grades`, `/stable/analyst-estimates`, `/stable/price-target-summary` | Historical analyst ratings for accuracy backtesting |
| `collectors/collect_dcf_data.py` | `/stable/discounted-cash-flow` | DCF valuations |

## Database Architecture

### `nasdaq_stocks.db` — Fast Daily Database
- Downloaded from GitHub release each CI run
- Single table: `stock_consensus` (~190 columns)
- Contains: current prices, analyst data, all scores, company info
- Updated by: daily pipeline + compass scores workflow

### `backtest.db` — Large Historical Database
- ~530 MB (trimmed for CI) / 4.9 GB (local full)
- Tables: historical_prices, historical_income_statements, historical_balance_sheets, historical_cash_flows, historical_key_metrics, historical_grades, analyst_estimates_snapshot, analyst_sector_accuracy
- Used by: compass/moonshot/valuation score computation
- Updated by: manual collector scripts

## Rate Limits

| Context | Rate | Setting |
|---------|------|---------|
| Async collectors (historical, DCF, analyst) | 2,500/min | `config.py: HISTORICAL_CALLS_PER_MINUTE` |
| `update_analyst_OPTIMIZED.py` | 700/min | Hardcoded in script |
| `enrich_stock_data.py` | ~300/min (10 threads x 0.2s delay) | `RATE_LIMIT_DELAY = 0.2` |
| FMP plan limit | 3,000/min | FMP account tier |

## CI Environment

### GitHub Secrets Required
| Secret | Purpose |
|--------|---------|
| `FMP_API_KEY` | Financial Modeling Prep API access |
| `SITE_DEPLOY_TOKEN` | GitHub PAT to push to compass-score-site repo |
| `CLOUDFLARE_API_TOKEN` | Cloudflare Pages deploy + KV sync |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account identifier |

### Build Environment Variables (non-secret)
| Variable | Value | Purpose |
|----------|-------|---------|
| `PUBLIC_SUPABASE_URL` | `https://jyadpvwxedqvqlrsghzy.supabase.co` | Astro build needs for Supabase client |
| `PUBLIC_SUPABASE_ANON_KEY` | `sb_publishable_mWnsNqWDuUYfbdDJrr5u6w_bzFwgQDk` | Astro build needs for Supabase client |

# Stockbrowse Deployment Reference

## Sites & Repos

| Site | URL | Repo Location | Deploys To |
|------|-----|---------------|------------|
| Landing Page | stockbrowse.co | `/Users/pcheev/Documents/stockbrowse-landing` | Cloudflare Pages (auto-deploy) |
| Main App | (app subdomain or separate) | `/tmp/stockbrowse-app-fix` | Cloudflare Pages (auto-deploy) |

## Landing Page (`stockbrowse-landing`)
- **Purpose**: Public-facing "Coming Soon" landing page
- **Tech**: Astro static site
- **Deploy**: Push to GitHub triggers Cloudflare Pages build
- **Content**: Waitlist signup, feature preview, timeline

## Main App (`stockbrowse-app-fix`)
- **Purpose**: Full stock browsing application
- **Tech**: Astro + Tailwind
- **Features**: Browse stocks, score lookup, watchlist, Learn/blog, pricing
- **Data**: Pulls from `stocks.json` exported by Python pipeline

## Data Pipeline
- **Database**: `/Users/pcheev/Documents/Stock Research V2/nasdaq_stocks.db`
- **Update Script**: `update_analyst_OPTIMIZED.py` (runs daily)
- **Export Script**: `export_website_stocks.py` (generates `stocks.json`)
- **Scoring**: `compute_compass_scores.py`, `compute_moonshot_scores.py`

## Key Files
- App stocks data: `/tmp/stockbrowse-app-fix/src/data/stocks.json`
- App types: `/tmp/stockbrowse-app-fix/src/data/stocks.ts`

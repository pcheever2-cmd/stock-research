# Stockbrowse Deployment Reference

## Sites & Repos

| Site | URL | Local Path | GitHub Repo | Deploys To |
|------|-----|------------|-------------|------------|
| Landing Page | stockbrowse.co | `/Users/pcheev/Documents/stockbrowse-landing` | `pcheever2-cmd/stockbrowse-landing` | Cloudflare Pages (auto-deploy) |
| Main App | app.stockbrowse.co | `/Users/pcheev/Documents/compass-score-site` | `pcheever2-cmd/stockbrowse-app` | Cloudflare Pages (CI deploy via wrangler) |
| Data Pipeline | (backend only) | `/Users/pcheev/Documents/Stock Research V2` | `pcheever2-cmd/stock-research-v2` | GitHub Actions (daily cron) |

## How Deployment Works

**Landing Page** uses Cloudflare Pages auto-deploy: push to `main` on GitHub → Cloudflare builds → deploys automatically.

**Main App** uses CI-driven deploy: the daily pipeline and compass-scores workflows build the site and run `wrangler pages deploy` to push to Cloudflare Pages (project: `stockbrowse-app-pages`). There is no git integration on the Pages project — deploys happen only via CI or manual `wrangler pages deploy dist --project-name=stockbrowse-app-pages --commit-dirty=true` from the local compass-score-site directory.

## Landing Page (`stockbrowse-landing`)
- **URL**: stockbrowse.co
- **Purpose**: Public-facing marketing landing page (currently stagnant "Coming Soon")
- **Tech**: Astro static site + Cloudflare Worker (`src/worker.ts`)
- **Backend**: Waitlist signup API via Worker + KV
- **Deploy**: Push to GitHub → auto-deploy
- **Secrets**: Set via Cloudflare dashboard (Workers & Pages > stockbrowse-landing > Settings)

## Main App (`compass-score-site`)
- **URL**: app.stockbrowse.co
- **Purpose**: Full stock browsing and research application
- **Tech**: Astro SSG + Tailwind CSS + Cloudflare Pages Functions (`/functions` directory)
- **Features**: Browse stocks, score lookup, watchlist, blog, pricing, auth, premium tiers
- **Data**: Pulls from `stocks-public.json` (free fields) at build time. Premium fields served at runtime via `/api/stocks/premium` from Cloudflare KV.
- **Deploy**: CI runs `wrangler pages deploy dist --project-name=stockbrowse-app-pages` (no git integration on Pages project)
- **Cloudflare Pages project name**: `stockbrowse-app-pages` (rename to `stockbrowse-app` after old Worker is deleted)
- **Build command**: `npm run build` (includes leakage canary check via `scripts/check-leakage.sh`)

### Main App — Premium Tier Architecture (added May 2026)

**Data split**: The export pipeline produces two files:
- `src/data/stocks-public.json` — 14 free-tier fields, committed to repo, baked into static HTML
- `src/data/stocks-premium.json` — 25 gated fields, NOT committed, uploaded to Cloudflare KV as sharded `stock:SYMBOL` keys

**Leakage canary**: `scripts/check-leakage.sh` runs as part of `npm run build`. It greps the built HTML for any premium field names. If any are found, the build fails. This prevents accidentally baking premium data into static pages.

**Auth**: Supabase Auth (email + password). Session stored by Supabase JS SDK. Tier stored in `profiles` table in Supabase Postgres with RLS.

**Payments**: Stripe Checkout + Customer Portal. Webhook at `/api/webhooks/stripe` updates `profiles.subscription_tier` on payment events.

**Tiers**:
| Tier | Price | Key features |
|------|-------|-------------|
| Free | $0 | Compass Score, grade, browse, 1Y chart |
| Newsletter | $1/mo | Monthly Compass Report |
| Plus | $10/mo | Score breakdown, Moonshot, analyst data, watchlist |
| Pro | $15/mo | Valuation Score, multi-watchlists, SMS alerts |

**Pages Functions** (in `/functions` directory, auto-deployed by Cloudflare):
- `/api/auth/me` — returns user profile + tier
- `/api/stocks/premium` — gated premium fields from KV
- `/api/checkout` — creates Stripe Checkout session
- `/api/portal` — Stripe Customer Portal
- `/api/webhooks/stripe` — Stripe webhook handler
- `/api/watchlists/*` — CRUD for server-backed watchlists
- `/api/newsletter/subscribe` — newsletter signup

**Secrets** (set via `wrangler pages secret put <NAME> --project-name=stockbrowse-app-pages` or Cloudflare dashboard > Pages > stockbrowse-app-pages > Settings):
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`
- `STRIPE_SECRET_KEY`
- `STRIPE_WEBHOOK_SECRET`

**Build-time env vars** (set in Cloudflare dashboard > Settings > Build):
- `PUBLIC_SUPABASE_URL`
- `PUBLIC_SUPABASE_ANON_KEY`

**Stripe Product IDs**:
- Newsletter: `prod_USo0WGwfiIcxqT`
- Plus: `prod_USo32e2kMoaXVS`
- Pro: `prod_USo9k60n0wpxdj`

**Supabase project**: `jyadpvwxedqvqlrsghzy` (https://supabase.com/dashboard/project/jyadpvwxedqvqlrsghzy)

**KV namespace** (STOCKS_PREMIUM_KV): `a16202c79b5b4372b3a686d6a4f8baf1`

## Data Pipeline (`Stock Research V2`)
- **Database**: `nasdaq_stocks.db` (current scores + fundamentals)
- **Backtest DB**: `backtest.db` (30 years of historical data)
- **Daily pipeline**: `run_pipeline_OPTIMIZED.py` via GitHub Actions (weekdays 6 AM UTC)
- **Scoring**: `compute_compass_scores.py` (quality), `compute_moonshot_scores.py` (growth), `compute_valuation_scores.py` (technical valuation)
- **Export**: `export_website_stocks.py` → generates `stocks.json` + `stocks-public.json` + `stocks-premium.json`, pushes to compass-score-site repo

### Export → Website data flow
```
nasdaq_stocks.db + backtest.db
        ↓
  export_website_stocks.py
        ↓
  stocks-public.json (committed to compass-score-site, build-time import)
  stocks-premium.json (uploaded to Cloudflare KV as stock:SYMBOL keys)
  stocks.json (backwards compat, will be removed)
        ↓
  CI: git push to compass-site repo → wrangler pages deploy → stockbrowse-app-pages.pages.dev
```

## Key Files

### Main App
- Stock data (public): `compass-score-site/src/data/stocks-public.json`
- Stock types: `compass-score-site/src/data/stocks.ts` (PublicStock + PremiumFields)
- Subscription logic: `compass-score-site/src/lib/subscription.ts`
- Supabase client: `compass-score-site/src/lib/supabase.ts`
- Auth fetch wrapper: `compass-score-site/src/lib/authFetch.ts`
- Leakage canary: `compass-score-site/scripts/check-leakage.sh`
- Cloudflare config: `compass-score-site/wrangler.toml`

### Data Pipeline
- Compass scoring: `Stock Research V2/compute_compass_scores.py`
- Moonshot scoring: `Stock Research V2/compute_moonshot_scores.py`
- Website export: `Stock Research V2/export_website_stocks.py`
- This file: `Stock Research V2/general/docs/DEPLOYMENT_REFERENCE.md`

### Database migrations (run in Supabase SQL Editor)
- `compass-score-site/migrations/0001_profiles.sql`
- `compass-score-site/migrations/0002_watchlists.sql`
- `compass-score-site/migrations/0003_newsletter_and_stripe.sql`

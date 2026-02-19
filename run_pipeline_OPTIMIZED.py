#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR - Run complete update & scoring pipeline
Coordinates all data fetching and scoring in optimal order
"""

import asyncio
import sqlite3
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
import sys
from pathlib import Path

# Import the optimized modules
sys.path.insert(0, str(Path(__file__).parent))
from config import FMP_API_KEY, DATABASE_NAME, PARQUET_PATH

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

PRICE_BATCH_SIZE = 400

# ==================== PIPELINE STEPS ====================

def run_batch_price_update():
    """Step 0: Fast batch price update for ALL stocks using bulk endpoint"""
    log.info("\n" + "="*60)
    log.info("STEP 0: Batch Price Update (all stocks)")
    log.info("="*60)

    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("SELECT symbol FROM stock_consensus", conn)
    conn.close()

    symbols = df['symbol'].dropna().tolist()
    if not symbols:
        log.warning("No symbols found in stock_consensus table!")
        return

    log.info(f"Updating prices for {len(symbols)} stocks in batches of {PRICE_BATCH_SIZE}...")
    updated_count = 0

    for i in range(0, len(symbols), PRICE_BATCH_SIZE):
        batch = symbols[i:i + PRICE_BATCH_SIZE]
        batch_str = ','.join(batch)
        url = f"https://financialmodelingprep.com/stable/batch-quote?symbols={batch_str}&apikey={FMP_API_KEY}"

        try:
            resp = requests.get(url, timeout=30)
            data = resp.json()

            if not isinstance(data, list):
                log.warning(f"Batch {i//PRICE_BATCH_SIZE + 1}: Unexpected response: {str(data)[:200]}")
                continue

            conn = sqlite3.connect(DATABASE_NAME)
            cur = conn.cursor()
            for quote in data:
                symbol = quote.get('symbol')
                price = quote.get('price')
                if symbol and price is not None:
                    cur.execute(
                        "UPDATE stock_consensus SET current_price = ?, price_updated_at = ? WHERE symbol = ?",
                        (price, datetime.now(timezone.utc).isoformat(), symbol)
                    )
                    if cur.rowcount > 0:
                        updated_count += 1
            conn.commit()
            conn.close()

            log.info(f"  Batch {i//PRICE_BATCH_SIZE + 1}: {len(data)} prices "
                     f"({min(i + PRICE_BATCH_SIZE, len(symbols))}/{len(symbols)})")
        except Exception as e:
            log.error(f"  Batch {i//PRICE_BATCH_SIZE + 1} error: {e}")

    log.info(f"  ✓ Price update complete: {updated_count} stocks updated")

async def run_analyst_update():
    """Step 1: Update analyst data"""
    log.info("\n" + "="*60)
    log.info("STEP 1: Updating Analyst Consensus Data")
    log.info("="*60)
    
    from update_analyst_OPTIMIZED import main as update_main
    await update_main()

async def run_scoring():
    """Step 2: Calculate scores"""
    log.info("\n" + "="*60)
    log.info("STEP 2: Calculating Long-Term & Value Scores")
    log.info("="*60)
    
    from score_long_term_OPTIMIZED import main as score_main
    await score_main()

def generate_stats_report():
    """Step 3: Generate summary statistics"""
    log.info("\n" + "="*60)
    log.info("STEP 3: Summary Statistics")
    log.info("="*60)
    
    conn = sqlite3.connect(DATABASE_NAME)
    
    # Total stocks
    total = conn.execute("SELECT COUNT(*) FROM stock_consensus").fetchone()[0]
    
    # Stocks with analyst data
    with_analysts = conn.execute("""
        SELECT COUNT(*) FROM stock_consensus 
        WHERE num_analysts >= 1 AND avg_price_target IS NOT NULL
    """).fetchone()[0]
    
    # Scored stocks
    scored = conn.execute("""
        SELECT COUNT(*) FROM stock_consensus 
        WHERE long_term_score IS NOT NULL
    """).fetchone()[0]
    
    # High quality opportunities
    high_quality = conn.execute("""
        SELECT COUNT(*) FROM stock_consensus 
        WHERE value_score >= 70 
          AND long_term_score >= 60
          AND num_analysts >= 5
          AND upside_percent >= 20
    """).fetchone()[0]
    
    # Coverage by market cap
    cap_coverage = conn.execute("""
        SELECT cap_category, COUNT(*) as count
        FROM stock_consensus
        WHERE num_analysts >= 1
        GROUP BY cap_category
        ORDER BY count DESC
    """).fetchall()
    
    # Top sectors
    sector_coverage = conn.execute("""
        SELECT industry, COUNT(*) as count
        FROM stock_consensus
        WHERE num_analysts >= 1
        GROUP BY industry
        ORDER BY count DESC
        LIMIT 10
    """).fetchall()
    
    conn.close()
    
    # Print report
    log.info(f"\n📊 Coverage Statistics:")
    log.info(f"  Total stocks in database: {total:,}")
    log.info(f"  Stocks with analyst coverage: {with_analysts:,} ({with_analysts/total*100:.1f}%)")
    log.info(f"  Stocks scored: {scored:,} ({scored/total*100:.1f}%)")
    log.info(f"  High-quality opportunities: {high_quality:,}")
    
    log.info(f"\n📈 Coverage by Market Cap:")
    for cap, count in cap_coverage:
        log.info(f"  {cap}: {count:,} stocks")
    
    log.info(f"\n🏭 Top 10 Sectors by Coverage:")
    for sector, count in sector_coverage:
        log.info(f"  {sector}: {count:,} stocks")
    
    log.info("\n✅ Pipeline Complete!")
    log.info("Next: Open the Streamlit dashboard to view results")

# ==================== COVERAGE ANALYZER ====================

def validate_data_freshness():
    """Validate that data was actually updated during this pipeline run"""
    log.info("\n" + "="*60)
    log.info("Data Freshness Validation")
    log.info("="*60)

    conn = sqlite3.connect(DATABASE_NAME)

    # Check how many stocks were updated today
    updated_today = conn.execute("""
        SELECT COUNT(*) FROM stock_consensus
        WHERE DATE(last_updated) = DATE('now')
    """).fetchone()[0]

    # Check how many have stale data (not updated today)
    stale_count = conn.execute("""
        SELECT COUNT(*) FROM stock_consensus
        WHERE DATE(last_updated) < DATE('now') OR last_updated IS NULL
    """).fetchone()[0]

    # Total stocks
    total = conn.execute("SELECT COUNT(*) FROM stock_consensus").fetchone()[0]

    # Get oldest last_updated
    oldest = conn.execute("""
        SELECT MIN(last_updated) FROM stock_consensus
        WHERE last_updated IS NOT NULL
    """).fetchone()[0]

    # Get sample of stale stocks
    stale_sample = conn.execute("""
        SELECT symbol, last_updated, num_analysts
        FROM stock_consensus
        WHERE DATE(last_updated) < DATE('now') OR last_updated IS NULL
        ORDER BY num_analysts DESC NULLS LAST
        LIMIT 5
    """).fetchall()

    conn.close()

    # Report
    freshness_pct = (updated_today / total * 100) if total > 0 else 0
    log.info(f"\n📊 Data Freshness Report:")
    log.info(f"  Updated today: {updated_today:,} / {total:,} ({freshness_pct:.1f}%)")
    log.info(f"  Stale (not updated today): {stale_count:,}")
    log.info(f"  Oldest timestamp: {oldest}")

    if stale_count > 0 and stale_sample:
        log.warning(f"\n⚠️  Sample of stale stocks:")
        for symbol, last_updated, analysts in stale_sample:
            analysts_str = str(analysts) if analysts else "N/A"
            log.warning(f"    {symbol}: last_updated={last_updated}, analysts={analysts_str}")

    # Validation threshold - warn if less than 50% updated
    if freshness_pct < 50:
        log.error(f"\n❌ CRITICAL: Only {freshness_pct:.1f}% of stocks updated today!")
        log.error("   Check API key, rate limits, or network connectivity.")
    elif stale_count > 0:
        log.warning(f"\n⚠️  {stale_count} stocks have stale data. This may be normal if:")
        log.warning("   - Some stocks have no analyst coverage")
        log.warning("   - API returned errors for some symbols")
    else:
        log.info(f"\n✅ All stocks have fresh data!")

    return updated_today, stale_count, total


def analyze_coverage_gaps():
    """Identify which stocks should have coverage but don't"""
    log.info("\n" + "="*60)
    log.info("Coverage Gap Analysis")
    log.info("="*60)
    
    conn = sqlite3.connect(DATABASE_NAME)
    
    # Find large/mid caps without coverage
    missing_coverage = conn.execute("""
        SELECT symbol, cap_category, industry, current_price
        FROM stock_consensus
        WHERE (num_analysts IS NULL OR num_analysts < 1)
          AND cap_category IN ('Large Cap (>$10B)', 'Mid Cap ($2B–$10B)')
        ORDER BY cap_category, industry
        LIMIT 50
    """).fetchall()
    
    if missing_coverage:
        log.warning(f"\n⚠️  Found {len(missing_coverage)} large/mid-cap stocks WITHOUT analyst coverage:")
        log.warning("These stocks might be:")
        log.warning("  1. Recently IPO'd")
        log.warning("  2. Foreign listings (ADRs)")
        log.warning("  3. Special purpose entities")
        log.warning("  4. Actually have coverage but FMP doesn't track it")
        
        log.info("\nSample of uncovered stocks:")
        for symbol, cap, industry, price in missing_coverage[:10]:
            log.info(f"  {symbol:<6} | {cap:<25} | {industry:<30} | ${price:.2f}")
    
    # Find stocks with stale data
    stale_stocks = conn.execute("""
        SELECT symbol, last_updated, upside_percent
        FROM stock_consensus
        WHERE last_updated < date('now', '-7 days')
          AND num_analysts >= 1
        ORDER BY upside_percent DESC
        LIMIT 20
    """).fetchall()
    
    if stale_stocks:
        log.warning(f"\n⚠️  Found {len(stale_stocks)} stocks with stale data (>7 days old):")
        for symbol, last_updated, upside in stale_stocks[:5]:
            upside_str = f"{upside:+.1f}%" if upside is not None else "N/A"
            log.warning(f"  {symbol}: Last updated {last_updated} (upside {upside_str})")
    
    conn.close()

# ==================== PARQUET EXPORT ====================

def export_dashboard_parquet():
    """Export dashboard data to parquet for Streamlit Cloud deployment"""
    log.info("\n" + "="*60)
    log.info("STEP 4: Exporting Dashboard Parquet")
    log.info("="*60)

    conn = sqlite3.connect(DATABASE_NAME)
    # Export ALL stocks (not just analyst-covered) per user request
    query = """
        SELECT symbol, company_name, company_description,
               current_price, avg_price_target, median_price_target,
               min_price_target, max_price_target,
               upside_percent, num_analysts, recommendation,
               consensus_rating, recent_ratings,
               cap_category, sector, industry, last_updated,
               enterprise_value, ebitda, ev_ebitda,
               total_debt, debt_ebitda, ocf_ev,
               peg_ratio, forward_pe, forward_ev_ebitda, ev_ebitda_reduction,
               projected_revenue_growth, projected_eps_growth,
               projected_ebitda_growth, earnings_growth,
               projected_revenue_next_year, projected_eps_next_year,
               sma50, sma200, rsi, adx, close_price_technical,
               long_term_score, value_score, value_score_v2,
               trend_score, fundamentals_score, valuation_score,
               momentum_score, market_risk_score,
               trend_signal, trend_signal_count,
               altman_z_score, piotroski_score, market_cap, exchange
        FROM stock_consensus
        ORDER BY value_score_v2 DESC NULLS LAST, upside_percent DESC NULLS LAST
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df.to_parquet(PARQUET_PATH, index=False)
    log.info(f"  Exported {len(df)} stocks to {PARQUET_PATH} ({Path(PARQUET_PATH).stat().st_size / 1024:.0f} KB)")

# ==================== MAIN ====================

async def main():
    start_time = datetime.now()
    
    log.info("="*60)
    log.info("🚀 MASTER PIPELINE - Optimized Stock Analysis")
    log.info("="*60)
    log.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    try:
        # Run pipeline
        run_batch_price_update()
        await run_analyst_update()

        # Validate data freshness after analyst update
        validate_data_freshness()

        await run_scoring()
        generate_stats_report()

        # Export parquet for Streamlit Cloud
        export_dashboard_parquet()

        # Optional: Analyze gaps
        log.info("\nRunning coverage gap analysis...")
        analyze_coverage_gaps()
        
    except Exception as e:
        log.error(f"\n❌ Pipeline failed: {e}")
        raise
    
    finally:
        elapsed = datetime.now() - start_time
        log.info("\n" + "="*60)
        log.info(f"Total runtime: {elapsed}")
        log.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
# File: update_analyst_consensus_fmp_v2.py — Smart daily update (only outdated analyst data)

import concurrent.futures
import requests
import sqlite3
import time
import random
import pandas as pd
import logging
from datetime import datetime, date

from config import FMP_API_KEY, DATABASE_NAME
BASE_STABLE = 'https://financialmodelingprep.com/stable'

TEST_MODE = False
TEST_TICKER = 'AAPL'

MAX_WORKERS = 3
MIN_DELAY = 0.2
MAX_DELAY = 0.7

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
else:
    logging.getLogger().setLevel(logging.INFO)

log = logging.getLogger(__name__)

def fetch_tickers_from_db():
    today = date.today().isoformat()

    try:
        conn = sqlite3.connect(DATABASE_NAME)
        
        if TEST_MODE:
            df = pd.read_sql_query("SELECT symbol FROM stock_consensus WHERE symbol = ?", conn, params=[TEST_TICKER])
            log.info(f"TEST MODE: Forcing update for {TEST_TICKER}")
        else:
            query = """
                SELECT symbol 
                FROM stock_consensus 
                WHERE last_updated < ? 
                   OR last_updated IS NULL 
                   OR avg_price_target IS NULL
                   OR num_analysts IS NULL
                ORDER BY last_updated ASC NULLS LAST
            """
            df = pd.read_sql_query(query, conn, params=[today])
        
        conn.close()
        tickers = df['symbol'].dropna().tolist()
        
        if tickers:
            log.info(f"Found {len(tickers)} stocks to update today")
        else:
            log.info("All stocks are already up to date!")
        
        return tickers
    except Exception as e:
        log.error(f"DB error: {e}")
        return []

def safe_get_first(data):
    if isinstance(data, list):
        return data[0] if data else {}
    elif isinstance(data, dict):
        return data
    return {}

def get_table_columns():
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(stock_consensus)")
        columns = {row[1] for row in cur.fetchall()}
        conn.close()
        return columns
    except Exception as e:
        log.error(f"Error getting table columns: {e}")
        return set()

def process_and_save(ticker):
    try:
        log.info(f"\nProcessing {ticker}...")

        # Quote
        quote_data = requests.get(f"{BASE_STABLE}/quote?symbol={ticker}&apikey={FMP_API_KEY}").json() or \
                     requests.get(f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={FMP_API_KEY}").json()
        quote = safe_get_first(quote_data)
        current_price = quote.get('price')
        market_cap = quote.get('marketCap')
        if not current_price or not market_cap:
            log.info(f"   → No quote data for {ticker}")
            return

        # Profile
        profile = safe_get_first(requests.get(f"{BASE_STABLE}/profile?symbol={ticker}&apikey={FMP_API_KEY}").json())
        industry = profile.get('industry') or profile.get('sector') or 'N/A'

        # Price Target Consensus
        target_data = requests.get(f"{BASE_STABLE}/price-target-consensus?symbol={ticker}&apikey={FMP_API_KEY}").json()
        target = safe_get_first(target_data)
        mean_target = target.get('targetConsensus') or target.get('targetMeanPrice')
        if not mean_target:
            log.info(f"   → No price target for {ticker} — skipping")
            return

        upside = (mean_target - current_price) / current_price * 100

        # Analyst estimates — robust growth + accurate count
        estimates_data = requests.get(f"{BASE_STABLE}/analyst-estimates?symbol={ticker}&period=annual&apikey={FMP_API_KEY}").json()

        num_analysts = 0
        projected_ebitda_growth = projected_eps_growth = projected_revenue_growth = None
        if estimates_data and len(estimates_data) >= 2:
            next_year = estimates_data[0]
            current_year = estimates_data[1]

            eps_curr = (current_year.get('estimatedEpsAvg') or current_year.get('estimatedEps') or 
                        current_year.get('epsAvg') or current_year.get('epsEstimated') or 0)
            eps_next = (next_year.get('estimatedEpsAvg') or next_year.get('estimatedEps') or 
                        next_year.get('epsAvg') or next_year.get('epsEstimated') or 0)

            rev_curr = (current_year.get('estimatedRevenueAvg') or current_year.get('estimatedRevenue') or 
                        current_year.get('revenueAvg') or current_year.get('revenueEstimated') or 0)
            rev_next = (next_year.get('estimatedRevenueAvg') or next_year.get('estimatedRevenue') or 
                        next_year.get('revenueAvg') or next_year.get('revenueEstimated') or 0)

            ebitda_curr = (current_year.get('estimatedEbitdaAvg') or current_year.get('estimatedEbitda') or 
                           current_year.get('ebitdaAvg') or 0)
            ebitda_next = (next_year.get('estimatedEbitdaAvg') or next_year.get('estimatedEbitda') or 
                           next_year.get('ebitdaAvg') or 0)

            if eps_curr > 0 and eps_next > 0:
                projected_eps_growth = (eps_next / eps_curr - 1) * 100

            if rev_curr > 0 and rev_next > 0:
                projected_revenue_growth = (rev_next / rev_curr - 1) * 100

            if ebitda_curr > 0 and ebitda_next > 0:
                projected_ebitda_growth = (ebitda_next / ebitda_curr - 1) * 100

            for year in estimates_data[:3]:
                count = (year.get('numberOfAnalysts') or year.get('numAnalysts') or
                         year.get('numAnalystsEps') or year.get('numberOfAnalystsEps') or
                         year.get('numAnalystsRevenue') or year.get('numberOfAnalystsRevenue') or 0)
                num_analysts = max(num_analysts, count)

        if num_analysts == 0:
            num_analysts = 1

        recommendation = safe_get_first(requests.get(f"{BASE_STABLE}/ratings-snapshot?symbol={ticker}&apikey={FMP_API_KEY}").json()).get('rating', 'N/A')

        grades_data = requests.get(f"{BASE_STABLE}/grades?symbol={ticker}&limit=5&apikey={FMP_API_KEY}").json()
        recent_ratings = "No recent rating changes"
        if grades_data:
            lines = []
            for g in grades_data:
                action = g.get('action', '').lower()
                emoji = "⬆️" if 'upgrade' in action else "⬇️" if 'downgrade' in action else "●"
                lines.append(f"{emoji} {g.get('date', '')[:10]}: {g.get('gradingCompany', '')} → {g.get('newGrade', '')}")
            recent_ratings = "\n".join(lines)

        metrics = safe_get_first(requests.get(f"{BASE_STABLE}/key-metrics-ttm?symbol={ticker}&apikey={FMP_API_KEY}").json())
        ev_ebitda = metrics.get('evToEBITDATTM')
        enterprise_value = metrics.get('enterpriseValueTTM')

        ratios = safe_get_first(requests.get(f"{BASE_STABLE}/ratios-ttm?symbol={ticker}&apikey={FMP_API_KEY}").json())
        ebitda = ratios.get('ebitdaTTM')

        cap = ('Micro/Penny Cap (<$300M)' if market_cap < 300e6 else
               'Small Cap ($300M–$2B)' if market_cap < 2e9 else
               'Mid Cap ($2B–$10B)' if market_cap < 10e9 else
               'Large Cap (>$10B)')

        log.info(f"{ticker:<8} | Upside {upside:+6.1f}% | Analysts {num_analysts} | EV/EBITDA {ev_ebitda or 0:.1f}x")

        columns = get_table_columns()

        base_fields = {
            'symbol': ticker,
            'avg_price_target': mean_target,
            'median_price_target': target.get('targetMedian'),
            'min_price_target': target.get('targetLow'),
            'max_price_target': target.get('targetHigh'),
            'num_analysts': num_analysts,
            'consensus_rating': recommendation,
            'last_updated': datetime.utcnow().isoformat(),  # Analyst data timestamp
            'current_price': current_price,
            'upside_percent': upside,
            'cap_category': cap,
            'industry': industry,
            'recent_ratings': recent_ratings,
            'recommendation': recommendation,
        }

        optional_fields = {
            'enterprise_value': enterprise_value,
            'ebitda': ebitda,
            'ev_ebitda': ev_ebitda,
            'projected_revenue_growth': projected_revenue_growth,
            'projected_eps_growth': projected_eps_growth,
            'projected_ebitda_growth': projected_ebitda_growth,
        }

        insert_fields = {k: v for k, v in {**base_fields, **optional_fields}.items() if k in columns}

        placeholders = ', '.join(['?'] * len(insert_fields))
        cols_str = ', '.join(insert_fields.keys())

        conn = sqlite3.connect(DATABASE_NAME)
        cur = conn.cursor()
        cur.execute(f"INSERT OR REPLACE INTO stock_consensus ({cols_str}) VALUES ({placeholders})",
                    tuple(insert_fields.values()))
        conn.commit()
        conn.close()

    except Exception as e:
        log.error(f"Failed {ticker}: {e}")

    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def main():
    tickers = fetch_tickers_from_db()
    if not tickers:
        return
    log.info(f"\nUpdating {len(tickers)} stocks today...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(process_and_save, tickers)
    log.info("\nDaily update complete! Run backfill_forward_valuation.py for forward metrics.")

if __name__ == "__main__":
    main()
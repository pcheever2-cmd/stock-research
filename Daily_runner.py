#!/usr/bin/env python3
# daily_runner.py - Full daily automation with separate price and analyst timestamps

import asyncio
import sqlite3
import requests
import pandas as pd
import time
import random
import logging
from datetime import datetime
import os
import sys

# ----------------------------- CONFIG -----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from config import FMP_API_KEY, DATABASE_NAME

LOG_FILE = os.path.join(SCRIPT_DIR, "daily_run.log")
BATCH_SIZE = 400

# ----------------------------- LOGGING -----------------------------
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(console)

# ----------------------------- BATCH PRICE UPDATE -----------------------------
def batch_update_prices():
    logging.info("=== Starting batch price update for ALL stocks ===")
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        df = pd.read_sql_query("SELECT symbol FROM stock_consensus", conn)
        symbols = df['symbol'].dropna().tolist()
        conn.close()

        if not symbols:
            logging.warning("No symbols found in stock_consensus table!")
            return

        total = len(symbols)
        logging.info(f"Found {total} stocks. Updating prices in batches of {BATCH_SIZE}...")

        updated_count = 0
        for i in range(0, total, BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            batch_str = ','.join(batch)
            url = f"https://financialmodelingprep.com/stable/batch-quote?symbols={batch_str}&apikey={FMP_API_KEY}"

            try:
                data = requests.get(url, timeout=30).json()
                
                if not isinstance(data, list):
                    if isinstance(data, dict) and 'Error Message' in data:
                        logging.warning(f"API error for batch {i//BATCH_SIZE + 1}: {data.get('Error Message')}")
                    else:
                        logging.warning(f"Unexpected API response for batch {i//BATCH_SIZE + 1}: {data}")
                    continue

                conn = sqlite3.connect(DATABASE_NAME)
                cur = conn.cursor()
                for quote in data:
                    symbol = quote.get('symbol')
                    price = quote.get('price')
                    if symbol and price is not None:
                        cur.execute("""
                            UPDATE stock_consensus 
                            SET current_price = ?, price_updated_at = ?
                            WHERE symbol = ?
                        """, (price, datetime.utcnow().isoformat(), symbol))
                        if cur.rowcount > 0:
                            updated_count += 1
                conn.commit()
                conn.close()

                logging.info(f"Batch {i//BATCH_SIZE + 1}: Updated {len(data)} prices (progress: {min(i + BATCH_SIZE, total)}/{total})")

            except Exception as e:
                logging.error(f"Request error in batch {i//BATCH_SIZE + 1}: {e}")

            time.sleep(random.uniform(0.5, 1.2))

        logging.info(f"Batch price update complete: {updated_count} prices updated.")

    except Exception as e:
        logging.error(f"Critical error in batch_update_prices: {e}")

# ----------------------------- MAIN -----------------------------
def main():
    logging.info("\n" + "="*60)
    logging.info("DAILY AUTOMATION RUN STARTED")
    logging.info("="*60)

    start_time = time.time()

    # 1. Update prices (only price_updated_at)
    batch_update_prices()

    # 2. Analyst consensus update (optimized async version)
    try:
        logging.info("Running optimized analyst consensus update...")
        from update_analyst_OPTIMIZED import main as analyst_main
        asyncio.run(analyst_main())
        logging.info("Analyst consensus update completed.")
    except Exception as e:
        logging.error(f"Consensus updater failed: {e}")

    # 3. Long-term scoring (optimized async version)
    try:
        logging.info("Running optimized long-term scoring...")
        from score_long_term_OPTIMIZED import main as score_main
        asyncio.run(score_main())
        logging.info("Long-term scoring completed.")
    except Exception as e:
        logging.error(f"Scoring failed: {e}")

    duration = time.time() - start_time
    logging.info(f"DAILY RUN COMPLETE in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    logging.info("="*60 + "\n")

if __name__ == "__main__":
    main()
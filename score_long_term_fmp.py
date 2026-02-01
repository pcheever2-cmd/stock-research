#!/usr/bin/env python3
# File: score_long_term_fmp.py
# Long-Term Score + Jack-style Value Score
# Fundamentals/Value from DB | Live Technicals (modern) | Market regime: SPY > SMA200 only

import concurrent.futures
import requests
import sqlite3
import time
import random
import pandas as pd
import logging
from datetime import datetime, date

from config import FMP_API_KEY, DATABASE_NAME

TEST_MODE = False
TEST_TICKER = 'AAPL'

MAX_WORKERS = 2
MIN_DELAY = 0.3
MAX_DELAY = 0.7

MODERN_BASE = "https://financialmodelingprep.com/stable/technical-indicators"

# ------------------------------------------------------------------
# Logging setup — ONLY runs when script is executed directly
# ------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Console output only when run standalone
        ]
    )
else:
    # When imported, use existing root logger (from daily_runner)
    logging.getLogger().setLevel(logging.INFO)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Rest of the script — all prints replaced with log.info/debug
# ------------------------------------------------------------------
def fetch_tickers_from_db():
    today = date.today().isoformat()
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        if TEST_MODE:
            df = pd.read_sql_query("""
                SELECT * FROM stock_consensus 
                WHERE symbol = ?
            """, conn, params=[TEST_TICKER])
            log.info(f"TEST MODE: Scoring {TEST_TICKER}")
        else:
            df = pd.read_sql_query("""
                SELECT * FROM stock_consensus 
                WHERE (scored_at < ? OR scored_at IS NULL)
                  AND ev_ebitda IS NOT NULL
                  AND current_price IS NOT NULL
            """, conn, params=[today])
        conn.close()
        log.info(f"Found {len(df)} stocks ready for scoring")
        return df
    except Exception as e:
        log.error(f"DB error: {e}")
        return pd.DataFrame()

def get_modern_indicator(ticker, indicator, period):
    try:
        params = {
            'symbol': ticker,
            'periodLength': period,
            'timeframe': '1day',
            'limit': 1,
            'apikey': FMP_API_KEY
        }
        url = f"{MODERN_BASE}/{indicator}"
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        if not data or len(data) == 0:
            log.debug(f"No data for {indicator.upper()}({period}) on {ticker}")
            return None
        return data[0]
    except Exception as e:
        log.debug(f"Error fetching {indicator.upper()} for {ticker}: {e}")
        return None

def is_market_bullish():
    try:
        spy_sma_data = get_modern_indicator("SPY", "sma", 200)
        if not spy_sma_data:
            log.debug("No SPY SMA200 data → Market bullish = False")
            return False
        
        spy_close = spy_sma_data.get('close')
        spy_sma200 = spy_sma_data.get('sma')
        if not spy_close or not spy_sma200:
            log.debug("Missing SPY close or SMA200 → Market bullish = False")
            return False
        
        bullish = spy_close > spy_sma200
        log.debug(f"SPY > SMA200: {bullish} ({spy_close:.2f} vs {spy_sma200:.2f}) → Market Bullish: {bullish}")
        return bullish
    except Exception as e:
        log.debug(f"Exception in market check: {e} → Market bullish = False")
        return False

def score_stock(row):
    ticker = row['symbol']
    try:
        log.info(f"\n=== Scoring {ticker} ===")

        sma200_data = get_modern_indicator(ticker, "sma", 200)
        if not sma200_data:
            log.info("   → No SMA200 data — skipping stock")
            return
        
        close = sma200_data['close']
        sma200 = sma200_data['sma']
        sma50_data = get_modern_indicator(ticker, "sma", 50)
        sma50 = sma50_data['sma'] if sma50_data else None
        rsi_data = get_modern_indicator(ticker, "rsi", 14)
        rsi = rsi_data['rsi'] if rsi_data else None
        adx_data = get_modern_indicator(ticker, "adx", 14)
        adx = adx_data['adx'] if adx_data else None

        log.info(f"   → Price: ${close:.2f} | Above SMA200 ({sma200:.2f}) | SMA50: {sma50 or 'N/A'}")
        log.info(f"   → RSI(14): {rsi or 'N/A'} | ADX(14): {adx or 'N/A'}")

        ev_ebitda = row.get('ev_ebitda')
        proj_rev_growth = row.get('projected_revenue_growth')
        proj_eps_growth = row.get('projected_eps_growth')
        proj_ebitda_growth = row.get('projected_ebitda_growth')

        log.info(f"   → EV/EBITDA: {ev_ebitda or 'N/A'}x")
        log.info(f"   → Proj Rev Growth: {proj_rev_growth or 'N/A'}% | Proj EPS Growth: {proj_eps_growth or 'N/A'}%")

        market_bullish = is_market_bullish()

        # Scoring logic unchanged...
        trend_score = 0
        if close > sma200: trend_score += 10
        if sma50 and sma200 and sma50 > sma200: trend_score += 10
        if sma50 and close > sma50: trend_score += 5

        fundamentals_score = 0
        if proj_rev_growth:
            if proj_rev_growth > 15: fundamentals_score += 15
            elif proj_rev_growth > 8: fundamentals_score += 8
        if proj_eps_growth:
            if proj_eps_growth > 15: fundamentals_score += 10
            elif proj_eps_growth > 8: fundamentals_score += 5

        valuation_score = 0
        if ev_ebitda:
            if ev_ebitda < 12: valuation_score += 10
            elif ev_ebitda < 20: valuation_score += 6
            elif ev_ebitda < 30: valuation_score += 3

        momentum_score = 0
        if rsi and 40 <= rsi <= 65: momentum_score += 5
        if adx and adx > 25: momentum_score += 5

        market_risk_score = 10 if market_bullish else 0

        lt_score = trend_score + fundamentals_score + valuation_score + momentum_score + market_risk_score

        value_score = 0
        if ev_ebitda:
            if ev_ebitda < 10: value_score += 30
            elif ev_ebitda < 15: value_score += 20
            elif ev_ebitda < 20: value_score += 10
            elif ev_ebitda < 30: value_score += 5
        if proj_rev_growth:
            if proj_rev_growth > 25: value_score += 30
            elif proj_rev_growth > 15: value_score += 20
            elif proj_rev_growth > 8: value_score += 10
        if proj_eps_growth and proj_eps_growth > 15: value_score += 15
        if proj_ebitda_growth and proj_ebitda_growth > 15: value_score += 10

        log.info(f"\n{ticker}: LT Score {lt_score}/100 | Value Score {value_score}/100")
        log.info(f"   Trend: {trend_score}/25 | Fundamentals: {fundamentals_score}/33 | Valuation: {valuation_score}/16")
        log.info(f"   Momentum: {momentum_score}/10 | Market Risk: {market_risk_score}/10")

        # Save scores
        conn = sqlite3.connect(DATABASE_NAME)
        cur = conn.cursor()
        cur.execute("""
            UPDATE stock_consensus 
            SET long_term_score = ?, value_score = ?, trend_score = ?, fundamentals_score = ?, 
                valuation_score = ?, momentum_score = ?, market_risk_score = ?, market_bullish = ?, scored_at = ?
            WHERE symbol = ?
        """, (lt_score, value_score, trend_score, fundamentals_score, valuation_score,
              momentum_score, market_risk_score, 1 if market_bullish else 0,
              datetime.utcnow().isoformat(), ticker))
        conn.commit()
        conn.close()
        log.info("   → Scores saved to database\n")

    except Exception as e:
        log.error(f"Error scoring {ticker}: {e}")

    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

def main():
    df = fetch_tickers_from_db()
    if df.empty:
        log.info("No stocks need scoring today.")
        return

    log.info(f"\nStarting scoring for {len(df)} stocks...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(score_stock, [row for _, row in df.iterrows()])

    log.info("Scoring complete!")

if __name__ == "__main__":
    main()
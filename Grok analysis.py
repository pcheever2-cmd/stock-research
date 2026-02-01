#!/usr/bin/env python3
# stock_analysis_with_grok.py — Risk-categorized (50 stocks) + Full Portfolio Analysis with Hold/Sell/Buy More

import sqlite3
import pandas as pd
import requests
import logging
from datetime import datetime

from config import DATABASE_NAME, GROK_API_KEY, PORTFOLIO_CSV

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ----------------------------- RISK CATEGORY QUERIES (50 stocks total) -----------------------------
def query_low_risk(limit=10):
    conn = sqlite3.connect(DATABASE_NAME)
    query = """
        SELECT symbol, upside_percent, long_term_score, value_score, num_analysts, cap_category, industry, 
               projected_eps_growth, projected_revenue_growth, ev_ebitda, current_price
        FROM stock_consensus
        WHERE num_analysts >= 10 
          AND cap_category IN ('Large Cap (>$10B)', 'Mid Cap ($2B–$10B)') 
          AND ev_ebitda < 15 
          AND long_term_score > 70 
          AND upside_percent BETWEEN 5 AND 20
        ORDER BY long_term_score DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    return df

def query_moderate_risk(limit=25):
    conn = sqlite3.connect(DATABASE_NAME)
    query = """
        SELECT symbol, upside_percent, long_term_score, value_score, num_analysts, cap_category, industry, 
               projected_eps_growth, projected_revenue_growth, ev_ebitda, current_price
        FROM stock_consensus
        WHERE num_analysts BETWEEN 5 AND 10 
          AND cap_category IN ('Mid Cap ($2B–$10B)', 'Small Cap ($300M–$2B)') 
          AND ev_ebitda BETWEEN 15 AND 25 
          AND long_term_score BETWEEN 50 AND 70 
          AND upside_percent BETWEEN 20 AND 50
        ORDER BY value_score DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    return df

def query_high_risk(limit=10):
    conn = sqlite3.connect(DATABASE_NAME)
    query = """
        SELECT symbol, upside_percent, long_term_score, value_score, num_analysts, cap_category, industry, 
               projected_eps_growth, projected_revenue_growth, ev_ebitda, current_price
        FROM stock_consensus
        WHERE num_analysts BETWEEN 1 AND 5 
          AND cap_category IN ('Small Cap ($300M–$2B)', 'Micro/Penny Cap (<$300M)') 
          AND (ev_ebitda > 25 OR ev_ebitda < 0 OR ev_ebitda IS NULL)
          AND long_term_score BETWEEN 30 AND 50 
          AND upside_percent > 50
        ORDER BY upside_percent DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    return df

def query_moon_shots(limit=5):
    conn = sqlite3.connect(DATABASE_NAME)
    query = """
        SELECT symbol, upside_percent, long_term_score, value_score, num_analysts, cap_category, industry, 
               projected_eps_growth, projected_revenue_growth, ev_ebitda, current_price
        FROM stock_consensus
        WHERE num_analysts <= 3 
          AND cap_category = 'Micro/Penny Cap (<$300M)' 
          AND (ev_ebitda > 50 OR ev_ebitda < -10 OR ev_ebitda IS NULL)
          AND long_term_score < 40 
          AND upside_percent > 100
        ORDER BY upside_percent DESC
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=[limit])
    conn.close()
    return df

def load_portfolio():
    try:
        df = pd.read_csv(PORTFOLIO_CSV)
        df.columns = df.columns.str.strip()
        column_map = {
            'purchase price': 'purchase_price',
            'Price Cost Basis': 'price_cost_basis',
            'Unrealized Gain/Loss $': 'unrealized_gain_loss'
        }
        df = df.rename(columns=column_map)
        logging.info(f"Loaded {len(df)} positions from portfolio.")
        return df
    except FileNotFoundError:
        logging.warning(f"Portfolio file not found at {PORTFOLIO_CSV}. Skipping portfolio review.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading portfolio: {e}")
        return pd.DataFrame()

# ----------------------------- GROK API CALL -----------------------------
def call_grok_api(prompt, max_tokens=3000):
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "grok-4-1-fast-reasoning",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=data, timeout=120)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        logging.error(f"Grok API error {response.status_code}: {response.text}")
        return f"API error (status {response.status_code}). Check your key/plan."

# ----------------------------- MAIN ANALYSIS -----------------------------
if __name__ == "__main__":
    low_risk = query_low_risk(10)
    moderate_risk = query_moderate_risk(25)
    high_risk = query_high_risk(10)
    moon_shots = query_moon_shots(5)
    portfolio = load_portfolio()

    all_stocks = pd.concat([low_risk, moderate_risk, high_risk, moon_shots], ignore_index=True)
    stock_list = ', '.join(all_stocks['symbol'].tolist())

    logging.info(f"Selected {len(all_stocks)} stocks for discovery analysis")

    stocks_summary = all_stocks.to_string(index=False)

    if not portfolio.empty:
        portfolio_display = portfolio.to_string(index=False)
    else:
        portfolio_display = "No portfolio loaded."

    prompt = f"""
Current date: {datetime.today().strftime('%Y-%m-%d')}

Scoring Background:
- Value Score /100: Low EV/EBITDA + high projected growth
- Long-Term Score /100: Trend (SMA), fundamentals, valuation, momentum (RSI/ADX), market risk

=== Discovery Stocks (50 total, categorized by risk) ===
Data (symbol, upside_percent, long_term_score, value_score, num_analysts, cap_category, industry, projected_eps_growth, projected_revenue_growth, ev_ebitda, current_price):
{stocks_summary}

For these discovery stocks, provide grouped analysis by risk category with top picks.

=== Portfolio Review (Apply SAME full analysis to each position) ===
Portfolio positions (columns: symbol, shares, purchase_price, price_cost_basis, unrealized_gain_loss):
{portfolio_display}

For EACH portfolio stock, perform the full analysis:
1. X sentiment (last 24-48h): Overall sentiment, key themes, catalysts, shift, retail vs institutional, hype, momentum prediction
2. Sector trends: Emerging trends, traction, catalysts, KOLs, beneficiaries
3. Viral momentum: Spike? Catalyst, quality, drivers, backing, risk, entry/exit, red flags
4. Red flags: Insider selling, accounting, threats, complaints, legal, downgrades, warnings
5. Beginner explanation: What company does, revenue model, short-term suitability
6. Fundamental summary: Revenue growth, profit/loss, debt, cash flow, conclusion
7. Sector condition: Sector sentiment, short-term price impact
8. Chart reading: Main trend, support/resistance, common beginner mistakes
9. Trading plan: Entry, buy area, stop loss, target (retail trader)

Then, for EACH portfolio stock, give a clear recommendation:
- HOLD (and why)
- SELL (and why — suggest replacement from discovery stocks if better)
- BUY MORE (and why — how much, at what price)

Include unrealized gain/loss in reasoning where relevant.

Be honest, data-driven, and concise.
"""

    logging.info("Sending 50 discovery stocks + full portfolio review to Grok API...")
    analysis = call_grok_api(prompt)
    print("\n=== Grok AI Full Analysis: 50 Discovery Stocks + Portfolio Review (Hold/Sell/Buy More) ===\n")
    print(analysis)
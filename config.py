#!/usr/bin/env python3
"""
Centralized configuration for Stock Research pipeline.
- Loads API keys from environment variables (GitHub Actions) or .env file (local)
- All paths are relative to project root
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if it exists (local development only; ignored in CI)
load_dotenv()

PROJECT_ROOT = Path(__file__).parent

# API Keys
FMP_API_KEY = os.environ.get('FMP_API_KEY', '')
GROK_API_KEY = os.environ.get('GROK_API_KEY', '')

# Paths
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')
CACHE_DIR = PROJECT_ROOT / 'cache'
PARQUET_DIR = PROJECT_ROOT / 'data'
PARQUET_PATH = str(PARQUET_DIR / 'dashboard_data.parquet')
PORTFOLIO_CSV = str(PROJECT_ROOT / 'Portfolio.csv')

# Backtest database (separate from daily pipeline DB to keep it fast)
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Historical data collection settings
HISTORICAL_BATCH_SIZE = 50       # Symbols per batch
HISTORICAL_MAX_CONCURRENT = 10   # Parallel requests
HISTORICAL_CALLS_PER_MINUTE = 700
HISTORICAL_REQUEST_TIMEOUT = 30  # Longer timeout for large responses

# FMP Historical Endpoints
FMP_BASE_URL = 'https://financialmodelingprep.com'
HISTORICAL_ENDPOINTS = {
    'prices': f'{FMP_BASE_URL}/stable/historical-price-eod/full',
    'income': f'{FMP_BASE_URL}/stable/income-statement',
    'balance': f'{FMP_BASE_URL}/stable/balance-sheet-statement',
    'cashflow': f'{FMP_BASE_URL}/stable/cash-flow-statement',
    'metrics': f'{FMP_BASE_URL}/stable/key-metrics',
}

# FMP Analyst Data Endpoints
ANALYST_ENDPOINTS = {
    'grades': f'{FMP_BASE_URL}/stable/grades',
    'analyst_estimates': f'{FMP_BASE_URL}/stable/analyst-estimates',
    'price_target_summary': f'{FMP_BASE_URL}/stable/price-target-summary',
}

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

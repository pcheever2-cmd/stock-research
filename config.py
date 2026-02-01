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

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)

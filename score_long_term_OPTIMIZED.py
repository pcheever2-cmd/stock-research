#!/usr/bin/env python3
"""
OPTIMIZED Long-Term Scoring System
- Async technical indicator fetching
- Batch processing
- Smart caching for market regime
- Estimated time: ~3-5 min for 1000 stocks (vs 15-20 min before)
"""

import asyncio
import aiohttp
import sqlite3
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Optional, Tuple
import json
from pathlib import Path

# ==================== CONFIGURATION ====================
from config import FMP_API_KEY, DATABASE_NAME, CACHE_DIR, BACKTEST_DB
from analyst_accuracy_scorer import calculate_analyst_signal_score

BATCH_SIZE = 30
MAX_CONCURRENT = 10  # Technical indicators are lighter
CALLS_PER_MINUTE = 700  # FMP limit is 750, leave headroom
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

MODERN_BASE = "https://financialmodelingprep.com/stable/technical-indicators"

# Sector-specific EV/EBITDA thresholds (based on empirical percentile analysis)
# "Cheap" for Tech is different from "Cheap" for Energy
SECTOR_VALUATION_THRESHOLDS = {
    'Basic Materials': {'p25': 9.2, 'p50': 13.0, 'p75': 20.5},
    'Communication Services': {'p25': 5.7, 'p50': 10.3, 'p75': 16.1},
    'Consumer Cyclical': {'p25': 8.8, 'p50': 12.0, 'p75': 18.4},
    'Consumer Defensive': {'p25': 9.0, 'p50': 12.5, 'p75': 17.3},
    'Energy': {'p25': 5.5, 'p50': 7.9, 'p75': 11.5},
    'Financial Services': {'p25': 7.6, 'p50': 10.7, 'p75': 15.7},
    'Healthcare': {'p25': 6.5, 'p50': 13.8, 'p75': 22.6},
    'Industrials': {'p25': 10.1, 'p50': 14.7, 'p75': 19.8},
    'Real Estate': {'p25': 11.0, 'p50': 15.0, 'p75': 19.3},
    'Technology': {'p25': 10.7, 'p50': 17.0, 'p75': 29.8},
    'Utilities': {'p25': 10.1, 'p50': 12.2, 'p75': 13.8},
}

# Default thresholds (market-wide median percentiles)
DEFAULT_VALUATION_THRESHOLDS = {'p25': 9.0, 'p50': 13.0, 'p75': 18.0}

# ==================== MOMENTUM & 52-WEEK HIGH CALCULATOR ====================
class MomentumCache:
    """
    Calculate and cache momentum-based factors:
    - 12-1 momentum: +0.0648 correlation, +5.17% Q5-Q1 spread
    - 52-week high proximity: +0.0670 correlation, +4.28% Q5-Q1 spread
    """

    def __init__(self):
        self._momentum_cache = {}
        self._high52w_cache = {}
        self._loaded = False

    def load_all_data(self):
        """Pre-load momentum and 52-week high data for all stocks from backtest.db"""
        if self._loaded:
            return

        try:
            conn = sqlite3.connect(BACKTEST_DB)
            # Get last 13 months of prices
            cutoff = (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d')
            prices = pd.read_sql_query(f"""
                SELECT symbol, date, adjusted_close
                FROM historical_prices
                WHERE date >= '{cutoff}'
                ORDER BY symbol, date
            """, conn)
            conn.close()

            # Calculate momentum and 52-week high for each symbol
            for symbol in prices['symbol'].unique():
                sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
                if len(sym_prices) < 252:  # Need ~1 year
                    continue

                closes = sym_prices['adjusted_close'].values

                # 12-1 momentum: price 1 month ago / price 12 months ago
                if len(closes) >= 252 and closes[-252] > 0 and closes[-21] > 0:
                    momentum = ((closes[-21] / closes[-252]) - 1) * 100
                    self._momentum_cache[symbol] = max(-80, min(momentum, 200))  # Cap extremes

                # 52-week high proximity: current price / 52-week high
                high_52w = max(closes[-252:])
                current_price = closes[-1]
                if high_52w > 0:
                    pct_from_high = ((current_price / high_52w) - 1) * 100
                    self._high52w_cache[symbol] = max(-80, min(pct_from_high, 0))  # Always <= 0

            self._loaded = True
            log.info(f"Loaded momentum for {len(self._momentum_cache)} stocks, "
                     f"52w-high for {len(self._high52w_cache)} stocks")

        except Exception as e:
            log.warning(f"Could not load momentum data: {e}")
            self._loaded = True  # Don't retry

    def get_momentum(self, symbol: str) -> Optional[float]:
        """Get 12-1 momentum for a symbol (returns None if not available)"""
        if not self._loaded:
            self.load_all_data()
        return self._momentum_cache.get(symbol)

    def get_high52w_pct(self, symbol: str) -> Optional[float]:
        """Get % from 52-week high (0 = at high, -50 = 50% below high)"""
        if not self._loaded:
            self.load_all_data()
        return self._high52w_cache.get(symbol)

    def get_momentum_score(self, symbol: str) -> int:
        """
        Convert momentum to score (0-25 points).
        Based on backtest quintiles:
        - Q1 (losers, <-30%): 0 points
        - Q2 (-30% to -10%): 5 points
        - Q3 (-10% to +10%): 10 points
        - Q4 (+10% to +40%): 18 points
        - Q5 (winners, >+40%): 25 points
        """
        momentum = self.get_momentum(symbol)
        if momentum is None:
            return 10  # Default to neutral if no data

        if momentum > 40:
            return 25  # Strong winners
        elif momentum > 10:
            return 18  # Moderate winners
        elif momentum > -10:
            return 10  # Neutral
        elif momentum > -30:
            return 5   # Moderate losers
        else:
            return 0   # Strong losers

    def get_high52w_score(self, symbol: str) -> int:
        """
        Convert 52-week high proximity to score (0-10 points).
        Based on backtest: stocks near highs continue to outperform.
        - At/near high (>-5%): 10 points
        - Within 15% of high (-15% to -5%): 7 points
        - Within 30% (-30% to -15%): 4 points
        - More than 30% below: 0 points
        """
        pct = self.get_high52w_pct(symbol)
        if pct is None:
            return 5  # Default to neutral if no data

        if pct > -5:
            return 10  # Near 52-week high
        elif pct > -15:
            return 7   # Within 15%
        elif pct > -30:
            return 4   # Within 30%
        else:
            return 0   # Far from high

momentum_cache = MomentumCache()


# ==================== GROSS PROFITABILITY CALCULATOR ====================
class GrossProfitabilityCache:
    """
    Calculate and cache Gross Profitability (GP/Assets) factor.
    Academic source: Novy-Marx 2013 "The Other Side of Value"

    Backtest results:
    - Correlation: +0.035
    - Combined with momentum: improves spread from +5.94% to +6.42%
    - Optimal weight: ~35% of factor mix
    """

    def __init__(self):
        self._gp_cache = {}
        self._loaded = False

    def load_all_data(self):
        """Pre-load gross profitability for all stocks from backtest.db"""
        if self._loaded:
            return

        try:
            conn = sqlite3.connect(BACKTEST_DB)

            # Load most recent annual income statements (Q4 = full year)
            income = pd.read_sql_query("""
                SELECT symbol, fiscal_year, gross_profit
                FROM historical_income_statements
                WHERE period = 'Q4'
                ORDER BY symbol, fiscal_year DESC
            """, conn)

            # Load most recent balance sheets
            balance = pd.read_sql_query("""
                SELECT symbol, fiscal_year, total_assets
                FROM historical_balance_sheets
                WHERE period = 'Q4'
                ORDER BY symbol, fiscal_year DESC
            """, conn)

            conn.close()

            # Get most recent year for each symbol
            income_latest = income.groupby('symbol').first().reset_index()
            balance_latest = balance.groupby('symbol').first().reset_index()

            # Merge and calculate GP/Assets
            merged = income_latest.merge(balance_latest[['symbol', 'total_assets']],
                                         on='symbol', how='inner')

            for _, row in merged.iterrows():
                symbol = row['symbol']
                gp = row['gross_profit']
                assets = row['total_assets']

                if pd.notna(gp) and pd.notna(assets) and assets > 0:
                    gp_ratio = gp / assets
                    # Cap at reasonable range (-0.2 to 0.5)
                    self._gp_cache[symbol] = max(-0.2, min(gp_ratio, 0.5))

            self._loaded = True
            log.info(f"Loaded gross profitability for {len(self._gp_cache)} stocks")

        except Exception as e:
            log.warning(f"Could not load gross profitability data: {e}")
            self._loaded = True  # Don't retry

    def get_gross_profitability(self, symbol: str) -> Optional[float]:
        """Get gross profitability ratio (GP/Assets) for a symbol"""
        if not self._loaded:
            self.load_all_data()
        return self._gp_cache.get(symbol)

    def get_gross_profitability_score(self, symbol: str) -> int:
        """
        Convert gross profitability to score (0-15 points).
        Based on backtest quintiles:
        - Q1 (lowest GP, <0.01): 0 points
        - Q2 (0.01-0.03): 4 points
        - Q3 (0.03-0.07): 8 points
        - Q4 (0.07-0.12): 12 points
        - Q5 (highest GP, >0.12): 15 points
        """
        gp = self.get_gross_profitability(symbol)
        if gp is None:
            return 7  # Default to neutral if no data

        if gp > 0.12:
            return 15  # Highly profitable
        elif gp > 0.07:
            return 12  # Good profitability
        elif gp > 0.03:
            return 8   # Moderate profitability
        elif gp > 0.01:
            return 4   # Low profitability
        else:
            return 0   # Very low/negative profitability


gross_profitability_cache = GrossProfitabilityCache()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# ==================== MARKET REGIME CACHE ====================
class MarketRegimeCache:
    """Cache market regime (SPY check) - only changes daily"""
    
    def __init__(self):
        self.cache_file = CACHE_DIR / 'market_regime.json'
        self.cache_duration = timedelta(hours=4)  # Refresh every 4 hours
    
    def get(self) -> Optional[bool]:
        if not self.cache_file.exists():
            return None
        
        with open(self.cache_file, 'r') as f:
            data = json.load(f)
        
        cached_time = datetime.fromisoformat(data['timestamp'])
        if datetime.now() - cached_time > self.cache_duration:
            return None
        
        return data['is_bullish']
    
    def set(self, is_bullish: bool):
        with open(self.cache_file, 'w') as f:
            json.dump({
                'is_bullish': is_bullish,
                'timestamp': datetime.now().isoformat()
            }, f)

market_cache = MarketRegimeCache()

# ==================== RATE LIMITER ====================
class RateLimiter:
    """Token bucket rate limiter for API calls"""

    def __init__(self, calls_per_minute: int):
        self.min_interval = 60.0 / calls_per_minute
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last_call + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call = asyncio.get_event_loop().time()

# ==================== ASYNC FETCHER ====================
class AsyncIndicatorFetcher:

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.rate_limiter = RateLimiter(CALLS_PER_MINUTE)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_indicator(self, ticker: str, indicator: str, period: int, retry_count: int = 0) -> Optional[dict]:
        """Fetch single technical indicator with retry logic - releases semaphore before retrying"""
        params = {
            'symbol': ticker,
            'periodLength': period,
            'timeframe': '1day',
            'limit': 1,
            'apikey': self.api_key
        }
        url = f"{MODERN_BASE}/{indicator}"

        should_retry = False
        await self.rate_limiter.acquire()
        async with self.semaphore:
            try:
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data[0] if data else None
                    elif response.status == 429 and retry_count < MAX_RETRIES:
                        should_retry = True
                    else:
                        return None
            except asyncio.TimeoutError:
                if retry_count < MAX_RETRIES:
                    log.debug(f"Timeout on {indicator} for {ticker}, retrying...")
                    should_retry = True
                else:
                    return None
            except Exception as e:
                log.debug(f"Error fetching {indicator} for {ticker}: {e}")
                return None

        # Retry OUTSIDE the semaphore to avoid deadlock
        if should_retry:
            await asyncio.sleep(2 ** retry_count)
            return await self.fetch_indicator(ticker, indicator, period, retry_count + 1)
        return None
    
    async def fetch_all_indicators(self, ticker: str) -> Dict[str, Optional[dict]]:
        """Fetch all indicators for one stock in parallel"""
        tasks = {
            'sma200': self.fetch_indicator(ticker, 'sma', 200),
            'sma50': self.fetch_indicator(ticker, 'sma', 50),
            'rsi': self.fetch_indicator(ticker, 'rsi', 14),
            'adx': self.fetch_indicator(ticker, 'adx', 14),
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        return dict(zip(tasks.keys(), results))

# ==================== MARKET REGIME CHECK ====================
async def check_market_regime(fetcher: AsyncIndicatorFetcher) -> bool:
    """Check if market is bullish (SPY > SMA200) - cached"""
    
    # Check cache first
    cached = market_cache.get()
    if cached is not None:
        log.info(f"Using cached market regime: {'Bullish' if cached else 'Bearish'}")
        return cached
    
    log.info("Checking market regime (SPY > SMA200)...")
    spy_sma200 = await fetcher.fetch_indicator('SPY', 'sma', 200)
    
    if not spy_sma200:
        log.warning("Could not fetch SPY data - assuming bearish")
        is_bullish = False
    else:
        spy_close = spy_sma200.get('close')
        spy_sma = spy_sma200.get('sma')
        is_bullish = spy_close > spy_sma if (spy_close and spy_sma) else False
        log.info(f"Market regime: {'Bullish' if is_bullish else 'Bearish'} "
                f"(SPY ${spy_close:.2f} vs SMA200 ${spy_sma:.2f})")
    
    # Cache result
    market_cache.set(is_bullish)
    return is_bullish

# ==================== SCORING LOGIC ====================
def calculate_scores(row: pd.Series, indicators: Dict, market_bullish: bool,
                     sector: str = None):
    """Calculate all scores for one stock using sector-relative valuation."""

    # Extract indicator values
    sma200_data = indicators.get('sma200')
    sma50_data = indicators.get('sma50')
    rsi_data = indicators.get('rsi')
    adx_data = indicators.get('adx')

    if not sma200_data:
        return (0, 0, 0, 0, 0, 0, 0, 0), {}  # Can't score without price data

    close = sma200_data.get('close', 0)
    sma200 = sma200_data.get('sma', 0)
    sma50 = sma50_data.get('sma') if sma50_data else None
    rsi = rsi_data.get('rsi') if rsi_data else None
    adx = adx_data.get('adx') if adx_data else None

    # Fundamental data from database
    ev_ebitda = row.get('ev_ebitda')
    proj_rev_growth = row.get('projected_revenue_growth')
    proj_eps_growth = row.get('projected_eps_growth')
    proj_ebitda_growth = row.get('projected_ebitda_growth')

    # Get sector-specific valuation thresholds
    val_thresholds = SECTOR_VALUATION_THRESHOLDS.get(sector, DEFAULT_VALUATION_THRESHOLDS)
    
    # === TREND SCORE (max 25) ===
    trend_score = 0
    if close > sma200:
        trend_score += 10
    if sma50 and sma200 and sma50 > sma200:
        trend_score += 10
    if sma50 and close > sma50:
        trend_score += 5
    
    # === FUNDAMENTALS SCORE (max 25) ===
    fundamentals_score = 0
    if proj_rev_growth:
        if proj_rev_growth > 15:
            fundamentals_score += 15
        elif proj_rev_growth > 8:
            fundamentals_score += 8
    if proj_eps_growth:
        if proj_eps_growth > 15:
            fundamentals_score += 10
        elif proj_eps_growth > 8:
            fundamentals_score += 5
    
    # === VALUATION SCORE (max 16) - SECTOR-RELATIVE ===
    # Uses sector percentiles: below P25 = cheap, P25-P75 = fair, above P75 = expensive
    valuation_score = 0
    if ev_ebitda and ev_ebitda > 0:
        p25, p50, p75 = val_thresholds['p25'], val_thresholds['p50'], val_thresholds['p75']
        if ev_ebitda < p25:
            valuation_score += 10  # Cheap for this sector
        elif ev_ebitda < p50:
            valuation_score += 6   # Below median
        elif ev_ebitda < p75:
            valuation_score += 3   # Fair
        # Above P75 = expensive, no points
    
    # === MOMENTUM SCORE (max 10) — V2 tweaks: RSI 40-55 (was 40-65), ADX > 20 (was > 25) ===
    momentum_score = 0
    if rsi and 40 <= rsi <= 55:
        momentum_score += 5
    if adx and adx > 20:
        momentum_score += 5
    
    # === MARKET RISK SCORE (max 10) ===
    market_risk_score = 10 if market_bullish else 0
    
    # === LONG-TERM SCORE (total) ===
    lt_score = trend_score + fundamentals_score + valuation_score + momentum_score + market_risk_score
    
    # === VALUE SCORE (max 100) - SECTOR-RELATIVE ===
    value_score = 0
    if ev_ebitda and ev_ebitda > 0:
        p25, p50, p75 = val_thresholds['p25'], val_thresholds['p50'], val_thresholds['p75']
        if ev_ebitda < p25 * 0.8:        # Very cheap (bottom ~10%)
            value_score += 30
        elif ev_ebitda < p25:            # Cheap (bottom ~25%)
            value_score += 20
        elif ev_ebitda < p50:            # Below median
            value_score += 10
        elif ev_ebitda < p75:            # Fair
            value_score += 5
    
    if proj_rev_growth:
        if proj_rev_growth > 25:
            value_score += 30
        elif proj_rev_growth > 15:
            value_score += 20
        elif proj_rev_growth > 8:
            value_score += 10
    
    if proj_eps_growth and proj_eps_growth > 15:
        value_score += 15

    if proj_ebitda_growth and proj_ebitda_growth > 15:
        value_score += 10

    # === VALUE SCORE V2 (continuous, max 100) - SECTOR-RELATIVE ===
    # Valuation component (max 40, min -10)
    if ev_ebitda is not None:
        p25, p50, p75 = val_thresholds['p25'], val_thresholds['p50'], val_thresholds['p75']
        if ev_ebitda < 0:
            v2_val = -10  # Negative EV/EBITDA is a red flag
        elif ev_ebitda < p25 * 0.7:  # Very cheap (bottom ~10%)
            v2_val = 40
        elif ev_ebitda < p25:  # Cheap (bottom ~25%)
            v2_val = 30
        elif ev_ebitda < p50:  # Below median
            v2_val = 20
        elif ev_ebitda < p75:  # Fair
            v2_val = 10
        else:
            v2_val = 0  # Expensive (above P75)
    else:
        v2_val = 0

    # Revenue Growth component (max 25)
    rev = proj_rev_growth or 0
    rev_capped = min(rev, 50)
    v2_rev = max(0, min(rev_capped / 2, 25))
    if rev > 60:
        v2_rev *= 0.7

    # EPS Growth component (max 20, min -5)
    eps = proj_eps_growth or 0
    v2_eps = max(-5, min(eps / 2, 20))

    # Quality component (max 15) - SECTOR-RELATIVE
    ebitda_g = proj_ebitda_growth or 0
    v2_quality = 10.0 if ebitda_g > 10 else 0.0
    # Reward reasonable valuation (below P75 for sector)
    if ev_ebitda is not None and 0 < ev_ebitda <= val_thresholds['p75']:
        v2_quality += 5.0

    value_score_v2 = int(max(0, min(v2_val + v2_rev + v2_eps + v2_quality, 100)))

    raw_indicators = {
        'close': close, 'sma50': sma50, 'sma200': sma200,
        'rsi': rsi, 'adx': adx,
    }

    return (lt_score, value_score, value_score_v2, trend_score, fundamentals_score,
            valuation_score, momentum_score, market_risk_score), raw_indicators

# ==================== TREND SIGNAL DETECTION ====================
def detect_trend_signals(current: dict, previous: dict) -> Tuple[Optional[str], int]:
    """
    Compare current vs previous indicator values to detect trend transitions.
    Returns (comma-separated signal string, count of active signals).
    """
    signals = []

    curr_sma50 = current.get('sma50')
    curr_sma200 = current.get('sma200')
    prev_sma50 = previous.get('sma50')
    prev_sma200 = previous.get('sma200')
    curr_close = current.get('close')
    prev_close = previous.get('close_price_technical')
    curr_rsi = current.get('rsi')
    prev_rsi = previous.get('rsi')

    # 1. Golden Cross: SMA50 crosses above SMA200
    if (curr_sma50 and curr_sma200 and prev_sma50 and prev_sma200 and
            curr_sma50 > curr_sma200 and prev_sma50 <= prev_sma200):
        signals.append("Golden Cross")

    # 2. Price breaks above SMA50
    if (curr_close and curr_sma50 and prev_close and prev_sma50 and
            curr_close > curr_sma50 and prev_close <= prev_sma50):
        signals.append("Price > SMA50")

    # 3. RSI recovery from oversold
    if (curr_rsi and prev_rsi and prev_rsi < 30 and curr_rsi >= 40):
        signals.append("RSI Recovery")

    # 4. Bullish alignment (current-state flag, not crossover)
    if (curr_close and curr_sma50 and curr_sma200 and
            curr_close > curr_sma50 > curr_sma200):
        signals.append("Bullish Aligned")

    signal_str = ", ".join(signals) if signals else None
    return signal_str, len(signals)

# ==================== MAIN PROCESSING ====================
async def score_stock(row: pd.Series, fetcher: AsyncIndicatorFetcher, market_bullish: bool) -> Optional[Dict]:
    """Score one stock"""
    ticker = row['symbol']

    try:
        # Fetch all indicators in parallel
        indicators = await fetcher.fetch_all_indicators(ticker)

        # Get sector for sector-relative valuation
        sector = row.get('sector')

        # Calculate scores + raw indicators (using sector-relative valuation)
        scores, raw = calculate_scores(row, indicators, market_bullish, sector=sector)

        if scores[0] == 0:  # No valid data
            log.debug(f"  {ticker}: No technical data")
            return None

        lt_score, value_score, value_score_v2, trend, fundamentals, valuation, momentum, market_risk = scores

        # === VALUE SCORE V3 (max 100) ===
        # Multi-factor score with backtested components:
        #
        # Factor correlations (from backtest):
        # - 12-1 Momentum:       +0.066 corr (STRONGEST price factor)
        # - 52-Week High:        +0.075 corr (second strongest)
        # - Gross Profitability: +0.035 corr (Novy-Marx 2013 - complements momentum)
        # - Trend Score:         +0.055 corr (price above SMAs)
        # - Fundamentals:        +0.051 corr (revenue/EPS growth)
        # - Valuation:           -0.071 corr (NEGATIVE! reduced weight)
        # - Analyst Signal:      +0.024 corr (top 3 per sector)
        #
        # V3 = Momentum (25) + High52w (10) + GP (15) + Trend (15) + Fundamentals (15) + Valuation (5) + Analyst (10) + Quality (5)
        #
        analyst_signal = calculate_analyst_signal_score(ticker, sector)
        analyst_points = analyst_signal['analyst_signal_score']  # -10 to +15

        # Get 12-1 momentum score (0-25 points) - STRONGEST factor
        momentum_12_1_score = momentum_cache.get_momentum_score(ticker)

        # Get 52-week high proximity score (0-10 points) - second strongest factor
        high52w_score = momentum_cache.get_high52w_score(ticker)

        # Get gross profitability score (0-15 points) - NEW academic factor
        gp_score = gross_profitability_cache.get_gross_profitability_score(ticker)

        # Quality score (max 5) - reduced to make room for GP
        proj_ebitda_growth = row.get('projected_ebitda_growth') or 0
        ev_ebitda = row.get('ev_ebitda')
        val_thresholds = SECTOR_VALUATION_THRESHOLDS.get(sector, DEFAULT_VALUATION_THRESHOLDS)

        v3_quality = 3 if proj_ebitda_growth > 10 else 0
        if ev_ebitda is not None and 0 < ev_ebitda <= val_thresholds['p75']:
            v3_quality += 2
        v3_quality = min(v3_quality, 5)

        # Valuation score (sector-relative, max 5) - UPDATED based on deep dive analysis:
        # - Extreme cheapness (<8x) is a VALUE TRAP signal, not a buy signal
        # - Sweet spot is 15-25x (D2/D3 in backtest had best returns: +3.9%)
        # - Very expensive (>50x) slightly underperforms
        if ev_ebitda is not None and ev_ebitda > 0:
            p25, p50, p75 = val_thresholds['p25'], val_thresholds['p50'], val_thresholds['p75']
            if ev_ebitda < 8:
                v3_valuation = 0   # VALUE TRAP - extreme cheapness is distress signal
            elif ev_ebitda < p50:
                v3_valuation = 5   # Sweet spot - moderately cheap
            elif ev_ebitda < p75:
                v3_valuation = 3   # Fair value
            else:
                v3_valuation = 0   # Expensive
        else:
            v3_valuation = 0

        # Scale trend and fundamentals from LT score (max 25 each) to max 15 each
        v3_trend = int(trend * 15 / 25)           # Scale 0-25 → 0-15
        v3_fundamentals = int(fundamentals * 15 / 25)  # Scale 0-25 → 0-15

        # Scale analyst signal from [-10, +15] to [-5, +10]
        v3_analyst = int(analyst_points * 10 / 15) if analyst_points > 0 else int(analyst_points * 5 / 10)

        # V3 total (max 100)
        # Weights: Mom(25) + 52wH(10) + GP(15) + Trend(15) + Fund(15) + Val(5) + Analyst(10) + Quality(5) = 100
        value_score_v3 = int(min(100, max(0,
            momentum_12_1_score +  # max 25 (STRONGEST - 12-month momentum)
            high52w_score +        # max 10 (52-week high proximity)
            gp_score +             # max 15 (gross profitability - Novy-Marx)
            v3_trend +             # max 15 (price above SMAs)
            v3_fundamentals +      # max 15 (revenue/EPS growth)
            v3_valuation +         # max 5 (sector-relative EV/EBITDA - reduced due to neg corr)
            v3_analyst +           # max 10, min -5 (top 3 analyst signal)
            v3_quality             # max 5 (EBITDA growth + reasonable valuation)
        )))

        # Detect trend signals by comparing current indicators to previous (stored in DB)
        previous = {
            'sma50': row.get('sma50'),
            'sma200': row.get('sma200'),
            'close_price_technical': row.get('close_price_technical'),
            'rsi': row.get('rsi'),
        }
        trend_signal, trend_signal_count = detect_trend_signals(raw, previous)

        momentum_info = f" | Mom {momentum_12_1_score:>2}/25" if momentum_12_1_score != 10 else ""
        high52w_info = f" | 52wH {high52w_score:>2}/10" if high52w_score != 5 else ""
        gp_info = f" | GP {gp_score:>2}/15" if gp_score != 7 else ""
        analyst_info = f" | Analyst {v3_analyst:+d}" if v3_analyst != 0 else ""
        signal_info = f" | Signals: {trend_signal}" if trend_signal else ""
        log.info(f"{ticker:<6} | LT {lt_score:>3} | V2 {value_score_v2:>3} | V3 {value_score_v3:>3} | "
                f"Trend {v3_trend:>2}/15 | Fund {v3_fundamentals:>2}/15{momentum_info}{high52w_info}{gp_info}{analyst_info}{signal_info}")

        return {
            'symbol': ticker,
            'long_term_score': lt_score,
            'value_score': value_score,
            'value_score_v2': value_score_v2,
            'value_score_v3': value_score_v3,
            'trend_score': trend,
            'fundamentals_score': fundamentals,
            'valuation_score': valuation,
            'momentum_score': momentum,
            'momentum_12_1': momentum_cache.get_momentum(ticker),  # Raw 12-1 momentum %
            'momentum_12_1_score': momentum_12_1_score,  # 0-25 score
            'high52w_pct': momentum_cache.get_high52w_pct(ticker),  # % from 52-week high
            'high52w_score': high52w_score,  # 0-10 score
            'gross_profitability': gross_profitability_cache.get_gross_profitability(ticker),  # GP/Assets ratio
            'gross_profitability_score': gp_score,  # 0-15 score
            'market_risk_score': market_risk,
            'analyst_signal_score': analyst_points,
            'market_bullish': 1 if market_bullish else 0,
            'scored_at': datetime.utcnow().isoformat(),
            # Raw indicators (current values)
            'sma50': raw.get('sma50'),
            'sma200': raw.get('sma200'),
            'rsi': raw.get('rsi'),
            'adx': raw.get('adx'),
            'close_price_technical': raw.get('close'),
            # Previous values (rotate old current -> prev)
            'prev_sma50': row.get('sma50'),
            'prev_sma200': row.get('sma200'),
            'prev_rsi': row.get('rsi'),
            'prev_close_technical': row.get('close_price_technical'),
            # Trend signals
            'trend_signal': trend_signal,
            'trend_signal_count': trend_signal_count,
        }

    except Exception as e:
        log.error(f"Error scoring {ticker}: {e}")
        return None

def save_scores_batch(results: list):
    """Bulk update scores in database"""
    if not results:
        return

    conn = sqlite3.connect(DATABASE_NAME)
    cur = conn.cursor()

    # Ensure new columns exist
    new_columns = [
        ("value_score_v3", "INTEGER"),
        ("analyst_signal_score", "REAL"),
        ("momentum_12_1", "REAL"),
        ("momentum_12_1_score", "INTEGER"),
        ("high52w_pct", "REAL"),
        ("high52w_score", "INTEGER"),
        ("gross_profitability", "REAL"),
        ("gross_profitability_score", "INTEGER"),
    ]
    for col_name, col_type in new_columns:
        try:
            cur.execute(f"ALTER TABLE stock_consensus ADD COLUMN {col_name} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists

    for row in results:
        cur.execute("""
            UPDATE stock_consensus
            SET long_term_score = ?, value_score = ?, value_score_v2 = ?, value_score_v3 = ?,
                trend_score = ?,
                fundamentals_score = ?, valuation_score = ?, momentum_score = ?,
                momentum_12_1 = ?, momentum_12_1_score = ?,
                high52w_pct = ?, high52w_score = ?,
                gross_profitability = ?, gross_profitability_score = ?,
                market_risk_score = ?, analyst_signal_score = ?, market_bullish = ?, scored_at = ?,
                sma50 = ?, sma200 = ?, rsi = ?, adx = ?, close_price_technical = ?,
                prev_sma50 = ?, prev_sma200 = ?, prev_rsi = ?, prev_close_technical = ?,
                trend_signal = ?, trend_signal_count = ?
            WHERE symbol = ?
        """, (
            row['long_term_score'], row['value_score'], row['value_score_v2'], row['value_score_v3'],
            row['trend_score'],
            row['fundamentals_score'], row['valuation_score'], row['momentum_score'],
            row.get('momentum_12_1'), row.get('momentum_12_1_score'),
            row.get('high52w_pct'), row.get('high52w_score'),
            row.get('gross_profitability'), row.get('gross_profitability_score'),
            row['market_risk_score'], row['analyst_signal_score'], row['market_bullish'], row['scored_at'],
            row['sma50'], row['sma200'], row['rsi'], row['adx'], row['close_price_technical'],
            row['prev_sma50'], row['prev_sma200'], row['prev_rsi'], row['prev_close_technical'],
            row['trend_signal'], row['trend_signal_count'],
            row['symbol']
        ))

    conn.commit()
    conn.close()
    log.info(f"  ✓ Saved scores for {len(results)} stocks")

# ==================== MAIN ====================
async def main():
    log.info("=" * 60)
    log.info("OPTIMIZED Long-Term Scoring System")
    log.info("=" * 60)
    
    # 1. Fetch stocks needing scoring
    today = date.today().isoformat()
    conn = sqlite3.connect(DATABASE_NAME)
    df = pd.read_sql_query("""
        SELECT * FROM stock_consensus 
        WHERE (scored_at < ? OR scored_at IS NULL)
          AND ev_ebitda IS NOT NULL
          AND current_price IS NOT NULL
    """, conn, params=[today])
    conn.close()
    
    if df.empty:
        log.info("✓ All stocks already scored today!")
        return
    
    log.info(f"Found {len(df)} stocks to score")
    
    # 2. Check market regime once
    async with AsyncIndicatorFetcher(FMP_API_KEY) as fetcher:
        market_bullish = await check_market_regime(fetcher)
        
        # 3. Process in batches
        total_scored = 0
        
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i + BATCH_SIZE]
            log.info(f"\n--- Batch {i//BATCH_SIZE + 1}/{(len(df)-1)//BATCH_SIZE + 1} "
                    f"({len(batch_df)} stocks) ---")
            
            # Score batch in parallel
            tasks = [
                score_stock(row, fetcher, market_bullish)
                for _, row in batch_df.iterrows()
            ]
            results = await asyncio.gather(*tasks)
            
            # Filter and save
            valid_results = [r for r in results if r is not None]
            save_scores_batch(valid_results)
            
            total_scored += len(valid_results)
            log.info(f"Progress: {total_scored}/{len(df)} stocks scored")
    
    log.info("\n" + "=" * 60)
    log.info(f"✓ Scoring complete! {total_scored}/{len(df)} stocks updated")
    log.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
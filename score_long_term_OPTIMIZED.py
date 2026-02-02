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
from config import FMP_API_KEY, DATABASE_NAME, CACHE_DIR

BATCH_SIZE = 30
MAX_CONCURRENT = 10  # Technical indicators are lighter
CALLS_PER_MINUTE = 700  # FMP limit is 750, leave headroom
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

MODERN_BASE = "https://financialmodelingprep.com/stable/technical-indicators"

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
def calculate_scores(row: pd.Series, indicators: Dict, market_bullish: bool):
    """Calculate all scores for one stock"""
    
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
    
    # === VALUATION SCORE (max 16) ===
    valuation_score = 0
    if ev_ebitda:
        if ev_ebitda < 12:
            valuation_score += 10
        elif ev_ebitda < 20:
            valuation_score += 6
        elif ev_ebitda < 30:
            valuation_score += 3
    
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
    
    # === VALUE SCORE (max 100) ===
    value_score = 0
    if ev_ebitda:
        if ev_ebitda < 10:
            value_score += 30
        elif ev_ebitda < 15:
            value_score += 20
        elif ev_ebitda < 20:
            value_score += 10
        elif ev_ebitda < 30:
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

    # === VALUE SCORE V2 (continuous, max 100) ===
    # Valuation component (max 40, min -10)
    if ev_ebitda is not None:
        if ev_ebitda < 0:
            v2_val = -10
        elif ev_ebitda < 8:
            v2_val = 40
        elif ev_ebitda < 12:
            v2_val = 30
        elif ev_ebitda < 16:
            v2_val = 20
        elif ev_ebitda < 22:
            v2_val = 10
        else:
            v2_val = 0
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

    # Quality component (max 15)
    ebitda_g = proj_ebitda_growth or 0
    v2_quality = 10.0 if ebitda_g > 10 else 0.0
    if ev_ebitda is not None and 0 < ev_ebitda <= 25:
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
        
        # Calculate scores + raw indicators
        scores, raw = calculate_scores(row, indicators, market_bullish)

        if scores[0] == 0:  # No valid data
            log.debug(f"  {ticker}: No technical data")
            return None

        lt_score, value_score, value_score_v2, trend, fundamentals, valuation, momentum, market_risk = scores

        # Detect trend signals by comparing current indicators to previous (stored in DB)
        previous = {
            'sma50': row.get('sma50'),
            'sma200': row.get('sma200'),
            'close_price_technical': row.get('close_price_technical'),
            'rsi': row.get('rsi'),
        }
        trend_signal, trend_signal_count = detect_trend_signals(raw, previous)

        signal_info = f" | Signals: {trend_signal}" if trend_signal else ""
        log.info(f"{ticker:<6} | LT {lt_score:>3}/100 | V1 {value_score:>3} | V2 {value_score_v2:>3} | "
                f"Trend {trend:>2}/25 | Fund {fundamentals:>2}/25 | Val {valuation:>2}/16{signal_info}")

        return {
            'symbol': ticker,
            'long_term_score': lt_score,
            'value_score': value_score,
            'value_score_v2': value_score_v2,
            'trend_score': trend,
            'fundamentals_score': fundamentals,
            'valuation_score': valuation,
            'momentum_score': momentum,
            'market_risk_score': market_risk,
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
    
    for row in results:
        cur.execute("""
            UPDATE stock_consensus
            SET long_term_score = ?, value_score = ?, value_score_v2 = ?,
                trend_score = ?,
                fundamentals_score = ?, valuation_score = ?, momentum_score = ?,
                market_risk_score = ?, market_bullish = ?, scored_at = ?,
                sma50 = ?, sma200 = ?, rsi = ?, adx = ?, close_price_technical = ?,
                prev_sma50 = ?, prev_sma200 = ?, prev_rsi = ?, prev_close_technical = ?,
                trend_signal = ?, trend_signal_count = ?
            WHERE symbol = ?
        """, (
            row['long_term_score'], row['value_score'], row['value_score_v2'],
            row['trend_score'],
            row['fundamentals_score'], row['valuation_score'], row['momentum_score'],
            row['market_risk_score'], row['market_bullish'], row['scored_at'],
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
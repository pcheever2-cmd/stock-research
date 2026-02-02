#!/usr/bin/env python3
"""
Backtesting Engine - Replays the scoring system across 5 years of historical data.
Identifies buy/sell signals and calculates forward returns.
All computation is local (no API calls) — uses data in backtest.db.

Usage:
    python run_backtest.py                          # Full backtest
    python run_backtest.py --symbols AAPL,MSFT      # Specific symbols
    python run_backtest.py --summary                # Show results summary
"""

import sqlite3
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB
from setup_database import setup_backtest_tables

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

BATCH_SIZE = 100  # Stocks per batch


# ==================== TECHNICAL INDICATORS ====================

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI calculation"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() /
                      atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() /
                       atr.replace(0, np.nan))

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return adx


def compute_technical_indicators(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA50, SMA200, RSI, ADX columns to a price DataFrame"""
    df = prices_df.sort_values('date').copy()
    df['sma50'] = df['adjusted_close'].rolling(50, min_periods=50).mean()
    df['sma200'] = df['adjusted_close'].rolling(200, min_periods=200).mean()
    df['rsi'] = compute_rsi(df['adjusted_close'], 14)
    df['adx'] = compute_adx(df['high'], df['low'], df['adjusted_close'], 14)
    return df


# ==================== MARKET REGIME ====================

def compute_spy_regime(conn: sqlite3.Connection) -> pd.DataFrame:
    """Compute daily SPY > SMA200 market regime"""
    spy = pd.read_sql_query(
        "SELECT date, adjusted_close, high, low FROM historical_prices WHERE symbol = 'SPY' ORDER BY date",
        conn
    )
    if spy.empty:
        log.error("No SPY data found in backtest.db")
        return pd.DataFrame(columns=['date', 'market_bullish'])

    spy['sma200'] = spy['adjusted_close'].rolling(200, min_periods=200).mean()
    spy['market_bullish'] = (spy['adjusted_close'] > spy['sma200']).astype(int)
    return spy[['date', 'market_bullish']].dropna()


# ==================== TTM FUNDAMENTALS ====================

def compute_ttm_fundamentals(symbol: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Compute TTM (trailing twelve months) revenue, EPS, EBITDA growth rates
    and EV/EBITDA at each filing date. Uses filing_date to prevent look-ahead bias.
    Returns a DataFrame with one row per filing_date, forward-fillable to daily.
    """
    # Load income statements sorted by filing date
    income = pd.read_sql_query("""
        SELECT date, period, filing_date, revenue, eps_diluted, ebitda
        FROM historical_income_statements
        WHERE symbol = ? AND filing_date IS NOT NULL
        ORDER BY date
    """, conn, params=(symbol,))

    # Load key metrics for EV/EBITDA (no filing_date in this table, use period date)
    metrics = pd.read_sql_query("""
        SELECT date, ev_to_ebitda
        FROM historical_key_metrics
        WHERE symbol = ?
        ORDER BY date
    """, conn, params=(symbol,))

    if income.empty:
        return pd.DataFrame()

    # Deduplicate: keep latest filing per period date
    income = income.drop_duplicates(subset=['date'], keep='last')
    income = income.sort_values('date')

    # Compute TTM (rolling sum of last 4 quarters)
    for col in ['revenue', 'eps_diluted', 'ebitda']:
        income[f'ttm_{col}'] = income[col].rolling(4, min_periods=4).sum()

    # Compute YoY growth (TTM now vs TTM 4 quarters ago)
    for col in ['revenue', 'eps_diluted', 'ebitda']:
        ttm_col = f'ttm_{col}'
        prior = income[ttm_col].shift(4)
        growth = ((income[ttm_col] / prior.replace(0, np.nan)) - 1) * 100
        # Cap extreme values
        income[f'{col}_growth'] = growth.clip(-200, 500)

    # Merge EV/EBITDA from key metrics
    if not metrics.empty:
        metrics = metrics.drop_duplicates(subset=['date'], keep='last')
        income = income.merge(
            metrics[['date', 'ev_to_ebitda']],
            on='date', how='left'
        )
    else:
        income['ev_to_ebitda'] = np.nan

    # Forward-fill EV/EBITDA
    income['ev_to_ebitda'] = income['ev_to_ebitda'].ffill()

    # Use filing_date as the availability date
    result = income[['filing_date', 'revenue_growth', 'eps_diluted_growth',
                      'ebitda_growth', 'ev_to_ebitda']].copy()
    result = result.rename(columns={
        'filing_date': 'date',
        'revenue_growth': 'rev_growth',
        'eps_diluted_growth': 'eps_growth',
        'ebitda_growth': 'ebitda_growth',
    })
    result = result.dropna(subset=['date'])
    result = result.sort_values('date')
    return result


# ==================== ANALYST GRADE FEATURES ====================

def compute_grade_features(symbol: str, conn: sqlite3.Connection,
                           date_range: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling analyst grade features from historical_grades table.
    Returns a DataFrame with daily grade metrics, merged onto the date range.
    """
    grades = pd.read_sql_query("""
        SELECT date, action FROM historical_grades
        WHERE symbol = ? ORDER BY date
    """, conn, params=(symbol,))

    if grades.empty:
        # Return empty features — will be filled with 0
        result = date_range[['date']].copy()
        result['upgrades_30d'] = 0
        result['downgrades_30d'] = 0
        result['grade_momentum'] = 0
        result['analyst_coverage_90d'] = 0
        return result

    # Normalize action values
    grades['action'] = grades['action'].str.lower().str.strip()
    grades['is_upgrade'] = grades['action'].isin(['upgrade']).astype(int)
    grades['is_downgrade'] = grades['action'].isin(['downgrade']).astype(int)
    grades['is_any'] = 1

    # Group by date (multiple grades can happen on the same day)
    daily = grades.groupby('date').agg(
        upgrades=('is_upgrade', 'sum'),
        downgrades=('is_downgrade', 'sum'),
        actions=('is_any', 'sum'),
    ).reset_index()

    # Merge onto full date range
    result = date_range[['date']].merge(daily, on='date', how='left')
    result = result.fillna(0)

    # Rolling windows
    result['upgrades_30d'] = result['upgrades'].rolling(30, min_periods=1).sum().astype(int)
    result['downgrades_30d'] = result['downgrades'].rolling(30, min_periods=1).sum().astype(int)
    result['grade_momentum'] = result['upgrades_30d'] - result['downgrades_30d']
    result['analyst_coverage_90d'] = result['actions'].rolling(90, min_periods=1).sum().astype(int)

    return result[['date', 'upgrades_30d', 'downgrades_30d', 'grade_momentum', 'analyst_coverage_90d']]


# ==================== SCORING ====================

def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply scoring thresholds to a DataFrame that has all required columns.
    Vectorized — no per-row loops.
    """
    # Trend Score (max 25)
    df['trend_score'] = 0
    df.loc[df['close'] > df['sma200'], 'trend_score'] += 10
    df.loc[(df['sma50'].notna()) & (df['sma200'].notna()) &
           (df['sma50'] > df['sma200']), 'trend_score'] += 10
    df.loc[(df['sma50'].notna()) & (df['close'] > df['sma50']), 'trend_score'] += 5

    # Fundamentals Score (max 25)
    df['fundamentals_score'] = 0
    df.loc[df['rev_growth'] > 15, 'fundamentals_score'] += 15
    df.loc[(df['rev_growth'] > 8) & (df['rev_growth'] <= 15), 'fundamentals_score'] += 8
    df.loc[df['eps_growth'] > 15, 'fundamentals_score'] += 10
    df.loc[(df['eps_growth'] > 8) & (df['eps_growth'] <= 15), 'fundamentals_score'] += 5

    # Valuation Score (max 16)
    df['valuation_score'] = 0
    df.loc[df['ev_ebitda'] < 12, 'valuation_score'] += 10
    df.loc[(df['ev_ebitda'] >= 12) & (df['ev_ebitda'] < 20), 'valuation_score'] += 6
    df.loc[(df['ev_ebitda'] >= 20) & (df['ev_ebitda'] < 30), 'valuation_score'] += 3

    # Momentum Score (max 10)
    df['momentum_score'] = 0
    df.loc[(df['rsi'] >= 40) & (df['rsi'] <= 65), 'momentum_score'] += 5
    df.loc[df['adx'] > 25, 'momentum_score'] += 5

    # Market Risk Score (max 10)
    df['market_risk_score'] = df['market_bullish'] * 10

    # LT Score
    df['lt_score'] = (df['trend_score'] + df['fundamentals_score'] +
                      df['valuation_score'] + df['momentum_score'] +
                      df['market_risk_score'])

    # Value Score (max 100)
    df['value_score'] = 0
    df.loc[df['ev_ebitda'] < 10, 'value_score'] += 30
    df.loc[(df['ev_ebitda'] >= 10) & (df['ev_ebitda'] < 15), 'value_score'] += 20
    df.loc[(df['ev_ebitda'] >= 15) & (df['ev_ebitda'] < 20), 'value_score'] += 10
    df.loc[(df['ev_ebitda'] >= 20) & (df['ev_ebitda'] < 30), 'value_score'] += 5

    df.loc[df['rev_growth'] > 25, 'value_score'] += 30
    df.loc[(df['rev_growth'] > 15) & (df['rev_growth'] <= 25), 'value_score'] += 20
    df.loc[(df['rev_growth'] > 8) & (df['rev_growth'] <= 15), 'value_score'] += 10

    df.loc[df['eps_growth'] > 15, 'value_score'] += 15
    df.loc[df['ebitda_growth'] > 15, 'value_score'] += 10

    return df


def score_dataframe_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply V1 + V2 scoring side-by-side. Keeps V1 value_score unchanged,
    applies tactical momentum tweaks to LT, adds continuous value_score_v2.
    """
    # Trend Score (max 25) — unchanged
    df['trend_score'] = 0
    df.loc[df['close'] > df['sma200'], 'trend_score'] += 10
    df.loc[(df['sma50'].notna()) & (df['sma200'].notna()) &
           (df['sma50'] > df['sma200']), 'trend_score'] += 10
    df.loc[(df['sma50'].notna()) & (df['close'] > df['sma50']), 'trend_score'] += 5

    # Fundamentals Score (max 25) — unchanged
    df['fundamentals_score'] = 0
    df.loc[df['rev_growth'] > 15, 'fundamentals_score'] += 15
    df.loc[(df['rev_growth'] > 8) & (df['rev_growth'] <= 15), 'fundamentals_score'] += 8
    df.loc[df['eps_growth'] > 15, 'fundamentals_score'] += 10
    df.loc[(df['eps_growth'] > 8) & (df['eps_growth'] <= 15), 'fundamentals_score'] += 5

    # Valuation Score (max 16) — unchanged
    df['valuation_score'] = 0
    df.loc[df['ev_ebitda'] < 12, 'valuation_score'] += 10
    df.loc[(df['ev_ebitda'] >= 12) & (df['ev_ebitda'] < 20), 'valuation_score'] += 6
    df.loc[(df['ev_ebitda'] >= 20) & (df['ev_ebitda'] < 30), 'valuation_score'] += 3

    # Momentum Score (max 10) — TWEAKED: RSI 40-55 (was 40-65), ADX > 20 (was > 25)
    df['momentum_score'] = 0
    df.loc[(df['rsi'] >= 40) & (df['rsi'] <= 55), 'momentum_score'] += 5
    df.loc[df['adx'] > 20, 'momentum_score'] += 5

    # Market Risk Score (max 10) — unchanged
    df['market_risk_score'] = df['market_bullish'] * 10

    # LT Score (with tactical momentum tweaks)
    df['lt_score'] = (df['trend_score'] + df['fundamentals_score'] +
                      df['valuation_score'] + df['momentum_score'] +
                      df['market_risk_score'])

    # Value Score V1 (max 100) — unchanged formula
    df['value_score'] = 0
    df.loc[df['ev_ebitda'] < 10, 'value_score'] += 30
    df.loc[(df['ev_ebitda'] >= 10) & (df['ev_ebitda'] < 15), 'value_score'] += 20
    df.loc[(df['ev_ebitda'] >= 15) & (df['ev_ebitda'] < 20), 'value_score'] += 10
    df.loc[(df['ev_ebitda'] >= 20) & (df['ev_ebitda'] < 30), 'value_score'] += 5

    df.loc[df['rev_growth'] > 25, 'value_score'] += 30
    df.loc[(df['rev_growth'] > 15) & (df['rev_growth'] <= 25), 'value_score'] += 20
    df.loc[(df['rev_growth'] > 8) & (df['rev_growth'] <= 15), 'value_score'] += 10

    df.loc[df['eps_growth'] > 15, 'value_score'] += 15
    df.loc[df['ebitda_growth'] > 15, 'value_score'] += 10

    # === Value Score V2 (continuous, max 100) ===

    # Valuation component (max 40, min -10)
    v2_val = np.where(df['ev_ebitda'] < 0, -10,
             np.where(df['ev_ebitda'] < 8, 40,
             np.where(df['ev_ebitda'] < 12, 30,
             np.where(df['ev_ebitda'] < 16, 20,
             np.where(df['ev_ebitda'] < 22, 10, 0)))))

    # Revenue Growth component (max 25)
    rev_capped = df['rev_growth'].clip(upper=50).fillna(0)
    v2_rev = (rev_capped / 2).clip(0, 25)
    # Penalize hyper-growth (>60%) by 0.7x
    v2_rev = np.where(df['rev_growth'] > 60, v2_rev * 0.7, v2_rev)

    # EPS Growth component (max 20, min -5)
    v2_eps = (df['eps_growth'].fillna(0) / 2).clip(-5, 20)

    # Quality component (max 15)
    v2_quality = np.where(df['ebitda_growth'] > 10, 10.0, 0.0)
    v2_quality = v2_quality + np.where((df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 25), 5.0, 0.0)

    df['value_score_v2'] = (v2_val + v2_rev + v2_eps + v2_quality).clip(0, 100).astype(int)

    return df


# ==================== SIGNAL DETECTION ====================

def detect_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect buy/sell signals from scored daily data.
    Returns a DataFrame of signal events.
    """
    signals = []

    prev_lt = df['lt_score'].shift(1)
    prev_value = df['value_score'].shift(1)
    prev_sma50 = df['sma50'].shift(1)
    prev_sma200 = df['sma200'].shift(1)
    prev_rsi = df['rsi'].shift(1)
    prev_close = df['close'].shift(1)

    # Strong Buy: LT >= 60 AND Value >= 60, crossed from below
    strong_buy = ((df['lt_score'] >= 60) & (df['value_score'] >= 60) &
                  ((prev_lt < 60) | (prev_value < 60)))
    if strong_buy.any():
        sig = df.loc[strong_buy, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'Strong Buy'
        signals.append(sig)

    # Buy: LT >= 50 AND Value >= 50, crossed from below
    buy = ((df['lt_score'] >= 50) & (df['value_score'] >= 50) &
           ((prev_lt < 50) | (prev_value < 50)))
    if buy.any():
        sig = df.loc[buy, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'Buy'
        signals.append(sig)

    # Golden Cross: SMA50 crosses above SMA200
    golden = ((df['sma50'] > df['sma200']) &
              (prev_sma50.notna()) & (prev_sma200.notna()) &
              (prev_sma50 <= prev_sma200))
    if golden.any():
        sig = df.loc[golden, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'Golden Cross'
        signals.append(sig)

    # Death Cross: SMA50 crosses below SMA200
    death = ((df['sma50'] < df['sma200']) &
             (prev_sma50.notna()) & (prev_sma200.notna()) &
             (prev_sma50 >= prev_sma200))
    if death.any():
        sig = df.loc[death, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'Death Cross'
        signals.append(sig)

    # RSI Recovery: RSI goes from < 30 to >= 40
    rsi_recovery = ((df['rsi'] >= 40) & (prev_rsi < 30))
    if rsi_recovery.any():
        sig = df.loc[rsi_recovery, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'RSI Recovery'
        signals.append(sig)

    # Bullish Alignment: close > SMA50 > SMA200 (new alignment, wasn't true yesterday)
    bullish_today = ((df['close'] > df['sma50']) & (df['sma50'] > df['sma200']))
    bullish_prev = ((prev_close > prev_sma50) & (prev_sma50 > prev_sma200))
    bullish_new = bullish_today & ~bullish_prev & prev_sma50.notna()
    if bullish_new.any():
        sig = df.loc[bullish_new, ['date', 'lt_score', 'value_score', 'close']].copy()
        sig['signal_type'] = 'Bullish Alignment'
        signals.append(sig)

    if not signals:
        return pd.DataFrame(columns=['date', 'lt_score', 'value_score', 'close', 'signal_type'])

    return pd.concat(signals, ignore_index=True)


def detect_signals_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all V1 + V2 signals. Preserves 6 original V1 signal types,
    adds 4 Buy Variants, 3 Conviction Tiers, and 2 Regime-Dependent signals.
    """
    signals = []

    prev_lt = df['lt_score'].shift(1)
    prev_value = df['value_score'].shift(1)
    prev_v2 = df['value_score_v2'].shift(1)
    prev_sma50 = df['sma50'].shift(1)
    prev_sma200 = df['sma200'].shift(1)
    prev_rsi = df['rsi'].shift(1)
    prev_close = df['close'].shift(1)
    prev_fund = df['fundamentals_score'].shift(1)

    cols = ['date', 'lt_score', 'value_score', 'value_score_v2', 'close']

    # === V1 SIGNALS (preserved) ===

    # Strong Buy: LT >= 60 AND Value >= 60, crossed from below
    strong_buy = ((df['lt_score'] >= 60) & (df['value_score'] >= 60) &
                  ((prev_lt < 60) | (prev_value < 60)))
    if strong_buy.any():
        sig = df.loc[strong_buy, cols].copy()
        sig['signal_type'] = 'Strong Buy'
        signals.append(sig)

    # Buy: LT >= 50 AND Value >= 50, crossed from below
    buy = ((df['lt_score'] >= 50) & (df['value_score'] >= 50) &
           ((prev_lt < 50) | (prev_value < 50)))
    if buy.any():
        sig = df.loc[buy, cols].copy()
        sig['signal_type'] = 'Buy'
        signals.append(sig)

    # Golden Cross: SMA50 crosses above SMA200
    golden = ((df['sma50'] > df['sma200']) &
              (prev_sma50.notna()) & (prev_sma200.notna()) &
              (prev_sma50 <= prev_sma200))
    if golden.any():
        sig = df.loc[golden, cols].copy()
        sig['signal_type'] = 'Golden Cross'
        signals.append(sig)

    # Death Cross: SMA50 crosses below SMA200
    death = ((df['sma50'] < df['sma200']) &
             (prev_sma50.notna()) & (prev_sma200.notna()) &
             (prev_sma50 >= prev_sma200))
    if death.any():
        sig = df.loc[death, cols].copy()
        sig['signal_type'] = 'Death Cross'
        signals.append(sig)

    # RSI Recovery: RSI goes from < 30 to >= 40
    rsi_recovery = ((df['rsi'] >= 40) & (prev_rsi < 30))
    if rsi_recovery.any():
        sig = df.loc[rsi_recovery, cols].copy()
        sig['signal_type'] = 'RSI Recovery'
        signals.append(sig)

    # Bullish Alignment: close > SMA50 > SMA200 (new alignment)
    bullish_today = ((df['close'] > df['sma50']) & (df['sma50'] > df['sma200']))
    bullish_prev = ((prev_close > prev_sma50) & (prev_sma50 > prev_sma200))
    bullish_new = bullish_today & ~bullish_prev & prev_sma50.notna()
    if bullish_new.any():
        sig = df.loc[bullish_new, cols].copy()
        sig['signal_type'] = 'Bullish Alignment'
        signals.append(sig)

    # === V2 BUY VARIANTS (crossover signals) ===

    # Buy_A: LT >= 50 & V2 >= 50 & eps_growth > 5
    buy_a = ((df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) &
             (df['eps_growth'] > 5) &
             ((prev_lt < 50) | (prev_v2 < 50)))
    if buy_a.any():
        sig = df.loc[buy_a, cols].copy()
        sig['signal_type'] = 'Buy_A'
        signals.append(sig)

    # Buy_B: LT >= 50 & V2 >= 50 & fundamentals >= 15
    buy_b = ((df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) &
             (df['fundamentals_score'] >= 15) &
             ((prev_lt < 50) | (prev_v2 < 50)))
    if buy_b.any():
        sig = df.loc[buy_b, cols].copy()
        sig['signal_type'] = 'Buy_B'
        signals.append(sig)

    # Buy_C: LT >= 50 & ev_ebitda 0-18 & rev_growth > 12
    buy_c = ((df['lt_score'] >= 50) &
             (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 18) &
             (df['rev_growth'] > 12) &
             (prev_lt < 50))
    if buy_c.any():
        sig = df.loc[buy_c, cols].copy()
        sig['signal_type'] = 'Buy_C'
        signals.append(sig)

    # Buy_D: LT >= 55 & fundamentals >= 18 & valuation >= 6
    buy_d = ((df['lt_score'] >= 55) & (df['fundamentals_score'] >= 18) &
             (df['valuation_score'] >= 6) &
             ((prev_lt < 55) | (prev_fund < 18)))
    if buy_d.any():
        sig = df.loc[buy_d, cols].copy()
        sig['signal_type'] = 'Buy_D'
        signals.append(sig)

    # === CONVICTION TIERS (state-entry signals) ===

    # Tier 1: LT >= 55 & V2 >= 55 & fundamentals >= 18 & ev_ebitda 0-22 & RSI 35-65
    tier1_cond = ((df['lt_score'] >= 55) & (df['value_score_v2'] >= 55) &
                  (df['fundamentals_score'] >= 18) &
                  (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 22) &
                  (df['rsi'] >= 35) & (df['rsi'] <= 65))
    tier1_prev = ((prev_lt >= 55) & (prev_v2 >= 55) &
                  (prev_fund >= 18) &
                  (df['ev_ebitda'].shift(1) > 0) & (df['ev_ebitda'].shift(1) <= 22) &
                  (prev_rsi >= 35) & (prev_rsi <= 65))
    tier1 = tier1_cond & ~tier1_prev.fillna(False)
    if tier1.any():
        sig = df.loc[tier1, cols].copy()
        sig['signal_type'] = 'Tier 1'
        signals.append(sig)

    # Tier 2: LT >= 50 & V2 >= 45 & (eps_growth > 8 OR rev_growth > 15)
    tier2_cond = ((df['lt_score'] >= 50) & (df['value_score_v2'] >= 45) &
                  ((df['eps_growth'] > 8) | (df['rev_growth'] > 15)))
    tier2_prev = ((prev_lt >= 50) & (prev_v2 >= 45) &
                  ((df['eps_growth'].shift(1) > 8) | (df['rev_growth'].shift(1) > 15)))
    tier2 = tier2_cond & ~tier2_prev.fillna(False)
    if tier2.any():
        sig = df.loc[tier2, cols].copy()
        sig['signal_type'] = 'Tier 2'
        signals.append(sig)

    # Tier 3: LT >= 40 & V2 >= 40 & RSI < 40 & fundamentals >= 15
    tier3_cond = ((df['lt_score'] >= 40) & (df['value_score_v2'] >= 40) &
                  (df['rsi'] < 40) & (df['fundamentals_score'] >= 15))
    tier3_prev = ((prev_lt >= 40) & (prev_v2 >= 40) &
                  (prev_rsi < 40) & (prev_fund >= 15))
    tier3 = tier3_cond & ~tier3_prev.fillna(False)
    if tier3.any():
        sig = df.loc[tier3, cols].copy()
        sig['signal_type'] = 'Tier 3'
        signals.append(sig)

    # === REGIME-DEPENDENT SIGNALS ===

    # Regime Buy Bull: bullish market & LT >= 55 & fundamentals >= 15 & ev_ebitda > 0
    regime_bull = ((df['market_bullish'] == 1) & (df['lt_score'] >= 55) &
                   (df['fundamentals_score'] >= 15) & (df['ev_ebitda'] > 0) &
                   ((prev_lt < 55) | (df['market_bullish'].shift(1) != 1)))
    if regime_bull.any():
        sig = df.loc[regime_bull, cols].copy()
        sig['signal_type'] = 'Regime Buy Bull'
        signals.append(sig)

    # Regime Buy Bear: bearish market & LT >= 45 & RSI < 40 & (eps > 8 OR rev > 10)
    regime_bear = ((df['market_bullish'] == 0) & (df['lt_score'] >= 45) &
                   (df['rsi'] < 40) &
                   ((df['eps_growth'] > 8) | (df['rev_growth'] > 10)) &
                   ((prev_lt < 45) | (df['market_bullish'].shift(1) != 0) |
                    (prev_rsi >= 40)))
    if regime_bear.any():
        sig = df.loc[regime_bear, cols].copy()
        sig['signal_type'] = 'Regime Buy Bear'
        signals.append(sig)

    # === ANALYST GRADE SIGNALS (require historical_grades data) ===

    if 'grade_momentum' in df.columns:
        prev_gm = df['grade_momentum'].shift(1)

        # Analyst Upgrade Cluster: grade_momentum >= 3 in 30d & LT >= 45
        upgrade_cluster = ((df['grade_momentum'] >= 3) & (df['lt_score'] >= 45) &
                           ((prev_gm < 3) | (prev_lt < 45)))
        if upgrade_cluster.any():
            sig = df.loc[upgrade_cluster, cols].copy()
            sig['signal_type'] = 'Analyst Upgrade Cluster'
            signals.append(sig)

        # Analyst Downgrade Recovery: downgrades_30d >= 2 & RSI < 35 & fundamentals >= 15
        prev_dg = df['downgrades_30d'].shift(1)
        dg_recovery = ((df['downgrades_30d'] >= 2) & (df['rsi'] < 35) &
                       (df['fundamentals_score'] >= 15) &
                       ((prev_dg < 2) | (prev_rsi >= 35)))
        if dg_recovery.any():
            sig = df.loc[dg_recovery, cols].copy()
            sig['signal_type'] = 'Analyst Downgrade Recovery'
            signals.append(sig)

        # High Coverage Buy: analyst_coverage_90d >= 5 & LT >= 50 & V2 >= 50
        prev_cov = df['analyst_coverage_90d'].shift(1)
        high_cov = ((df['analyst_coverage_90d'] >= 5) & (df['lt_score'] >= 50) &
                    (df['value_score_v2'] >= 50) &
                    ((prev_cov < 5) | (prev_lt < 50) | (prev_v2 < 50)))
        if high_cov.any():
            sig = df.loc[high_cov, cols].copy()
            sig['signal_type'] = 'High Coverage Buy'
            signals.append(sig)

    if not signals:
        return pd.DataFrame(columns=cols + ['signal_type'])

    return pd.concat(signals, ignore_index=True)


# ==================== FORWARD RETURNS ====================

def compute_forward_returns(signals_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate forward returns for each signal date"""
    if signals_df.empty:
        return signals_df

    prices_sorted = prices_df[['date', 'adjusted_close']].sort_values('date').reset_index(drop=True)
    date_to_idx = dict(zip(prices_sorted['date'], range(len(prices_sorted))))
    closes = prices_sorted['adjusted_close'].values

    periods = {'return_1w': 5, 'return_1m': 21, 'return_3m': 63,
               'return_6m': 126, 'return_1y': 252}

    for col, offset in periods.items():
        returns = []
        for _, row in signals_df.iterrows():
            idx = date_to_idx.get(row['date'])
            if idx is not None and idx + offset < len(closes):
                entry = closes[idx]
                exit_price = closes[idx + offset]
                returns.append(((exit_price / entry) - 1) * 100 if entry > 0 else None)
            else:
                returns.append(None)
        signals_df[col] = returns

    return signals_df


# ==================== PROCESS ONE STOCK ====================

def process_stock(symbol: str, conn: sqlite3.Connection,
                  spy_regime: pd.DataFrame) -> tuple:
    """
    Process one stock: compute indicators, score, detect signals, calc returns.
    Returns (scores_df, signals_df) or (None, None) if insufficient data.
    """
    # Load prices
    prices = pd.read_sql_query(
        "SELECT date, open, high, low, close, volume, adjusted_close "
        "FROM historical_prices WHERE symbol = ? ORDER BY date",
        conn, params=(symbol,)
    )

    if len(prices) < 200:
        return None, None

    # Compute technical indicators
    prices = compute_technical_indicators(prices)

    # Merge market regime
    prices = prices.merge(spy_regime, on='date', how='left')
    prices['market_bullish'] = prices['market_bullish'].fillna(0).astype(int)

    # Compute TTM fundamentals
    fundamentals = compute_ttm_fundamentals(symbol, conn)

    # Merge fundamentals into daily prices (forward-fill between filing dates)
    if not fundamentals.empty:
        prices = prices.merge(fundamentals, on='date', how='left')
        for col in ['rev_growth', 'eps_growth', 'ebitda_growth', 'ev_to_ebitda']:
            if col in prices.columns:
                prices[col] = prices[col].ffill()
    else:
        prices['rev_growth'] = np.nan
        prices['eps_growth'] = np.nan
        prices['ebitda_growth'] = np.nan
        prices['ev_to_ebitda'] = np.nan

    # Rename for scoring
    prices = prices.rename(columns={'ev_to_ebitda': 'ev_ebitda'})

    # Fill NaN for scoring (treat missing as 0/neutral)
    for col in ['rev_growth', 'eps_growth', 'ebitda_growth', 'ev_ebitda']:
        if col not in prices.columns:
            prices[col] = np.nan

    # Only score rows where we have SMA200 (need 200 days of history)
    scoreable = prices.dropna(subset=['sma200']).copy()
    if scoreable.empty:
        return None, None

    # Apply V2 scoring (includes V1 value_score + V2 value_score_v2)
    scoreable = score_dataframe_v2(scoreable)

    # Merge analyst grade features (if grades table exists)
    try:
        grade_features = compute_grade_features(symbol, conn, scoreable)
        scoreable = scoreable.merge(grade_features, on='date', how='left')
        for col in ['upgrades_30d', 'downgrades_30d', 'grade_momentum', 'analyst_coverage_90d']:
            scoreable[col] = scoreable[col].fillna(0).astype(int)
    except Exception:
        # Grades table may not exist yet — add zero columns
        for col in ['upgrades_30d', 'downgrades_30d', 'grade_momentum', 'analyst_coverage_90d']:
            scoreable[col] = 0

    # Build scores output
    scores_df = scoreable[['date', 'close', 'sma50', 'sma200', 'rsi', 'adx',
                            'ev_ebitda', 'rev_growth', 'eps_growth', 'ebitda_growth',
                            'trend_score', 'fundamentals_score', 'valuation_score',
                            'momentum_score', 'market_risk_score',
                            'lt_score', 'value_score', 'value_score_v2']].copy()
    scores_df.insert(0, 'symbol', symbol)

    # Detect V2 signals (includes all V1 + V2 + analyst signals)
    signals_df = detect_signals_v2(scoreable)
    if not signals_df.empty:
        signals_df.insert(0, 'symbol', symbol)
        signals_df = signals_df.rename(columns={'close': 'close_price'})
        signals_df = compute_forward_returns(signals_df, prices)

    return scores_df, signals_df


# ==================== BATCH SAVE ====================

def save_scores_batch(scores_list: list, conn: sqlite3.Connection):
    """Bulk insert daily scores using INSERT OR REPLACE"""
    if not scores_list:
        return 0
    all_scores = pd.concat(scores_list, ignore_index=True)
    cols = ['symbol', 'date', 'close', 'sma50', 'sma200', 'rsi', 'adx',
            'ev_ebitda', 'rev_growth', 'eps_growth', 'ebitda_growth',
            'trend_score', 'fundamentals_score', 'valuation_score',
            'momentum_score', 'market_risk_score', 'lt_score', 'value_score',
            'value_score_v2']
    placeholders = ','.join(['?'] * len(cols))
    sql = f"INSERT OR REPLACE INTO backtest_daily_scores ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(row[c] if c in row and pd.notna(row[c]) else None for c in cols)
            for _, row in all_scores.iterrows()]
    conn.executemany(sql, rows)
    return len(rows)


def save_signals_batch(signals_list: list, conn: sqlite3.Connection):
    """Bulk insert signals using INSERT OR REPLACE"""
    if not signals_list:
        return 0
    all_signals = pd.concat(signals_list, ignore_index=True)
    cols = ['symbol', 'date', 'signal_type', 'lt_score', 'value_score', 'value_score_v2',
            'close_price', 'return_1w', 'return_1m', 'return_3m', 'return_6m', 'return_1y']
    for c in cols:
        if c not in all_signals.columns:
            all_signals[c] = None
    placeholders = ','.join(['?'] * len(cols))
    sql = f"INSERT OR REPLACE INTO backtest_signals ({','.join(cols)}) VALUES ({placeholders})"
    rows = [tuple(row[c] if c in row and pd.notna(row[c]) else None for c in cols)
            for _, row in all_signals.iterrows()]
    conn.executemany(sql, rows)
    return len(rows)


# ==================== SUMMARY ====================

def show_summary():
    """Display backtest results summary"""
    conn = sqlite3.connect(BACKTEST_DB)

    # Total stats
    total_scores = conn.execute("SELECT COUNT(*) FROM backtest_daily_scores").fetchone()[0]
    total_signals = conn.execute("SELECT COUNT(*) FROM backtest_signals").fetchone()[0]
    unique_stocks = conn.execute("SELECT COUNT(DISTINCT symbol) FROM backtest_daily_scores").fetchone()[0]

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Stocks scored: {unique_stocks:,}")
    print(f"  Daily score rows: {total_scores:,}")
    print(f"  Total signals detected: {total_signals:,}")

    # Signal counts by type
    print("\n--- Signal Counts ---")
    signal_counts = pd.read_sql_query("""
        SELECT signal_type, COUNT(*) as count
        FROM backtest_signals
        GROUP BY signal_type
        ORDER BY count DESC
    """, conn)
    for _, row in signal_counts.iterrows():
        print(f"  {row['signal_type']:<25s}: {row['count']:>6,}")

    # Average forward returns by signal type
    print("\n--- Average Forward Returns by Signal Type ---")
    print(f"  {'Signal':<25s} {'1W':>8s} {'1M':>8s} {'3M':>8s} {'6M':>8s} {'1Y':>8s}  {'Win% 3M':>8s}")
    print("  " + "-" * 78)

    for _, sig_row in signal_counts.iterrows():
        sig_type = sig_row['signal_type']
        stats = pd.read_sql_query("""
            SELECT
                AVG(return_1w) as avg_1w,
                AVG(return_1m) as avg_1m,
                AVG(return_3m) as avg_3m,
                AVG(return_6m) as avg_6m,
                AVG(return_1y) as avg_1y,
                SUM(CASE WHEN return_3m > 0 THEN 1 ELSE 0 END) * 100.0 /
                    NULLIF(COUNT(return_3m), 0) as win_3m
            FROM backtest_signals
            WHERE signal_type = ?
        """, conn, params=(sig_type,))

        if not stats.empty:
            r = stats.iloc[0]
            def fmt(v):
                return f"{v:+.1f}%" if pd.notna(v) else "   N/A"
            win = f"{r['win_3m']:.0f}%" if pd.notna(r['win_3m']) else "N/A"
            print(f"  {sig_type:<25s} {fmt(r['avg_1w']):>8s} {fmt(r['avg_1m']):>8s} "
                  f"{fmt(r['avg_3m']):>8s} {fmt(r['avg_6m']):>8s} {fmt(r['avg_1y']):>8s}  {win:>8s}")

    # Score distribution
    print("\n--- LT Score Distribution (latest date per stock) ---")
    score_dist = pd.read_sql_query("""
        SELECT
            CASE
                WHEN lt_score >= 70 THEN '70+ (Strong)'
                WHEN lt_score >= 50 THEN '50-69 (Good)'
                WHEN lt_score >= 30 THEN '30-49 (Fair)'
                ELSE '0-29 (Weak)'
            END as bucket,
            COUNT(*) as count
        FROM backtest_daily_scores
        WHERE date = (SELECT MAX(date) FROM backtest_daily_scores)
        GROUP BY bucket
        ORDER BY bucket DESC
    """, conn)
    for _, row in score_dist.iterrows():
        print(f"  {row['bucket']:<25s}: {row['count']:>6,}")

    # Value Score V2 distribution
    print("\n--- Value Score V2 Distribution (latest date per stock) ---")
    v2_dist = pd.read_sql_query("""
        SELECT
            CASE
                WHEN value_score_v2 >= 70 THEN '70+ (Strong)'
                WHEN value_score_v2 >= 50 THEN '50-69 (Good)'
                WHEN value_score_v2 >= 30 THEN '30-49 (Fair)'
                WHEN value_score_v2 >= 0 THEN '0-29 (Weak)'
                ELSE 'N/A'
            END as bucket,
            COUNT(*) as count
        FROM backtest_daily_scores
        WHERE date = (SELECT MAX(date) FROM backtest_daily_scores)
              AND value_score_v2 IS NOT NULL
        GROUP BY bucket
        ORDER BY bucket DESC
    """, conn)
    for _, row in v2_dist.iterrows():
        print(f"  {row['bucket']:<25s}: {row['count']:>6,}")

    # Top performing signal combos
    print("\n--- Top 10 Strongest 3-Month Return Signals ---")
    top_signals = pd.read_sql_query("""
        SELECT symbol, date, signal_type, lt_score, value_score,
               value_score_v2, close_price, return_3m
        FROM backtest_signals
        WHERE return_3m IS NOT NULL
        ORDER BY return_3m DESC
        LIMIT 10
    """, conn)
    for _, row in top_signals.iterrows():
        v2 = int(row['value_score_v2']) if pd.notna(row['value_score_v2']) else 0
        print(f"  {row['symbol']:<6s} {row['date']} {row['signal_type']:<20s} "
              f"LT:{row['lt_score']:>2} V1:{row['value_score']:>2} V2:{v2:>2} "
              f"3M: {row['return_3m']:+.1f}%")

    print("=" * 70)
    conn.close()


# ==================== WINNER vs LOSER ANALYSIS ====================

def analyze_winners_losers():
    """
    Deep analysis: what indicators are commonly seen in winning signals vs losing ones.
    Joins signals with daily scores to get full indicator context at signal time.
    """
    conn = sqlite3.connect(BACKTEST_DB)

    # Join signals with daily scores to get all indicators at signal time
    df = pd.read_sql_query("""
        SELECT
            s.symbol, s.date, s.signal_type,
            s.lt_score, s.value_score, s.value_score_v2, s.close_price,
            s.return_1w, s.return_1m, s.return_3m, s.return_6m, s.return_1y,
            d.rsi, d.adx, d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth,
            d.sma50, d.sma200, d.close, d.market_risk_score as market_bullish_score,
            d.trend_score, d.fundamentals_score, d.valuation_score,
            d.momentum_score, d.market_risk_score
        FROM backtest_signals s
        LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
        WHERE s.return_3m IS NOT NULL
    """, conn)

    if df.empty:
        print("No signals with 3M returns found.")
        conn.close()
        return

    # Cap EV/EBITDA outliers for meaningful analysis (some stocks have near-zero EBITDA)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-100, 200)

    # Classify winners/losers (using 3M return as primary benchmark)
    df['outcome'] = np.where(df['return_3m'] > 0, 'Winner', 'Loser')
    winners = df[df['outcome'] == 'Winner']
    losers = df[df['outcome'] == 'Loser']

    print("\n" + "=" * 80)
    print("WINNER vs LOSER ANALYSIS")
    print(f"(Using 3-month forward return to classify; {len(df):,} signals with 3M data)")
    print("=" * 80)

    # ---- 1. Indicator Profiles: Winners vs Losers (all signal types) ----
    print("\n--- 1. INDICATOR PROFILES AT SIGNAL TIME (All Signals) ---")
    indicators = ['lt_score', 'value_score', 'value_score_v2', 'rsi', 'adx',
                  'ev_ebitda_clean', 'rev_growth', 'eps_growth', 'ebitda_growth',
                  'trend_score', 'fundamentals_score', 'valuation_score',
                  'momentum_score', 'market_risk_score']

    print(f"  {'Indicator':<25s} {'W Avg':>8s} {'L Avg':>8s} {'W Med':>8s} {'L Med':>8s} {'AvgDelta':>10s}")
    print("  " + "-" * 70)
    for ind in indicators:
        w_mean = winners[ind].mean()
        l_mean = losers[ind].mean()
        w_med = winners[ind].median()
        l_med = losers[ind].median()
        delta = w_mean - l_mean if pd.notna(w_mean) and pd.notna(l_mean) else np.nan
        label = 'ev_ebitda' if ind == 'ev_ebitda_clean' else ind
        w_str = f"{w_mean:.1f}" if pd.notna(w_mean) else "N/A"
        l_str = f"{l_mean:.1f}" if pd.notna(l_mean) else "N/A"
        wm_str = f"{w_med:.1f}" if pd.notna(w_med) else "N/A"
        lm_str = f"{l_med:.1f}" if pd.notna(l_med) else "N/A"
        d_str = f"{delta:+.1f}" if pd.notna(delta) else "N/A"
        print(f"  {label:<25s} {w_str:>8s} {l_str:>8s} {wm_str:>8s} {lm_str:>8s} {d_str:>10s}")

    # ---- 2. Median Returns (less outlier-sensitive) ----
    print("\n--- 2. MEDIAN vs AVERAGE RETURNS BY SIGNAL ---")
    print(f"  {'Signal':<25s} {'Avg 3M':>8s} {'Med 3M':>8s} {'Avg 1Y':>8s} {'Med 1Y':>8s} {'Count':>8s}")
    print("  " + "-" * 62)
    for sig_type in df['signal_type'].unique():
        sub = df[df['signal_type'] == sig_type]
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        med_1y = sub['return_1y'].median()
        print(f"  {sig_type:<25s} {avg_3m:+7.1f}% {med_3m:+7.1f}% "
              f"{avg_1y:+7.1f}% {med_1y:+7.1f}% {len(sub):>7,}")

    # ---- 3. Win Rate by LT Score Bracket ----
    print("\n--- 3. WIN RATE BY LT SCORE BRACKET ---")
    lt_bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    print(f"  {'LT Score':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s} {'Avg 1Y':>10s}")
    print("  " + "-" * 65)
    for lo, hi in lt_bins:
        sub = df[(df['lt_score'] >= lo) & (df['lt_score'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        label = f"{lo}-{hi-1}"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}% "
              f"{avg_1y:>+9.1f}%" if pd.notna(avg_1y) else
              f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%       N/A")

    # ---- 4. Win Rate by Value Score Bracket ----
    print("\n--- 4. WIN RATE BY VALUE SCORE BRACKET ---")
    vs_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]
    print(f"  {'Value Score':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 55)
    for lo, hi in vs_bins:
        sub = df[(df['value_score'] >= lo) & (df['value_score'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        label = f"{lo}-{hi-1}"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 5. Win Rate by Market Regime ----
    print("\n--- 5. WIN RATE BY MARKET REGIME (SPY > SMA200) ---")
    print(f"  {'Regime':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s} {'Avg 1Y':>10s}")
    print("  " + "-" * 65)
    for regime, label in [(10, 'Bullish'), (0, 'Bearish')]:
        sub = df[df['market_risk_score'] == regime]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}% "
              f"{avg_1y:>+9.1f}%" if pd.notna(avg_1y) else
              f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%       N/A")

    # ---- 6. Win Rate by RSI Range ----
    print("\n--- 6. WIN RATE BY RSI RANGE AT SIGNAL ---")
    rsi_bins = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 100)]
    print(f"  {'RSI Range':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 55)
    for lo, hi in rsi_bins:
        sub = df[(df['rsi'] >= lo) & (df['rsi'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 7. Win Rate by EV/EBITDA Range ----
    print("\n--- 7. WIN RATE BY EV/EBITDA RANGE AT SIGNAL ---")
    ev_bins = [(-100, 0), (0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 9999)]
    print(f"  {'EV/EBITDA':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 55)
    for lo, hi in ev_bins:
        sub = df[(df['ev_ebitda_clean'] >= lo) & (df['ev_ebitda_clean'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        if hi >= 9999:
            label = f"{lo}+"
        elif lo < 0:
            label = f"Negative"
        else:
            label = f"{lo}-{hi}"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 8. Win Rate by ADX (Trend Strength) ----
    print("\n--- 8. WIN RATE BY ADX (TREND STRENGTH) AT SIGNAL ---")
    adx_bins = [(0, 15), (15, 25), (25, 35), (35, 50), (50, 100)]
    print(f"  {'ADX Range':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 55)
    for lo, hi in adx_bins:
        sub = df[(df['adx'] >= lo) & (df['adx'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        label = f"{lo}-{hi}" if hi < 100 else f"{lo}+"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 9. Win Rate by Revenue Growth Range ----
    print("\n--- 9. WIN RATE BY REVENUE GROWTH AT SIGNAL ---")
    rg_bins = [(-200, -10), (-10, 0), (0, 8), (8, 15), (15, 25), (25, 50), (50, 500)]
    print(f"  {'Rev Growth':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 55)
    for lo, hi in rg_bins:
        sub = df[(df['rev_growth'] >= lo) & (df['rev_growth'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        if hi >= 500:
            label = f"{lo}%+"
        elif lo <= -200:
            label = f"< {hi}%"
        else:
            label = f"{lo}%-{hi}%"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 10. Sub-Score Combination Analysis ----
    print("\n--- 10. BEST SUB-SCORE COMBINATIONS (Top 15 by Win Rate, min 50 signals) ---")
    # Create buckets for each sub-score
    df['trend_bucket'] = pd.cut(df['trend_score'], bins=[-1, 5, 15, 25],
                                 labels=['Low(0-5)', 'Mid(10-15)', 'High(20-25)'])
    df['fund_bucket'] = pd.cut(df['fundamentals_score'], bins=[-1, 5, 13, 25],
                                labels=['Low(0-5)', 'Mid(8-13)', 'High(15-25)'])
    df['val_bucket'] = pd.cut(df['valuation_score'], bins=[-1, 3, 6, 16],
                               labels=['Low(0-3)', 'Mid(6)', 'High(10-16)'])

    combos = df.groupby(['trend_bucket', 'fund_bucket', 'val_bucket']).agg(
        count=('return_3m', 'count'),
        win_pct=('return_3m', lambda x: (x > 0).mean() * 100),
        avg_3m=('return_3m', 'mean'),
        med_3m=('return_3m', 'median'),
    ).reset_index()
    combos = combos[combos['count'] >= 50].sort_values('win_pct', ascending=False).head(15)

    print(f"  {'Trend':<12s} {'Fundmntl':<12s} {'Valuation':<12s} {'Count':>7s} {'Win%':>7s} {'Avg 3M':>9s} {'Med 3M':>9s}")
    print("  " + "-" * 72)
    for _, r in combos.iterrows():
        print(f"  {str(r['trend_bucket']):<12s} {str(r['fund_bucket']):<12s} {str(r['val_bucket']):<12s} "
              f"{r['count']:>6.0f} {r['win_pct']:>6.1f}% {r['avg_3m']:>+8.1f}% {r['med_3m']:>+8.1f}%")

    # ---- 11. By Signal Type: Winners vs Losers Indicator Breakdown ----
    print("\n--- 11. PER-SIGNAL WINNER vs LOSER INDICATOR COMPARISON ---")
    for sig_type in sorted(df['signal_type'].unique()):
        sub = df[df['signal_type'] == sig_type]
        if len(sub) < 20:
            continue
        w = sub[sub['outcome'] == 'Winner']
        l = sub[sub['outcome'] == 'Loser']
        print(f"\n  [{sig_type}] — {len(w):,} winners, {len(l):,} losers")
        print(f"    {'Indicator':<22s} {'W Avg':>8s} {'L Avg':>8s} {'W Med':>8s} {'L Med':>8s} {'AvgDelta':>10s}")
        print(f"    " + "-" * 62)
        for ind in ['lt_score', 'value_score', 'rsi', 'adx', 'ev_ebitda_clean',
                     'rev_growth', 'eps_growth', 'trend_score',
                     'fundamentals_score', 'valuation_score', 'momentum_score']:
            w_m = w[ind].mean()
            l_m = l[ind].mean()
            w_md = w[ind].median()
            l_md = l[ind].median()
            d = w_m - l_m if pd.notna(w_m) and pd.notna(l_m) else np.nan
            label = 'ev_ebitda' if ind == 'ev_ebitda_clean' else ind
            w_s = f"{w_m:.1f}" if pd.notna(w_m) else "N/A"
            l_s = f"{l_m:.1f}" if pd.notna(l_m) else "N/A"
            wm_s = f"{w_md:.1f}" if pd.notna(w_md) else "N/A"
            lm_s = f"{l_md:.1f}" if pd.notna(l_md) else "N/A"
            d_s = f"{d:+.1f}" if pd.notna(d) else "N/A"
            print(f"    {label:<22s} {w_s:>8s} {l_s:>8s} {wm_s:>8s} {lm_s:>8s} {d_s:>10s}")

    # ---- 12. Year-over-Year Win Rate ----
    print("\n--- 12. WIN RATE BY YEAR ---")
    df['year'] = df['date'].str[:4]
    print(f"  {'Year':<8s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s}")
    print("  " + "-" * 48)
    for year in sorted(df['year'].unique()):
        sub = df[df['year'] == year]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        print(f"  {year:<8s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}%")

    # ---- 13. Strongest Multi-Factor Filters ----
    print("\n--- 13. MULTI-FACTOR FILTER PERFORMANCE ---")
    print("  Combining filters to find highest-conviction setups:\n")

    filters = [
        ("Baseline (all signals)", df),
        ("LT >= 50", df[df['lt_score'] >= 50]),
        ("LT >= 60", df[df['lt_score'] >= 60]),
        ("LT >= 50 + Bull market", df[(df['lt_score'] >= 50) & (df['market_risk_score'] == 10)]),
        ("LT >= 50 + RSI 40-60", df[(df['lt_score'] >= 50) & (df['rsi'] >= 40) & (df['rsi'] <= 60)]),
        ("LT >= 50 + ADX > 25", df[(df['lt_score'] >= 50) & (df['adx'] > 25)]),
        ("LT >= 50 + Rev Growth > 15%", df[(df['lt_score'] >= 50) & (df['rev_growth'] > 15)]),
        ("LT >= 50 + EV/EBITDA < 15", df[(df['lt_score'] >= 50) & (df['ev_ebitda'] > 0) & (df['ev_ebitda'] < 15)]),
        ("LT >= 60 + Bull + Rev > 15%", df[(df['lt_score'] >= 60) & (df['market_risk_score'] == 10) & (df['rev_growth'] > 15)]),
        ("LT >= 60 + EV/EBITDA < 15 + ADX > 25", df[(df['lt_score'] >= 60) & (df['ev_ebitda'] > 0) & (df['ev_ebitda'] < 15) & (df['adx'] > 25)]),
        ("Value >= 60 + Bull market", df[(df['value_score'] >= 60) & (df['market_risk_score'] == 10)]),
        ("LT >= 50 + Value >= 50 + Bull", df[(df['lt_score'] >= 50) & (df['value_score'] >= 50) & (df['market_risk_score'] == 10)]),
    ]

    print(f"  {'Filter':<40s} {'Count':>7s} {'Win%':>7s} {'Avg 3M':>9s} {'Med 3M':>9s} {'Avg 1Y':>9s}")
    print("  " + "-" * 85)
    for name, sub in filters:
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        y_str = f"{avg_1y:>+8.1f}%" if pd.notna(avg_1y) else "     N/A"
        print(f"  {name:<40s} {len(sub):>6,} {win_pct:>6.1f}% {avg_3m:>+8.1f}% {med_3m:>+8.1f}% {y_str}")

    # ---- 14. WIN RATE BY VALUE SCORE V2 BRACKET ----
    print("\n--- 14. WIN RATE BY VALUE SCORE V2 BRACKET ---")
    v2_bins = [(0, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 101)]
    print(f"  {'V2 Score':<15s} {'Count':>8s} {'Win%':>8s} {'Avg 3M':>10s} {'Med 3M':>10s} {'Avg 1Y':>10s}")
    print("  " + "-" * 65)
    for lo, hi in v2_bins:
        sub = df[(df['value_score_v2'] >= lo) & (df['value_score_v2'] < hi)]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        label = f"{lo}-{hi-1}"
        y_str = f"{avg_1y:>+9.1f}%" if pd.notna(avg_1y) else "      N/A"
        print(f"  {label:<15s} {len(sub):>7,} {win_pct:>7.1f}% {avg_3m:>+9.1f}% {med_3m:>+9.1f}% {y_str}")

    # ---- 15. V1 vs V2 SIGNAL COMPARISON ----
    print("\n--- 15. V1 vs V2 SIGNAL COMPARISON ---")
    v1_signals = ['Strong Buy', 'Buy', 'Golden Cross', 'Death Cross',
                  'RSI Recovery', 'Bullish Alignment']
    v2_signals = ['Buy_A', 'Buy_B', 'Buy_C', 'Buy_D',
                  'Tier 1', 'Tier 2', 'Tier 3',
                  'Regime Buy Bull', 'Regime Buy Bear']

    print(f"\n  {'Signal':<25s} {'Count':>7s} {'Win%':>7s} {'Avg 3M':>9s} {'Med 3M':>9s} "
          f"{'Avg 6M':>9s} {'Avg 1Y':>9s}")
    print("  " + "-" * 80)

    print("  --- V1 Signals ---")
    for sig_type in v1_signals:
        sub = df[df['signal_type'] == sig_type]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_6m = sub['return_6m'].mean()
        avg_1y = sub['return_1y'].mean()
        s6 = f"{avg_6m:>+8.1f}%" if pd.notna(avg_6m) else "     N/A"
        s1y = f"{avg_1y:>+8.1f}%" if pd.notna(avg_1y) else "     N/A"
        print(f"  {sig_type:<25s} {len(sub):>6,} {win_pct:>6.1f}% {avg_3m:>+8.1f}% "
              f"{med_3m:>+8.1f}% {s6} {s1y}")

    print("  --- V2 Signals ---")
    for sig_type in v2_signals:
        sub = df[df['signal_type'] == sig_type]
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_6m = sub['return_6m'].mean()
        avg_1y = sub['return_1y'].mean()
        s6 = f"{avg_6m:>+8.1f}%" if pd.notna(avg_6m) else "     N/A"
        s1y = f"{avg_1y:>+8.1f}%" if pd.notna(avg_1y) else "     N/A"
        print(f"  {sig_type:<25s} {len(sub):>6,} {win_pct:>6.1f}% {avg_3m:>+8.1f}% "
              f"{med_3m:>+8.1f}% {s6} {s1y}")

    # ---- 16. V2 MULTI-FACTOR FILTER PERFORMANCE ----
    print("\n--- 16. V2 MULTI-FACTOR FILTER PERFORMANCE ---")
    v2_filters = [
        ("Baseline (all signals)", df),
        ("V2 >= 50", df[df['value_score_v2'] >= 50]),
        ("V2 >= 60", df[df['value_score_v2'] >= 60]),
        ("LT >= 50 + V2 >= 50", df[(df['lt_score'] >= 50) & (df['value_score_v2'] >= 50)]),
        ("LT >= 50 + V2 >= 50 + Bull", df[(df['lt_score'] >= 50) & (df['value_score_v2'] >= 50) &
                                            (df['market_risk_score'] == 10)]),
        ("LT >= 55 + V2 >= 55 + Fund >= 18", df[(df['lt_score'] >= 55) & (df['value_score_v2'] >= 55) &
                                                  (df['fundamentals_score'] >= 18)]),
        ("LT >= 55 + V2 >= 55 + RSI 35-65", df[(df['lt_score'] >= 55) & (df['value_score_v2'] >= 55) &
                                                 (df['rsi'] >= 35) & (df['rsi'] <= 65)]),
        ("V2 >= 50 + EV/EBITDA 0-22", df[(df['value_score_v2'] >= 50) &
                                           (df['ev_ebitda'] > 0) & (df['ev_ebitda'] <= 22)]),
    ]

    print(f"  {'Filter':<40s} {'Count':>7s} {'Win%':>7s} {'Avg 3M':>9s} {'Med 3M':>9s} {'Avg 1Y':>9s}")
    print("  " + "-" * 85)
    for name, sub in v2_filters:
        if len(sub) < 10:
            continue
        win_pct = (sub['return_3m'] > 0).mean() * 100
        avg_3m = sub['return_3m'].mean()
        med_3m = sub['return_3m'].median()
        avg_1y = sub['return_1y'].mean()
        y_str = f"{avg_1y:>+8.1f}%" if pd.notna(avg_1y) else "     N/A"
        print(f"  {name:<40s} {len(sub):>6,} {win_pct:>6.1f}% {avg_3m:>+8.1f}% {med_3m:>+8.1f}% {y_str}")

    # ---- 17. ANALYST GRADE ACCURACY ANALYSIS ----
    try:
        grades_df = pd.read_sql_query("""
            SELECT g.symbol, g.date, g.action
            FROM historical_grades g
            WHERE g.action IN ('upgrade', 'downgrade')
            ORDER BY g.symbol, g.date
        """, conn)

        if not grades_df.empty and len(grades_df) >= 50:
            print("\n--- 17. ANALYST GRADE ACCURACY ---")
            print("  Forward returns after analyst upgrades vs downgrades:\n")

            # Efficient: group by symbol, load prices once per symbol
            grade_returns = []
            for symbol, sym_grades in grades_df.groupby('symbol'):
                sym_prices = pd.read_sql_query(
                    "SELECT date, adjusted_close FROM historical_prices "
                    "WHERE symbol = ? ORDER BY date",
                    conn, params=(symbol,)
                )
                if sym_prices.empty:
                    continue
                date_to_idx = dict(zip(sym_prices['date'], range(len(sym_prices))))
                closes = sym_prices['adjusted_close'].values

                for _, gr in sym_grades.iterrows():
                    idx = date_to_idx.get(gr['date'])
                    if idx is None:
                        continue
                    entry = closes[idx]
                    if entry <= 0:
                        continue
                    r = {'action': gr['action']}
                    for label, offset in [('1m', 21), ('3m', 63), ('6m', 126)]:
                        if idx + offset < len(closes):
                            r[f'return_{label}'] = ((closes[idx + offset] / entry) - 1) * 100
                        else:
                            r[f'return_{label}'] = None
                    grade_returns.append(r)

            if grade_returns:
                gr_df = pd.DataFrame(grade_returns)
                print(f"  {'Action':<15s} {'Count':>8s} {'Avg 1M':>9s} {'Avg 3M':>9s} {'Avg 6M':>9s} {'Win% 3M':>9s}")
                print("  " + "-" * 55)
                for action in ['upgrade', 'downgrade']:
                    sub = gr_df[gr_df['action'] == action]
                    if len(sub) < 10:
                        continue
                    avg_1m = sub['return_1m'].mean()
                    avg_3m = sub['return_3m'].mean()
                    avg_6m = sub['return_6m'].mean()
                    win_3m = (sub['return_3m'] > 0).mean() * 100 if sub['return_3m'].notna().sum() > 0 else 0
                    m1 = f"{avg_1m:>+8.1f}%" if pd.notna(avg_1m) else "     N/A"
                    m3 = f"{avg_3m:>+8.1f}%" if pd.notna(avg_3m) else "     N/A"
                    m6 = f"{avg_6m:>+8.1f}%" if pd.notna(avg_6m) else "     N/A"
                    print(f"  {action:<15s} {len(sub):>7,} {m1} {m3} {m6} {win_3m:>8.1f}%")
            else:
                print("  No grade events matched to price data")
        else:
            print("\n--- 17. ANALYST GRADE ACCURACY ---")
            print("  Insufficient grade data (run collect_analyst_data.py first)")
    except Exception as e:
        print(f"\n--- 17. ANALYST GRADE ACCURACY ---")
        print(f"  Skipped (grades table not available: {e})")

    print("\n" + "=" * 80)
    conn.close()


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description='Run backtest on historical data')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols')
    parser.add_argument('--summary', action='store_true', help='Show results summary')
    parser.add_argument('--analyze', action='store_true', help='Deep winner vs loser analysis')
    args = parser.parse_args()

    if args.analyze:
        analyze_winners_losers()
        return

    if args.summary:
        show_summary()
        return

    # Ensure tables exist
    setup_backtest_tables()

    conn = sqlite3.connect(BACKTEST_DB)

    # Get symbol list
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM historical_prices WHERE symbol != 'SPY' ORDER BY symbol"
        ).fetchall()
        symbols = [r[0] for r in rows]

    if not symbols:
        log.error("No symbols found in backtest.db")
        conn.close()
        return

    # Clear existing results
    if args.symbols:
        # Specific symbols: delete in batches (SQLite variable limit is 999)
        for j in range(0, len(symbols), 500):
            batch_syms = symbols[j:j + 500]
            ph = ','.join(['?'] * len(batch_syms))
            conn.execute(f"DELETE FROM backtest_daily_scores WHERE symbol IN ({ph})", batch_syms)
            conn.execute(f"DELETE FROM backtest_signals WHERE symbol IN ({ph})", batch_syms)
    else:
        # Full run: clear everything
        conn.execute("DELETE FROM backtest_daily_scores")
        conn.execute("DELETE FROM backtest_signals")
    conn.commit()

    # Compute SPY market regime once
    log.info("Computing SPY market regime...")
    spy_regime = compute_spy_regime(conn)
    log.info(f"  SPY regime data: {len(spy_regime)} trading days")
    bullish_pct = spy_regime['market_bullish'].mean() * 100
    log.info(f"  Bullish days: {bullish_pct:.1f}%")

    start_time = datetime.now()
    log.info(f"\nBacktesting {len(symbols)} stocks...")

    total_scores = 0
    total_signals = 0
    processed = 0
    skipped = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        scores_list = []
        signals_list = []

        for symbol in batch:
            scores_df, signals_df = process_stock(symbol, conn, spy_regime)
            if scores_df is not None:
                scores_list.append(scores_df)
                processed += 1
            else:
                skipped += 1

            if signals_df is not None and not signals_df.empty:
                signals_list.append(signals_df)

        # Save batch
        batch_scores = save_scores_batch(scores_list, conn)
        batch_signals = save_signals_batch(signals_list, conn)
        conn.commit()

        total_scores += batch_scores
        total_signals += batch_signals

        done = min(i + BATCH_SIZE, len(symbols))
        log.info(f"  Progress: {done}/{len(symbols)} stocks | "
                 f"{total_scores:,} score rows | {total_signals:,} signals")

    conn.close()

    elapsed = datetime.now() - start_time
    log.info("\n" + "=" * 60)
    log.info(f"Backtest complete!")
    log.info(f"  Stocks processed: {processed:,} (skipped {skipped} with < 200 days)")
    log.info(f"  Daily scores: {total_scores:,} rows")
    log.info(f"  Signals detected: {total_signals:,}")
    log.info(f"  Runtime: {elapsed}")
    log.info("=" * 60)

    # Show summary
    show_summary()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyst Accuracy Scorer
=======================
Provides accuracy-weighted analyst signal scoring.
Uses historical accuracy to weight current recommendations.

IMPORTANT: Rolling window accuracy is now available to eliminate look-ahead bias.
- Use get_rolling_top3_analysts(sector, as_of_date) for backtest scenarios
- Use the hardcoded SECTOR_TOP3_ANALYSTS for live/production scoring

Integration with scoring system:
    from analyst_accuracy_scorer import get_analyst_signal_boost
    boost = get_analyst_signal_boost(symbol, sector, recent_grades)

For backtesting (no look-ahead bias):
    from analyst_accuracy_scorer import get_rolling_analyst_signal_score
    signal = get_rolling_analyst_signal_score(symbol, sector, as_of_date)
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from functools import lru_cache

PROJECT_ROOT = Path(__file__).parent
DATABASE_NAME = str(PROJECT_ROOT / 'nasdaq_stocks.db')
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Cache for rolling analyst accuracy (cleared when parameters change)
_rolling_accuracy_cache = {}

# Top analysts by overall hit rate (hardcoded for performance)
# These are updated periodically by running analyze_analyst_accuracy.py
TOP_ANALYSTS = {
    'Hovde Group': 0.738,
    'Daiwa Capital': 0.690,
    'Melius Research': 0.667,
    'Atlantic Equities': 0.640,
    'Stephens & Co.': 0.630,
    'Telsey Advisory Group': 0.630,
    'New Street Research': 0.625,
    'Northland Capital Markets': 0.620,
    'Cantor Fitzgerald': 0.604,
    'Bernstein': 0.602,
}

# Worst analysts (potential contrarian signals)
WORST_ANALYSTS = {
    'Odeon Capital': 0.365,
    'Lake Street': 0.385,
    'DZ Bank': 0.408,
    'Berenberg': 0.419,
    'MoffettNathanson': 0.426,
    'Sidoti & Co.': 0.424,
}

# TOP 3 ANALYSTS PER SECTOR (validated by backtest - 71% hit rate, +0.254 correlation)
# Using top 3 instead of top 1 for better sample size while maintaining high accuracy
SECTOR_TOP3_ANALYSTS = {
    'Basic Materials': [
        ('Raymond James', 0.786),
        ('Deutsche Bank', 0.771),
        ('JP Morgan', 0.695),
    ],
    'Communication Services': [
        ('Bernstein', 0.769),
        ('Benchmark', 0.727),
        ('UBS', 0.700),
    ],
    'Consumer Cyclical': [
        ('CFRA', 0.727),
        ('Canaccord Genuity', 0.700),
        ('Telsey Advisory Group', 0.690),
    ],
    'Consumer Defensive': [
        ('Evercore ISI Group', 0.818),
        ('Stephens & Co.', 0.800),
        ('Bernstein', 0.786),
    ],
    'Energy': [
        ('Roth MKM', 0.812),
        ('Evercore ISI Group', 0.722),
        ('Wells Fargo', 0.672),
    ],
    'Financial Services': [
        ('Hovde Group', 0.744),
        ('Atlantic Equities', 0.714),
        ('Deutsche Bank', 0.676),
    ],
    'Healthcare': [
        ('Bernstein', 0.688),
        ('Guggenheim', 0.680),
        ('William Blair', 0.652),
    ],
    'Industrials': [
        ('CJS Securities', 0.800),
        ('Seaport Global', 0.762),
        ('Stephens & Co.', 0.700),
    ],
    'Real Estate': [
        ('Argus Research', 0.800),
        ('RBC Capital', 0.750),
        ('BTIG', 0.727),
    ],
    'Technology': [
        ('Daiwa Capital', 0.929),  # 92.9% - exceptional!
        ('Redburn Atlantic', 0.769),
        ('Northland Capital Markets', 0.712),
    ],
    'Utilities': [
        ('Wolfe Research', 0.684),
        ('B of A Securities', 0.659),
        ('Citigroup', 0.636),
    ],
}

# Average hit rate (baseline)
BASELINE_HIT_RATE = 0.527

# Rolling window settings (for bias-free backtesting)
ROLLING_WINDOW_YEARS = 3  # Use 3 years of historical data
FORWARD_DAYS = 63  # 3-month forward returns for accuracy calculation
MIN_CALLS_FOR_RANKING = 15  # Minimum calls to be considered for top 3


def compute_rolling_analyst_accuracy(as_of_date: str, sector: str = None,
                                      training_years: int = 3,
                                      forward_days: int = 63) -> pd.DataFrame:
    """
    Compute analyst accuracy using ONLY data before as_of_date.

    This eliminates look-ahead bias by only using historical information
    that would have been available at the decision point.

    Args:
        as_of_date: The date to compute accuracy AS OF (YYYY-MM-DD)
        sector: Optional sector to filter grades
        training_years: Years of historical data to use
        forward_days: Days to measure forward return

    Returns:
        DataFrame with columns: grading_company, sector, hit_rate, avg_return, n_calls
    """
    conn = sqlite3.connect(BACKTEST_DB)

    # Training window: grades from (as_of_date - training_years) to (as_of_date - forward_days)
    # We need forward_days gap to ensure we have forward return data
    as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')
    train_end = as_of_dt - timedelta(days=forward_days)
    train_start = as_of_dt - timedelta(days=training_years * 365)

    train_start_str = train_start.strftime('%Y-%m-%d')
    train_end_str = train_end.strftime('%Y-%m-%d')

    # Load grades in training window
    sector_filter = f"AND sector = '{sector}'" if sector else ""

    grades_query = f"""
        SELECT g.symbol, g.date, g.grading_company, g.action,
               s.sector
        FROM historical_grades g
        LEFT JOIN (SELECT DISTINCT symbol, sector FROM stock_consensus) s
            ON g.symbol = s.symbol
        WHERE g.date >= ? AND g.date <= ?
        AND g.action IN ('upgrade', 'downgrade')
        {sector_filter}
        ORDER BY g.symbol, g.date
    """

    try:
        grades = pd.read_sql_query(grades_query, conn,
                                   params=(train_start_str, train_end_str))
    except Exception as e:
        # stock_consensus may not exist - try without sector
        grades_query_simple = """
            SELECT symbol, date, grading_company, action
            FROM historical_grades
            WHERE date >= ? AND date <= ?
            AND action IN ('upgrade', 'downgrade')
            ORDER BY symbol, date
        """
        grades = pd.read_sql_query(grades_query_simple, conn,
                                   params=(train_start_str, train_end_str))
        grades['sector'] = 'Unknown'

    if grades.empty:
        conn.close()
        return pd.DataFrame()

    # Load prices for forward return calculation
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close
        FROM historical_prices
        WHERE date >= ? AND date <= ?
        ORDER BY symbol, date
    """, conn, params=(train_start_str, as_of_date))

    conn.close()

    if prices.empty:
        return pd.DataFrame()

    # Calculate forward returns for each grade
    results = []

    for symbol in grades['symbol'].unique():
        symbol_grades = grades[grades['symbol'] == symbol]
        symbol_prices = prices[prices['symbol'] == symbol].sort_values('date')

        if len(symbol_prices) < forward_days + 10:
            continue

        price_dict = dict(zip(symbol_prices['date'], symbol_prices['adjusted_close']))
        dates_list = symbol_prices['date'].tolist()

        for _, grade in symbol_grades.iterrows():
            grade_date = grade['date']

            # Find price on grade date
            if grade_date not in price_dict:
                close_dates = [d for d in dates_list if d >= grade_date]
                if not close_dates:
                    continue
                grade_date = close_dates[0]

            entry_price = price_dict.get(grade_date)
            if not entry_price or entry_price <= 0:
                continue

            # Find forward date
            try:
                idx = dates_list.index(grade_date)
                if idx + forward_days >= len(dates_list):
                    continue
                forward_date = dates_list[idx + forward_days]
                exit_price = price_dict.get(forward_date)
            except (ValueError, IndexError):
                continue

            if not exit_price or exit_price <= 0:
                continue

            forward_return = ((exit_price / entry_price) - 1) * 100

            # Determine if call was correct
            is_upgrade = grade['action'] == 'upgrade'
            correct = (forward_return > 0) if is_upgrade else (forward_return < 0)

            results.append({
                'grading_company': grade['grading_company'],
                'sector': grade['sector'],
                'action': grade['action'],
                'forward_return': forward_return,
                'correct': correct,
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # Aggregate by analyst + sector
    stats = df.groupby(['grading_company', 'sector']).agg({
        'correct': ['sum', 'count', 'mean'],
        'forward_return': 'mean',
    })
    stats.columns = ['correct_calls', 'n_calls', 'hit_rate', 'avg_return']
    stats = stats.reset_index()

    # Filter to analysts with enough calls
    stats = stats[stats['n_calls'] >= MIN_CALLS_FOR_RANKING]

    return stats


def get_rolling_top3_analysts(sector: str, as_of_date: str) -> List[tuple]:
    """
    Get top 3 analysts for a sector using ONLY data before as_of_date.

    This is the bias-free version for backtesting.

    Args:
        sector: The sector to get top analysts for
        as_of_date: Date to compute rankings as of (YYYY-MM-DD)

    Returns:
        List of (analyst_name, hit_rate) tuples for top 3
    """
    # Check cache
    cache_key = f"{sector}_{as_of_date}"
    if cache_key in _rolling_accuracy_cache:
        return _rolling_accuracy_cache[cache_key]

    # Compute rolling accuracy
    accuracy = compute_rolling_analyst_accuracy(as_of_date, sector=sector)

    if accuracy.empty:
        _rolling_accuracy_cache[cache_key] = []
        return []

    # Filter to this sector
    sector_accuracy = accuracy[accuracy['sector'] == sector]

    if sector_accuracy.empty:
        _rolling_accuracy_cache[cache_key] = []
        return []

    # Get top 3 by hit rate
    top3 = sector_accuracy.nlargest(3, 'hit_rate')
    result = [(row['grading_company'], row['hit_rate'])
              for _, row in top3.iterrows()]

    # Cache result
    _rolling_accuracy_cache[cache_key] = result

    return result


def clear_rolling_accuracy_cache():
    """Clear the rolling accuracy cache."""
    global _rolling_accuracy_cache
    _rolling_accuracy_cache = {}


def is_rolling_top3_analyst(analyst: str, sector: str, as_of_date: str) -> bool:
    """Check if analyst is in rolling top 3 for sector as of a date."""
    top3 = get_rolling_top3_analysts(sector, as_of_date)
    return analyst in [name for name, _ in top3]


def get_rolling_analyst_signal_score(symbol: str, sector: str,
                                      as_of_date: str,
                                      lookback_days: int = 90) -> Dict:
    """
    Calculate analyst signal score using rolling accuracy (no look-ahead bias).

    This version uses get_rolling_top3_analysts() instead of hardcoded SECTOR_TOP3_ANALYSTS.

    Args:
        symbol: Stock ticker
        sector: Stock sector
        as_of_date: Date to compute signal as of (YYYY-MM-DD)
        lookback_days: Days to look back for recent grades

    Returns:
        Same format as calculate_analyst_signal_score()
    """
    conn = sqlite3.connect(BACKTEST_DB)

    as_of_dt = datetime.strptime(as_of_date, '%Y-%m-%d')
    cutoff = (as_of_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')

    try:
        grades = pd.read_sql_query("""
            SELECT grading_company, new_grade, action, date
            FROM historical_grades
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date DESC
        """, conn, params=(symbol, cutoff, as_of_date))
    except:
        conn.close()
        return {
            'analyst_signal_score': 0,
            'upgrade_count': 0,
            'downgrade_count': 0,
            'top3_upgrade_count': 0,
            'top3_downgrade_count': 0,
            'top_analyst_actions': [],
        }

    conn.close()

    if grades.empty:
        return {
            'analyst_signal_score': 0,
            'upgrade_count': 0,
            'downgrade_count': 0,
            'top3_upgrade_count': 0,
            'top3_downgrade_count': 0,
            'top_analyst_actions': [],
        }

    # Get rolling top 3 analysts
    rolling_top3 = get_rolling_top3_analysts(sector, as_of_date)
    top3_names = [name for name, _ in rolling_top3]

    upgrades = 0
    downgrades = 0
    top3_upgrades = 0
    top3_downgrades = 0
    other_upgrades = 0
    top_actions = []

    for _, grade in grades.iterrows():
        analyst = grade.get('grading_company', '')
        action = grade.get('action', '').lower()

        is_top3 = analyst in top3_names

        if action == 'upgrade':
            upgrades += 1
            if is_top3:
                top3_upgrades += 1
                top_actions.append(f"⭐ {analyst}: Upgrade (Rolling Top 3 {sector})")
            elif analyst in TOP_ANALYSTS:
                other_upgrades += 1
                top_actions.append(f"{analyst}: Upgrade")

        elif action == 'downgrade':
            downgrades += 1
            if is_top3:
                top3_downgrades += 1
                top_actions.append(f"⚠️ {analyst}: Downgrade (Rolling Top 3 {sector})")

    # Calculate score
    analyst_signal_score = 0
    analyst_signal_score += min(top3_upgrades * 5, 15)
    analyst_signal_score += min(other_upgrades * 2, 6)
    analyst_signal_score -= min(top3_downgrades * 5, 10)
    analyst_signal_score = max(-10, min(analyst_signal_score, 15))

    return {
        'analyst_signal_score': round(analyst_signal_score, 2),
        'upgrade_count': upgrades,
        'downgrade_count': downgrades,
        'top3_upgrade_count': top3_upgrades,
        'top3_downgrade_count': top3_downgrades,
        'top_analyst_actions': top_actions,
    }


def get_analyst_accuracy(analyst: str, sector: str = None) -> float:
    """
    Get the accuracy rating for an analyst, optionally sector-adjusted.

    Returns a multiplier:
    - > 1.0: More accurate than baseline
    - < 1.0: Less accurate than baseline
    """
    # Check if in top 3 for sector (highest priority - best signal)
    if sector and sector in SECTOR_TOP3_ANALYSTS:
        for top_analyst, hit_rate in SECTOR_TOP3_ANALYSTS[sector]:
            if analyst == top_analyst:
                return hit_rate / BASELINE_HIT_RATE  # Strong boost for sector expert

    # Check top analysts (overall)
    if analyst in TOP_ANALYSTS:
        return TOP_ANALYSTS[analyst] / BASELINE_HIT_RATE

    # Check worst analysts (contrarian potential)
    if analyst in WORST_ANALYSTS:
        return WORST_ANALYSTS[analyst] / BASELINE_HIT_RATE

    # Unknown analyst - use baseline
    return 1.0


def is_top3_analyst(analyst: str, sector: str) -> bool:
    """Check if an analyst is in the top 3 for a sector."""
    if not sector or sector not in SECTOR_TOP3_ANALYSTS:
        return False
    top3_names = [name for name, _ in SECTOR_TOP3_ANALYSTS[sector]]
    return analyst in top3_names


def get_recent_grades(symbol: str, days: int = 90) -> List[Dict]:
    """Get recent analyst grades for a symbol."""
    conn = sqlite3.connect(DATABASE_NAME)

    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    # Check if we have a grades table in main DB
    # If not, return empty - grades are typically only in backtest.db
    try:
        grades = pd.read_sql_query("""
            SELECT grading_company, new_grade, action, date
            FROM analyst_grades
            WHERE symbol = ? AND date >= ?
            ORDER BY date DESC
        """, conn, params=(symbol, cutoff))
    except:
        conn.close()
        # Try backtest DB
        conn = sqlite3.connect(BACKTEST_DB)
        try:
            grades = pd.read_sql_query("""
                SELECT grading_company, new_grade, action, date
                FROM historical_grades
                WHERE symbol = ? AND date >= ?
                ORDER BY date DESC
            """, conn, params=(symbol, cutoff))
        except:
            conn.close()
            return []

    conn.close()

    return grades.to_dict('records')


def calculate_analyst_signal_score(symbol: str, sector: str = None) -> Dict:
    """
    Calculate an analyst signal score for a stock (V3 validated).

    IMPORTANT FINDINGS from 2022+ backtest analysis:
    - Top 3 Upgrades: 61.5% hit rate, +3.50% avg return → Bullish signal
    - Top 3 Downgrades: 66.7% hit rate (for drops), -6.06% avg return → Bearish signal
    - Spread: +9.56% between upgrades and downgrades

    Both directions are predictive for Top 3 analysts per sector.

    Returns:
        {
            'analyst_signal_score': float (-10 to +15),
            'upgrade_count': int,
            'downgrade_count': int,
            'top3_upgrade_count': int,
            'top3_downgrade_count': int,
            'top_analyst_actions': list,
        }
    """
    grades = get_recent_grades(symbol, days=90)

    if not grades:
        return {
            'analyst_signal_score': 0,
            'upgrade_count': 0,
            'downgrade_count': 0,
            'top3_upgrade_count': 0,
            'top3_downgrade_count': 0,
            'top_analyst_actions': [],
        }

    upgrades = 0
    downgrades = 0
    top3_upgrades = 0
    top3_downgrades = 0
    other_upgrades = 0
    top_actions = []

    for grade in grades:
        analyst = grade.get('grading_company', '')
        action = grade.get('action', '').lower()

        is_top3 = is_top3_analyst(analyst, sector)

        if action == 'upgrade':
            upgrades += 1
            if is_top3:
                top3_upgrades += 1
                top_actions.append(f"⭐ {analyst}: Upgrade (Top 3 {sector})")
            elif analyst in TOP_ANALYSTS:
                other_upgrades += 1
                top_actions.append(f"{analyst}: Upgrade")

        elif action == 'downgrade':
            downgrades += 1
            if is_top3:
                top3_downgrades += 1
                top_actions.append(f"⚠️ {analyst}: Downgrade (Top 3 {sector})")

    # Calculate score using BOTH upgrades (+) and downgrades (-)
    # Based on 2022+ data: both directions are predictive for Top 3 analysts
    #
    # Scoring:
    # - Top 3 sector upgrade:   +5 points each (max +15)
    # - Top 3 sector downgrade: -5 points each (min -10)
    # - Other top analyst upgrade: +2 points each (max +6)
    #
    analyst_signal_score = 0

    # Positive: upgrades
    analyst_signal_score += min(top3_upgrades * 5, 15)      # Top 3 sector upgrades
    analyst_signal_score += min(other_upgrades * 2, 6)      # Other top analyst upgrades

    # Negative: downgrades (from top 3 only - they're the ones that are predictive)
    analyst_signal_score -= min(top3_downgrades * 5, 10)    # Top 3 sector downgrades

    # Clamp to range [-10, 15]
    analyst_signal_score = max(-10, min(analyst_signal_score, 15))

    return {
        'analyst_signal_score': round(analyst_signal_score, 2),
        'upgrade_count': upgrades,
        'downgrade_count': downgrades,
        'top3_upgrade_count': top3_upgrades,
        'top3_downgrade_count': top3_downgrades,
        'top_analyst_actions': top_actions,
    }


def get_analyst_signal_boost(symbol: str, sector: str, current_score: float) -> float:
    """
    Get a score boost/penalty based on analyst signals.

    Args:
        symbol: Stock ticker
        sector: Stock sector
        current_score: Current long-term or value score

    Returns:
        Adjusted score (original score + analyst boost)
    """
    signal = calculate_analyst_signal_score(symbol, sector)

    # Boost is +/- up to 5 points based on analyst signal
    boost = signal['analyst_signal_score'] / 2  # Scale to +/- 5

    return current_score + boost


# Quick test
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test analyst accuracy scoring')
    parser.add_argument('--rolling', action='store_true',
                        help='Test rolling accuracy (bias-free)')
    parser.add_argument('--date', default='2024-01-01',
                        help='As-of date for rolling test (YYYY-MM-DD)')
    args = parser.parse_args()

    print("=== Analyst Accuracy Scorer Test ===")

    # Test analyst accuracy lookup
    print("\nAnalyst Accuracy Multipliers (Hardcoded):")
    for analyst in ['Hovde Group', 'Bernstein', 'Unknown Analyst', 'Odeon Capital']:
        mult = get_analyst_accuracy(analyst)
        print(f"  {analyst}: {mult:.2f}x")

    # Test sector specialist
    print("\nSector Specialist Lookup (Hardcoded):")
    print(f"  Bernstein for Healthcare: {get_analyst_accuracy('Bernstein', 'Healthcare'):.2f}x")
    print(f"  Bernstein for Technology: {get_analyst_accuracy('Bernstein', 'Technology'):.2f}x")

    # Test signal calculation (if data available)
    print("\nTesting signal calculation for AAPL (Hardcoded Top 3):")
    signal = calculate_analyst_signal_score('AAPL', 'Technology')
    print(f"  Signal Score: {signal['analyst_signal_score']}")
    print(f"  Upgrades: {signal['upgrade_count']}, Downgrades: {signal['downgrade_count']}")

    if args.rolling:
        print(f"\n=== Rolling Accuracy Test (as of {args.date}) ===")
        print("(Using only data BEFORE this date - no look-ahead bias)\n")

        # Test rolling top 3 for each sector
        sectors = ['Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical']

        for sector in sectors:
            print(f"\n{sector} - Rolling Top 3 Analysts:")
            top3 = get_rolling_top3_analysts(sector, args.date)
            if top3:
                for analyst, hit_rate in top3:
                    print(f"  {analyst}: {hit_rate:.1%}")
            else:
                print("  (No data available)")

        # Compare rolling vs hardcoded for Technology
        print(f"\n=== Comparison: Rolling vs Hardcoded (Technology) ===")
        print("\nHardcoded Top 3:")
        for analyst, hit_rate in SECTOR_TOP3_ANALYSTS.get('Technology', []):
            print(f"  {analyst}: {hit_rate:.1%}")

        print(f"\nRolling Top 3 (as of {args.date}):")
        rolling_top3 = get_rolling_top3_analysts('Technology', args.date)
        for analyst, hit_rate in rolling_top3:
            print(f"  {analyst}: {hit_rate:.1%}")

        # Test rolling signal score
        print(f"\n=== Rolling Signal Score for AAPL (as of {args.date}) ===")
        rolling_signal = get_rolling_analyst_signal_score('AAPL', 'Technology', args.date)
        print(f"  Signal Score: {rolling_signal['analyst_signal_score']}")
        print(f"  Top 3 Upgrades: {rolling_signal['top3_upgrade_count']}")
        print(f"  Top 3 Downgrades: {rolling_signal['top3_downgrade_count']}")

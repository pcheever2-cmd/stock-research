#!/usr/bin/env python3
"""
Options Strategy Backtester
============================
Tests call options strategies using the same bucket signals.

Key differences from stock trading:
- Leverage: Options amplify gains and losses (2-5x typical)
- Time decay: Options lose value over time (theta)
- Max loss: Can lose 100% of premium
- Expiration: Must match hold period to expiration

Scenarios tested:
1. ATM Calls (At-The-Money) - Delta ~0.50, moderate leverage
2. Slightly OTM Calls (5% OTM) - Delta ~0.35, higher leverage
3. Deep OTM Calls (10% OTM) - Delta ~0.25, maximum leverage
4. Conservative ATM - Smaller positions, tighter management
5. Earnings Beat + ATM - Only trade with earnings confirmation
6. Strong Sectors + ATM - Sector-filtered with options

Features:
- Full compounding (reinvest profits)
- Detailed trade tracking (buys/sells/activity)
- 2-year time horizon (2024-2025)
- Realistic option pricing model
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import math

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

from bucket_config import BUCKETS, BUCKET_1, BUCKET_2, BUCKET_3


# ══════════════════════════════════════════════════════════════════════════════
# OPTIONS PRICING MODEL (Simplified but realistic)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionParams:
    """Parameters for option pricing."""
    strike_pct: float      # Strike as % of stock price (1.0 = ATM, 1.05 = 5% OTM)
    delta: float           # Option delta (sensitivity to stock price)
    theta_daily: float     # Daily time decay as % of premium
    premium_pct: float     # Premium as % of stock price
    gamma: float           # Rate of delta change
    vega: float            # Sensitivity to volatility


# Predefined option types (realistic 60-90 day options)
OPTION_TYPES = {
    'ATM': OptionParams(
        strike_pct=1.00,
        delta=0.50,
        theta_daily=0.005,     # ~0.5% daily decay
        premium_pct=0.06,      # 6% of stock price for 2-3 month ATM
        gamma=0.02,
        vega=0.15,
    ),
    'OTM_5': OptionParams(
        strike_pct=1.05,
        delta=0.35,
        theta_daily=0.006,     # Slightly higher decay for OTM
        premium_pct=0.035,     # 3.5% premium (cheaper)
        gamma=0.025,
        vega=0.12,
    ),
    'OTM_10': OptionParams(
        strike_pct=1.10,
        delta=0.22,
        theta_daily=0.007,     # Higher decay
        premium_pct=0.02,      # 2% premium (cheapest)
        gamma=0.03,
        vega=0.10,
    ),
}


def calculate_option_return(stock_return_pct: float, hold_days: int,
                            option_type: str = 'ATM',
                            volatility_change: float = 0.0) -> float:
    """
    Calculate option return based on stock return.

    Uses a realistic leverage-based model:
    - Options provide leverage = stock_price / premium
    - Time decay reduces value over time
    - Winning trades can have asymmetric upside

    Args:
        stock_return_pct: Stock return in percentage (e.g., 10.0 for +10%)
        hold_days: Number of days held
        option_type: 'ATM', 'OTM_5', or 'OTM_10'
        volatility_change: Change in implied volatility (e.g., 0.05 for +5%)

    Returns:
        Option return in percentage
    """
    params = OPTION_TYPES[option_type]
    stock_return = stock_return_pct / 100  # Convert to decimal

    # Calculate effective leverage
    # ATM: ~12x leverage (8% premium), OTM_5: ~20x, OTM_10: ~33x
    leverage = 1.0 / params.premium_pct

    # Time decay cost (as % of premium lost)
    # More realistic: lose about 30-50% of premium over 60 days if flat
    hold_fraction = min(1.0, hold_days / 90)
    time_decay_cost = 0.35 * hold_fraction  # 35% of premium over full period

    if stock_return > 0:
        # WINNING TRADE
        # Calculate intrinsic value gain
        # For ATM: if stock up 10%, option up ~60-80% (less time value lost)
        # Leverage effect but capped by actual intrinsic gain

        # Intrinsic gain = max(0, stock_price * (1 + return) - strike) / premium - 1
        # For ATM (strike = stock_price):
        intrinsic_pct = stock_return / params.premium_pct  # e.g., 10% / 8% = 125%

        # But we also have remaining time value (less decay on winners)
        time_value_remaining = (1 - time_decay_cost * 0.5)  # Winners lose less time value

        # Option return = intrinsic gain * delta_factor + remaining_time_value - 1
        # Simplified: leverage * stock_return - reduced_decay
        effective_leverage = params.delta * leverage
        option_return = effective_leverage * stock_return - (time_decay_cost * 0.3)

        # Big winners get gamma boost
        if stock_return > 0.15:  # >15% stock gain
            option_return *= 1.3  # Gamma acceleration

    else:
        # LOSING TRADE
        # Options lose faster on the downside due to delta + theta

        # Delta loss
        delta_loss = params.delta * abs(stock_return) / params.premium_pct

        # Time decay hits harder on losers
        total_loss = delta_loss + time_decay_cost

        option_return = -total_loss

        # If stock down significantly, likely total loss
        if stock_return < -0.15:
            option_return = max(-0.95, option_return)  # Near total loss

    # Cap losses at -100% (can't lose more than premium)
    option_return = max(-1.0, option_return)

    # Cap gains at realistic levels (300-400% for big winners)
    option_return = min(4.0, option_return)

    return option_return * 100  # Convert back to percentage


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data_2year():
    """Load signal data for 2-year period (2024-2025)."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Filter to 2024-2025 for 2-year horizon
    df = pd.read_sql_query("""
        SELECT
            s.symbol, s.date, s.signal_type,
            s.lt_score, s.value_score, s.value_score_v2, s.close_price,
            s.return_1w, s.return_1m, s.return_3m, s.return_6m, s.return_1y,
            d.rsi, d.adx, d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth,
            d.fundamentals_score, d.market_risk_score
        FROM backtest_signals s
        LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
        WHERE s.date >= '2024-01-01' AND s.return_3m IS NOT NULL
    """, conn)

    # Supplementary data
    mcaps = pd.read_sql_query("""
        SELECT symbol, market_cap FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol HAVING date = MAX(date)
    """, conn).set_index('symbol')['market_cap']

    acov = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot GROUP BY symbol
    """, conn).set_index('symbol')['analyst_count']

    # Earnings surprise
    income = pd.read_sql_query("""
        SELECT symbol, fiscal_year, SUM(eps_diluted) as actual_eps, COUNT(*) as quarters
        FROM historical_income_statements
        WHERE eps_diluted IS NOT NULL AND fiscal_year IS NOT NULL
        GROUP BY symbol, fiscal_year HAVING quarters >= 4
    """, conn)
    estimates = pd.read_sql_query("""
        SELECT symbol, fiscal_year, eps_avg FROM analyst_estimates_snapshot
        WHERE eps_avg IS NOT NULL AND fiscal_year IS NOT NULL
    """, conn)
    conn.close()

    conn2 = sqlite3.connect(MAIN_DB)
    sectors = pd.read_sql_query(
        "SELECT symbol, sector FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    # Enrich
    df['market_cap'] = df['symbol'].map(mcaps)
    df['analyst_count'] = df['symbol'].map(acov)
    df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    df['month'] = df['date_dt'].dt.month

    # Earnings surprise
    if not income.empty and not estimates.empty:
        merged = income.merge(estimates, on=['symbol', 'fiscal_year'], how='inner')
        if not merged.empty:
            merged = merged.sort_values('fiscal_year').groupby('symbol').last().reset_index()
            merged['earnings_surprise_pct'] = np.where(
                merged['eps_avg'].abs() > 0.01,
                (merged['actual_eps'] - merged['eps_avg']) / merged['eps_avg'].abs() * 100,
                np.nan
            )
            surprise_map = merged.set_index('symbol')['earnings_surprise_pct'].clip(-200, 500)
            df['earnings_surprise_pct'] = df['symbol'].map(surprise_map)
        else:
            df['earnings_surprise_pct'] = np.nan
    else:
        df['earnings_surprise_pct'] = np.nan

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OptionsStrategy:
    """Options trading strategy configuration."""
    name: str
    description: str
    option_type: str              # 'ATM', 'OTM_5', 'OTM_10'
    position_pct: float           # % of capital per trade
    max_positions: int            # max concurrent positions
    stop_loss_pct: float          # exit if option down this much
    profit_target_pct: float      # optional: take profit at this level (0 = no target)
    hold_days: int                # target hold period
    require_earnings_beat: bool = False
    require_strong_sector: bool = False


def get_options_strategies() -> List[OptionsStrategy]:
    """
    Define options strategies to test.

    CONSTRAINTS:
    - Max stop loss: 25% (tight risk management)
    - Cash only: No margin/leverage - position sizes based on available cash
    - All positions sized as % of current capital
    """
    strategies = []

    # 1. ATM Baseline - Tight stops
    strategies.append(OptionsStrategy(
        name="ATM_Baseline",
        description="ATM calls, 25% max stop, cash only",
        option_type='ATM',
        position_pct=3.0,
        max_positions=10,
        stop_loss_pct=-25.0,       # TIGHT STOP
        profit_target_pct=75.0,    # Take profits earlier
        hold_days=60,
    ))

    # 2. ATM Conservative - Very tight stops
    strategies.append(OptionsStrategy(
        name="ATM_Conservative",
        description="ATM calls, 20% stop, small positions",
        option_type='ATM',
        position_pct=2.0,
        max_positions=12,
        stop_loss_pct=-20.0,       # VERY TIGHT
        profit_target_pct=50.0,
        hold_days=45,
    ))

    # 3. ATM Moderate - Balanced approach
    strategies.append(OptionsStrategy(
        name="ATM_Moderate",
        description="ATM calls, 25% stop, moderate sizing",
        option_type='ATM',
        position_pct=2.5,
        max_positions=10,
        stop_loss_pct=-25.0,
        profit_target_pct=60.0,
        hold_days=45,
    ))

    # 4. OTM 5% Tight - Cheaper calls with tight stops
    strategies.append(OptionsStrategy(
        name="OTM5_TightStop",
        description="5% OTM calls, 25% stop",
        option_type='OTM_5',
        position_pct=2.0,
        max_positions=12,
        stop_loss_pct=-25.0,
        profit_target_pct=100.0,
        hold_days=60,
    ))

    # 5. ATM + Earnings Beat
    strategies.append(OptionsStrategy(
        name="ATM_EarningsBeat",
        description="ATM calls, earnings beat filter, 25% stop",
        option_type='ATM',
        position_pct=3.0,
        max_positions=8,
        stop_loss_pct=-25.0,
        profit_target_pct=75.0,
        hold_days=60,
        require_earnings_beat=True,
    ))

    # 6. ATM + Strong Sectors
    strategies.append(OptionsStrategy(
        name="ATM_StrongSectors",
        description="ATM calls, strong sectors only, 25% stop",
        option_type='ATM',
        position_pct=3.0,
        max_positions=8,
        stop_loss_pct=-25.0,
        profit_target_pct=75.0,
        hold_days=60,
        require_strong_sector=True,
    ))

    # 7. OTM 5% + Earnings Beat
    strategies.append(OptionsStrategy(
        name="OTM5_EarningsBeat",
        description="5% OTM, earnings beat, 25% stop",
        option_type='OTM_5',
        position_pct=2.5,
        max_positions=10,
        stop_loss_pct=-25.0,
        profit_target_pct=100.0,
        hold_days=60,
        require_earnings_beat=True,
    ))

    # 8. OTM 5% + Strong Sectors
    strategies.append(OptionsStrategy(
        name="OTM5_StrongSectors",
        description="5% OTM, strong sectors, 25% stop",
        option_type='OTM_5',
        position_pct=2.5,
        max_positions=10,
        stop_loss_pct=-25.0,
        profit_target_pct=100.0,
        hold_days=60,
        require_strong_sector=True,
    ))

    # 9. Quick Flip - Short holds
    strategies.append(OptionsStrategy(
        name="ATM_QuickFlip",
        description="ATM, 30-day hold, 20% stop, quick exits",
        option_type='ATM',
        position_pct=2.0,
        max_positions=15,
        stop_loss_pct=-20.0,
        profit_target_pct=40.0,
        hold_days=30,
    ))

    # 10. Combined Best - All filters, tight management
    strategies.append(OptionsStrategy(
        name="Combined_Best",
        description="OTM 5%, earnings beat, strong sectors, 25% stop",
        option_type='OTM_5',
        position_pct=3.0,
        max_positions=8,
        stop_loss_pct=-25.0,
        profit_target_pct=100.0,
        hold_days=60,
        require_earnings_beat=True,
        require_strong_sector=True,
    ))

    return strategies


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    bucket_id: int
    entry_date: str
    exit_date: str
    entry_price: float
    stock_return_pct: float
    option_return_pct: float
    position_value: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stop_loss', 'profit_target', 'expiration', 'hold_complete'
    hold_days: int


@dataclass
class SimulationResult:
    """Results from options simulation."""
    strategy_name: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    cagr_pct: float

    # Trade activity
    total_trades: int
    total_buys: int
    total_sells: int
    avg_trades_per_month: float

    # Performance
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float

    # Option-specific
    avg_option_return: float
    best_trade_pct: float
    worst_trade_pct: float
    total_premium_spent: float

    # Risk
    sharpe_ratio: float
    sortino_ratio: float

    # Breakdown
    wins: int
    losses: int
    stopped_out: int
    profit_targets_hit: int
    expirations: int


def simulate_options_strategy(df: pd.DataFrame, strategy: OptionsStrategy,
                               starting_capital: float = 100_000) -> Tuple[SimulationResult, List[Trade]]:
    """
    Run options strategy simulation with full compounding.

    Returns:
        SimulationResult and list of all trades
    """
    capital = starting_capital
    trades = []
    equity_curve = [(df['date_dt'].min(), capital)]
    max_equity = capital
    max_drawdown = 0

    # Get bucket signals
    all_signals = []
    for bucket in BUCKETS:
        sub = bucket.filter_fn(df)
        sub = sub[sub['return_3m'].notna()].copy()

        # Apply strategy filters
        if strategy.require_earnings_beat:
            sub = sub[sub['earnings_surprise_pct'] > 0]
        if strategy.require_strong_sector:
            sub = sub[sub['sector'].isin(bucket.strong_sectors)]

        sub['bucket_id'] = bucket.id
        all_signals.append(sub)

    combined = pd.concat(all_signals).sort_values('date_dt')
    if len(combined) == 0:
        return SimulationResult(
            strategy_name=strategy.name,
            starting_capital=starting_capital,
            ending_capital=starting_capital,
            total_return_pct=0, cagr_pct=0,
            total_trades=0, total_buys=0, total_sells=0,
            avg_trades_per_month=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0,
            max_drawdown_pct=0, avg_option_return=0,
            best_trade_pct=0, worst_trade_pct=0,
            total_premium_spent=0, sharpe_ratio=0, sortino_ratio=0,
            wins=0, losses=0, stopped_out=0,
            profit_targets_hit=0, expirations=0,
        ), []

    # Track active positions
    active_positions = []  # [(entry_date, symbol, premium_paid, entry_stock_price)]
    held_symbols = set()
    total_buys = 0
    total_sells = 0
    total_premium = 0
    stopped_out = 0
    profit_targets = 0
    expirations = 0
    monthly_returns = []
    current_month = None
    month_start_capital = capital

    for _, row in combined.iterrows():
        sig_date = row['date_dt']

        # Track monthly returns for Sharpe
        row_month = (sig_date.year, sig_date.month)
        if current_month is None:
            current_month = row_month
            month_start_capital = capital
        elif row_month != current_month:
            monthly_ret = (capital - month_start_capital) / month_start_capital
            monthly_returns.append(monthly_ret)
            current_month = row_month
            month_start_capital = capital

        # Check and close expired/stopped positions
        positions_to_close = []
        for i, (entry_date, sym, premium, entry_price, bucket_id) in enumerate(active_positions):
            days_held = (sig_date - entry_date).days

            # Get the stock return for this holding period
            # Approximate using the return_3m scaled by hold period
            sym_signals = combined[(combined['symbol'] == sym) &
                                   (combined['date_dt'] >= entry_date)]
            if len(sym_signals) > 0:
                stock_ret = sym_signals.iloc[0]['return_3m'] * (days_held / 90)
            else:
                stock_ret = 0

            # Calculate option return
            opt_ret = calculate_option_return(stock_ret, days_held, strategy.option_type)

            # Check exit conditions
            exit_reason = None
            if opt_ret <= strategy.stop_loss_pct:
                exit_reason = 'stop_loss'
                opt_ret = strategy.stop_loss_pct
                stopped_out += 1
            elif strategy.profit_target_pct > 0 and opt_ret >= strategy.profit_target_pct:
                exit_reason = 'profit_target'
                opt_ret = strategy.profit_target_pct
                profit_targets += 1
            elif days_held >= strategy.hold_days:
                exit_reason = 'hold_complete'
                expirations += 1

            if exit_reason:
                positions_to_close.append(i)

                # Calculate P&L
                exit_value = premium * (1 + opt_ret / 100)
                pnl = exit_value - premium

                # Record trade
                trades.append(Trade(
                    symbol=sym,
                    bucket_id=bucket_id,
                    entry_date=entry_date.strftime('%Y-%m-%d'),
                    exit_date=sig_date.strftime('%Y-%m-%d'),
                    entry_price=entry_price,
                    stock_return_pct=stock_ret,
                    option_return_pct=opt_ret,
                    position_value=premium,
                    pnl=pnl,
                    pnl_pct=opt_ret,
                    exit_reason=exit_reason,
                    hold_days=days_held,
                ))

                # Update capital (compounding)
                capital += exit_value
                total_sells += 1
                held_symbols.discard(sym)

        # Remove closed positions (in reverse order to maintain indices)
        for i in sorted(positions_to_close, reverse=True):
            active_positions.pop(i)

        # Check if we can open a new position
        if len(active_positions) >= strategy.max_positions:
            continue
        if row['symbol'] in held_symbols:
            continue

        # Calculate position size (compounding - use current capital)
        position_value = capital * (strategy.position_pct / 100)

        # Don't trade if position would be too small or capital depleted
        if position_value < 100 or capital < position_value:
            continue

        # Open position (buy call option)
        premium = position_value  # Premium paid = position value
        active_positions.append((
            sig_date,
            row['symbol'],
            premium,
            row['close_price'],
            row['bucket_id']
        ))
        held_symbols.add(row['symbol'])
        capital -= premium  # Deduct premium from capital
        total_buys += 1
        total_premium += premium

        # Track equity
        equity_curve.append((sig_date, capital + sum(p[2] for p in active_positions)))

        # Track drawdown
        current_equity = capital + sum(p[2] for p in active_positions)
        if current_equity > max_equity:
            max_equity = current_equity
        dd = (current_equity - max_equity) / max_equity * 100
        if dd < max_drawdown:
            max_drawdown = dd

    # Close any remaining positions at end
    for entry_date, sym, premium, entry_price, bucket_id in active_positions:
        days_held = (combined['date_dt'].max() - entry_date).days
        sym_signals = combined[(combined['symbol'] == sym)]
        if len(sym_signals) > 0:
            stock_ret = sym_signals.iloc[-1]['return_3m'] * (days_held / 90)
        else:
            stock_ret = 0

        opt_ret = calculate_option_return(stock_ret, days_held, strategy.option_type)
        exit_value = premium * (1 + opt_ret / 100)
        pnl = exit_value - premium

        trades.append(Trade(
            symbol=sym,
            bucket_id=bucket_id,
            entry_date=entry_date.strftime('%Y-%m-%d'),
            exit_date=combined['date_dt'].max().strftime('%Y-%m-%d'),
            entry_price=entry_price,
            stock_return_pct=stock_ret,
            option_return_pct=opt_ret,
            position_value=premium,
            pnl=pnl,
            pnl_pct=opt_ret,
            exit_reason='end_of_period',
            hold_days=days_held,
        ))

        capital += exit_value
        total_sells += 1
        expirations += 1

    # Calculate final metrics
    total_trades = len(trades)
    if total_trades == 0:
        return SimulationResult(
            strategy_name=strategy.name,
            starting_capital=starting_capital,
            ending_capital=capital,
            total_return_pct=0, cagr_pct=0,
            total_trades=0, total_buys=0, total_sells=0,
            avg_trades_per_month=0, win_rate=0,
            avg_win_pct=0, avg_loss_pct=0, profit_factor=0,
            max_drawdown_pct=max_drawdown, avg_option_return=0,
            best_trade_pct=0, worst_trade_pct=0,
            total_premium_spent=total_premium, sharpe_ratio=0, sortino_ratio=0,
            wins=0, losses=0, stopped_out=stopped_out,
            profit_targets_hit=profit_targets, expirations=expirations,
        ), trades

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
    avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
    pf = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

    # Time-based metrics
    date_range = (combined['date_dt'].max() - combined['date_dt'].min()).days
    years = max(0.1, date_range / 365.25)
    months = max(1, date_range / 30)
    total_return = (capital / starting_capital - 1) * 100
    cagr = ((capital / starting_capital) ** (1 / years) - 1) * 100 if capital > 0 else -100

    # Sharpe and Sortino
    if len(monthly_returns) > 2:
        sharpe = (np.mean(monthly_returns) / np.std(monthly_returns)) * np.sqrt(12) if np.std(monthly_returns) > 0 else 0
        downside = [r for r in monthly_returns if r < 0]
        sortino = (np.mean(monthly_returns) / np.std(downside)) * np.sqrt(12) if len(downside) > 0 and np.std(downside) > 0 else 0
    else:
        sharpe = 0
        sortino = 0

    all_returns = [t.pnl_pct for t in trades]

    return SimulationResult(
        strategy_name=strategy.name,
        starting_capital=starting_capital,
        ending_capital=capital,
        total_return_pct=total_return,
        cagr_pct=cagr,
        total_trades=total_trades,
        total_buys=total_buys,
        total_sells=total_sells,
        avg_trades_per_month=total_trades / months,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=max_drawdown,
        avg_option_return=np.mean(all_returns),
        best_trade_pct=max(all_returns),
        worst_trade_pct=min(all_returns),
        total_premium_spent=total_premium,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        wins=len(wins),
        losses=len(losses),
        stopped_out=stopped_out,
        profit_targets_hit=profit_targets,
        expirations=expirations,
    ), trades


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start = datetime.now()
    print("=" * 130)
    print("  OPTIONS STRATEGY BACKTEST — CALL OPTIONS")
    print(f"  Time Period: 2024-2025 (2 years)")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 130)

    print("\n  Loading 2-year data (2024-2025)...")
    df = load_data_2year()
    print(f"  Loaded {len(df):,} signals")
    print(f"  Date range: {df['date_dt'].min().date()} to {df['date_dt'].max().date()}")

    strategies = get_options_strategies()
    results = []
    all_trades = {}

    print(f"\n  Testing {len(strategies)} options strategies...")
    print(f"  {'─'*120}")

    for strat in strategies:
        result, trades = simulate_options_strategy(df, strat)
        results.append(result)
        all_trades[strat.name] = trades

        print(f"  {strat.name:<22s}: ${result.ending_capital:>10,.0f} ({result.total_return_pct:>+7.1f}%) | "
              f"{result.total_trades:>4} trades | {result.win_rate:>5.1f}% win | "
              f"DD: {result.max_drawdown_pct:>+6.1f}%")

    # Sort by return
    results.sort(key=lambda x: x.total_return_pct, reverse=True)

    # Print detailed comparison
    print(f"\n\n{'='*130}")
    print("  OPTIONS STRATEGY COMPARISON (sorted by return)")
    print(f"{'='*130}")

    print(f"\n  {'Strategy':<22s} {'End Cap':>12s} {'Return':>8s} {'CAGR':>7s} "
          f"{'Trades':>7s} {'Buys':>6s} {'Sells':>6s} {'Win%':>6s} {'PF':>5s} "
          f"{'MaxDD':>7s} {'Sharpe':>7s} {'AvgRet':>7s}")
    print(f"  {'─'*125}")

    for r in results:
        print(f"  {r.strategy_name:<22s} ${r.ending_capital:>10,.0f} {r.total_return_pct:>+7.1f}% "
              f"{r.cagr_pct:>+6.1f}% {r.total_trades:>6} {r.total_buys:>5} {r.total_sells:>5} "
              f"{r.win_rate:>5.1f}% {r.profit_factor:>4.2f} {r.max_drawdown_pct:>+6.1f}% "
              f"{r.sharpe_ratio:>6.2f} {r.avg_option_return:>+6.1f}%")

    # Trading activity summary
    print(f"\n\n{'='*130}")
    print("  TRADING ACTIVITY SUMMARY")
    print(f"{'='*130}")

    print(f"\n  {'Strategy':<22s} {'Total':>7s} {'Buys':>6s} {'Sells':>6s} {'Per Mo':>7s} "
          f"{'Stopped':>8s} {'Target':>8s} {'Expire':>8s} {'Best':>8s} {'Worst':>8s}")
    print(f"  {'─'*100}")

    for r in results:
        print(f"  {r.strategy_name:<22s} {r.total_trades:>6} {r.total_buys:>5} {r.total_sells:>5} "
              f"{r.avg_trades_per_month:>6.1f} {r.stopped_out:>7} {r.profit_targets_hit:>7} "
              f"{r.expirations:>7} {r.best_trade_pct:>+7.0f}% {r.worst_trade_pct:>+7.0f}%")

    # Win/Loss breakdown
    print(f"\n\n{'='*130}")
    print("  WIN/LOSS BREAKDOWN")
    print(f"{'='*130}")

    print(f"\n  {'Strategy':<22s} {'Wins':>6s} {'Losses':>7s} {'Win%':>6s} "
          f"{'Avg Win':>9s} {'Avg Loss':>9s} {'PF':>6s} {'Premium$':>12s}")
    print(f"  {'─'*90}")

    for r in results:
        print(f"  {r.strategy_name:<22s} {r.wins:>5} {r.losses:>6} {r.win_rate:>5.1f}% "
              f"{r.avg_win_pct:>+8.1f}% {r.avg_loss_pct:>+8.1f}% {r.profit_factor:>5.2f} "
              f"${r.total_premium_spent:>10,.0f}")

    # Comparison: Options vs Stocks (reference from previous backtest)
    print(f"\n\n{'='*130}")
    print("  OPTIONS vs STOCKS COMPARISON (same signals)")
    print(f"{'='*130}")
    print(f"\n  Reference: Stock strategies over 4 years returned +16% to +27%")
    print(f"  Options strategies over 2 years (2024-2025):")

    best = results[0]
    print(f"\n  Best Options Strategy: {best.strategy_name}")
    print(f"    Return: {best.total_return_pct:+.1f}% ({best.cagr_pct:+.1f}% CAGR)")
    print(f"    Trades: {best.total_trades} ({best.total_buys} buys, {best.total_sells} sells)")
    print(f"    Win Rate: {best.win_rate:.1f}%")
    print(f"    Max Drawdown: {best.max_drawdown_pct:+.1f}%")
    print(f"    Best Trade: {best.best_trade_pct:+.0f}%")
    print(f"    Worst Trade: {best.worst_trade_pct:+.0f}%")

    # Top 3 detailed
    print(f"\n\n{'='*130}")
    print("  TOP 3 STRATEGIES — DETAILED VIEW")
    print(f"{'='*130}")

    for i, r in enumerate(results[:3]):
        strat = next(s for s in strategies if s.name == r.strategy_name)
        print(f"\n  #{i+1} {r.strategy_name}")
        print(f"      Description: {strat.description}")
        print(f"      Option Type: {strat.option_type} | Position: {strat.position_pct}% | "
              f"Stop: {strat.stop_loss_pct}% | Target: {strat.profit_target_pct}%")
        print(f"      Filters: earnings_beat={strat.require_earnings_beat}, "
              f"strong_sector={strat.require_strong_sector}")
        print(f"      Results: ${r.starting_capital:,.0f} → ${r.ending_capital:,.0f} "
              f"({r.total_return_pct:+.1f}%)")
        print(f"      Activity: {r.total_buys} buys, {r.total_sells} sells, "
              f"{r.avg_trades_per_month:.1f}/month")
        print(f"      Wins: {r.wins} | Losses: {r.losses} | "
              f"Stopped: {r.stopped_out} | Targets: {r.profit_targets_hit}")

    # Recommendation
    print(f"\n\n{'='*130}")
    print("  RECOMMENDATION")
    print(f"{'='*130}")

    # Find best risk-adjusted
    best_sharpe = max(results, key=lambda x: x.sharpe_ratio)

    print(f"\n  For MAXIMUM RETURNS: {results[0].strategy_name}")
    print(f"    {results[0].total_return_pct:+.1f}% return, {results[0].win_rate:.1f}% win rate")

    if best_sharpe.strategy_name != results[0].strategy_name:
        print(f"\n  For BEST RISK-ADJUSTED: {best_sharpe.strategy_name}")
        print(f"    {best_sharpe.total_return_pct:+.1f}% return, Sharpe: {best_sharpe.sharpe_ratio:.2f}")

    # Warning about options
    print(f"\n  ⚠️  OPTIONS WARNING:")
    print(f"      - Options can lose 100% of premium (max loss in simulation: {min(r.worst_trade_pct for r in results):.0f}%)")
    print(f"      - Higher leverage = higher risk")
    print(f"      - Time decay works against you")
    print(f"      - These returns assume perfect execution (real slippage may differ)")

    elapsed = datetime.now() - start
    print(f"\n\n{'='*130}")
    print(f"  Options backtest complete! Runtime: {elapsed}")
    print(f"{'='*130}")

    return results, all_trades


if __name__ == "__main__":
    results, trades = main()

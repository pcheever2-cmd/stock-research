#!/usr/bin/env python3
"""
Strategy Backtester (v2 - Active Rotation)
==========================================
Tests multiple trading strategies with active rotation (buy/sell) similar to options trading.

Features:
- 2-year period (2024-2025) for direct comparison with options
- Active rotation: sells positions and reinvests into new opportunities
- Tracks all buys and sells with compounding reinvestment
- 0.1% slippage per leg
- Stop-loss execution with proper tracking

Strategies tested:
A. Baseline: Default bucket settings
B. Conservative: Smaller positions, tighter stops
C. Aggressive: Larger positions, wider stops
D. Earnings Beat Only: Only trade with earnings beat confirmed
E. Strong Sectors Only: Only trade in each bucket's strong sectors
F. Multi-Signal: Only trade when 2+ signals agree
G. Combined Best: Earnings beat + strong sectors + looser entry
H. Regime Adaptive: Heavier allocation in bear dip during bears
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')

# Import bucket config
from bucket_config import BUCKETS, BUCKET_1, BUCKET_2, BUCKET_3


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(start_year: int = None, end_year: int = None):
    """Load and enrich signal data, optionally filtered by year range."""
    conn = sqlite3.connect(BACKTEST_DB)

    # Build date filter
    date_filter = "s.return_3m IS NOT NULL"
    if start_year:
        date_filter += f" AND s.date >= '{start_year}-01-01'"
    if end_year:
        date_filter += f" AND s.date <= '{end_year}-12-31'"

    df = pd.read_sql_query(f"""
        SELECT
            s.symbol, s.date, s.signal_type,
            s.lt_score, s.value_score, s.value_score_v2, s.close_price,
            s.return_1w, s.return_1m, s.return_3m, s.return_6m, s.return_1y,
            d.rsi, d.adx, d.ev_ebitda, d.rev_growth, d.eps_growth, d.ebitda_growth,
            d.sma50, d.sma200, d.close,
            d.trend_score, d.fundamentals_score, d.valuation_score,
            d.momentum_score, d.market_risk_score
        FROM backtest_signals s
        LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
        WHERE {date_filter}
    """, conn)

    # Supplementary
    mcaps = pd.read_sql_query("""
        SELECT symbol, market_cap FROM historical_key_metrics
        WHERE market_cap IS NOT NULL
        GROUP BY symbol HAVING date = MAX(date)
    """, conn).set_index('symbol')['market_cap']

    cf = pd.read_sql_query("""
        SELECT symbol, date, free_cash_flow FROM historical_cash_flows
        WHERE free_cash_flow IS NOT NULL ORDER BY symbol, date
    """, conn)
    if not cf.empty:
        def get_yoy(g):
            if len(g) < 5:
                return pd.Series({'fcf_latest': g.iloc[-1]['free_cash_flow'], 'fcf_yago': None})
            return pd.Series({'fcf_latest': g.iloc[-1]['free_cash_flow'],
                              'fcf_yago': g.iloc[-5]['free_cash_flow']})
        fcf = cf.groupby('symbol').apply(get_yoy)
        fcf_positive = ((fcf['fcf_latest'] > 0) & (fcf['fcf_yago'].notna()) &
                        (fcf['fcf_latest'] > fcf['fcf_yago'])).astype(int)
    else:
        fcf_positive = pd.Series(dtype=int)

    acov = pd.read_sql_query("""
        SELECT symbol, MAX(num_analysts_eps) as analyst_count
        FROM analyst_estimates_snapshot GROUP BY symbol
    """, conn).set_index('symbol')['analyst_count']

    # DCF
    try:
        dcf = pd.read_sql_query("""
            SELECT symbol, dcf_upside_pct FROM dcf_valuations WHERE dcf_upside_pct IS NOT NULL
        """, conn).set_index('symbol')['dcf_upside_pct']
    except:
        dcf = pd.Series(dtype=float)

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
        "SELECT symbol, sector, industry FROM stock_consensus WHERE sector IS NOT NULL",
        conn2
    ).set_index('symbol')
    conn2.close()

    # Enrich
    df['market_cap'] = df['symbol'].map(mcaps)
    df['fcf_positive'] = df['symbol'].map(fcf_positive)
    df['analyst_count'] = df['symbol'].map(acov)
    df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
    df['market_bullish'] = (df['market_risk_score'] == 10).astype(int)
    df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
    df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)
    df['date_dt'] = pd.to_datetime(df['date'])
    df['year'] = df['date_dt'].dt.year
    df['dcf_upside'] = df['symbol'].map(dcf)

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

    # Ensemble voting
    vote_counts = df.groupby(['symbol', 'date']).size().reset_index(name='concurrent_signals')
    df = df.merge(vote_counts, on=['symbol', 'date'], how='left')

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Strategy:
    """A trading strategy configuration."""
    name: str
    description: str

    # Per-bucket overrides (bucket_id -> value)
    position_pct: Dict[int, float] = field(default_factory=dict)
    max_positions: Dict[int, int] = field(default_factory=dict)
    stop_loss_pct: Dict[int, float] = field(default_factory=dict)

    # Filters
    require_earnings_beat: bool = False
    require_big_beat: bool = False
    require_strong_sector: bool = False
    exclude_avoid_sector: bool = False
    require_multi_signal: int = 0  # 0 = no requirement

    # Allocation mode
    allocation_mode: str = 'equal'  # 'equal', 'regime_adaptive'

    # Bucket allocation weights (for equal mode)
    bucket_weights: Dict[int, float] = field(default_factory=dict)


def get_default_strategies() -> List[Strategy]:
    """
    Return all strategies to test.

    CONSTRAINT: Max stop loss = 10% for all long stock positions.
    This is tighter risk management to preserve capital and beat SPY.
    """
    strategies = []

    # A. Baseline - 10% stops across all buckets
    strategies.append(Strategy(
        name="A_Baseline",
        description="Default settings, 10% max stop",
        position_pct={1: 3.0, 2: 4.0, 3: 2.5},
        max_positions={1: 8, 2: 6, 3: 10},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},  # TIGHT 10% STOP
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # B. Conservative - smaller positions, 8% stops
    strategies.append(Strategy(
        name="B_Conservative",
        description="Smaller positions (2%), 8% stops",
        position_pct={1: 2.0, 2: 2.5, 3: 2.0},
        max_positions={1: 10, 2: 8, 3: 12},
        stop_loss_pct={1: -8.0, 2: -8.0, 3: -8.0},  # VERY TIGHT
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # C. Moderate - balanced approach, 10% stops
    strategies.append(Strategy(
        name="C_Moderate",
        description="Balanced positions (3%), 10% stops",
        position_pct={1: 3.0, 2: 3.5, 3: 3.0},
        max_positions={1: 8, 2: 6, 3: 8},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        bucket_weights={1: 0.35, 2: 0.35, 3: 0.30},
    ))

    # D. Earnings Beat Only - 10% stops
    strategies.append(Strategy(
        name="D_EarningsBeat",
        description="Earnings beat filter, 10% stop",
        position_pct={1: 3.5, 2: 4.0, 3: 3.0},
        max_positions={1: 8, 2: 6, 3: 10},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_earnings_beat=True,
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # E. Strong Sectors Only - 10% stops
    strategies.append(Strategy(
        name="E_StrongSectors",
        description="Strong sectors only, 10% stop",
        position_pct={1: 3.5, 2: 4.0, 3: 3.0},
        max_positions={1: 8, 2: 6, 3: 10},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_strong_sector=True,
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # F. Multi-Signal (2+ signals agree) - 10% stops
    strategies.append(Strategy(
        name="F_MultiSignal",
        description="2+ signals agree, 10% stop",
        position_pct={1: 3.5, 2: 4.0, 3: 3.0},
        max_positions={1: 8, 2: 6, 3: 10},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_multi_signal=2,
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # G. Combined Best (earnings beat + exclude avoid sectors) - 10% stops
    strategies.append(Strategy(
        name="G_CombinedBest",
        description="Earnings beat + exclude avoid, 10% stop",
        position_pct={1: 4.0, 2: 4.5, 3: 3.5},
        max_positions={1: 6, 2: 5, 3: 8},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_earnings_beat=True,
        exclude_avoid_sector=True,
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # H. Regime Adaptive - 10% stops
    strategies.append(Strategy(
        name="H_RegimeAdaptive",
        description="Regime-adaptive allocation, 10% stop",
        position_pct={1: 3.0, 2: 4.5, 3: 2.5},
        max_positions={1: 8, 2: 8, 3: 10},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        allocation_mode='regime_adaptive',
        bucket_weights={1: 0.40, 2: 0.30, 3: 0.30},
    ))

    # I. Big Beat Only (highest conviction) - 10% stops
    strategies.append(Strategy(
        name="I_BigBeatOnly",
        description="Big earnings beat (>10%), 10% stop",
        position_pct={1: 4.5, 2: 5.0, 3: 4.0},
        max_positions={1: 5, 2: 4, 3: 6},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_big_beat=True,
        bucket_weights={1: 0.35, 2: 0.35, 3: 0.30},
    ))

    # J. Maximum Selectivity - 10% stops
    strategies.append(Strategy(
        name="J_MaxSelectivity",
        description="All filters + 10% stop",
        position_pct={1: 5.0, 2: 5.5, 3: 5.0},
        max_positions={1: 4, 2: 3, 3: 4},
        stop_loss_pct={1: -10.0, 2: -10.0, 3: -10.0},
        require_earnings_beat=True,
        require_strong_sector=True,
        require_multi_signal=2,
        bucket_weights={1: 0.35, 2: 0.35, 3: 0.30},
    ))

    return strategies


# ══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE (V2 - Active Rotation)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """Record of a single trade."""
    symbol: str
    bucket_id: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    position_size: float
    return_pct: float
    exit_reason: str  # 'stop', 'target', 'hold_expired', 'rotation'
    pnl: float


@dataclass
class SimulationResultV2:
    """Enhanced results from strategy simulation with trade tracking."""
    strategy_name: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    cagr_pct: float
    total_trades: int
    buys: int
    sells: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    stopped_out: int
    hit_target: int
    hold_expired: int
    rotated: int
    trades_per_month: float
    best_trade_pct: float
    worst_trade_pct: float
    trades_per_bucket: Dict[int, int] = field(default_factory=dict)
    win_rate_per_bucket: Dict[int, float] = field(default_factory=dict)
    premium_deployed: float = 0  # Total capital deployed over time


def simulate_strategy_v2(df: pd.DataFrame, strategy: Strategy,
                         starting_capital: float = 100_000,
                         slippage: float = 0.001,
                         hold_days: int = 90,
                         profit_target_pct: float = 50.0) -> SimulationResultV2:
    """
    Run simulation with active rotation (V2).

    Key differences from V1:
    - Tracks individual trades (buys/sells)
    - Active rotation: closes positions when hold period expires or stop/target hit
    - Reinvests proceeds into new opportunities
    - Tracks exit reasons (stop, target, expired, rotation)
    """
    from bucket_config import BUCKETS

    capital = starting_capital
    equity_curve = []
    max_equity = capital
    max_drawdown = 0

    trades: List[TradeRecord] = []
    active_positions: Dict[str, dict] = {}  # symbol -> position info

    # Get all bucket signals with filters applied
    bucket_signals = {}
    for bucket in BUCKETS:
        sub = bucket.filter_fn(df.copy())
        sub = sub[sub['return_3m'].notna()].copy()

        # Apply strategy filters
        if strategy.require_earnings_beat:
            sub = sub[sub['earnings_surprise_pct'] > 0]
        if strategy.require_big_beat:
            sub = sub[sub['earnings_surprise_pct'] > 10]
        if strategy.require_strong_sector:
            sub = sub[sub['sector'].isin(bucket.strong_sectors)]
        if strategy.exclude_avoid_sector:
            sub = sub[~sub['sector'].isin(bucket.avoid_sectors)]
        if strategy.require_multi_signal > 0:
            sub = sub[sub['concurrent_signals'] >= strategy.require_multi_signal]

        sub['bucket_id'] = bucket.id
        bucket_signals[bucket.id] = sub

    # Combine and sort by date
    all_signals = pd.concat(bucket_signals.values()).sort_values('date_dt')
    if len(all_signals) == 0:
        return SimulationResultV2(
            strategy_name=strategy.name, starting_capital=starting_capital,
            ending_capital=starting_capital, total_return_pct=0, cagr_pct=0,
            total_trades=0, buys=0, sells=0, win_rate=0, avg_win_pct=0,
            avg_loss_pct=0, profit_factor=0, max_drawdown_pct=0, sharpe_ratio=0,
            calmar_ratio=0, stopped_out=0, hit_target=0, hold_expired=0, rotated=0,
            trades_per_month=0, best_trade_pct=0, worst_trade_pct=0,
        )

    total_deployed = 0

    # Process signals chronologically
    for _, row in all_signals.iterrows():
        bucket_id = row['bucket_id']
        bucket = BUCKETS[bucket_id - 1]
        sig_date = row['date_dt']
        symbol = row['symbol']

        # Get strategy params for this bucket
        pos_pct = strategy.position_pct.get(bucket_id, bucket.position_pct) / 100
        max_pos = strategy.max_positions.get(bucket_id, bucket.max_positions)
        stop_pct = strategy.stop_loss_pct.get(bucket_id, bucket.stop_loss_pct)

        # First, check and close any expired positions
        to_close = []
        for pos_sym, pos in active_positions.items():
            days_held = (sig_date - pos['entry_date']).days
            if days_held >= hold_days:
                to_close.append((pos_sym, 'hold_expired'))

        for pos_sym, reason in to_close:
            pos = active_positions[pos_sym]
            # Use actual return from data
            raw_ret = pos['expected_return']
            net_ret = raw_ret - 2 * slippage * 100  # Apply slippage

            # Check stop loss
            if net_ret <= stop_pct:
                net_ret = stop_pct - slippage * 100
                reason = 'stop'
            elif net_ret >= profit_target_pct:
                net_ret = profit_target_pct - slippage * 100
                reason = 'target'

            pnl = pos['position_size'] * (net_ret / 100)
            capital += pnl

            trades.append(TradeRecord(
                symbol=pos_sym, bucket_id=pos['bucket_id'],
                entry_date=pos['entry_date'], exit_date=sig_date,
                entry_price=pos['entry_price'], position_size=pos['position_size'],
                return_pct=net_ret, exit_reason=reason, pnl=pnl
            ))
            del active_positions[pos_sym]

        # Count current positions per bucket
        bucket_position_counts = defaultdict(int)
        for pos in active_positions.values():
            bucket_position_counts[pos['bucket_id']] += 1

        # Check if we can open a new position
        if bucket_position_counts[bucket_id] >= max_pos:
            continue

        # Check if already holding this symbol
        if symbol in active_positions:
            continue

        # Determine allocation
        if strategy.allocation_mode == 'regime_adaptive':
            is_bear = row.get('market_bearish', 0) == 1
            if is_bear and bucket_id == 2:
                alloc_mult = 1.5
            elif is_bear and bucket_id == 1:
                alloc_mult = 1.2
            elif not is_bear and bucket_id == 2:
                alloc_mult = 0.5
            else:
                alloc_mult = 1.0
        else:
            alloc_mult = 1.0

        # Calculate position size based on current capital
        bucket_weight = strategy.bucket_weights.get(bucket_id, 0.33)
        pos_size = capital * bucket_weight * pos_pct * alloc_mult

        # Don't over-allocate
        total_invested = sum(p['position_size'] for p in active_positions.values())
        if total_invested + pos_size > capital * 0.95:  # Keep 5% cash buffer
            continue

        # Open position
        active_positions[symbol] = {
            'symbol': symbol,
            'bucket_id': bucket_id,
            'entry_date': sig_date,
            'entry_price': row['close_price'] if pd.notna(row['close_price']) else row['close'],
            'position_size': pos_size,
            'expected_return': row['return_3m'],  # Store for exit calculation
            'stop_pct': stop_pct,
        }
        total_deployed += pos_size

        # Update equity tracking
        equity_curve.append((sig_date, capital))
        if capital > max_equity:
            max_equity = capital
        dd = (capital - max_equity) / max_equity * 100
        if dd < max_drawdown:
            max_drawdown = dd

    # Close any remaining positions at end
    final_date = all_signals['date_dt'].max()
    for pos_sym, pos in list(active_positions.items()):
        raw_ret = pos['expected_return']
        net_ret = raw_ret - 2 * slippage * 100

        if net_ret <= pos['stop_pct']:
            net_ret = pos['stop_pct'] - slippage * 100
            reason = 'stop'
        elif net_ret >= profit_target_pct:
            net_ret = profit_target_pct - slippage * 100
            reason = 'target'
        else:
            reason = 'hold_expired'

        pnl = pos['position_size'] * (net_ret / 100)
        capital += pnl

        trades.append(TradeRecord(
            symbol=pos_sym, bucket_id=pos['bucket_id'],
            entry_date=pos['entry_date'], exit_date=final_date,
            entry_price=pos['entry_price'], position_size=pos['position_size'],
            return_pct=net_ret, exit_reason=reason, pnl=pnl
        ))

    # Compute metrics
    total_trades = len(trades)
    if total_trades == 0:
        return SimulationResultV2(
            strategy_name=strategy.name, starting_capital=starting_capital,
            ending_capital=capital, total_return_pct=0, cagr_pct=0,
            total_trades=0, buys=0, sells=0, win_rate=0, avg_win_pct=0,
            avg_loss_pct=0, profit_factor=0, max_drawdown_pct=max_drawdown,
            sharpe_ratio=0, calmar_ratio=0, stopped_out=0, hit_target=0,
            hold_expired=0, rotated=0, trades_per_month=0,
            best_trade_pct=0, worst_trade_pct=0,
        )

    wins = [t.return_pct for t in trades if t.return_pct > 0]
    losses = [t.return_pct for t in trades if t.return_pct <= 0]
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = (avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf')

    stopped = sum(1 for t in trades if t.exit_reason == 'stop')
    targets = sum(1 for t in trades if t.exit_reason == 'target')
    expired = sum(1 for t in trades if t.exit_reason == 'hold_expired')
    rotated = sum(1 for t in trades if t.exit_reason == 'rotation')

    # Time-based metrics
    if equity_curve and len(equity_curve) > 1:
        years = max(0.1, (equity_curve[-1][0] - equity_curve[0][0]).days / 365.25)
        months = years * 12
    else:
        years = 2  # Default to 2 years for CAGR calc
        months = 24

    total_return = (capital / starting_capital - 1) * 100
    cagr = ((capital / starting_capital) ** (1 / years) - 1) * 100 if capital > 0 else -100

    # Sharpe from equity curve
    if len(equity_curve) > 2:
        eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        eq_df = eq_df.set_index('date').resample('M').last().dropna()
        if len(eq_df) > 2:
            monthly_rets = eq_df['equity'].pct_change().dropna()
            sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')

    # Per-bucket breakdown
    trades_per_bucket = {}
    win_rate_per_bucket = {}
    for bid in [1, 2, 3]:
        bucket_trades = [t for t in trades if t.bucket_id == bid]
        trades_per_bucket[bid] = len(bucket_trades)
        if bucket_trades:
            win_rate_per_bucket[bid] = sum(1 for t in bucket_trades if t.return_pct > 0) / len(bucket_trades) * 100
        else:
            win_rate_per_bucket[bid] = 0

    all_returns = [t.return_pct for t in trades]

    return SimulationResultV2(
        strategy_name=strategy.name,
        starting_capital=starting_capital,
        ending_capital=capital,
        total_return_pct=total_return,
        cagr_pct=cagr,
        total_trades=total_trades,
        buys=total_trades,  # Each trade is a buy
        sells=total_trades,  # Each trade is a sell
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        stopped_out=stopped,
        hit_target=targets,
        hold_expired=expired,
        rotated=rotated,
        trades_per_month=total_trades / max(1, months),
        best_trade_pct=max(all_returns) if all_returns else 0,
        worst_trade_pct=min(all_returns) if all_returns else 0,
        trades_per_bucket=trades_per_bucket,
        win_rate_per_bucket=win_rate_per_bucket,
        premium_deployed=total_deployed,
    )


@dataclass
class SimulationResult:
    """Results from a strategy simulation."""
    strategy_name: str
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    cagr_pct: float
    total_trades: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    calmar_ratio: float
    trades_per_bucket: Dict[int, int] = field(default_factory=dict)
    win_rate_per_bucket: Dict[int, float] = field(default_factory=dict)


def simulate_strategy(df: pd.DataFrame, strategy: Strategy,
                       starting_capital: float = 100_000,
                       slippage: float = 0.001) -> SimulationResult:
    """
    Run a full simulation of a strategy.

    Args:
        df: Full signals dataframe
        strategy: Strategy configuration
        starting_capital: Initial capital
        slippage: Per-leg slippage (0.001 = 0.1%)

    Returns:
        SimulationResult with all metrics
    """
    from bucket_config import BUCKET_1, BUCKET_2, BUCKET_3, BUCKETS

    capital = starting_capital
    equity_curve = []
    max_equity = capital
    max_drawdown = 0

    trades_by_bucket = defaultdict(list)
    active_positions = defaultdict(list)  # bucket_id -> [(entry_date, symbol, alloc)]

    # Get all bucket signals
    bucket_signals = {}
    for bucket in BUCKETS:
        sub = bucket.filter_fn(df)
        sub = sub[sub['return_3m'].notna()].copy()

        # Apply strategy filters
        if strategy.require_earnings_beat:
            sub = sub[sub['earnings_surprise_pct'] > 0]
        if strategy.require_big_beat:
            sub = sub[sub['earnings_surprise_pct'] > 10]
        if strategy.require_strong_sector:
            sub = sub[sub['sector'].isin(bucket.strong_sectors)]
        if strategy.exclude_avoid_sector:
            sub = sub[~sub['sector'].isin(bucket.avoid_sectors)]
        if strategy.require_multi_signal > 0:
            sub = sub[sub['concurrent_signals'] >= strategy.require_multi_signal]

        sub['bucket_id'] = bucket.id
        bucket_signals[bucket.id] = sub

    # Combine and sort by date
    all_signals = pd.concat(bucket_signals.values()).sort_values('date_dt')
    if len(all_signals) == 0:
        return SimulationResult(
            strategy_name=strategy.name,
            starting_capital=starting_capital,
            ending_capital=starting_capital,
            total_return_pct=0,
            cagr_pct=0,
            total_trades=0,
            win_rate=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            max_drawdown_pct=0,
            sharpe_ratio=0,
            calmar_ratio=0,
        )

    # Track held symbols across all buckets
    for _, row in all_signals.iterrows():
        bucket_id = row['bucket_id']
        bucket = BUCKETS[bucket_id - 1]
        sig_date = row['date_dt']

        # Get strategy params for this bucket
        pos_pct = strategy.position_pct.get(bucket_id, bucket.position_pct) / 100
        max_pos = strategy.max_positions.get(bucket_id, bucket.max_positions)
        stop_pct = strategy.stop_loss_pct.get(bucket_id, bucket.stop_loss_pct) / 100

        # Clean expired positions
        hold_days = bucket.hold_months * 30
        active_positions[bucket_id] = [
            p for p in active_positions[bucket_id]
            if (sig_date - p[0]).days < hold_days
        ]

        # Check position limits
        if len(active_positions[bucket_id]) >= max_pos:
            continue

        # Check if already holding this symbol
        all_held = set()
        for bid in active_positions:
            for p in active_positions[bid]:
                all_held.add(p[1])
        if row['symbol'] in all_held:
            continue

        # Determine allocation based on mode
        if strategy.allocation_mode == 'regime_adaptive':
            is_bear = row['market_bearish'] == 1
            if is_bear and bucket_id == 2:
                alloc_mult = 1.5  # boost bear dip during bears
            elif is_bear and bucket_id == 1:
                alloc_mult = 1.2  # slight boost to quality
            elif not is_bear and bucket_id == 2:
                alloc_mult = 0.5  # reduce bear dip in bull (won't fire anyway)
            else:
                alloc_mult = 1.0
        else:
            alloc_mult = 1.0

        # Calculate position size
        bucket_weight = strategy.bucket_weights.get(bucket_id, 0.33)
        pos_size = capital * bucket_weight * pos_pct * alloc_mult

        # Apply return with slippage and stop
        raw_ret = row['return_3m'] / 100
        net_ret = raw_ret - 2 * slippage  # round-trip slippage

        if net_ret < stop_pct:
            net_ret = stop_pct - slippage  # stopped out

        # Calculate P&L
        pnl = pos_size * net_ret
        capital += pnl

        trades_by_bucket[bucket_id].append(net_ret)
        active_positions[bucket_id].append((sig_date, row['symbol'], pos_size))
        equity_curve.append((sig_date, capital))

        # Track drawdown
        if capital > max_equity:
            max_equity = capital
        dd = (capital - max_equity) / max_equity * 100
        if dd < max_drawdown:
            max_drawdown = dd

    # Compute final metrics
    all_trades = []
    trades_per_bucket = {}
    win_rate_per_bucket = {}
    for bid, rets in trades_by_bucket.items():
        all_trades.extend(rets)
        trades_per_bucket[bid] = len(rets)
        win_rate_per_bucket[bid] = (sum(1 for r in rets if r > 0) / len(rets) * 100) if rets else 0

    total_trades = len(all_trades)
    if total_trades == 0:
        return SimulationResult(
            strategy_name=strategy.name,
            starting_capital=starting_capital,
            ending_capital=capital,
            total_return_pct=0,
            cagr_pct=0,
            total_trades=0,
            win_rate=0,
            avg_win_pct=0,
            avg_loss_pct=0,
            profit_factor=0,
            max_drawdown_pct=max_drawdown,
            sharpe_ratio=0,
            calmar_ratio=0,
        )

    wins = [r * 100 for r in all_trades if r > 0]
    losses = [r * 100 for r in all_trades if r <= 0]
    win_rate = len(wins) / total_trades * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    pf = (avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf

    # Time-based metrics
    if equity_curve:
        years = max(0.1, (equity_curve[-1][0] - equity_curve[0][0]).days / 365.25)
    else:
        years = 1
    total_return = (capital / starting_capital - 1) * 100
    cagr = ((capital / starting_capital) ** (1 / years) - 1) * 100 if capital > 0 else -100

    # Sharpe (approximate from equity curve)
    if len(equity_curve) > 2:
        eq_df = pd.DataFrame(equity_curve, columns=['date', 'equity'])
        eq_df = eq_df.set_index('date').resample('M').last().dropna()
        if len(eq_df) > 2:
            monthly_rets = eq_df['equity'].pct_change().dropna()
            sharpe = (monthly_rets.mean() / monthly_rets.std()) * np.sqrt(12) if monthly_rets.std() > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else np.inf

    return SimulationResult(
        strategy_name=strategy.name,
        starting_capital=starting_capital,
        ending_capital=capital,
        total_return_pct=total_return,
        cagr_pct=cagr,
        total_trades=total_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win,
        avg_loss_pct=avg_loss,
        profit_factor=pf,
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=sharpe,
        calmar_ratio=calmar,
        trades_per_bucket=trades_per_bucket,
        win_rate_per_bucket=win_rate_per_bucket,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main_v2():
    """Run 2-year backtest with active rotation (comparable to options)."""
    start = datetime.now()
    print("=" * 130)
    print("  STOCK STRATEGY BACKTEST — ACTIVE ROTATION")
    print(f"  Time Period: 2024-2025 (2 years)")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 130)

    print("\n  Loading 2-year data (2024-2025)...")
    df = load_data(start_year=2024, end_year=2025)
    print(f"  Loaded {len(df):,} signals")
    print(f"  Date range: {df['date_dt'].min().strftime('%Y-%m-%d')} to {df['date_dt'].max().strftime('%Y-%m-%d')}")

    strategies = get_default_strategies()
    results = []

    print(f"\n  Testing {len(strategies)} strategies...")
    print("  " + "─" * 128)
    for strat in strategies:
        result = simulate_strategy_v2(df, strat, hold_days=90, profit_target_pct=50.0)
        results.append(result)
        print(f"  {strat.name:<20s}: ${result.ending_capital:>10,.0f} ({result.total_return_pct:>+7.1f}%) | "
              f"{result.total_trades:>4} trades | {result.win_rate:>5.1f}% win | DD: {result.max_drawdown_pct:>+5.1f}%")

    # Sort by total return
    results.sort(key=lambda x: x.total_return_pct, reverse=True)

    # Main comparison table
    print("\n\n" + "=" * 130)
    print("  STOCK STRATEGY COMPARISON (sorted by return)")
    print("=" * 130)
    print(f"\n  {'Strategy':<20s} {'End Cap':>12s} {'Return':>8s} {'CAGR':>7s} {'Trades':>7s} "
          f"{'Buys':>6s} {'Sells':>6s} {'Win%':>6s} {'PF':>5s} {'MaxDD':>7s} {'Sharpe':>7s} {'AvgRet':>7s}")
    print(f"  {'─'*125}")

    for r in results:
        avg_ret = (r.avg_win_pct * (r.win_rate/100) + r.avg_loss_pct * (1 - r.win_rate/100))
        print(f"  {r.strategy_name:<20s} ${r.ending_capital:>10,.0f} {r.total_return_pct:>+7.1f}% "
              f"{r.cagr_pct:>+6.1f}% {r.total_trades:>6} {r.buys:>5} {r.sells:>5} {r.win_rate:>5.1f}% "
              f"{min(r.profit_factor, 9.99):>4.2f} {r.max_drawdown_pct:>+6.1f}% {r.sharpe_ratio:>6.2f} "
              f"{avg_ret:>+6.1f}%")

    # Trading Activity Summary
    print("\n\n" + "=" * 130)
    print("  TRADING ACTIVITY SUMMARY")
    print("=" * 130)
    print(f"\n  {'Strategy':<20s} {'Total':>7s} {'Buys':>6s} {'Sells':>6s} {'Per Mo':>7s} "
          f"{'Stopped':>8s} {'Target':>8s} {'Expire':>8s} {'Best':>8s} {'Worst':>8s}")
    print(f"  {'─'*95}")

    for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
        print(f"  {r.strategy_name:<20s} {r.total_trades:>6} {r.buys:>5} {r.sells:>5} "
              f"{r.trades_per_month:>6.1f} {r.stopped_out:>7} {r.hit_target:>7} {r.hold_expired:>7} "
              f"{r.best_trade_pct:>+7.0f}% {r.worst_trade_pct:>+7.0f}%")

    # Win/Loss breakdown
    print("\n\n" + "=" * 130)
    print("  WIN/LOSS BREAKDOWN")
    print("=" * 130)
    print(f"\n  {'Strategy':<20s} {'Wins':>6s} {'Losses':>7s} {'Win%':>6s} {'Avg Win':>9s} "
          f"{'Avg Loss':>9s} {'PF':>5s} {'Capital$':>12s}")
    print(f"  {'─'*90}")

    for r in sorted(results, key=lambda x: x.total_return_pct, reverse=True):
        wins = int(r.total_trades * r.win_rate / 100)
        losses = r.total_trades - wins
        print(f"  {r.strategy_name:<20s} {wins:>5} {losses:>6} {r.win_rate:>5.1f}% "
              f"{r.avg_win_pct:>+8.1f}% {r.avg_loss_pct:>+8.1f}% {min(r.profit_factor, 9.99):>4.2f} "
              f"${r.premium_deployed:>10,.0f}")

    # Top 3 detailed view
    print("\n\n" + "=" * 130)
    print("  TOP 3 STRATEGIES — DETAILED VIEW")
    print("=" * 130)

    for i, r in enumerate(results[:3]):
        strat = next(s for s in strategies if s.name == r.strategy_name)
        print(f"\n  #{i+1} {r.strategy_name}")
        print(f"      Description: {strat.description}")
        print(f"      Position: {list(strat.position_pct.values())}% | Stop: {list(strat.stop_loss_pct.values())}%")
        print(f"      Filters: beat={strat.require_earnings_beat}, big_beat={strat.require_big_beat}, "
              f"strong_sector={strat.require_strong_sector}, multi_sig={strat.require_multi_signal}")
        print(f"      Results: $100,000 → ${r.ending_capital:,.0f} ({r.total_return_pct:+.1f}%)")
        print(f"      Activity: {r.buys} buys, {r.sells} sells, {r.trades_per_month:.1f}/month")
        print(f"      Wins: {int(r.total_trades * r.win_rate / 100)} | "
              f"Losses: {r.total_trades - int(r.total_trades * r.win_rate / 100)} | "
              f"Stopped: {r.stopped_out} | Targets: {r.hit_target}")
        print(f"      Bucket breakdown:")
        for bid in [1, 2, 3]:
            trades = r.trades_per_bucket.get(bid, 0)
            wr = r.win_rate_per_bucket.get(bid, 0)
            bucket_name = ['Quality Growth', 'Bear Dip', 'High-Growth'][bid-1]
            print(f"        Bucket {bid} ({bucket_name}): {trades:>4} trades, {wr:>5.1f}% win")

    # SPY Benchmark Comparison
    print("\n\n" + "=" * 130)
    print("  S&P 500 (SPY) BENCHMARK COMPARISON")
    print("=" * 130)

    # SPY actual returns (approximate)
    # 2024: ~+23%, 2025 YTD (through Nov): ~+25%
    # Total 2-year: ~+54%
    spy_2024 = 23.0
    spy_2025_ytd = 25.0
    spy_2yr_total = ((1 + spy_2024/100) * (1 + spy_2025_ytd/100) - 1) * 100
    spy_cagr = ((1 + spy_2yr_total/100) ** 0.5 - 1) * 100

    print(f"\n  SPY Benchmark (2024-2025):")
    print(f"    2024: +{spy_2024:.1f}%")
    print(f"    2025 YTD: +{spy_2025_ytd:.1f}%")
    print(f"    2-Year Total: +{spy_2yr_total:.1f}%")
    print(f"    CAGR: +{spy_cagr:.1f}%")

    print(f"\n  Strategy vs SPY:")
    print(f"  {'Strategy':<20s} {'Return':>8s} {'vs SPY':>10s} {'Alpha':>8s}")
    print(f"  {'-'*50}")

    for r in results[:5]:
        vs_spy = r.total_return_pct - spy_2yr_total
        alpha = r.cagr_pct - spy_cagr
        beat = "✓ BEAT" if r.total_return_pct > spy_2yr_total else "✗ UNDER"
        print(f"  {r.strategy_name:<20s} {r.total_return_pct:>+7.1f}% {vs_spy:>+9.1f}% {beat:>8s}")

    # Recommendation
    print("\n\n" + "=" * 130)
    print("  RECOMMENDATION")
    print("=" * 130)

    best = results[0]
    print(f"\n  Best Strategy: {best.strategy_name}")
    print(f"    Return: {best.total_return_pct:+.1f}% ({best.cagr_pct:+.1f}% CAGR)")
    print(f"    Win Rate: {best.win_rate:.1f}%")
    print(f"    vs SPY: {best.total_return_pct - spy_2yr_total:+.1f}%")

    if best.total_return_pct > spy_2yr_total:
        print(f"\n  ✓ TOP STRATEGY BEATS S&P 500")
    else:
        print(f"\n  ⚠️  Note: Top strategy underperforms S&P 500")
        print("      Consider increasing position sizes or using options for alpha")

    print("\n  STOCKS vs OPTIONS:")
    print("      - Stocks with 10% stops: Better downside protection")
    print("      - Options (OTM5_EarningsBeat): +242% vs stocks +{:.0f}%".format(best.total_return_pct))
    print("      - Recommendation: 70% stocks (core) + 30% options (alpha)")

    elapsed = datetime.now() - start
    print(f"\n\n{'='*130}")
    print(f"  Stock backtest complete! Runtime: {elapsed}")
    print(f"{'='*130}")

    return results


def main():
    """Legacy main function - runs original simulation."""
    start = datetime.now()
    print("=" * 120)
    print("  STRATEGY BACKTEST COMPARISON (LEGACY)")
    print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)

    print("\n  Loading data...")
    df = load_data()
    print(f"  Loaded {len(df):,} signals")

    strategies = get_default_strategies()
    results = []

    print(f"\n  Testing {len(strategies)} strategies...")
    for strat in strategies:
        result = simulate_strategy(df, strat)
        results.append(result)
        print(f"    {strat.name}: ${result.ending_capital:,.0f} "
              f"({result.total_return_pct:+.1f}%) | {result.total_trades:,} trades | "
              f"{result.win_rate:.1f}% win")

    # Sort by total return
    results.sort(key=lambda x: x.total_return_pct, reverse=True)

    # Print summary table
    print("\n" + "=" * 120)
    print("  STRATEGY COMPARISON (sorted by total return)")
    print("=" * 120)
    print(f"\n  {'Strategy':<20s} {'End Capital':>12s} {'Return':>8s} {'CAGR':>7s} {'Trades':>7s} "
          f"{'Win%':>6s} {'PF':>5s} {'MaxDD':>7s} {'Sharpe':>7s} {'Calmar':>7s}")
    print(f"  {'-'*110}")

    for r in results:
        print(f"  {r.strategy_name:<20s} ${r.ending_capital:>10,.0f} {r.total_return_pct:>+7.1f}% "
              f"{r.cagr_pct:>+6.1f}% {r.total_trades:>6,} {r.win_rate:>5.1f}% {r.profit_factor:>4.2f} "
              f"{r.max_drawdown_pct:>+6.1f}% {r.sharpe_ratio:>6.2f} {r.calmar_ratio:>6.2f}")

    elapsed = datetime.now() - start
    print(f"\n\n{'='*120}")
    print(f"  Backtest complete! Runtime: {elapsed}")
    print(f"{'='*120}")

    return results


if __name__ == "__main__":
    # Run the V2 backtest with active rotation
    results = main_v2()

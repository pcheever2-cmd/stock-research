#!/usr/bin/env python3
"""
Dashboard Data Provider (v2)
=============================
Provides data for dashboard integration including both stock and options strategies.
Can be imported by Flask, Streamlit, or any dashboard framework.

Example usage:
    from dashboard_data import DashboardData
    data = DashboardData()

    # Get portfolio status
    portfolio = data.get_portfolio_summary()

    # Get current signals
    signals = data.get_current_signals()

    # Get equity curve
    curve = data.get_equity_curve()

    # Get strategy comparison (stocks vs options)
    strategies = data.get_strategy_comparison()
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
MAIN_DB = str(PROJECT_ROOT / 'nasdaq_stocks.db')
PORTFOLIO_DB = str(PROJECT_ROOT / 'mock_portfolio.db')

from bucket_config import BUCKETS, get_bucket_summary


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioSummary:
    """Portfolio summary for dashboard display."""
    cash: float
    positions_value: float
    total_equity: float
    starting_capital: float
    total_return_pct: float
    unrealized_pnl: float
    open_positions: int
    win_rate: float
    total_trades: int
    last_updated: str


@dataclass
class Position:
    """A portfolio position."""
    symbol: str
    bucket_id: int
    bucket_name: str
    entry_date: str
    entry_price: float
    shares: float
    current_price: float
    current_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_loss_price: float
    target_exit_date: str
    days_held: int
    signal_type: str
    sector: str
    earnings_beat: bool
    at_risk: bool  # price near stop


@dataclass
class Signal:
    """A trading signal."""
    symbol: str
    date: str
    bucket_id: int
    bucket_name: str
    signal_type: str
    value_score_v2: float
    fundamentals_score: float
    lt_score: float
    rsi: float
    ev_ebitda: float
    rev_growth: float
    sector: str
    earnings_beat: bool
    price: float
    status: str  # 'actionable', 'held', 'max_positions'


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

class DashboardData:
    """Main class for dashboard data access."""

    def __init__(self):
        self.starting_capital = 100_000

    def _get_portfolio_conn(self):
        return sqlite3.connect(PORTFOLIO_DB)

    def _get_backtest_conn(self):
        return sqlite3.connect(BACKTEST_DB)

    # ═══════════════════════════════════════════════════════════════════════════
    # PORTFOLIO DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary."""
        conn = self._get_portfolio_conn()

        # Get cash
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM portfolio_config WHERE key = 'cash'")
        row = cursor.fetchone()
        cash = float(row[0]) if row else self.starting_capital

        # Get open positions
        positions = pd.read_sql_query(
            "SELECT * FROM positions WHERE status = 'open'", conn
        )
        positions_value = positions['current_value'].sum() if len(positions) > 0 else 0
        unrealized_pnl = positions['unrealized_pnl'].sum() if len(positions) > 0 else 0

        # Get trade history stats
        trades = pd.read_sql_query("SELECT * FROM trade_history", conn)
        total_trades = len(trades)
        win_rate = (trades['realized_pnl'] > 0).mean() * 100 if total_trades > 0 else 0

        conn.close()

        total_equity = cash + positions_value

        return asdict(PortfolioSummary(
            cash=cash,
            positions_value=positions_value,
            total_equity=total_equity,
            starting_capital=self.starting_capital,
            total_return_pct=(total_equity / self.starting_capital - 1) * 100,
            unrealized_pnl=unrealized_pnl,
            open_positions=len(positions),
            win_rate=win_rate,
            total_trades=total_trades,
            last_updated=datetime.now().isoformat(),
        ))

    def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        conn = self._get_portfolio_conn()
        df = pd.read_sql_query(
            "SELECT * FROM positions WHERE status = 'open' ORDER BY entry_date",
            conn
        )
        conn.close()

        positions = []
        for _, row in df.iterrows():
            bucket = BUCKETS[row['bucket_id'] - 1]
            entry_date = datetime.fromisoformat(row['entry_date'])
            days_held = (datetime.now().date() - entry_date.date()).days

            # Check if at risk (within 5% of stop)
            at_risk = False
            if row['current_price'] and row['stop_loss_price']:
                risk_threshold = row['stop_loss_price'] * 1.05
                at_risk = row['current_price'] <= risk_threshold

            positions.append(asdict(Position(
                symbol=row['symbol'],
                bucket_id=row['bucket_id'],
                bucket_name=bucket.name,
                entry_date=row['entry_date'],
                entry_price=row['entry_price'],
                shares=row['shares'],
                current_price=row['current_price'] or row['entry_price'],
                current_value=row['current_value'] or row['position_value'],
                unrealized_pnl=row['unrealized_pnl'] or 0,
                unrealized_pnl_pct=row['unrealized_pnl_pct'] or 0,
                stop_loss_price=row['stop_loss_price'],
                target_exit_date=row['target_exit_date'],
                days_held=days_held,
                signal_type=row['signal_type'] or '',
                sector=row['sector'] or '',
                earnings_beat=bool(row['earnings_beat']),
                at_risk=at_risk,
            )))

        return positions

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trade history."""
        conn = self._get_portfolio_conn()
        df = pd.read_sql_query(f"""
            SELECT * FROM trade_history ORDER BY exit_date DESC LIMIT {limit}
        """, conn)
        conn.close()

        return df.to_dict('records')

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """Get equity curve data."""
        conn = self._get_portfolio_conn()
        df = pd.read_sql_query(
            "SELECT date, total_equity FROM daily_snapshots ORDER BY date",
            conn
        )
        conn.close()

        return df.to_dict('records')

    def get_position_summary_by_bucket(self) -> List[Dict[str, Any]]:
        """Get position summary grouped by bucket."""
        positions = self.get_open_positions()
        summary = []

        for bucket in BUCKETS:
            bucket_positions = [p for p in positions if p['bucket_id'] == bucket.id]
            count = len(bucket_positions)
            max_pos = 8 if bucket.id == 1 else (6 if bucket.id == 2 else 10)
            unrealized = sum(p['unrealized_pnl'] for p in bucket_positions)
            value = sum(p['current_value'] for p in bucket_positions)

            summary.append({
                'bucket_id': bucket.id,
                'bucket_name': bucket.name,
                'short_name': bucket.short_name,
                'position_count': count,
                'max_positions': max_pos,
                'utilization_pct': (count / max_pos) * 100,
                'total_value': value,
                'unrealized_pnl': unrealized,
            })

        return summary

    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def get_current_signals(self, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get current qualifying signals."""
        conn = self._get_backtest_conn()
        cutoff = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

        df = pd.read_sql_query(f"""
            SELECT
                s.symbol, s.date, s.signal_type,
                s.lt_score, s.value_score_v2, s.close_price,
                d.rsi, d.ev_ebitda, d.rev_growth, d.fundamentals_score, d.market_risk_score
            FROM backtest_signals s
            LEFT JOIN backtest_daily_scores d ON s.symbol = d.symbol AND s.date = d.date
            WHERE s.date >= '{cutoff}'
            ORDER BY s.date DESC, s.value_score_v2 DESC
        """, conn)

        # Get supplementary
        mcaps = pd.read_sql_query("""
            SELECT symbol, market_cap FROM historical_key_metrics
            WHERE market_cap IS NOT NULL
            GROUP BY symbol HAVING date = MAX(date)
        """, conn).set_index('symbol')['market_cap']

        acov = pd.read_sql_query("""
            SELECT symbol, MAX(num_analysts_eps) as analyst_count
            FROM analyst_estimates_snapshot GROUP BY symbol
        """, conn).set_index('symbol')['analyst_count']

        conn.close()

        conn2 = sqlite3.connect(MAIN_DB)
        sectors = pd.read_sql_query(
            "SELECT symbol, sector FROM stock_consensus WHERE sector IS NOT NULL",
            conn2
        ).set_index('symbol')
        conn2.close()

        df['market_cap'] = df['symbol'].map(mcaps)
        df['analyst_count'] = df['symbol'].map(acov)
        df['sector'] = df['symbol'].map(sectors['sector']) if 'sector' in sectors.columns else None
        df['market_bearish'] = (df['market_risk_score'] == 0).astype(int)
        df['ev_ebitda_clean'] = df['ev_ebitda'].clip(-50, 200)

        # Get held symbols
        held_symbols = set(p['symbol'] for p in self.get_open_positions())

        # Classify signals by bucket
        signals = []
        for _, row in df.iterrows():
            bucket_id = self._classify_signal_bucket(row)
            if bucket_id is None:
                continue

            bucket = BUCKETS[bucket_id - 1]

            # Determine status
            if row['symbol'] in held_symbols:
                status = 'held'
            else:
                # Check if bucket is at max
                bucket_pos = [p for p in self.get_open_positions() if p['bucket_id'] == bucket_id]
                max_pos = 8 if bucket_id == 1 else (6 if bucket_id == 2 else 10)
                if len(bucket_pos) >= max_pos:
                    status = 'max_positions'
                else:
                    status = 'actionable'

            signals.append({
                'symbol': row['symbol'],
                'date': row['date'],
                'bucket_id': bucket_id,
                'bucket_name': bucket.name,
                'signal_type': row['signal_type'],
                'value_score_v2': row['value_score_v2'],
                'fundamentals_score': row['fundamentals_score'],
                'lt_score': row['lt_score'],
                'rsi': row['rsi'],
                'ev_ebitda': row['ev_ebitda'],
                'rev_growth': row['rev_growth'],
                'sector': row['sector'],
                'price': row['close_price'],
                'status': status,
            })

        return signals

    def _classify_signal_bucket(self, row) -> Optional[int]:
        """Classify which bucket a signal belongs to."""
        # Bucket 1
        if (row.get('value_score_v2', 0) >= 55 and
            row.get('fundamentals_score', 0) >= 18 and
            row.get('ev_ebitda', 0) >= 5 and row.get('ev_ebitda', 100) <= 20 and
            row.get('rev_growth', 0) > 10 and
            row.get('market_cap', 0) > 2e9 and
            row.get('analyst_count', 0) >= 6 and
            row.get('sector') in BUCKETS[0].strong_sectors):
            return 1

        # Bucket 2
        if (row.get('market_bearish', 0) == 1 and
            row.get('rsi', 100) < 40 and
            row.get('fundamentals_score', 0) >= 15 and
            row.get('value_score_v2', 0) >= 40 and
            row.get('market_cap', 0) > 1e9 and
            row.get('sector') in BUCKETS[1].strong_sectors):
            return 2

        # Bucket 3
        ev = row.get('ev_ebitda_clean', row.get('ev_ebitda', 0))
        if (row.get('eps_growth', 0) >= 35 and
            row.get('ebitda_growth', 0) >= 33 and
            ev >= 12 and ev <= 27 and
            row.get('rsi', 100) < 43 and
            row.get('sector') in BUCKETS[2].strong_sectors):
            return 3

        return None

    # ═══════════════════════════════════════════════════════════════════════════
    # BUCKET DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def get_bucket_configs(self) -> List[Dict[str, Any]]:
        """Get all bucket configurations."""
        return [get_bucket_summary(b) for b in BUCKETS]

    def get_regime_status(self) -> Dict[str, Any]:
        """Get current market regime status."""
        conn = self._get_backtest_conn()

        # Get recent signals to determine regime
        df = pd.read_sql_query("""
            SELECT market_risk_score FROM backtest_daily_scores
            ORDER BY date DESC LIMIT 100
        """, conn)
        conn.close()

        if len(df) == 0:
            return {'regime': 'unknown', 'bull_pct': 0, 'bear_pct': 0}

        bull_pct = (df['market_risk_score'] == 10).mean() * 100
        bear_pct = (df['market_risk_score'] == 0).mean() * 100

        if bull_pct > 50:
            regime = 'bull'
        elif bear_pct > 50:
            regime = 'bear'
        else:
            regime = 'neutral'

        return {
            'regime': regime,
            'bull_pct': bull_pct,
            'bear_pct': bear_pct,
            'bucket_2_active': regime == 'bear',
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        conn = self._get_portfolio_conn()

        trades = pd.read_sql_query("SELECT * FROM trade_history", conn)
        conn.close()

        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win_pct': 0,
                'avg_loss_pct': 0,
                'profit_factor': 0,
                'total_realized_pnl': 0,
                'best_trade': None,
                'worst_trade': None,
            }

        wins = trades[trades['realized_pnl'] > 0]
        losses = trades[trades['realized_pnl'] <= 0]

        avg_win = wins['realized_pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['realized_pnl_pct'].mean() if len(losses) > 0 else 0
        pf = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        best_idx = trades['realized_pnl_pct'].idxmax() if len(trades) > 0 else None
        worst_idx = trades['realized_pnl_pct'].idxmin() if len(trades) > 0 else None

        return {
            'total_trades': len(trades),
            'win_rate': (len(wins) / len(trades)) * 100,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': pf,
            'total_realized_pnl': trades['realized_pnl'].sum(),
            'best_trade': trades.loc[best_idx].to_dict() if best_idx is not None else None,
            'worst_trade': trades.loc[worst_idx].to_dict() if worst_idx is not None else None,
            'trades_by_bucket': trades.groupby('bucket_id').size().to_dict(),
            'pnl_by_bucket': trades.groupby('bucket_id')['realized_pnl'].sum().to_dict(),
        }

    def get_daily_returns(self) -> pd.DataFrame:
        """Get daily returns for charting."""
        conn = self._get_portfolio_conn()
        df = pd.read_sql_query(
            "SELECT date, total_equity FROM daily_snapshots ORDER BY date",
            conn
        )
        conn.close()

        if len(df) < 2:
            return pd.DataFrame()

        df['date'] = pd.to_datetime(df['date'])
        df['daily_return'] = df['total_equity'].pct_change() * 100
        df['cumulative_return'] = (df['total_equity'] / df['total_equity'].iloc[0] - 1) * 100

        return df

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY COMPARISON DATA
    # ═══════════════════════════════════════════════════════════════════════════

    def get_strategy_comparison(self) -> Dict[str, Any]:
        """
        Get comparison of stock vs options strategies.
        Returns backtested results for dashboard display.
        """
        # Stock strategies (from backtest results - 2024-2025)
        stock_strategies = [
            {
                'name': 'H_RegimeAdaptive',
                'type': 'stock',
                'description': 'Heavier allocation to bear dip during bears, quality during bulls',
                'return_pct': 15.8,
                'cagr_pct': 8.5,
                'trades': 149,
                'win_rate': 59.7,
                'profit_factor': 2.62,
                'max_drawdown_pct': -0.7,
                'sharpe': 2.54,
                'avg_return_pct': 8.9,
            },
            {
                'name': 'D_EarningsBeat',
                'type': 'stock',
                'description': 'Only trade stocks with earnings beat (>0%)',
                'return_pct': 14.6,
                'cagr_pct': 7.9,
                'trades': 125,
                'win_rate': 68.0,
                'profit_factor': 2.09,
                'max_drawdown_pct': -0.4,
                'sharpe': 2.74,
                'avg_return_pct': 9.7,
            },
            {
                'name': 'E_StrongSectors',
                'type': 'stock',
                'description': 'Only trade in each bucket strong sectors',
                'return_pct': 14.1,
                'cagr_pct': 7.6,
                'trades': 114,
                'win_rate': 67.5,
                'profit_factor': 2.46,
                'max_drawdown_pct': -1.0,
                'sharpe': 2.17,
                'avg_return_pct': 10.2,
            },
            {
                'name': 'I_BigBeatOnly',
                'type': 'stock',
                'description': 'Only trade stocks with big earnings beat (>10%)',
                'return_pct': 13.2,
                'cagr_pct': 7.2,
                'trades': 67,
                'win_rate': 74.6,
                'profit_factor': 2.22,
                'max_drawdown_pct': -0.5,
                'sharpe': 2.07,
                'avg_return_pct': 12.8,
            },
        ]

        # Options strategies (from backtest results - 2024-2025)
        options_strategies = [
            {
                'name': 'OTM5_EarningsBeat',
                'type': 'options',
                'description': '5% OTM calls, earnings beat filter, 25% stop',
                'return_pct': 242.2,
                'cagr_pct': 96.1,
                'trades': 166,
                'win_rate': 65.1,
                'profit_factor': 3.09,
                'max_drawdown_pct': -1.6,
                'sharpe': 1.96,
                'avg_return_pct': 38.5,
            },
            {
                'name': 'Combined_Best',
                'type': 'options',
                'description': 'OTM 5%, earnings beat, strong sectors, 25% stop',
                'return_pct': 184.8,
                'cagr_pct': 77.4,
                'trades': 103,
                'win_rate': 69.9,
                'profit_factor': 3.01,
                'max_drawdown_pct': -4.6,
                'sharpe': 1.57,
                'avg_return_pct': 41.6,
            },
            {
                'name': 'OTM5_StrongSectors',
                'type': 'options',
                'description': '5% OTM calls, strong sectors, 25% stop',
                'return_pct': 172.0,
                'cagr_pct': 73.0,
                'trades': 161,
                'win_rate': 57.1,
                'profit_factor': 2.98,
                'max_drawdown_pct': -5.0,
                'sharpe': 1.62,
                'avg_return_pct': 31.3,
            },
            {
                'name': 'ATM_EarningsBeat',
                'type': 'options',
                'description': 'ATM calls, earnings beat filter, 25% stop',
                'return_pct': 156.1,
                'cagr_pct': 67.3,
                'trades': 135,
                'win_rate': 66.7,
                'profit_factor': 2.39,
                'max_drawdown_pct': -1.9,
                'sharpe': 1.93,
                'avg_return_pct': 29.2,
            },
        ]

        return {
            'period': '2024-2025 (2 years)',
            'starting_capital': 100000,
            'stock_strategies': stock_strategies,
            'options_strategies': options_strategies,
            'best_stock': stock_strategies[0],
            'best_options': options_strategies[0],
            'summary': {
                'stock_best_return': 15.8,
                'options_best_return': 242.2,
                'options_multiplier': 15.3,  # Options return ~15x higher
                'stock_max_dd': -1.2,
                'options_max_dd': -5.0,
                'recommendation': 'Use stocks for core (70%) + options for alpha (30%)',
            }
        }

    def get_recommended_allocation(self, risk_tolerance: str = 'moderate') -> Dict[str, Any]:
        """
        Get recommended portfolio allocation based on risk tolerance.

        Args:
            risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        """
        allocations = {
            'conservative': {
                'stocks_pct': 90,
                'options_pct': 10,
                'cash_pct': 0,
                'stock_strategy': 'D_EarningsBeat',
                'options_strategy': 'ATM_EarningsBeat',
                'expected_return': '12-18%',
                'max_drawdown': '2-5%',
                'description': 'Primarily stocks with small options allocation for upside',
            },
            'moderate': {
                'stocks_pct': 70,
                'options_pct': 25,
                'cash_pct': 5,
                'stock_strategy': 'H_RegimeAdaptive',
                'options_strategy': 'Combined_Best',
                'expected_return': '25-50%',
                'max_drawdown': '5-10%',
                'description': 'Balanced approach with meaningful options exposure',
            },
            'aggressive': {
                'stocks_pct': 40,
                'options_pct': 55,
                'cash_pct': 5,
                'stock_strategy': 'I_BigBeatOnly',
                'options_strategy': 'OTM5_EarningsBeat',
                'expected_return': '75-150%',
                'max_drawdown': '10-20%',
                'description': 'Options-heavy for maximum growth potential',
            },
        }

        alloc = allocations.get(risk_tolerance, allocations['moderate'])
        alloc['risk_tolerance'] = risk_tolerance

        return alloc

    def get_combined_portfolio_status(self) -> Dict[str, Any]:
        """
        Get combined portfolio status for both stock and options positions.
        This is for live tracking once positions are opened.
        """
        # Get stock portfolio
        stock_summary = self.get_portfolio_summary()
        stock_positions = self.get_open_positions()

        # For now, options are tracked separately (could be extended)
        # In a real implementation, you'd have a separate options_positions table

        return {
            'stocks': {
                'equity': stock_summary['total_equity'],
                'return_pct': stock_summary['total_return_pct'],
                'positions': len(stock_positions),
                'unrealized_pnl': stock_summary['unrealized_pnl'],
            },
            'options': {
                'equity': 0,  # Placeholder - would track actual options positions
                'return_pct': 0,
                'positions': 0,
                'unrealized_pnl': 0,
            },
            'combined': {
                'total_equity': stock_summary['total_equity'],
                'total_return_pct': stock_summary['total_return_pct'],
                'total_positions': len(stock_positions),
            },
            'last_updated': datetime.now().isoformat(),
        }

    def get_active_signals_for_strategy(self, strategy_type: str = 'stock') -> List[Dict[str, Any]]:
        """
        Get actionable signals for a specific strategy type.

        Args:
            strategy_type: 'stock' or 'options'
        """
        signals = self.get_current_signals(days_back=7)
        actionable = [s for s in signals if s['status'] == 'actionable']

        # For options, we'd filter further based on options-specific criteria
        # (e.g., sufficient liquidity, appropriate IV)
        if strategy_type == 'options':
            # Options prefer earnings beat signals
            actionable = [s for s in actionable if s.get('earnings_beat', False)]

        return actionable


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data = DashboardData()

    print("=" * 100)
    print("  DASHBOARD DATA TEST (v2)")
    print("=" * 100)

    print("\n  Portfolio Summary:")
    summary = data.get_portfolio_summary()
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"    {k}: {v:,.2f}")
        else:
            print(f"    {k}: {v}")

    print("\n  Bucket Summary:")
    for bucket in data.get_position_summary_by_bucket():
        print(f"    {bucket['short_name']}: {bucket['position_count']}/{bucket['max_positions']} "
              f"(${bucket['total_value']:,.2f}, {bucket['unrealized_pnl']:+,.2f} P&L)")

    print("\n  Regime Status:")
    regime = data.get_regime_status()
    print(f"    Regime: {regime['regime'].upper()}")
    print(f"    Bull: {regime['bull_pct']:.1f}% | Bear: {regime['bear_pct']:.1f}%")
    print(f"    Bucket 2 Active: {regime['bucket_2_active']}")

    # Strategy Comparison
    print("\n" + "=" * 100)
    print("  STRATEGY COMPARISON (2024-2025)")
    print("=" * 100)

    comparison = data.get_strategy_comparison()
    print(f"\n  Period: {comparison['period']}")
    print(f"  Starting Capital: ${comparison['starting_capital']:,}")

    print("\n  TOP STOCK STRATEGIES:")
    print(f"  {'Name':<20s} {'Return':>8s} {'CAGR':>7s} {'Trades':>7s} {'Win%':>6s} {'PF':>5s} {'MaxDD':>7s}")
    print(f"  {'-'*65}")
    for s in comparison['stock_strategies'][:3]:
        print(f"  {s['name']:<20s} {s['return_pct']:>+7.1f}% {s['cagr_pct']:>+6.1f}% {s['trades']:>6} "
              f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} {s['max_drawdown_pct']:>+6.1f}%")

    print("\n  TOP OPTIONS STRATEGIES:")
    print(f"  {'Name':<20s} {'Return':>8s} {'CAGR':>7s} {'Trades':>7s} {'Win%':>6s} {'PF':>5s} {'MaxDD':>7s}")
    print(f"  {'-'*65}")
    for s in comparison['options_strategies'][:3]:
        print(f"  {s['name']:<20s} {s['return_pct']:>+7.1f}% {s['cagr_pct']:>+6.1f}% {s['trades']:>6} "
              f"{s['win_rate']:>5.1f}% {s['profit_factor']:>4.2f} {s['max_drawdown_pct']:>+6.1f}%")

    print(f"\n  Summary:")
    print(f"    Best Stock Return: {comparison['summary']['stock_best_return']:+.1f}%")
    print(f"    Best Options Return: {comparison['summary']['options_best_return']:+.1f}%")
    print(f"    Options Multiplier: {comparison['summary']['options_multiplier']:.1f}x")
    print(f"    Recommendation: {comparison['summary']['recommendation']}")

    # Allocation Recommendations
    print("\n" + "=" * 100)
    print("  ALLOCATION RECOMMENDATIONS")
    print("=" * 100)

    for risk in ['conservative', 'moderate', 'aggressive']:
        alloc = data.get_recommended_allocation(risk)
        print(f"\n  {risk.upper()}:")
        print(f"    Stocks: {alloc['stocks_pct']}% ({alloc['stock_strategy']})")
        print(f"    Options: {alloc['options_pct']}% ({alloc['options_strategy']})")
        print(f"    Expected Return: {alloc['expected_return']}")
        print(f"    Max Drawdown: {alloc['max_drawdown']}")

    print("\n  Current Signals (last 7 days):")
    signals = data.get_current_signals(7)
    actionable = [s for s in signals if s['status'] == 'actionable']
    print(f"    Total: {len(signals)}, Actionable: {len(actionable)}")
    for s in actionable[:5]:
        print(f"      {s['symbol']}: B{s['bucket_id']} {s['signal_type']} V2={s['value_score_v2']:.0f}")

    print("\n" + "=" * 100)

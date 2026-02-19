#!/usr/bin/env python3
"""
Score Validation Backtest
=========================
Validates that high Value Score and Long-Term Score predict better returns.
Tests quintile performance, component contributions, and overall predictive power.

This is the definitive test of whether the scoring system works.

Usage:
    python validate_scoring.py              # Full validation
    python validate_scoring.py --quick      # Quick summary only
"""

import sqlite3
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import BACKTEST_DB

# Import scoring functions from run_backtest
from run_backtest import (
    compute_technical_indicators,
    compute_spy_regime,
    compute_ttm_fundamentals,
    score_dataframe_v2,
    score_dataframe_v3,
)
from config import DATABASE_NAME


def load_gross_profitability_lookup(conn: sqlite3.Connection) -> dict:
    """Load gross profitability (GP/Assets) for all stocks."""
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

    # Get most recent year for each symbol
    income_latest = income.groupby('symbol').first().reset_index()
    balance_latest = balance.groupby('symbol').first().reset_index()

    # Merge and calculate GP/Assets
    merged = income_latest.merge(balance_latest[['symbol', 'total_assets']],
                                 on='symbol', how='inner')

    gp_lookup = {}
    for _, row in merged.iterrows():
        symbol = row['symbol']
        gp = row['gross_profit']
        assets = row['total_assets']

        if pd.notna(gp) and pd.notna(assets) and assets > 0:
            gp_ratio = gp / assets
            gp_lookup[symbol] = max(-0.2, min(gp_ratio, 0.5))

    return gp_lookup


def load_all_scored_data(conn: sqlite3.Connection, spy_regime: pd.DataFrame,
                         sample_size: int = None,
                         min_price: float = 5.0) -> pd.DataFrame:
    """
    Load and score all stocks. Returns a DataFrame with daily scores and forward returns.

    Args:
        conn: Database connection
        spy_regime: SPY market regime DataFrame
        sample_size: If set, only process this many symbols (for quick testing)
        min_price: Minimum stock price to include (filters penny stocks)
    """
    # Load gross profitability lookup
    print("Loading gross profitability data...")
    gp_lookup = load_gross_profitability_lookup(conn)
    print(f"  Loaded GP data for {len(gp_lookup)} stocks")

    # Get all symbols with sufficient price data AND minimum price
    # This filters out penny stocks that cause extreme return outliers
    symbols = pd.read_sql_query(f"""
        SELECT DISTINCT symbol, COUNT(*) as days, AVG(adjusted_close) as avg_price
        FROM historical_prices
        GROUP BY symbol
        HAVING days >= 300 AND avg_price >= {min_price}
        ORDER BY symbol
    """, conn)['symbol'].tolist()

    if sample_size:
        symbols = symbols[:sample_size]

    print(f"Processing {len(symbols)} symbols...")

    all_data = []
    processed = 0

    for symbol in symbols:
        try:
            # Load prices
            prices = pd.read_sql_query("""
                SELECT date, open, high, low, close, volume, adjusted_close
                FROM historical_prices WHERE symbol = ? ORDER BY date
            """, conn, params=(symbol,))

            if len(prices) < 250:
                continue

            # Compute technical indicators
            prices = compute_technical_indicators(prices)

            # Merge market regime
            prices = prices.merge(spy_regime, on='date', how='left')
            prices['market_bullish'] = prices['market_bullish'].fillna(0).astype(int)

            # Compute fundamentals
            fundamentals = compute_ttm_fundamentals(symbol, conn)
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

            prices = prices.rename(columns={'ev_to_ebitda': 'ev_ebitda'})

            # Score the data
            scoreable = prices.dropna(subset=['sma200']).copy()
            if scoreable.empty:
                continue

            scoreable['symbol'] = symbol  # Ensure symbol is available for GP lookup
            scored = score_dataframe_v3(scoreable, gp_lookup=gp_lookup)

            # Calculate forward returns
            scored = scored.sort_values('date').reset_index(drop=True)
            closes = scored['adjusted_close'].values

            # 1M, 2M, 3M, 6M, 12M forward returns
            for name, days in [('fwd_1m', 21), ('fwd_2m', 42), ('fwd_3m', 63),
                               ('fwd_6m', 126), ('fwd_12m', 252)]:
                returns = []
                for i in range(len(closes)):
                    if i + days < len(closes) and closes[i] > 0:
                        returns.append(((closes[i + days] / closes[i]) - 1) * 100)
                    else:
                        returns.append(np.nan)
                scored[name] = returns

            scored['symbol'] = symbol

            # Winsorize returns to cap extreme outliers (cap at +/- 100% for short, +/- 200% for long)
            for col in ['fwd_1m', 'fwd_2m', 'fwd_3m']:
                scored[col] = scored[col].clip(-100, 100)
            for col in ['fwd_6m', 'fwd_12m']:
                scored[col] = scored[col].clip(-100, 200)

            all_data.append(scored[['symbol', 'date', 'lt_score', 'value_score',
                                    'value_score_v2', 'value_score_v3', 'trend_score',
                                    'fundamentals_score', 'valuation_score', 'momentum_score',
                                    'momentum_12_1_score', 'high52w_score', 'gross_profitability_score',
                                    'market_risk_score', 'fwd_1m', 'fwd_2m', 'fwd_3m',
                                    'fwd_6m', 'fwd_12m']].dropna(subset=['fwd_3m']))

            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(symbols)} symbols...")

        except Exception as e:
            continue

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def analyze_quintiles(df: pd.DataFrame, score_col: str, return_col: str) -> pd.DataFrame:
    """Analyze performance by score quintile."""
    df_clean = df.dropna(subset=[score_col, return_col]).copy()

    if len(df_clean) < 100:
        return pd.DataFrame()

    # Use percentile-based bins instead of qcut to handle duplicates
    try:
        # Try qcut first with duplicates='drop'
        df_clean['quintile'] = pd.qcut(
            df_clean[score_col],
            q=5,
            labels=False,
            duplicates='drop'
        )
        # Map numeric bins to labels
        unique_bins = sorted(df_clean['quintile'].dropna().unique())
        if len(unique_bins) >= 5:
            label_map = {0: 'Q1 (Low)', 1: 'Q2', 2: 'Q3', 3: 'Q4', 4: 'Q5 (High)'}
        elif len(unique_bins) == 4:
            label_map = {0: 'Q1 (Low)', 1: 'Q2', 2: 'Q4', 3: 'Q5 (High)'}
        elif len(unique_bins) == 3:
            label_map = {0: 'Q1 (Low)', 1: 'Q3', 2: 'Q5 (High)'}
        else:
            # Fallback to percentile-based grouping
            raise ValueError("Not enough unique bins")

        df_clean['quintile'] = df_clean['quintile'].map(label_map)

    except (ValueError, KeyError):
        # Fallback: use percentiles directly
        percentiles = df_clean[score_col].rank(pct=True)
        df_clean['quintile'] = pd.cut(
            percentiles,
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'],
            include_lowest=True
        )

    # Aggregate by quintile
    stats = df_clean.groupby('quintile', observed=False).agg({
        score_col: 'mean',
        return_col: ['mean', 'median', 'std', 'count']
    })
    stats.columns = ['avg_score', 'avg_return', 'median_return', 'std_return', 'count']

    return stats


def analyze_sector_neutral_quintiles(df: pd.DataFrame, score_col: str,
                                       return_col: str = 'fwd_3m') -> dict:
    """
    Analyze performance using sector-neutral quintiles.

    Ranks stocks within each sector first, then combines quintiles across sectors.
    This isolates stock-picking skill from sector timing.
    """
    # Get sector mapping
    main_conn = sqlite3.connect(DATABASE_NAME)
    sector_map = dict(main_conn.execute(
        "SELECT symbol, sector FROM stock_consensus").fetchall())
    main_conn.close()

    df_clean = df.dropna(subset=[score_col, return_col]).copy()
    df_clean['sector'] = df_clean['symbol'].map(sector_map)
    df_clean = df_clean.dropna(subset=['sector'])

    if len(df_clean) < 500:
        return {'error': 'Insufficient data'}

    # Rank within each sector
    sector_quintiles = []

    for sector in df_clean['sector'].unique():
        sector_df = df_clean[df_clean['sector'] == sector].copy()

        if len(sector_df) < 50:  # Need enough stocks per sector
            continue

        try:
            sector_df['sector_quintile'] = pd.qcut(
                sector_df[score_col], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
        except ValueError:
            # Fallback to percentile ranks
            pcts = sector_df[score_col].rank(pct=True)
            sector_df['sector_quintile'] = pd.cut(
                pcts, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], include_lowest=True
            )

        sector_quintiles.append(sector_df)

    if not sector_quintiles:
        return {'error': 'No valid sectors'}

    combined = pd.concat(sector_quintiles, ignore_index=True)

    # Aggregate by sector-neutral quintile
    stats = combined.groupby('sector_quintile', observed=False).agg({
        return_col: ['mean', 'std', 'count']
    })
    stats.columns = ['avg_return', 'std_return', 'count']
    stats = stats.reset_index()

    # Calculate spread
    q5_ret = stats[stats['sector_quintile'] == 'Q5']['avg_return'].values
    q1_ret = stats[stats['sector_quintile'] == 'Q1']['avg_return'].values

    spread = (q5_ret[0] - q1_ret[0]) if len(q5_ret) > 0 and len(q1_ret) > 0 else 0

    return {
        'quintile_stats': stats,
        'spread': spread,
        'n_sectors': len(sector_quintiles),
        'n_observations': len(combined),
    }


def compare_to_benchmarks(df: pd.DataFrame, return_col: str = 'fwd_3m') -> dict:
    """
    Compare V3 scoring to naive benchmarks.

    Benchmarks:
    1. Equal-weight (all stocks average return)
    2. Pure 12-1 momentum only
    3. Pure gross profitability only
    4. Random selection (Q3 average as proxy)
    """
    results = {}

    # V3 Q5 (our top picks)
    v3_clean = df.dropna(subset=['value_score_v3', return_col])
    if len(v3_clean) > 100:
        v3_q5 = v3_clean[v3_clean['value_score_v3'] > v3_clean['value_score_v3'].quantile(0.8)]
        results['V3 Top 20%'] = {
            'avg_return': v3_q5[return_col].mean(),
            'count': len(v3_q5),
        }

    # Benchmark 1: Equal-weight (all stocks)
    all_returns = df.dropna(subset=[return_col])
    results['Equal Weight (All)'] = {
        'avg_return': all_returns[return_col].mean(),
        'count': len(all_returns),
    }

    # Benchmark 2: Pure 12-1 momentum
    mom_clean = df.dropna(subset=['momentum_12_1_score', return_col])
    if len(mom_clean) > 100:
        mom_q5 = mom_clean[mom_clean['momentum_12_1_score'] >
                           mom_clean['momentum_12_1_score'].quantile(0.8)]
        results['Pure Momentum Q5'] = {
            'avg_return': mom_q5[return_col].mean(),
            'count': len(mom_q5),
        }

    # Benchmark 3: Pure gross profitability
    gp_clean = df.dropna(subset=['gross_profitability_score', return_col])
    if len(gp_clean) > 100:
        gp_q5 = gp_clean[gp_clean['gross_profitability_score'] >
                          gp_clean['gross_profitability_score'].quantile(0.8)]
        results['Pure Gross Profit Q5'] = {
            'avg_return': gp_q5[return_col].mean(),
            'count': len(gp_q5),
        }

    # Benchmark 4: Random (middle quintile proxy)
    if len(v3_clean) > 100:
        q_low = v3_clean['value_score_v3'].quantile(0.4)
        q_high = v3_clean['value_score_v3'].quantile(0.6)
        random_proxy = v3_clean[(v3_clean['value_score_v3'] >= q_low) &
                                 (v3_clean['value_score_v3'] <= q_high)]
        results['Random (Q3 proxy)'] = {
            'avg_return': random_proxy[return_col].mean(),
            'count': len(random_proxy),
        }

    return results


def analyze_multi_horizon_returns(df: pd.DataFrame, score_col: str = 'value_score_v3') -> dict:
    """
    Analyze how V3 score predicts returns across multiple time horizons.
    """
    horizons = ['fwd_1m', 'fwd_2m', 'fwd_3m', 'fwd_6m', 'fwd_12m']
    results = {}

    for horizon in horizons:
        if horizon not in df.columns:
            continue

        clean = df.dropna(subset=[score_col, horizon])
        if len(clean) < 500:
            continue

        # Correlation
        corr = clean[score_col].corr(clean[horizon])

        # Q5-Q1 spread
        q5 = clean[clean[score_col] > clean[score_col].quantile(0.8)][horizon].mean()
        q1 = clean[clean[score_col] < clean[score_col].quantile(0.2)][horizon].mean()
        spread = q5 - q1

        results[horizon] = {
            'correlation': corr,
            'q5_return': q5,
            'q1_return': q1,
            'spread': spread,
            'n_obs': len(clean),
        }

    return results


def calculate_portfolio_statistics(df: pd.DataFrame, score_col: str = 'value_score_v3') -> dict:
    """
    Calculate portfolio-level statistics for Q5 portfolio.

    Simulates monthly rebalancing of top quintile stocks.
    Returns Sharpe, Sortino, max drawdown, and other risk metrics.
    """
    # Get monthly returns for Q5 portfolio
    df_clean = df.dropna(subset=[score_col, 'fwd_1m']).copy()

    if len(df_clean) < 500:
        return {'error': 'Insufficient data'}

    # Group by month-end date
    df_clean['year_month'] = df_clean['date'].str[:7]
    monthly_data = []

    for ym in sorted(df_clean['year_month'].unique()):
        month_df = df_clean[df_clean['year_month'] == ym]

        if len(month_df) < 50:
            continue

        # Q5 portfolio (top 20%)
        threshold = month_df[score_col].quantile(0.8)
        q5 = month_df[month_df[score_col] >= threshold]

        if len(q5) < 10:
            continue

        # Equal-weight monthly return
        q5_return = q5['fwd_1m'].mean()

        # Market (all stocks) return
        market_return = month_df['fwd_1m'].mean()

        monthly_data.append({
            'year_month': ym,
            'q5_return': q5_return,
            'market_return': market_return,
            'n_stocks': len(q5),
        })

    if len(monthly_data) < 12:
        return {'error': 'Not enough monthly data'}

    monthly_df = pd.DataFrame(monthly_data)

    # Calculate statistics
    q5_returns = monthly_df['q5_return'].values
    market_returns = monthly_df['market_return'].values

    # Annualize
    ann_return = np.mean(q5_returns) * 12
    ann_vol = np.std(q5_returns) * np.sqrt(12)

    # Risk-free rate assumption (2% annual)
    rf = 2.0 / 12  # Monthly

    # Sharpe ratio
    excess_returns = q5_returns - rf
    sharpe = np.mean(excess_returns) / np.std(q5_returns) * np.sqrt(12) if np.std(q5_returns) > 0 else 0

    # Sortino ratio (downside deviation only)
    negative_returns = q5_returns[q5_returns < rf]
    downside_dev = np.std(negative_returns) * np.sqrt(12) if len(negative_returns) > 0 else ann_vol
    sortino = (ann_return - 2.0) / downside_dev if downside_dev > 0 else 0

    # Max drawdown (cumulative returns)
    cumulative = np.cumprod(1 + q5_returns / 100)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) * 100

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Hit rate (% months Q5 beats market)
    hit_rate = np.mean(q5_returns > market_returns) * 100

    # Information ratio (vs market)
    excess_vs_market = q5_returns - market_returns
    tracking_error = np.std(excess_vs_market) * np.sqrt(12)
    info_ratio = np.mean(excess_vs_market) * 12 / tracking_error if tracking_error > 0 else 0

    return {
        'annualized_return': ann_return,
        'annualized_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar,
        'hit_rate': hit_rate,
        'information_ratio': info_ratio,
        'n_months': len(monthly_df),
        'avg_stocks_per_month': monthly_df['n_stocks'].mean(),
    }


def analyze_interaction_terms(df: pd.DataFrame, return_col: str = 'fwd_3m') -> dict:
    """
    Test interaction terms between factors.

    Key interactions:
    - Momentum × Valuation (GARP strategy)
    - Profitability × Momentum
    - Profitability × Valuation (Quality at Value)
    """
    results = {}

    # Need momentum, valuation, and profitability scores
    required = ['momentum_12_1_score', 'valuation_score', 'gross_profitability_score']
    df_clean = df.dropna(subset=required + [return_col]).copy()

    if len(df_clean) < 500:
        return {'error': 'Insufficient data'}

    # Create high/low groups for each factor
    for col in required:
        median = df_clean[col].median()
        df_clean[f'{col}_high'] = df_clean[col] > median

    # Test interaction: High Momentum × High Profitability
    for interaction_name, factor1, factor2 in [
        ('Momentum × Profitability', 'momentum_12_1_score', 'gross_profitability_score'),
        ('Momentum × Valuation', 'momentum_12_1_score', 'valuation_score'),
        ('Profitability × Valuation', 'gross_profitability_score', 'valuation_score'),
    ]:
        f1_high = f'{factor1}_high'
        f2_high = f'{factor2}_high'

        # Four quadrants
        quadrants = {
            'Both High': (df_clean[f1_high]) & (df_clean[f2_high]),
            'F1 High Only': (df_clean[f1_high]) & (~df_clean[f2_high]),
            'F2 High Only': (~df_clean[f1_high]) & (df_clean[f2_high]),
            'Both Low': (~df_clean[f1_high]) & (~df_clean[f2_high]),
        }

        quad_returns = {}
        for quad_name, mask in quadrants.items():
            subset = df_clean[mask]
            quad_returns[quad_name] = {
                'avg_return': subset[return_col].mean(),
                'count': len(subset),
            }

        # Calculate interaction effect
        # Interaction = (Both High - F1 High Only) - (F2 High Only - Both Low)
        both_high = quad_returns['Both High']['avg_return']
        f1_only = quad_returns['F1 High Only']['avg_return']
        f2_only = quad_returns['F2 High Only']['avg_return']
        both_low = quad_returns['Both Low']['avg_return']

        interaction_effect = (both_high - f1_only) - (f2_only - both_low)

        results[interaction_name] = {
            'quadrants': quad_returns,
            'interaction_effect': interaction_effect,
            'best_quadrant': max(quad_returns.items(), key=lambda x: x[1]['avg_return'])[0],
            'best_return': max(quad_returns.values(), key=lambda x: x['avg_return'])['avg_return'],
        }

    return results


def analyze_component_contribution(df: pd.DataFrame, return_col: str = 'fwd_3m') -> dict:
    """Analyze how each score component correlates with forward returns."""
    components = ['momentum_12_1_score', 'high52w_score', 'gross_profitability_score',
                  'trend_score', 'fundamentals_score', 'valuation_score', 'momentum_score',
                  'market_risk_score']

    results = {}
    for comp in components:
        if comp in df.columns:
            clean = df.dropna(subset=[comp, return_col])
            if len(clean) > 100:
                corr = clean[comp].corr(clean[return_col])
                results[comp] = {
                    'correlation': corr,
                    'avg_when_high': clean[clean[comp] > clean[comp].median()][return_col].mean(),
                    'avg_when_low': clean[clean[comp] <= clean[comp].median()][return_col].mean(),
                }

    return results


def run_validation(quick: bool = False):
    """Run the full scoring validation."""
    print("=" * 70)
    print("SCORE VALIDATION BACKTEST")
    print("=" * 70)
    print(f"Database: {BACKTEST_DB}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    conn = sqlite3.connect(BACKTEST_DB)

    # Compute SPY regime
    print("\nComputing market regime...")
    spy_regime = compute_spy_regime(conn)

    # Load and score all data
    sample_size = 200 if quick else None
    print(f"\nLoading and scoring {'sample of 200' if quick else 'all'} stocks...")
    df = load_all_scored_data(conn, spy_regime, sample_size=sample_size)

    if df.empty:
        print("ERROR: No data available for validation")
        conn.close()
        return

    print(f"\nTotal observations: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique symbols: {df['symbol'].nunique()}")

    # ==================== QUINTILE ANALYSIS ====================
    print("\n" + "=" * 70)
    print("QUINTILE ANALYSIS")
    print("=" * 70)

    for score_name, score_col in [('Value Score V3', 'value_score_v3'),
                                   ('Value Score V2', 'value_score_v2'),
                                   ('Value Score V1', 'value_score'),
                                   ('Long-Term Score', 'lt_score')]:
        print(f"\n--- {score_name} vs 3-Month Forward Returns ---")

        quintiles = analyze_quintiles(df, score_col, 'fwd_3m')
        if quintiles.empty:
            print("  Insufficient data")
            continue

        print(f"\n{'Quintile':<12} {'Avg Score':>10} {'Avg Ret%':>10} {'Med Ret%':>10} {'Std':>8} {'Count':>10}")
        print("-" * 65)

        for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
            if q in quintiles.index:
                row = quintiles.loc[q]
                print(f"{q:<12} {row['avg_score']:>10.1f} {row['avg_return']:>+9.2f}% "
                      f"{row['median_return']:>+9.2f}% {row['std_return']:>7.1f} {int(row['count']):>10,}")

        # Calculate Q5-Q1 spread
        if 'Q5 (High)' in quintiles.index and 'Q1 (Low)' in quintiles.index:
            q5_ret = quintiles.loc['Q5 (High)', 'avg_return']
            q1_ret = quintiles.loc['Q1 (Low)', 'avg_return']
            spread = q5_ret - q1_ret
            print("-" * 65)
            print(f"{'Q5-Q1 Spread':<12} {'':>10} {spread:>+9.2f}%")

            if spread > 0:
                print(f"  ✅ POSITIVE SPREAD: High {score_name} stocks outperform by {spread:.2f}%")
            else:
                print(f"  ❌ NEGATIVE SPREAD: High {score_name} stocks underperform by {abs(spread):.2f}%")

    # ==================== CORRELATION ANALYSIS ====================
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    for return_col, period in [('fwd_1m', '1-Month'), ('fwd_2m', '2-Month'), ('fwd_3m', '3-Month')]:
        print(f"\n--- {period} Forward Returns ---")

        correlations = {}
        for score_col in ['value_score_v3', 'value_score_v2', 'value_score', 'lt_score']:
            clean = df.dropna(subset=[score_col, return_col])
            if len(clean) > 100:
                correlations[score_col] = clean[score_col].corr(clean[return_col])

        for name, corr in correlations.items():
            indicator = "✅" if corr > 0 else "❌"
            print(f"  {name:20s}: r = {corr:+.4f} {indicator}")

    # ==================== COMPONENT CONTRIBUTION ====================
    print("\n" + "=" * 70)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 70)
    print("Which score components best predict 3-month returns?\n")

    contributions = analyze_component_contribution(df)

    print(f"{'Component':<22} {'Correlation':>12} {'High→Return':>14} {'Low→Return':>12} {'Spread':>10}")
    print("-" * 75)

    for comp, stats in sorted(contributions.items(), key=lambda x: x[1]['correlation'], reverse=True):
        spread = stats['avg_when_high'] - stats['avg_when_low']
        indicator = "✅" if stats['correlation'] > 0 else "❌"
        print(f"{comp:<22} {stats['correlation']:>+11.4f} {stats['avg_when_high']:>+13.2f}% "
              f"{stats['avg_when_low']:>+11.2f}% {spread:>+9.2f}% {indicator}")

    # ==================== SECTOR-NEUTRAL ANALYSIS ====================
    print("\n" + "=" * 70)
    print("SECTOR-NEUTRAL PORTFOLIO ANALYSIS")
    print("=" * 70)
    print("Ranking stocks within each sector, then combining quintiles.\n")

    sector_neutral = analyze_sector_neutral_quintiles(df, 'value_score_v3', 'fwd_3m')

    if 'error' not in sector_neutral:
        sn_stats = sector_neutral['quintile_stats']
        print(f"{'Quintile':<12} {'Avg Return':>12} {'Std Dev':>10} {'Count':>10}")
        print("-" * 50)

        for _, row in sn_stats.iterrows():
            print(f"{row['sector_quintile']:<12} {row['avg_return']:>+11.2f}% "
                  f"{row['std_return']:>9.1f} {int(row['count']):>10,}")

        print("-" * 50)
        print(f"{'Q5-Q1 Spread':<12} {sector_neutral['spread']:>+11.2f}%")
        print(f"\nCoverage: {sector_neutral['n_sectors']} sectors, {sector_neutral['n_observations']:,} observations")

        if sector_neutral['spread'] > 0:
            print(f"  V3 stock-picking generates {sector_neutral['spread']:.2f}% spread WITHIN sectors")
        else:
            print(f"  V3 spread may be partially due to sector timing, not stock-picking")
    else:
        print(f"  Error: {sector_neutral.get('error', 'Unknown')}")

    # ==================== BENCHMARK COMPARISONS ====================
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISONS")
    print("=" * 70)
    print("Comparing V3 Top 20% to naive benchmarks (3-month returns).\n")

    benchmarks = compare_to_benchmarks(df, 'fwd_3m')

    print(f"{'Strategy':<25} {'Avg Return':>12} {'Count':>10}")
    print("-" * 50)

    for name, stats in sorted(benchmarks.items(),
                               key=lambda x: x[1]['avg_return'], reverse=True):
        print(f"{name:<25} {stats['avg_return']:>+11.2f}% {stats['count']:>10,}")

    # Calculate excess returns
    print("\n--- Excess Returns vs Benchmarks ---")
    v3_return = benchmarks.get('V3 Top 20%', {}).get('avg_return', 0)

    for name, stats in benchmarks.items():
        if name != 'V3 Top 20%':
            excess = v3_return - stats['avg_return']
            indicator = "outperforms" if excess > 0 else "underperforms"
            print(f"  V3 {indicator} {name}: {excess:+.2f}%")

    # ==================== MULTI-HORIZON ANALYSIS ====================
    print("\n" + "=" * 70)
    print("MULTI-HORIZON RETURN ANALYSIS")
    print("=" * 70)
    print("How does V3 predictive power change across time horizons?\n")

    multi_horizon = analyze_multi_horizon_returns(df, 'value_score_v3')

    print(f"{'Horizon':<12} {'Correlation':>12} {'Q5 Ret':>10} {'Q1 Ret':>10} {'Spread':>10}")
    print("-" * 60)

    for horizon, stats in multi_horizon.items():
        horizon_label = {'fwd_1m': '1-Month', 'fwd_2m': '2-Month', 'fwd_3m': '3-Month',
                         'fwd_6m': '6-Month', 'fwd_12m': '12-Month'}.get(horizon, horizon)
        print(f"{horizon_label:<12} {stats['correlation']:>+11.4f} {stats['q5_return']:>+9.2f}% "
              f"{stats['q1_return']:>+9.2f}% {stats['spread']:>+9.2f}%")

    # ==================== INTERACTION TERM ANALYSIS ====================
    print("\n" + "=" * 70)
    print("INTERACTION TERM ANALYSIS")
    print("=" * 70)
    print("Testing factor combinations (momentum x valuation, etc.)\n")

    interactions = analyze_interaction_terms(df, 'fwd_3m')

    if 'error' not in interactions:
        for interaction_name, stats in interactions.items():
            print(f"\n--- {interaction_name} ---")
            print(f"{'Quadrant':<20} {'Avg Return':>12} {'Count':>10}")
            print("-" * 45)

            for quad_name, quad_stats in stats['quadrants'].items():
                print(f"{quad_name:<20} {quad_stats['avg_return']:>+11.2f}% {quad_stats['count']:>10,}")

            print("-" * 45)
            print(f"Interaction Effect: {stats['interaction_effect']:+.2f}%")
            print(f"Best Quadrant: {stats['best_quadrant']} ({stats['best_return']:+.2f}%)")
    else:
        print(f"  Error: {interactions.get('error', 'Unknown')}")

    # ==================== SUB-PERIOD REGIME ANALYSIS ====================
    print("\n" + "=" * 70)
    print("SUB-PERIOD REGIME ANALYSIS")
    print("=" * 70)
    print("How does V3 perform across different market regimes?\n")

    # Define regime periods
    regimes = [
        ('Pre-2010', '2000-01-01', '2009-12-31'),
        ('2010s Bull', '2010-01-01', '2019-12-31'),
        ('COVID Crash', '2020-01-01', '2020-03-31'),
        ('COVID Recovery', '2020-04-01', '2021-12-31'),
        ('2022 Bear', '2022-01-01', '2022-12-31'),
        ('2023-2025 AI Bull', '2023-01-01', '2025-12-31'),
    ]

    print(f"{'Regime':<20} {'Dates':<25} {'V3 Corr':>10} {'Spread':>10} {'N':>10}")
    print("-" * 80)

    for regime_name, start_date, end_date in regimes:
        regime_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        clean = regime_df.dropna(subset=['value_score_v3', 'fwd_3m'])

        if len(clean) < 100:
            print(f"{regime_name:<20} {start_date[:7]}–{end_date[:7]:<13} {'N/A':>10} {'N/A':>10} {len(clean):>10,}")
            continue

        corr = clean['value_score_v3'].corr(clean['fwd_3m'])

        # Q5-Q1 spread
        q5 = clean[clean['value_score_v3'] > clean['value_score_v3'].quantile(0.8)]['fwd_3m'].mean()
        q1 = clean[clean['value_score_v3'] < clean['value_score_v3'].quantile(0.2)]['fwd_3m'].mean()
        spread = q5 - q1

        print(f"{regime_name:<20} {start_date[:7]}–{end_date[:7]:<13} {corr:>+9.4f} {spread:>+9.2f}% {len(clean):>10,}")

    # ==================== STATISTICAL SIGNIFICANCE ====================
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE")
    print("=" * 70)

    # T-test for Q5 vs Q1
    for score_col, score_name in [('value_score_v3', 'V3'), ('value_score_v2', 'V2'), ('lt_score', 'LT')]:
        clean = df.dropna(subset=[score_col, 'fwd_3m'])
        median_score = clean[score_col].median()

        high_returns = clean[clean[score_col] > clean[score_col].quantile(0.8)]['fwd_3m']
        low_returns = clean[clean[score_col] < clean[score_col].quantile(0.2)]['fwd_3m']

        if len(high_returns) > 30 and len(low_returns) > 30:
            # Simple t-statistic
            mean_diff = high_returns.mean() - low_returns.mean()
            pooled_std = np.sqrt((high_returns.std()**2 / len(high_returns)) +
                                  (low_returns.std()**2 / len(low_returns)))
            t_stat = mean_diff / pooled_std if pooled_std > 0 else 0

            print(f"\n{score_name} Score: Top 20% vs Bottom 20%")
            print(f"  Mean difference: {mean_diff:+.2f}%")
            print(f"  T-statistic: {t_stat:.2f}")
            print(f"  Significance: {'Strong (|t|>2.0)' if abs(t_stat) > 2 else 'Moderate (|t|>1.5)' if abs(t_stat) > 1.5 else 'Weak'}")

    # ==================== PORTFOLIO STATISTICS ====================
    print("\n" + "=" * 70)
    print("PORTFOLIO STATISTICS (Q5 Portfolio)")
    print("=" * 70)
    print("Simulating monthly rebalancing of top 20% V3 stocks.\n")

    port_stats = calculate_portfolio_statistics(df, 'value_score_v3')

    if 'error' not in port_stats:
        print(f"Performance Metrics:")
        print(f"  Annualized Return:     {port_stats['annualized_return']:>+8.2f}%")
        print(f"  Annualized Volatility: {port_stats['annualized_volatility']:>8.2f}%")
        print(f"  Max Drawdown:          {port_stats['max_drawdown']:>8.2f}%")
        print(f"  Months Analyzed:       {port_stats['n_months']:>8}")
        print(f"  Avg Stocks/Month:      {port_stats['avg_stocks_per_month']:>8.0f}")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio:          {port_stats['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:         {port_stats['sortino_ratio']:>8.2f}")
        print(f"  Calmar Ratio:          {port_stats['calmar_ratio']:>8.2f}")

        print(f"\nRelative Performance:")
        print(f"  Hit Rate vs Market:    {port_stats['hit_rate']:>7.1f}%")
        print(f"  Information Ratio:     {port_stats['information_ratio']:>8.2f}")

        # Interpretation
        if port_stats['sharpe_ratio'] > 1.0:
            print(f"\n  V3 Q5 portfolio shows EXCELLENT risk-adjusted returns (Sharpe > 1)")
        elif port_stats['sharpe_ratio'] > 0.5:
            print(f"\n  V3 Q5 portfolio shows GOOD risk-adjusted returns (Sharpe > 0.5)")
        elif port_stats['sharpe_ratio'] > 0:
            print(f"\n  V3 Q5 portfolio shows POSITIVE risk-adjusted returns")
        else:
            print(f"\n  V3 Q5 portfolio needs improvement (negative Sharpe)")
    else:
        print(f"  Error: {port_stats.get('error', 'Unknown')}")

    # ==================== FINAL VERDICT ====================
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    # Calculate key metrics for verdict
    v3_quintiles = analyze_quintiles(df, 'value_score_v3', 'fwd_3m')
    v2_quintiles = analyze_quintiles(df, 'value_score_v2', 'fwd_3m')
    lt_quintiles = analyze_quintiles(df, 'lt_score', 'fwd_3m')

    v3_spread = 0
    v2_spread = 0
    lt_spread = 0

    if not v3_quintiles.empty and 'Q5 (High)' in v3_quintiles.index and 'Q1 (Low)' in v3_quintiles.index:
        v3_spread = v3_quintiles.loc['Q5 (High)', 'avg_return'] - v3_quintiles.loc['Q1 (Low)', 'avg_return']

    if not v2_quintiles.empty and 'Q5 (High)' in v2_quintiles.index and 'Q1 (Low)' in v2_quintiles.index:
        v2_spread = v2_quintiles.loc['Q5 (High)', 'avg_return'] - v2_quintiles.loc['Q1 (Low)', 'avg_return']

    if not lt_quintiles.empty and 'Q5 (High)' in lt_quintiles.index and 'Q1 (Low)' in lt_quintiles.index:
        lt_spread = lt_quintiles.loc['Q5 (High)', 'avg_return'] - lt_quintiles.loc['Q1 (Low)', 'avg_return']

    clean = df.dropna(subset=['value_score_v3', 'fwd_3m'])
    v3_corr = clean['value_score_v3'].corr(clean['fwd_3m']) if len(clean) > 100 else 0

    clean = df.dropna(subset=['value_score_v2', 'fwd_3m'])
    v2_corr = clean['value_score_v2'].corr(clean['fwd_3m']) if len(clean) > 100 else 0

    clean = df.dropna(subset=['lt_score', 'fwd_3m'])
    lt_corr = clean['lt_score'].corr(clean['fwd_3m']) if len(clean) > 100 else 0

    print(f"""
Scoring System Performance Summary:
-----------------------------------
Value Score V3 (with 12-1 momentum):
  - Q5-Q1 Spread: {v3_spread:+.2f}% (3-month)
  - Correlation:  {v3_corr:+.4f}

Value Score V2:
  - Q5-Q1 Spread: {v2_spread:+.2f}% (3-month)
  - Correlation:  {v2_corr:+.4f}

Long-Term Score:
  - Q5-Q1 Spread: {lt_spread:+.2f}% (3-month)
  - Correlation:  {lt_corr:+.4f}

V3 vs V2 Improvement: {v3_spread - v2_spread:+.2f}% spread increase
""")

    # Determine overall verdict
    if v3_spread > 2.0:
        print("""
✅ V3 SCORING SYSTEM IS STRONGLY PREDICTIVE

Value Score V3 (with 12-1 momentum) shows a strong Q5-Q1 spread.
High-scoring stocks significantly outperform low-scoring stocks.

Recommendation: Use V3 as the primary scoring metric.
""")
    elif v3_spread > 1.0:
        print("""
✅ V3 SCORING SYSTEM IS PREDICTIVE

Value Score V3 shows meaningful predictive power.
12-1 momentum is providing additional signal.

Recommendation: Use V3 for ranking, continue exploring additional factors.
""")
    elif v3_spread > 0:
        print("""
⚠️ V3 SCORING SYSTEM HAS MODERATE PREDICTIVE POWER

Positive but moderate spread. Momentum is helping.

Recommendation:
- Use V3 as one factor among many
- Continue exploring additional factors
""")
    else:
        print("""
❌ V3 SCORING NEEDS IMPROVEMENT

Negative or zero spread indicates the V3 methodology needs work.

Recommendation:
- Check momentum calculation for errors
- Review component weightings
- Consider sector-adjusting momentum thresholds
""")

    conn.close()
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate scoring system predictive power')
    parser.add_argument('--quick', action='store_true', help='Quick test with 200 symbols')
    args = parser.parse_args()

    run_validation(quick=args.quick)

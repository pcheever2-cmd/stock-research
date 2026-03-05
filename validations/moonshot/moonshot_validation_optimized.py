#!/usr/bin/env python3
"""
Optimized Moonshot Score Validation
====================================
Memory-efficient version that processes data in chunks to avoid 18GB memory usage.

Processes data by year chunks and saves intermediate results to disk.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import pickle
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')
CACHE_DIR = PROJECT_ROOT / 'moonshot_cache'
CACHE_DIR.mkdir(exist_ok=True)

# Split dates
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

FWD_HORIZON = 252  # trading days


def load_fundamentals():
    """Load fundamentals with TTM computations."""
    print("=" * 80)
    print("LOADING FUNDAMENTALS")
    print("=" * 80)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    fund = pd.read_sql_query("""
        SELECT i.symbol, i.date, i.period,
               i.revenue, i.gross_profit, i.eps_diluted as eps,
               i.net_income, i.ebitda,
               b.total_assets, b.total_equity,
               m.market_cap, m.enterprise_value, m.ev_to_ebitda,
               m.pe_ratio, m.pb_ratio
        FROM historical_income_statements i
        LEFT JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
        LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
        ORDER BY i.symbol, i.date
    """, conn)
    fund['date'] = pd.to_datetime(fund['date'])
    print(f"  {len(fund):,} fundamental records")

    conn.close()
    sys.stdout.flush()

    return fund


def compute_ttm_fundamentals(fund):
    """Compute TTM metrics efficiently."""
    print("\nComputing TTM fundamentals...")
    sys.stdout.flush()

    fund = fund.sort_values(['symbol', 'date']).copy()

    # TTM for flow metrics
    for col in ['revenue', 'gross_profit', 'eps', 'net_income', 'ebitda']:
        if col in fund.columns:
            fund[f'{col}_ttm'] = fund.groupby('symbol')[col].transform(
                lambda x: x.rolling(4, min_periods=4).sum()
            )

    # Ratios
    fund['gross_margin'] = fund['gross_profit_ttm'] / fund['revenue_ttm']
    fund['roe'] = fund['net_income_ttm'] / fund['total_equity']

    # YoY growth
    fund['revenue_growth'] = fund.groupby('symbol')['revenue_ttm'].pct_change(4, fill_method=None)
    fund['eps_growth'] = fund.groupby('symbol')['eps_ttm'].pct_change(4, fill_method=None)
    fund['ebitda_growth'] = fund.groupby('symbol')['ebitda_ttm'].pct_change(4, fill_method=None)

    # Margin trends
    fund['margin_improvement'] = fund.groupby('symbol')['gross_margin'].diff(4)

    # Clean infinities
    for col in ['gross_margin', 'roe', 'revenue_growth', 'eps_growth',
                'ebitda_growth', 'margin_improvement']:
        if col in fund.columns:
            fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

    print(f"  TTM metrics computed for {len(fund):,} records")
    sys.stdout.flush()

    return fund


def load_prices_for_symbol_batch(symbols, start_date=None, end_date=None):
    """Load prices for a batch of symbols."""
    conn = sqlite3.connect(BACKTEST_DB)

    placeholders = ','.join(['?' for _ in symbols])
    query = f"""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE symbol IN ({placeholders})
          AND adjusted_close > 0.5
    """
    params = list(symbols)

    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)

    query += " ORDER BY symbol, date"

    prices = pd.read_sql_query(query, conn, params=params)
    prices['date'] = pd.to_datetime(prices['date'])

    conn.close()

    return prices


def compute_forward_returns_batched(batch_size=500):
    """Compute forward returns in batches to save memory."""
    cache_file = CACHE_DIR / 'forward_returns.pkl'

    if cache_file.exists():
        print(f"\nLoading cached forward returns from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\nComputing {FWD_HORIZON}-day forward returns in batches...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Get all symbols
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM historical_prices ORDER BY symbol",
        conn
    )['symbol'].tolist()

    conn.close()

    all_fwd_returns = {}

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        prices = load_prices_for_symbol_batch(batch_symbols)

        for symbol in batch_symbols:
            sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
            if len(sym_prices) < FWD_HORIZON:
                continue

            sym_prices = sym_prices.set_index('date')
            fwd_ret = sym_prices['close'].pct_change(FWD_HORIZON).shift(-FWD_HORIZON)
            all_fwd_returns[symbol] = fwd_ret.to_dict()

        if (i + batch_size) % 1000 == 0:
            print(f"  Processed {i + batch_size:,}/{len(symbols):,} symbols")
            sys.stdout.flush()

    print(f"  Forward returns for {len(all_fwd_returns):,} symbols")

    # Cache results
    with open(cache_file, 'wb') as f:
        pickle.dump(all_fwd_returns, f)
    print(f"  Cached to {cache_file}")

    sys.stdout.flush()
    return all_fwd_returns


def compute_momentum_12_1_batched(batch_size=500):
    """Compute momentum in batches."""
    cache_file = CACHE_DIR / 'momentum_data.pkl'

    if cache_file.exists():
        print(f"\nLoading cached momentum from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f"\nComputing 12-1 month momentum in batches...")
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM historical_prices ORDER BY symbol",
        conn
    )['symbol'].tolist()
    conn.close()

    price_dict = {}

    for i in range(0, len(symbols), batch_size):
        batch_symbols = symbols[i:i+batch_size]
        prices = load_prices_for_symbol_batch(batch_symbols)

        for symbol in batch_symbols:
            sym_prices = prices[prices['symbol'] == symbol].sort_values('date')
            if len(sym_prices) < 252:
                continue

            sym_prices = sym_prices.set_index('date')
            price_dict[symbol] = {
                'close': sym_prices['close'].to_dict()
            }

        if (i + batch_size) % 1000 == 0:
            print(f"  Processed {i + batch_size:,}/{len(symbols):,} symbols")
            sys.stdout.flush()

    print(f"  Momentum data for {len(price_dict):,} symbols")

    # Cache results
    with open(cache_file, 'wb') as f:
        pickle.dump(price_dict, f)
    print(f"  Cached to {cache_file}")

    sys.stdout.flush()
    return price_dict


def get_momentum(price_dict, symbol, date):
    """Get 12-1 month momentum for a symbol at a date."""
    if symbol not in price_dict:
        return np.nan

    prices = price_dict[symbol]['close']
    dates = sorted([d for d in prices.keys() if d <= date])

    if len(dates) < 252:
        return np.nan

    price_12m = prices.get(dates[-252], np.nan)
    price_1m = prices.get(dates[-21] if len(dates) >= 21 else dates[0], np.nan)

    if pd.isna(price_12m) or price_12m <= 0:
        return np.nan

    return (price_1m - price_12m) / price_12m


def build_factor_dataset_chunked(fund, price_dict, fwd_returns, period_start, period_end):
    """Build factor dataset by processing quarterly chunks."""
    print(f"\nBuilding factor dataset for {period_start} to {period_end}...")
    sys.stdout.flush()

    # Generate quarterly rebalance dates
    rebal_dates = pd.date_range(period_start, period_end, freq='QE')

    all_data = []

    for idx, rebal_date in enumerate(rebal_dates):
        print(f"  Processing {rebal_date.date()} ({idx+1}/{len(rebal_dates)})...", end='')
        sys.stdout.flush()

        # Get latest fundamentals as of rebalance date
        fund_at_date = fund[fund['date'] <= rebal_date].groupby('symbol').last().reset_index()

        count = 0
        for _, row in fund_at_date.iterrows():
            symbol = row['symbol']

            # Get factors
            revenue_growth = row.get('revenue_growth', np.nan)
            eps_growth = row.get('eps_growth', np.nan)
            ebitda_growth = row.get('ebitda_growth', np.nan)
            gross_margin = row.get('gross_margin', np.nan)
            margin_improvement = row.get('margin_improvement', np.nan)
            roe = row.get('roe', np.nan)
            market_cap = row.get('market_cap', np.nan)
            ev_ebitda = row.get('ev_to_ebitda', np.nan)
            pe_ratio = row.get('pe_ratio', np.nan)

            # Skip if missing critical data
            if pd.isna(revenue_growth) or pd.isna(gross_margin) or pd.isna(market_cap):
                continue
            if market_cap <= 0:
                continue

            # Quality filters
            if revenue_growth > 5.0 or revenue_growth < -0.9:
                continue
            if gross_margin < 0 or gross_margin > 1:
                continue

            # Price factors
            momentum_12_1 = get_momentum(price_dict, symbol, rebal_date)

            # Forward return
            if symbol not in fwd_returns:
                continue
            fwd_dates = [d for d in fwd_returns[symbol].keys() if d <= rebal_date]
            if not fwd_dates:
                continue
            fwd_ret = fwd_returns[symbol][max(fwd_dates)]

            if pd.isna(fwd_ret):
                continue

            # Handle missing values
            eps_growth = 0 if pd.isna(eps_growth) else eps_growth
            ebitda_growth = 0 if pd.isna(ebitda_growth) else ebitda_growth
            margin_improvement = 0 if pd.isna(margin_improvement) else margin_improvement
            momentum_12_1 = 0 if pd.isna(momentum_12_1) else momentum_12_1
            roe = 0 if pd.isna(roe) else roe

            # Cap extreme values
            revenue_growth = np.clip(revenue_growth, -0.5, 2.0)
            eps_growth = np.clip(eps_growth, -0.5, 3.0)
            ebitda_growth = np.clip(ebitda_growth, -0.5, 3.0)
            momentum_12_1 = np.clip(momentum_12_1, -0.5, 2.0)
            roe = np.clip(roe, -0.5, 0.5)

            # Market cap segmentation
            if market_cap < 300e6:
                cap_segment = 'micro'
            elif market_cap < 2e9:
                cap_segment = 'small'
            elif market_cap < 10e9:
                cap_segment = 'mid'
            else:
                cap_segment = 'large'

            all_data.append({
                'symbol': symbol,
                'date': rebal_date,
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth,
                'ebitda_growth': ebitda_growth,
                'gross_margin': gross_margin,
                'margin_improvement': margin_improvement,
                'roe': roe,
                'momentum_12_1': momentum_12_1,
                'market_cap': market_cap,
                'cap_segment': cap_segment,
                'ev_ebitda': ev_ebitda,
                'pe_ratio': pe_ratio,
                'fwd_return': fwd_ret
            })
            count += 1

        print(f" {count} observations")
        sys.stdout.flush()

    df = pd.DataFrame(all_data)
    print(f"\n  Total: {len(df):,} observations collected")
    sys.stdout.flush()

    return df


def univariate_analysis(df, period_name):
    """Analyze individual factor correlations."""
    print(f"\n{'=' * 80}")
    print(f"UNIVARIATE ANALYSIS - {period_name}")
    print(f"{'=' * 80}")

    factors = ['revenue_growth', 'eps_growth', 'ebitda_growth', 'gross_margin',
               'margin_improvement', 'roe', 'momentum_12_1']

    print(f"\n{'Factor':<25} {'Correlation':<15} {'Obs':<12}")
    print("-" * 80)

    results = {}
    for factor in factors:
        test_df = df[[factor, 'fwd_return']].dropna()
        if len(test_df) > 100:
            corr = test_df[factor].corr(test_df['fwd_return'])
            results[factor] = corr
            print(f"{factor:<25} {corr:+.4f}          {len(test_df):,}")
        else:
            results[factor] = np.nan
            print(f"{factor:<25} {'N/A':<15} {len(test_df):,}")

    sys.stdout.flush()
    return results


def multivariate_regression(df, period_name):
    """Run multivariate regression."""
    print(f"\n{'=' * 80}")
    print(f"MULTIVARIATE REGRESSION - {period_name}")
    print(f"{'=' * 80}")

    factors = ['revenue_growth', 'eps_growth', 'ebitda_growth', 'gross_margin',
               'margin_improvement', 'roe', 'momentum_12_1']

    reg_df = df[factors + ['fwd_return']].dropna().copy()

    if len(reg_df) < 1000:
        print(f"  Insufficient data: {len(reg_df):,} observations")
        return None

    # Standardize factors
    for factor in factors:
        reg_df[f'{factor}_z'] = (reg_df[factor] - reg_df[factor].mean()) / reg_df[factor].std()

    z_cols = [f'{f}_z' for f in factors]
    reg_df = reg_df.dropna(subset=z_cols)

    # Run regression
    from sklearn.linear_model import LinearRegression

    X = reg_df[z_cols].values
    y = reg_df['fwd_return'].values

    model = LinearRegression()
    model.fit(X, y)

    print(f"\nObservations: {len(reg_df):,}")
    print(f"R²: {model.score(X, y):.4f}")
    print(f"\n{'Factor':<25} {'Coefficient':<15}")
    print("-" * 80)

    coefs = {}
    for i, factor in enumerate(factors):
        coef = model.coef_[i]
        coefs[factor] = coef
        print(f"{factor:<25} {coef:+.4f}")

    sys.stdout.flush()
    return coefs


def compute_moonshot_score(df_is, df_test, weights):
    """Compute Moonshot Score."""
    print(f"\nComputing Moonshot Score...")
    sys.stdout.flush()

    # Compute stats from IS data
    factor_stats = {}
    for factor in weights.keys():
        factor_stats[factor] = {
            'mean': df_is[factor].mean(),
            'std': df_is[factor].std()
        }

    # Apply to test set
    df_scored = df_test.copy()

    for factor in weights.keys():
        df_scored[f'{factor}_z'] = (
            (df_scored[factor] - factor_stats[factor]['mean']) /
            factor_stats[factor]['std']
        )

    # Compute weighted score
    df_scored['moonshot_raw'] = sum(
        df_scored[f'{factor}_z'] * weight
        for factor, weight in weights.items()
    )

    return df_scored


def quintile_analysis(df, score_col='moonshot_raw', return_col='fwd_return', name=''):
    """Run quintile analysis."""
    test_df = df[[score_col, return_col, 'market_cap']].dropna().copy()

    if len(test_df) < 100:
        return None

    try:
        test_df['quintile'] = pd.qcut(test_df[score_col], 5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        return None

    q_rets = test_df.groupby('quintile', observed=False)[return_col].mean()

    result = {
        'Q1': q_rets.get(1, np.nan),
        'Q2': q_rets.get(2, np.nan),
        'Q3': q_rets.get(3, np.nan),
        'Q4': q_rets.get(4, np.nan),
        'Q5': q_rets.get(5, np.nan),
        'spread': q_rets.get(5, 0) - q_rets.get(1, 0),
        'count': len(test_df)
    }

    print(f"\n{name}")
    print(f"  Q1 (Low Growth):  {result['Q1']:+.2%}")
    print(f"  Q2:               {result['Q2']:+.2%}")
    print(f"  Q3:               {result['Q3']:+.2%}")
    print(f"  Q4:               {result['Q4']:+.2%}")
    print(f"  Q5 (High Growth): {result['Q5']:+.2%}")
    print(f"  Spread (Q5-Q1):   {result['spread']:+.2%}")
    print(f"  Observations:     {result['count']:,}")

    is_monotonic = all(q_rets.get(i, 0) <= q_rets.get(i+1, 0) for i in range(1, 5))
    print(f"  Monotonic:        {'✓ Yes' if is_monotonic else '✗ No'}")

    sys.stdout.flush()
    return result


def market_cap_segmentation(df, score_col='moonshot_raw'):
    """Analyze by market cap."""
    print(f"\n{'=' * 80}")
    print("MARKET CAP SEGMENTATION")
    print(f"{'=' * 80}")

    results = {}
    for segment in ['micro', 'small', 'mid', 'large']:
        seg_df = df[df['cap_segment'] == segment]
        if len(seg_df) >= 100:
            result = quintile_analysis(seg_df, score_col, 'fwd_return', f'{segment.upper()} CAP')
            results[segment] = result

    return results


def main():
    print("=" * 80)
    print("OPTIMIZED MOONSHOT SCORE VALIDATION")
    print(f"In-Sample:  {IS_START} to {IS_END}")
    print(f"Out-of-Sample: {OOS_START} to {OOS_END}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    # Load fundamentals
    fund = load_fundamentals()
    fund = compute_ttm_fundamentals(fund)

    # Precompute momentum and forward returns (batched, cached)
    price_dict = compute_momentum_12_1_batched(batch_size=500)
    fwd_returns = compute_forward_returns_batched(batch_size=500)

    # Build datasets (chunked by quarter)
    print(f"\n{'=' * 80}")
    print("BUILDING IN-SAMPLE DATASET (1995-2019)")
    print(f"{'=' * 80}")
    df_is = build_factor_dataset_chunked(fund, price_dict, fwd_returns, IS_START, IS_END)

    # Save IS dataset
    is_cache = CACHE_DIR / 'df_is.pkl'
    with open(is_cache, 'wb') as f:
        pickle.dump(df_is, f)
    print(f"Saved IS dataset to {is_cache}")

    print(f"\n{'=' * 80}")
    print("BUILDING OUT-OF-SAMPLE DATASET (2020-2026)")
    print(f"{'=' * 80}")
    df_oos = build_factor_dataset_chunked(fund, price_dict, fwd_returns, OOS_START, OOS_END)

    # Save OOS dataset
    oos_cache = CACHE_DIR / 'df_oos.pkl'
    with open(oos_cache, 'wb') as f:
        pickle.dump(df_oos, f)
    print(f"Saved OOS dataset to {oos_cache}")

    # Univariate analysis
    print(f"\n{'=' * 80}")
    print("STEP 1: UNIVARIATE FACTOR ANALYSIS")
    print(f"{'=' * 80}")

    is_corrs = univariate_analysis(df_is, "IN-SAMPLE")
    oos_corrs = univariate_analysis(df_oos, "OUT-OF-SAMPLE")

    # Multivariate regression
    print(f"\n{'=' * 80}")
    print("STEP 2: MULTIVARIATE REGRESSION")
    print(f"{'=' * 80}")

    coefs = multivariate_regression(df_is, "IN-SAMPLE")

    # Design weights
    print(f"\n{'=' * 80}")
    print("STEP 3: OPTIMAL WEIGHT DESIGN")
    print(f"{'=' * 80}")

    if coefs:
        positive_coefs = {k: max(v, 0) for k, v in coefs.items()}
        total = sum(positive_coefs.values())
        if total > 0:
            optimal_weights = {k: v/total for k, v in positive_coefs.items()}
        else:
            print("  WARNING: No positive coefficients, using equal weights")
            optimal_weights = {k: 1/len(coefs) for k in coefs.keys()}
    else:
        optimal_weights = {
            'revenue_growth': 0.25,
            'eps_growth': 0.20,
            'gross_margin': 0.15,
            'margin_improvement': 0.10,
            'momentum_12_1': 0.15,
            'ebitda_growth': 0.10,
            'roe': 0.05
        }

    print("\nOptimal Weights:")
    for factor, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
        print(f"  {factor:<25} {weight:.1%}")

    # Compute scores
    print(f"\n{'=' * 80}")
    print("STEP 4: COMPUTE MOONSHOT SCORES")
    print(f"{'=' * 80}")

    df_is_scored = compute_moonshot_score(df_is, df_is, optimal_weights)
    df_oos_scored = compute_moonshot_score(df_is, df_oos, optimal_weights)

    # Test performance
    print(f"\n{'=' * 80}")
    print("STEP 5: QUINTILE PERFORMANCE")
    print(f"{'=' * 80}")

    print("\n--- IN-SAMPLE (1995-2019) ---")
    is_result = quintile_analysis(df_is_scored, 'moonshot_raw', 'fwd_return', 'FULL UNIVERSE')

    print("\n--- OUT-OF-SAMPLE (2020-2026) ---")
    oos_result = quintile_analysis(df_oos_scored, 'moonshot_raw', 'fwd_return', 'FULL UNIVERSE')

    # Market cap segmentation
    print(f"\n{'=' * 80}")
    print("STEP 6: MARKET CAP SEGMENTATION (OOS)")
    print(f"{'=' * 80}")

    cap_results = market_cap_segmentation(df_oos_scored, 'moonshot_raw')

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    if is_result and oos_result:
        print(f"\nFull Universe Performance:")
        print(f"  In-Sample (1995-2019):     {is_result['spread']:+.2%} spread ({is_result['count']:,} obs)")
        print(f"  Out-of-Sample (2020-2026): {oos_result['spread']:+.2%} spread ({oos_result['count']:,} obs)")

        if oos_result['spread'] > 0.02:
            print(f"\n  ✓ PREDICTIVE - OOS spread > 2%")
        elif oos_result['spread'] > 0:
            print(f"\n  ⚠ WEAK - OOS spread positive but < 2%")
        else:
            print(f"\n  ✗ NOT PREDICTIVE - OOS spread negative")

    if cap_results:
        print(f"\nMarket Cap Segmentation (OOS):")
        for segment, result in cap_results.items():
            if result:
                print(f"  {segment.capitalize():<8} {result['spread']:+.2%}")

    print(f"\n{'=' * 80}")
    print("COMPLETED")
    print(f"{'=' * 80}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

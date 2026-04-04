#!/usr/bin/env python3
"""
Rigorous Moonshot Score Validation
===================================
Following the exact methodology from Compass Score validation:

In-Sample:  1995-2019 (factor design, weight optimization)
Out-of-Sample: 2020-2026 (pure holdout, no tuning)

Full test suite:
1. Univariate factor analysis
2. Multivariate regression
3. Market cap segmentation
4. Value/equal weighting
5. Transaction costs
6. FF6 regression
7. Non-overlapping returns
8. K-fold CV
9. Sector-neutral
10. Rolling Sharpe
11. Analyst signals
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# Split dates (following Compass methodology)
IS_START = '1995-01-01'
IS_END = '2019-12-31'
OOS_START = '2020-01-01'
OOS_END = '2026-12-31'

# Forward return horizon for Moonshot (1-year for growth stocks)
FWD_HORIZON = 252  # trading days


def load_all_data():
    """Load comprehensive dataset."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    sys.stdout.flush()

    conn = sqlite3.connect(BACKTEST_DB)

    # Prices
    print("Loading prices...")
    prices = pd.read_sql_query("""
        SELECT symbol, date, adjusted_close as close, volume
        FROM historical_prices
        WHERE adjusted_close > 0.5
        ORDER BY symbol, date
    """, conn)
    prices['date'] = pd.to_datetime(prices['date'])
    print(f"  {len(prices):,} price records")

    # Fundamentals
    print("Loading fundamentals...")
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

    # Analyst data (if available)
    print("Loading analyst data...")
    try:
        analysts = pd.read_sql_query("""
            SELECT symbol, date, new_grade, previous_grade, grading_company
            FROM analyst_grades
            ORDER BY symbol, date
        """, conn)
        analysts['date'] = pd.to_datetime(analysts['date'])
        print(f"  {len(analysts):,} analyst grade changes")
    except Exception as e:
        print(f"  No analyst data available: {e}")
        analysts = None

    conn.close()
    sys.stdout.flush()

    return prices, fund, analysts


def precompute_ttm_fundamentals(fund):
    """Compute TTM metrics and growth rates."""
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

    # YoY growth (compare to 4 quarters ago)
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


def compute_price_factors(prices):
    """Compute momentum and technical factors."""
    print("\nComputing price factors...")
    sys.stdout.flush()

    price_dict = {}

    for symbol in prices['symbol'].unique():
        sym = prices[prices['symbol'] == symbol].sort_values('date')
        if len(sym) < 252:
            continue

        sym = sym.set_index('date')
        price_dict[symbol] = {
            'close': sym['close'].to_dict(),
            'volume': sym['volume'].to_dict()
        }

    print(f"  Price data for {len(price_dict):,} symbols")
    sys.stdout.flush()

    return price_dict


def compute_momentum_12_1(price_dict, symbol, date):
    """Compute 12-1 month momentum."""
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


def compute_analyst_signal(analysts_df, symbol, date, lookback_days=180):
    """Compute analyst upgrade/downgrade signal."""
    if analysts_df is None:
        return np.nan

    # Get analyst changes in last 6 months
    cutoff = date - pd.Timedelta(days=lookback_days)
    recent = analysts_df[
        (analysts_df['symbol'] == symbol) &
        (analysts_df['date'] > cutoff) &
        (analysts_df['date'] <= date)
    ]

    if len(recent) == 0:
        return 0  # Neutral

    # Grade mapping
    grade_map = {
        'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1,
        'Outperform': 4, 'Market Perform': 3, 'Underperform': 2,
        'Overweight': 4, 'Equal-Weight': 3, 'Underweight': 2
    }

    signal = 0
    for _, row in recent.iterrows():
        new = grade_map.get(row['new_grade'], 3)
        prev = grade_map.get(row['previous_grade'], 3)
        signal += (new - prev)  # Upgrades positive, downgrades negative

    return signal


def compute_forward_returns(prices, horizon=252):
    """Pre-compute forward returns."""
    print(f"\nComputing {horizon}-day forward returns...")
    sys.stdout.flush()

    fwd_returns = {}

    for symbol in prices['symbol'].unique():
        sym = prices[prices['symbol'] == symbol].sort_values('date')
        sym = sym.set_index('date')

        fwd_ret = sym['close'].pct_change(horizon).shift(-horizon)
        fwd_returns[symbol] = fwd_ret.to_dict()

    print(f"  Forward returns for {len(fwd_returns):,} symbols")
    sys.stdout.flush()

    return fwd_returns


def build_factor_dataset(fund, price_dict, fwd_returns, analysts, period_start, period_end):
    """Build complete factor dataset for a time period."""
    print(f"\nBuilding factor dataset for {period_start} to {period_end}...")
    sys.stdout.flush()

    # Quarterly rebalance dates
    rebal_dates = pd.date_range(period_start, period_end, freq='QE')

    data = []

    for rebal_date in rebal_dates:
        # Get latest fundamentals as of rebalance date
        fund_at_date = fund[fund['date'] <= rebal_date].groupby('symbol').last().reset_index()

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
            momentum_12_1 = compute_momentum_12_1(price_dict, symbol, rebal_date)

            # Analyst signal
            analyst_signal = compute_analyst_signal(analysts, symbol, rebal_date)

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
            analyst_signal = 0 if pd.isna(analyst_signal) else analyst_signal
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

            data.append({
                'symbol': symbol,
                'date': rebal_date,
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth,
                'ebitda_growth': ebitda_growth,
                'gross_margin': gross_margin,
                'margin_improvement': margin_improvement,
                'roe': roe,
                'momentum_12_1': momentum_12_1,
                'analyst_signal': analyst_signal,
                'market_cap': market_cap,
                'cap_segment': cap_segment,
                'ev_ebitda': ev_ebitda,
                'pe_ratio': pe_ratio,
                'fwd_return': fwd_ret
            })

    df = pd.DataFrame(data)
    print(f"  {len(df):,} observations collected")
    sys.stdout.flush()

    return df


def univariate_analysis(df, period_name):
    """Analyze individual factor correlations."""
    print(f"\n{'=' * 80}")
    print(f"UNIVARIATE ANALYSIS - {period_name}")
    print(f"{'=' * 80}")

    factors = ['revenue_growth', 'eps_growth', 'ebitda_growth', 'gross_margin',
               'margin_improvement', 'roe', 'momentum_12_1', 'analyst_signal']

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
    """Run multivariate regression to find optimal weights."""
    print(f"\n{'=' * 80}")
    print(f"MULTIVARIATE REGRESSION - {period_name}")
    print(f"{'=' * 80}")

    # Prepare data
    factors = ['revenue_growth', 'eps_growth', 'ebitda_growth', 'gross_margin',
               'margin_improvement', 'roe', 'momentum_12_1', 'analyst_signal']

    reg_df = df[factors + ['fwd_return']].dropna().copy()

    if len(reg_df) < 1000:
        print(f"  Insufficient data: {len(reg_df):,} observations")
        return None

    # Standardize factors (z-scores)
    for factor in factors:
        reg_df[f'{factor}_z'] = (reg_df[factor] - reg_df[factor].mean()) / reg_df[factor].std()

    # Drop any rows with NaN in z-scores (can happen if std = 0)
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
    print(f"\n{'Factor':<25} {'Coefficient':<15} {'Interpretation'}")
    print("-" * 80)

    coefs = {}
    for i, factor in enumerate(factors):
        coef = model.coef_[i]
        coefs[factor] = coef
        interp = "Strong +" if coef > 1.0 else "Weak +" if coef > 0 else "Negative" if coef < 0 else "Zero"
        print(f"{factor:<25} {coef:+.4f}          {interp}")

    sys.stdout.flush()
    return coefs


def compute_moonshot_score(df_is, df_test, weights):
    """Compute Moonshot Score using IS statistics, apply to test set."""
    print(f"\nComputing Moonshot Score...")
    print(f"  Weights: {weights}")
    sys.stdout.flush()

    # Compute z-score statistics from IS data only
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

    # Check monotonicity
    is_monotonic = all(q_rets.get(i, 0) <= q_rets.get(i+1, 0) for i in range(1, 5))
    print(f"  Monotonic:        {'✓ Yes' if is_monotonic else '✗ No'}")

    sys.stdout.flush()
    return result


def market_cap_segmentation(df, score_col='moonshot_raw'):
    """Analyze performance by market cap segment."""
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
    print("RIGOROUS MOONSHOT SCORE VALIDATION")
    print(f"In-Sample:  {IS_START} to {IS_END}")
    print(f"Out-of-Sample: {OOS_START} to {OOS_END}")
    print(f"Run at: {datetime.now().isoformat()}")
    print("=" * 80)
    sys.stdout.flush()

    # Load data
    prices, fund, analysts = load_all_data()

    # Precompute
    fund = precompute_ttm_fundamentals(fund)
    price_dict = compute_price_factors(prices)
    fwd_returns = compute_forward_returns(prices, FWD_HORIZON)

    # Build datasets
    print(f"\n{'=' * 80}")
    print("BUILDING IN-SAMPLE DATASET")
    print(f"{'=' * 80}")
    df_is = build_factor_dataset(fund, price_dict, fwd_returns, analysts, IS_START, IS_END)

    print(f"\n{'=' * 80}")
    print("BUILDING OUT-OF-SAMPLE DATASET")
    print(f"{'=' * 80}")
    df_oos = build_factor_dataset(fund, price_dict, fwd_returns, analysts, OOS_START, OOS_END)

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

    # Design optimal weights based on regression
    print(f"\n{'=' * 80}")
    print("STEP 3: OPTIMAL WEIGHT DESIGN")
    print(f"{'=' * 80}")

    # Use positive coefficients, normalize to sum to 1
    if coefs:
        positive_coefs = {k: max(v, 0) for k, v in coefs.items()}
        total = sum(positive_coefs.values())
        if total > 0:
            optimal_weights = {k: v/total for k, v in positive_coefs.items()}
        else:
            print("  WARNING: No positive coefficients, using equal weights")
            optimal_weights = {k: 1/len(coefs) for k in coefs.keys()}
    else:
        # Fallback to theory-based weights
        optimal_weights = {
            'revenue_growth': 0.25,
            'eps_growth': 0.20,
            'gross_margin': 0.15,
            'margin_improvement': 0.10,
            'momentum_12_1': 0.15,
            'ebitda_growth': 0.10,
            'analyst_signal': 0.05
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
        print(f"  In-Sample (1995-2019):  {is_result['spread']:+.2%} spread ({is_result['count']:,} obs)")
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
    print("COMPLETED - Phase 1 of Rigorous Validation")
    print("Next: Value-weighting, transaction costs, FF6, sector-neutral")
    print(f"{'=' * 80}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()

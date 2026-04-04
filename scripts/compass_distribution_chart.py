#!/usr/bin/env python3
"""
Compass Score Distribution Visualization
=========================================
Creates a histogram showing:
1. Full universe distribution of Compass scores (0-100)
2. S&P 500 overlay showing quality clustering
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).parent
BACKTEST_DB = str(PROJECT_ROOT / 'backtest.db')

# S&P 500 components (approximate - major names)
SP500_CORE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'BRK-B', 'TSLA', 'UNH', 'JNJ',
    'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'LLY',
    'PEP', 'KO', 'COST', 'AVGO', 'PFE', 'TMO', 'MCD', 'WMT', 'CSCO', 'ABT',
    'DHR', 'ACN', 'CRM', 'NKE', 'TXN', 'NEE', 'PM', 'ORCL', 'UPS', 'MS',
    'RTX', 'HON', 'INTC', 'UNP', 'IBM', 'QCOM', 'LOW', 'GS', 'CAT', 'BA',
    'AMGN', 'AMD', 'SPGI', 'GE', 'SBUX', 'INTU', 'BLK', 'ISRG', 'AXP', 'MDLZ',
    'LMT', 'ADI', 'SYK', 'GILD', 'BKNG', 'TJX', 'MMC', 'VRTX', 'PLD', 'ADP',
    'CVS', 'C', 'REGN', 'CI', 'NOW', 'MO', 'SO', 'DUK', 'CL', 'EOG',
    'BSX', 'BDX', 'SLB', 'CME', 'MMM', 'NOC', 'WM', 'ITW', 'EQIX', 'CSX',
    'PNC', 'USB', 'TGT', 'APD', 'FCX', 'EMR', 'NSC', 'FDX', 'SHW', 'COP'
]

print("Loading data...")
conn = sqlite3.connect(BACKTEST_DB)

prices = pd.read_sql_query("""
    SELECT symbol, date, adjusted_close as close, volume
    FROM historical_prices
    WHERE adjusted_close > 0.5 AND date >= '2024-01-01'
    ORDER BY symbol, date
""", conn)
prices['date'] = pd.to_datetime(prices['date'])

fund = pd.read_sql_query("""
    SELECT i.symbol, i.date, i.gross_profit, i.net_income,
           b.total_assets, c.operating_cash_flow, c.free_cash_flow,
           m.market_cap
    FROM historical_income_statements i
    JOIN historical_balance_sheets b ON i.symbol = b.symbol AND i.date = b.date
    JOIN historical_cash_flows c ON i.symbol = c.symbol AND i.date = c.date
    LEFT JOIN historical_key_metrics m ON i.symbol = m.symbol AND i.date = m.date
    WHERE i.date >= '2023-01-01'
""", conn)
fund['date'] = pd.to_datetime(fund['date'])
conn.close()

# Compute factors
fund['roa'] = fund['net_income'] / fund['total_assets']
fund['ocf_assets'] = fund['operating_cash_flow'] / fund['total_assets']
fund['fcf_assets'] = fund['free_cash_flow'] / fund['total_assets']
fund['gp_assets'] = fund['gross_profit'] / fund['total_assets']
fund = fund.sort_values(['symbol', 'date'])
fund['asset_growth'] = fund.groupby('symbol')['total_assets'].pct_change(4)

for col in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth']:
    fund[col] = fund[col].replace([np.inf, -np.inf], np.nan)

print(f"  {len(prices):,} price records")
print(f"  {len(fund):,} fundamental records")

def compute_compass_score(symbol, as_of_date='2025-01-01'):
    """Compute raw Compass Score for a symbol."""
    as_of_dt = pd.to_datetime(as_of_date)

    sym_fund = fund[(fund['symbol'] == symbol) & (fund['date'] <= as_of_dt)]
    if len(sym_fund) == 0:
        return None, {}

    latest_fund = sym_fund.sort_values('date').iloc[-1]

    sym_prices = prices[(prices['symbol'] == symbol) & (prices['date'] <= as_of_dt)]
    if len(sym_prices) < 60:
        return None, {}

    sym_prices = sym_prices.sort_values('date').tail(60)
    rets = sym_prices['close'].pct_change().dropna()
    vol = rets.std() * np.sqrt(252) * 100

    factors = {
        'roa': latest_fund['roa'],
        'ocf_assets': latest_fund['ocf_assets'],
        'fcf_assets': latest_fund['fcf_assets'],
        'gp_assets': latest_fund['gp_assets'],
        'asset_growth': latest_fund['asset_growth'],
        'vol_60d': vol,
        'market_cap': latest_fund['market_cap']
    }

    stats = {
        'roa': {'mean': 0.02, 'std': 0.15},
        'ocf_assets': {'mean': 0.05, 'std': 0.12},
        'fcf_assets': {'mean': 0.03, 'std': 0.15},
        'gp_assets': {'mean': 0.25, 'std': 0.20},
        'asset_growth': {'mean': 0.10, 'std': 0.30},
        'vol_60d': {'mean': 45, 'std': 25}
    }

    z_scores = {}
    for f in ['roa', 'ocf_assets', 'fcf_assets', 'gp_assets', 'asset_growth', 'vol_60d']:
        val = factors[f]
        if pd.isna(val):
            return None, factors
        z_scores[f] = (val - stats[f]['mean']) / stats[f]['std']

    score = (
        z_scores['roa'] * 0.20 +
        z_scores['ocf_assets'] * 0.15 +
        z_scores['fcf_assets'] * 0.15 +
        z_scores['gp_assets'] * 0.20 +
        (-z_scores['vol_60d']) * 0.15 +
        (-z_scores['asset_growth']) * 0.15
    )

    return score, factors

# Compute scores for universe
print("\nComputing scores for universe...")
all_symbols = prices['symbol'].unique()
scores = []

for symbol in all_symbols:
    score, factors = compute_compass_score(symbol)
    if score is not None:
        scores.append({
            'symbol': symbol,
            'raw_score': score,
            'market_cap': factors.get('market_cap')
        })

scores_df = pd.DataFrame(scores)
print(f"  {len(scores_df):,} stocks with valid scores")

# Transform to 0-100 percentile scale
scores_df['percentile'] = scores_df['raw_score'].rank(pct=True) * 100

# Separate S&P 500
sp500_scores = scores_df[scores_df['symbol'].isin(SP500_CORE)]
print(f"  {len(sp500_scores)} S&P 500 stocks found")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("\nGenerating visualization...")

# Set up the figure with a clean style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Grade boundaries and colors
grade_boundaries = [0, 20, 40, 60, 85, 100]
grade_colors = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4']  # Red to Blue
grade_labels = ['F (High Risk)', 'D (Speculative)', 'C (Neutral)', 'B (Above Avg)', 'A (High Quality)']

# Plot full universe histogram
bins = np.arange(0, 101, 5)
n_full, bins_full, patches_full = ax.hist(
    scores_df['percentile'],
    bins=bins,
    alpha=0.6,
    color='#888888',
    edgecolor='white',
    linewidth=0.5,
    label=f'Full Universe (n={len(scores_df):,})'
)

# Color the bars by grade
for i, patch in enumerate(patches_full):
    bin_center = (bins[i] + bins[i+1]) / 2
    if bin_center < 20:
        patch.set_facecolor(grade_colors[0])
    elif bin_center < 40:
        patch.set_facecolor(grade_colors[1])
    elif bin_center < 60:
        patch.set_facecolor(grade_colors[2])
    elif bin_center < 85:
        patch.set_facecolor(grade_colors[3])
    else:
        patch.set_facecolor(grade_colors[4])
    patch.set_alpha(0.5)

# Overlay S&P 500 histogram
ax.hist(
    sp500_scores['percentile'],
    bins=bins,
    alpha=0.8,
    color='#2166ac',
    edgecolor='white',
    linewidth=0.5,
    label=f'S&P 500 (n={len(sp500_scores)})'
)

# Add vertical lines for grade boundaries
for boundary in [20, 40, 60, 85]:
    ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.3, linewidth=1)

# Add grade labels at top
y_max = ax.get_ylim()[1]
grade_centers = [10, 30, 50, 72.5, 92.5]
for center, label in zip(grade_centers, grade_labels):
    ax.text(center, y_max * 0.95, label, ha='center', va='top', fontsize=9,
            fontweight='bold', color='#333333')

# Add S&P 500 median line
sp500_median = sp500_scores['percentile'].median()
ax.axvline(x=sp500_median, color='#2166ac', linestyle='-', linewidth=2, alpha=0.8)
ax.text(sp500_median + 1, y_max * 0.7, f'S&P 500\nMedian: {sp500_median:.0f}',
        fontsize=10, color='#2166ac', fontweight='bold', va='top')

# Add full universe median line
full_median = scores_df['percentile'].median()
ax.axvline(x=full_median, color='#888888', linestyle='-', linewidth=2, alpha=0.6)
ax.text(full_median - 1, y_max * 0.5, f'Universe\nMedian: {full_median:.0f}',
        fontsize=10, color='#555555', fontweight='bold', va='top', ha='right')

# Labels and title
ax.set_xlabel('Compass Score (0-100 Percentile)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Stocks', fontsize=12, fontweight='bold')
ax.set_title('Compass Score Distribution: Full Universe vs S&P 500\n' +
             'S&P 500 stocks cluster in high-quality (A/B) grades',
             fontsize=14, fontweight='bold')

# Legend
ax.legend(loc='upper left', fontsize=10)

# Set x-axis limits
ax.set_xlim(0, 100)

# Add text box with key stats
stats_text = (
    f"Full Universe: {len(scores_df):,} stocks\n"
    f"S&P 500: {len(sp500_scores)} stocks\n"
    f"\n"
    f"S&P 500 in A/B grades: {len(sp500_scores[sp500_scores['percentile'] >= 60]) / len(sp500_scores) * 100:.0f}%\n"
    f"Full Universe in A/B: {len(scores_df[scores_df['percentile'] >= 60]) / len(scores_df) * 100:.0f}%"
)
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#cccccc')
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, family='monospace')

plt.tight_layout()

# Save the figure
output_path = PROJECT_ROOT / 'compass_distribution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"\nSaved: {output_path}")

# Also save a high-res version
output_path_hires = PROJECT_ROOT / 'compass_distribution_hires.png'
plt.savefig(output_path_hires, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path_hires}")

plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 70)
print("DISTRIBUTION SUMMARY")
print("=" * 70)

print(f"\nFull Universe ({len(scores_df):,} stocks):")
print(f"  Median: {scores_df['percentile'].median():.0f}")
print(f"  Mean: {scores_df['percentile'].mean():.0f}")
print(f"  Std Dev: {scores_df['percentile'].std():.1f}")

print(f"\nS&P 500 ({len(sp500_scores)} stocks):")
print(f"  Median: {sp500_scores['percentile'].median():.0f}")
print(f"  Mean: {sp500_scores['percentile'].mean():.0f}")
print(f"  Std Dev: {sp500_scores['percentile'].std():.1f}")

print(f"\nGrade Distribution Comparison:")
print(f"{'Grade':<20} {'Full Universe':<20} {'S&P 500':<20}")
print("-" * 60)
for grade, (low, high) in zip(grade_labels, [(0,20), (20,40), (40,60), (60,85), (85,100)]):
    full_pct = len(scores_df[(scores_df['percentile'] >= low) & (scores_df['percentile'] < high)]) / len(scores_df) * 100
    sp_pct = len(sp500_scores[(sp500_scores['percentile'] >= low) & (sp500_scores['percentile'] < high)]) / len(sp500_scores) * 100
    print(f"{grade:<20} {full_pct:>6.1f}%              {sp_pct:>6.1f}%")

print("\n" + "=" * 70)
print("COMPLETED")
print("=" * 70)

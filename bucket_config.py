#!/usr/bin/env python3
"""
Finalized Bucket Configuration
===============================
The definitive bucket definitions based on all analysis (v1, v2, final bucket analysis, trading strategy).
This file serves as the single source of truth for all dashboards and trading systems.

Bucket Summary:
- Bucket 1: Quality Growth Compounder (all-weather, stable)
- Bucket 2: Bear Market Dip Buy (bear-only, highest conviction when active)
- Bucket 3: High-Growth Momentum (all-weather, fastest)

All buckets benefit from Earnings Beat overlay (+4-5% win rate boost).
"""

from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional
import pandas as pd

# ══════════════════════════════════════════════════════════════════════════════
# BUCKET DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BucketConfig:
    """Configuration for a trading bucket."""
    id: int
    name: str
    short_name: str
    criteria_text: str

    # Position management
    position_pct: float          # % of portfolio per position
    max_positions: int           # max concurrent positions
    hold_months: int             # target holding period
    stop_loss_pct: float         # exit if down this much

    # Regime
    regime: str                  # 'all', 'bull', 'bear'

    # Historical performance (from analysis)
    historical_win_rate: float   # baseline win rate (3M)
    historical_win_rate_with_beat: float  # win rate with earnings beat overlay
    profit_factor: float
    avg_return_3m: float
    avg_return_1y: float
    bootstrap_ci_lo: float
    bootstrap_ci_hi: float

    # Sector guidance
    strong_sectors: List[str] = field(default_factory=list)
    avoid_sectors: List[str] = field(default_factory=list)

    # Filter function (set after class definition)
    filter_fn: Optional[Callable] = None


# ══════════════════════════════════════════════════════════════════════════════
# THE THREE FINALIZED BUCKETS
# ══════════════════════════════════════════════════════════════════════════════

BUCKET_1 = BucketConfig(
    id=1,
    name="Quality Growth Compounder",
    short_name="QGC",
    criteria_text="V2≥55 + Fund≥18 + EV[5-20] + RevG>10 + Cap>$2B + Analysts≥6",

    # Position management
    position_pct=3.0,
    max_positions=8,
    hold_months=6,
    stop_loss_pct=-15.0,

    # Regime
    regime='all',

    # Historical performance
    historical_win_rate=56.3,
    historical_win_rate_with_beat=62.6,
    profit_factor=1.35,
    avg_return_3m=3.8,
    avg_return_1y=19.1,
    bootstrap_ci_lo=55.2,
    bootstrap_ci_hi=57.4,

    # Sector guidance
    strong_sectors=['Consumer Cyclical', 'Industrials', 'Technology', 'Financial Services'],
    avoid_sectors=['Healthcare', 'Communication Services', 'Real Estate'],
)

BUCKET_2 = BucketConfig(
    id=2,
    name="Bear Market Dip Buy",
    short_name="BMD",
    criteria_text="Bear regime + RSI<40 + Fund≥15 + V2≥40 + Cap>$1B",

    # Position management
    position_pct=4.0,      # larger size - high conviction
    max_positions=6,
    hold_months=6,
    stop_loss_pct=-20.0,   # wider stop - volatile entries

    # Regime
    regime='bear',

    # Historical performance
    historical_win_rate=61.2,
    historical_win_rate_with_beat=70.3,
    profit_factor=2.16,
    avg_return_3m=10.0,
    avg_return_1y=25.1,
    bootstrap_ci_lo=59.3,
    bootstrap_ci_hi=63.1,

    # Sector guidance
    strong_sectors=['Technology', 'Consumer Defensive', 'Industrials', 'Consumer Cyclical', 'Utilities'],
    avoid_sectors=['Energy', 'Basic Materials'],
)

BUCKET_3 = BucketConfig(
    id=3,
    name="High-Growth Momentum",
    short_name="HGM",
    criteria_text="EPSG≥35 + EBITDAG≥33 + EV[12-27] + RSI<43",

    # Position management
    position_pct=2.5,
    max_positions=10,
    hold_months=3,         # shorter hold - momentum
    stop_loss_pct=-12.0,   # tighter stop - momentum reversal

    # Regime
    regime='all',

    # Historical performance
    historical_win_rate=60.9,
    historical_win_rate_with_beat=66.8,
    profit_factor=1.53,
    avg_return_3m=6.0,
    avg_return_1y=18.0,
    bootstrap_ci_lo=59.6,
    bootstrap_ci_hi=62.2,

    # Sector guidance
    strong_sectors=['Technology', 'Consumer Cyclical', 'Utilities', 'Financial Services'],
    avoid_sectors=['Energy', 'Healthcare'],
)

# All buckets in order
BUCKETS = [BUCKET_1, BUCKET_2, BUCKET_3]
BUCKET_MAP = {b.id: b for b in BUCKETS}
BUCKET_NAME_MAP = {b.name: b for b in BUCKETS}


# ══════════════════════════════════════════════════════════════════════════════
# FILTER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def filter_bucket_1(df: pd.DataFrame) -> pd.DataFrame:
    """Quality Growth Compounder filter."""
    return df[
        (df['value_score_v2'] >= 55) &
        (df['fundamentals_score'] >= 18) &
        (df['ev_ebitda'] >= 5) & (df['ev_ebitda'] <= 20) &
        (df['rev_growth'] > 10) &
        (df['market_cap'] > 2e9) &
        (df['analyst_count'] >= 6)
    ]

def filter_bucket_2(df: pd.DataFrame) -> pd.DataFrame:
    """Bear Market Dip Buy filter."""
    return df[
        (df['market_bearish'] == 1) &
        (df['rsi'] < 40) &
        (df['fundamentals_score'] >= 15) &
        (df['value_score_v2'] >= 40) &
        (df['market_cap'] > 1e9)
    ]

def filter_bucket_3(df: pd.DataFrame) -> pd.DataFrame:
    """High-Growth Momentum filter."""
    ev = df['ev_ebitda'].clip(-50, 200)  # ev_ebitda_clean
    return df[
        (df['eps_growth'] >= 35) &
        (df['ebitda_growth'] >= 33) &
        (ev >= 12) & (ev <= 27) &
        (df['rsi'] < 43)
    ]

# Attach filter functions
BUCKET_1.filter_fn = filter_bucket_1
BUCKET_2.filter_fn = filter_bucket_2
BUCKET_3.filter_fn = filter_bucket_3


# ══════════════════════════════════════════════════════════════════════════════
# OVERLAY FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def apply_earnings_beat_overlay(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to stocks with earnings surprise > 0%."""
    return df[df['earnings_surprise_pct'] > 0]

def apply_big_beat_overlay(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to stocks with earnings surprise > 10%."""
    return df[df['earnings_surprise_pct'] > 10]

def apply_multi_signal_overlay(df: pd.DataFrame, min_signals: int = 2) -> pd.DataFrame:
    """Filter to stocks with multiple concurrent signals."""
    return df[df['concurrent_signals'] >= min_signals]

def apply_sector_filter(df: pd.DataFrame, bucket: BucketConfig, mode: str = 'strong') -> pd.DataFrame:
    """
    Filter by sector guidance.
    mode='strong': only strong sectors
    mode='exclude_avoid': exclude avoid sectors
    """
    if mode == 'strong' and bucket.strong_sectors:
        return df[df['sector'].isin(bucket.strong_sectors)]
    elif mode == 'exclude_avoid' and bucket.avoid_sectors:
        return df[~df['sector'].isin(bucket.avoid_sectors)]
    return df


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_regime(df: pd.DataFrame, lookback: int = 100) -> str:
    """
    Detect current market regime based on recent signals.
    Returns: 'bull', 'bear', or 'neutral'
    """
    recent = df.sort_values('date_dt').tail(lookback)
    bull_pct = recent['market_bullish'].mean() * 100
    bear_pct = recent['market_bearish'].mean() * 100

    if bull_pct > 50:
        return 'bull'
    elif bear_pct > 50:
        return 'bear'
    else:
        return 'neutral'

def is_bucket_active(bucket: BucketConfig, regime: str) -> bool:
    """Check if a bucket should be active in the current regime."""
    if bucket.regime == 'all':
        return True
    return bucket.regime == regime


# ══════════════════════════════════════════════════════════════════════════════
# SCORING / RANKING
# ══════════════════════════════════════════════════════════════════════════════

def compute_conviction_score(row: pd.Series, bucket: BucketConfig) -> float:
    """
    Compute a conviction score for a signal (0-100).
    Higher = more confident.
    """
    score = 50.0  # baseline

    # Earnings beat bonus
    if pd.notna(row.get('earnings_surprise_pct')):
        if row['earnings_surprise_pct'] > 10:
            score += 20  # big beat
        elif row['earnings_surprise_pct'] > 0:
            score += 10  # beat
        elif row['earnings_surprise_pct'] < -10:
            score -= 15  # big miss

    # Multi-signal bonus
    if row.get('concurrent_signals', 1) >= 3:
        score += 10
    elif row.get('concurrent_signals', 1) >= 2:
        score += 5

    # Sector bonus/penalty
    sector = row.get('sector', '')
    if sector in bucket.strong_sectors:
        score += 10
    elif sector in bucket.avoid_sectors:
        score -= 15

    # Score quality bonus
    v2 = row.get('value_score_v2', 50)
    if v2 >= 70:
        score += 10
    elif v2 >= 60:
        score += 5

    # FCF bonus
    if row.get('fcf_positive') == 1:
        score += 5

    return min(100, max(0, score))


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def get_bucket_signals(df: pd.DataFrame, bucket: BucketConfig,
                       apply_beat_overlay: bool = False,
                       apply_sector_filter_mode: Optional[str] = None) -> pd.DataFrame:
    """
    Get signals for a bucket with optional overlays.

    Args:
        df: Full signals dataframe
        bucket: Bucket configuration
        apply_beat_overlay: If True, only include earnings beat stocks
        apply_sector_filter_mode: 'strong', 'exclude_avoid', or None

    Returns:
        Filtered dataframe
    """
    result = bucket.filter_fn(df)

    if apply_beat_overlay:
        result = apply_earnings_beat_overlay(result)

    if apply_sector_filter_mode:
        result = apply_sector_filter(result, bucket, apply_sector_filter_mode)

    return result

def get_bucket_summary(bucket: BucketConfig) -> Dict:
    """Get a summary dict for a bucket (for dashboard/API use)."""
    return {
        'id': bucket.id,
        'name': bucket.name,
        'short_name': bucket.short_name,
        'criteria': bucket.criteria_text,
        'regime': bucket.regime,
        'position_pct': bucket.position_pct,
        'max_positions': bucket.max_positions,
        'hold_months': bucket.hold_months,
        'stop_loss_pct': bucket.stop_loss_pct,
        'max_exposure_pct': bucket.position_pct * bucket.max_positions,
        'historical_win_rate': bucket.historical_win_rate,
        'historical_win_rate_with_beat': bucket.historical_win_rate_with_beat,
        'profit_factor': bucket.profit_factor,
        'avg_return_3m': bucket.avg_return_3m,
        'avg_return_1y': bucket.avg_return_1y,
        'strong_sectors': bucket.strong_sectors,
        'avoid_sectors': bucket.avoid_sectors,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN (for testing)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print("  FINALIZED BUCKET CONFIGURATION")
    print("=" * 80)

    for bucket in BUCKETS:
        print(f"\n  BUCKET {bucket.id}: {bucket.name} ({bucket.short_name})")
        print(f"  {'-'*70}")
        print(f"  Criteria: {bucket.criteria_text}")
        print(f"  Regime: {bucket.regime.upper()}")
        print(f"  Position: {bucket.position_pct}% | Max: {bucket.max_positions} | "
              f"Hold: {bucket.hold_months}mo | Stop: {bucket.stop_loss_pct}%")
        print(f"  Max Exposure: {bucket.position_pct * bucket.max_positions}%")
        print(f"  Win Rate: {bucket.historical_win_rate:.1f}% (base) → "
              f"{bucket.historical_win_rate_with_beat:.1f}% (+beat)")
        print(f"  Profit Factor: {bucket.profit_factor:.2f}")
        print(f"  Avg Return: {bucket.avg_return_3m:+.1f}% (3M), {bucket.avg_return_1y:+.1f}% (1Y)")
        print(f"  Strong Sectors: {', '.join(bucket.strong_sectors[:3])}...")
        print(f"  Avoid Sectors: {', '.join(bucket.avoid_sectors)}")

    print(f"\n{'='*80}")
    print("  Configuration ready for dashboard integration.")
    print("=" * 80)

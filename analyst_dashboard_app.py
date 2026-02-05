#!/usr/bin/env python3
# File: analyst_dashboard_app.py — 3-Tab Dashboard: Research, Analysis, Movers
import streamlit as st
import sqlite3
import pandas as pd
import os
import requests
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
try:
    from config import DATABASE_NAME, PARQUET_PATH, BACKTEST_DB
except Exception:
    _root = Path(__file__).parent
    DATABASE_NAME = str(_root / 'nasdaq_stocks.db')
    PARQUET_PATH = str(_root / 'data' / 'dashboard_data.parquet')
    BACKTEST_DB = str(_root / 'backtest.db')

# Additional paths for cloud deployment
_root = Path(__file__).parent
MOVERS_PARQUET = str(_root / 'data' / 'movers_data.parquet')
HYBRID_DB = str(_root / 'mock_portfolio.db')
GITHUB_REPO = "pcheever2-cmd/stock-research"


def download_db_from_release(db_name: str, dest_path: str) -> bool:
    """Download database from GitHub release for Streamlit Cloud deployment."""
    try:
        # GitHub release asset URL
        url = f"https://github.com/{GITHUB_REPO}/releases/download/data/{db_name}"
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(resp.content)
            return True
    except Exception:
        pass
    return False

st.set_page_config(page_title="Stock Research Dashboard", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def is_market_open():
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() <= market_close

def get_fmp_api_key():
    """Get FMP API key from Streamlit secrets or environment"""
    try:
        return st.secrets["FMP_API_KEY"]
    except Exception:
        return os.environ.get("FMP_API_KEY", "")

# ── Live Price Fetching ──────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_prices(symbols_csv: str) -> dict:
    """Fetch live prices from FMP batch-quote endpoint (cached 5 min)"""
    api_key = get_fmp_api_key()
    if not api_key:
        return {}
    prices = {}
    symbol_list = symbols_csv.split(",")
    for i in range(0, len(symbol_list), 400):
        batch = symbol_list[i:i+400]
        try:
            resp = requests.get(
                f"https://financialmodelingprep.com/stable/batch-quote",
                params={"symbols": ",".join(batch), "apikey": api_key},
                timeout=15
            )
            if resp.status_code == 200:
                for q in resp.json():
                    if isinstance(q, dict) and 'symbol' in q and 'price' in q:
                        prices[q['symbol']] = q['price']
        except Exception:
            pass
    return prices

# ── Conviction Tier Logic ────────────────────────────────────────────────────
def conviction_tier(lt, v2, fund, ev, rsi_val, rev_g, eps_g):
    """Compute conviction tier from score components"""
    lt = lt or 0
    v2 = v2 or 0
    fund = fund or 0
    rev_g = rev_g or 0
    eps_g = eps_g or 0

    # Tier 1 — Quality Compounder (strict ev_ebitda > 0)
    if (lt >= 55 and v2 >= 55 and fund >= 18 and
        ev is not None and 0 < ev <= 22 and
        rsi_val is not None and 35 <= rsi_val <= 65):
        return 'Tier 1'

    # Tier 2 — Balanced Setup (strict ev_ebitda > 0)
    if (lt >= 50 and v2 >= 45 and
        ev is not None and ev > 0 and
        (eps_g > 8 or rev_g > 15)):
        return 'Tier 2'

    # Tier 3 — Oversold Dip Buy (lenient on ev_ebitda)
    if (lt >= 40 and v2 >= 40 and
        rsi_val is not None and rsi_val < 40 and
        fund >= 15):
        return 'Tier 3'

    return None

# ── Styling Functions ────────────────────────────────────────────────────────
def color_upside(val):
    if pd.isna(val) or val == '-':
        return ''
    try:
        val_num = float(str(val).replace('%', '').replace('+', ''))
        if val_num > 30:
            return 'background-color: #c6f6d5; color: #155724'
        elif val_num > 0:
            return 'background-color: #d4edda; color: #155724'
        elif val_num < 0:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    except Exception:
        return ''

def color_trend_signal(val):
    if pd.isna(val) or val is None or val == '':
        return ''
    count = str(val).count(',') + 1
    if count >= 3:
        return 'background-color: #c6f6d5; color: #155724; font-weight: bold'
    elif count >= 2:
        return 'background-color: #d4edda; color: #155724'
    return 'background-color: #fff3cd; color: #856404'

def color_tier(val):
    if val == 'Tier 1':
        return 'background-color: #c6f6d5; color: #155724; font-weight: bold'
    elif val == 'Tier 2':
        return 'background-color: #d4edda; color: #155724'
    elif val == 'Tier 3':
        return 'background-color: #fff3cd; color: #856404'
    return ''

def color_change(val):
    """Color positive changes green, negative red"""
    if pd.isna(val):
        return ''
    if val > 0:
        intensity = min(val / 20, 1.0)
        return f'background-color: rgba(198, 246, 213, {intensity}); color: #155724; font-weight: bold'
    elif val < 0:
        intensity = min(abs(val) / 20, 1.0)
        return f'background-color: rgba(248, 215, 218, {intensity}); color: #721c24; font-weight: bold'
    return ''

def color_grade_action(val):
    """Color analyst grade actions"""
    if val == 'upgrade':
        return 'background-color: #c6f6d5; color: #155724; font-weight: bold'
    elif val == 'downgrade':
        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
    return ''

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner="Loading stock data...")
def load_data():
    """Load all stock_consensus data + compute conviction tiers.
    Returns (df, data_source_label) tuple."""
    df = None
    source = "none"

    if Path(DATABASE_NAME).exists():
        try:
            conn = sqlite3.connect(DATABASE_NAME)
            query = """
                SELECT symbol, company_name, company_description,
                       current_price, avg_price_target, median_price_target,
                       min_price_target, max_price_target,
                       upside_percent, num_analysts, recommendation,
                       consensus_rating, recent_ratings,
                       cap_category, sector, industry, last_updated,
                       enterprise_value, ebitda, ev_ebitda,
                       total_debt, debt_ebitda, ocf_ev,
                       peg_ratio, forward_pe, forward_ev_ebitda, ev_ebitda_reduction,
                       projected_revenue_growth, projected_eps_growth,
                       projected_ebitda_growth, earnings_growth,
                       projected_revenue_next_year, projected_eps_next_year,
                       sma50, sma200, rsi, adx, close_price_technical,
                       long_term_score, value_score, value_score_v2,
                       trend_score, fundamentals_score, valuation_score,
                       momentum_score, market_risk_score,
                       trend_signal, trend_signal_count
                FROM stock_consensus
                WHERE num_analysts >= 1
                ORDER BY upside_percent DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            source = "SQLite"
        except Exception as e:
            st.warning(f"SQLite not available ({e}), trying parquet...")
            df = None

    if df is None or df.empty:
        parquet = Path(PARQUET_PATH)
        if parquet.exists():
            df = pd.read_parquet(parquet)
            source = "Parquet"
        else:
            st.error("No data source found. Run the pipeline first.")
            return None, "none"

    if df.empty:
        return None, source

    # Ensure columns exist (handles older parquet files)
    for col, default in [('company_name', None), ('company_description', None),
                          ('sector', None), ('ev_ebitda', None), ('debt_ebitda', None),
                          ('ocf_ev', None), ('trend_signal', None), ('trend_signal_count', 0),
                          ('value_score_v2', None), ('rsi', None), ('median_price_target', None),
                          ('peg_ratio', None), ('forward_pe', None), ('forward_ev_ebitda', None),
                          ('ev_ebitda_reduction', None), ('projected_ebitda_growth', None),
                          ('earnings_growth', None), ('adx', None), ('sma50', None),
                          ('sma200', None), ('close_price_technical', None),
                          ('projected_revenue_next_year', None), ('projected_eps_next_year', None),
                          ('consensus_rating', None)]:
        if col not in df.columns:
            df[col] = default

    # Compute conviction tiers
    df['conviction_tier'] = df.apply(
        lambda row: conviction_tier(
            row.get('long_term_score'), row.get('value_score_v2'),
            row.get('fundamentals_score'), row.get('ev_ebitda'),
            row.get('rsi'), row.get('projected_revenue_growth'),
            row.get('projected_eps_growth')
        ), axis=1
    )

    # Derived columns
    df['upside_low'] = ((df['min_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
    df['upside_high'] = ((df['max_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
    df['min_price_target_display'] = df['min_price_target'].apply(lambda x: '-' if pd.isna(x) else f"${x:,.2f}")
    df['max_price_target_display'] = df['max_price_target'].apply(lambda x: '-' if pd.isna(x) else f"${x:,.2f}")
    df['upside_low_display'] = df['upside_low'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
    df['upside_high_display'] = df['upside_high'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='ISO8601', errors='coerce').dt.date

    # SMA status for Research tab
    df['sma_status'] = df.apply(
        lambda r: 'Above' if (pd.notna(r.get('sma50')) and pd.notna(r.get('sma200'))
                               and r['sma50'] > r['sma200']) else
                  ('Below' if pd.notna(r.get('sma50')) and pd.notna(r.get('sma200')) else '-'),
        axis=1
    )

    df = df.round({
        'current_price': 2, 'avg_price_target': 2, 'median_price_target': 2,
        'upside_percent': 1, 'long_term_score': 0, 'value_score': 0, 'value_score_v2': 0,
        'projected_revenue_growth': 1, 'projected_eps_growth': 1,
        'projected_ebitda_growth': 1, 'earnings_growth': 1,
        'ev_ebitda': 1, 'debt_ebitda': 1, 'ocf_ev': 4,
        'peg_ratio': 1, 'forward_pe': 1, 'rsi': 1,
    })
    return df, source


@st.cache_data(ttl=3600, show_spinner="Loading analyst estimates...")
def load_analyst_estimates():
    """Load analyst estimates from backtest.db"""
    if not Path(BACKTEST_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BACKTEST_DB)
    current_year = datetime.now().year
    query = """
        SELECT symbol, fiscal_year,
               revenue_low, revenue_high, revenue_avg,
               ebitda_low, ebitda_high, ebitda_avg,
               eps_low, eps_high, eps_avg,
               num_analysts_revenue, num_analysts_eps
        FROM analyst_estimates_snapshot
        WHERE fiscal_year BETWEEN ? AND ?
        ORDER BY symbol, fiscal_year
    """
    df = pd.read_sql_query(query, conn, params=[current_year, current_year + 2])
    conn.close()
    return df


@st.cache_data(ttl=3600, show_spinner="Loading price targets...")
def load_price_target_summary():
    """Load price target summary from backtest.db"""
    if not Path(BACKTEST_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BACKTEST_DB)
    df = pd.read_sql_query("SELECT * FROM price_target_summary", conn)
    conn.close()
    return df


@st.cache_data(ttl=3600, show_spinner="Loading analyst grades...")
def load_recent_grades(days=60):
    """Load recent analyst grade changes from backtest.db"""
    if not Path(BACKTEST_DB).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(BACKTEST_DB)
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = """
        SELECT symbol, date, grading_company, previous_grade, new_grade, action
        FROM historical_grades
        WHERE date >= ?
        ORDER BY date DESC
    """
    df = pd.read_sql_query(query, conn, params=[cutoff])
    conn.close()
    return df


@st.cache_data(ttl=3600, show_spinner="Computing score changes...")
def load_score_movers():
    """Load score comparison data from movers parquet or backtest_daily_scores"""
    # Try parquet first (for Streamlit Cloud deployment)
    if Path(MOVERS_PARQUET).exists():
        try:
            all_scores = pd.read_parquet(MOVERS_PARQUET)
            if not all_scores.empty and 'date' in all_scores.columns:
                dates = sorted(all_scores['date'].unique(), reverse=True)
                if len(dates) >= 1:
                    date_now = dates[0]
                    date_7d = dates[min(6, len(dates)-1)]
                    date_30d = dates[min(21, len(dates)-1)]

                    # Get data for each date
                    a = all_scores[all_scores['date'] == date_now].copy()
                    b = all_scores[all_scores['date'] == date_7d][['symbol', 'lt_score', 'value_score_v2',
                        'trend_score', 'fundamentals_score', 'rsi', 'ev_ebitda']].copy()
                    c = all_scores[all_scores['date'] == date_30d][['symbol', 'lt_score', 'value_score_v2']].copy()

                    # Rename columns
                    a = a.rename(columns={
                        'lt_score': 'lt_now', 'value_score_v2': 'v2_now',
                        'trend_score': 'trend_now', 'fundamentals_score': 'fund_now',
                        'valuation_score': 'val_now', 'momentum_score': 'mom_now',
                        'close': 'close_now', 'rsi': 'rsi_now', 'ev_ebitda': 'ev_ebitda_now'
                    })
                    b = b.rename(columns={
                        'lt_score': 'lt_7d', 'value_score_v2': 'v2_7d',
                        'trend_score': 'trend_7d', 'fundamentals_score': 'fund_7d',
                        'rsi': 'rsi_7d', 'ev_ebitda': 'ev_ebitda_7d'
                    })
                    c = c.rename(columns={'lt_score': 'lt_30d', 'value_score_v2': 'v2_30d'})

                    # Merge
                    df = a.merge(b, on='symbol', how='left').merge(c, on='symbol', how='left')

                    # Compute changes
                    df['lt_change_7d'] = df['lt_now'] - df['lt_7d']
                    df['v2_change_7d'] = df['v2_now'] - df['v2_7d']
                    df['lt_change_30d'] = df['lt_now'] - df['lt_30d']
                    df['v2_change_30d'] = df['v2_now'] - df['v2_30d']

                    return df, date_now, date_7d, date_30d
        except Exception:
            pass  # Fall through to database

    # Fall back to database (for local development)
    if not Path(BACKTEST_DB).exists():
        return pd.DataFrame(), None, None, None
    conn = sqlite3.connect(BACKTEST_DB)

    # Get recent distinct dates
    dates = conn.execute(
        "SELECT DISTINCT date FROM backtest_daily_scores ORDER BY date DESC LIMIT 30"
    ).fetchall()
    if not dates:
        conn.close()
        return pd.DataFrame(), None, None, None
    dates = [d[0] for d in dates]

    date_now = dates[0]
    date_7d = dates[min(6, len(dates)-1)]
    date_30d = dates[min(21, len(dates)-1)]

    query = """
        SELECT a.symbol,
               a.lt_score as lt_now, a.value_score_v2 as v2_now,
               a.trend_score as trend_now, a.fundamentals_score as fund_now,
               a.valuation_score as val_now, a.momentum_score as mom_now,
               a.close as close_now, a.rsi as rsi_now, a.ev_ebitda as ev_ebitda_now,
               a.rev_growth, a.eps_growth,
               b.lt_score as lt_7d, b.value_score_v2 as v2_7d,
               b.trend_score as trend_7d, b.fundamentals_score as fund_7d,
               b.rsi as rsi_7d, b.ev_ebitda as ev_ebitda_7d,
               c.lt_score as lt_30d, c.value_score_v2 as v2_30d
        FROM backtest_daily_scores a
        LEFT JOIN backtest_daily_scores b ON a.symbol = b.symbol AND b.date = ?
        LEFT JOIN backtest_daily_scores c ON a.symbol = c.symbol AND c.date = ?
        WHERE a.date = ?
    """
    df = pd.read_sql_query(query, conn, params=[date_7d, date_30d, date_now])
    conn.close()

    # Compute changes
    df['lt_change_7d'] = df['lt_now'] - df['lt_7d']
    df['v2_change_7d'] = df['v2_now'] - df['v2_7d']
    df['lt_change_30d'] = df['lt_now'] - df['lt_30d']
    df['v2_change_30d'] = df['v2_now'] - df['v2_30d']

    return df, date_now, date_7d, date_30d


# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ══════════════════════════════════════════════════════════════════════════════

result = load_data()
if result is None or result[0] is None or result[0].empty:
    st.warning("No data found. Run your update and scoring scripts first.")
    st.stop()
df, _data_source = result

# ── Live Price Overlay (runs before tabs so all tabs get updated prices) ─────
market_open = is_market_open()
if market_open:
    col_status, col_refresh = st.columns([4, 1])
    with col_status:
        st.success("Market OPEN -- Prices refresh every 5 minutes")
    with col_refresh:
        if st.button("Refresh Prices Now"):
            st.cache_data.clear()
            st.rerun()

    symbols_csv = ",".join(df['symbol'].tolist())
    live_prices = fetch_live_prices(symbols_csv)
    if live_prices:
        df['current_price'] = df.apply(
            lambda row: live_prices.get(row['symbol'], row['current_price']), axis=1
        ).round(2)
        df['upside_percent'] = ((df['avg_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
        df['upside_low'] = ((df['min_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
        df['upside_high'] = ((df['max_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
        df['upside_low_display'] = df['upside_low'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
        df['upside_high_display'] = df['upside_high'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
else:
    st.info("Market CLOSED -- Showing last known prices")

# ── Common Sidebar Filters ───────────────────────────────────────────────────
st.sidebar.header("Filters")
st.sidebar.caption("These filters apply to all tabs")

_scored = df['value_score_v2'].notna().sum()
st.sidebar.caption(f"Data: {_data_source} | {len(df):,} stocks | {_scored:,} scored")

ticker_search = st.sidebar.text_input(
    "Quick Ticker Search",
    placeholder="e.g. AAPL, VRT, ETN",
    help="Case-insensitive partial match"
)

cap_options = ['All'] + sorted(df['cap_category'].dropna().unique().tolist())
selected_cap = st.sidebar.selectbox("Market Cap Category", cap_options)

sector_options = ['All'] + sorted(df['sector'].dropna().unique().tolist())
selected_sector = st.sidebar.selectbox("Sector", sector_options)

if selected_sector != 'All':
    avail_industries = df[df['sector'] == selected_sector]['industry'].dropna().unique().tolist()
else:
    avail_industries = df['industry'].dropna().unique().tolist()
industry_options = ['All'] + sorted(avail_industries)
selected_industry = st.sidebar.selectbox("Industry", industry_options)

# Apply common filters to get base dataset
base = df.copy()
if ticker_search:
    base = base[base['symbol'].str.contains(ticker_search.strip(), case=False)]
if selected_cap != 'All':
    base = base[base['cap_category'] == selected_cap]
if selected_sector != 'All':
    base = base[base['sector'] == selected_sector]
if selected_industry != 'All':
    base = base[base['industry'] == selected_industry]

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["Research", "Analysis", "Movers", "Hybrid Portfolio"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Stock Research")
    st.markdown("**Fundamentals, analyst projections, price targets & valuation metrics**")

    # ── Tab-specific filters ─────────────────────────────────────────────────
    with st.expander("Research Filters", expanded=False):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            r_min_analysts = st.slider("Min Analysts", 0, int(df['num_analysts'].max()), 0, key="r_analysts")
            r_min_upside = st.slider("Min Upside %", -50, 300, -50, step=5, key="r_upside")
        with fc2:
            rec_options = ['All'] + sorted(df['recommendation'].dropna().unique().tolist())
            r_rec = st.selectbox("Recommendation", rec_options, key="r_rec")
            r_min_rev = st.slider("Min Rev Growth %", -50, 100, -50, step=5, key="r_rev")
        with fc3:
            r_max_ev = st.slider("Max EV/EBITDA", 0.0, 100.0, 100.0, step=1.0, key="r_ev")
            r_min_eps = st.slider("Min EPS Growth %", -50, 100, -50, step=5, key="r_eps")
        with fc4:
            r_max_debt = st.slider("Max Debt/EBITDA", 0.0, 20.0, 20.0, step=0.5, key="r_debt")
            r_max_pe = st.slider("Max Forward P/E", 0.0, 200.0, 200.0, step=5.0, key="r_pe")

    # Apply research filters
    r_filtered = base.copy()
    mask = (r_filtered['num_analysts'].fillna(0) >= r_min_analysts)
    if r_min_upside > -50:
        mask = mask & (r_filtered['upside_percent'].fillna(-999) >= r_min_upside)
    if r_rec != 'All':
        mask = mask & (r_filtered['recommendation'] == r_rec)
    if r_min_rev > -50:
        mask = mask & (r_filtered['projected_revenue_growth'].fillna(-999) >= r_min_rev)
    if r_min_eps > -50:
        mask = mask & (r_filtered['projected_eps_growth'].fillna(-999) >= r_min_eps)
    if r_max_ev < 100.0:
        mask = mask & ((r_filtered['ev_ebitda'].fillna(0) <= r_max_ev) | r_filtered['ev_ebitda'].isna())
    if r_max_debt < 20.0:
        mask = mask & ((r_filtered['debt_ebitda'].fillna(0) <= r_max_debt) | r_filtered['debt_ebitda'].isna())
    if r_max_pe < 200.0:
        mask = mask & ((r_filtered['forward_pe'].fillna(0) <= r_max_pe) | r_filtered['forward_pe'].isna())
    r_filtered = r_filtered[mask].copy()

    # ── Key Metrics ──────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Stocks", f"{len(base):,}")
    m2.metric("After Filters", f"{len(r_filtered):,}")
    med_upside = r_filtered['upside_percent'].median() if len(r_filtered) > 0 else 0
    m3.metric("Median Upside", f"{med_upside:+.1f}%")
    med_ev = r_filtered['ev_ebitda'].dropna().median() if len(r_filtered) > 0 else 0
    m4.metric("Median EV/EBITDA", f"{med_ev:.1f}x" if pd.notna(med_ev) else "N/A")
    avg_analysts = r_filtered['num_analysts'].mean() if len(r_filtered) > 0 else 0
    m5.metric("Avg Analyst Coverage", f"{avg_analysts:.0f}")

    # ── Main Research Table ──────────────────────────────────────────────────
    st.subheader(f"All Stocks ({len(r_filtered):,})")

    r_display = r_filtered[[
        'symbol', 'company_name', 'sector', 'industry', 'cap_category',
        'current_price',
        'min_price_target_display', 'avg_price_target', 'median_price_target', 'max_price_target_display',
        'upside_low_display', 'upside_percent', 'upside_high_display',
        'num_analysts', 'recommendation',
        'ev_ebitda', 'forward_pe', 'peg_ratio', 'debt_ebitda', 'ocf_ev',
        'projected_revenue_growth', 'projected_eps_growth',
        'projected_ebitda_growth', 'earnings_growth',
        'rsi', 'sma_status',
    ]].copy()

    r_display.rename(columns={
        'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector',
        'industry': 'Industry', 'cap_category': 'Cap',
        'current_price': 'Price',
        'min_price_target_display': 'Low Target', 'avg_price_target': 'Avg Target',
        'median_price_target': 'Med Target', 'max_price_target_display': 'High Target',
        'upside_low_display': 'Low Up%', 'upside_percent': 'Mean Up%',
        'upside_high_display': 'High Up%',
        'num_analysts': 'Analysts', 'recommendation': 'Rating',
        'ev_ebitda': 'EV/EBITDA', 'forward_pe': 'Fwd P/E', 'peg_ratio': 'PEG',
        'debt_ebitda': 'Debt/EBITDA', 'ocf_ev': 'OCF/EV',
        'projected_revenue_growth': 'Rev Gr%', 'projected_eps_growth': 'EPS Gr%',
        'projected_ebitda_growth': 'EBITDA Gr%', 'earnings_growth': 'Earn Gr%',
        'rsi': 'RSI', 'sma_status': 'SMA 50v200',
    }, inplace=True)

    r_format = {
        'Price': '${:.2f}', 'Avg Target': '${:.2f}', 'Med Target': '${:.2f}',
        'Analysts': '{:.0f}',
        'EV/EBITDA': '{:.1f}x', 'Fwd P/E': '{:.1f}x', 'PEG': '{:.1f}',
        'Debt/EBITDA': '{:.1f}x', 'OCF/EV': '{:.1%}',
        'Rev Gr%': '{:+.1f}%', 'EPS Gr%': '{:+.1f}%',
        'EBITDA Gr%': '{:+.1f}%', 'Earn Gr%': '{:+.1f}%',
        'RSI': '{:.0f}',
    }

    r_styled = r_display.style \
        .format(r_format, na_rep='-') \
        .map(color_upside, subset=['Low Up%', 'Mean Up%', 'High Up%']) \
        .background_gradient(subset=['EV/EBITDA'], cmap='RdYlGn_r', vmin=0, vmax=30) \
        .background_gradient(subset=['Debt/EBITDA'], cmap='RdYlGn_r', vmin=0, vmax=10) \
        .background_gradient(subset=['OCF/EV'], cmap='YlGn', vmin=0, vmax=0.15) \
        .background_gradient(subset=['Rev Gr%'], cmap='YlGn', vmin=0, vmax=50) \
        .background_gradient(subset=['EPS Gr%'], cmap='YlGn', vmin=0, vmax=50) \
        .bar(subset=['Analysts'], color='#5fba7d', vmin=0)

    st.dataframe(r_styled, use_container_width=True, height=620)

    # ── Expander Details (Top 20 by upside) ──────────────────────────────────
    if len(r_filtered) > 0:
        st.subheader("Stock Details -- Top 20 by Mean Upside")

        # Load supplementary data
        estimates_df = load_analyst_estimates()
        targets_df = load_price_target_summary()
        grades_df = load_recent_grades(days=60)

        top_20 = r_filtered.sort_values('upside_percent', ascending=False).head(20)
        for _, row in top_20.iterrows():
            sym = row['symbol']
            company_label = f" ({row['company_name']})" if pd.notna(row.get('company_name')) else ""
            with st.expander(
                f"**{sym}**{company_label}  |  {row['upside_percent']:+.1f}% upside  |  "
                f"{row['num_analysts']:.0f} analysts  |  {row['recommendation'] or 'N/A'}  |  "
                f"EV/EBITDA {row['ev_ebitda']:.1f}x" if pd.notna(row.get('ev_ebitda')) else
                f"**{sym}**{company_label}  |  {row['upside_percent']:+.1f}% upside  |  "
                f"{row['num_analysts']:.0f} analysts  |  {row['recommendation'] or 'N/A'}"
            ):
                # Company description
                desc = row.get('company_description')
                if pd.notna(desc) and desc:
                    st.markdown(f"**About:** {str(desc)[:500]}{'...' if len(str(desc)) > 500 else ''}")
                    st.markdown("---")

                # Analyst Estimates
                if not estimates_df.empty:
                    sym_est = estimates_df[estimates_df['symbol'] == sym]
                    if len(sym_est) > 0:
                        st.markdown("**Analyst Estimates (Consensus)**")
                        est_table = []
                        for _, e in sym_est.iterrows():
                            fy = int(e['fiscal_year'])
                            est_table.append({
                                'Fiscal Year': fy,
                                'Revenue Low': f"${e['revenue_low']/1e9:.2f}B" if pd.notna(e.get('revenue_low')) and e['revenue_low'] else '-',
                                'Revenue Avg': f"${e['revenue_avg']/1e9:.2f}B" if pd.notna(e.get('revenue_avg')) and e['revenue_avg'] else '-',
                                'Revenue High': f"${e['revenue_high']/1e9:.2f}B" if pd.notna(e.get('revenue_high')) and e['revenue_high'] else '-',
                                'EPS Low': f"${e['eps_low']:.2f}" if pd.notna(e.get('eps_low')) else '-',
                                'EPS Avg': f"${e['eps_avg']:.2f}" if pd.notna(e.get('eps_avg')) else '-',
                                'EPS High': f"${e['eps_high']:.2f}" if pd.notna(e.get('eps_high')) else '-',
                                'Analysts (EPS)': int(e['num_analysts_eps']) if pd.notna(e.get('num_analysts_eps')) else '-',
                            })
                        if est_table:
                            st.dataframe(pd.DataFrame(est_table), use_container_width=True, hide_index=True)

                # Price Target Breakdown
                if not targets_df.empty:
                    sym_pt = targets_df[targets_df['symbol'] == sym]
                    if len(sym_pt) > 0:
                        pt = sym_pt.iloc[0]
                        st.markdown("**Price Target Breakdown**")
                        pt_data = []
                        for period, avg_col, cnt_col in [
                            ('Last Month', 'last_month_avg', 'last_month_count'),
                            ('Last Quarter', 'last_quarter_avg', 'last_quarter_count'),
                            ('Last Year', 'last_year_avg', 'last_year_count'),
                            ('All Time', 'all_time_avg', 'all_time_count'),
                        ]:
                            avg_val = pt.get(avg_col)
                            cnt_val = pt.get(cnt_col)
                            pt_data.append({
                                'Period': period,
                                'Avg Target': f"${avg_val:,.2f}" if pd.notna(avg_val) and avg_val else '-',
                                'Count': int(cnt_val) if pd.notna(cnt_val) and cnt_val else 0,
                            })
                        st.dataframe(pd.DataFrame(pt_data), use_container_width=True, hide_index=True)

                # Recent Grade Changes
                if not grades_df.empty:
                    sym_grades = grades_df[grades_df['symbol'] == sym].head(10)
                    if len(sym_grades) > 0:
                        st.markdown("**Recent Analyst Grade Changes**")
                        g_display = sym_grades[['date', 'grading_company', 'previous_grade', 'new_grade', 'action']].copy()
                        g_display.columns = ['Date', 'Firm', 'Previous', 'New', 'Action']
                        styled_grades = g_display.style.map(color_grade_action, subset=['Action'])
                        st.dataframe(styled_grades, use_container_width=True, hide_index=True)

                # Recent ratings (fallback)
                if row.get('recent_ratings') and "No recent" not in str(row.get('recent_ratings', '')):
                    if grades_df.empty or len(grades_df[grades_df['symbol'] == sym]) == 0:
                        st.markdown("**Recent Rating Changes**")
                        st.markdown(str(row['recent_ratings']).replace('\n', '  \n'))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Scoring & Conviction Tiers")
    st.markdown("**Our algorithmic analysis -- backtested over 5 years on 2,847 stocks**")

    # ── Tier Summary Cards ───────────────────────────────────────────────────
    t1_count = (base['conviction_tier'] == 'Tier 1').sum()
    t2_count = (base['conviction_tier'] == 'Tier 2').sum()
    t3_count = (base['conviction_tier'] == 'Tier 3').sum()

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("""
        <div style="background-color: #c6f6d5; padding: 12px; border-radius: 8px; border-left: 4px solid #155724;">
        <strong style="color: #155724;">Tier 1: Quality Compounder</strong><br>
        <small>LT>=55, V2>=55, Fund>=18, EV/EBITDA 0-22, RSI 35-65</small><br>
        <small>3M Win: 57% | 1Y Avg: +29%</small><br>
        <strong>""" + str(t1_count) + """ stocks qualify</strong>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown("""
        <div style="background-color: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745;">
        <strong style="color: #155724;">Tier 2: Balanced Setup</strong><br>
        <small>LT>=50, V2>=45, EV/EBITDA>0, Growth present</small><br>
        <small>3M Win: 56% | 1Y Avg: +22%</small><br>
        <strong>""" + str(t2_count) + """ stocks qualify</strong>
        </div>
        """, unsafe_allow_html=True)
    with tc3:
        st.markdown("""
        <div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #856404;">
        <strong style="color: #856404;">Tier 3: Oversold Dip Buy</strong><br>
        <small>LT>=40, V2>=40, RSI<40, Fund>=15</small><br>
        <small>3M Win: 62% | 1Y Avg: +23%</small><br>
        <strong>""" + str(t3_count) + """ stocks qualify</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tab-specific filters ─────────────────────────────────────────────────
    with st.expander("Analysis Filters", expanded=False):
        af1, af2, af3, af4 = st.columns(4)
        with af1:
            a_tier = st.multiselect("Conviction Tier", ['Tier 1', 'Tier 2', 'Tier 3', 'No Tier'],
                                     default=['Tier 1', 'Tier 2', 'Tier 3', 'No Tier'], key="a_tier")
        with af2:
            a_min_lt = st.slider("Min LT Score", 0, 100, 0, key="a_lt")
            a_min_v2 = st.slider("Min V2 Score", 0, 100, 0, key="a_v2")
        with af3:
            a_min_fund = st.slider("Min Fundamentals (/25)", 0, 25, 0, key="a_fund")
            a_trending = st.checkbox("Show Trending Only", key="a_trend")
        with af4:
            a_sort = st.selectbox("Sort By", ["V2 Score", "LT Score", "Tier then V2", "Mean Upside %"],
                                   key="a_sort")

    # Apply analysis filters
    a_filtered = base.copy()
    tier_mask = pd.Series(False, index=a_filtered.index)
    if 'Tier 1' in a_tier:
        tier_mask = tier_mask | (a_filtered['conviction_tier'] == 'Tier 1')
    if 'Tier 2' in a_tier:
        tier_mask = tier_mask | (a_filtered['conviction_tier'] == 'Tier 2')
    if 'Tier 3' in a_tier:
        tier_mask = tier_mask | (a_filtered['conviction_tier'] == 'Tier 3')
    if 'No Tier' in a_tier:
        tier_mask = tier_mask | a_filtered['conviction_tier'].isna()
    a_filtered = a_filtered[tier_mask].copy()

    mask = (
        (a_filtered['long_term_score'].fillna(0) >= a_min_lt) &
        (a_filtered['value_score_v2'].fillna(0) >= a_min_v2) &
        (a_filtered['fundamentals_score'].fillna(0) >= a_min_fund)
    )
    if a_trending:
        mask = mask & (a_filtered['trend_signal_count'].fillna(0) > 0)
    a_filtered = a_filtered[mask].copy()

    # Sort
    if a_sort == "V2 Score":
        a_filtered = a_filtered.sort_values('value_score_v2', ascending=False)
    elif a_sort == "LT Score":
        a_filtered = a_filtered.sort_values('long_term_score', ascending=False)
    elif a_sort == "Tier then V2":
        tier_order = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
        a_filtered['_tier_sort'] = a_filtered['conviction_tier'].map(tier_order).fillna(9)
        a_filtered = a_filtered.sort_values(['_tier_sort', 'value_score_v2'], ascending=[True, False])
        a_filtered.drop(columns='_tier_sort', inplace=True)
    else:
        a_filtered = a_filtered.sort_values('upside_percent', ascending=False)

    # ── Key Metrics ──────────────────────────────────────────────────────────
    am1, am2, am3, am4, am5 = st.columns(5)
    am1.metric("Tier 1", t1_count)
    am2.metric("Tier 2", t2_count)
    am3.metric("Tier 3", t3_count)
    avg_v2 = a_filtered['value_score_v2'].mean() if len(a_filtered) > 0 else 0
    am4.metric("Avg V2 Score", f"{avg_v2:.0f}/100" if pd.notna(avg_v2) else "N/A")
    trending_count = (a_filtered['trend_signal_count'].fillna(0) > 0).sum()
    am5.metric("With Trend Signals", trending_count)

    # ── Main Analysis Table ──────────────────────────────────────────────────
    st.subheader(f"Scored Stocks ({len(a_filtered):,})")

    a_display = a_filtered[[
        'symbol', 'company_name', 'sector', 'cap_category', 'conviction_tier',
        'value_score_v2', 'long_term_score',
        'trend_score', 'fundamentals_score', 'valuation_score',
        'momentum_score', 'market_risk_score',
        'current_price', 'upside_percent',
        'ev_ebitda', 'rsi',
        'projected_revenue_growth', 'projected_eps_growth',
        'trend_signal',
    ]].copy()

    a_display.rename(columns={
        'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector',
        'cap_category': 'Cap', 'conviction_tier': 'Tier',
        'value_score_v2': 'V2 Score', 'long_term_score': 'LT Score',
        'trend_score': 'Trend (/25)', 'fundamentals_score': 'Fund (/25)',
        'valuation_score': 'Val (/16)', 'momentum_score': 'Mom (/10)',
        'market_risk_score': 'Mkt (/10)',
        'current_price': 'Price', 'upside_percent': 'Mean Up%',
        'ev_ebitda': 'EV/EBITDA', 'rsi': 'RSI',
        'projected_revenue_growth': 'Rev Gr%', 'projected_eps_growth': 'EPS Gr%',
        'trend_signal': 'Trend Signals',
    }, inplace=True)

    a_format = {
        'V2 Score': '{:.0f}', 'LT Score': '{:.0f}',
        'Trend (/25)': '{:.0f}', 'Fund (/25)': '{:.0f}', 'Val (/16)': '{:.0f}',
        'Mom (/10)': '{:.0f}', 'Mkt (/10)': '{:.0f}',
        'Price': '${:.2f}', 'Mean Up%': '{:+.1f}%',
        'EV/EBITDA': '{:.1f}x', 'RSI': '{:.0f}',
        'Rev Gr%': '{:+.1f}%', 'EPS Gr%': '{:+.1f}%',
    }

    a_styled = a_display.style \
        .format(a_format, na_rep='-') \
        .map(color_tier, subset=['Tier']) \
        .map(color_trend_signal, subset=['Trend Signals']) \
        .map(color_upside, subset=['Mean Up%']) \
        .background_gradient(subset=['V2 Score'], cmap='Blues', vmin=0, vmax=100) \
        .background_gradient(subset=['LT Score'], cmap='Oranges', vmin=0, vmax=100) \
        .background_gradient(subset=['Rev Gr%'], cmap='YlGn', vmin=0, vmax=50) \
        .background_gradient(subset=['EPS Gr%'], cmap='YlGn', vmin=0, vmax=50) \
        .background_gradient(subset=['EV/EBITDA'], cmap='RdYlGn_r', vmin=0, vmax=30)

    st.dataframe(a_styled, use_container_width=True, height=620)

    # ── Score Breakdown Expanders (Top 15 by V2) ────────────────────────────
    if len(a_filtered) > 0:
        st.subheader("Score Breakdown -- Top 15 by V2 Score")
        top_15 = a_filtered.sort_values('value_score_v2', ascending=False).head(15)
        for _, row in top_15.iterrows():
            sym = row['symbol']
            company_label = f" ({row['company_name']})" if pd.notna(row.get('company_name')) else ""
            tier_label = f"  |  **{row['conviction_tier']}**" if pd.notna(row.get('conviction_tier')) else ""
            trend_label = f"  |  Trends: {row['trend_signal']}" if pd.notna(row.get('trend_signal')) and row.get('trend_signal') else ""
            v2_val = row['value_score_v2'] if pd.notna(row.get('value_score_v2')) else 0
            with st.expander(
                f"**{sym}**{company_label}  |  V2 **{v2_val:.0f}**  |  LT **{row['long_term_score']:.0f}**  |  "
                f"{row['upside_percent']:+.1f}%{tier_label}{trend_label}"
            ):
                # Tier qualification checklist
                lt = row.get('long_term_score', 0) or 0
                v2 = row.get('value_score_v2', 0) or 0
                fund = row.get('fundamentals_score', 0) or 0
                ev = row.get('ev_ebitda')
                rsi_v = row.get('rsi')
                rev_g = row.get('projected_revenue_growth', 0) or 0
                eps_g = row.get('projected_eps_growth', 0) or 0

                checks = []
                checks.append(f"{'[x]' if lt >= 55 else '[ ]'} LT Score >= 55 ({lt:.0f})")
                checks.append(f"{'[x]' if v2 >= 55 else '[ ]'} V2 Score >= 55 ({v2:.0f})")
                checks.append(f"{'[x]' if fund >= 18 else '[ ]'} Fundamentals >= 18 ({fund:.0f})")
                ev_check = pd.notna(ev) and 0 < ev <= 22
                checks.append(f"{'[x]' if ev_check else '[ ]'} EV/EBITDA 0-22 ({ev:.1f}x)" if pd.notna(ev) else "[ ] EV/EBITDA 0-22 (N/A)")
                rsi_check = pd.notna(rsi_v) and 35 <= rsi_v <= 65
                checks.append(f"{'[x]' if rsi_check else '[ ]'} RSI 35-65 ({rsi_v:.0f})" if pd.notna(rsi_v) else "[ ] RSI 35-65 (N/A)")
                growth_check = eps_g > 8 or rev_g > 15
                checks.append(f"{'[x]' if growth_check else '[ ]'} Growth (EPS>{eps_g:.1f}% or Rev>{rev_g:.1f}%)")

                st.markdown("**Tier 1 Qualification Checklist:**")
                st.text('\n'.join(checks))

                st.markdown("**Score Components:**")
                sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                sc1.metric("Trend", f"{row.get('trend_score', 0):.0f}/25")
                sc2.metric("Fundamentals", f"{fund:.0f}/25")
                sc3.metric("Valuation", f"{row.get('valuation_score', 0):.0f}/16")
                sc4.metric("Momentum", f"{row.get('momentum_score', 0):.0f}/10")
                sc5.metric("Mkt/Risk", f"{row.get('market_risk_score', 0):.0f}/10")

                desc = row.get('company_description')
                if pd.notna(desc) and desc:
                    st.markdown(f"**About:** {str(desc)[:500]}{'...' if len(str(desc)) > 500 else ''}")

    # ── Methodology (collapsed) ──────────────────────────────────────────────
    with st.expander("Scoring Methodology & Details"):
        st.markdown("""
### Value Score V2 /100 -- Continuous Fundamentals Scoring

**Backtested over 5 years on 2,847 stocks.** V2 replaces the old binary tier system with continuous
scoring. Higher V2 = monotonically higher win rate (47% at V2=0 to 59% at V2=60-69).

| Component | Max | Formula |
|-----------|-----|---------|
| **Valuation** | 40 | EV/EBITDA <8: +40, <12: +30, <16: +20, <22: +10. Negative: -10 |
| **Revenue Growth** | 25 | `min(rev_growth, 50) / 2`, capped at 25. >60% penalized 0.7x |
| **EPS Growth** | 20 | `eps_growth / 2`, range -5 to +20 |
| **Quality** | 15 | EBITDA growth >10%: +10. Profitable (EV/EBITDA 0-25): +5 |

### Long-Term Score /100

| Category | Max | Criteria |
|----------|-----|----------|
| **Trend** | 25 | Price > 200 SMA (+10), 50 SMA > 200 SMA (+10), Price > 50 SMA (+5) |
| **Fundamentals** | 25 | Rev Growth >15% (+15) / >8% (+8), EPS Growth >15% (+10) / >8% (+5) |
| **Valuation** | 16 | EV/EBITDA <12 (+10) / <20 (+6) / <30 (+3) |
| **Momentum** | 10 | RSI 40-55 (+5), ADX > 20 (+5) |
| **Market Regime** | 10 | Bull market (SPY > 200 SMA) -> +10 |

### Valuation Metrics

- **EV/EBITDA**: Enterprise Value / EBITDA. Lower = cheaper. <10x attractive, >30x expensive.
- **Debt/EBITDA**: Net Debt / EBITDA. Lower = less leveraged. <2x healthy, >5x concerning.
- **OCF/EV**: Operating Cash Flow / Enterprise Value. Higher = better cash yield. >10% strong.

### Trend Signals

- **Golden Cross**: SMA50 crosses above SMA200
- **Price > SMA50**: Price breaks above 50-day moving average
- **RSI Recovery**: RSI recovers from oversold (<30) to neutral (>40)
- **Bullish Aligned**: Price > SMA50 > SMA200 (uptrend confirmed)
        """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MOVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.title("Movers & Category Changes")
    st.markdown("**Stocks jumping categories and improving buying position in our algorithm**")

    # Refresh button for movers data
    if st.button("🔄 Refresh Movers Data", key="refresh_movers"):
        load_score_movers.clear()
        st.rerun()

    # ── Load movers data ─────────────────────────────────────────────────────
    score_movers, date_now, date_7d, date_30d = load_score_movers()
    grades_all = load_recent_grades(days=30)

    if score_movers.empty:
        st.warning("No backtest data available for movers detection. Run the backtest first.")
    else:
        # ── Filters ──────────────────────────────────────────────────────────
        with st.expander("Movers Filters", expanded=False):
            mf1, mf2, mf3 = st.columns(3)
            with mf1:
                m_lookback = st.radio("Lookback", ["7 Days", "30 Days"], key="m_look", horizontal=True)
            with mf2:
                m_direction = st.radio("Direction", ["Improving Only", "All Changes", "Declining Only"],
                                        key="m_dir", horizontal=True)
            with mf3:
                m_min_change = st.slider("Min Score Change", 0, 50, 5, key="m_change")

        # Apply common sidebar filters to score_movers
        sm = score_movers.copy()
        if ticker_search:
            sm = sm[sm['symbol'].str.contains(ticker_search.strip(), case=False)]

        # Join company info from main df
        info_cols = df[['symbol', 'company_name', 'sector', 'cap_category', 'conviction_tier',
                        'recommendation', 'upside_percent']].copy()
        sm = sm.merge(info_cols, on='symbol', how='left')

        if selected_cap != 'All':
            sm = sm[sm['cap_category'] == selected_cap]
        if selected_sector != 'All':
            sm = sm[sm['sector'] == selected_sector]
        if selected_industry != 'All':
            # Industry comes from main df
            industry_syms = df[df['industry'] == selected_industry]['symbol'].tolist()
            sm = sm[sm['symbol'].isin(industry_syms)]

        # Select change columns based on lookback
        if m_lookback == "7 Days":
            lt_col, v2_col = 'lt_change_7d', 'v2_change_7d'
            lt_prev, v2_prev = 'lt_7d', 'v2_7d'
        else:
            lt_col, v2_col = 'lt_change_30d', 'v2_change_30d'
            lt_prev, v2_prev = 'lt_30d', 'v2_30d'

        # ── Compute tier transitions ─────────────────────────────────────────
        sm['tier_now'] = sm.apply(
            lambda r: conviction_tier(
                r.get('lt_now'), r.get('v2_now'), r.get('fund_now'),
                r.get('ev_ebitda_now'), r.get('rsi_now'),
                r.get('rev_growth'), r.get('eps_growth')
            ), axis=1
        )
        sm['tier_prev'] = sm.apply(
            lambda r: conviction_tier(
                r.get(lt_prev), r.get(v2_prev), r.get('fund_7d' if m_lookback == "7 Days" else None),
                r.get('ev_ebitda_7d' if m_lookback == "7 Days" else None),
                r.get('rsi_7d' if m_lookback == "7 Days" else None),
                r.get('rev_growth'), r.get('eps_growth')
            ), axis=1
        )

        # ── Key Metrics ──────────────────────────────────────────────────────
        score_up = sm[(sm[lt_col].fillna(0) >= m_min_change) | (sm[v2_col].fillna(0) >= m_min_change)]
        score_down = sm[(sm[lt_col].fillna(0) <= -m_min_change) | (sm[v2_col].fillna(0) <= -m_min_change)]

        new_tiers = sm[(sm['tier_prev'].isna()) & (sm['tier_now'].notna())]
        tier_upgrades = sm[
            ((sm['tier_prev'] == 'Tier 3') & (sm['tier_now'].isin(['Tier 1', 'Tier 2']))) |
            ((sm['tier_prev'] == 'Tier 2') & (sm['tier_now'] == 'Tier 1'))
        ]

        upgrade_count = 0
        downgrade_count = 0
        if not grades_all.empty:
            upgrade_count = (grades_all['action'] == 'upgrade').sum()
            downgrade_count = (grades_all['action'] == 'downgrade').sum()

        mm1, mm2, mm3, mm4, mm5 = st.columns(5)
        mm1.metric("Score Improvers", len(score_up))
        mm2.metric("Score Decliners", len(score_down))
        mm3.metric("New Tier Entries", len(new_tiers))
        mm4.metric("Analyst Upgrades (30d)", upgrade_count)
        mm5.metric("Analyst Downgrades (30d)", downgrade_count)

        st.caption(f"Backtest data: {date_now} vs {date_7d} (7d) / {date_30d} (30d)")

        # ── SECTION A: Score Movers ──────────────────────────────────────────
        st.subheader("Score Movers")

        # Filter by direction
        if m_direction == "Improving Only":
            s_movers = sm[(sm[lt_col].fillna(0) >= m_min_change) | (sm[v2_col].fillna(0) >= m_min_change)]
            s_movers = s_movers.sort_values(lt_col, ascending=False)
        elif m_direction == "Declining Only":
            s_movers = sm[(sm[lt_col].fillna(0) <= -m_min_change) | (sm[v2_col].fillna(0) <= -m_min_change)]
            s_movers = s_movers.sort_values(lt_col, ascending=True)
        else:
            s_movers = sm[
                (sm[lt_col].fillna(0).abs() >= m_min_change) |
                (sm[v2_col].fillna(0).abs() >= m_min_change)
            ]
            s_movers = s_movers.sort_values(lt_col, ascending=False)

        if len(s_movers) > 0:
            sm_display = s_movers[[
                'symbol', 'company_name', 'sector', 'cap_category',
                'lt_now', lt_prev, lt_col,
                'v2_now', v2_prev, v2_col,
                'conviction_tier', 'rsi_now',
            ]].copy()

            sm_display.rename(columns={
                'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector',
                'cap_category': 'Cap',
                'lt_now': 'LT Now', lt_prev: 'LT Prev', lt_col: 'LT Change',
                'v2_now': 'V2 Now', v2_prev: 'V2 Prev', v2_col: 'V2 Change',
                'conviction_tier': 'Tier', 'rsi_now': 'RSI',
            }, inplace=True)

            sm_format = {
                'LT Now': '{:.0f}', 'LT Prev': '{:.0f}', 'LT Change': '{:+.0f}',
                'V2 Now': '{:.0f}', 'V2 Prev': '{:.0f}', 'V2 Change': '{:+.0f}',
                'RSI': '{:.0f}',
            }

            sm_styled = sm_display.style \
                .format(sm_format, na_rep='-') \
                .map(color_change, subset=['LT Change', 'V2 Change']) \
                .map(color_tier, subset=['Tier']) \
                .background_gradient(subset=['LT Now'], cmap='Oranges', vmin=0, vmax=100) \
                .background_gradient(subset=['V2 Now'], cmap='Blues', vmin=0, vmax=100)

            st.dataframe(sm_styled, use_container_width=True, height=400)
        else:
            st.info(f"No score movers with >= {m_min_change} point change in the selected period.")

        # ── SECTION B: Tier Transitions ──────────────────────────────────────
        st.subheader("Tier Transitions")

        tier_changed = sm[sm['tier_now'] != sm['tier_prev']].copy()
        # Include None -> Tier and Tier -> None
        tier_changed = sm[
            (sm['tier_now'] != sm['tier_prev']) |
            (sm['tier_now'].notna() & sm['tier_prev'].isna()) |
            (sm['tier_now'].isna() & sm['tier_prev'].notna())
        ].copy()
        # Deduplicate
        tier_changed = tier_changed.drop_duplicates(subset='symbol')

        if m_direction == "Improving Only":
            # Keep entries and upgrades only
            tier_order = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
            tier_changed['_now_rank'] = tier_changed['tier_now'].map(tier_order).fillna(0)
            tier_changed['_prev_rank'] = tier_changed['tier_prev'].map(tier_order).fillna(0)
            tier_changed = tier_changed[tier_changed['_now_rank'] > tier_changed['_prev_rank']]
            tier_changed.drop(columns=['_now_rank', '_prev_rank'], inplace=True, errors='ignore')
        elif m_direction == "Declining Only":
            tier_order = {'Tier 1': 3, 'Tier 2': 2, 'Tier 3': 1}
            tier_changed['_now_rank'] = tier_changed['tier_now'].map(tier_order).fillna(0)
            tier_changed['_prev_rank'] = tier_changed['tier_prev'].map(tier_order).fillna(0)
            tier_changed = tier_changed[tier_changed['_now_rank'] < tier_changed['_prev_rank']]
            tier_changed.drop(columns=['_now_rank', '_prev_rank'], inplace=True, errors='ignore')

        if len(tier_changed) > 0:
            tier_changed['transition'] = tier_changed.apply(
                lambda r: f"{r['tier_prev'] or 'None'} -> {r['tier_now'] or 'None'}", axis=1
            )

            tt_display = tier_changed[[
                'symbol', 'company_name', 'sector',
                'tier_prev', 'tier_now', 'transition',
                'lt_now', 'v2_now', 'rsi_now', 'ev_ebitda_now',
            ]].copy()

            tt_display.rename(columns={
                'symbol': 'Symbol', 'company_name': 'Company', 'sector': 'Sector',
                'tier_prev': 'Previous Tier', 'tier_now': 'Current Tier',
                'transition': 'Transition',
                'lt_now': 'LT Score', 'v2_now': 'V2 Score',
                'rsi_now': 'RSI', 'ev_ebitda_now': 'EV/EBITDA',
            }, inplace=True)

            tt_format = {
                'LT Score': '{:.0f}', 'V2 Score': '{:.0f}',
                'RSI': '{:.0f}', 'EV/EBITDA': '{:.1f}x',
            }

            tt_styled = tt_display.style \
                .format(tt_format, na_rep='-') \
                .map(color_tier, subset=['Previous Tier', 'Current Tier'])

            st.dataframe(tt_styled, use_container_width=True, height=400)
        else:
            st.info("No tier transitions detected in the selected period.")

        # ── SECTION C: Analyst Grade Activity ────────────────────────────────
        st.subheader("Analyst Grade Activity (Last 30 Days)")

        if not grades_all.empty:
            # Aggregate per symbol
            grade_summary = grades_all.groupby('symbol').agg(
                total=('action', 'count'),
                upgrades=('action', lambda x: (x == 'upgrade').sum()),
                downgrades=('action', lambda x: (x == 'downgrade').sum()),
                latest_date=('date', 'max'),
            ).reset_index()
            grade_summary['net'] = grade_summary['upgrades'] - grade_summary['downgrades']

            # Get latest action per symbol
            latest = grades_all.sort_values('date', ascending=False).drop_duplicates('symbol')
            latest = latest[['symbol', 'grading_company', 'action', 'previous_grade', 'new_grade']].rename(
                columns={'grading_company': 'latest_firm', 'action': 'latest_action',
                         'previous_grade': 'latest_prev', 'new_grade': 'latest_new'}
            )
            grade_summary = grade_summary.merge(latest, on='symbol', how='left')

            # Join company info
            grade_summary = grade_summary.merge(
                info_cols[['symbol', 'company_name']], on='symbol', how='left'
            )

            # Apply sidebar filters
            if ticker_search:
                grade_summary = grade_summary[grade_summary['symbol'].str.contains(ticker_search.strip(), case=False)]
            if selected_sector != 'All':
                sector_syms = df[df['sector'] == selected_sector]['symbol'].tolist()
                grade_summary = grade_summary[grade_summary['symbol'].isin(sector_syms)]

            # Apply direction filter
            if m_direction == "Improving Only":
                grade_summary = grade_summary[grade_summary['net'] > 0]
            elif m_direction == "Declining Only":
                grade_summary = grade_summary[grade_summary['net'] < 0]

            grade_summary = grade_summary.sort_values('net', ascending=False)

            if len(grade_summary) > 0:
                g_display = grade_summary[[
                    'symbol', 'company_name', 'upgrades', 'downgrades', 'net',
                    'latest_action', 'latest_firm', 'latest_date',
                    'latest_prev', 'latest_new',
                ]].copy()

                g_display.rename(columns={
                    'symbol': 'Symbol', 'company_name': 'Company',
                    'upgrades': 'Upgrades', 'downgrades': 'Downgrades', 'net': 'Net',
                    'latest_action': 'Latest Action', 'latest_firm': 'Latest Firm',
                    'latest_date': 'Date',
                    'latest_prev': 'From', 'latest_new': 'To',
                }, inplace=True)

                g_styled = g_display.style \
                    .map(color_change, subset=['Net']) \
                    .map(color_grade_action, subset=['Latest Action'])

                st.dataframe(g_styled, use_container_width=True, height=400)
            else:
                st.info("No analyst grade activity matching your filters.")
        else:
            st.info("No analyst grade data available. Run `collect_analyst_data.py` first.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: HYBRID PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.title("Hybrid Portfolio Tracker")
    st.markdown("**55% Mega-Cap Core | 25% Options Alpha | 20% Growth Picks**")

    # Live refresh function
    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def fetch_live_prices(symbols: list) -> dict:
        """Fetch live prices from FMP API."""
        import os
        api_key = os.environ.get('FMP_API_KEY', '')
        if not api_key:
            return {}
        try:
            symbols_str = ','.join(symbols)
            url = f"https://financialmodelingprep.com/stable/batch-quote?symbols={symbols_str}&apikey={api_key}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    return {item['symbol']: item['price'] for item in data if 'price' in item}
        except Exception:
            pass
        return {}

    # Selection criteria explanation
    with st.expander("📋 Selection Criteria (click to expand)", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            **Bucket 1: Quality Growth Compounder**
            - Value Score V2 >= 55
            - Fundamentals Score >= 18
            - EV/EBITDA between 5-20
            - Revenue Growth > 10%
            - Market Cap > $2B
            - Analyst Coverage >= 6
            """)
        with col_b:
            st.markdown("""
            **Bucket 3: High-Growth Momentum**
            - EPS Growth >= 35%
            - EBITDA Growth >= 33%
            - EV/EBITDA between 12-27
            - RSI < 43 (oversold)

            **Options requirement:** EPS Growth > 15%
            """)

    # Always download hybrid portfolio database from GitHub release on Streamlit Cloud
    # to ensure we have the latest data (it's small and changes frequently)
    if _data_source == 'GitHub Release':
        with st.spinner("Downloading portfolio data from GitHub..."):
            download_db_from_release("mock_portfolio.db", HYBRID_DB)
    elif not Path(HYBRID_DB).exists():
        with st.spinner("Downloading portfolio data from GitHub..."):
            download_db_from_release("mock_portfolio.db", HYBRID_DB)

    if Path(HYBRID_DB).exists() and Path(HYBRID_DB).stat().st_size > 0:
        h_conn = sqlite3.connect(HYBRID_DB)

        # Check if tables exist
        tables = h_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='hybrid_positions'"
        ).fetchall()

        if tables:
            # Check which columns exist
            cols_info = pd.read_sql_query("PRAGMA table_info(hybrid_positions)", h_conn)
            existing_cols = cols_info['name'].values if not cols_info.empty else []
            has_strike = 'strike_price' in existing_cols
            has_expiration = 'expiration_date' in existing_cols
            has_contracts = 'contracts' in existing_cols
            has_premium = 'premium' in existing_cols

            # Build SELECT based on available columns
            base_cols = "symbol, position_type, entry_date, entry_price, cost_basis, current_price, current_value, pnl, pnl_pct, status, notes, updated_at"
            if has_strike:
                base_cols = base_cols.replace("entry_price,", "entry_price, strike_price,")
            if has_expiration:
                base_cols = base_cols.replace("cost_basis,", "expiration_date, cost_basis,")
            if has_contracts:
                base_cols = base_cols.replace("cost_basis,", "contracts, cost_basis,")
            if has_premium:
                base_cols = base_cols.replace("contracts,", "contracts, premium,")

            positions = pd.read_sql_query(f'''
                SELECT {base_cols}
                FROM hybrid_positions WHERE status = 'open'
                ORDER BY position_type, symbol
            ''', h_conn)

            if not has_strike:
                positions['strike_price'] = None
            if not has_expiration:
                positions['expiration_date'] = None
            if not has_contracts:
                positions['contracts'] = None
            if not has_premium:
                positions['premium'] = None

            # Load metrics from backtest.db or movers parquet for transparency
            metrics_df = pd.DataFrame()
            if Path(MOVERS_PARQUET).exists():
                try:
                    all_scores = pd.read_parquet(MOVERS_PARQUET)
                    if not all_scores.empty and 'date' in all_scores.columns:
                        latest_date = all_scores['date'].max()
                        metrics_df = all_scores[all_scores['date'] == latest_date][[
                            'symbol', 'lt_score', 'value_score_v2', 'fundamentals_score',
                            'ev_ebitda', 'eps_growth', 'ebitda_growth', 'rev_growth', 'rsi'
                        ]].copy()
                except Exception:
                    pass
            elif Path(BACKTEST_DB).exists():
                try:
                    b_conn = sqlite3.connect(BACKTEST_DB)
                    latest = b_conn.execute('SELECT MAX(date) FROM backtest_daily_scores').fetchone()[0]
                    if latest:
                        metrics_df = pd.read_sql_query(f'''
                            SELECT symbol, lt_score, value_score_v2, fundamentals_score,
                                   ev_ebitda, eps_growth, ebitda_growth, rev_growth, rsi
                            FROM backtest_daily_scores WHERE date = '{latest}'
                        ''', b_conn)
                    b_conn.close()
                except Exception:
                    pass

            if not positions.empty:
                # Summary metrics
                total_cost = positions['cost_basis'].sum()
                total_value = positions['current_value'].sum() if positions['current_value'].notna().any() else total_cost
                total_pnl = total_value - total_cost
                total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

                # Live refresh button
                refresh_col1, refresh_col2 = st.columns([1, 4])
                with refresh_col1:
                    if st.button("🔄 Refresh Prices", help="Fetch live prices (requires FMP API key)"):
                        symbols = positions['symbol'].unique().tolist() + ['SPY']
                        live_prices = fetch_live_prices(symbols)
                        if live_prices:
                            st.session_state['live_prices'] = live_prices
                            st.session_state['live_refresh_time'] = pd.Timestamp.now()
                            st.rerun()
                        else:
                            st.warning("Could not fetch live prices. API key may not be set.")

                # Use live prices if available, otherwise use stored prices
                live_prices = st.session_state.get('live_prices', {})
                if live_prices:
                    # Recalculate values with live prices
                    for idx, row in positions.iterrows():
                        entry_price = row.get('entry_price') if 'entry_price' in row.index else None
                        if row['symbol'] in live_prices and entry_price and pd.notna(entry_price) and entry_price > 0:
                            new_price = live_prices[row['symbol']]
                            shares = row['cost_basis'] / entry_price
                            positions.at[idx, 'current_price'] = new_price
                            positions.at[idx, 'current_value'] = shares * new_price
                            positions.at[idx, 'pnl'] = (shares * new_price) - row['cost_basis']
                            positions.at[idx, 'pnl_pct'] = (positions.at[idx, 'pnl'] / row['cost_basis']) * 100
                    total_value = positions['current_value'].sum()
                    total_pnl = total_value - total_cost
                    total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

                # Show summary cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Value", f"${total_value:,.0f}", f"{total_pnl_pct:+.1f}%")
                with col2:
                    mega_val = positions[positions['position_type'] == 'mega_cap']['current_value'].sum()
                    mega_pnl_pct = ((mega_val / (total_cost * 0.55)) - 1) * 100 if total_cost else 0
                    st.metric("Mega-Cap (55%)", f"${mega_val:,.0f}", f"{mega_pnl_pct:+.1f}%")
                with col3:
                    opt_val = positions[positions['position_type'] == 'options']['current_value'].sum()
                    opt_pnl_pct = ((opt_val / (total_cost * 0.25)) - 1) * 100 if total_cost else 0
                    st.metric("Options (25%)", f"${opt_val:,.0f}", f"{opt_pnl_pct:+.1f}%")
                with col4:
                    growth_val = positions[positions['position_type'] == 'growth']['current_value'].sum()
                    growth_pnl_pct = ((growth_val / (total_cost * 0.20)) - 1) * 100 if total_cost else 0
                    st.metric("Growth (20%)", f"${growth_val:,.0f}", f"{growth_pnl_pct:+.1f}%")

                # S&P 500 Benchmark Comparison
                spy_entry = None
                spy_current = live_prices.get('SPY')

                try:
                    # Check if spy_entry_price column exists
                    cols = pd.read_sql_query("PRAGMA table_info(hybrid_daily_snapshot)", h_conn)
                    has_spy_entry = 'spy_entry_price' in cols['name'].values if not cols.empty else False

                    if has_spy_entry:
                        spy_snapshot = pd.read_sql_query('''
                            SELECT spy_entry_price, spy_value FROM hybrid_daily_snapshot
                            WHERE spy_entry_price IS NOT NULL
                            ORDER BY date DESC LIMIT 1
                        ''', h_conn)
                        if not spy_snapshot.empty:
                            spy_entry = spy_snapshot['spy_entry_price'].iloc[0]
                            if not spy_current:
                                spy_current = spy_snapshot['spy_value'].iloc[0]
                    else:
                        # Fallback: try to get just spy_value
                        spy_snapshot = pd.read_sql_query('''
                            SELECT spy_value FROM hybrid_daily_snapshot
                            WHERE spy_value IS NOT NULL
                            ORDER BY date DESC LIMIT 1
                        ''', h_conn)
                        if not spy_snapshot.empty and not spy_current:
                            spy_current = spy_snapshot['spy_value'].iloc[0]
                except Exception:
                    pass  # SPY data not available

                if spy_entry and spy_current:
                    spy_return_pct = ((spy_current - spy_entry) / spy_entry) * 100
                    spy_equiv_value = total_cost * (1 + spy_return_pct / 100)
                    alpha = total_pnl_pct - spy_return_pct

                    st.markdown("---")
                    st.subheader("📈 S&P 500 Benchmark Comparison")
                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    with bcol1:
                        st.metric("SPY Entry", f"${spy_entry:,.2f}")
                    with bcol2:
                        st.metric("SPY Current", f"${spy_current:,.2f}", f"{spy_return_pct:+.2f}%")
                    with bcol3:
                        st.metric("$100K in SPY", f"${spy_equiv_value:,.0f}")
                    with bcol4:
                        alpha_color = "normal" if alpha >= 0 else "inverse"
                        st.metric("Alpha", f"{alpha:+.2f}%", delta_color=alpha_color)

                    if 'live_refresh_time' in st.session_state:
                        st.caption(f"Live prices as of: {st.session_state['live_refresh_time'].strftime('%Y-%m-%d %H:%M:%S')}")

                st.markdown("---")

                # Helper to determine which bucket a stock qualifies for
                def get_bucket_reason(row):
                    reasons = []
                    # Check Bucket 1 criteria
                    b1 = (row.get('value_score_v2', 0) >= 55 and
                          row.get('fundamentals_score', 0) >= 18 and
                          5 <= (row.get('ev_ebitda') or 0) <= 20 and
                          (row.get('rev_growth') or 0) > 10)
                    # Check Bucket 3 criteria
                    b3 = ((row.get('eps_growth') or 0) >= 35 and
                          (row.get('ebitda_growth') or 0) >= 33 and
                          12 <= (row.get('ev_ebitda') or 0) <= 27 and
                          (row.get('rsi') or 100) < 43)
                    if b1:
                        reasons.append('B1')
                    if b3:
                        reasons.append('B3')
                    return ', '.join(reasons) if reasons else '-'

                # Positions by type with metrics
                for pos_type, emoji, title, description in [
                    ('mega_cap', '📊', 'Mega-Cap Core', 'Top 10 mega-caps by market cap (buy & hold)'),
                    ('options', '🎯', 'Options Alpha', 'OTM5 calls on Bucket 1/3 stocks with EPS Growth > 15%'),
                    ('growth', '🌱', 'Growth Picks', 'Long-term holds from Bucket 1/3 signals')
                ]:
                    type_pos = positions[positions['position_type'] == pos_type]
                    if not type_pos.empty:
                        st.subheader(f"{emoji} {title}")
                        st.caption(description)

                        if pos_type == 'mega_cap':
                            # Mega-caps are hardcoded, just show basic info
                            display_df = type_pos[['symbol', 'cost_basis', 'current_price', 'pnl', 'pnl_pct', 'notes']].copy()
                            display_df['Weight'] = display_df['notes'].apply(
                                lambda x: x.split(' - ')[1] if ' - ' in str(x) else ''
                            )
                            display_df = display_df[['symbol', 'Weight', 'cost_basis', 'current_price', 'pnl', 'pnl_pct']]
                            display_df.columns = ['Symbol', 'Weight', 'Cost', 'Price', 'P&L', 'P&L %']
                        else:
                            # Options and Growth - show why they were selected
                            base_cols = ['symbol', 'cost_basis', 'current_price', 'pnl', 'pnl_pct']
                            if pos_type == 'options':
                                if 'strike_price' in type_pos.columns:
                                    base_cols.insert(2, 'strike_price')  # Add strike after symbol
                                if 'expiration_date' in type_pos.columns:
                                    base_cols.insert(3 if 'strike_price' in type_pos.columns else 2, 'expiration_date')
                            display_df = type_pos[base_cols].copy()

                            # Merge metrics if available
                            if not metrics_df.empty:
                                display_df = display_df.merge(
                                    metrics_df[['symbol', 'eps_growth', 'ebitda_growth', 'ev_ebitda', 'rsi', 'value_score_v2', 'fundamentals_score', 'rev_growth']],
                                    on='symbol', how='left'
                                )
                                display_df['Bucket'] = display_df.apply(get_bucket_reason, axis=1)

                                if pos_type == 'options':
                                    # Include contracts, strike price and expiration for options
                                    has_strike = 'strike_price' in display_df.columns
                                    has_exp = 'expiration_date' in display_df.columns
                                    has_contracts = 'contracts' in type_pos.columns and type_pos['contracts'].notna().any()

                                    if has_contracts and has_strike and has_exp:
                                        # Add contracts from type_pos
                                        display_df = display_df.merge(type_pos[['symbol', 'contracts']], on='symbol', how='left')
                                        display_df = display_df[['symbol', 'contracts', 'strike_price', 'expiration_date', 'Bucket', 'cost_basis', 'pnl_pct']]
                                        display_df.columns = ['Symbol', 'Contracts', 'Strike $', 'Expires', 'Bucket', 'Cost', 'P&L %']
                                    elif has_strike and has_exp:
                                        display_df = display_df[['symbol', 'current_price', 'strike_price', 'expiration_date', 'Bucket', 'eps_growth', 'cost_basis', 'pnl_pct']]
                                        display_df.columns = ['Symbol', 'Stock $', 'Strike $', 'Expires', 'Bucket', 'EPS Gr%', 'Cost', 'P&L %']
                                    elif has_strike:
                                        display_df = display_df[['symbol', 'current_price', 'strike_price', 'Bucket', 'eps_growth', 'cost_basis', 'pnl_pct']]
                                        display_df.columns = ['Symbol', 'Stock $', 'Strike $', 'Bucket', 'EPS Gr%', 'Cost', 'P&L %']
                                    else:
                                        display_df = display_df[['symbol', 'Bucket', 'eps_growth', 'ebitda_growth', 'ev_ebitda', 'rsi', 'cost_basis', 'pnl_pct']]
                                        display_df.columns = ['Symbol', 'Bucket', 'EPS Gr%', 'EBITDA Gr%', 'EV/EBITDA', 'RSI', 'Cost', 'P&L %']
                                else:  # growth
                                    display_df = display_df[['symbol', 'Bucket', 'value_score_v2', 'fundamentals_score', 'rev_growth', 'ev_ebitda', 'cost_basis', 'pnl_pct']]
                                    display_df.columns = ['Symbol', 'Bucket', 'V2 Score', 'Fund Score', 'Rev Gr%', 'EV/EBITDA', 'Cost', 'P&L %']
                            else:
                                display_df.columns = ['Symbol', 'Cost', 'Price', 'P&L', 'P&L %']

                        def color_pnl(val):
                            if pd.isna(val):
                                return ''
                            return 'color: green' if val > 0 else 'color: red' if val < 0 else ''

                        def color_bucket(val):
                            if val == 'B1':
                                return 'background-color: #e6f3ff'
                            elif val == 'B3':
                                return 'background-color: #fff3e6'
                            elif val == 'B1, B3':
                                return 'background-color: #e6ffe6'
                            return ''

                        # Build format dict based on columns present
                        fmt = {}
                        if 'Cost' in display_df.columns:
                            fmt['Cost'] = '${:,.0f}'
                        if 'Price' in display_df.columns:
                            fmt['Price'] = lambda x: f'${x:.2f}' if pd.notna(x) else 'N/A'
                        if 'Stock $' in display_df.columns:
                            fmt['Stock $'] = lambda x: f'${x:.2f}' if pd.notna(x) else 'N/A'
                        if 'Strike $' in display_df.columns:
                            fmt['Strike $'] = lambda x: f'${x:.2f}' if pd.notna(x) else 'N/A'
                        if 'P&L' in display_df.columns:
                            fmt['P&L'] = lambda x: f'${x:+,.0f}' if pd.notna(x) else '-'
                        if 'P&L %' in display_df.columns:
                            fmt['P&L %'] = lambda x: f'{x:+.1f}%' if pd.notna(x) else '-'
                        if 'EPS Gr%' in display_df.columns:
                            fmt['EPS Gr%'] = lambda x: f'{x:.0f}%' if pd.notna(x) else '-'
                        if 'EBITDA Gr%' in display_df.columns:
                            fmt['EBITDA Gr%'] = lambda x: f'{x:.0f}%' if pd.notna(x) else '-'
                        if 'Rev Gr%' in display_df.columns:
                            fmt['Rev Gr%'] = lambda x: f'{x:.0f}%' if pd.notna(x) else '-'
                        if 'EV/EBITDA' in display_df.columns:
                            fmt['EV/EBITDA'] = lambda x: f'{x:.1f}' if pd.notna(x) else '-'
                        if 'RSI' in display_df.columns:
                            fmt['RSI'] = lambda x: f'{x:.0f}' if pd.notna(x) else '-'
                        if 'V2 Score' in display_df.columns:
                            fmt['V2 Score'] = lambda x: f'{x:.0f}' if pd.notna(x) else '-'
                        if 'Fund Score' in display_df.columns:
                            fmt['Fund Score'] = lambda x: f'{x:.0f}' if pd.notna(x) else '-'

                        styled = display_df.style.format(fmt, na_rep='-')
                        if 'P&L %' in display_df.columns:
                            styled = styled.map(color_pnl, subset=['P&L %'])
                        if 'P&L' in display_df.columns:
                            styled = styled.map(color_pnl, subset=['P&L'])
                        if 'Bucket' in display_df.columns:
                            styled = styled.map(color_bucket, subset=['Bucket'])

                        st.dataframe(styled, use_container_width=True, hide_index=True)

                # Daily snapshot chart with SPY comparison
                try:
                    # Check which columns exist
                    cols = pd.read_sql_query("PRAGMA table_info(hybrid_daily_snapshot)", h_conn)
                    available_cols = cols['name'].tolist() if not cols.empty else []

                    # Build query based on available columns
                    select_cols = ['date', 'total_value', 'cumulative_pnl']
                    if 'spy_value' in available_cols:
                        select_cols.append('spy_value')
                    if 'spy_entry_price' in available_cols:
                        select_cols.append('spy_entry_price')

                    snapshots = pd.read_sql_query(f'''
                        SELECT {', '.join(select_cols)}
                        FROM hybrid_daily_snapshot
                        ORDER BY date
                    ''', h_conn)
                except Exception:
                    snapshots = pd.DataFrame()

                if len(snapshots) >= 1:
                    st.subheader("📊 Performance Over Time")

                    # Calculate portfolio and SPY returns as percentages
                    has_spy_entry = 'spy_entry_price' in snapshots.columns and snapshots['spy_entry_price'].notna().any()
                    has_spy_value = 'spy_value' in snapshots.columns and snapshots['spy_value'].notna().any()

                    if has_spy_entry and has_spy_value:
                        spy_entry_val = snapshots['spy_entry_price'].dropna().iloc[0]
                        snapshots['Portfolio %'] = ((snapshots['total_value'] / total_cost) - 1) * 100
                        snapshots['SPY %'] = ((snapshots['spy_value'] / spy_entry_val) - 1) * 100

                        chart_data = snapshots.set_index('date')[['Portfolio %', 'SPY %']]
                        st.line_chart(chart_data)
                    else:
                        st.line_chart(snapshots.set_index('date')['total_value'])

                # Last update time
                last_update = positions['updated_at'].max()
                st.caption(f"Last updated: {last_update}")
            else:
                st.info("No positions in hybrid portfolio yet. Run `setup_hybrid_portfolio.py` to initialize.")
        else:
            st.warning("Hybrid portfolio tables not found. Run `setup_hybrid_portfolio.py` to create them.")

        h_conn.close()
    else:
        st.warning("Hybrid portfolio database not found. Run `setup_hybrid_portfolio.py` to create it.")

    # Instructions
    with st.expander("How to use this portfolio"):
        st.markdown("""
        **Initial Setup:**
        1. Run `python setup_hybrid_portfolio.py` to create the portfolio

        **Daily Updates:**
        2. Run `python update_hybrid_portfolio.py` to fetch prices and track P&L
        3. Or click **🔄 Refresh Prices** above for live updates in the dashboard

        **Live Refresh Requirements:**
        - Set `FMP_API_KEY` environment variable for live price updates
        - On Streamlit Cloud, add it to your app secrets

        **Portfolio Structure:**
        - **Mega-Cap Core (55%)**: AAPL, MSFT, NVDA, GOOG, AMZN, META, TSLA, BRK-B, AVGO, LLY
        - **Options Alpha (25%)**: OTM5 calls on Bucket 1/3 signals with earnings beat
        - **Growth Picks (20%)**: Long-term holds from quality signals

        **Based on backtest showing:**
        - +370% cumulative return vs SPY +83% (2021-2025)
        - 36.3% CAGR vs SPY 12.8%
        """)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Data from Financial Modeling Prep | 6,600+ NYSE/NASDAQ/AMEX stocks | "
           "Pipeline runs daily via GitHub Actions | "
           "Live prices during market hours")

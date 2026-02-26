#!/usr/bin/env python3
# File: analyst_dashboard_app.py — Compass Score Dashboard
import streamlit as st
import sqlite3
import pandas as pd
import os
import requests
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

# ══════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION (Optional - disable by setting ENABLE_AUTH=False)
# ══════════════════════════════════════════════════════════════════════════════

ENABLE_AUTH = True  # Set to False to disable authentication

def check_auth():
    """Simple password authentication."""
    if not ENABLE_AUTH:
        return True

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Try to get password from secrets (Streamlit Cloud) or environment
    correct_password = None
    try:
        correct_password = st.secrets["APP_PASSWORD"]
    except (KeyError, FileNotFoundError):
        correct_password = os.environ.get("APP_PASSWORD", "")

    if not correct_password:
        # No password configured - allow access (for development)
        return True

    # Show login form
    st.title("🧭 Compass Score Dashboard")
    st.markdown("Please enter the password to access the dashboard.")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Login"):
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")

    st.stop()
    return False

# ══════════════════════════════════════════════════════════════════════════════

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

st.set_page_config(page_title="Compass Score Dashboard", layout="wide")

# Check authentication before loading the rest of the app
check_auth()

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
    # TEMPORARILY DISABLED - causing page freezes
    # Remove this return statement to re-enable live prices
    return {}

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
                       trend_signal, trend_signal_count,
                       compass_score, compass_grade
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
                          ('consensus_rating', None),
                          ('compass_score', None), ('compass_grade', None)]:
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
tab1, tab2, tab3 = st.tabs(["Compass", "Research", "Movers"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: COMPASS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.title("Compass Score")
    st.markdown("**Quality-focused stock scoring based on profitability, cash flows, and stability**")

    # ── Grade Summary Cards ───────────────────────────────────────────────────
    scored_stocks = base[base['compass_score'].notna()].copy()
    a_count = (scored_stocks['compass_grade'] == 'A').sum()
    b_count = (scored_stocks['compass_grade'] == 'B').sum()
    c_count = (scored_stocks['compass_grade'] == 'C').sum()
    d_count = (scored_stocks['compass_grade'] == 'D').sum()
    f_count = (scored_stocks['compass_grade'] == 'F').sum()

    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    with gc1:
        st.markdown(f"""
        <div style="background-color: #c6f6d5; padding: 12px; border-radius: 8px; border-left: 4px solid #155724; text-align: center;">
        <strong style="color: #155724; font-size: 24px;">A</strong><br>
        <small style="color: #155724;">High Quality (85-100)</small><br>
        <strong style="font-size: 20px;">{a_count:,}</strong>
        </div>
        """, unsafe_allow_html=True)
    with gc2:
        st.markdown(f"""
        <div style="background-color: #d4edda; padding: 12px; border-radius: 8px; border-left: 4px solid #28a745; text-align: center;">
        <strong style="color: #155724; font-size: 24px;">B</strong><br>
        <small style="color: #155724;">Above Average (60-84)</small><br>
        <strong style="font-size: 20px;">{b_count:,}</strong>
        </div>
        """, unsafe_allow_html=True)
    with gc3:
        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 12px; border-radius: 8px; border-left: 4px solid #856404; text-align: center;">
        <strong style="color: #856404; font-size: 24px;">C</strong><br>
        <small style="color: #856404;">Neutral (40-59)</small><br>
        <strong style="font-size: 20px;">{c_count:,}</strong>
        </div>
        """, unsafe_allow_html=True)
    with gc4:
        st.markdown(f"""
        <div style="background-color: #ffe5d0; padding: 12px; border-radius: 8px; border-left: 4px solid #d35400; text-align: center;">
        <strong style="color: #d35400; font-size: 24px;">D</strong><br>
        <small style="color: #d35400;">Speculative (20-39)</small><br>
        <strong style="font-size: 20px;">{d_count:,}</strong>
        </div>
        """, unsafe_allow_html=True)
    with gc5:
        st.markdown(f"""
        <div style="background-color: #f8d7da; padding: 12px; border-radius: 8px; border-left: 4px solid #721c24; text-align: center;">
        <strong style="color: #721c24; font-size: 24px;">F</strong><br>
        <small style="color: #721c24;">High Risk (0-19)</small><br>
        <strong style="font-size: 20px;">{f_count:,}</strong>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Tab-specific filters ─────────────────────────────────────────────────
    with st.expander("Compass Filters", expanded=True):
        cf1, cf2, cf3, cf4 = st.columns(4)
        with cf1:
            c_grades = st.multiselect("Grade", ['A', 'B', 'C', 'D', 'F'],
                                       default=['A', 'B'], key="c_grades")
        with cf2:
            c_min_score = st.slider("Min Score", 0, 100, 0, key="c_min_score")
        with cf3:
            c_sort = st.selectbox("Sort By", ["Compass Score", "Company Name", "Sector"],
                                   key="c_sort")
        with cf4:
            c_show_unscored = st.checkbox("Show Unscored", value=False, key="c_show_unscored")

    # Apply compass filters
    if c_show_unscored:
        c_filtered = base.copy()
    else:
        c_filtered = scored_stocks.copy()
        grade_mask = c_filtered['compass_grade'].isin(c_grades)
        c_filtered = c_filtered[grade_mask].copy()
        c_filtered = c_filtered[c_filtered['compass_score'] >= c_min_score]

    # Sort
    if c_sort == "Compass Score":
        c_filtered = c_filtered.sort_values('compass_score', ascending=False)
    elif c_sort == "Company Name":
        c_filtered = c_filtered.sort_values('company_name', ascending=True)
    else:
        c_filtered = c_filtered.sort_values(['sector', 'compass_score'], ascending=[True, False])

    # ── Key Metrics ──────────────────────────────────────────────────────────
    cm1, cm2, cm3, cm4 = st.columns(4)
    cm1.metric("Total Scored", f"{len(scored_stocks):,}")
    cm2.metric("After Filters", f"{len(c_filtered):,}")
    avg_score = c_filtered['compass_score'].mean() if len(c_filtered) > 0 else 0
    cm3.metric("Avg Score", f"{avg_score:.0f}/100" if pd.notna(avg_score) else "N/A")
    cm4.metric("A-Grade Stocks", f"{a_count:,}")

    # ── Main Compass Table ────────────────────────────────────────────────────
    st.subheader(f"Stocks by Compass Score ({len(c_filtered):,})")

    c_display = c_filtered[[
        'symbol', 'company_name', 'compass_score', 'compass_grade',
        'sector', 'cap_category', 'current_price',
        'ev_ebitda', 'debt_ebitda', 'projected_revenue_growth', 'projected_eps_growth',
    ]].copy()

    c_display.rename(columns={
        'symbol': 'Symbol', 'company_name': 'Company',
        'compass_score': 'Score', 'compass_grade': 'Grade',
        'sector': 'Sector', 'cap_category': 'Cap',
        'current_price': 'Price',
        'ev_ebitda': 'EV/EBITDA', 'debt_ebitda': 'Debt/EBITDA',
        'projected_revenue_growth': 'Rev Gr%', 'projected_eps_growth': 'EPS Gr%',
    }, inplace=True)

    c_format = {
        'Score': '{:.0f}',
        'Price': '${:.2f}',
        'EV/EBITDA': '{:.1f}x', 'Debt/EBITDA': '{:.1f}x',
        'Rev Gr%': '{:+.1f}%', 'EPS Gr%': '{:+.1f}%',
    }

    def color_grade(val):
        if val == 'A':
            return 'background-color: #c6f6d5; color: #155724; font-weight: bold;'
        elif val == 'B':
            return 'background-color: #d4edda; color: #155724;'
        elif val == 'C':
            return 'background-color: #fff3cd; color: #856404;'
        elif val == 'D':
            return 'background-color: #ffe5d0; color: #d35400;'
        elif val == 'F':
            return 'background-color: #f8d7da; color: #721c24;'
        return ''

    c_styled = c_display.style \
        .format(c_format, na_rep='-') \
        .map(color_grade, subset=['Grade']) \
        .background_gradient(subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100)

    st.dataframe(c_styled, use_container_width=True, height=620)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.title("Stock Research")
    st.markdown("**Analyst coverage, price targets, and fundamental metrics**")

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
    avg_compass = a_filtered['compass_score'].mean() if len(a_filtered) > 0 else 0
    am4.metric("Avg Compass Score", f"{avg_compass:.0f}/100" if pd.notna(avg_compass) else "N/A")
    trending_count = (a_filtered['trend_signal_count'].fillna(0) > 0).sum()
    am5.metric("With Trend Signals", trending_count)

    # ── Main Analysis Table ──────────────────────────────────────────────────
    st.subheader(f"Scored Stocks ({len(a_filtered):,})")

    a_display = a_filtered[[
        'symbol', 'company_name', 'compass_score', 'compass_grade',
        'sector', 'cap_category', 'conviction_tier',
        'value_score_v2', 'long_term_score',
        'trend_score', 'fundamentals_score', 'valuation_score',
        'momentum_score', 'market_risk_score',
        'current_price', 'upside_percent',
        'ev_ebitda', 'rsi',
        'projected_revenue_growth', 'projected_eps_growth',
        'trend_signal',
    ]].copy()

    a_display.rename(columns={
        'symbol': 'Symbol', 'company_name': 'Company',
        'compass_score': 'Compass', 'compass_grade': 'Grade',
        'sector': 'Sector', 'cap_category': 'Cap', 'conviction_tier': 'Tier',
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
        'Compass': '{:.0f}', 'V2 Score': '{:.0f}', 'LT Score': '{:.0f}',
        'Trend (/25)': '{:.0f}', 'Fund (/25)': '{:.0f}', 'Val (/16)': '{:.0f}',
        'Mom (/10)': '{:.0f}', 'Mkt (/10)': '{:.0f}',
        'Price': '${:.2f}', 'Mean Up%': '{:+.1f}%',
        'EV/EBITDA': '{:.1f}x', 'RSI': '{:.0f}',
        'Rev Gr%': '{:+.1f}%', 'EPS Gr%': '{:+.1f}%',
    }

    def color_compass_grade(val):
        """Color code Compass Grade A-F."""
        colors = {'A': '#1a7431', 'B': '#2e8b57', 'C': '#b8860b', 'D': '#cd853f', 'F': '#b22222'}
        return f'background-color: {colors.get(val, "white")}; color: white; font-weight: bold'

    a_styled = a_display.style \
        .format(a_format, na_rep='-') \
        .map(color_tier, subset=['Tier']) \
        .map(color_compass_grade, subset=['Grade']) \
        .map(color_trend_signal, subset=['Trend Signals']) \
        .map(color_upside, subset=['Mean Up%']) \
        .background_gradient(subset=['Compass'], cmap='RdYlGn', vmin=0, vmax=100) \
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


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Compass Score Dashboard | Quality-focused stock analysis | "
           "Data from Financial Modeling Prep")


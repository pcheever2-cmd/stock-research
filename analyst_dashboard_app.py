#!/usr/bin/env python3
# File: analyst_dashboard_app.py — Full dashboard with company names, sector filter,
# EV/EBITDA, Debt/EBITDA, OCF/EV, live prices, and trend signals
import streamlit as st
import sqlite3
import pandas as pd
import os
import requests
from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from config import DATABASE_NAME, PARQUET_PATH

st.set_page_config(page_title="Analyst Undervalued Stocks Dashboard", layout="wide")

# ── Market Hours Detection ─────────────────────────────────────────────────
def is_market_open():
    """Check if US stock market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)"""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:  # Saturday=5, Sunday=6
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

# ── Live Price Fetching ────────────────────────────────────────────────────
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

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Top Undervalued Opportunities", "Scoring Methodology", "Long-Term Score Details"])

with tab1:
    st.title("Analyst Consensus Undervalued Stocks Dashboard")
    st.markdown("**Analyst price targets, upside potential, growth estimates, scoring & trend signals**  \n"
                "Data from Financial Modeling Prep | Updated daily via GitHub Actions")

    @st.cache_data(ttl=3600, show_spinner="Loading stock data...")
    def load_data():
        df = None

        # Try SQLite first (local development)
        if Path(DATABASE_NAME).exists():
            try:
                conn = sqlite3.connect(DATABASE_NAME)
                query = """
                    SELECT symbol, company_name, current_price, avg_price_target,
                           min_price_target, max_price_target,
                           upside_percent, num_analysts, recommendation,
                           cap_category, sector, industry, recent_ratings, last_updated,
                           long_term_score, value_score,
                           trend_score, fundamentals_score, valuation_score,
                           momentum_score, market_risk_score,
                           projected_revenue_growth, projected_eps_growth,
                           ev_ebitda, debt_ebitda, ocf_ev,
                           trend_signal, trend_signal_count
                    FROM stock_consensus
                    WHERE num_analysts >= 1
                    ORDER BY upside_percent DESC
                """
                df = pd.read_sql_query(query, conn)
                conn.close()
            except Exception as e:
                st.warning(f"SQLite not available ({e}), trying parquet...")
                df = None

        # Fallback to parquet (Streamlit Cloud)
        if df is None or df.empty:
            parquet = Path(PARQUET_PATH)
            if parquet.exists():
                df = pd.read_parquet(parquet)
            else:
                st.error("No data source found. Run the pipeline first.")
                return None

        if df.empty:
            return None

        # Ensure new columns exist (handles older parquet files missing them)
        for col, default in [('company_name', None), ('sector', None), ('ev_ebitda', None),
                              ('debt_ebitda', None), ('ocf_ev', None),
                              ('trend_signal', None), ('trend_signal_count', 0)]:
            if col not in df.columns:
                df[col] = default

        # Calculate low/high upsides
        df['upside_low'] = ((df['min_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
        df['upside_high'] = ((df['max_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)

        # Display formatting
        df['min_price_target_display'] = df['min_price_target'].apply(lambda x: '-' if pd.isna(x) else f"${x:,.2f}")
        df['max_price_target_display'] = df['max_price_target'].apply(lambda x: '-' if pd.isna(x) else f"${x:,.2f}")
        df['upside_low_display'] = df['upside_low'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
        df['upside_high_display'] = df['upside_high'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")

        df['last_updated'] = pd.to_datetime(df['last_updated']).dt.date

        # Rounding
        df = df.round({
            'current_price': 2,
            'avg_price_target': 2,
            'upside_percent': 1,
            'long_term_score': 0,
            'value_score': 0,
            'projected_revenue_growth': 1,
            'projected_eps_growth': 1,
            'ev_ebitda': 1,
            'debt_ebitda': 1,
            'ocf_ev': 4,
        })
        return df

    df = load_data()
    if df is None or df.empty:
        st.warning("No data found. Run your update and scoring scripts first.")
        st.stop()

    # ── Live Price Overlay ─────────────────────────────────────────────────
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
            # Recalculate upsides with live prices
            df['upside_percent'] = ((df['avg_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
            df['upside_low'] = ((df['min_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
            df['upside_high'] = ((df['max_price_target'] - df['current_price']) / df['current_price'] * 100).round(1)
            df['upside_low_display'] = df['upside_low'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
            df['upside_high_display'] = df['upside_high'].apply(lambda x: '-' if pd.isna(x) else f"{x:+.1f}%")
    else:
        st.info("Market CLOSED -- Showing last known prices")

    # ── Sidebar Filters ───────────────────────────────────────────────────
    st.sidebar.header("Filters")

    cap_options = ['All'] + sorted(df['cap_category'].dropna().unique().tolist())
    selected_cap = st.sidebar.selectbox("Market Cap Category", cap_options)

    # Sector filter (broad ~11 categories) replaces old industry filter
    sector_options = ['All'] + sorted(df['sector'].dropna().unique().tolist())
    selected_sector = st.sidebar.selectbox("Sector", sector_options)

    # Industry filter (narrower, filtered by selected sector)
    if selected_sector != 'All':
        avail_industries = df[df['sector'] == selected_sector]['industry'].dropna().unique().tolist()
    else:
        avail_industries = df['industry'].dropna().unique().tolist()
    industry_options = ['All'] + sorted(avail_industries)
    selected_industry = st.sidebar.selectbox("Industry", industry_options)

    rec_options = ['All'] + sorted(df['recommendation'].dropna().unique().tolist())
    selected_rec = st.sidebar.selectbox("Consensus Recommendation", rec_options)

    col1s, col2s = st.sidebar.columns(2)
    with col1s:
        min_analysts = st.slider("Min Analysts", 1, int(df['num_analysts'].max()), 5)
        min_upside = st.slider("Min Mean Upside %", -50, 300, 30, step=5)
    with col2s:
        min_lt = st.slider("Min LT Score", 0, 100, 40)
        min_value = st.slider("Min Value Score", 0, 100, 40)

    # Growth filters
    min_rev_growth = st.sidebar.slider("Min Projected Revenue Growth %", 0, 100, 10, step=5)
    min_eps_growth = st.sidebar.slider("Min Projected EPS Growth %", 0, 100, 10, step=5)

    # Debt filter
    max_debt_ebitda = st.sidebar.slider("Max Debt/EBITDA", 0.0, 20.0, 10.0, step=0.5,
                                         help="Filter out highly leveraged companies")

    # Trend filter
    show_trending_only = st.sidebar.checkbox("Show Trending Stocks Only",
                                              help="Only show stocks with active trend signals")

    # Quick Ticker Search
    ticker_search = st.sidebar.text_input(
        "Quick Ticker Search",
        placeholder="e.g. AAPL, VRT, ETN, GEV",
        help="Type part of a ticker symbol (case-insensitive partial match)"
    )

    # Apply filters
    filtered = df.copy()

    # 1. Apply ticker search first (fastest to narrow down)
    if ticker_search:
        filtered = filtered[filtered['symbol'].str.contains(ticker_search.strip(), case=False)]

    # 2. Then apply other filters
    if selected_cap != 'All':
        filtered = filtered[filtered['cap_category'] == selected_cap]
    if selected_sector != 'All':
        filtered = filtered[filtered['sector'] == selected_sector]
    if selected_industry != 'All':
        filtered = filtered[filtered['industry'] == selected_industry]
    if selected_rec != 'All':
        filtered = filtered[filtered['recommendation'] == selected_rec]

    mask = (
        (filtered['num_analysts'] >= min_analysts) &
        (filtered['upside_percent'].fillna(-999) >= min_upside) &
        (filtered['long_term_score'].fillna(0) >= min_lt) &
        (filtered['value_score'].fillna(0) >= min_value)
    )
    if min_rev_growth > 0:
        mask = mask & (filtered['projected_revenue_growth'].fillna(-999) >= min_rev_growth)
    if min_eps_growth > 0:
        mask = mask & (filtered['projected_eps_growth'].fillna(-999) >= min_eps_growth)
    if max_debt_ebitda < 20.0:
        mask = mask & ((filtered['debt_ebitda'].fillna(0) <= max_debt_ebitda) | filtered['debt_ebitda'].isna())
    if show_trending_only:
        mask = mask & (filtered['trend_signal_count'].fillna(0) > 0)
    filtered = filtered[mask].copy()

    # ── Key Metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Covered Stocks", len(df))
    col2.metric("Stocks After Filters", len(filtered))
    col3.metric("Highest Mean Upside", f"{df['upside_percent'].max():+.1f}%")
    col4.metric("Highest Value Score", f"{df['value_score'].max():.0f}/100")

    # ── Main Table ────────────────────────────────────────────────────────
    st.subheader(f"Top Undervalued Opportunities ({len(filtered)} stocks)")

    display_cols = filtered[[
        'symbol', 'company_name', 'sector', 'cap_category', 'current_price',
        'min_price_target_display', 'avg_price_target', 'max_price_target_display',
        'upside_low_display', 'upside_percent', 'upside_high_display',
        'num_analysts', 'recommendation',
        'ev_ebitda', 'debt_ebitda', 'ocf_ev',
        'value_score',
        'projected_revenue_growth',
        'projected_eps_growth',
        'long_term_score',
        'trend_score', 'fundamentals_score', 'valuation_score',
        'momentum_score', 'market_risk_score',
        'trend_signal',
    ]].copy()

    display_cols.rename(columns={
        'symbol': 'Symbol',
        'company_name': 'Company',
        'sector': 'Sector',
        'cap_category': 'Cap Category',
        'current_price': 'Current $',
        'min_price_target_display': 'Low Target $',
        'avg_price_target': 'Mean Target $',
        'max_price_target_display': 'High Target $',
        'upside_low_display': 'Low Up %',
        'upside_percent': 'Mean Up %',
        'upside_high_display': 'High Up %',
        'num_analysts': 'Analysts',
        'recommendation': 'Rating',
        'ev_ebitda': 'EV/EBITDA',
        'debt_ebitda': 'Debt/EBITDA',
        'ocf_ev': 'OCF/EV',
        'value_score': 'Value Score',
        'projected_revenue_growth': 'Rev Growth %',
        'projected_eps_growth': 'EPS Growth %',
        'long_term_score': 'LT Score',
        'trend_score': 'Trend (/25)',
        'fundamentals_score': 'Fund (/25)',
        'valuation_score': 'Valuation (/16)',
        'momentum_score': 'Momentum (/10)',
        'market_risk_score': 'Mkt/Risk (/10)',
        'trend_signal': 'Trend Signals',
    }, inplace=True)

    # ── Styling ───────────────────────────────────────────────────────────
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
            else:
                return ''
        except:
            return ''

    def color_trend_signal(val):
        if pd.isna(val) or val is None or val == '':
            return ''
        count = str(val).count(',') + 1
        if count >= 3:
            return 'background-color: #c6f6d5; color: #155724; font-weight: bold'
        elif count >= 2:
            return 'background-color: #d4edda; color: #155724'
        else:
            return 'background-color: #fff3cd; color: #856404'

    format_dict = {
        'Current $': '${:.2f}',
        'Mean Target $': '${:.2f}',
        'Analysts': '{:.0f}',
        'EV/EBITDA': '{:.1f}x',
        'Debt/EBITDA': '{:.1f}x',
        'OCF/EV': '{:.1%}',
        'Value Score': '{:.0f}',
        'LT Score': '{:.0f}',
        'Rev Growth %': '{:+.1f}%',
        'EPS Growth %': '{:+.1f}%',
        'Trend (/25)': '{:.0f}',
        'Fund (/25)': '{:.0f}',
        'Valuation (/16)': '{:.0f}',
        'Momentum (/10)': '{:.0f}',
        'Mkt/Risk (/10)': '{:.0f}',
    }

    styled_table = display_cols.style \
        .format(format_dict, na_rep='-') \
        .map(color_upside, subset=['Low Up %', 'Mean Up %', 'High Up %']) \
        .map(color_trend_signal, subset=['Trend Signals']) \
        .background_gradient(subset=['Value Score'], cmap='Blues', vmin=0, vmax=100) \
        .background_gradient(subset=['LT Score'], cmap='Oranges', vmin=0, vmax=100) \
        .background_gradient(subset=['Rev Growth %'], cmap='YlGn', vmin=0, vmax=50) \
        .background_gradient(subset=['EPS Growth %'], cmap='YlGn', vmin=0, vmax=50) \
        .background_gradient(subset=['EV/EBITDA'], cmap='RdYlGn_r', vmin=0, vmax=30) \
        .background_gradient(subset=['Debt/EBITDA'], cmap='RdYlGn_r', vmin=0, vmax=10) \
        .background_gradient(subset=['OCF/EV'], cmap='YlGn', vmin=0, vmax=0.15) \
        .bar(subset=['Analysts'], color='#5fba7d', vmin=0)

    st.dataframe(styled_table, use_container_width=True, height=620)

    # ── Recent Analyst Actions ────────────────────────────────────────────
    if len(filtered) > 0:
        st.subheader("Recent Analyst Actions -- Top 15 by Value Score")
        top_n = filtered.sort_values('value_score', ascending=False).head(15)
        top_display = display_cols.loc[top_n.index]
        for _, row in top_display.iterrows():
            company_label = f" ({row['Company']})" if pd.notna(row.get('Company')) and row.get('Company') else ""
            trend_label = f"  |  Trend: {row['Trend Signals']}" if pd.notna(row.get('Trend Signals')) and row.get('Trend Signals') else ""
            with st.expander(
                f"**{row['Symbol']}**{company_label}  |  Value **{row['Value Score']:.0f}**  |  LT **{row['LT Score']:.0f}**  |  "
                f"{row['Mean Up %']:+.1f}%  |  {row['Analysts']:.0f} analysts  |  {row['Rating'] or 'N/A'}{trend_label}"
            ):
                orig_row = filtered.loc[row.name]
                if orig_row['recent_ratings'] and "No recent" not in str(orig_row['recent_ratings']):
                    st.markdown(orig_row['recent_ratings'].replace('\n', '  \n'))
                else:
                    st.info("No recent rating changes found")

with tab2:
    st.header("Scoring Methodology")
    st.markdown("""
    ### Value Score /100 -- Reward cheap & growing companies

    **Higher = better value** (only highest tier per category applies)

    - **EV/EBITDA**
      < 10 -> +30
      < 15 -> +20
      < 20 -> +10
      < 30 -> +5

    - **Projected Revenue Growth**
      > 25% -> +30
      > 15% -> +20
      > 8%  -> +10

    - **Projected EPS Growth** > 15% -> +15
    - **Projected EBITDA Growth** > 15% -> +10

    Max 100 points.

    ### New Metrics

    - **EV/EBITDA**: Enterprise Value / EBITDA. Lower = cheaper valuation. <10x is attractive, >30x is expensive.
    - **Debt/EBITDA**: Net Debt / EBITDA. Lower = less leveraged. <2x is healthy, >5x is concerning.
    - **OCF/EV**: Operating Cash Flow / Enterprise Value. Higher = better cash generation relative to valuation. >10% is strong.

    ### Trend Signals

    - **Golden Cross**: SMA50 crosses above SMA200 (strong bullish signal)
    - **Price > SMA50**: Price breaks above 50-day moving average
    - **RSI Recovery**: RSI recovers from oversold (<30) to neutral (>40)
    - **Bullish Aligned**: Price > SMA50 > SMA200 (uptrend confirmed)
    """)

with tab3:
    st.header("Long-Term Score Details /100")
    st.markdown("""
    Combines technical trend, fundamentals, valuation, momentum and market regime.

    | Category         | Max | Criteria |
    |------------------|-----|----------|
    | **Trend**        | 25  | Price > 200 SMA (+10), 50 SMA > 200 SMA (+10), Price > 50 SMA (+5) |
    | **Fundamentals** | 25  | Projected Revenue Growth >15% (+15) / >8% (+8), Projected EPS Growth >15% (+10) / >8% (+5) |
    | **Valuation**    | 16  | EV/EBITDA <12 (+10) / <20 (+6) / <30 (+3) |
    | **Momentum**     | 10  | RSI 40-65 (+5), ADX > 25 (+5) |
    | **Market Regime** | 10 | Bull market (SPY > 200 SMA) -> +10 |

    Use filters to find high LT + high Value + strong growth stocks.
    """)

st.markdown("---")
st.caption("Data from Financial Modeling Prep | Stocks with >= 1 analyst | "
           "Pipeline runs daily via GitHub Actions | "
           "Live prices during market hours")

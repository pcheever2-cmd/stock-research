#!/usr/bin/env python3
"""
Per-share FAIR VALUE (v2) — "what is it worth, and is it priced right today?" (DESCRIPTIVE product).
=====================================================================================================
Deliverable (A) from the fair-value plan. Triangulates an absolute fair-value RANGE per share from four
non-redundant methods, then a verdict (Under/Fair/Over) + the % gap vs today's price. This is the
USER-FACING number; it is NOT a return forecast (that's deliverable B, the validated mispricing composite
that supplies the CONVICTION). Per the panel: a per-share point estimate is plausibility, not alpha — so we
show a RANGE and lead with "not a forecast; can be a value trap."

Four methods (each -> an implied $/share; the spread across them IS the honesty):
  1. COMPS (comparable companies) — peer-group (industry, sector fallback) MEDIAN EV/EBITDA, P/E, P/S, P/B,
     P/FCF applied to the stock's own TTM fundamentals. The classic relative valuation, with the peer set
     surfaced for transparency ("cheap/expensive vs THESE peers").
  2. OWN-HISTORY reversion — the stock's own winsorized 5yr median EV/EBITDA & P/E applied to current TTM
     (ports compute_fair_value.py LENS 1). "Apple usually ~23x; at 30x -> above its norm."
  3. RESIDUAL INCOME (Edwards-Bell-Ohlson) — V = book + PV of (ROE - r)*book, ROE faded to r over a horizon.
     Accounting-anchored, far less terminal-sensitive than DCF.
  4. DCF (FCFF, conservative) — TTM FCF grown a few years then faded to a terminal g <= long-run GDP/inflation,
     discounted at a sector cost of equity. Reported as a base/pess/opt BAND (the assumption sensitivity).

Honest guards: clean market cap = latest price x diluted shares (never key_metrics ratios); require >=8 qtrs
& >=3yr history for own-history; winsorize peer multiples; drop absurd/again-noisy values; cap displayed gap.
Print-only (no DB/site writes). Run AFTER backfill_statement_extras.py.

Run:  python3 compute_fair_value_v2.py
"""
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
BACKTEST_DB = str(ROOT / 'backtest.db')
NASDAQ_DB = str(ROOT / 'nasdaq_stocks.db')

YEARS = 5
MIN_QTRS, MIN_SPAN_DAYS = 8, 1000
MIN_PEERS = 6                 # need >=6 industry peers for comps, else fall back to sector
WINS = (0.10, 0.90)          # winsorize peer/own multiples
TERMINAL_G = 0.025           # DCF terminal growth cap (<= long-run GDP/inflation)
SAMPLE = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'JPM', 'PFE', 'F', 'INTC', 'KO', 'XOM', 'WMT']

# Sector cost of equity (fixed risk-free + sector premium; robust anchor — beta-from-prices is noisy for
# small caps, per the panel). Risk-free ~4.3%; ERP scaled by rough sector beta. Used by RIM + DCF.
RF = 0.043
SECTOR_COE = {
    'Technology': 0.105, 'Communication Services': 0.100, 'Consumer Cyclical': 0.105,
    'Consumer Defensive': 0.080, 'Healthcare': 0.090, 'Financial Services': 0.095,
    'Industrials': 0.095, 'Energy': 0.105, 'Basic Materials': 0.100, 'Real Estate': 0.090,
    'Utilities': 0.075,
}
DEFAULT_COE = 0.095


def _ttm_panel():
    """Latest TTM fundamentals + latest balance snapshot + latest price, per symbol."""
    conn = sqlite3.connect(BACKTEST_DB)
    inc = pd.read_sql_query("""SELECT symbol,date,period,eps_diluted,ebitda,revenue,net_income,
        weighted_avg_shares_diluted AS shares, rd_expense, sga_expense, filing_date
        FROM historical_income_statements WHERE period LIKE 'Q%'""", conn)
    cf = pd.read_sql_query("""SELECT symbol,date,period,free_cash_flow,operating_cash_flow,
        dividends_paid, common_stock_repurchased FROM historical_cash_flows WHERE period LIKE 'Q%'""", conn)
    bal = pd.read_sql_query("""SELECT symbol,date,period,total_equity,net_debt,total_assets,
        total_current_assets,total_current_liabilities,goodwill,intangible_assets
        FROM historical_balance_sheets WHERE period LIKE 'Q%'""", conn)
    px = pd.read_sql_query("""SELECT symbol, close FROM historical_prices p
        WHERE date=(SELECT MAX(date) FROM historical_prices WHERE symbol=p.symbol)""", conn)
    conn.close()
    for d in (inc, cf, bal, px):
        d['symbol'] = d['symbol'].str.upper()

    df = inc.merge(cf, on=['symbol', 'date', 'period'], how='left').merge(
        bal, on=['symbol', 'date', 'period'], how='left').sort_values(['symbol', 'date'])
    g = df.groupby('symbol', group_keys=False)
    # TTM = rolling 4-quarter sum of FLOWS (need 4 quarters)
    for c in ['eps_diluted', 'ebitda', 'revenue', 'net_income', 'free_cash_flow',
              'operating_cash_flow', 'dividends_paid', 'common_stock_repurchased', 'rd_expense', 'sga_expense']:
        df[f'ttm_{c}'] = g[c].apply(lambda s: s.rolling(4, min_periods=4).sum())
    last = df.groupby('symbol').tail(1).copy()       # latest quarter row (balance = point-in-time snapshot)
    last = last.merge(px, on='symbol', how='left', suffixes=('', '_px'))
    last['price'] = last['close']
    last['mktcap'] = last['price'] * last['shares']
    return last


def _own_history():
    """Per-symbol current trailing P/E & EV/EBITDA + own winsorized 5yr median + history breadth.
    Ported from compute_fair_value.py (raw-close x shares / TTM, filing-date aligned)."""
    conn = sqlite3.connect(BACKTEST_DB)
    inc = pd.read_sql_query("""SELECT symbol,date,eps_diluted,ebitda,weighted_avg_shares_diluted AS shares,
        filing_date FROM historical_income_statements WHERE period LIKE 'Q%'""", conn)
    bal = pd.read_sql_query("SELECT symbol,date,net_debt FROM historical_balance_sheets WHERE period LIKE 'Q%'", conn)
    cutoff = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=YEARS * 365)).strftime('%Y-%m-%d')
    prices = pd.read_sql_query("SELECT symbol,date,close FROM historical_prices WHERE date >= ?", conn, params=(cutoff,))
    conn.close()
    for d in (inc, bal, prices):
        d['symbol'] = d['symbol'].str.upper()
    df = inc.merge(bal, on=['symbol', 'date'], how='left')
    df['asof'] = pd.to_datetime(df['filing_date'], errors='coerce').fillna(
        pd.to_datetime(df['date'], errors='coerce') + pd.Timedelta(days=75))
    df = df.dropna(subset=['asof']).sort_values(['symbol', 'date'])
    g = df.groupby('symbol', group_keys=False)
    df['ttm_eps'] = g['eps_diluted'].apply(lambda s: s.rolling(4, min_periods=4).sum())
    df['ttm_ebitda'] = g['ebitda'].apply(lambda s: s.rolling(4, min_periods=4).sum())
    fund = df[['symbol', 'date', 'asof', 'ttm_eps', 'ttm_ebitda', 'shares', 'net_debt']].rename(
        columns={'date': 'period_end'}).dropna(subset=['asof']).sort_values('asof')
    prices['dt'] = pd.to_datetime(prices['date'], errors='coerce')
    prices = prices.dropna(subset=['dt', 'close']).sort_values('dt')
    m = pd.merge_asof(prices, fund, left_on='dt', right_on='asof', by='symbol', direction='backward')
    mc = m['close'] * m['shares']
    m['pe'] = (m['close'] / m['ttm_eps']).where(m['ttm_eps'] > 0)
    m['ev'] = ((mc + m['net_debt']) / m['ttm_ebitda']).where(m['ttm_ebitda'] > 0)
    m = m[((m['pe'] > 0) & (m['pe'] < 500)) | ((m['ev'] > 0) & (m['ev'] < 200))]

    def wmed(s):
        s = s.dropna()
        return float(np.clip(s, s.quantile(0.05), s.quantile(0.95)).median()) if len(s) >= 30 else np.nan
    rows = []
    for sym, grp in m.groupby('symbol'):
        rows.append(dict(symbol=sym, own_med_pe=wmed(grp['pe']), own_med_ev=wmed(grp['ev']),
                         n_qtrs=grp['period_end'].nunique(), span_days=(grp['dt'].max() - grp['dt'].min()).days))
    return pd.DataFrame(rows)


def _wins_median(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    s = s[s > 0]
    if len(s) < MIN_PEERS:
        return np.nan
    return float(np.clip(s, s.quantile(WINS[0]), s.quantile(WINS[1])).median())


def add_peer_multiples(df):
    """COMPS: peer-MEDIAN EV/EBITDA, P/E, P/S, P/B, P/FCF by industry (sector fallback if <MIN_PEERS)."""
    df = df.copy()
    ev = df['mktcap'] + df['net_debt'].fillna(0)
    df['m_ev_ebitda'] = (ev / df['ttm_ebitda']).where(df['ttm_ebitda'] > 0)
    df['m_pe'] = (df['price'] / df['ttm_eps_diluted']).where(df['ttm_eps_diluted'] > 0)
    df['m_ps'] = (df['mktcap'] / df['ttm_revenue']).where(df['ttm_revenue'] > 0)
    df['m_pb'] = (df['mktcap'] / df['total_equity']).where(df['total_equity'] > 0)
    df['m_pfcf'] = (df['mktcap'] / df['ttm_free_cash_flow']).where(df['ttm_free_cash_flow'] > 0)
    mults = ['m_ev_ebitda', 'm_pe', 'm_ps', 'm_pb', 'm_pfcf']
    for key, tag in [('industry', 'ind'), ('sector', 'sec')]:
        for m in mults:
            df[f'{tag}_{m}'] = df.groupby(key)[m].transform(_wins_median)
            df[f'{tag}_n_{m}'] = df.groupby(key)[m].transform(lambda s: (s.replace([np.inf, -np.inf], np.nan) > 0).sum())
    # peer median: industry if enough peers else sector
    for m in mults:
        use_ind = df[f'ind_n_{m}'] >= MIN_PEERS
        df[f'peer_{m}'] = np.where(use_ind, df[f'ind_{m}'], df[f'sec_{m}'])
    df['peer_basis'] = np.where(df['ind_n_m_ev_ebitda'] >= MIN_PEERS, 'industry', 'sector')
    return df


def comps_value(r):
    """Implied $/share from each peer multiple applied to the stock's own TTM fundamental."""
    sh = r['shares']
    out = {}
    if sh and sh > 0:
        if r['ttm_ebitda'] > 0 and pd.notna(r['peer_m_ev_ebitda']):
            out['EV/EBITDA'] = (r['peer_m_ev_ebitda'] * r['ttm_ebitda'] - r['net_debt']) / sh
        if r['ttm_eps_diluted'] > 0 and pd.notna(r['peer_m_pe']):
            out['P/E'] = r['peer_m_pe'] * r['ttm_eps_diluted']
        if r['ttm_revenue'] > 0 and pd.notna(r['peer_m_ps']):
            out['P/S'] = r['peer_m_ps'] * r['ttm_revenue'] / sh
        if r['total_equity'] > 0 and pd.notna(r['peer_m_pb']):
            out['P/B'] = r['peer_m_pb'] * r['total_equity'] / sh
        if r['ttm_free_cash_flow'] > 0 and pd.notna(r['peer_m_pfcf']):
            out['P/FCF'] = r['peer_m_pfcf'] * r['ttm_free_cash_flow'] / sh
    return {k: v for k, v in out.items() if pd.notna(v) and v > 0}


def own_history_value(r):
    """Implied $/share from the stock's own 5yr median multiple (EV/EBITDA primary, P/E confirm)."""
    if not (r['n_qtrs'] >= MIN_QTRS and r['span_days'] >= MIN_SPAN_DAYS):
        return {}
    sh, out = r['shares'], {}
    if sh and sh > 0 and r['ttm_ebitda'] > 0 and pd.notna(r['own_med_ev']):
        out['own EV/EBITDA'] = (r['own_med_ev'] * r['ttm_ebitda'] - r['net_debt']) / sh
    if r['ttm_eps_diluted'] > 0 and pd.notna(r['own_med_pe']):
        out['own P/E'] = r['own_med_pe'] * r['ttm_eps_diluted']
    return {k: v for k, v in out.items() if pd.notna(v) and v > 0}


def residual_income_value(r, coe):
    """EBO: V = B0 + sum_t (ROE_t - r) * B_{t-1} / (1+r)^t, ROE faded linearly to r over T yrs + terminal."""
    book, ni, sh = r['total_equity'], r['ttm_net_income'], r['shares']
    if not (book and book > 0 and sh and sh > 0 and pd.notna(ni)):
        return np.nan
    roe0 = ni / book
    if not np.isfinite(roe0) or roe0 < -1 or roe0 > 1.5:    # drop absurd ROE (data noise)
        return np.nan
    r_, T = coe, 8
    B, V = book, book
    roe = roe0
    for t in range(1, T + 1):
        roe = roe + (r_ - roe0) * (t / T)                  # fade ROE0 -> r over T years
        ri = (roe - r_) * B
        V += ri / (1 + r_) ** t
        B = B * (1 + roe)                                   # clean-surplus growth (no payout split)
    # terminal residual income ~ 0 at fade end (ROE->r), so no perpetuity needed
    return V / sh if V > 0 else np.nan


def dcf_value(r, coe, g_explicit):
    """Conservative FCFF DCF: TTM FCF grows g_explicit for 5y, fades to TERMINAL_G, discount at coe."""
    fcf, nd, sh = r['ttm_free_cash_flow'], r['net_debt'], r['shares']
    if not (fcf and fcf > 0 and sh and sh > 0):
        return np.nan
    r_, N = coe, 5
    pv, f = 0.0, fcf
    for t in range(1, N + 1):
        gt = g_explicit + (TERMINAL_G - g_explicit) * (t / N)   # fade growth to terminal
        f = f * (1 + gt)
        pv += f / (1 + r_) ** t
    tv = f * (1 + TERMINAL_G) / (r_ - TERMINAL_G)                # Gordon terminal
    pv += tv / (1 + r_) ** N
    equity = pv - (nd if pd.notna(nd) else 0)
    return equity / sh if equity > 0 else np.nan


def value_one(r):
    """Fair-value RANGE + verdict. The CENTRAL estimate is the MARKET-CALIBRATED RELATIVE methods (comps +
    own-history multiples, sector-aware) — they directly answer "priced right vs peers and its own history"
    without the discount-rate landmine. Residual income & DCF are notoriously discount-rate-sensitive (a ~9%
    cost of equity makes most quality stocks look overvalued at today's ~3% yields), so they're shown as a
    flagged INTRINSIC cross-check, NOT in the central range. P/S, P/B, P/FCF shown for transparency."""
    coe = SECTOR_COE.get(r.get('sector'), DEFAULT_COE)
    comps = comps_value(r)
    own = own_history_value(r)
    is_fin = r.get('sector') == 'Financial Services'
    p = r['price']
    if not (p and p > 0):
        return None

    # CENTRAL = relative anchors. Banks: P/E & P/B (EV/EBITDA is meaningless for financials). Else: EV/EBITDA & P/E.
    core = {}
    if is_fin:
        for k in ('P/E', 'P/B'):
            if k in comps:
                core[f'comps {k}'] = comps[k]
        if 'own P/E' in own:
            core['own P/E'] = own['own P/E']
    else:
        for k in ('EV/EBITDA', 'P/E'):
            if k in comps:
                core[f'comps {k}'] = comps[k]
        core.update(own)                                # own EV/EBITDA, own P/E
    core = {k: v for k, v in core.items() if 0.1 * p <= v <= 10 * p}
    if len(core) < 2:
        return None
    vals = np.array(list(core.values()))
    lo, mid, hi = np.percentile(vals, 25), np.median(vals), np.percentile(vals, 75)
    upside = mid / p - 1
    verdict = 'Undervalued' if upside >= 0.15 else 'Overvalued' if upside <= -0.15 else 'Fair value'

    # Intrinsic cross-check (discount-rate sensitive) + supplementary multiples — display only.
    intrinsic = {}
    ri = residual_income_value(r, coe)
    if pd.notna(ri):
        intrinsic['Residual income'] = ri
    db = dcf_value(r, coe, 0.06)
    if pd.notna(db):
        intrinsic['DCF'] = db
    display = dict(core)
    for k in ('P/S', 'P/B', 'P/FCF'):
        if not (is_fin and k == 'P/B') and k in comps:
            display[f'comps {k}'] = comps[k]
    return dict(fv_low=lo, fv_mid=mid, fv_high=hi, upside=upside, verdict=verdict,
                n_methods=len(core), methods=core, intrinsic=intrinsic, display=display,
                peer_basis=r.get('peer_basis'))


def write_to_db(res):
    """Persist the fair-value $ RANGE to nasdaq_stocks.db.stock_consensus (keyed by symbol). upside%/verdict
    are NOT stored — they are derived at export time from the site's current_price (price-source consistency)."""
    now = datetime.now(timezone.utc).isoformat()
    rows = [(round(float(v['fv_low']), 2), round(float(v['fv_mid']), 2), round(float(v['fv_high']), 2),
             int(v['n_methods']), v.get('peer_basis'), now, sym) for sym, v in res.items()]
    conn = sqlite3.connect(NASDAQ_DB, timeout=60)
    cur = conn.cursor()
    cur.executemany("""
        UPDATE stock_consensus SET fair_value_low=?, fair_value_mid=?, fair_value_high=?,
               fair_value_n_methods=?, fair_value_basis=?, fair_value_computed_at=?
        WHERE UPPER(symbol)=?
    """, rows)
    n = cur.rowcount
    conn.commit(); conn.close()
    print(f"\nWrote fair value to stock_consensus for {n:,} symbols (of {len(res):,} valued).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true', help='persist fair-value range to nasdaq_stocks.db')
    args = ap.parse_args()
    print("Loading TTM fundamentals + own-history multiples + peers...")
    df = _ttm_panel()
    oh = _own_history()
    conn = sqlite3.connect(NASDAQ_DB)
    cons = pd.read_sql_query("SELECT symbol, sector, industry FROM stock_consensus", conn)
    conn.close()
    cons['symbol'] = cons['symbol'].str.upper()
    df = df.merge(cons, on='symbol', how='left').merge(oh, on='symbol', how='left')
    df['sector'] = df['sector'].fillna('Unknown'); df['industry'] = df['industry'].fillna(df['sector'])
    df = add_peer_multiples(df)

    res = {}
    for _, r in df.iterrows():
        v = value_one(r)
        if v:
            res[r['symbol']] = v
    out = pd.DataFrame.from_dict({k: {kk: vv for kk, vv in d.items() if kk != 'methods'}
                                  for k, d in res.items()}, orient='index')
    print(f"\nValued {len(out):,} stocks (>=3 methods agreeing within [0.1x,10x] price).\n")
    print("=" * 104)
    print("EXAMPLES — fair-value range vs price")
    print("=" * 104)
    print(f"{'Sym':<6}{'price':>9}{'FV_low':>9}{'FV_mid':>9}{'FV_high':>9}{'upside':>8}  {'Verdict':<12} methods")
    print("-" * 104)
    dd = df.set_index('symbol')
    for s in SAMPLE:
        if s not in res:
            print(f"{s:<6}(insufficient data)"); continue
        v = res[s]; p = dd.loc[s, 'price']
        ms = ', '.join(f"{k} ${vv:.0f}" for k, vv in v['methods'].items())
        intr = '  | intrinsic: ' + ', '.join(f"{k} ${vv:.0f}" for k, vv in v['intrinsic'].items()) if v['intrinsic'] else ''
        print(f"{s:<6}{p:>9.2f}{v['fv_low']:>9.0f}{v['fv_mid']:>9.0f}{v['fv_high']:>9.0f}"
              f"{v['upside']*100:>7.0f}%  {v['verdict']:<12} {ms}{intr}")
    print("\n" + "=" * 104)
    print("DISTRIBUTION")
    print("=" * 104)
    for lab in ['Undervalued', 'Fair value', 'Overvalued']:
        n = (out['verdict'] == lab).sum()
        print(f"  {lab:<13}{n:>6,}  ({n/len(out)*100:4.1f}%)")
    print("\nDESCRIPTIVE only — a fair-value RANGE is plausibility, not a forecast; can be a value trap. "
          "Conviction comes from the validated mispricing composite (deliverable B).")
    if args.write:
        write_to_db(res)
    else:
        print("(no DB write — pass --write to persist to stock_consensus)")


if __name__ == '__main__':
    main()

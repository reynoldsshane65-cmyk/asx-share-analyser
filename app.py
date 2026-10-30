import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
from pathlib import Path
import sqlite3
import json

st.set_page_config(page_title="ASX Share Analyser", page_icon="📈", layout="wide")
st.title("📈 ASX Share Analyser")
st.caption("Research prototype for Australian shares — no automatic trading")
st.warning("Not financial advice. Free data can be incomplete, delayed or incorrect. Verify company figures against ASX announcements and official reports. Scores are screening aids, not predictions.")

STARTER_TICKERS = ["BHP","CBA","CSL","WES","WOW","RIO","NAB","WBC","ANZ","FMG","MQG","GMG","REA","CAR","XRO","ALL","COL","TLS","STO","WDS","ORG","PLS","MIN","IGO","LYC","DRO","DMP","ZIP","TLX","PDN","BOE","LTR","SYR","CXO","A2M","BRG","IEL","ARB","CPU","QAN"]

def load_master_tickers():
    """Load bundled ASX company list; fall back to starter tickers if unavailable."""
    for path in [Path(__file__).with_name("ASX_Listed_Companies.csv"), Path("ASX_Listed_Companies.csv")]:
        try:
            if path.exists():
                companies = pd.read_csv(path)
                col = next((c for c in companies.columns if str(c).strip().lower() in {"asx code", "ticker", "symbol"}), None)
                if col:
                    codes = companies[col].dropna().astype(str).str.strip().str.upper()
                    codes = codes[codes.str.fullmatch(r"[A-Z0-9]{1,6}")]
                    codes = list(dict.fromkeys(codes.tolist()))
                    if codes:
                        return codes
        except Exception:
            pass
    return STARTER_TICKERS


@st.cache_data(ttl=900, show_spinner=False)
def load_company(symbol, period):
    t = yf.Ticker(symbol)
    try: hist = t.history(period=period, auto_adjust=True)
    except Exception: hist = pd.DataFrame()
    try: info = t.info or {}
    except Exception: info = {}
    try: income = t.financials
    except Exception: income = pd.DataFrame()
    try: balance = t.balance_sheet
    except Exception: balance = pd.DataFrame()
    try: cash = t.cashflow
    except Exception: cash = pd.DataFrame()
    return hist, info, income, balance, cash

def load_prices(symbol, period):
    """Fetch prices with one retry. Do not cache empty responses, which may be temporary Yahoo throttling."""
    for attempt in range(2):
        try:
            h = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if h is not None and not h.empty and "Close" in h:
                result = h[[c for c in ["Close", "Volume"] if c in h.columns]].dropna(subset=["Close"])
                if not result.empty:
                    return result
        except Exception:
            pass
        if attempt == 0:
            time.sleep(0.25)
    return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_info(symbol):
    """Load Yahoo fundamentals with fallbacks and a diagnostic status."""
    t = yf.Ticker(symbol)
    errors = []
    info = {}
    # get_info() and .info use the same Yahoo endpoint on many yfinance versions,
    # but trying both helps with version-specific behaviour.
    for getter in (lambda: t.get_info(), lambda: t.info):
        try:
            candidate = getter()
            if isinstance(candidate, dict) and candidate:
                info.update(candidate)
                break
        except Exception as exc:
            errors.append(type(exc).__name__)
    fast = {}
    try:
        fi = t.fast_info
        for key in ("market_cap", "shares", "last_price", "currency"):
            try:
                value = fi.get(key)
                if value is not None:
                    fast[key] = value
            except Exception:
                pass
    except Exception as exc:
        errors.append(type(exc).__name__)
    return {"info": info, "fast": fast, "error": "; ".join(errors)}

def financial_fallback(symbol):
    """Derive basic growth/margin metrics from annual statements if Yahoo info is missing."""
    result = {"revenue_growth": None, "profit_margin": None, "net_income": None, "revenue": None}
    try:
        statements = yf.Ticker(symbol).financials
        if statements is None or statements.empty:
            return result
        def latest(names):
            for name in names:
                if name in statements.index:
                    vals = statements.loc[name].dropna()
                    if len(vals):
                        return vals
            return pd.Series(dtype=float)
        revenue = latest(["Total Revenue", "Operating Revenue"])
        net = latest(["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"])
        if len(revenue):
            result["revenue"] = num(revenue.iloc[0])
            if len(revenue) >= 2 and num(revenue.iloc[1]) not in (None, 0):
                result["revenue_growth"] = (float(revenue.iloc[0]) / float(revenue.iloc[1]) - 1) * 100
        if len(net):
            result["net_income"] = num(net.iloc[0])
            if result["revenue"] not in (None, 0):
                result["profit_margin"] = float(net.iloc[0]) / result["revenue"] * 100
    except Exception:
        pass
    return result

def num(x):
    try: return None if x is None or pd.isna(x) else float(x)
    except Exception: return None

def fmt(x, kind="num"):
    x = num(x)
    if x is None: return "N/A"
    if kind == "money": return f"${x:,.3f}"
    if kind == "pct": return f"{x*100:.2f}%"
    if kind == "large":
        if abs(x) >= 1e9: return f"${x/1e9:.2f}B"
        if abs(x) >= 1e6: return f"${x/1e6:.2f}M"
    return f"{x:,.3f}"

def stmt(frame, names):
    if frame is None or frame.empty: return None
    for name in names:
        if name in frame.index:
            vals = frame.loc[name].dropna()
            if len(vals): return num(vals.iloc[0])
    return None

def pct(series, days):
    if len(series) <= days: return np.nan
    old, new = num(series.iloc[-days-1]), num(series.iloc[-1])
    return (new/old-1)*100 if old not in (None, 0) and new is not None else np.nan

def early_mover_score(row):
    """Transparent technical screen; this is not a probability or forecast."""
    parts = []
    def add(value, weight, transform):
        value = num(value)
        if value is not None and np.isfinite(value):
            parts.append((transform(value), weight))
    add(row.get("Return 1W %"), 15, lambda v: float(np.clip(50 + v * 5, 0, 100)))
    add(row.get("Return 1M %"), 15, lambda v: float(np.clip(50 + v * 2, 0, 100)))
    add(row.get("Return 3M %"), 10, lambda v: float(np.clip(50 + v, 0, 100)))
    add(row.get("Price vs 50D %"), 10, lambda v: 100 if v > 0 else float(np.clip(50 + v * 3, 0, 50)))
    add(row.get("Above 200D %"), 10, lambda v: 100 if v > 0 else float(np.clip(50 + v, 0, 50)))
    add(row.get("Volume Change %"), 20, lambda v: float(np.clip(50 + v * 0.35, 0, 100)))
    add(row.get("Distance from 20D High %"), 10, lambda v: float(np.clip(100 + v * 5, 0, 100)))
    add(row.get("Return 1W %"), 10, lambda v: 100 if 0 <= v <= 12 else (65 if -3 <= v < 0 or 12 < v <= 20 else 25))
    return round(sum(v*w for v,w in parts) / sum(w for _,w in parts), 1) if parts else np.nan

def score_row(row):
    weighted = []
    for col, weight, scale in [("Return 1M %",.25,2),("Return 3M %",.25,1),("Return 6M %",.2,.5),("Above 200D %",.2,2),("Volume Change %",.1,.25)]:
        v = num(row.get(col))
        if v is not None:
            part = 100 if col == "Above 200D %" and v > 0 else (25 if col == "Above 200D %" else float(np.clip(50+v*scale,0,100)))
            weighted.append((part,weight))
    mom = sum(v*w for v,w in weighted)/sum(w for _,w in weighted) if weighted else np.nan
    pe, margin, growth = num(row.get("P/E")), num(row.get("Profit Margin %")), num(row.get("Revenue Growth %"))
    val_parts = ([float(np.clip(100-pe*1.5,0,100))] if pe is not None and pe > 0 else []) + ([float(np.clip(50+margin,0,100))] if margin is not None else [])
    growth_parts = ([float(np.clip(50+growth,0,100))] if growth is not None else []) + ([float(np.clip(50+margin,0,100))] if margin is not None else [])
    val = float(np.mean(val_parts)) if val_parts else np.nan
    gro = float(np.mean(growth_parts)) if growth_parts else np.nan
    available = [(mom,.4),(val,.3),(gro,.3)]
    available = [(v,w) for v,w in available if pd.notna(v)]
    overall = sum(v*w for v,w in available)/sum(w for _,w in available) if available else np.nan
    return pd.Series({"Momentum Score":round(mom,1) if pd.notna(mom) else np.nan,"Value Score":round(val,1) if pd.notna(val) else np.nan,"Growth Score":round(gro,1) if pd.notna(gro) else np.nan,"Overall Score":round(overall,1) if pd.notna(overall) else np.nan})

@st.cache_data(ttl=900, show_spinner=False)
def download_price_batches(tickers, period, batch_size=30):
    """
    Download price history in small batches instead of making a separate price
    request for every ticker. This reduces request pressure and avoids the
    cascade of failures seen when Yahoo throttles repeated per-ticker requests.
    """
    output = {}
    symbols = [str(x).strip().upper().replace(".AX", "") for x in tickers if str(x).strip()]
    for start in range(0, len(symbols), batch_size):
        batch = symbols[start:start + batch_size]
        yahoo_symbols = [s + ".AX" for s in batch]
        data = pd.DataFrame()
        for attempt in range(3):
            try:
                data = yf.download(
                    tickers=" ".join(yahoo_symbols),
                    period=period,
                    auto_adjust=True,
                    group_by="ticker",
                    threads=False,
                    progress=False,
                    timeout=25
                )
                if data is not None and not data.empty:
                    break
            except Exception:
                data = pd.DataFrame()
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
        for sym, ysym in zip(batch, yahoo_symbols):
            try:
                if data is None or data.empty:
                    output[sym] = pd.DataFrame()
                    continue
                if isinstance(data.columns, pd.MultiIndex):
                    # group_by="ticker" normally gives ticker as level 0.
                    if ysym in data.columns.get_level_values(0):
                        h = data[ysym].copy()
                    elif ysym in data.columns.get_level_values(-1):
                        h = data.xs(ysym, axis=1, level=-1).copy()
                    else:
                        output[sym] = pd.DataFrame()
                        continue
                else:
                    # Single-symbol downloads can return flat columns.
                    h = data.copy() if len(batch) == 1 else pd.DataFrame()
                wanted = [c for c in ["Close", "Volume"] if c in h.columns]
                h = h[wanted].dropna(subset=["Close"]) if "Close" in wanted else pd.DataFrame()
                output[sym] = h
            except Exception:
                output[sym] = pd.DataFrame()
        time.sleep(1.0)  # modest pause between batches
    return output


def run_scan(tickers, period):
    rows = []
    if not tickers:
        return pd.DataFrame()

    # Fetch price history in batches first. This prevents individual price
    # requests and financial-data requests interleaving until Yahoo throttles.
    progress = st.progress(0, text="Downloading price history in batches…")
    price_data = download_price_batches(tickers, period, batch_size=30)
    progress.progress(0.15, text="Price downloads finished; calculating results…")
    total = len(tickers)

    for i, raw_sym in enumerate(tickers):
        sym = raw_sym.strip().upper().replace(".AX", "")
        h = price_data.get(sym, pd.DataFrame())
        if h is None or h.empty or len(h) < 30:
            status = "No price data from batch download" if h is None or h.empty else f"Insufficient price history ({len(h)} rows; need 30)"
            rows.append({
                "Ticker": sym, "Company": sym, "Price": np.nan,
                "Return 1W %": np.nan, "Return 1M %": np.nan,
                "Return 3M %": np.nan, "Return 6M %": np.nan,
                "Return 12M %": np.nan, "50D Average": np.nan, "Price vs 50D %": np.nan, "20D High": np.nan, "Distance from 20D High %": np.nan,
                "200D Average": np.nan, "Above 200D %": np.nan,
                "Volume Change %": np.nan, "Market Cap": np.nan,
                "P/E": np.nan, "Revenue Growth %": np.nan,
                "Profit Margin %": np.nan, "Fundamentals Status": "Not fetched (no prices)",
                "Data Date": "", "Scan Status": status
            })
            progress.progress(0.15 + 0.85 * (i + 1) / total, text=f"Checking {sym} ({i+1}/{total})…")
            continue

        close = h["Close"].dropna()
        vol = h["Volume"].fillna(0) if "Volume" in h else pd.Series(dtype=float)
        ma200 = close.tail(200).mean() if len(close) >= 200 else np.nan
        v20 = num(vol.tail(20).mean()) if len(vol) >= 20 else None
        vprev = num(vol.iloc[-40:-20].mean()) if len(vol) >= 40 else None

        # Limit fundamental calls during a full-market scan. Fetch one info
        # payload only; don't request financial statements for every missing field.
        loaded = load_info(sym + ".AX")
        info = loaded.get("info", {}) if isinstance(loaded, dict) else {}
        fast = loaded.get("fast", {}) if isinstance(loaded, dict) else {}
        shares = num(info.get("sharesOutstanding")) or num(info.get("impliedSharesOutstanding")) or num(fast.get("shares"))
        market_cap = num(info.get("marketCap")) or num(fast.get("market_cap")) or (num(close.iloc[-1]) * shares if shares else None)
        pe = num(info.get("trailingPE"))
        revenue_growth = num(info.get("revenueGrowth"))
        revenue_growth = revenue_growth * 100 if revenue_growth is not None else np.nan
        profit_margin = num(info.get("profitMargins"))
        profit_margin = profit_margin * 100 if profit_margin is not None else np.nan
        fundamentals_status = "Yahoo info OK" if info else "Fundamentals unavailable (price data available)"

        rows.append({
            "Ticker": sym, "Company": info.get("longName") or info.get("shortName") or sym,
            "Price": num(close.iloc[-1]), "Return 1W %": pct(close, 5),
            "Return 1M %": pct(close, 21), "Return 3M %": pct(close, 63),
            "Return 6M %": pct(close, 126), "Return 12M %": pct(close, 252),
            "50D Average": num(close.tail(50).mean()) if len(close) >= 50 else np.nan,
            "Price vs 50D %": (close.iloc[-1] / close.tail(50).mean() - 1) * 100 if len(close) >= 50 and close.tail(50).mean() else np.nan,
            "20D High": num(close.tail(20).max()) if len(close) >= 20 else np.nan,
            "Distance from 20D High %": (close.iloc[-1] / close.tail(20).max() - 1) * 100 if len(close) >= 20 and close.tail(20).max() else np.nan,
            "200D Average": num(ma200),
            "Above 200D %": (close.iloc[-1] / ma200 - 1) * 100 if pd.notna(ma200) and ma200 else np.nan,
            "Volume Change %": (v20 / vprev - 1) * 100 if v20 is not None and vprev not in (None, 0) else np.nan,
            "Market Cap": market_cap, "P/E": pe, "Revenue Growth %": revenue_growth,
            "Profit Margin %": profit_margin, "Fundamentals Status": fundamentals_status,
            "Data Date": close.index[-1].strftime("%Y-%m-%d"),
            "Scan Status": "Scanned successfully"
        })
        progress.progress(0.15 + 0.85 * (i + 1) / total, text=f"Processing {sym} ({i+1}/{total})…")

    progress.empty()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return pd.concat([df, df.apply(score_row, axis=1)], axis=1)


# ---------------------------------------------------------------------------
# Prediction history: snapshots are saved separately from the live scan cache.
# SQLite is convenient for this prototype; Streamlit Cloud storage is not a
# guaranteed permanent database, so export backups from the Performance page.
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).with_name("prediction_history.db")
PREDICTION_MODEL_VERSION = "v1"

def db_connect():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

def init_prediction_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_type TEXT NOT NULL,
                model_version TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                price REAL,
                overall_score REAL,
                momentum_score REAL,
                value_score REAL,
                growth_score REAL,
                early_mover_score REAL,
                return_1w REAL,
                return_1m REAL,
                return_3m REAL,
                price_vs_50d REAL,
                above_200d REAL,
                volume_change REAL,
                distance_20d_high REAL,
                outcome_5d REAL,
                outcome_20d REAL,
                outcome_60d REAL,
                UNIQUE(ticker, prediction_type, model_version, prediction_date)
            )
        """)
        conn.commit()

def record_prediction_snapshots(frame):
    if frame is None or frame.empty or "Ticker" not in frame.columns:
        return 0
    init_prediction_db()
    saved = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with db_connect() as conn:
        for _, row in frame.iterrows():
            price = num(row.get("Price"))
            date_value = str(row.get("Data Date", "") or "")
            if price is None or not date_value:
                continue
            early = early_mover_score(row)
            common = (
                str(row.get("Ticker", "")).upper(), PREDICTION_MODEL_VERSION,
                date_value, now, price, num(row.get("Overall Score")),
                num(row.get("Momentum Score")), num(row.get("Value Score")),
                num(row.get("Growth Score")), num(early),
                num(row.get("Return 1W %")), num(row.get("Return 1M %")),
                num(row.get("Return 3M %")), num(row.get("Price vs 50D %")),
                num(row.get("Above 200D %")), num(row.get("Volume Change %")),
                num(row.get("Distance from 20D High %"))
            )
            for prediction_type in ("Overall", "Early-Mover"):
                cur = conn.execute("""
                    INSERT OR IGNORE INTO predictions
                    (ticker, prediction_type, model_version, prediction_date, recorded_at,
                     price, overall_score, momentum_score, value_score, growth_score,
                     early_mover_score, return_1w, return_1m, return_3m, price_vs_50d,
                     above_200d, volume_change, distance_20d_high)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (common[0], prediction_type, *common[1:]))
                saved += cur.rowcount
        conn.commit()
    return saved

def read_prediction_history():
    init_prediction_db()
    with db_connect() as conn:
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY prediction_date DESC, ticker", conn)

def update_prediction_outcomes():
    history = read_prediction_history()
    if history.empty:
        return 0, 0
    pending = history[
        history[["outcome_5d", "outcome_20d", "outcome_60d"]].isna().any(axis=1)
    ]
    if pending.empty:
        return 0, 0
    symbols = sorted(pending["ticker"].dropna().astype(str).unique().tolist())
    price_map = download_price_batches(symbols, "2y", batch_size=30)
    updated = checked = 0
    with db_connect() as conn:
        for _, prediction in pending.iterrows():
            hist = price_map.get(prediction["ticker"], pd.DataFrame())
            if hist is None or hist.empty or "Close" not in hist:
                continue
            checked += 1
            dates = pd.to_datetime(hist.index)
            future = hist.loc[dates.strftime("%Y-%m-%d") > str(prediction["prediction_date"]), "Close"].dropna()
            if future.empty:
                continue
            values = {}
            for horizon in (5, 20, 60):
                col = f"outcome_{horizon}d"
                if pd.isna(prediction[col]) and len(future) >= horizon:
                    future_price = num(future.iloc[horizon - 1])
                    start_price = num(prediction["price"])
                    if start_price not in (None, 0) and future_price is not None:
                        values[col] = (future_price / start_price - 1) * 100
            if values:
                assignments = ", ".join(f"{key}=?" for key in values)
                conn.execute(
                    f"UPDATE predictions SET {assignments} WHERE id=?",
                    (*values.values(), int(prediction["id"]))
                )
                updated += 1
        conn.commit()
    return updated, checked

def prediction_performance_page():
    st.header("Prediction Performance & Learning")
    st.write("This page checks recorded scanner signals against subsequent share prices. It measures historical outcomes; it does not prove that the signals will work in future.")
    history = read_prediction_history()
    if history.empty:
        st.info("No prediction history has been recorded yet. Run a scan after installing this version; each ticker/date will be recorded once for both Overall and Early-Mover rankings.")
        return

    st.caption(f"Recorded predictions: {len(history):,} | First recorded date: {history['prediction_date'].min()} | Latest: {history['prediction_date'].max()}")
    if st.button("Update prediction outcomes", type="primary"):
        with st.spinner("Checking subsequent ASX share prices…"):
            updated, checked = update_prediction_outcomes()
        st.success(f"Checked {checked} prediction records and updated {updated}.")
        history = read_prediction_history()

    type_choice = st.selectbox("Prediction model", ["Early-Mover", "Overall"])
    data = history[history["prediction_type"] == type_choice].copy()
    st.subheader("Performance by timeframe")
    metric_cols = st.columns(3)
    for col, horizon in zip(metric_cols, (5, 20, 60)):
        outcome = data[f"outcome_{horizon}d"].dropna()
        col.metric(f"{horizon}-trading-day win rate", f"{(outcome > 0).mean()*100:.1f}%" if len(outcome) else "Pending")
        col.caption(f"{len(outcome)} evaluated")
    summary_rows = []
    for horizon in (5, 20, 60):
        outcome = data[f"outcome_{horizon}d"].dropna()
        summary_rows.append({
            "Horizon": f"{horizon} trading days",
            "Evaluated": len(outcome),
            "Positive outcomes": int((outcome > 0).sum()),
            "Win rate %": round((outcome > 0).mean()*100, 1) if len(outcome) else np.nan,
            "Average return %": round(outcome.mean(), 2) if len(outcome) else np.nan,
            "Median return %": round(outcome.median(), 2) if len(outcome) else np.nan,
            "20%+ rises": int((outcome >= 20).sum()) if len(outcome) else 0,
            "10%+ falls": int((outcome <= -10).sum()) if len(outcome) else 0
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

    st.subheader("Which signals worked better?")
    evaluated = data[data["outcome_20d"].notna()].copy()
    if len(evaluated) >= 5:
        diagnostics = []
        for label, field, condition in [
            ("1-week return positive", "return_1w", lambda s: s > 0),
            ("1-month return positive", "return_1m", lambda s: s > 0),
            ("Above 50-day average", "price_vs_50d", lambda s: s > 0),
            ("Above 200-day average", "above_200d", lambda s: s > 0),
            ("Volume expanding", "volume_change", lambda s: s > 0),
            ("Within 2% of 20-day high", "distance_20d_high", lambda s: s >= -2)
        ]:
            valid = evaluated.dropna(subset=[field, "outcome_20d"])
            for group_name, mask in [("Signal present", condition(valid[field])), ("Signal absent", ~condition(valid[field]))]:
                subset = valid[mask]
                if len(subset):
                    diagnostics.append({
                        "Indicator": label, "Group": group_name, "Samples": len(subset),
                        "20-day win rate %": round((subset["outcome_20d"] > 0).mean()*100, 1),
                        "Average 20-day return %": round(subset["outcome_20d"].mean(), 2)
                    })
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), hide_index=True, use_container_width=True)
            st.caption("Descriptive comparisons only. These indicators overlap and are not causal proof. Treat small sample sizes cautiously.")
    else:
        st.info("At least five evaluated 20-day outcomes are needed before the indicator comparison becomes useful.")

    st.subheader("Prediction history")
    display_cols = ["ticker", "prediction_type", "prediction_date", "price", "overall_score", "early_mover_score", "outcome_5d", "outcome_20d", "outcome_60d"]
    st.dataframe(data[display_cols].round(2), use_container_width=True, hide_index=True)
    st.download_button(
        "Download prediction history backup (CSV)",
        history.to_csv(index=False).encode("utf-8"),
        file_name="asx_prediction_history_backup.csv",
        mime="text/csv"
    )
    st.warning("Storage note: this prototype uses a local SQLite database beside app.py. Streamlit Community Cloud can reset local files during restarts or redeployments. Download the CSV backup regularly; for dependable long-term cloud history, the next step is connecting a persistent database.")

MASTER_TICKERS = load_master_tickers()
with st.sidebar:
    page=st.radio("Page",["Single-Share Analysis","Automatic Share Scanner","Early-Mover Tracker","Lower-Risk Investing","Value Investing","Momentum Trading","High-Growth Shares","Long-Term Investing","Portfolio Planner","Prediction Performance"])
    st.divider()
    ticker_text=st.text_area("ASX tickers (commas or new lines)",", ".join(MASTER_TICKERS),height=160,
        help="Defaults to the bundled ASX company list. Edit or paste your own tickers if needed.")
    tickers=list(dict.fromkeys(x.strip().upper().replace(".AX","") for x in ticker_text.replace("\n",",").split(",") if x.strip()))
    scan_mode=st.selectbox("Scan mode",["Full-market scan","Quick promising-shares scan"])
    max_tickers=st.slider("Maximum tickers per scan",5,2000,min(2000,max(5,len(tickers))),5)
    tickers=tickers[:max_tickers]
    if scan_mode == "Quick promising-shares scan":
        prior=st.session_state.get("scan_results",pd.DataFrame())
        if isinstance(prior,pd.DataFrame) and not prior.empty and "Overall Score" in prior.columns:
            promising=prior[prior["Price"].notna()].sort_values("Overall Score",ascending=False,na_position="last")["Ticker"].head(150).tolist()
            if promising:
                tickers=promising
        else:
            tickers=[t for t in STARTER_TICKERS if t in tickers][:150] or tickers[:150]
    scan_label=st.selectbox("Scanner price history",["1 year","2 years","5 years"])
    scan_period={"1 year":"1y","2 years":"2y","5 years":"5y"}[scan_label]
    if st.button("Clear cached data"):
        st.cache_data.clear()
        st.session_state.pop("scan_results", None)
        st.session_state.pop("scan_period", None)
        st.session_state.pop("scan_time", None)
        st.session_state.pop("scan_report", None)

    force_full_scan = st.checkbox(
        "Force full rescan (refresh all tickers)",
        value=False,
        help="Normally, existing results are reused and only tickers not already analysed are scanned."
    )

    if st.button("Run / refresh multi-share scan",type="primary",use_container_width=True):
        if not tickers:
            st.error("Enter at least one ticker.")
        else:
            old_df = st.session_state.get("scan_results", pd.DataFrame())
            old_period = st.session_state.get("scan_period")
            period_changed = old_period != scan_period
            can_reuse = (
                not force_full_scan
                and not period_changed
                and isinstance(old_df, pd.DataFrame)
                and not old_df.empty
                and "Ticker" in old_df.columns
            )

            # Reuse previous rows for tickers that are still selected.
            if can_reuse:
                selected_set = set(tickers)
                # Reuse only rows with usable prices. Failed/empty results are retried on the next run.
                kept_df = old_df[old_df["Ticker"].isin(selected_set)].copy()
                if "Price" in kept_df.columns:
                    kept_df = kept_df[kept_df["Price"].notna()].copy()
                already_done = set(kept_df["Ticker"].astype(str))
                to_scan = [sym for sym in tickers if sym not in already_done]
            else:
                kept_df = pd.DataFrame()
                to_scan = list(tickers)

            with st.spinner(f"Scanning {len(to_scan)} ticker(s); reusing {len(kept_df)} existing result(s)…"):
                new_df = run_scan(to_scan, scan_period) if to_scan else pd.DataFrame()

                if kept_df.empty:
                    combined = new_df
                elif new_df.empty:
                    combined = kept_df
                else:
                    combined = pd.concat([kept_df, new_df], ignore_index=True, sort=False)

                # Preserve the user's ticker-list order in the stored results.
                if not combined.empty and "Ticker" in combined.columns:
                    order = {sym: i for i, sym in enumerate(tickers)}
                    combined["_order"] = combined["Ticker"].map(order)
                    combined = combined.sort_values("_order").drop(columns="_order").reset_index(drop=True)

                st.session_state["scan_results"] = combined
                recorded_count = record_prediction_snapshots(combined)
                st.session_state["last_prediction_record_count"] = recorded_count
                st.session_state["scan_period"] = scan_period
                st.session_state["scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.session_state["scan_report"] = {
                    "requested": len(tickers),
                    "reused": len(kept_df),
                    "newly_scanned": len(to_scan),
                    "usable": int(combined["Price"].notna().sum()) if "Price" in combined.columns else 0,
                    "without_usable_data": max(0, len(tickers) - (int(combined["Price"].notna().sum()) if "Price" in combined.columns else 0)),
                    "force_full": bool(force_full_scan or period_changed or not can_reuse),
                }

scan=st.session_state.get("scan_results",pd.DataFrame())
report = st.session_state.get("scan_report")
if report:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tickers requested", report["requested"])
    m2.metric("Existing results reused", report["reused"])
    m3.metric("Newly scanned", report["newly_scanned"])
    m4.metric("Results with usable data", report["usable"])
    if report["without_usable_data"]:
        st.warning(
            f'{report["without_usable_data"]} ticker(s) have no usable price data. '
            "They are retained in the results with a Scan Status explaining the failure. Some may be valid tickers temporarily blocked by Yahoo Finance."
        )
        if not scan.empty and "Scan Status" in scan.columns:
            failed = scan[scan["Price"].isna()][["Ticker", "Scan Status"]]
            st.download_button("Download missing / failed tickers CSV", failed.to_csv(index=False).encode(), file_name="asx_failed_tickers.csv", mime="text/csv")
    else:
        st.success("Every selected ticker has a result row. This does not guarantee every data field is available.")

def show_scan(score_col,description):
    st.write(description)
    if scan.empty:
        st.info("No scan results yet. Use **Run / refresh multi-share scan** in the sidebar.")
        return
    d=scan.copy()
    if score_col=="Lower-Risk Score":
        d=d[(d["Above 200D %"].fillna(-999)>0)&(d["Return 1M %"].fillna(-999)>-10)].copy()
        d["Lower-Risk Score"]=(d["Momentum Score"].fillna(50)*.5+d["Value Score"].fillna(50)*.25+(100-d["Return 1M %"].abs().clip(0,100))*.25).round(1)
    elif score_col=="Long-Term Score":
        d["Long-Term Score"]=((d["Return 12M %"].fillna(0).clip(-50,50)+50)*.6+d["Value Score"].fillna(50)*.2+d["Growth Score"].fillna(50)*.2).round(1)
    if d.empty:
        st.warning("No shares met this screen. Try a broader list or scan again later.")
        return
    d=d.sort_values(score_col,ascending=False,na_position="last")
    st.caption(f"Shares with usable price data: {len(scan)}. Last scan: {st.session_state.get('scan_time','unknown')}.")
    st.caption("Scores are heuristics for comparison, not probabilities, forecasts, or buy/sell recommendations.")
    search=st.text_input("Filter by ticker or company",key="search_"+score_col)
    if search: d=d[d["Ticker"].str.contains(search,case=False,na=False)|d["Company"].str.contains(search,case=False,na=False)]
    cols=["Ticker","Company","Price","Return 1W %","Return 1M %","Return 3M %","Return 6M %","Return 12M %","Momentum Score","Value Score","Growth Score","Overall Score","Market Cap","P/E","Revenue Growth %","Profit Margin %","Fundamentals Status","Data Date","Scan Status"]
    if score_col not in cols: cols.insert(8,score_col)
    st.dataframe(d[[c for c in cols if c in d]].round(2),use_container_width=True,hide_index=True)
    st.download_button("Download results CSV",d.to_csv(index=False).encode(),file_name=f"asx_{score_col.lower().replace(' ','_')}.csv",mime="text/csv",key="csv_"+score_col)
    if not d.empty:
        chosen=st.selectbox("Inspect a share chart",d["Ticker"].tolist(),key="chart_"+score_col)
        hist=load_prices(chosen+".AX",scan_period)
        if not hist.empty: st.line_chart(hist["Close"],height=230)

if page=="Single-Share Analysis":
    st.header("Single-Share Analysis")
    with st.sidebar:
        raw=st.text_input("ASX ticker to analyse","BHP",key="single")
        period=st.selectbox("Single-share price history",["1y","2y","5y","10y"],index=2)
    ticker=raw.strip().upper()
    if ticker and "." not in ticker: ticker+=".AX"
    if not ticker: st.info("Enter a ticker."); st.stop()
    with st.spinner(f"Loading {ticker}…"): hist,info,income,balance,cash=load_company(ticker,period)
    if hist is None or hist.empty: st.error("No price history returned. Check ticker or try later."); st.stop()
    close=hist["Close"].dropna(); price=num(close.iloc[-1]); prev=num(close.iloc[-2]) if len(close)>1 else None
    change=price/prev-1 if price is not None and prev not in (None,0) else None
    name=info.get("longName") or info.get("shortName") or ticker
    st.subheader(f"{name} ({ticker})")
    a,b,c,d=st.columns(4)
    a.metric("Latest available close",fmt(price,"money"),fmt(change,"pct") if change is not None else None)
    b.metric("Market capitalisation",fmt(info.get("marketCap"),"large")); c.metric("Sector",info.get("sector") or "N/A"); d.metric("Currency",info.get("currency") or "N/A")
    ocf=stmt(cash,["Operating Cash Flow","Total Cash From Operating Activities","Cash Flow From Continuing Operating Activities"])
    shares=num(info.get("sharesOutstanding")) or num(info.get("impliedSharesOutstanding"))
    cash_ps=ocf/shares if ocf is not None and shares and shares>0 else None
    price_cash=price/cash_ps if price is not None and cash_ps not in (None,0) else None
    yield_raw=num(info.get("dividendYield")); yield_pct=yield_raw*100 if yield_raw is not None else None
    pe=num(info.get("trailingPE")); equity=stmt(balance,["Stockholders Equity","Total Stockholder Equity","Total Equity Gross Minority Interest"])
    equity_ps=equity/shares if equity is not None and shares and shares>0 else None
    eps_auto=num(info.get("trailingEps")); f_eps=num(info.get("forwardEps"))
    price_tab,inputs_tab,score_tab,data_tab=st.tabs(["Price & technicals","Checklist inputs","Rule scorecard","Raw data"])
    with price_tab:
        chart=hist[["Close"]].copy(); chart["50-day average"]=chart["Close"].rolling(50).mean(); chart["200-day average"]=chart["Close"].rolling(200).mean()
        st.line_chart(chart,use_container_width=True)
        delta=close.diff(); gain=delta.clip(lower=0).rolling(14).mean(); loss=(-delta.clip(upper=0)).rolling(14).mean()
        rsi=100-100/(1+gain/loss.replace(0,np.nan)); macd=close.ewm(span=12,adjust=False).mean()-close.ewm(span=26,adjust=False).mean(); signal=macd.ewm(span=9,adjust=False).mean()
        x,y,z,w=st.columns(4); x.metric("RSI (14)",fmt(rsi.iloc[-1] if len(rsi) else None)); y.metric("50-day average",fmt(chart["50-day average"].iloc[-1],"money")); z.metric("200-day average",fmt(chart["200-day average"].iloc[-1],"money")); w.metric("MACD vs signal","Positive" if len(macd) and macd.iloc[-1]>signal.iloc[-1] else "Negative")
        st.info("3-point trend-line sentiment remains a manual assessment in this version.")
    with inputs_tab:
        auto=[("Operating cash flow",fmt(ocf,"large")),("Shares outstanding",fmt(shares)),("Operating cash flow per share",fmt(cash_ps,"money")),("Price / cash ratio",fmt(price_cash)),("Dividend yield",fmt(yield_raw,"pct")),("Trailing P/E",fmt(pe)),("Total equity",fmt(equity,"large")),("Equity per share",fmt(equity_ps,"money")),("Trailing EPS",fmt(eps_auto,"money")),("Forward EPS estimate",fmt(f_eps,"money"))]
        st.dataframe(pd.DataFrame(auto,columns=["Metric","Value"]),hide_index=True,use_container_width=True)
        st.markdown("### Manual checklist inputs"); st.caption("Use current company reports. Leave unverified items unassessed.")
        l,r=st.columns(2)
        with l:
            sentiment=st.selectbox("3PTL sentiment",["Not assessed","Positive / above buy line","Negative / sell line breached","Wait / needs uptick"])
            upturn=st.selectbox("Positive upturn since latest financial report?",["Not assessed","Yes","No"])
            mortgage=st.number_input("Mortgage reference rate (%)",0.0,30.0,6.0,.25)
            hist_pe=st.selectbox("P/E vs historical period",["Not assessed","Lowest","Neither / unknown","Highest","Unprofitable / unavailable"])
            equity_trend=st.selectbox("Equity trend",["Not assessed","Consistently increasing","Not consistently increasing","Insufficient history"])
            audit=st.selectbox("Latest audit opinion",["Not checked","Clean / unmodified","Qualified / adverse / disclaimer","Emphasis of matter / going concern concern","Other / unclear"])
            founder=st.selectbox("Founder/board member owns at least 10%?",["Not assessed","Yes","No"])
        with r:
            eps=st.number_input("Current EPS (override if needed)",value=float(eps_auto or 0),step=.01,format="%.4f")
            future_eps=st.number_input("Future EPS estimate (0 if unknown)",value=float(f_eps or 0),step=.01,format="%.4f")
            hurdle=st.number_input("Market hurdle rate (%)",.1,50.0,6.1,.1)
            star=st.selectbox("Verified external Star Stock score",["Not assessed","0","0.5","1","1.5","2"])
            external_iv=st.number_input("External intrinsic value per share (0 if unknown)",min_value=0.0,value=0.0,step=.1)
            health_trend=st.selectbox("Verified financial-health trend",["Not assessed","Increasing","Stable","Deteriorating","Other / unclear"])
            health_rating=st.selectbox("Verified financial-health rating",["Not assessed","Strong","Satisfactory","Other / weak"])
            commodity=st.selectbox("Relevant commodity trend?",["Not assessed / not applicable","Yes","No"])
    with score_tab:
        rows=[]
        def add(rule,result,score=None,notes=""): rows.append({"Rule":rule,"Result":result,"Score":score if score is not None else "—","Notes":notes})
        add("3PTL sentiment gate",sentiment,1 if sentiment=="Positive / above buy line" else (0 if sentiment=="Negative / sell line breached" else None),"Manual chart assessment")
        add("Positive upturn since latest report",upturn,1 if upturn=="Yes" else (0 if upturn=="No" else None))
        add("Operating cash flow positive","Positive" if ocf is not None and ocf>0 else ("Non-positive" if ocf is not None else "Missing"))
        add("Price / cash ratio below 7","Pass" if price_cash is not None and 0<price_cash<7 else ("Fail" if price_cash is not None else "Missing"),2 if price_cash is not None and 0<price_cash<7 else (0 if price_cash is not None else None))
        add("Dividend yield above mortgage rate","Pass" if yield_pct is not None and yield_pct>mortgage else ("Fail" if yield_pct is not None else "Missing"),1 if yield_pct is not None and yield_pct>mortgage else (0 if yield_pct is not None else None))
        add("P/E below dividend yield","Pass" if pe is not None and yield_pct is not None and pe<yield_pct else ("Fail" if pe is not None and yield_pct is not None else "Missing"),1 if pe is not None and yield_pct is not None and pe<yield_pct else (0 if pe is not None and yield_pct is not None else None))
        add("P/E lowest in historical period",hist_pe,{"Lowest":2,"Highest":-1,"Unprofitable / unavailable":0}.get(hist_pe),"Manual historical check")
        add("Equity consistently increasing",equity_trend,1 if equity_trend=="Consistently increasing" else (0 if equity_trend=="Not consistently increasing" else None))
        add("Price below equity per share","Pass" if price is not None and equity_ps is not None and price<equity_ps else ("Fail" if price is not None and equity_ps is not None else "Missing"),1 if price is not None and equity_ps is not None and price<equity_ps else (0 if price is not None and equity_ps is not None else None))
        add("Price <= 1.3 x equity per share","Pass" if price is not None and equity_ps is not None and price<=1.3*equity_ps else ("Fail" if price is not None and equity_ps is not None else "Missing"),1 if price is not None and equity_ps is not None and price<=1.3*equity_ps else (0 if price is not None and equity_ps is not None else None))
        growth_pe=((future_eps-eps)/eps)/pe if eps>0 and future_eps>0 and pe is not None and pe>0 else None
        add("EPS growth / P-E ratio",fmt(growth_pe),2 if growth_pe is not None and growth_pe>1.5 else (-1 if growth_pe is not None and growth_pe<0 else (0 if growth_pe is not None else None)),"Estimate-based; verify inputs")
        iv1=eps/.195 if eps>0 else None; iv2=future_eps/(hurdle/100) if future_eps>0 and hurdle>0 else None
        add("Intrinsic value #1: current EPS / 19.5%",fmt(iv1,"money"))
        add("Price below intrinsic value #1","Pass" if price is not None and iv1 is not None and price<iv1 else ("Fail" if price is not None and iv1 is not None else "Missing"),1 if price is not None and iv1 is not None and price<iv1 else (0 if price is not None and iv1 is not None else None))
        add(f"Intrinsic value #2: future EPS / {hurdle:.1f}%",fmt(iv2,"money"),notes="Check the hurdle rate against an authoritative source")
        add("Price below intrinsic value #2","Pass" if price is not None and iv2 is not None and price<iv2 else ("Fail" if price is not None and iv2 is not None else "Missing"),1 if price is not None and iv2 is not None and price<iv2 else (0 if price is not None and iv2 is not None and price<iv2 else None))
        add("Intrinsic value #2 > 2 x share price","Pass" if price is not None and iv2 is not None and iv2>2*price else ("Fail" if price is not None and iv2 is not None else "Missing"),1 if price is not None and iv2 is not None and iv2>2*price else (0 if price is not None and iv2 is not None else None))
        add("External Star Stock score",star,float(star) if star!="Not assessed" else None,"Manual verified input")
        add("Price below external intrinsic value","Pass" if external_iv>0 and price is not None and price<external_iv else ("Fail" if external_iv>0 and price is not None else "Not assessed"),1 if external_iv>0 and price is not None and price<external_iv else (0 if external_iv>0 and price is not None else None))
        add("External financial-health trend",health_trend,{"Increasing":2,"Stable":1,"Deteriorating":-1}.get(health_trend))
        add("External financial-health rating",health_rating,1 if health_rating in ["Strong","Satisfactory"] else (0 if health_rating=="Other / weak" else None))
        add("Founder/board ownership >= 10%",founder,2 if founder=="Yes" else (0 if founder=="No" else None))
        audit_result={"Not checked":"Not checked","Clean / unmodified":"No warning recorded","Qualified / adverse / disclaimer":"FAIL — review carefully","Emphasis of matter / going concern concern":"WARNING — manual review","Other / unclear":"Manual review required"}[audit]
        add("Audit opinion",audit_result,notes="Read the latest audit report"); add("Commodity trend",commodity,notes="Manual check where relevant")
        checklist=pd.DataFrame(rows); st.dataframe(checklist,hide_index=True,use_container_width=True)
        scores=[float(v) for v in checklist["Score"] if isinstance(v,(int,float,np.integer,np.floating)) and not pd.isna(v)]
        if scores:
            m1,m2=st.columns(2); m1.metric("Sum of assessed points",f"{sum(scores):.1f}"); m2.metric("Number of scored rules",len(scores))
        if sentiment=="Negative / sell line breached": st.error("Sentiment gate failed. Treat as a screening flag, not an automatic trade.")
        elif sentiment=="Wait / needs uptick": st.warning("Wait state: verify an uptick before treating sentiment as positive.")
        if audit=="Qualified / adverse / disclaimer": st.error("Audit warning selected. Read the latest report before proceeding.")
        elif audit=="Emphasis of matter / going concern concern": st.warning("Review audit wording and going-concern disclosures carefully.")
        st.download_button("Download checklist CSV",checklist.to_csv(index=False).encode(),file_name=f"{ticker.replace('.','_')}_checklist.csv",mime="text/csv")
    with data_tab:
        for label,frame in [("Income statement",income),("Balance sheet",balance),("Cash flow",cash)]:
            st.markdown(f"### {label}")
            if frame is not None and not frame.empty: st.dataframe(frame,use_container_width=True)
            else: st.info("No statement data returned.")
        st.markdown("### Recent prices"); st.dataframe(hist.tail(30).sort_index(ascending=False),use_container_width=True)
        st.download_button("Download price history CSV",hist.to_csv().encode(),file_name=f"{ticker.replace('.','_')}_prices.csv",mime="text/csv")

elif page=="Automatic Share Scanner":
    st.header(page); show_scan("Overall Score","Combined ranking: momentum 40%, value 30%, growth 30%, using available fields.")
elif page=="Early-Mover Tracker":
    st.header("🚀 Early-Mover Tracker")
    st.write("Ranks shares for early technical signs of a possible move: improving short-term momentum, expanding volume, price strength versus moving averages, and proximity to recent highs.")
    st.warning("This is a screening heuristic, not a probability or prediction. A high score can still precede a sharp fall; small-cap shares can be illiquid and volatile.")
    if scan.empty:
        st.info("Run a full-market or quick promising-shares scan first. This tracker uses the latest scanner results.")
    else:
        movers=scan.copy()
        movers["Early-Mover Score"]=movers.apply(early_mover_score,axis=1)
        movers=movers[movers["Price"].notna()].sort_values("Early-Mover Score",ascending=False,na_position="last")
        movers["Signal Tier"]=np.select([movers["Early-Mover Score"]>=80,movers["Early-Mover Score"]>=65],["Priority review","Watch closely"],default="Lower priority")
        st.caption("Score weights: 1-week momentum 15%, 1-month momentum 15%, 3-month momentum 10%, price vs 50-day average 10%, price vs 200-day average 10%, volume expansion 20%, distance from 20-day high 10%, and avoiding extreme short-term extension 10%. Missing fields are excluded and weights renormalised.")
        st.caption(f"Latest scan: {st.session_state.get('scan_time','unknown')}. Use each share's Data Date to check freshness.")
        cols=["Ticker","Company","Price","Early-Mover Score","Signal Tier","Return 1W %","Return 1M %","Return 3M %","Price vs 50D %","Above 200D %","Volume Change %","Distance from 20D High %","Market Cap","Data Date","Scan Status"]
        st.dataframe(movers[[c for c in cols if c in movers.columns]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download early-mover rankings CSV",movers.to_csv(index=False).encode(),file_name="asx_early_mover_rankings.csv",mime="text/csv")
        selected=st.multiselect("Add shares to your tracker",movers["Ticker"].tolist(),key="early_mover_select")
        if st.button("Save selected shares to tracker",type="primary",disabled=not selected):
            watch=st.session_state.get("early_mover_watchlist",[])
            now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            indexed=movers.set_index("Ticker",drop=False)
            for sym in selected:
                row=indexed.loc[sym]
                if isinstance(row,pd.DataFrame): row=row.iloc[0]
                watch=[item for item in watch if item.get("Ticker")!=sym]
                watch.append({"Ticker":sym,"Company":row.get("Company",sym),"Added At":now,"Score at Add":row.get("Early-Mover Score"),
                    "Price at Add":row.get("Price"),"Data Date at Add":row.get("Data Date"),"Status":"Tracking"})
            st.session_state["early_mover_watchlist"]=watch
            st.success(f"Saved {len(selected)} share(s) to the tracker for this app session.")
        watch=st.session_state.get("early_mover_watchlist",[])
        st.subheader("Tracked shares")
        if watch:
            wd=pd.DataFrame(watch)
            indexed=movers.set_index("Ticker",drop=False)
            wd["Latest Price"]=wd["Ticker"].map(indexed["Price"])
            wd["Latest Score"]=wd["Ticker"].map(indexed["Early-Mover Score"])
            wd["Price Change Since Added %"]=(wd["Latest Price"]/wd["Price at Add"]-1)*100
            wd["Latest Data Date"]=wd["Ticker"].map(indexed["Data Date"])
            st.dataframe(wd.round(2),use_container_width=True,hide_index=True)
            st.download_button("Download tracker CSV",wd.to_csv(index=False).encode(),file_name="asx_early_mover_tracker.csv",mime="text/csv")
            remove=st.multiselect("Remove shares from tracker",wd["Ticker"].tolist(),key="early_mover_remove")
            if st.button("Remove selected tracked shares",disabled=not remove):
                st.session_state["early_mover_watchlist"]=[item for item in watch if item.get("Ticker") not in remove]
                st.rerun()
        else:
            st.caption("No shares tracked yet. Select candidates above and save them.")
        st.info("The tracker currently lives in Streamlit session state. Download its CSV to keep a copy; it may reset if the app session restarts. Automatic ASX-announcement/catalyst analysis is a later step.")

elif page=="Lower-Risk Investing":
    st.header(page); show_scan("Lower-Risk Score","A relative screen for shares above their 200-day average and without a severe monthly decline. These are not guaranteed to be safe.")
elif page=="Value Investing":
    st.header(page); show_scan("Value Score","A rough screen based on available P/E and profit margin. Low valuation can reflect genuine business problems.")
elif page=="Momentum Trading":
    st.header(page); show_scan("Momentum Score","Ranks recent returns, long trend and volume change. Momentum can reverse quickly.")
elif page=="High-Growth Shares":
    st.header(page); show_scan("Growth Score","Ranks available revenue growth and profit margin. Small-cap data can be incomplete.")
elif page=="Long-Term Investing":
    st.header(page); show_scan("Long-Term Score","Emphasises 12-month performance plus value/growth fields. Past performance does not predict future returns.")
elif page=="Portfolio Planner":
    st.header(page); st.write("Educational allocation calculator, not personalised financial advice.")
    amount=st.number_input("Portfolio amount (AUD)",min_value=0.0,value=10000.0,step=500.0)
    low=st.slider("Lower-volatility diversified investments (%)",0,100,40)
    growth=st.slider("Growth investments (%)",0,100,30)
    high=st.slider("Higher-risk / speculative investments (%)",0,100,10)
    cash=st.slider("Cash (%)",0,100,20)
    total=low+growth+high+cash; st.metric("Total allocation",f"{total}%")
    if total==100:
        plan=pd.DataFrame({"Category":["Lower-volatility diversified investments","Growth investments","Higher-risk / speculative investments","Cash"],"Allocation %":[low,growth,high,cash],"Amount AUD":[amount*low/100,amount*growth/100,amount*high/100,amount*cash/100]})
        st.dataframe(plan,hide_index=True,use_container_width=True)
        st.download_button("Download portfolio plan CSV",plan.to_csv(index=False).encode(),file_name="portfolio_plan.csv",mime="text/csv")
    else: st.warning("Adjust the sliders until the allocation totals 100%.")
    st.info("Diversification does not eliminate risk. Consider fees, tax, emergency savings, time horizon and the possibility of loss.")

elif page=="Prediction Performance":
    prediction_performance_page()

st.divider()
st.subheader("Data and limitations")
st.markdown("- Data provider: Yahoo Finance via yfinance; availability, freshness and definitions vary.\n- The built-in list is a sample, not the full ASX. Paste your own ticker list for wider coverage.\n- Small caps may have missing data, low liquidity, wide spreads and high volatility.\n- Scores are rule-based comparisons, not predictions, probabilities, or buy/sell recommendations.\n- Verify ASX announcements, audited reports, debt, cash flow, dilution, liquidity and trading halts.\n- The app does not connect to a broker or place trades.")
st.caption("Personal-use research prototype. Not affiliated with QAV, Stock Doctor, Yahoo Finance or ASX.")

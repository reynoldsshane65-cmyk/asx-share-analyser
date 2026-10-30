import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sqlite3
import json
import hashlib
import tempfile
from pathlib import Path

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
                wanted = [c for c in ["Open","High","Low","Close","Volume"] if c in h.columns]
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
        day_high=num(h["High"].iloc[-1]) if "High" in h else None
        day_low=num(h["Low"].iloc[-1]) if "Low" in h else None
        day_open=num(h["Open"].iloc[-1]) if "Open" in h else None
        day_range=(day_high-day_low)/float(close.iloc[-1])*100 if day_high is not None and day_low is not None and float(close.iloc[-1]) else np.nan
        close_location=(float(close.iloc[-1])-day_low)/(day_high-day_low)*100 if day_high is not None and day_low is not None and day_high>day_low else np.nan
        day_gap=(day_open/float(close.iloc[-2])-1)*100 if day_open is not None and len(close)>1 and float(close.iloc[-2]) else np.nan
        volatility=close.pct_change().tail(20).std()*100 if len(close)>=21 else np.nan

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
            "Daily range %": day_range, "Close in daily range %": close_location,
            "Gap %": day_gap, "20D volatility %": volatility,
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


# ---------------- Self-learning prediction engine ----------------
DB_PATH = Path(__file__).with_name("asx_prediction_history.sqlite")
FEATURES = ["Return 1W %", "Return 1M %", "Return 3M %", "Price vs 50D %", "Above 200D %", "Volume Change %", "Distance from 20D High %", "Short-term extension", "Daily range %", "Close in daily range %", "Gap %", "20D volatility %"]
# Preserve the original Phase 3 factor balance while reserving 20% for four OHLCV features.
BASE_WEIGHTS = np.array([0.12, 0.12, 0.08, 0.08, 0.08, 0.16, 0.08, 0.08, 0.05, 0.05, 0.05, 0.05], dtype=float)

def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, company TEXT,
        signal_date TEXT NOT NULL, entry_price REAL NOT NULL, base_score REAL,
        components_json TEXT, features_json TEXT, weights_json TEXT,
        price_5d REAL, return_5d REAL, checked_5d TEXT,
        price_10d REAL, return_10d REAL, checked_10d TEXT,
        price_20d REAL, return_20d REAL, checked_20d TEXT,
        price_30d REAL, return_30d REAL, checked_30d TEXT,
        price_60d REAL, return_60d REAL, checked_60d TEXT,
        UNIQUE(ticker, signal_date))''')
    # Upgrade existing Phase 3 databases in place; preserve all saved predictions.
    existing = {row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
    for horizon in (10, 30):
        for suffix, kind in (("price", "REAL"), ("return", "REAL"), ("checked", "TEXT")):
            column = f"{suffix}_{horizon}d"
            if column not in existing:
                con.execute(f"ALTER TABLE predictions ADD COLUMN {column} {kind}")
    con.execute('''CREATE TABLE IF NOT EXISTS historical_learning (
        ticker TEXT NOT NULL, signal_date TEXT NOT NULL, entry_price REAL, features_json TEXT,
        return_5d REAL, return_10d REAL, return_20d REAL, return_30d REAL,
        PRIMARY KEY(ticker, signal_date))''')
    learning_columns={row[1] for row in con.execute("PRAGMA table_info(historical_learning)").fetchall()}
    if "entry_price" not in learning_columns:
        con.execute("ALTER TABLE historical_learning ADD COLUMN entry_price REAL")
    con.execute('''CREATE TABLE IF NOT EXISTS historical_learning_monthly (
        ticker TEXT NOT NULL, month_key TEXT NOT NULL, signal_date TEXT NOT NULL,
        features_json TEXT, return_20d REAL, PRIMARY KEY(ticker,month_key))''')
    con.execute("CREATE TABLE IF NOT EXISTS app_meta (meta_key TEXT PRIMARY KEY,meta_value TEXT)")
    migrated=con.execute("SELECT 1 FROM app_meta WHERE meta_key='monthly_learning_backfill_v1'").fetchone()
    if not migrated:
        con.execute("""INSERT OR IGNORE INTO historical_learning_monthly
            (ticker,month_key,signal_date,features_json,return_20d)
            SELECT ticker,substr(signal_date,1,7),signal_date,features_json,return_20d
            FROM historical_learning WHERE return_20d IS NOT NULL
            ORDER BY ticker,substr(signal_date,1,7),signal_date""")
        con.execute("INSERT INTO app_meta (meta_key,meta_value) VALUES ('monthly_learning_backfill_v1','done')")
    con.execute('''CREATE TABLE IF NOT EXISTS historical_prices (
        ticker TEXT NOT NULL, price_date TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY(ticker, price_date))''')
    price_columns={row[1] for row in con.execute("PRAGMA table_info(historical_prices)").fetchall()}
    for column in ("open","high","low"):
        if column not in price_columns: con.execute(f"ALTER TABLE historical_prices ADD COLUMN {column} REAL")
    con.execute('''CREATE TABLE IF NOT EXISTS historical_batch_jobs (
        job_key TEXT PRIMARY KEY, universe_json TEXT NOT NULL, next_index INTEGER NOT NULL DEFAULT 0,
        period TEXT NOT NULL, interval_days INTEGER NOT NULL, updated_at TEXT NOT NULL)''')
    con.execute('''CREATE TABLE IF NOT EXISTS historical_batch_failures (
        job_key TEXT NOT NULL, ticker TEXT NOT NULL, reason TEXT, updated_at TEXT NOT NULL,
        PRIMARY KEY(job_key,ticker))''')
    con.commit()
    return con

def sqlite_backup_bytes():
    """Create a consistent SQLite snapshot, including committed WAL data, for download."""
    if not DB_PATH.exists():
        return None
    with tempfile.TemporaryDirectory() as folder:
        backup_path=Path(folder)/"asx_prediction_history.sqlite"
        source=sqlite3.connect(str(DB_PATH),timeout=30)
        target=sqlite3.connect(str(backup_path),timeout=30)
        try:
            source.backup(target)
        finally:
            target.close(); source.close()
        return backup_path.read_bytes()

def render_database_backup(key_prefix):
    st.caption("This backup contains the saved prediction database, batch progress, cached historical OHLCV prices, and learning outcomes.")
    if st.button("Prepare SQLite database backup",key=f"{key_prefix}_prepare_db_backup"):
        try:
            st.session_state["sqlite_backup_bytes"]=sqlite_backup_bytes()
            st.session_state["sqlite_backup_filename"]=f"asx_prediction_history_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.sqlite"
        except Exception as exc:
            st.error(f"Could not prepare the database backup: {exc}")
    backup=st.session_state.get("sqlite_backup_bytes")
    if backup:
        st.download_button("Download SQLite backup",data=backup,file_name=st.session_state.get("sqlite_backup_filename","asx_prediction_history_backup.sqlite"),mime="application/vnd.sqlite3",key=f"{key_prefix}_download_db_backup")

def render_database_backup(key_prefix):
    st.caption("This backup contains the saved prediction database, batch progress, cached historical OHLCV prices, and learning outcomes.")
    if st.button("Prepare SQLite database backup",key=f"{key_prefix}_prepare_db_backup"):
        try:
            st.session_state["sqlite_backup_bytes"]=sqlite_backup_bytes()
            st.session_state["sqlite_backup_filename"]=f"asx_prediction_history_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.sqlite"
        except Exception as exc:
            st.error(f"Could not prepare the database backup: {exc}")
    backup=st.session_state.get("sqlite_backup_bytes")
    if backup:
        st.download_button("Download SQLite backup",data=backup,file_name=st.session_state.get("sqlite_backup_filename","asx_prediction_history_backup.sqlite"),mime="application/vnd.sqlite3",key=f"{key_prefix}_download_db_backup")

def component_scores(row):
    def val(k):
        x=num(row.get(k)); return x if x is not None and np.isfinite(x) else None
    w1,w4,w12,p50,p200,vol,high= [val(k) for k in FEATURES[:7]]
    extension = w1
    values = [
        np.clip(50+w1*5,0,100) if w1 is not None else np.nan,
        np.clip(50+w4*2,0,100) if w4 is not None else np.nan,
        np.clip(50+w12,0,100) if w12 is not None else np.nan,
        (100 if p50>0 else np.clip(50+p50*3,0,50)) if p50 is not None else np.nan,
        (100 if p200>0 else np.clip(50+p200,0,50)) if p200 is not None else np.nan,
        np.clip(50+vol*.35,0,100) if vol is not None else np.nan,
        np.clip(100+high*5,0,100) if high is not None else np.nan,
        (100 if 0<=extension<=12 else (65 if -3<=extension<0 or 12<extension<=20 else 25)) if extension is not None else np.nan,
        np.clip(80-val("Daily range %")*2,0,100) if val("Daily range %") is not None else np.nan,
        np.clip(val("Close in daily range %"),0,100) if val("Close in daily range %") is not None else np.nan,
        np.clip(50+val("Gap %")*4,0,100) if val("Gap %") is not None else np.nan,
        np.clip(100-val("20D volatility %")*4,0,100) if val("20D volatility %") is not None else np.nan,
    ]
    return [None if pd.isna(x) else float(x) for x in values]

def learned_weights():
    con=db_connect()
    # A full-universe history may contain millions of rows. Stream through them
    # and retain only the sums needed to calculate each indicator correlation.
    cursors=[con.execute("""WITH observations AS (
          SELECT h.ticker,h.signal_date,h.features_json,h.return_20d FROM historical_learning_monthly h
          WHERE h.return_20d IS NOT NULL
          UNION ALL
          SELECT p.ticker,p.signal_date,p.features_json,p.return_20d FROM predictions p WHERE p.return_20d IS NOT NULL
        ), monthly AS (
          SELECT features_json,return_20d,
                 ROW_NUMBER() OVER (PARTITION BY ticker,substr(signal_date,1,7) ORDER BY signal_date) AS month_row
          FROM observations
        ) SELECT features_json,return_20d FROM monthly WHERE month_row=1""")]
    stats={feature:[0,0.0,0.0,0.0,0.0,0.0] for feature in FEATURES}
    observations=0
    for cursor in cursors:
        while True:
            chunk=cursor.fetchmany(5000)
            if not chunk: break
            for features_json,target_value in chunk:
                target=num(target_value)
                if target is None: continue
                observations+=1
                try: features=json.loads(features_json or "{}")
                except Exception: features={}
                for feature in FEATURES:
                    value=num(features.get(feature))
                    if value is None or not np.isfinite(value): continue
                    item=stats[feature]; item[0]+=1; item[1]+=value; item[2]+=target
                    item[3]+=value*value; item[4]+=target*target; item[5]+=value*target
    con.close()
    if observations<30:
        return BASE_WEIGHTS.copy(), observations, False, []
    correlations=[]
    for feature in FEATURES:
        n,sx,sy,sxx,syy,sxy=stats[feature]
        vx=sxx-sx*sx/n if n else 0.0; vy=syy-sy*sy/n if n else 0.0
        cov=sxy-sx*sy/n if n else 0.0
        corr=cov/(vx*vy)**0.5 if n>=20 and vx>0 and vy>0 else 0.0
        correlations.append(float(corr) if np.isfinite(corr) else 0.0)
    # Blend empirical relationships conservatively with the original transparent weights.
    multipliers=np.clip(1+1.5*np.array(correlations),0.5,1.5)
    weights=BASE_WEIGHTS*multipliers
    weights=weights/weights.sum()
    return weights, observations, True, correlations

def score_with_weights(row, weights):
    comps=component_scores(row)
    usable=[(i,v) for i,v in enumerate(comps) if v is not None]
    if not usable: return np.nan
    denom=sum(weights[i] for i,_ in usable)
    return round(sum(v*weights[i] for i,v in usable)/denom,1) if denom else np.nan

def explain_early_mover_score(row, weights):
    """Return the visible indicator inputs and weighted contributions for one live score."""
    labels=["1-week momentum","1-month momentum","3-month momentum","Price vs 50-day average","Price vs 200-day average","Volume change","Distance from 20-day high","Short-term extension","Daily trading range","Close position in daily range","Opening gap","20-day volatility"]
    values=component_scores(row)
    observed=[row.get(name) for name in FEATURES[:7]]+[row.get("Return 1W %")]+[row.get(name) for name in FEATURES[8:]]
    rows=[]; usable_weight=0
    for i,(label,value,input_value) in enumerate(zip(labels,values,observed)):
        if value is None: continue
        usable_weight += float(weights[i])
        rows.append({"Indicator":label,"Observed value":input_value,"Indicator score (0–100)":round(value,1),"Weight":float(weights[i])})
    for item in rows:
        item["Weighted contribution"] = round(item["Indicator score (0–100)"]*item["Weight"]/usable_weight,2) if usable_weight else 0
    risks=[]
    for key,label in [("Return 1W %","Short-term move may be extended"),("Return 1M %","Recent monthly momentum is negative"),("Price vs 50D %","Price is below its 50-day average"),("Above 200D %","Price is below its 200-day average"),("Volume Change %","Volume confirmation is weak or declining"),("Distance from 20D High %","Price is far below its recent high")]:
        value=num(row.get(key))
        if value is not None and ((key=="Return 1W %" and value>20) or (key=="Return 1M %" and value<0) or (key in ("Price vs 50D %","Above 200D %") and value<0) or (key=="Volume Change %" and value<0) or (key=="Distance from 20D High %" and value<-10)):
            risks.append(f"{label} ({value:+.1f}%)")
    vol20=num(row.get("20D volatility %")); close_pos=num(row.get("Close in daily range %")); gap=num(row.get("Gap %"))
    if vol20 is not None and vol20>5: risks.append(f"Recent daily volatility is high ({vol20:.1f}%)")
    if close_pos is not None and close_pos<25: risks.append(f"Latest close is near the bottom of its daily range ({close_pos:.0f}%)")
    if gap is not None and gap<-3: risks.append(f"Latest session opened down sharply ({gap:.1f}%)")
    return pd.DataFrame(rows), risks

def save_predictions(frame, weights):
    con=db_connect(); saved=0
    for _,r in frame.iterrows():
        ticker=str(r.get("Ticker","")); price=num(r.get("Price")); date=str(r.get("Data Date", ""))[:10]
        if not ticker or price is None or price<=0 or not date: continue
        # Calculate component scores before using them in the feature record.
        # Previously `comps` was referenced one line too early, causing
        # UnboundLocalError whenever a prediction was saved.
        comps=component_scores(r)
        features={k:num(r.get(k)) for k in FEATURES}
        features[FEATURES[7]]=comps[7]
        try:
            cur=con.execute("INSERT OR IGNORE INTO predictions (ticker,company,signal_date,entry_price,base_score,components_json,features_json,weights_json) VALUES (?,?,?,?,?,?,?,?)",
                (ticker,str(r.get("Company",ticker)),date,float(price),num(r.get("Early-Mover Score")),json.dumps(comps),json.dumps(features),json.dumps(weights.tolist())))
            if cur.rowcount > 0: saved+=1
        except Exception: pass
    con.commit(); con.close(); return saved

def refresh_prediction_outcomes():
    con=db_connect()
    pending=pd.read_sql_query("SELECT * FROM predictions WHERE return_5d IS NULL OR return_10d IS NULL OR return_20d IS NULL OR return_30d IS NULL OR return_60d IS NULL",con)
    con.close(); updated=0
    if pending.empty: return updated
    progress=st.progress(0,text="Checking prediction outcomes…")
    for n,(_,r) in enumerate(pending.iterrows()):
        try:
            hist=yf.Ticker(str(r.ticker)+".AX").history(start=str(r.signal_date),auto_adjust=True)
            if hist is None or hist.empty or "Close" not in hist: continue
            hist=hist["Close"].dropna()
            # Entry date is the close used when the signal was recorded; count later sessions only.
            future=hist[hist.index.date > pd.Timestamp(r.signal_date).date()]
            changes={}
            for horizon in (5,10,20,30,60):
                if pd.isna(getattr(r,f"return_{horizon}d")) and len(future)>=horizon:
                    px=num(future.iloc[horizon-1]); ret=(px/float(r.entry_price)-1)*100 if px and r.entry_price else None
                    changes[horizon]=(px,ret,datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if changes:
                con=db_connect()
                for h,(px,ret,checked) in changes.items():
                    con.execute(f"UPDATE predictions SET price_{h}d=?, return_{h}d=?, checked_{h}d=? WHERE id=?",(px,ret,checked,int(r.id)))
                con.commit(); con.close(); updated+=1
        except Exception: pass
        progress.progress((n+1)/len(pending),text=f"Checking prediction {n+1}/{len(pending)}…")
    progress.empty(); return updated

def store_historical_learning(frame):
    """Persist historical outcomes so completed, earlier observations train future live scores."""
    if frame is None or frame.empty:
        return 0
    con=db_connect(); saved=0
    for _, row in frame.iterrows():
        features={key:num(row.get(key)) for key in FEATURES}
        comps=component_scores(row)
        features[FEATURES[7]]=comps[7]
        try:
            con.execute("""INSERT OR REPLACE INTO historical_learning
                (ticker,signal_date,entry_price,features_json,return_5d,return_10d,return_20d,return_30d)
                VALUES (?,?,?,?,?,?,?,?)""",
                (str(row.get("Ticker","")),str(row.get("Signal date",""))[:10],num(row.get("Entry price")),json.dumps(features),
                 num(row.get("Return 5d %")),num(row.get("Return 10d %")),num(row.get("Return 20d %")),num(row.get("Return 30d %"))))
            signal_date=str(row.get("Signal date",""))[:10]
            con.execute("""INSERT INTO historical_learning_monthly
                (ticker,month_key,signal_date,features_json,return_20d) VALUES (?,?,?,?,?)
                ON CONFLICT(ticker,month_key) DO UPDATE SET signal_date=excluded.signal_date,
                features_json=excluded.features_json,return_20d=excluded.return_20d
                WHERE excluded.signal_date<=historical_learning_monthly.signal_date""",
                (str(row.get("Ticker","")),signal_date[:7],signal_date,json.dumps(features),num(row.get("Return 20d %"))))
            saved+=1
        except Exception:
            continue
    con.commit(); con.close(); return saved

def save_historical_prices(ticker, hist):
    """Cache downloaded adjusted OHLCV data so completed batches are reusable."""
    if hist is None or hist.empty or "Close" not in hist:
        return 0
    records=[]
    for date,row in hist.iterrows():
        opn=num(row.get("Open")); high=num(row.get("High")); low=num(row.get("Low"))
        close=num(row.get("Close")); volume=num(row.get("Volume")) if "Volume" in hist.columns else None
        if close is not None:
            records.append((ticker,pd.Timestamp(date).strftime("%Y-%m-%d"),opn,high,low,close,volume))
    if not records: return 0
    con=db_connect()
    con.executemany("INSERT OR REPLACE INTO historical_prices (ticker,price_date,open,high,low,close,volume) VALUES (?,?,?,?,?,?,?)",records)
    con.commit(); con.close(); return len(records)

def load_historical_prices(ticker):
    con=db_connect()
    frame=pd.read_sql_query("SELECT price_date AS Date,open AS Open,high AS High,low AS Low,close AS Close,volume AS Volume FROM historical_prices WHERE ticker=? ORDER BY price_date",con,params=(ticker,))
    con.close()
    if frame.empty: return frame
    frame["Date"]=pd.to_datetime(frame["Date"])
    return frame.set_index("Date")

def process_historical_batch(batch_codes, job_key, interval, batch_number=None, batch_total=None):
    """Download, cache, evaluate, and checkpoint one small group of tickers."""
    failures=[]
    label=f"Downloading maximum available history for batch {batch_number}/{batch_total}…" if batch_number is not None else "Downloading maximum available history…"
    progress=st.progress(0,text=label)
    price_sets=download_price_batches(batch_codes,"max",batch_size=30)
    for idx,ticker in enumerate(batch_codes):
        try:
            prices=price_sets.get(ticker,pd.DataFrame())
            if prices is not None and not prices.empty:
                save_historical_prices(ticker,prices)
            else:
                prices=load_historical_prices(ticker)
            one=historical_backtest_frame(prices,ticker,int(interval))
            if one.empty:
                failures.append({"Ticker":ticker,"Reason":"No cached or online history long enough to calculate the 200-day trend and later outcomes."})
            else:
                store_historical_learning(one)
        except Exception as exc:
            failures.append({"Ticker":ticker,"Reason":str(exc)[:200]})
        con=db_connect()
        failure=next((item["Reason"] for item in failures if item["Ticker"]==ticker),None)
        if failure:
            con.execute("INSERT OR REPLACE INTO historical_batch_failures (job_key,ticker,reason,updated_at) VALUES (?,?,?,?)",(job_key,ticker,failure,datetime.now().isoformat(timespec="seconds")))
        else:
            con.execute("DELETE FROM historical_batch_failures WHERE job_key=? AND ticker=?",(job_key,ticker))
        con.commit(); con.close()
        progress.progress((idx+1)/max(1,len(batch_codes)),text=f"Processed {ticker} ({idx+1}/{len(batch_codes)} in batch)…")
    progress.empty()
    return failures

def load_historical_learning(tickers):
    """Load a representative display sample; all rows remain stored and feed training."""
    if not tickers: return pd.DataFrame()
    con=db_connect()
    if len(tickers)==len(MASTER_TICKERS) and set(tickers)==set(MASTER_TICKERS):
        where=""
        params=()
    else:
        where="WHERE ticker IN ("+",".join("?" for _ in tickers)+")"
        params=tuple(tickers)
    total=int(con.execute("SELECT COUNT(*) FROM historical_learning "+where,params).fetchone()[0])
    sql="""WITH ranked AS (
        SELECT ticker,signal_date,entry_price,features_json,return_5d,return_10d,return_20d,return_30d,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY signal_date) AS rn,
               COUNT(*) OVER (PARTITION BY ticker) AS ticker_n
        FROM historical_learning """+where+"""
    ) SELECT ticker,signal_date,entry_price,features_json,return_5d,return_10d,return_20d,return_30d
      FROM ranked WHERE rn=1 OR rn % CASE WHEN ticker_n>30 THEN CAST((ticker_n+29)/30 AS INTEGER) ELSE 1 END=0"""
    data=pd.read_sql_query(sql,con,params=params)
    con.close()
    if data.empty: return pd.DataFrame()
    rows=[]
    for _,record in data.iterrows():
        try: features=json.loads(record.get("features_json") or "{}")
        except Exception: features={}
        row={**features,"Ticker":record["ticker"],"Signal date":record["signal_date"],"Entry price":record["entry_price"]}
        row["Score"]=score_with_weights(row,BASE_WEIGHTS)
        for h in (5,10,20,30): row[f"Return {h}d %"]=record[f"return_{h}d"]
        rows.append(row)
    result=pd.DataFrame(rows)
    result.attrs["total_observations"]=total
    result.attrs["display_sampled"]=total>len(result)
    return result

def batch_job_state(tickers, interval):
    universe=json.dumps(list(tickers),separators=(",",":"))
    job_key=hashlib.sha1((universe+"|max|"+str(interval)).encode("utf-8")).hexdigest()[:20]
    con=db_connect()
    record=con.execute("SELECT universe_json,next_index FROM historical_batch_jobs WHERE job_key=?",(job_key,)).fetchone()
    if record is None:
        con.execute("INSERT INTO historical_batch_jobs (job_key,universe_json,next_index,period,interval_days,updated_at) VALUES (?,?,?,?,?,?)",
                    (job_key,universe,0,"max",int(interval),datetime.now().isoformat(timespec="seconds")))
        con.commit(); next_index=0
    else: next_index=int(record[1])
    con.close()
    return job_key,next_index

def set_batch_job_index(job_key, tickers, interval, next_index):
    con=db_connect()
    con.execute("UPDATE historical_batch_jobs SET next_index=?,updated_at=? WHERE job_key=?",
                (int(next_index),datetime.now().isoformat(timespec="seconds"),job_key))
    con.commit(); con.close()

def historical_backtest_frame(hist, ticker, interval=5):
    """Create a simple historical walk-forward sample using only trailing data at each date."""
    if hist is None or hist.empty or "Close" not in hist:
        return pd.DataFrame()
    close = hist["Close"].dropna().astype(float)
    volume = hist["Volume"].fillna(0).astype(float) if "Volume" in hist else pd.Series(0.0, index=close.index)
    horizons = (5, 10, 20, 30)
    rows = []
    # 200 sessions of prior context are required for the long-trend indicator.
    for i in range(200, len(close) - max(horizons), interval):
        past = close.iloc[:i+1]
        vpast = volume.reindex(past.index).fillna(0)
        price = float(past.iloc[-1])
        ma50 = float(past.tail(50).mean())
        ma200 = float(past.tail(200).mean())
        v20 = float(vpast.tail(20).mean())
        vprev = float(vpast.iloc[-40:-20].mean()) if len(vpast) >= 40 else 0.0
        day_open=num(hist["Open"].iloc[i]) if "Open" in hist else None
        day_high=num(hist["High"].iloc[i]) if "High" in hist else None
        day_low=num(hist["Low"].iloc[i]) if "Low" in hist else None
        day_range=(day_high-day_low)/price*100 if day_high is not None and day_low is not None and price else np.nan
        close_location=(price-day_low)/(day_high-day_low)*100 if day_high is not None and day_low is not None and day_high>day_low else np.nan
        day_gap=(day_open/float(close.iloc[i-1])-1)*100 if day_open is not None and i>0 and float(close.iloc[i-1]) else np.nan
        volatility=past.pct_change().tail(20).std()*100 if len(past)>=21 else np.nan
        row = {
            "Return 1W %": pct(past, 5), "Return 1M %": pct(past, 21),
            "Return 3M %": pct(past, 63), "Price vs 50D %": (price/ma50-1)*100 if ma50 else np.nan,
            "Above 200D %": (price/ma200-1)*100 if ma200 else np.nan,
            "Volume Change %": (v20/vprev-1)*100 if vprev else np.nan,
            "Daily range %":day_range,"Close in daily range %":close_location,
            "Gap %":day_gap,"20D volatility %":volatility,
            "Distance from 20D High %": (price/float(past.tail(20).max())-1)*100 if len(past)>=20 else np.nan,
            "Ticker": ticker, "Signal date": str(past.index[-1])[:10], "Entry price": price,
        }
        row["Score"] = score_with_weights(row, BASE_WEIGHTS)
        for horizon in horizons:
            future = float(close.iloc[i+horizon])
            row[f"Return {horizon}d %"] = (future/price-1)*100
        rows.append(row)
    return pd.DataFrame(rows)

def backtest_metrics(frame, horizon):
    col = f"Return {horizon}d %"
    values = pd.to_numeric(frame[col], errors="coerce").dropna()
    return {
        "Signals": len(values),
        "Positive return %": (values.gt(0).mean()*100) if len(values) else np.nan,
        "Average return %": values.mean() if len(values) else np.nan,
        "Median return %": values.median() if len(values) else np.nan,
        "Negative return %": values.lt(0).mean()*100 if len(values) else np.nan,
    }


MASTER_TICKERS = load_master_tickers()
with st.sidebar:
    page=st.radio("Page",["Single-Share Analysis","Automatic Share Scanner","Early-Mover Tracker","Prediction Lab","Phase 4 Backtest","Lower-Risk Investing","Value Investing","Momentum Trading","High-Growth Shares","Long-Term Investing","Portfolio Planner"])
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
    st.caption("Scores are heuristic research signals, not calibrated probabilities or buy/sell recommendations. Prediction outcomes require saved signals and later price checks.")
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
        learn_w, learned_n, learning_active, learn_corr = learned_weights()
        movers["Early-Mover Score"]=movers.apply(lambda row: score_with_weights(row, learn_w),axis=1)
        if learning_active:
            st.success(f"Adaptive weights active: trained on {learned_n} completed 20-trading-day outcomes. The model blends learned relationships with the original weights.")
        else:
            st.info(f"Learning engine warming up: {learned_n}/30 completed 20-day outcomes. Original transparent weights remain in use until there is a minimum sample.")
        movers=movers[movers["Price"].notna()].sort_values("Early-Mover Score",ascending=False,na_position="last")
        movers["Signal Tier"]=np.select([movers["Early-Mover Score"]>=80,movers["Early-Mover Score"]>=65],["Priority review","Watch closely"],default="Lower priority")
        st.caption("The score combines momentum, 50/200-day trend, volume, distance from highs, opening gaps, intraday range, close position within the day's range, and recent volatility. Missing fields are excluded and weights renormalised; the exact live weights and contributions appear below.")
        st.caption(f"Latest scan: {st.session_state.get('scan_time','unknown')}. Use each share's Data Date to check freshness.")
        cols=["Ticker","Company","Price","Early-Mover Score","Signal Tier","Return 1W %","Return 1M %","Return 3M %","Price vs 50D %","Above 200D %","Volume Change %","Distance from 20D High %","Market Cap","Data Date","Scan Status"]
        st.dataframe(movers[[c for c in cols if c in movers.columns]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download early-mover rankings CSV",movers.to_csv(index=False).encode(),file_name="asx_early_mover_rankings.csv",mime="text/csv")
        explain_ticker=st.selectbox("Explain a score",movers["Ticker"].tolist(),key="explain_early_mover")
        explain_row=movers[movers["Ticker"]==explain_ticker].iloc[0]
        factor_table,risk_notes=explain_early_mover_score(explain_row,learn_w)
        st.markdown(f"**{explain_ticker} score: {explain_row['Early-Mover Score']:.1f}/100** — indicator scores are combined using the weights shown. This is a ranking score, not a likelihood.")
        st.dataframe(factor_table,use_container_width=True,hide_index=True)
        st.write("**Factors that may weaken the setup:**", "; ".join(risk_notes) if risk_notes else "No major technical warning from these indicators; missing announcements, dilution, liquidity, halts, and sudden news can still reverse a move.")
        prediction_selection=st.multiselect("Record these as formal predictions",movers["Ticker"].tolist(),key="prediction_select")
        if st.button("Save selected predictions to history",type="primary",disabled=not prediction_selection):
            chosen=movers[movers["Ticker"].isin(prediction_selection)].copy()
            saved=save_predictions(chosen,learn_w)
            st.success(f"Added {saved} new prediction(s). Existing ticker/date duplicates are ignored.")
            st.caption("Predictions are recorded at the latest available closing price and data date, not at a live executable price.")
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

elif page=="Prediction Lab":
    st.header("🧠 Prediction Lab — accuracy and learning")
    st.write("Saved predictions are measured after 5, 10, 20, 30 and 60 subsequent trading sessions. The learning engine combines completed live predictions with completed 20-day observations from Phase 4 historical backtests to adjust indicator weights for future Early-Mover scores.")
    st.warning("This is a research experiment, not a proven forecasting model. It needs a meaningful number of saved predictions across different market conditions. It does not place trades.")
    a,b=st.columns([1,3])
    if a.button("Refresh outcomes from Yahoo Finance",type="primary"):
        with st.spinner("Checking saved predictions against later prices…"):
            count=refresh_prediction_outcomes()
        st.success(f"Updated outcomes for {count} prediction(s).")
    con=db_connect(); predictions=pd.read_sql_query("SELECT * FROM predictions ORDER BY signal_date DESC,id DESC",con); con.close()
    if predictions.empty:
        st.info("No predictions recorded yet. Go to Early-Mover Tracker, select shares and choose 'Save selected predictions to history'.")
    else:
        total=len(predictions)
        st.subheader("Prediction performance")
        metrics=st.columns(6); metrics[0].metric("Predictions recorded",total)
        for col,h in zip(metrics[1:],(5,10,20,30,60)):
            vals=pd.to_numeric(predictions[f"return_{h}d"],errors="coerce").dropna()
            wins=int((vals>0).sum()); count=len(vals)
            col.metric(f"{h}-day positive rate",f"{wins/count*100:.1f}%" if count else "Not ready",help=f"{wins} positive outcomes out of {count} completed predictions")
        summaries=[]
        for h in (5,10,20,30,60):
            vals=pd.to_numeric(predictions[f"return_{h}d"],errors="coerce").dropna()
            summaries.append({"Horizon":f"{h} trading days","Completed":len(vals),"Positive %":round((vals>0).mean()*100,1) if len(vals) else np.nan,"Average return %":round(vals.mean(),2) if len(vals) else np.nan,"Median return %":round(vals.median(),2) if len(vals) else np.nan})
        st.dataframe(pd.DataFrame(summaries),use_container_width=True,hide_index=True)
        st.subheader("What the system is learning")
        weights,n,active,corr=learned_weights()
        if active:
            learned=pd.DataFrame({"Indicator":FEATURES,"Original weight %":BASE_WEIGHTS*100,"Learned weight %":weights*100,"Correlation with 20-day return":corr})
            st.dataframe(learned.round(3),use_container_width=True,hide_index=True)
            st.caption("Training combines saved live predictions and historical backtest observations. It uses about one matured 20-day outcome per ticker per month to reduce overlap; at least 30 such observations are required. Correlations are descriptive, not causal, and weights are blended conservatively with the base weights.")
        else:
            st.info(f"The adaptive scoring is not active yet. {n}/30 completed 20-day outcomes are available. Until then, the original weights are used.")
        st.subheader("Individual predictions")
        view_cols=["ticker","company","signal_date","entry_price","base_score","price_5d","return_5d","price_10d","return_10d","price_20d","return_20d","price_30d","return_30d","price_60d","return_60d"]
        view=predictions[view_cols].copy()
        view.columns=["Ticker","Company","Signal date","Entry price","Signal score","5d price","5d return %","10d price","10d return %","20d price","20d return %","30d price","30d return %","60d price","60d return %"]
        st.dataframe(view.round(3),use_container_width=True,hide_index=True)
        st.download_button("Download complete prediction history CSV",predictions.to_csv(index=False).encode(),file_name="asx_prediction_history.csv",mime="text/csv")
        if st.button("Delete all saved prediction history",type="secondary"):
            st.session_state["confirm_delete_predictions"]=True
        if st.session_state.get("confirm_delete_predictions"):
            st.error("This permanently deletes the local prediction database. Download a backup first if needed.")
            if st.button("Confirm delete prediction history"):
                con=db_connect(); con.execute("DELETE FROM predictions"); con.commit(); con.close()
                st.session_state["confirm_delete_predictions"]=False; st.rerun()
    st.caption(f"Local database: {DB_PATH.name}. Streamlit Community Cloud local storage may be reset by app rebuilds or restarts. Download the CSV backup regularly; a later upgrade can connect durable cloud storage.")
    render_database_backup("prediction_lab")

elif page=="Phase 4 Backtest":
    st.header("🧪 Phase 4 — Historical Backtest & Diagnostics")
    st.write("Replay the Phase 3 Early-Mover scoring rules at past dates, compare them with later prices, and add completed historical observations to the learning engine. Those past results then inform the adaptive weights used for future live Early-Mover scores.")
    st.warning("This downloads maximum available Yahoo Finance adjusted OHLCV history for current ASX codes, in resumable batches. It cannot include the full history of delisted companies or guarantee 50 years for newer listings. It evaluates momentum, trend, volume, opening gaps, daily range, close location and volatility; historical announcements and fundamentals are not in this free-data backtest. Fees, spreads and market impact are excluded, and historical adjustments may be revised. Results are descriptive, not proof of causes or future performance.")
    scope=st.radio("Backtest universe",["Entire listed universe","Selected shares"],horizontal=True,key="p4_scope")
    if scope=="Entire listed universe":
        backtest_codes=MASTER_TICKERS
        st.caption(f"Using all {len(backtest_codes):,} codes in the bundled current ASX list. This is not a historical list of every company that was ever listed.")
    else:
        backtest_codes=st.multiselect("Select shares for historical test",MASTER_TICKERS,default=MASTER_TICKERS[:min(10,len(MASTER_TICKERS))],key="p4_backtest_codes")
    interval=st.select_slider("Signal spacing (trading days)",[1,5,10,20],value=5,key="p4_interval")
    batch_size=st.select_slider("Shares to fetch per batch",[10,20,30,50,75],value=20,key="p4_batch_size")
    signal_threshold=st.slider("Score treated as a signal",40,90,65,key="p4_threshold")
    job_key,next_index=batch_job_state(backtest_codes,interval)
    done=min(next_index,len(backtest_codes))
    st.progress(done/max(1,len(backtest_codes)),text=f"Universe progress: {done:,} of {len(backtest_codes):,} shares processed")
    con=db_connect()
    cached_tickers,cached_rows=con.execute("SELECT COUNT(DISTINCT ticker),COUNT(*) FROM historical_prices").fetchone()
    con.close()
    st.caption(f"Local cache currently holds {cached_rows:,} daily OHLCV records across {cached_tickers:,} tickers. Completed batches and learning observations remain in the app's SQLite database between reruns; back up that database before replacing or migrating the app.")
    col_auto,col_single,col_reset=st.columns(3)
    run_all=col_auto.button("Run all remaining batches automatically",type="primary",key="p4_run_all",disabled=(done>=len(backtest_codes) or not backtest_codes))
    run_batch=col_single.button(f"Run one batch ({min(batch_size,max(0,len(backtest_codes)-done))})",key="p4_run_batch",disabled=(done>=len(backtest_codes) or not backtest_codes))
    reset_batch=col_reset.button("Restart this backtest run",key="p4_reset_batch",disabled=(done==0))
    st.caption("The automatic run may take a long time. Progress and data are saved after each batch; if Streamlit interrupts it, return here and click the automatic button again to resume.")
    if reset_batch:
        set_batch_job_index(job_key,backtest_codes,interval,0)
        st.rerun()
    if run_all:
        current=done
        batch_total=int(np.ceil((len(backtest_codes)-current)/batch_size))
        overall=st.progress(0,text=f"Starting {batch_total} remaining batches…")
        failed_codes=[]; batch_number=0
        while current<len(backtest_codes):
            batch_number+=1
            batch_codes=backtest_codes[current:min(current+batch_size,len(backtest_codes))]
            failed_codes.extend(process_historical_batch(batch_codes,job_key,interval,batch_number,batch_total))
            current=min(current+len(batch_codes),len(backtest_codes))
            set_batch_job_index(job_key,backtest_codes,interval,current)
            overall.progress(current/max(1,len(backtest_codes)),text=f"Completed {current:,} of {len(backtest_codes):,} shares")
        overall.empty()
        st.session_state["p4_backtest_threshold"]=signal_threshold
        st.success(f"Finished the full run. {len(failed_codes)} ticker(s) had unavailable or insufficient history; see the failed-share list below.")
        st.rerun()
    elif run_batch:
        batch_codes=backtest_codes[done:min(done+batch_size,len(backtest_codes))]
        process_historical_batch(batch_codes,job_key,interval)
        next_index=min(done+len(batch_codes),len(backtest_codes))
        set_batch_job_index(job_key,backtest_codes,interval,next_index)
        st.session_state["p4_backtest_threshold"]=signal_threshold
        st.rerun()
    bt=load_historical_learning(backtest_codes)
    if not bt.empty:
        cutoff=st.session_state.get("p4_backtest_threshold",signal_threshold)
        signals=bt[pd.to_numeric(bt["Score"],errors="coerce")>=cutoff].copy()
        total_rows=bt.attrs.get("total_observations",len(bt))
        sample_text=f"Showing a representative sample of {len(bt):,} from {total_rows:,} stored observations." if bt.attrs.get("display_sampled",False) else f"Showing all {total_rows:,} stored observations."
        st.caption(f"{sample_text} {len(signals):,} displayed rows met the score threshold of {cutoff}. Evaluation uses original fixed weights to avoid look-ahead. All completed monthly learning observations train future live Early-Mover scores.")
        st.subheader("Accuracy by timeframe")
        horizon_rows=[]
        for horizon in (5,10,20,30):
            metrics=backtest_metrics(signals,horizon)
            horizon_rows.append({"Timeframe (trading days)":horizon,**{k:(round(v,2) if pd.notna(v) and isinstance(v,(int,float,np.integer,np.floating)) else v) for k,v in metrics.items()}})
        st.dataframe(pd.DataFrame(horizon_rows),use_container_width=True,hide_index=True)
        horizon=st.selectbox("Diagnostic timeframe",[5,10,20,30],index=2,key="p4_horizon")
        retcol=f"Return {horizon}d %"
        st.subheader("Calibration by score bucket")
        work=bt.copy()
        work["Score bucket"]=pd.cut(pd.to_numeric(work["Score"],errors="coerce"),[-np.inf,49,64,79,np.inf],labels=["0–49","50–64","65–79","80–100"])
        bucket=work.groupby("Score bucket",observed=False).agg(
            Samples=(retcol,"count"),Positive_return_rate=(retcol,lambda x:round((x>0).mean()*100,1)),
            Mean_return=(retcol,"mean"),Median_return=(retcol,"median"),False_positive_rate=(retcol,lambda x:round((x<0).mean()*100,1))).reset_index()
        bucket.columns=["Score bucket","Samples","Positive return %","Average return %","Median return %","False positive %"]
        st.dataframe(bucket.round(2),use_container_width=True,hide_index=True)
        st.caption("A false positive is a score at or above the selected threshold followed by a negative return at the chosen timeframe. Calibration is descriptive: score bands are not probabilities.")
        st.subheader("What went wrong or right?")
        st.write("**False positives / weakest signal outcomes**")
        st.dataframe(signals.sort_values(retcol).head(12)[["Ticker","Signal date","Score","Entry price",retcol]],use_container_width=True,hide_index=True)
        st.write("**Best signal outcomes**")
        st.dataframe(signals.sort_values(retcol,ascending=False).head(12)[["Ticker","Signal date","Score","Entry price",retcol]],use_container_width=True,hide_index=True)
        st.subheader("Indicator diagnostics")
        st.caption("These correlations compare trailing indicator values with the selected forward return in this sample. They are exploratory and do not establish cause.")
        diagnostics=[]
        for feature in FEATURES:
            if feature in work:
                pair=work[[feature,retcol]].apply(pd.to_numeric,errors="coerce").dropna()
                corr=pair.iloc[:,0].corr(pair.iloc[:,1]) if len(pair)>=10 and pair.iloc[:,0].nunique()>1 else np.nan
                diagnostics.append({"Indicator":feature,"Sample count":len(pair),"Correlation with forward return":corr})
        st.dataframe(pd.DataFrame(diagnostics).round(3),use_container_width=True,hide_index=True)
        st.download_button("Download displayed backtest rows CSV",bt.to_csv(index=False).encode(),file_name="asx_phase4_historical_backtest_sample.csv",mime="text/csv")
    else:
        st.info("Run the next download batch. Each share's maximum available daily history is cached locally, and completed historical observations are added to the learning pool.")
    con=db_connect()
    failures_df=pd.read_sql_query("SELECT ticker AS Ticker,reason AS Reason FROM historical_batch_failures WHERE job_key=? ORDER BY ticker",con,params=(job_key,))
    con.close()
    failures=failures_df.to_dict("records")
    if failures:
        with st.expander(f"Unavailable shares ({len(failures)})"):
            st.dataframe(pd.DataFrame(failures),use_container_width=True,hide_index=True)
    render_database_backup("phase4_backtest")

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

st.divider()
st.subheader("Data and limitations")
st.markdown("- Data provider: Yahoo Finance via yfinance; availability, freshness and definitions vary.\n- The bundled company file contains 1,810 current ASX codes; it is not a historical record of delisted companies or past ASX membership.\n- Maximum available history varies by ticker; newer listings have shorter histories. Small caps may have missing prices, low liquidity, wide spreads and high volatility.\n- Scores are rule-based comparisons, not probabilities or buy/sell recommendations.\n- Historical correlations do not show that a factor caused a price rise. Verify ASX announcements, audited reports, debt, cash flow, dilution, liquidity and trading halts.\n- The app does not connect to a broker or place trades.")
st.caption("Personal-use research prototype. Not affiliated with QAV, Stock Doctor, Yahoo Finance or ASX.")

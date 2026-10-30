import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import time
import sqlite3
import json
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
                wanted = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in h.columns]
                h = h[wanted].dropna(subset=["Close"]) if "Close" in wanted else pd.DataFrame()
                output[sym] = h
            except Exception:
                output[sym] = pd.DataFrame()
        time.sleep(1.0)  # modest pause between batches
    return output


def refresh_vpa_results(frame, period="1y"):
    """Backfill VPA from OHLCV without rerunning fundamentals or losing prior scores."""
    updated = frame.copy()
    fields = ["VPA Score", "Accumulation Score", "Distribution Risk",
              "Effort Result Score", "VPA Signal", "VPA Notes"]
    for field in fields:
        if field not in updated.columns:
            updated[field] = np.nan if field.endswith("Score") or field == "Distribution Risk" else ""
    symbols = updated.loc[updated["Price"].notna(), "Ticker"].astype(str).tolist()
    bar = st.progress(0, text="Downloading OHLCV data for VPA…")
    for offset in range(0, len(symbols), 30):
        batch = symbols[offset:offset + 30]
        histories = download_price_batches(batch, period, batch_size=30)
        for symbol in batch:
            history = histories.get(symbol, pd.DataFrame())
            try:
                result = calculate_vpa(history)
            except Exception as exc:
                result = {"VPA Score": np.nan, "Accumulation Score": np.nan,
                          "Distribution Risk": np.nan, "Effort Result Score": np.nan,
                          "VPA Signal": "VPA calculation error",
                          "VPA Notes": f"{type(exc).__name__}: {str(exc)[:150]}"}
            mask = updated["Ticker"].astype(str).eq(symbol)
            for field, value in result.items():
                updated.loc[mask, field] = value
        bar.progress(min(1.0, (offset + len(batch)) / max(len(symbols), 1)),
                     text=f"Calculated VPA for {min(offset+len(batch),len(symbols))}/{len(symbols)} tickers")
    bar.empty()
    return updated


def calculate_vpa(history):
    """Rule-based Volume Price Analysis signals from daily OHLCV data.

    Signals are evidence/flags, not proof of institutional activity. Scores are
    normalized to 0-100 and deliberately conservative for thinly traded shares.
    """
    empty = {
        "VPA Score": np.nan, "Accumulation Score": np.nan,
        "Distribution Risk": np.nan, "Effort Result Score": np.nan,
        "VPA Signal": "Insufficient OHLCV data", "VPA Notes": "Fewer than 30 usable OHLCV sessions; check Yahoo history, OHLC columns and trading volume."
    }
    if history is None or history.empty or len(history) < 30:
        return empty
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in history.columns for c in required):
        return empty
    x = history[required].dropna(subset=["High", "Low", "Close"]).copy()
    x = x[(x["Volume"].fillna(0) >= 0) & (x["High"] >= x["Low"])]
    if len(x) < 30:
        return empty
    x = x.tail(120)
    o,h,l,c,v = (x[k].astype(float) for k in required)
    spread=(h-l).replace(0,np.nan)
    close_loc=((c-l)/spread).clip(0,1).fillna(.5)
    vol_avg=v.rolling(20,min_periods=10).mean()
    vol_ratio=(v/(vol_avg.replace(0,np.nan))).replace([np.inf,-np.inf],np.nan)
    ret=c.pct_change()*100
    up=ret>0.15; down=ret<-.15
    recent=x.tail(20)
    recent_ret=ret.tail(20)
    recent_v=v.tail(20)
    recent_avg=vol_avg.tail(20)
    recent_loc=close_loc.tail(20)
    # Volume-weighted directional pressure; bounded to avoid one print dominating.
    vr=(recent_v/(recent_avg.replace(0,np.nan))).clip(0,4).fillna(0)
    up_pressure=float(vr[up.reindex(vr.index,fill_value=False)].sum())
    down_pressure=float(vr[down.reindex(vr.index,fill_value=False)].sum())
    total_pressure=up_pressure+down_pressure
    pressure_balance=(up_pressure-down_pressure)/total_pressure if total_pressure>0 else 0
    price_range=(recent["High"].max()-recent["Low"].min())/max(float(c.iloc[-1]),1e-9)*100
    # Accumulation is suggested by positive directional volume, closes firm in bars,
    # and a contained range; this is a heuristic, not participant identification.
    firm_closes=float(recent_loc.mean())
    accumulation=float(np.clip(50+pressure_balance*28+(firm_closes-.5)*35-np.clip(price_range-18,0,30)*.35,0,100))
    # Latest-bar event tests against the preceding 20 sessions (exclude today).
    avg20=num(vol_avg.iloc[-1]); today_v=num(v.iloc[-1]); ratio=(today_v/avg20) if avg20 and avg20>0 else np.nan
    prev_high=num(h.iloc[-21:-1].max()) if len(h)>=21 else None
    today_close=float(c.iloc[-1]); today_high=float(h.iloc[-1]); today_low=float(l.iloc[-1])
    today_loc=float(close_loc.iloc[-1]); today_ret=float(ret.iloc[-1]) if pd.notna(ret.iloc[-1]) else 0.0
    today_spread=float(spread.iloc[-1]) if pd.notna(spread.iloc[-1]) else 0.0
    median_spread=float(spread.tail(20).median()) if spread.tail(20).notna().any() else 0.0
    no_supply=bool(today_ret<0 and ratio is not None and np.isfinite(ratio) and ratio<.75 and today_spread<=median_spread*1.15)
    no_demand=bool(today_ret>0 and ratio is not None and np.isfinite(ratio) and ratio<.75 and today_spread<=median_spread*1.15)
    stopping=bool(today_ret<0 and ratio is not None and np.isfinite(ratio) and ratio>=1.8 and today_loc>=.60)
    upthrust=bool(prev_high is not None and today_high>prev_high and today_close<prev_high and today_loc<.45 and ratio is not None and np.isfinite(ratio) and ratio>=1.3)
    breakout=bool(prev_high is not None and today_close>prev_high and ratio is not None and np.isfinite(ratio) and ratio>=1.5)
    distribution=bool(ratio is not None and np.isfinite(ratio) and ratio>=1.8 and today_loc<.4 and (today_ret<=.5 or today_close>=float(c.tail(20).max())*.95))
    # Effort/result: large relative volume with muted price progress is a caution;
    # high volume with meaningful directional movement scores better.
    absret=abs(today_ret)
    effort=float(np.clip(50 + (min(absret,5)*8 if ratio and ratio>=1.3 else 0) - (max(0,(ratio or 0)-1.5)*12 if absret<1.0 else 0),0,100))
    dist_risk=float(np.clip((35 if distribution else 0)+(40 if upthrust else 0)+(15 if no_demand else 0)+(10 if today_ret<0 and ratio and ratio>=1.5 and today_loc<.4 else 0),0,100))
    event_bonus=(12 if no_supply else 0)+(18 if stopping else 0)+(18 if breakout else 0)
    event_penalty=(18 if no_demand else 0)+(25 if upthrust else 0)+(18 if distribution else 0)
    vpa_score=float(np.clip(accumulation*.55+effort*.20+25+event_bonus-event_penalty,0,100))
    signals=[]
    if breakout: signals.append("Volume-confirmed breakout")
    if stopping: signals.append("Stopping volume")
    if no_supply: signals.append("No supply")
    if no_demand: signals.append("No demand")
    if upthrust: signals.append("Upthrust / failed breakout")
    if distribution: signals.append("Possible distribution")
    if not signals:
        signals.append("No strong latest-bar event")
    notes=f"20-session accumulation {accumulation:.0f}/100; latest volume {ratio:.2f}x 20-day average" if ratio is not None and np.isfinite(ratio) else f"20-session accumulation {accumulation:.0f}/100; volume baseline unavailable"
    return {"VPA Score":round(vpa_score,1),"Accumulation Score":round(accumulation,1),
            "Distribution Risk":round(dist_risk,1),"Effort Result Score":round(effort,1),
            "VPA Signal":"; ".join(signals),"VPA Notes":notes}


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
                "Profit Margin %": np.nan, "VPA Score": np.nan, "Accumulation Score": np.nan, "Distribution Risk": np.nan, "Effort Result Score": np.nan, "VPA Signal": "No price data", "VPA Notes": "", "Fundamentals Status": "Not fetched (no prices)",
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
            "Profit Margin %": profit_margin, **calculate_vpa(h), "Fundamentals Status": fundamentals_status,
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
FEATURES = ["Return 1W %", "Return 1M %", "Return 3M %", "Price vs 50D %", "Above 200D %", "Volume Change %", "Distance from 20D High %", "Short-term extension", "VPA Score", "Accumulation Score", "Distribution Safety", "Effort Result Score"]
BASE_WEIGHTS = np.array([0.12, 0.12, 0.08, 0.08, 0.08, 0.16, 0.08, 0.08, 0.12, 0.04, 0.02, 0.02], dtype=float)

def db_connect():
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, company TEXT,
        signal_date TEXT NOT NULL, entry_price REAL NOT NULL, base_score REAL,
        components_json TEXT, features_json TEXT, weights_json TEXT,
        price_5d REAL, return_5d REAL, checked_5d TEXT,
        price_20d REAL, return_20d REAL, checked_20d TEXT,
        price_60d REAL, return_60d REAL, checked_60d TEXT,
        UNIQUE(ticker, signal_date))''')
    con.commit()
    return con

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
    ]
    vpa, accum, effort = val("VPA Score"), val("Accumulation Score"), val("Effort Result Score")
    dist = val("Distribution Risk")
    values.extend([vpa, accum, (100-dist) if dist is not None else np.nan, effort])
    return [None if pd.isna(x) else float(x) for x in values]

def learned_weights():
    con=db_connect()
    df=pd.read_sql_query("SELECT features_json, return_20d FROM predictions WHERE return_20d IS NOT NULL",con)
    con.close()
    if len(df)<30:
        return BASE_WEIGHTS.copy(), len(df), False, []
    rows=[]
    for _,r in df.iterrows():
        try: rows.append(json.loads(r["features_json"] or "{}"))
        except Exception: rows.append({})
    f=pd.DataFrame(rows)
    target=pd.to_numeric(df["return_20d"],errors="coerce")
    correlations=[]
    for feature in FEATURES:
        x=pd.to_numeric(f[feature] if feature in f.columns else pd.Series(np.nan,index=f.index),errors="coerce")
        pair=pd.concat([x,target],axis=1).dropna()
        corr=float(pair.iloc[:,0].corr(pair.iloc[:,1])) if len(pair)>=20 and pair.iloc[:,0].nunique()>1 else 0.0
        correlations.append(corr if np.isfinite(corr) else 0.0)
    # Blend empirical relationships conservatively with the original transparent weights.
    multipliers=np.clip(1+1.5*np.array(correlations),0.5,1.5)
    weights=BASE_WEIGHTS*multipliers
    weights=weights/weights.sum()
    return weights, len(df), True, correlations

def score_with_weights(row, weights):
    comps=component_scores(row)
    usable=[(i,v) for i,v in enumerate(comps) if v is not None]
    if not usable: return np.nan
    denom=sum(weights[i] for i,_ in usable)
    return round(sum(v*weights[i] for i,v in usable)/denom,1) if denom else np.nan

def save_predictions(frame, weights):
    con=db_connect(); saved=0
    for _,r in frame.iterrows():
        ticker=str(r.get("Ticker","")); price=num(r.get("Price")); date=str(r.get("Data Date", ""))[:10]
        if not ticker or price is None or price<=0 or not date: continue
        # Calculate component scores before using them in the feature record.
        # Previously `comps` was referenced one line too early, causing
        # UnboundLocalError whenever a prediction was saved.
        comps=component_scores(r)
        features={k:num(r.get(k)) for k in FEATURES[:7]}
        features[FEATURES[7]]=comps[7]
        features["VPA Score"]=num(r.get("VPA Score"))
        features["Accumulation Score"]=num(r.get("Accumulation Score"))
        dr=num(r.get("Distribution Risk")); features["Distribution Safety"]=(100-dr) if dr is not None else None
        features["Effort Result Score"]=num(r.get("Effort Result Score"))
        try:
            cur=con.execute("INSERT OR IGNORE INTO predictions (ticker,company,signal_date,entry_price,base_score,components_json,features_json,weights_json) VALUES (?,?,?,?,?,?,?,?)",
                (ticker,str(r.get("Company",ticker)),date,float(price),num(r.get("Early-Mover Score")),json.dumps(comps),json.dumps(features),json.dumps(weights.tolist())))
            if cur.rowcount > 0: saved+=1
        except Exception: pass
    con.commit(); con.close(); return saved

def refresh_prediction_outcomes():
    con=db_connect()
    pending=pd.read_sql_query("SELECT * FROM predictions WHERE return_5d IS NULL OR return_20d IS NULL OR return_60d IS NULL",con)
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
            for horizon in (5,20,60):
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


MASTER_TICKERS = load_master_tickers()
with st.sidebar:
    page=st.radio("Page",["Single-Share Analysis","Automatic Share Scanner","Early-Mover Tracker","VPA Scanner","Prediction Lab","Lower-Risk Investing","Value Investing","Momentum Trading","High-Growth Shares","Long-Term Investing","Portfolio Planner"])
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
                # Older Phase 3 rows have prices but no VPA columns. Do not silently
                # treat those rows as Phase 5-complete results.
                if "VPA Score" not in kept_df.columns:
                    already_done = set()
                else:
                    missing_vpa = set(kept_df.loc[kept_df["VPA Score"].isna(), "Ticker"].astype(str))
                    already_done -= missing_vpa
                kept_df = kept_df[kept_df["Ticker"].astype(str).isin(already_done)]
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
    st.write("Ranks shares for early technical signs of a possible move: improving short-term momentum, expanding volume, price strength versus moving averages, proximity to recent highs, and Volume Price Analysis evidence such as accumulation, no-supply tests and volume-confirmed breakouts.")
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
        st.caption("Score weights: 1-week momentum 15%, 1-month momentum 15%, 3-month momentum 10%, price vs 50-day average 10%, price vs 200-day average 10%, volume expansion 20%, distance from 20-day high 10%, and avoiding extreme short-term extension 8%, plus VPA score 12%, accumulation 4%, distribution-risk inverse 2%, and effort-versus-result 2%. Missing fields are excluded and weights renormalised. The adaptive learner can adjust these weights after enough completed outcomes.")
        st.caption(f"Latest scan: {st.session_state.get('scan_time','unknown')}. Use each share's Data Date to check freshness.")
        cols=["Ticker","Company","Price","Early-Mover Score","Signal Tier","Return 1W %","Return 1M %","Return 3M %","Price vs 50D %","Above 200D %","Volume Change %","Distance from 20D High %","VPA Score","Accumulation Score","Distribution Risk","VPA Signal","Market Cap","Data Date","Scan Status"]
        st.dataframe(movers[[c for c in cols if c in movers.columns]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download early-mover rankings CSV",movers.to_csv(index=False).encode(),file_name="asx_early_mover_rankings.csv",mime="text/csv")
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

elif page=="VPA Scanner":
    st.header("📊 Volume Price Analysis Scanner")
    st.write("Ranks ASX shares using the relationship between daily price spread, candle close location and trading volume. It looks for possible accumulation, no-supply tests, stopping volume, volume-confirmed breakouts, distribution and upthrusts.")
    st.warning("VPA signals are heuristic interpretations of OHLCV data. They cannot identify who is trading or prove institutional accumulation/distribution. Thinly traded ASX shares, announcements and stale data can distort signals.")
    if scan.empty:
        st.info("Run a full-market or quick promising-shares scan first. VPA uses the same downloaded price history.")
    else:
        missing_count = (int(scan["Price"].notna().sum()) if "VPA Score" not in scan.columns
                         else int((scan["Price"].notna() & scan["VPA Score"].isna()).sum()))
        if missing_count:
            st.warning(f"{missing_count} tickers have prices but no VPA score. These may be reused Phase 3 results. You can repair VPA without rescanning fundamentals.")
            if st.button("Calculate / repair missing VPA scores", type="primary"):
                with st.spinner("Refreshing VPA from historical prices and volume…"):
                    repaired = refresh_vpa_results(scan, st.session_state.get("scan_period", "1y"))
                    st.session_state["scan_results"] = repaired
                st.rerun()
        vpa=scan[scan["Price"].notna()].copy()
        # Backward compatibility: cached/session scan results from Phase 3 or
        # earlier Phase 5 builds may not contain the newer VPA columns.
        # Add them safely so the VPA page can still render instead of raising KeyError.
        vpa_defaults = {
            "VPA Score": np.nan,
            "Accumulation Score": np.nan,
            "Distribution Risk": np.nan,
            "Effort Result Score": np.nan,
            "VPA Signal": "VPA data not available in this saved scan",
            "VPA Notes": "Run a fresh scan to calculate VPA from OHLCV history."
        }
        for column, default_value in vpa_defaults.items():
            if column not in vpa.columns:
                vpa[column] = default_value
        vpa=vpa.sort_values(["VPA Score","Accumulation Score"],ascending=False,na_position="last")
        a,b,c=st.columns(3)
        a.metric("Shares with VPA score",int(vpa["VPA Score"].notna().sum()))
        b.metric("Possible accumulation (60+)",int((vpa["Accumulation Score"]>=60).sum()))
        c.metric("Distribution warnings",int((vpa["Distribution Risk"]>=35).sum()))
        st.caption("VPA score combines a 20-session directional-volume/range assessment, effort-versus-result and latest-bar event flags. It is a research ranking, not a probability or buy/sell signal.")
        cols=["Ticker","Company","Price","VPA Score","Accumulation Score","Distribution Risk","Effort Result Score","VPA Signal","VPA Notes","Volume Change %","Return 1W %","Return 1M %","Data Date"]
        st.dataframe(vpa[[k for k in cols if k in vpa]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download VPA rankings CSV",vpa.to_csv(index=False).encode(),file_name="asx_vpa_rankings.csv",mime="text/csv")
        st.subheader("How to interpret the signals")
        st.markdown("- **Possible accumulation:** directional volume pressure and firm closes suggest demand may be absorbing supply.\n- **No supply:** a low-volume decline with a relatively narrow spread may indicate reduced selling interest.\n- **Stopping volume:** heavy volume during a decline with a recovery from the lows may indicate demand absorbing selling.\n- **Volume-confirmed breakout:** close above the prior 20-session high with volume at least 1.5× its 20-day average.\n- **Distribution / upthrust:** heavy volume with weak closes or a failed move above recent highs can warn of supply.")
        st.info("The Early-Mover Score now includes VPA features. The Prediction Lab learns their historical relationship with subsequent 20-day returns alongside the original indicators. Existing Phase 3 prediction history remains readable; older records simply have no VPA feature values.")

elif page=="Prediction Lab":
    st.header("🧠 Prediction Lab — accuracy and learning")
    st.write("Every saved prediction is measured against its entry close after 5, 20 and 60 subsequent trading sessions. The learning engine reviews completed 20-day outcomes to estimate which signals have been more useful, including the new VPA features. VPA values are saved with each new prediction so their contribution can be evaluated.")
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
        # Same-prediction comparison: original eight-feature score vs VPA-enhanced
        # 12-feature score, using only rows that actually contain saved VPA values.
        compare_rows=[]
        original_weights=np.array([0.15,0.15,0.10,0.10,0.10,0.20,0.10,0.10],dtype=float)
        for _,pr in predictions.iterrows():
            try: parts=json.loads(pr.get("components_json") or "[]")
            except Exception: parts=[]
            outcome=num(pr.get("return_20d"))
            if len(parts)>=12 and outcome is not None and any(x is not None for x in parts[8:12]):
                old=[(i,num(v)) for i,v in enumerate(parts[:8]) if num(v) is not None]
                new=[(i,num(v)) for i,v in enumerate(parts[:12]) if num(v) is not None]
                old_score=(sum(v*original_weights[i] for i,v in old)/sum(original_weights[i] for i,v in old)) if old else np.nan
                new_score=(sum(v*BASE_WEIGHTS[i] for i,v in new)/sum(BASE_WEIGHTS[i] for i,v in new)) if new else np.nan
                compare_rows.append({"Original score":old_score,"VPA-enhanced score":new_score,"20-day return %":outcome})
        if len(compare_rows)>=10:
            cmp=pd.DataFrame(compare_rows)
            st.subheader("Does VPA improve the ranking? — same-prediction comparison")
            st.caption(f"Compares {len(cmp)} completed predictions that contain saved VPA data. Both scores are recalculated with their original baseline weights; adaptive weights are excluded here to keep the comparison interpretable.")
            summary=[]
            for label in ["Original score","VPA-enhanced score"]:
                pairs=cmp[[label,"20-day return %"]].dropna()
                corr=pairs.iloc[:,0].corr(pairs.iloc[:,1]) if len(pairs)>=3 and pairs.iloc[:,0].nunique()>1 else np.nan
                top=pairs[pairs[label]>=pairs[label].quantile(.75)]
                summary.append({"Model":label,"Score / return correlation":round(corr,3) if pd.notna(corr) else np.nan,"Top-quartile predictions":len(top),"Top-quartile positive rate %":round((top["20-day return %"]>0).mean()*100,1) if len(top) else np.nan,"Top-quartile average return %":round(top["20-day return %"].mean(),2) if len(top) else np.nan})
            st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True)
            st.caption("Treat this as an early diagnostic, not proof of improvement. Use a larger sample across different market conditions and avoid tuning on the same observations used to judge performance.")
        elif compare_rows:
            st.info(f"VPA comparison warming up: {len(compare_rows)}/10 completed predictions with VPA data. More completed 20-day outcomes are needed.")
        st.subheader("Prediction performance")
        metrics=st.columns(4); metrics[0].metric("Predictions recorded",total)
        for col,h in zip(metrics[1:],(5,20,60)):
            vals=pd.to_numeric(predictions[f"return_{h}d"],errors="coerce").dropna()
            wins=int((vals>0).sum()); count=len(vals)
            col.metric(f"{h}-day positive rate",f"{wins/count*100:.1f}%" if count else "Not ready",help=f"{wins} positive outcomes out of {count} completed predictions")
        summaries=[]
        for h in (5,20,60):
            vals=pd.to_numeric(predictions[f"return_{h}d"],errors="coerce").dropna()
            summaries.append({"Horizon":f"{h} trading days","Completed":len(vals),"Positive %":round((vals>0).mean()*100,1) if len(vals) else np.nan,"Average return %":round(vals.mean(),2) if len(vals) else np.nan,"Median return %":round(vals.median(),2) if len(vals) else np.nan})
        st.dataframe(pd.DataFrame(summaries),use_container_width=True,hide_index=True)
        st.subheader("What the system is learning")
        weights,n,active,corr=learned_weights()
        if active:
            learned=pd.DataFrame({"Indicator":FEATURES,"Original weight %":BASE_WEIGHTS*100,"Learned weight %":weights*100,"Correlation with 20-day return":corr})
            st.dataframe(learned.round(3),use_container_width=True,hide_index=True)
            st.caption("Correlations are descriptive, not causal. The model blends them conservatively with original weights and only activates after 30 completed 20-day observations.")
        else:
            st.info(f"The adaptive scoring is not active yet. {n}/30 completed 20-day outcomes are available. Until then, the original weights are used.")
        st.subheader("Individual predictions")
        view_cols=["ticker","company","signal_date","entry_price","base_score","price_5d","return_5d","price_20d","return_20d","price_60d","return_60d"]
        view=predictions[view_cols].copy()
        view.columns=["Ticker","Company","Signal date","Entry price","Signal score","5d price","5d return %","20d price","20d return %","60d price","60d return %"]
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
st.markdown("- Data provider: Yahoo Finance via yfinance; availability, freshness and definitions vary.\n- The built-in list is a sample, not the full ASX. Paste your own ticker list for wider coverage.\n- Small caps may have missing data, low liquidity, wide spreads and high volatility.\n- Scores are rule-based comparisons, not predictions, probabilities, or buy/sell recommendations.\n- Verify ASX announcements, audited reports, debt, cash flow, dilution, liquidity and trading halts.\n- The app does not connect to a broker or place trades.")
st.caption("Personal-use research prototype. Not affiliated with QAV, Stock Doctor, Yahoo Finance or ASX.")

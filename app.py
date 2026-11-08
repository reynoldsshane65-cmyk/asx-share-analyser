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
import zipfile
import shutil
import gc
import os
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
    fields = ["VPA Score", "Accumulation Score", "Distribution Risk", "20D Distribution Risk",
              "Effort Result Score", "VPA Signal", "VPA Notes"]
    for field in fields:
        if field not in updated.columns:
            updated[field] = np.nan if field.endswith("Score") or "Distribution Risk" in field else ""
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
                          "Distribution Risk": np.nan, "20D Distribution Risk": np.nan, "Effort Result Score": np.nan,
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
        "Distribution Risk": np.nan, "20D Distribution Risk": np.nan, "Effort Result Score": np.nan,
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
    # Persistent distribution: evaluate the same four warnings for each of
    # the latest 20 completed daily bars, with more weight on recent sessions.
    # Use preceding 20 bars for reference highs and average volume to avoid
    # look-ahead bias; a minimum of 30 usable sessions is required above.
    prior_avg = v.shift(1).rolling(20, min_periods=10).mean()
    rolling_ratio = (v / prior_avg.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    prior_highs = h.shift(1).rolling(20, min_periods=10).max()
    prior_max_close = c.shift(1).rolling(20, min_periods=10).max()
    median_spreads = spread.shift(1).rolling(20, min_periods=10).median()
    possible_dist = ((rolling_ratio >= 1.8) & (close_loc < .4) &
                     ((ret <= .5) | (c >= prior_max_close * .95)))
    failed_breakout = ((h > prior_highs) & (c < prior_highs) &
                       (close_loc < .45) & (rolling_ratio >= 1.3))
    weak_demand = ((ret > 0) & (rolling_ratio < .75) &
                   (spread <= median_spreads * 1.15))
    bearish_volume = ((ret < 0) & (rolling_ratio >= 1.5) & (close_loc < .4))
    daily_risk = (possible_dist.astype(float) * 35 +
                  failed_breakout.astype(float) * 40 +
                  weak_demand.astype(float) * 15 +
                  bearish_volume.astype(float) * 10).clip(0, 100)
    last20 = daily_risk.tail(20).to_numpy(dtype=float)
    weights = np.arange(1, len(last20) + 1, dtype=float)
    # Recency-weighted mean plus persistence penalty for repeated warning days.
    # Heuristic score (not a calibrated probability).
    persistent_days = int(np.count_nonzero(last20 >= 35))
    dist_risk_20d = float(np.clip(
        np.average(last20, weights=weights) + min(30, persistent_days * 4),
        0, 100))
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
            "Distribution Risk":round(dist_risk,1),"20D Distribution Risk":round(dist_risk_20d,1),"Effort Result Score":round(effort,1),
            "VPA Signal":"; ".join(signals),"VPA Notes":notes}


def phase4_technical_inputs(hist):
    if hist is None or hist.empty or "Close" not in hist:
        return {}
    h=hist.dropna(subset=["Close"])
    if h.empty: return {}
    last=h.iloc[-1]; price=num(last.get("Close"))
    high=num(last.get("High")); low=num(last.get("Low")); opn=num(last.get("Open"))
    prev=num(h["Close"].iloc[-2]) if len(h)>1 else None
    return {
        "Daily range %":(high-low)/price*100 if high is not None and low is not None and price else np.nan,
        "Close in daily range %":(price-low)/(high-low)*100 if high is not None and low is not None and high>low else np.nan,
        "Gap %":(opn/prev-1)*100 if opn is not None and prev else np.nan,
        "20D volatility %":h["Close"].pct_change().tail(20).std()*100 if len(h)>=21 else np.nan,
    }

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
                "Volume Change %": np.nan, "20D dollar volume AUD": np.nan, "Market Cap": np.nan,
                "P/E": np.nan, "Revenue Growth %": np.nan,
                "Profit Margin %": np.nan, "VPA Score": np.nan, "Accumulation Score": np.nan, "Distribution Risk": np.nan, "20D Distribution Risk": np.nan, "Effort Result Score": np.nan, "VPA Signal": "No price data", "VPA Notes": "", "Fundamentals Status": "Not fetched (no prices)",
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
            "20D dollar volume AUD": float((h["Close"].tail(20)*h["Volume"].tail(20)).mean()) if "Volume" in h and len(h)>=20 else np.nan,
            "Market Cap": market_cap, "P/E": pe, "Revenue Growth %": revenue_growth,
            "Profit Margin %": profit_margin, **calculate_vpa(h), **phase4_technical_inputs(h), "Fundamentals Status": fundamentals_status,
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
FEATURES = ["Return 1W %", "Return 1M %", "Return 3M %", "Price vs 50D %", "Above 200D %", "Volume Change %", "Distance from 20D High %", "Short-term extension", "VPA Score", "Accumulation Score", "Distribution Safety", "Effort Result Score"] + ["Daily range %", "Close in daily range %", "Gap %", "20D volatility %"]
# Preserve the original Phase 3 factor balance while reserving 20% for four OHLCV features.
BASE_WEIGHTS = np.array([0.12, 0.12, 0.08, 0.08, 0.08, 0.16, 0.08, 0.08, 0.12, 0.04, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05], dtype=float)
BASE_WEIGHTS = BASE_WEIGHTS / BASE_WEIGHTS.sum()

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

# Backups are staged on disk to avoid holding a multi-GB database in Streamlit session state.
BACKUP_CHUNK_BYTES = 100 * 1024 * 1024  # below Streamlit's default 200 MB upload limit
BACKUP_STAGE = Path(tempfile.gettempdir()) / "asx_backup_staging"
RESTORE_STAGE = Path(tempfile.gettempdir()) / "asx_restore_staging"


def _stream_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sqlite_validate(path):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60) as check:
        if check.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
        tables = {r[0] for r in check.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"predictions", "historical_learning"}.issubset(tables):
            raise ValueError("Required prediction tables are missing")


def prepare_split_backup():
    if not DB_PATH.exists():
        raise FileNotFoundError("No prediction database exists yet")
    BACKUP_STAGE.mkdir(parents=True, exist_ok=True)
    folder = BACKUP_STAGE / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir()
    snapshot = folder / "snapshot.sqlite"
    source = sqlite3.connect(str(DB_PATH), timeout=60)
    target = sqlite3.connect(str(snapshot), timeout=60)
    try:
        source.backup(target, pages=1000, sleep=0.1)
    finally:
        target.close()
        source.close()
    archive = folder / "database.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        z.write(snapshot, arcname="asx_prediction_history.sqlite")
    snapshot.unlink()
    pieces = []
    with open(archive, "rb") as stream:
        n = 0
        while True:
            block = stream.read(BACKUP_CHUNK_BYTES)
            if not block:
                break
            n += 1
            name = f"asx_backup.part{n:03d}"
            part = folder / name
            part.write_bytes(block)
            pieces.append({"name": name, "size": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    manifest = {"format": "asx-split-backup-v1", "archive_size": archive.stat().st_size,
                "archive_sha256": _stream_sha256(archive), "parts": pieces}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    archive.unlink()
    return folder, manifest


def render_database_backup(key_prefix):
    st.caption("Compressed, split backups include prediction history, cached historical prices and batch progress. Keep ALL downloaded parts and the manifest together.")
    if st.button("Prepare compressed backup", key=f"{key_prefix}_prepare_db_backup"):
        try:
            with st.spinner("Taking a consistent SQLite snapshot and compressing it. Large databases can take several minutes…"):
                folder, manifest = prepare_split_backup()
            st.session_state["split_backup_folder"] = str(folder)
            st.success(f"Prepared {len(manifest['parts'])} part(s), each at most 100 MiB.")
        except Exception as exc:
            st.error(f"Backup preparation failed: {exc}")
    folder_name = st.session_state.get("split_backup_folder")
    if folder_name and Path(folder_name, "manifest.json").exists():
        folder = Path(folder_name)
        manifest = json.loads((folder / "manifest.json").read_text())
        st.download_button("Download manifest.json FIRST", data=(folder / "manifest.json").read_bytes(),
                           file_name="asx_backup_manifest.json", mime="application/json", key=f"{key_prefix}_manifest")
        for item in manifest["parts"]:
            part = folder / item["name"]
            # Download buttons hold their payload in memory; only read the requested 100 MiB part.
            with st.expander(f"Download {item['name']} ({item['size'] / 1048576:.1f} MiB)"):
                st.download_button(f"Download {item['name']}", data=part.read_bytes(),
                                   file_name=item["name"], mime="application/octet-stream",
                                   key=f"{key_prefix}_{item['name']}")
        st.info("Save every part and the manifest somewhere permanent. Streamlit Cloud storage is temporary.")


def render_database_restore():
    with st.expander("Restore prediction database (split backup or existing SQLite file)"):
        st.warning("Restore REPLACES the current database. Download a backup of the current database first. Cloud disk storage is temporary; do not reboot during restoration.")
        st.caption("For a new backup: upload its manifest once, then upload ONE part at a time in order. The app saves parts on disk between uploads. For your existing 2 GB .sqlite file, split it locally into 100 MiB parts first; see the included helper script.")
        manifest_upload = st.file_uploader("Upload backup manifest (.json)", type=["json"], key="restore_manifest_upload")
        if st.button("Start new restore from manifest", disabled=manifest_upload is None, key="restore_manifest_start"):
            try:
                manifest = json.loads(manifest_upload.getvalue())
                parts = manifest.get("parts", [])
                if (manifest.get("format") != "asx-split-backup-v1" or not parts or
                    len(parts) > 1000 or any(x.get("name") != f"asx_backup.part{i:03d}" or
                    not isinstance(x.get("size"), int) or not 0 < x["size"] <= BACKUP_CHUNK_BYTES or
                    len(str(x.get("sha256", ""))) != 64 for i, x in enumerate(parts, 1))):
                    raise ValueError("Invalid backup manifest")
                RESTORE_STAGE.mkdir(parents=True, exist_ok=True)
                for old in RESTORE_STAGE.glob("asx_backup.part*"):
                    old.unlink()
                (RESTORE_STAGE / "manifest.json").write_text(json.dumps(manifest))
                st.success("Manifest accepted. Upload part 1 below.")
            except Exception as exc:
                st.error(f"Invalid manifest: {exc}")
        manifest_path = RESTORE_STAGE / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            parts = manifest["parts"]
            next_index = next((i for i, part in enumerate(parts) if not (RESTORE_STAGE / part["name"]).exists()), len(parts))
            st.write(f"Received {next_index} of {len(parts)} parts")
            if next_index < len(parts):
                expected = parts[next_index]
                upload = st.file_uploader(f"Upload {expected['name']} (part {next_index+1}/{len(parts)})", key=f"restore_part_{next_index}")
                if st.button("Save uploaded part", disabled=upload is None, key=f"restore_save_{next_index}"):
                    try:
                        if upload.size != expected["size"]:
                            raise ValueError("Part size does not match manifest")
                        candidate = RESTORE_STAGE / "incoming.part"
                        with open(candidate, "wb") as out:
                            while True:
                                block = upload.read(4 * 1024 * 1024)
                                if not block:
                                    break
                                out.write(block)
                        if _stream_sha256(candidate) != expected["sha256"]:
                            candidate.unlink(missing_ok=True)
                            raise ValueError("Part checksum mismatch")
                        os.replace(candidate, RESTORE_STAGE / expected["name"])
                        st.success("Part saved.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Part rejected: {exc}")
            else:
                st.success("All parts received and verified.")
                confirmed = st.checkbox("I understand this replaces ALL existing prediction and backtest records", key="restore_split_confirm")
                if st.button("Verify and restore database", disabled=not confirmed, key="restore_split_apply"):
                    try:
                        with st.spinner("Reassembling, checking and restoring database…"):
                            archive = RESTORE_STAGE / "reassembled.zip"
                            with open(archive, "wb") as out:
                                for part in parts:
                                    with open(RESTORE_STAGE / part["name"], "rb") as inp:
                                        shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
                            if archive.stat().st_size != manifest["archive_size"] or _stream_sha256(archive) != manifest["archive_sha256"]:
                                raise ValueError("Archive verification failed")
                            candidate = DB_PATH.with_name("asx_restore_candidate.sqlite")
                            with zipfile.ZipFile(archive) as z:
                                names = z.namelist()
                                if names != ["asx_prediction_history.sqlite"]:
                                    raise ValueError("Unexpected backup archive contents")
                                with z.open(names[0]) as inp, open(candidate, "wb") as out:
                                    shutil.copyfileobj(inp, out, 4 * 1024 * 1024)
                            _sqlite_validate(candidate)
                            # Preserve the previous database until the replacement is validated.
                            # Close any active app DB users before replacing it.
                            if DB_PATH.exists():
                                with sqlite3.connect(str(DB_PATH), timeout=60) as current:
                                    current.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                            os.replace(candidate, DB_PATH)
                            st.session_state.pop("sqlite_backup_bytes", None)
                        st.success("Database restored and validated. Refresh the app before running another scan.")
                    except Exception as exc:
                        st.error(f"Restore failed: {exc}. The current database was not intentionally replaced unless the final replacement completed.")

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
    if dist is None and val("Distribution Safety") is not None: dist = 100-val("Distribution Safety")
    values.extend([vpa, accum, (100-dist) if dist is not None else np.nan, effort])
    values.extend([
        np.clip(80-val("Daily range %")*2,0,100) if val("Daily range %") is not None else np.nan,
        val("Close in daily range %") if val("Close in daily range %") is not None else np.nan,
        np.clip(50+val("Gap %")*4,0,100) if val("Gap %") is not None else np.nan,
        np.clip(100-val("20D volatility %")*4,0,100) if val("20D volatility %") is not None else np.nan,
    ])
    return [None if pd.isna(x) else float(x) for x in values]

def learned_weights(cutoff_date=None):
    con=db_connect()
    # A full-universe history may contain millions of rows. Stream through them
    # and retain only the sums needed to calculate each indicator correlation.
    historical_filter="WHERE h.return_20d IS NOT NULL"
    live_filter="WHERE p.return_20d IS NOT NULL"
    query_params=[]
    if cutoff_date:
        historical_filter+=" AND h.signal_date < ?"
        live_filter+=" AND p.signal_date < ?"
        query_params=[cutoff_date,cutoff_date]
    query="""WITH observations AS (
          SELECT h.ticker,h.signal_date,h.features_json,h.return_20d FROM historical_learning_monthly h
          """+historical_filter+"""
          UNION ALL
          SELECT p.ticker,p.signal_date,p.features_json,p.return_20d FROM predictions p """+live_filter+"""
        ), monthly AS (
          SELECT features_json,return_20d,
                 ROW_NUMBER() OVER (PARTITION BY ticker,substr(signal_date,1,7) ORDER BY signal_date) AS month_row
          FROM observations
        ) SELECT features_json,return_20d FROM monthly WHERE month_row=1"""
    cursors=[con.execute(query,query_params)]
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



def render_learning_dashboard():
    st.subheader("Learning status")
    weights,n,active,corr=learned_weights()
    historical_examples,live_examples=learning_source_counts()
    source_cols=st.columns(3)
    source_cols[0].metric("Historical monthly examples",historical_examples,help="Phase 4 backtest observations with a completed 20-trading-day outcome.")
    source_cols[1].metric("Live predictions with 20-day outcomes",live_examples)
    source_cols[2].metric("Effective examples used",n,help="The model keeps about one 20-day training example per ticker per calendar month to reduce overlap.")
    if active:
        learned=pd.DataFrame({"Indicator":FEATURES,"Original weight %":BASE_WEIGHTS*100,"Learned weight %":weights*100,"Change (percentage points)":(weights-BASE_WEIGHTS)*100,"Correlation with 20-day return":corr})
        st.dataframe(learned.round(3),use_container_width=True,hide_index=True)
        st.caption("Adaptive weights are active. Correlations are descriptive, not causal, and weights are blended conservatively with the base weights.")
    else:
        st.info(f"Adaptive scoring is not active yet. {n}/30 effective completed 20-day examples are available; the original weights are still used.")

def learning_source_counts():
    con=db_connect()
    historical=int(con.execute("SELECT COUNT(*) FROM historical_learning_monthly WHERE return_20d IS NOT NULL").fetchone()[0])
    live=int(con.execute("SELECT COUNT(*) FROM predictions WHERE return_20d IS NOT NULL").fetchone()[0])
    con.close()
    return historical,live

def score_with_weights(row, weights):
    comps=component_scores(row)
    usable=[(i,v) for i,v in enumerate(comps) if v is not None]
    if not usable: return np.nan
    denom=sum(weights[i] for i,_ in usable)
    return round(sum(v*weights[i] for i,v in usable)/denom,1) if denom else np.nan

def explain_early_mover_score(row, weights):
    """Return the visible indicator inputs and weighted contributions for one live score."""
    labels=["1-week momentum","1-month momentum","3-month momentum","Price vs 50-day average","Price vs 200-day average","Volume change","Distance from 20-day high","Short-term extension","VPA score","Accumulation","Distribution safety","Effort / result","Daily trading range","Close position in daily range","Opening gap","20-day volatility"]
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
        risk=num(r.get("Distribution Risk"))
        safety=num(r.get("Distribution Safety"))
        features["Distribution Safety"]=(100-risk) if risk is not None else safety
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
        risk=num(row.get("Distribution Risk"))
        safety=num(row.get("Distribution Safety"))
        features["Distribution Safety"]=(100-risk) if risk is not None else safety
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
    hist = hist.loc[hist["Close"].notna()].sort_index()
    close = hist["Close"].astype(float)
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
        row.update(calculate_vpa(hist.loc[:past.index[-1]].tail(120)))
        row["Distribution Safety"] = 100-row["Distribution Risk"] if pd.notna(row["Distribution Risk"]) else np.nan
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



def run_strategy_optimiser(frame, fee_pct=0.6, minimum_price=0.05):
    """Select rules on validation only; report final unseen chronological test separately.

    Forward close-to-close returns are proxies, not executable trade fills.
    """
    if frame is None or frame.empty:
        return None, "No historical signals are available."
    data=frame.copy()
    data["Signal date"]=pd.to_datetime(data["Signal date"],errors="coerce")
    for field in ("Score","VPA Score","Accumulation Score","Distribution Safety","Entry price"):
        data[field]=pd.to_numeric(data[field],errors="coerce") if field in data else np.nan
    for h in (5,10,20,30):
        col=f"Return {h}d %"
        data[col]=pd.to_numeric(data[col],errors="coerce") if col in data else np.nan
    data=data.dropna(subset=["Signal date","Ticker","Entry price"])
    data=data[(data["Entry price"]>=minimum_price)&(data["Entry price"]>0)]
    data=data.sort_values("Signal date")
    dates=np.array(sorted(data["Signal date"].unique()))
    if len(dates)<90:
        return None,"Need at least 90 distinct signal dates to make a meaningful time-separated comparison."
    first=pd.Timestamp(dates[int(len(dates)*0.60)])
    second=pd.Timestamp(dates[int(len(dates)*0.80)])
    gap=pd.Timedelta(days=45)
    train=data[data["Signal date"]<first-gap]
    validation=data[(data["Signal date"]>=first)&(data["Signal date"]<second-gap)]
    test=data[data["Signal date"]>=second]
    if len(validation)<80 or len(test)<80:
        return None,"Insufficient validation or test observations after the 45-day outcome gaps."
    candidates=[]
    # Predeclared grid avoids trying hundreds of arbitrary thresholds on small samples.
    for score in (55,65,75,85):
        for vpa in (None,65,75):
            for accum in (None,65):
                for safety in (None,70):
                    candidates.append((score,vpa,accum,safety))
    def select(df,rule):
        score,vpa,accum,safety=rule
        mask=df["Score"]>=score
        if vpa is not None: mask &= df["VPA Score"]>=vpa
        if accum is not None: mask &= df["Accumulation Score"]>=accum
        if safety is not None: mask &= df["Distribution Safety"]>=safety
        return df.loc[mask]
    def metrics(df,h):
        values=(df[f"Return {h}d %"]-fee_pct).replace([np.inf,-np.inf],np.nan).dropna()
        if values.empty: return None
        return {"Trades":int(len(values)),"Average net return %":float(values.mean()),
                "Median net return %":float(values.median()),"Win rate %":float(values.gt(0).mean()*100),
                "Total positive %":float(values[values>0].sum()),"Total negative %":float(values[values<0].sum())}
    ranked=[]
    for rule in candidates:
        for h in (5,10,20,30):
            val=metrics(select(validation,rule),h)
            if val is None or val["Trades"]<30: continue
            # Prefer robust median and positive expectancy; penalty for few observations.
            quality=val["Average net return %"]*0.5+val["Median net return %"]*0.5
            ranked.append((quality,rule,h,val))
    if not ranked: return None,"No strategy had at least 30 completed validation signals. Try lowering the minimum price."
    ranked.sort(key=lambda x:x[0],reverse=True)
    rows=[]
    for rank,(_,rule,h,val) in enumerate(ranked[:10],1):
        tst=metrics(select(test,rule),h)
        rows.append({"Validation rank":rank,"Score min":rule[0],"VPA min":rule[1],
                     "Accumulation min":rule[2],"Distribution risk max":100-rule[3] if rule[3] is not None else None,
                     "Hold days":h,"Validation trades":val["Trades"],
                     "Validation mean net %":round(val["Average net return %"],2),
                     "Validation median net %":round(val["Median net return %"],2),
                     "Test trades":tst["Trades"] if tst else 0,
                     "Test mean net %":round(tst["Average net return %"],2) if tst else np.nan,
                     "Test median net %":round(tst["Median net return %"],2) if tst else np.nan,
                     "Test win %":round(tst["Win rate %"],1) if tst else np.nan})
    winner=ranked[0]
    selected=select(test,winner[1])
    # Daily close equity simulation is not justified by overlapping fixed-horizon outcomes.
    return {"table":pd.DataFrame(rows),"train_rows":len(train),"validation_rows":len(validation),
            "test_rows":len(test),"validation_start":str(first.date()),"test_start":str(second.date()),
            "best_test":metrics(selected,winner[2]),"grid_size":len(candidates)*4},None


def render_strategy_optimiser(bt):
    st.subheader("Trading Strategy Optimiser — experimental")
    st.caption("Selects rules on a middle validation period, then reports results on a later untouched test period. "
               "No model is fitted here; the earlier training segment is reserved for future model training. "
               "The test results must not be used to repeatedly tune the same rules.")
    a,b=st.columns(2)
    with a:
        costs=st.slider("Estimated round-trip trading costs (%)",0.0,5.0,0.8,0.1,key="opt_costs")
    with b:
        min_price=st.selectbox("Minimum historical share price (AUD)",[0.01,0.05,0.10,0.20,0.50,1.00],index=2,key="opt_minprice")
    st.caption("Costs combine estimated entry/exit brokerage, spread and slippage. The minimum price is NOT a liquidity test. "
               "Historical average dollar volume, real bid/ask fills, stop-loss/take-profit paths and portfolio drawdown "
               "are not available from this sampled signal table and are not simulated.")
    if st.button("Run strategy optimiser",key="opt_run"):
        with st.spinner("Comparing historical strategies..."):
            result,error=run_strategy_optimiser(bt,costs,min_price)
        st.session_state["opt_result"]={"result":result,"error":error,"costs":costs,"price":min_price,
                                         "n":len(bt),"last":str(bt["Signal date"].max())}
    saved=st.session_state.get("opt_result")
    if saved:
        if (saved["costs"],saved["price"],saved["n"],saved["last"])!=(costs,min_price,len(bt),str(bt["Signal date"].max())):
            st.info("Settings or historical sample changed. Run the optimiser again.")
        elif saved["error"]:
            st.warning(saved["error"])
        else:
            result=saved["result"]
            st.caption(f"Compared {result['grid_size']} rule/horizon combinations; "
                       f"validation begins {result['validation_start']}, unseen test begins {result['test_start']}. "
                       f"Rows: reserved training {result['train_rows']:,}, validation {result['validation_rows']:,}, "
                       f"test {result['test_rows']:,}.")
            st.dataframe(result["table"],use_container_width=True,hide_index=True)
            st.download_button("Download optimiser results CSV",result["table"].to_csv(index=False).encode(),
                               file_name="asx_strategy_optimiser_results.csv",mime="text/csv",key="opt_csv")
            st.warning("Exploratory only: fixed-horizon close returns assume fills at the signal close, "
                       "overlapping trades are not portfolio returns, historical delisted shares are missing, "
                       "and the displayed historical sample is downsampled by ticker. "
                       "No max drawdown, stop-loss/take-profit execution or benchmark outperformance is claimed. "
                       "Validate the best rule on fresh, never-used signals before risking capital.")



def simulate_historical_portfolio(signals, rule, hold_days, start_date, capital=10000.0,
                                  position_pct=10, max_positions=5, costs_pct=0.8,
                                  min_price=0.10, min_dollar_volume=20000.0,
                                  participation_pct=2.0, stop_pct=0.0,
                                  target_pct=0.0, trailing_pct=0.0,
                                  rank_method="Score", max_daily_volatility=0.0, require_uptrend=False):
    """Chronological daily-bar research simulation. Stops are conservative on ambiguous bars.

    Signals known at close -> next available opening entry; prior 20-session liquidity.
    Exit stop/target based on daily OHLC, open gaps filled at opening price.
    Trailing peak updated AFTER evaluating each session to avoid lookahead.
    """
    data=signals.copy()
    data['Signal date']=pd.to_datetime(data['Signal date'],errors='coerce').dt.normalize()
    for col in ('Score','VPA Score','Accumulation Score','Distribution Safety'):
        data[col]=pd.to_numeric(data[col],errors='coerce') if col in data else np.nan
    score,vpa,accum,safety=rule
    mask=(data['Score']>=score)&(data['Signal date']>=pd.Timestamp(start_date))
    if vpa is not None: mask &= data['VPA Score']>=vpa
    if accum is not None: mask &= data['Accumulation Score']>=accum
    if safety is not None: mask &= data['Distribution Safety']>=safety
    data=data.loc[mask].dropna(subset=['Signal date','Ticker']).copy()
    if data.empty: return None,'No signals qualified in the later test period.'
    price_map={}; candidates=[]
    skipped={'missing prices':0,'liquidity':0,'price':0,'no next open':0,'volatility filter':0,'trend filter':0,'trend below MA':0,'trend missing MA':0,'trend eligible before volatility':0,'trend passes before volatility':0,'trend fails before volatility':0,'trend missing before volatility':0}
    for ticker,group in data.groupby('Ticker'):
        bars=load_historical_prices(str(ticker))
        if bars.empty: skipped['missing prices']+=len(group); continue
        bars=bars[~bars.index.duplicated(keep='last')].sort_index().copy()
        for col in ('Open','High','Low','Close','Volume'):
            bars[col]=pd.to_numeric(bars[col],errors='coerce')
        bars['Prior dollar volume']=(bars['Close']*bars['Volume']).shift(1).rolling(20,min_periods=20).mean()
        # Entry filters only use bars completed BEFORE the entry session.
        bars['Prior volatility %']=bars['Close'].pct_change().shift(1).rolling(20,min_periods=20).std()*100
        bars['Prior 50D MA']=bars['Close'].shift(1).rolling(50,min_periods=50).mean()
        price_map[ticker]=bars
        for _,row in group.iterrows():
            ix=int(bars.index.searchsorted(row['Signal date'],side='right'))
            if ix>=len(bars): skipped['no next open']+=1; continue
            entry=bars.iloc[ix]
            if not np.isfinite(entry['Open']) or entry['Open']<=0: skipped['no next open']+=1; continue
            if entry['Open']<min_price: skipped['price']+=1; continue
            adv=entry['Prior dollar volume']
            if not np.isfinite(adv) or adv<min_dollar_volume: skipped['liquidity']+=1; continue
            # Independent trend diagnostics: measure trend BEFORE the volatility filter,
            # while keeping the actual trading filter order unchanged.
            prior_close = bars.iloc[ix-1]['Close'] if ix > 0 else np.nan
            prior_ma = entry['Prior 50D MA']
            skipped['trend eligible before volatility'] += 1
            if not np.isfinite(prior_close) or not np.isfinite(prior_ma):
                skipped['trend missing before volatility'] += 1
            elif prior_close > prior_ma:
                skipped['trend passes before volatility'] += 1
            else:
                skipped['trend fails before volatility'] += 1
            if max_daily_volatility>0 and (not np.isfinite(entry['Prior volatility %']) or entry['Prior volatility %']>max_daily_volatility):
                skipped.setdefault('volatility filter',0); skipped['volatility filter']+=1; continue
            if require_uptrend and (not np.isfinite(entry['Prior 50D MA']) or not np.isfinite(bars.iloc[ix-1]['Close']) or bars.iloc[ix-1]['Close']<=entry['Prior 50D MA']):
                skipped['trend filter'] += 1
                if not np.isfinite(prior_close) or not np.isfinite(prior_ma):
                    skipped['trend missing MA'] += 1
                else:
                    skipped['trend below MA'] += 1
                continue
            end_ix=ix+int(hold_days)-1
            if end_ix>=len(bars): skipped['missing prices']+=1; continue
            window=bars.iloc[ix:end_ix+1]
            if window[['Open','High','Low','Close']].isna().any().any() or (window[['Open','High','Low','Close']]<=0).any().any():
                skipped['missing prices']+=1; continue
            rank=float(row['Score'])
            if rank_method=='Score + VPA + Accumulation':
                rank += 0.2*(float(row['VPA Score']) if pd.notna(row['VPA Score']) else 0)
                rank += 0.2*(float(row['Accumulation Score']) if pd.notna(row['Accumulation Score']) else 0)
            candidates.append({'ticker':ticker,'entry_date':bars.index[ix],
                               'deadline':bars.index[end_ix], 'entry_price':float(entry['Open']),
                               'adv':float(adv),'score':float(row['Score']),'rank':rank})
    if not candidates: return None,'No trades passed cached-price and liquidity checks.'
    candidates.sort(key=lambda x:(x['entry_date'],-x['rank'],x['ticker']))
    end_date=max(x['deadline'] for x in candidates)
    dates=sorted(set().union(*(set(b.index[(b.index>=pd.Timestamp(start_date)) &
                                   (b.index<=end_date)]) for b in price_map.values())))
    by_date={}
    for x in candidates: by_date.setdefault(x['entry_date'],[]).append(x)
    cash=float(capital); positions={}; ledger=[]; equity_rows=[]; last_prices={}
    half_cost=costs_pct/200.0
    rejected={'full portfolio':0,'already held':0,'cash or participation':0}
    for day in dates:
        # Open new trades at session open. Signals from previous close only.
        for x in by_date.get(day,[]):
            t=x['ticker']
            if t in positions: rejected['already held']+=1; continue
            if len(positions)>=max_positions: rejected['full portfolio']+=1; continue
            marked=cash+sum(pos['shares']*last_prices.get(tk,pos['entry_price']) for tk,pos in positions.items())
            budget=min(marked*position_pct/100.0,x['adv']*participation_pct/100.0,cash)
            shares=int(budget/(x['entry_price']*(1+half_cost)))
            if shares<1: rejected['cash or participation']+=1; continue
            spend=shares*x['entry_price']*(1+half_cost)
            cash-=spend
            positions[t]={**x,'shares':shares,'spend':spend,'peak':x['entry_price']}
        # Evaluate exits using today's OHLC; stop gets priority if both hit.
        for t,pos in list(positions.items()):
            bars=price_map[t]
            if day not in bars.index: continue
            bar=bars.loc[day]
            op,hi,lo,cl=(float(bar[k]) for k in ('Open','High','Low','Close'))
            if not all(np.isfinite(z) and z>0 for z in (op,hi,lo,cl)): continue
            fixed_stop=pos['entry_price']*(1-stop_pct/100) if stop_pct>0 else -np.inf
            trailing_stop=pos['peak']*(1-trailing_pct/100) if trailing_pct>0 else -np.inf
            stop=max(fixed_stop,trailing_stop)
            target=pos['entry_price']*(1+target_pct/100) if target_pct>0 else np.inf
            exit_price=None; reason=None
            if op<=stop:
                exit_price=op; reason='Stop gap at open'
            elif op>=target:
                exit_price=op; reason='Target gap at open'
            elif lo<=stop:
                exit_price=stop; reason='Stop (conservative if both hit)'
            elif hi>=target:
                exit_price=target; reason='Take profit'
            elif day>=pos['deadline']:
                exit_price=cl; reason='Holding period'
            if exit_price is not None:
                proceeds=pos['shares']*exit_price*(1-half_cost)
                cash+=proceeds
                ledger.append({'Ticker':t,'Entry date':str(pos['entry_date'].date()),
                               'Exit date':str(day.date()),'Shares':pos['shares'],
                               'Entry open':pos['entry_price'],'Exit price':exit_price,
                               'Exit reason':reason,'Cost AUD':round(pos['spend'],2),
                               'Proceeds AUD':round(proceeds,2),'Net P&L AUD':round(proceeds-pos['spend'],2),
                               'Net return %':round((proceeds/pos['spend']-1)*100,3),
                               'Prior 20d avg dollar volume':round(pos['adv'],2)})
                del positions[t]
            else:
                # Peak updated after intraday stop evaluation; no intraday sequencing assumed.
                pos['peak']=max(pos['peak'],hi)
        for t in positions:
            bars=price_map[t]
            if day in bars.index and pd.notna(bars.loc[day,'Close']):
                last_prices[t]=float(bars.loc[day,'Close'])
        equity=cash+sum(pos['shares']*last_prices.get(t,pos['entry_price']) for t,pos in positions.items())
        equity_rows.append({'Date':day,'Equity AUD':equity,'Cash AUD':cash,'Open positions':len(positions)})
    eq=pd.DataFrame(equity_rows)
    eq['Peak AUD']=eq['Equity AUD'].cummax().clip(lower=capital)
    eq['Drawdown %']=(eq['Equity AUD']/eq['Peak AUD']-1)*100
    trades=pd.DataFrame(ledger)
    end_equity=float(eq['Equity AUD'].iloc[-1]); days=max((eq['Date'].iloc[-1]-eq['Date'].iloc[0]).days,1)
    stats={'Starting balance AUD':capital,'Ending balance AUD':end_equity,
           'Total return %':(end_equity/capital-1)*100,
           'Annualised return %':((end_equity/capital)**(365.25/days)-1)*100 if end_equity>0 else -100,
           'Maximum drawdown %':float(eq['Drawdown %'].min()),
           'Closed trades':len(trades),'Win rate %':float((trades['Net P&L AUD']>0).mean()*100) if len(trades) else np.nan,
           'Open positions at end':len(positions),'Candidate signals':len(candidates),'Eligible tickers':len(set(x['ticker'] for x in candidates))}
    return {'equity':eq,'trades':trades,'stats':stats,'skipped':skipped,'rejected':rejected},None


def analyse_trend_filter_outcomes(signals, rule, hold_days, costs_pct, min_price,
                                  min_dollar_volume, max_daily_volatility, start_date=None):
    """Compare signal outcomes on both sides of the 50D filter, without portfolio sizing.

    All checks use data available before the next-session open. Fixed-horizon
    outcomes use that open and a later close; no stop/target execution is implied.
    """
    data = signals.copy()
    data['Signal date'] = pd.to_datetime(data['Signal date'], errors='coerce').dt.normalize()
    for col in ('Score', 'VPA Score', 'Accumulation Score', 'Distribution Safety'):
        data[col] = pd.to_numeric(data[col], errors='coerce') if col in data else np.nan
    score, vpa, accum, safety = rule
    mask = data['Score'] >= score
    if vpa is not None: mask &= data['VPA Score'] >= vpa
    if accum is not None: mask &= data['Accumulation Score'] >= accum
    if safety is not None: mask &= data['Distribution Safety'] >= safety
    if start_date is not None: mask &= data['Signal date'] >= pd.Timestamp(start_date)
    data = data.loc[mask].dropna(subset=['Signal date', 'Ticker'])
    rows = []
    skipped = {'missing cached prices': 0, 'no next opening': 0,
               'minimum price': 0, 'liquidity': 0, 'volatility': 0,
               'insufficient future sessions': 0, 'invalid future prices': 0}
    horizons = (5, 10, 20, 30)
    for ticker, group in data.groupby('Ticker'):
        bars = load_historical_prices(str(ticker))
        if bars.empty:
            skipped['missing cached prices'] += len(group)
            continue
        bars = bars[~bars.index.duplicated(keep='last')].sort_index().copy()
        for col in ('Open', 'Close', 'Volume'):
            bars[col] = pd.to_numeric(bars[col], errors='coerce')
        close = bars['Close']
        prior_ma = close.shift(1).rolling(50, min_periods=50).mean()
        prior_vol = close.pct_change().shift(1).rolling(20, min_periods=20).std() * 100
        prior_adv = (close * bars['Volume']).shift(1).rolling(20, min_periods=20).mean()
        for _, signal in group.iterrows():
            ix = int(bars.index.searchsorted(signal['Signal date'], side='right'))
            if ix >= len(bars):
                skipped['no next opening'] += 1
                continue
            op = num(bars['Open'].iloc[ix])
            if op is None or not np.isfinite(op) or op <= 0:
                skipped['no next opening'] += 1
                continue
            if op < min_price:
                skipped['minimum price'] += 1
                continue
            adv = num(prior_adv.iloc[ix])
            if adv is None or not np.isfinite(adv) or adv < min_dollar_volume:
                skipped['liquidity'] += 1
                continue
            volatility = num(prior_vol.iloc[ix])
            if max_daily_volatility > 0 and (volatility is None or not np.isfinite(volatility) or volatility > max_daily_volatility):
                skipped['volatility'] += 1
                continue
            previous_close = num(close.iloc[ix - 1]) if ix else None
            moving_average = num(prior_ma.iloc[ix])
            if previous_close is None or moving_average is None or not np.isfinite(previous_close) or not np.isfinite(moving_average):
                classification = 'Rejected: missing 50D data'
            elif previous_close > moving_average:
                classification = 'Accepted: above 50D MA'
            else:
                classification = 'Rejected: at/below 50D MA'
            record = {'Ticker': str(ticker), 'Signal date': str(signal['Signal date'].date()),
                      'Entry date': str(bars.index[ix].date()), 'Trend group': classification,
                      'Previous close': previous_close, 'Prior 50D MA': moving_average,
                      'Entry open': op, 'Prior volatility %': volatility}
            for horizon in horizons:
                # Horizon 1 is entry-session close; horizon 5 is the fifth session close.
                exit_ix = ix + horizon - 1
                outcome = np.nan
                if exit_ix < len(bars):
                    exit_price = num(close.iloc[exit_ix])
                    if exit_price is not None and np.isfinite(exit_price) and exit_price > 0:
                        outcome = (exit_price / op - 1) * 100 - costs_pct
                record[f'Return {horizon}d net %'] = outcome
            exit_ix = ix + int(hold_days) - 1
            if exit_ix >= len(bars):
                skipped['insufficient future sessions'] += 1
                continue
            exit_price = num(close.iloc[exit_ix])
            if exit_price is None or not np.isfinite(exit_price) or exit_price <= 0:
                skipped['invalid future prices'] += 1
                continue
            record['Strategy-hold net %'] = (exit_price / op - 1) * 100 - costs_pct
            rows.append(record)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame(), skipped
    summary = []
    for group_name, part in detail.groupby('Trend group'):
        outcome = part['Strategy-hold net %'].dropna()
        wins = outcome[outcome > 0]
        losses = outcome[outcome < 0]
        item = {'Trend group': group_name, 'Signals': len(part),
                'Hold win rate %': float((outcome > 0).mean() * 100),
                'Hold average net %': float(outcome.mean()),
                'Average winner %': float(wins.mean()) if len(wins) else np.nan,
                'Average loser %': float(losses.mean()) if len(losses) else np.nan}
        for horizon in horizons:
            values = part[f'Return {horizon}d net %'].dropna()
            item[f'{horizon}d observations'] = len(values)
            item[f'{horizon}d avg net %'] = float(values.mean()) if len(values) else np.nan
            item[f'{horizon}d win rate %'] = float((values > 0).mean() * 100) if len(values) else np.nan
        summary.append(item)
    return detail, pd.DataFrame(summary), skipped


def render_filter_effectiveness_analyser(bt, rule, hold, saved, liquidity):
    st.markdown('### Filter Effectiveness Analyser — 50-day trend')
    st.caption('Compares the SAME score-qualified signals above and at/below their prior 50-day average. '
               'Checks price, liquidity and volatility before grouping. Entry is the next available session open; '
               'returns use later session closes and subtract estimated round-trip costs. '
               'This is a signal-level comparison, NOT a portfolio simulation.')
    a, b = st.columns(2)
    with a:
        vol = st.selectbox('Volatility cap for comparison (%)', [0, 2, 3, 4, 5, 7, 10],
                           index=5, format_func=lambda x: 'Off' if x == 0 else f'{x}%', key='fe_vol')
    with b:
        sample = st.selectbox('Historical sample', ['All cached signals', 'Latest 25% of signal dates'], key='fe_sample')
    signature = (len(bt), str(pd.to_datetime(bt['Signal date'], errors='coerce').max()),
                 tuple(rule), hold, saved['costs'], saved['price'], liquidity, vol, sample)
    if st.button('Analyse accepted vs rejected shares', key='fe_run'):
        start = None
        if sample != 'All cached signals':
            dates = pd.to_datetime(bt['Signal date'], errors='coerce').dropna().sort_values().unique()
            if len(dates): start = pd.Timestamp(dates[int(len(dates) * .75)])
        with st.spinner('Comparing historical outcomes from cached daily prices...'):
            detail, summary, skipped = analyse_trend_filter_outcomes(
                bt, rule, hold, saved['costs'], saved['price'], liquidity, vol, start)
        st.session_state['fe_result'] = {'signature': signature, 'detail': detail,
                                         'summary': summary, 'skipped': skipped}
    result = st.session_state.get('fe_result')
    if not result: return
    if result['signature'] != signature:
        st.info('Settings changed. Run the comparison again.')
        return
    if result['detail'].empty:
        st.warning('No eligible signals with complete future prices were found. Check cached history and settings.')
    else:
        st.dataframe(result['summary'].round(2), hide_index=True, use_container_width=True)
        st.download_button('Download filter comparison summary CSV',
                           result['summary'].to_csv(index=False).encode(),
                           file_name='asx_filter_effectiveness_summary.csv', mime='text/csv', key='fe_summary_csv')
        with st.expander('Individual signal outcomes and CSV'):
            st.dataframe(result['detail'].round(3), hide_index=True, use_container_width=True)
            st.download_button('Download all accepted/rejected signal outcomes CSV',
                               result['detail'].to_csv(index=False).encode(),
                               file_name='asx_filter_effectiveness_signals.csv', mime='text/csv', key='fe_detail_csv')
    with st.expander('Excluded signals and data availability'):
        st.dataframe(pd.DataFrame([{'Reason': k, 'Signals': v} for k, v in result['skipped'].items()]),
                     hide_index=True, use_container_width=True)
    st.warning('Exploratory historical comparison only. Signals can overlap, and missing/delisted stocks can bias results. '
               'A rejected signal is not an executed losing trade; results omit portfolio constraints, stop/target paths, '
               'market impact and taxes. Do not use the same sample repeatedly to tune and then claim independent validation.')


def render_walk_forward_testing(bt, rule, hold, saved, capital, positions, allocation, liquidity, participation):
    """Rolling, chronological parameter selection; only earlier windows choose each fold's rule."""
    st.markdown('### Walk-forward strategy testing')
    st.caption('Each fold selects a volatility/trend combination using an earlier validation window, then evaluates that selection once on the next window. A 45-calendar-day gap separates windows. The underlying score rule and holding period were selected elsewhere and are NOT independently validated by this test.')
    nfolds=st.selectbox('Number of walk-forward tests', [3,4],key='wf_nfolds')
    min_count=st.number_input('Minimum validation closed trades per fold',min_value=5,max_value=100,value=15,step=5,key='wf_min_count')
    dates=pd.to_datetime(bt['Signal date'],errors='coerce').dropna().dt.normalize().sort_values().unique()
    signature=(len(bt),str(pd.to_datetime(bt['Signal date']).max()),tuple(rule),hold,saved['costs'],saved['price'],capital,positions,allocation,liquidity,participation,nfolds,min_count)
    if st.button('Run walk-forward tests',key='wf_run'):
        if len(dates)<160:
            st.session_state['wf_result']={'signature':signature,'error':'Need at least 160 distinct historical signal dates.'}
        else:
            # Expanding validation followed by a strictly later test. Avoid adjacent outcome overlap.
            boundaries=np.linspace(int(len(dates)*0.38),int(len(dates)*0.88),nfolds+1,dtype=int)
            configs=[(v,t) for v in (0,2,3,4,5,7,10) for t in (False,True)]
            rows=[]; diag=[]; progress=st.progress(0,text='Testing chronological folds...')
            total=nfolds*(len(configs)+1); done=0
            signal_dates=pd.to_datetime(bt['Signal date'],errors='coerce')
            for f in range(nfolds):
                val_start=pd.Timestamp(dates[0]); val_end=pd.Timestamp(dates[boundaries[f]])-pd.Timedelta(days=45)
                test_start=pd.Timestamp(dates[boundaries[f]]); test_end=pd.Timestamp(dates[boundaries[f+1]])
                validation=bt[(signal_dates>=val_start)&(signal_dates<=val_end)]
                candidates=[]
                for vol,trend in configs:
                    r,e=simulate_historical_portfolio(validation,rule,hold,val_start,capital,allocation,positions,saved['costs'],saved['price'],liquidity,participation,0,0,0,'Score',vol,trend)
                    if r is not None and e is None:
                        a=r['stats']; n=int(a['Closed trades'])
                        if n>=min_count:
                            pnl=pd.to_numeric(r['trades']['Net P&L AUD'],errors='coerce')
                            loss=-float(pnl[pnl<0].sum()); profit=float(pnl[pnl>0].sum())
                            pf=profit/loss if loss>0 else (10.0 if profit>0 else 0.0)
                            score=float(a['Total return %'])+1.5*float(a['Maximum drawdown %'])
                            candidates.append((score,pf,vol,trend,n))
                        diag.append({'Fold':f+1,'Volatility max %':vol,'Trend ON':trend,'Validation trades':n,'Trend rejected':r['skipped'].get('trend filter',0),'Volatility rejected':r['skipped'].get('volatility filter',0),'Trend checked before volatility':r['skipped'].get('trend eligible before volatility',0),'Above 50D MA':r['skipped'].get('trend passes before volatility',0),'At/below 50D MA':r['skipped'].get('trend fails before volatility',0),'Missing 50D MA':r['skipped'].get('trend missing before volatility',0),'Trend rejected below MA':r['skipped'].get('trend below MA',0),'Trend rejected missing MA':r['skipped'].get('trend missing MA',0)})
                    done+=1; progress.progress(done/total)
                if candidates:
                    _,_,vol,trend,n=max(candidates,key=lambda x:(x[0],x[1]))
                    test=bt[(signal_dates>=test_start)&(signal_dates<test_end)]
                    r,e=simulate_historical_portfolio(test,rule,hold,test_start,capital,allocation,positions,saved['costs'],saved['price'],liquidity,participation,0,0,0,'Score',vol,trend)
                    if r is not None and e is None:
                        a=r['stats']; rows.append({'Fold':f+1,'Validation through':str(val_end.date()),'Test start':str(test_start.date()),'Test end':str(test_end.date()),'Chosen volatility %':vol,'Trend ON':trend,'Validation trades':n,'Test trades':a['Closed trades'],'Test return %':a['Total return %'],'Test max drawdown %':a['Maximum drawdown %'],'Test win rate %':a['Win rate %']})
                    else:
                        rows.append({'Fold':f+1,'Test start':str(test_start.date()),'Error':str(e)})
                else:
                    rows.append({'Fold':f+1,'Test start':str(test_start.date()),'Error':'No configuration met minimum trades'})
                done+=1; progress.progress(done/total)
            progress.empty()
            st.session_state['wf_result']={'signature':signature,'rows':pd.DataFrame(rows),'diagnostics':pd.DataFrame(diag)}
    record=st.session_state.get('wf_result')
    if not record:return
    if record['signature']!=signature:
        st.info('Settings changed. Rerun walk-forward tests.');return
    if record.get('error'):
        st.warning(record['error']);return
    frame=record['rows']
    st.dataframe(frame.round(2),hide_index=True,use_container_width=True)
    st.download_button('Download walk-forward results CSV',frame.to_csv(index=False).encode(),file_name='asx_walk_forward_results.csv',mime='text/csv',key='wf_csv')
    with st.expander('Filter rejection diagnostics by fold'):
        d=record['diagnostics'];st.dataframe(d,hide_index=True,use_container_width=True)
        st.download_button('Download filter diagnostics CSV',d.to_csv(index=False).encode(),file_name='asx_filter_diagnostics.csv',mime='text/csv',key='wf_diag_csv')
        if len(d):
            st.caption('Independent trend counts are measured after price/liquidity checks but BEFORE volatility rejection. They show whether the trend filter is redundant, missing data, or simply hidden by the volatility filter. These are candidate signals, not executed trades.')
            paired=d.pivot_table(index=['Fold','Volatility max %'],columns='Trend ON',values='Trend rejected',aggfunc='first')
            if True in paired.columns and (paired[True].fillna(0)==0).all():
                st.warning('Trend ON rejected zero signals in every tested validation fold. Verify price-history availability, signal timestamps and moving-average calculation before relying on this filter.')
    st.warning('Research only: folds share a historical dataset, may contain repeated tickers, and the score rule was previously selected. No guarantee of live performance. Validate on future paper trades and account for delistings, corporate actions and execution.')


def render_automatic_strategy_testing(bt):
    """Validation-only parameter search with one later chronological holdout run."""
    st.subheader('Automatic Strategy Testing — experimental')
    st.caption('Automatically compares entry quality filters on an earlier validation window, then runs only the validation winner on a later holdout. The holdout is not reusable as an untouched test after you inspect it.')
    saved=st.session_state.get('opt_result')
    if not saved or saved.get('error') or not saved.get('result'):
        st.info('Run the Trading Strategy Optimiser above first.')
        return
    if saved.get('n')!=len(bt) or saved.get('last')!=str(bt['Signal date'].max()):
        st.info('Historical signals have changed. Rerun the optimiser first.')
        return
    opt=saved['result']
    top=opt['table'].iloc[0]
    rule=(int(top['Score min']), None if pd.isna(top['VPA min']) else float(top['VPA min']),
          None if pd.isna(top['Accumulation min']) else float(top['Accumulation min']),
          None if pd.isna(top['Distribution risk max']) else 100-float(top['Distribution risk max']))
    hold=int(top['Hold days'])
    # Keep a 45-day separation between signal windows to reduce overlap of outcomes.
    dates=pd.to_datetime(bt['Signal date'],errors='coerce').dropna().sort_values().unique()
    if len(dates)<90:
        st.info('At least 90 distinct historical signal dates are needed for time-separated testing.')
        return
    boundary=pd.Timestamp(dates[int(len(dates)*0.8)]).normalize()
    val_start=pd.Timestamp(dates[int(len(dates)*0.6)]).normalize()
    val_end=boundary-pd.Timedelta(days=45)
    if val_end<=val_start:
        st.info('Not enough time between validation and holdout periods.')
        return
    a,b,c=st.columns(3)
    with a:
        capital=st.number_input('Test starting capital (AUD)',min_value=1000.0,value=10000.0,step=1000.0,key='auto_capital')
        positions=st.slider('Test maximum concurrent positions',1,20,5,key='auto_positions')
    with b:
        allocation=st.slider('Test allocation per trade (%)',1,50,10,key='auto_allocation')
        liquidity=st.number_input('Test minimum daily dollar volume (AUD)',min_value=0.0,value=20000.0,step=10000.0,key='auto_liquidity')
    with c:
        participation=st.slider('Test maximum daily-volume participation (%)',0.1,10.0,2.0,0.1,key='auto_participation')
        min_trades=st.slider('Minimum validation trades',5,50,15,key='auto_min_trades')
    st.caption(f'Validation signals: {val_start.date()} to {val_end.date()}; holdout signals from {boundary.date()}. The rule and holding period come from validation-ranked optimiser results. Costs: {saved["costs"]:.1f}%.')
    # Predefined, limited grid: avoid an unrestricted search that would overfit small samples.
    configs=[(v,t) for v in (0,2,3,4,5,7,10) for t in (False,True)]
    signature=(len(bt),str(bt['Signal date'].max()),tuple(rule),hold,capital,positions,allocation,liquidity,participation,min_trades,saved['costs'],saved['price'])
    if st.button('Run automatic strategy comparison',key='auto_run'):
        rows=[]
        validation=bt[(pd.to_datetime(bt['Signal date'])>=val_start)&(pd.to_datetime(bt['Signal date'])<=val_end)].copy()
        bar=st.progress(0,text='Comparing entry filters on validation data...')
        for i,(vol,trend) in enumerate(configs):
            result,error=simulate_historical_portfolio(validation,rule,hold,val_start,capital,
                allocation,positions,saved['costs'],saved['price'],liquidity,participation,
                0,0,0,'Score',vol,trend)
            if result and not error:
                stats=result['stats']; trades=result['trades']
                n=int(stats['Closed trades'])
                pnl=pd.to_numeric(trades['Net P&L AUD'],errors='coerce') if not trades.empty else pd.Series(dtype=float)
                gross_profit=float(pnl[pnl>0].sum()); gross_loss=-float(pnl[pnl<0].sum())
                pf=gross_profit/gross_loss if gross_loss>0 else (float('inf') if gross_profit>0 else 0)
                rows.append({'Volatility max %':vol,'Require 50D uptrend':trend,
                    'Total return %':stats['Total return %'],'Maximum drawdown %':stats['Maximum drawdown %'],
                    'Win rate %':stats['Win rate %'],'Trades':n,'Profit factor':pf})
            bar.progress((i+1)/len(configs),text=f'Tested {i+1}/{len(configs)} combinations')
        bar.empty()
        frame=pd.DataFrame(rows)
        if frame.empty:
            st.session_state['auto_result']={'signature':signature,'error':'No valid validation simulations.','table':None}
        else:
            # Prefer net profitability and manageable drawdown; reject sparse samples.
            eligible=frame[frame['Trades']>=min_trades].copy()
            if eligible.empty:
                st.session_state['auto_result']={'signature':signature,'error':'No combination met the minimum validation trade count.','table':frame}
            else:
                eligible['Ranking score']=eligible['Total return %']+eligible['Maximum drawdown %']*1.5
                eligible=eligible.sort_values(['Ranking score','Profit factor'],ascending=False)
                winner=eligible.iloc[0]
                holdout=bt[pd.to_datetime(bt['Signal date'])>=boundary].copy()
                final,final_error=simulate_historical_portfolio(holdout,rule,hold,boundary,capital,
                    allocation,positions,saved['costs'],saved['price'],liquidity,participation,
                    0,0,0,'Score',float(winner['Volatility max %']),bool(winner['Require 50D uptrend']))
                st.session_state['auto_result']={'signature':signature,'error':final_error,'table':eligible,
                    'winner':winner.to_dict(),'holdout':final,'validation_start':str(val_start.date()),
                    'validation_end':str(val_end.date()),'holdout_start':str(boundary.date())}
    record=st.session_state.get('auto_result')
    if not record: return
    if record['signature']!=signature:
        st.info('Inputs changed. Rerun the automatic comparison.'); return
    if record.get('table') is not None:
        st.markdown('**Validation ranking (not holdout results)**')
        st.dataframe(record['table'].replace([np.inf,-np.inf],np.nan).round(2),hide_index=True,use_container_width=True)
        st.download_button('Download strategy comparison CSV',record['table'].to_csv(index=False).encode(),
                           file_name='asx_automatic_strategy_comparison.csv',mime='text/csv',key='auto_csv')
    if record.get('error'):
        st.warning(record['error']); return
    winner=record['winner']; result=record['holdout']; stats=result['stats']
    st.markdown('**Later holdout — selected strategy only**')
    st.write(f"Selected on validation: volatility {'OFF' if winner['Volatility max %']==0 else str(int(winner['Volatility max %']))+'%'}; 50-day uptrend {'ON' if winner['Require 50D uptrend'] else 'OFF'}.")
    metrics=st.columns(4)
    metrics[0].metric('Holdout total return',f"{stats['Total return %']:+.2f}%")
    metrics[1].metric('Holdout max drawdown',f"{stats['Maximum drawdown %']:.2f}%")
    metrics[2].metric('Holdout win rate',f"{stats['Win rate %']:.1f}%")
    metrics[3].metric('Holdout closed trades',str(stats['Closed trades']))
    st.line_chart(result['equity'].set_index('Date')['Equity AUD'])
    st.markdown('**Entry filter diagnostics**')
    st.caption('Counts are rejected signal rows, before portfolio capacity constraints. A zero trend rejection with trend ON may indicate the filter is ineffective or the eligible universe is already entirely above its moving average.')
    st.dataframe(pd.DataFrame([{'Reason':k,'Rejected signals':v} for k,v in result.get('skipped',{}).items()]),hide_index=True,use_container_width=True)
    st.caption(f"Eligible candidates: {stats['Candidate signals']}; eligible tickers: {stats.get('Eligible tickers', 'unknown')}.")
    render_walk_forward_testing(bt, rule, hold, saved, capital, positions, allocation, liquidity, participation)
    st.warning('The optimiser rule was already selected using this historical dataset, so this is not a fully independent test of the entire strategy. Repeated runs and parameter changes on the same holdout introduce overfitting. Signals are sampled; delisted shares, execution uncertainty and corporate-action data can affect results. Forward paper testing is needed.')


def render_portfolio_simulator(bt):
    st.subheader('Trade-by-Trade Portfolio Simulator — experimental')
    st.caption('Uses the validation-ranked strategy from the optimiser and tests it chronologically on the later period. '
               'Entries are at the next session open, exits at the close after the chosen number of trading sessions. '
               'Uses cached historical daily prices, not broker executions.')
    saved=st.session_state.get('opt_result')
    if not saved or saved.get('error') or not saved.get('result'):
        st.info('Run the Trading Strategy Optimiser above first to select a rule using validation data only.')
        return
    opt=saved['result']
    if saved.get('n')!=len(bt) or saved.get('last')!=str(bt['Signal date'].max()):
        st.info('The historical sample changed. Run the optimiser again before simulating.')
        return
    table=opt['table']
    if table.empty: return
    rank=st.selectbox('Strategy chosen by validation rank',list(range(1,len(table)+1)),index=0,key='sim_rank')
    r=table.iloc[rank-1]
    rule=(int(r['Score min']),None if pd.isna(r['VPA min']) else float(r['VPA min']),
          None if pd.isna(r['Accumulation min']) else float(r['Accumulation min']),
          None if pd.isna(r['Distribution risk max']) else 100-float(r['Distribution risk max']))
    st.write(f"Score ≥{rule[0]}, VPA ≥{rule[1] if rule[1] is not None else 'any'}, accumulation ≥{rule[2] if rule[2] is not None else 'any'}, "
             f"distribution risk ≤{r['Distribution risk max'] if pd.notna(r['Distribution risk max']) else 'any'}; hold {int(r['Hold days'])} sessions")
    a,b,c=st.columns(3)
    with a:
        capital=st.number_input('Starting capital (AUD)',min_value=1000.0,value=10000.0,step=1000.0,key='sim_capital')
        max_positions=st.slider('Maximum concurrent positions',1,20,5,key='sim_max_positions')
    with b:
        allocation=st.slider('Target allocation per trade (%)',1,50,10,key='sim_allocation')
        liquidity=st.number_input('Minimum prior 20-session avg dollar volume (AUD)',min_value=0.0,value=20000.0,step=10000.0,key='sim_liquidity')
    with c:
        participation=st.slider('Maximum position as % of average daily dollar volume',0.1,10.0,2.0,0.1,key='sim_participation')
        st.caption(f"Round-trip costs {saved['costs']:.1f}%; minimum share price ${saved['price']:.2f} (from optimiser).")
    st.markdown('**Experimental exits and entry ranking**')
    d,e,f=st.columns(3)
    with d:
        stop=st.selectbox('Stop-loss (%)',[0,5,8,10,15,20],index=0,key='sim_stop',format_func=lambda v:'Off' if v==0 else f'{v}%')
    with e:
        target=st.selectbox('Take-profit (%)',[0,10,15,20,30],index=0,key='sim_target',format_func=lambda v:'Off' if v==0 else f'{v}%')
    with f:
        trailing=st.selectbox('Trailing stop (%)',[0,5,8,10,15,20],index=0,key='sim_trailing',format_func=lambda v:'Off' if v==0 else f'{v}%')
    ranking=st.selectbox('Choose candidates when positions are limited',
                         ['Score','Score + VPA + Accumulation'],key='sim_ranking')
    st.markdown('**Consistent Returns — optional entry quality filters**')
    q1,q2=st.columns(2)
    with q1:
        max_vol=st.selectbox('Maximum prior 20-day daily volatility (%)',
                             [0,2,3,4,5,7,10],index=0,key='sim_quality_vol',
                             format_func=lambda v:'Off' if v==0 else f'{v}%')
    with q2:
        uptrend=st.checkbox('Require previous close above prior 50-day average',
                            value=False,key='sim_quality_trend')
    st.caption('Filters use historical prices available before entry. They can reduce trades and may reduce returns; test against the unchanged baseline.')
    st.caption('Stops/targets use cached daily OHLC. Opening gaps fill at the open; if both intraday thresholds are touched, '
               'the stop is assumed first. Trailing peaks update after each day, so same-day high-to-low ordering is not assumed. '
               'These are approximate fills, not real orders.')
    st.warning('Research simulation only. Cached Yahoo prices may be adjusted or inconsistent across corporate actions; '
               'liquidity uses historical daily volume rather than bid/ask depth. Stop/target fills are approximate, '
               'dividends, tax, benchmark, delisted-stock coverage, or guaranteed fills. Signals are sampled, not all daily opportunities.')
    settings=(rank,capital,max_positions,allocation,liquidity,participation,saved['costs'],saved['price'],len(bt),opt['test_start'],stop,target,trailing,ranking,max_vol,uptrend)
    if st.button('Run trade-by-trade simulation',key='sim_run'):
        with st.spinner('Loading cached daily bars and simulating portfolio...'):
            result,error=simulate_historical_portfolio(bt,rule,int(r['Hold days']),opt['test_start'],capital,
                            allocation,max_positions,saved['costs'],saved['price'],liquidity,participation,stop,target,trailing,ranking,max_vol,uptrend)
        st.session_state['sim_result']={'settings':settings,'result':result,'error':error}
    record=st.session_state.get('sim_result')
    if not record: return
    if record['settings']!=settings:
        st.info('Simulation settings changed. Run again to refresh results.'); return
    if record['error']:
        st.warning(record['error']); return
    result=record['result']; stats=result['stats']
    cols=st.columns(4)
    for col,key,fmt in zip(cols,('Ending balance AUD','Total return %','Maximum drawdown %','Closed trades'),
                           ('${:,.2f}','{:+.2f}%','{:.2f}%','{:,.0f}')):
        col.metric(key,fmt.format(stats[key]))
    st.caption(f"Win rate: {stats['Win rate %']:.1f}% | Annualised return: {stats['Annualised return %']:+.2f}% | "
               f"Candidates: {stats['Candidate signals']:,} | Open positions at end: {stats['Open positions at end']}")
    if stop or target or trailing or ranking!='Score' or max_vol or uptrend:
        st.caption('Comparison with unchanged baseline using identical capital, costs, liquidity and position limits:')
        if st.button('Compare with unchanged baseline',key='sim_compare_baseline'):
            with st.spinner('Running baseline with no exits and score-only ranking...'):
                baseline,base_err=simulate_historical_portfolio(bt,rule,int(r['Hold days']),opt['test_start'],capital,
                    allocation,max_positions,saved['costs'],saved['price'],liquidity,participation)
            if base_err: st.warning(base_err)
            else:
                st.dataframe(pd.DataFrame([
                    {'Version':'Unchanged baseline',**{k:baseline['stats'][k] for k in ('Total return %','Maximum drawdown %','Closed trades','Win rate %')}},
                    {'Version':'Experimental',**{k:stats[k] for k in ('Total return %','Maximum drawdown %','Closed trades','Win rate %')}}
                ]),hide_index=True,use_container_width=True)
                st.caption('Exploratory comparison on the same historical test period. Changing settings based on this test makes it no longer untouched.')
    st.markdown('**Consistent Returns Analyser**')
    trades=result['trades']
    if not trades.empty and 'Net return %' in trades:
        net=pd.to_numeric(trades['Net return %'],errors='coerce').dropna()
        wins=net[net>0]; losses=net[net<0]
        gross_profit=float(pd.to_numeric(trades.loc[trades['Net P&L AUD']>0,'Net P&L AUD']).sum())
        gross_loss=abs(float(pd.to_numeric(trades.loc[trades['Net P&L AUD']<0,'Net P&L AUD']).sum()))
        factor=gross_profit/gross_loss if gross_loss>0 else float('inf') if gross_profit>0 else float('nan')
        longest=streak=0
        for value in net:
            streak=streak+1 if value<=0 else 0
            longest=max(longest,streak)
        m=st.columns(5)
        m[0].metric('Profit factor',f'{factor:.2f}' if np.isfinite(factor) else 'N/A')
        m[1].metric('Avg winner',f'{wins.mean():+.2f}%' if len(wins) else 'N/A')
        m[2].metric('Avg loser',f'{losses.mean():+.2f}%' if len(losses) else 'N/A')
        m[3].metric('Median trade',f'{net.median():+.2f}%')
        m[4].metric('Longest losing streak',str(longest))
        st.caption(f'Positive trades: {len(wins)} | Losing or flat trades: {len(net)-len(wins)} | '
                   f'Average trade: {net.mean():+.2f}% | Trades evaluated: {len(net)}')
        if stats['Total return %']<=0:
            st.warning('This configuration lost money after estimated costs. A higher win rate alone does not make it viable.')
        elif stats['Maximum drawdown %'] < -10:
            st.info('This configuration was profitable in this sample, but historical drawdown exceeded the 10% research target.')
        else:
            st.success('This configuration met the positive-return and under-10%-drawdown research targets on this sample. This is not evidence of future performance.')
    else:
        st.info('No closed trades to evaluate for consistency.')
    st.caption('These results reuse the same historical test data. Repeatedly changing filters against it creates overfitting; '
               'confirm promising rules on a fresh later period and compare with a market benchmark before using real money.')
    st.line_chart(result['equity'].set_index('Date')['Equity AUD'])
    # Optional price-only ASX ETF reference from existing cache. Never fetch future prices.
    bench=load_historical_prices('STW')
    if not bench.empty:
        bench=bench.sort_index()
        bench=bench[(bench.index>=result['equity']['Date'].min()) & (bench.index<=result['equity']['Date'].max())]
        close=pd.to_numeric(bench['Close'],errors='coerce').dropna()
        if len(close)>=30 and close.iloc[0]>0:
            bench_end=capital*close.iloc[-1]/close.iloc[0]
            st.caption(f'STW ETF cached close-price-only reference: ${bench_end:,.2f} from ${capital:,.0f} '
                       f'({(bench_end/capital-1)*100:+.2f}%). Excludes dividends, fees and exact matching execution dates; not a total-return benchmark.')
    else:
        st.caption('Benchmark unavailable: no cached STW ETF daily history. No external benchmark data fetched.')
    st.write('**Closed trades**')
    st.dataframe(result['trades'],use_container_width=True,hide_index=True)
    st.download_button('Download simulated trades CSV',result['trades'].to_csv(index=False).encode(),
                       'asx_simulated_trades.csv','text/csv',key='sim_trades_csv')
    st.download_button('Download daily equity CSV',result['equity'].to_csv(index=False).encode(),
                       'asx_simulated_equity.csv','text/csv',key='sim_equity_csv')
    st.caption(f"Rejected before candidate creation: {result['skipped']}; portfolio constraints: {result['rejected']}.")

def evaluate_learned_holdout(frame, threshold):
    """Train on early matured observations, then compare fixed/adaptive scores on later rows."""
    if frame is None or len(frame)<60:
        return None,"Need at least 60 sampled historical rows to make a time-separated comparison."
    dates=pd.to_datetime(frame["Signal date"],errors="coerce").dropna().sort_values().reset_index(drop=True)
    if len(dates)<60:
        return None,"Not enough valid signal dates for a holdout comparison."
    split_date=dates.iloc[int(len(dates)*0.70)]
    # Exclude training signals whose 20-session outcomes could extend into the test period.
    train_cutoff=split_date-pd.Timedelta(days=45)
    weights,train_count,active,_=learned_weights(train_cutoff.strftime("%Y-%m-%d"))
    test=frame[pd.to_datetime(frame["Signal date"],errors="coerce")>=split_date].copy()
    if not active or train_count<30:
        return None,f"Only {train_count} effective matured examples were available before the holdout date; at least 30 are needed."
    if len(test)<20:
        return None,"The later holdout portion has too few rows to compare reliably."
    test["Fixed score"]=pd.to_numeric(test["Score"],errors="coerce")
    test["Learned score"]=test.apply(lambda row:score_with_weights(row,weights),axis=1)
    result_rows=[]
    top_n=max(1,int(np.ceil(len(test)*0.20)))
    for method,score_col in (("Fixed weights","Fixed score"),("Learned weights","Learned score")):
        selections=[("At score threshold",test[test[score_col]>=threshold]),("Top 20% of scores",test.nlargest(top_n,score_col))]
        for basis,selected in selections:
            for horizon in (5,10,20,30):
                values=pd.to_numeric(selected[f"Return {horizon}d %"],errors="coerce").dropna()
                result_rows.append({"Scoring method":method,"Selection rule":basis,"Horizon (trading days)":horizon,
                    "Signals":len(values),"Positive return %":values.gt(0).mean()*100 if len(values) else np.nan,
                    "Average return %":values.mean() if len(values) else np.nan,
                    "Median return %":values.median() if len(values) else np.nan,
                    "False positive %":values.lt(0).mean()*100 if len(values) else np.nan})
    weights_table=pd.DataFrame({"Indicator":FEATURES,"Base weight %":BASE_WEIGHTS*100,"Weights learned before holdout %":weights*100,"Change (percentage points)":(weights-BASE_WEIGHTS)*100})
    return {"metrics":pd.DataFrame(result_rows),"weights":weights_table,"train_count":train_count,
            "train_cutoff":train_cutoff.strftime("%Y-%m-%d"),"test_start":pd.Timestamp(split_date).strftime("%Y-%m-%d"),
            "test_rows":len(test)},None




# Sector baskets are illustrative, equal-weighted groups of ASX-listed shares,
# not official GICS sector indices. Survivorship and selection bias apply.
SECTOR_BASKETS = {
    "Materials & Mining": ["BHP", "RIO", "FMG", "S32", "MIN"],
    "Financials": ["CBA", "NAB", "WBC", "ANZ", "MQG"],
    "Energy": ["WDS", "STO", "ORG", "WHC", "BPT"],
    "Healthcare": ["CSL", "RMD", "COH", "SHL", "RHC"],
    "Technology": ["XRO", "WTC", "TNE", "NXT", "CPU"],
    "Consumer Discretionary": ["WES", "JBH", "HVN", "ALL", "FLT"],
    "Consumer Staples": ["WOW", "COL", "A2M", "BGA", "TWE"],
    "Industrials": ["QAN", "TCL", "BXB", "REH", "ALQ"],
    "Real Estate": ["GMG", "SCG", "VCX", "GPT", "MGR"],
    "Communication Services": ["TLS", "REA", "CAR", "TPG", "SEK"],
    "Utilities": ["AGL", "APA", "ORG", "MEZ", "SKI"],
}

@st.cache_data(ttl=3600, show_spinner=False)
def sector_price_history(symbols):
    out = {}
    for symbol in symbols:
        try:
            h = yf.Ticker(symbol).history(period="2y", auto_adjust=True)
            if h is not None and not h.empty and "Close" in h:
                close = h["Close"].dropna()
                if len(close) >= 6:
                    out[symbol] = close
        except Exception:
            pass
    return out

def sector_return(series, sessions):
    if series is None or len(series) <= sessions:
        return None
    first, last = float(series.iloc[-sessions-1]), float(series.iloc[-1])
    return (last / first - 1) * 100 if first > 0 else None

def basket_sector_for_ticker(ticker):
    matches=[name for name,codes in SECTOR_BASKETS.items() if ticker in codes]
    return " / ".join(matches) if matches else "Not in representative baskets"


def sector_accuracy_page():
    st.header("Sector-aware prediction accuracy")
    st.write("Compare saved Early-Mover predictions with the ASX 200 and inspect whether a sector basket was leading or lagging **at the signal date**. This uses only the representative basket constituents, not the full ASX sector universe.")
    horizon=st.selectbox("Outcome horizon (trading sessions)",[5,10,20,30,60],index=2,key="sector_accuracy_horizon")
    con=db_connect()
    records=pd.read_sql_query(f"SELECT ticker,signal_date,base_score,return_{horizon}d AS actual_return,features_json FROM predictions WHERE return_{horizon}d IS NOT NULL",con)
    con.close()
    if records.empty:
        st.info("No completed predictions at this horizon yet. Save predictions in Early-Mover Tracker, then refresh outcomes in Prediction Lab after enough trading sessions.")
        return
    records["Sector basket"]=records["ticker"].map(basket_sector_for_ticker)
    st.caption(f"{len(records)} completed predictions available. Sector membership covers only the selected representative shares; other ASX tickers remain unclassified.")
    if not st.button("Calculate benchmark and historical sector context",key="sector_accuracy_run"):
        st.dataframe(records[["ticker","signal_date","Sector basket","base_score","actual_return"]].rename(columns={"ticker":"Ticker","signal_date":"Signal date","base_score":"Saved score","actual_return":"Forward return %"}),hide_index=True,use_container_width=True)
        return
    needed={t for code in records.ticker for t in SECTOR_BASKETS.get(basket_sector_for_ticker(code),[]) }  # exclude ambiguous baskets
    symbols=["^AXJO"]+[f"{t}.AX" for t in sorted(needed)]
    with st.spinner("Loading historical benchmark and basket prices..."):
        histories=sector_price_history(tuple(symbols))
    benchmark=histories.get("^AXJO")
    if benchmark is None:
        st.error("ASX 200 history unavailable; cannot calculate relative outcomes.")
        return
    def historical_return_at(series,signal_date,back=63,forward=None):
        if series is None: return np.nan
        idx=pd.to_datetime(series.index).tz_localize(None) if getattr(series.index,"tz",None) is not None else pd.to_datetime(series.index)
        position=idx.searchsorted(pd.Timestamp(signal_date),side="right")-1
        if position<0: return np.nan
        if forward is not None:
            target=position+forward
            if target>=len(series): return np.nan
            return (float(series.iloc[target])/float(series.iloc[position])-1)*100
        if position<back: return np.nan
        return (float(series.iloc[position])/float(series.iloc[position-back])-1)*100
    results=[]
    for _,r in records.iterrows():
        sector=basket_sector_for_ticker(r.ticker)
        asx_future=historical_return_at(benchmark,r.signal_date,forward=horizon)
        asx_prior=historical_return_at(benchmark,r.signal_date)
        member_returns=[]
        if sector in SECTOR_BASKETS:
            for ticker in SECTOR_BASKETS[sector]:
                value=historical_return_at(histories.get(f"{ticker}.AX"),r.signal_date)
                if np.isfinite(value): member_returns.append(value)
        sector_prior=float(np.mean(member_returns)) if len(member_returns)>=3 else np.nan
        relative_sector=sector_prior-asx_prior if np.isfinite(sector_prior) and np.isfinite(asx_prior) else np.nan
        results.append({"Ticker":r.ticker,"Signal date":r.signal_date,"Sector basket":sector,"Sector status at signal":"Leading" if np.isfinite(relative_sector) and relative_sector>0 else ("Lagging" if np.isfinite(relative_sector) else "Unknown"),"Prior sector vs ASX (pp)":relative_sector,"Saved score":r.base_score,"Forward return %":r.actual_return,"ASX forward return %":asx_future,"Excess return (pp)":r.actual_return-asx_future if np.isfinite(asx_future) else np.nan})
    result=pd.DataFrame(results)
    usable=result.dropna(subset=["Excess return (pp)"])
    if usable.empty:
        st.warning("No benchmark-matched outcomes. Historical price coverage is limited to two years.")
    else:
        a,b,c=st.columns(3)
        a.metric("Benchmark-matched signals",len(usable))
        b.metric("Outperformed ASX 200",f"{(usable['Excess return (pp)']>0).mean()*100:.1f}%")
        c.metric("Average excess return",f"{usable['Excess return (pp)'].mean():+.2f} pp")
        summary=usable.groupby("Sector status at signal",dropna=False).agg(Signals=("Ticker","size"),Positive_rate=("Forward return %",lambda v:round((v>0).mean()*100,1)),Average_return=("Forward return %","mean"),Average_excess=("Excess return (pp)","mean")).reset_index()
        st.subheader("Does prior sector strength matter?")
        st.dataframe(summary.round(2),hide_index=True,use_container_width=True)
        st.caption("Positive rate measures gains above zero; excess return compares against the ASX 200. 'Unknown' includes tickers outside our small baskets and dates without enough price history.")
    st.dataframe(result.round(2),hide_index=True,use_container_width=True)
    st.download_button("Download sector accuracy CSV",result.to_csv(index=False).encode(),file_name="asx_sector_prediction_accuracy.csv",mime="text/csv")
    st.caption("Important: this is descriptive, not causal. The fixed baskets have survivorship bias, and prices may be missing. The benchmark uses a price index while share returns are adjusted for corporate actions/dividends, so comparisons are not perfectly like-for-like. Sector status uses the prior 63 trading sessions and is not used to modify scores. Historical data limited to 2 years; outcomes must already be completed in Prediction Lab.")


def sector_performance_page():
    st.header("Sector Performance")
    st.write("Compare representative ASX sector baskets with the S&P/ASX 200 price index (^AXJO). Identify strong and weak sectors before reviewing individual shares.")
    horizon = st.selectbox("Comparison period", ["1 week", "1 month", "3 months", "6 months", "12 months"], index=2)
    sessions = {"1 week": 5, "1 month": 21, "3 months": 63, "6 months": 126, "12 months": 252}[horizon]
    if st.button("Refresh sector data", key="refresh_sector"):
        sector_price_history.clear()
    symbols = ["^AXJO"] + sorted({f"{t}.AX" for basket in SECTOR_BASKETS.values() for t in basket})
    with st.spinner("Fetching sector share prices..."):
        prices = sector_price_history(tuple(symbols))
    benchmark = sector_return(prices.get("^AXJO"), sessions)
    if benchmark is None:
        st.warning("The ASX 200 benchmark has insufficient data for this period. Try a shorter period or refresh later.")
        return
    st.metric("ASX 200 benchmark return", f"{benchmark:+.2f}%", help="Price index only; excludes dividends")
    records = []
    for sector, tickers in SECTOR_BASKETS.items():
        members = []
        for t in tickers:
            ret = sector_return(prices.get(f"{t}.AX"), sessions)
            if ret is not None:
                members.append((t, ret))
        if not members:
            continue
        avg = float(np.mean([r for _, r in members]))
        records.append({"Sector": sector, "Sector return %": round(avg, 2), "ASX 200 return %": round(benchmark, 2), "Relative performance (pp)": round(avg-benchmark, 2), "Outperforming stocks": sum(r > benchmark for _, r in members), "Stocks with data": len(members), "Basket size": len(tickers), "Status": "Outperforming" if avg > benchmark else "Underperforming"})
    if not records:
        st.warning("No usable sector prices returned. Try refreshing later.")
        return
    df = pd.DataFrame(records).sort_values("Relative performance (pp)", ascending=False)
    a, b, c = st.columns(3)
    a.metric("Sectors beating ASX 200", int((df["Relative performance (pp)"] > 0).sum()))
    b.metric("Sectors lagging ASX 200", int((df["Relative performance (pp)"] < 0).sum()))
    c.metric("Strongest basket", df.iloc[0]["Sector"])
    st.subheader("Sector performance relative to ASX 200")
    st.bar_chart(df.set_index("Sector")["Relative performance (pp)"], horizontal=True)
    st.dataframe(df, hide_index=True, use_container_width=True)
    chosen = st.selectbox("Inspect a sector's individual shares", df["Sector"].tolist())
    details = []
    for t in SECTOR_BASKETS[chosen]:
        r = sector_return(prices.get(f"{t}.AX"), sessions)
        details.append({"Ticker": t, "Return %": round(r, 2) if r is not None else None, "Vs ASX 200 (pp)": round(r-benchmark, 2) if r is not None else None})
    st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)
    st.download_button("Download sector comparison CSV", df.to_csv(index=False).encode(), file_name="asx_sector_comparison.csv", mime="text/csv")
    st.caption("Method: simple average of available constituent price returns, equal weight per stock. Sector baskets are selected examples, NOT comprehensive ASX sectors or investable sector indices. Constituents can overlap, and missing stocks are excluded. 12 months requires 253 price observations; some stocks may lack data. Benchmark excludes dividends. This is a sector screen, not proof of future performance.")


MASTER_TICKERS = load_master_tickers()
def apply_custom_score_filters(frame, key_prefix):
    """Optional AND-combined filters; missing scores never pass an enabled filter."""
    specs = [
        ("Early-Mover Score", "EMS — minimum", 70, "min"),
        ("VPA Score", "VPA — minimum", 65, "min"),
        ("Accumulation Score", "Accumulation — minimum", 60, "min"),
        ("Distribution Risk", "Today distribution risk — maximum", 35, "max"),
        ("20D Distribution Risk", "20-day distribution risk — maximum", 35, "max"),
        ("Effort Result Score", "Effort / result — minimum", 50, "min"),
        ("Momentum Score", "Momentum — minimum", 60, "min"),
        ("Value Score", "Value — minimum", 50, "min"),
        ("Growth Score", "Growth — minimum", 50, "min"),
        ("Overall Score", "Overall — minimum", 60, "min"),
    ]
    with st.expander("🎚️ Custom score filters — show only shares meeting ALL enabled rules", expanded=True):
        st.caption("Tick the scores you want to filter, then move each slider. Unticked scores are ignored. All enabled rules must pass. Missing values do not pass an enabled filter.")
        a,b=st.columns(2)
        active=[]
        for i,(column,label,default,mode) in enumerate(specs):
            with (a if i%2==0 else b):
                enabled=st.checkbox(f"Use {label.split(' — ')[0]}",value=False,key=f"{key_prefix}_use_{i}")
                threshold=st.slider(label,0,100,default,1,key=f"{key_prefix}_threshold_{i}",disabled=not enabled)
                if enabled:
                    active.append((column,threshold,mode))
        if st.button("Clear all filters",key=f"{key_prefix}_clear"):
            for i in range(len(specs)):
                st.session_state[f"{key_prefix}_use_{i}"]=False
            st.rerun()
    filtered=frame.copy()
    for column,threshold,mode in active:
        values=pd.to_numeric(filtered[column],errors="coerce") if column in filtered else pd.Series(np.nan,index=filtered.index)
        filtered=filtered[values.ge(threshold) if mode=="min" else values.le(threshold)]
    st.caption(f"Showing **{len(filtered):,} of {len(frame):,}** shares with price data. {len(active)} active score filter(s).")
    if filtered.empty:
        st.warning("No shares match every enabled filter. Reduce minimums, increase maximum distribution risk, or untick some rules.")
    return filtered

# ---------------- Simple evidence-first opportunity shortlist ----------------
def opportunity_conditions(frame):
    """A deliberately small, fixed set of explainable 20-session setups."""
    f=frame
    def n(k):
        return pd.to_numeric(f[k],errors="coerce") if k in f else pd.Series(np.nan,index=f.index)
    trend=(n("Return 1M %")>0)&(n("Price vs 50D %")>0)&(n("Volume Change %")>0)
    breakout=(n("Return 1W %")>0)&(n("Distance from 20D High %")>=-3)&(n("Volume Change %")>20)
    recovery=(n("Return 1M %")<0)&(n("VPA Score")>=65)&(n("Distribution Risk")<=20)&(n("Return 1W %")>-20)
    return {"Trend + volume":trend,"Near-high breakout":breakout,"Possible recovery":recovery}

def opportunity_evidence():
    """Fit the strategy on earlier outcomes; audit it on a later untouched period."""
    con=db_connect()
    try:
        raw=pd.read_sql_query("SELECT signal_date,features_json,return_20d FROM historical_learning_monthly WHERE return_20d IS NOT NULL",con)
    finally:
        con.close()
    if raw.empty or len(raw)<120:
        return {},"At least 120 completed historical monthly examples are needed. Build them on the Phase 4 Backtest page."
    rows=[]
    for r in raw.itertuples(index=False):
        try: features=json.loads(r.features_json or '{}')
        except (ValueError,TypeError): continue
        rows.append({**features,"Signal date":r.signal_date,"Return 20d %":r.return_20d})
    hist=pd.DataFrame(rows)
    hist["Signal date"]=pd.to_datetime(hist["Signal date"],errors="coerce")
    hist["Return 20d %"]=pd.to_numeric(hist["Return 20d %"],errors="coerce")
    hist=hist.dropna(subset=["Signal date","Return 20d %"]).sort_values("Signal date")
    if len(hist)<120:return {},"Not enough valid completed historical examples."
    split=hist["Signal date"].iloc[int(len(hist)*.7)]
    train=hist[hist["Signal date"]<split-pd.Timedelta(days=45)].copy()
    test=hist[hist["Signal date"]>=split].copy()
    if len(train)<60 or len(test)<30:return {},"Not enough separate earlier and later observations for validation."
    train_conditions=opportunity_conditions(train)
    test_conditions=opportunity_conditions(test)
    baseline=float((test["Return 20d %"]>0).mean()*100)
    results={}
    for name,train_mask in train_conditions.items():
        t=train.loc[train_mask,"Return 20d %"]
        v=test.loc[test_conditions[name],"Return 20d %"]
        results[name]={"train_n":len(t),"test_n":len(v),"win_rate":float((v>0).mean()*100) if len(v) else None,
                       "avg_return":float(v.mean()) if len(v) else None,"baseline":baseline,
                       "validated":bool(len(t)>=25 and len(v)>=15 and (v>0).mean()*100>baseline and v.mean()>0)}
    return results,None

def opportunity_reason(row,name):
    reasons={
        "Trend + volume":"Positive monthly trend, above 50-day average, with increasing volume",
        "Near-high breakout":"Positive weekly momentum, near its 20-day high, with increasing volume",
        "Possible recovery":"Weak monthly price trend but stronger VPA and limited recent distribution warnings"}
    return reasons[name]

def render_opportunities(scan):
    st.header("Today's Opportunities")
    st.write("A simple 2–4 week research shortlist. Candidates must match a transparent price/volume setup; historical results determine whether the setup has demonstrated value.")
    st.caption("Uses the latest completed scan, not a live price feed. No ranking here is a probability of profit or a personalised buy recommendation.")
    if scan is None or scan.empty:
        st.info("Run / refresh multi-share scan in the sidebar first. Then return here to view the shortlist.")
        return
    report=st.session_state.get("scan_report",{})
    if report:
        st.caption(f"Universe coverage: {report.get('requested',len(scan)):,} requested; {report.get('usable',0):,} usable results. The shortlist ranks matching shares across ALL scanned results, not just the first 20 tickers.")
        if report.get("requested",0)<len(MASTER_TICKERS):
            st.warning("This scan did not cover the entire bundled ASX list. Select Full-market scan in the sidebar and run the scan to search all tickers.")
    data=scan[scan["Price"].notna()].copy()
    if data.empty:
        st.warning("No usable price data in the latest scan.")
        return
    evidence,error=opportunity_evidence()
    if error: st.warning("Historical validation unavailable: "+error+" The app will show research candidates, not verified opportunities.")
    else:
        with st.expander("See historical evidence (later holdout period)"):
            table=[{"Setup":k,"Earlier examples":v["train_n"],"Later examples":v["test_n"],
                    "Later positive %":round(v["win_rate"],1) if v["win_rate"] is not None else None,
                    "Later average return %":round(v["avg_return"],2) if v["avg_return"] is not None else None,
                    "Later all-share positive %":round(v["baseline"],1),
                    "Passed basic check":v["validated"]} for k,v in evidence.items()]
            st.dataframe(pd.DataFrame(table),hide_index=True,use_container_width=True)
            st.caption("Historical tests use monthly examples with a 45-calendar-day training gap. This is an exploratory holdout check, not a trading simulation: it excludes costs, delistings may be missing, and the historical ticker universe may suffer survivorship bias. Do not interpret win rates as calibrated forecasts.")
    candidates=[]
    masks=opportunity_conditions(data)
    for name,mask in masks.items():
        for _,r in data.loc[mask].iterrows():
            e=evidence.get(name,{})
            candidates.append({"Ticker":r.get("Ticker"),"Company":r.get("Company"),"Price":r.get("Price"),
                "Setup":name,"Why it appeared":opportunity_reason(r,name),
                "Historical positive %":round(e["win_rate"],1) if e.get("win_rate") is not None else np.nan,
                "Historical average return %":round(e["avg_return"],2) if e.get("avg_return") is not None else np.nan,
                "Evidence":"Passed initial holdout" if e.get("validated") else "Not validated",
                "Risk to check":"Thin trading / sharp reversal / company announcements",
                "Data Date":r.get("Data Date","")})
    if not candidates:
        st.info("No shares match the three setups in this scan. That is a valid outcome; there is no need to force a buy list.")
        return
    result=pd.DataFrame(candidates)
    # Avoid claiming one ticker is stronger simply because it matches more rules.
    result["_pass"]=(result["Evidence"]=="Passed initial holdout").astype(int)
    # Rank on current, share-specific evidence rather than alphabetic ticker order.
    # Previously most setups shared the same validation status and historical
    # win rate, so sorting by Ticker made the shortlist almost entirely A/B shares.
    lookup=data.drop_duplicates("Ticker").set_index("Ticker")
    def numeric_metric(column, default=50):
        values=pd.to_numeric(lookup.get(column, pd.Series(index=lookup.index, dtype=float)), errors="coerce")
        return values.reindex(result["Ticker"]).reset_index(drop=True).fillna(default)
    result=result.reset_index(drop=True)
    ems=numeric_metric("Early-Mover Score", np.nan)
    vpa=numeric_metric("VPA Score", 50)
    accumulation=numeric_metric("Accumulation Score", 50)
    risk=numeric_metric("Distribution Risk", 50)
    momentum=numeric_metric("Momentum Score", 50)
    ems=ems.fillna(momentum)
    # A transparent prioritisation heuristic, NOT a calibrated success probability.
    result["Priority score"]=(.35*ems.clip(0,100)+.30*vpa.clip(0,100)+
        .20*accumulation.clip(0,100)+.15*(100-risk.clip(0,100))).round(1)
    result=result.sort_values(["_pass","Priority score","Ticker"],ascending=[False,False,True],na_position="last")
    top=result.drop_duplicates("Ticker").head(20).drop(columns="_pass")
    st.metric("Candidates matching a setup",result["Ticker"].nunique())
    st.caption("Ranking: 35% EMS (or momentum if unavailable), 30% VPA, 20% accumulation, 15% distribution safety. These weights are heuristic, not validated probabilities. Tickers are only used to break exact ties.")
    letters=sorted(set(data["Ticker"].astype(str).str[0].str.upper()))
    st.caption(f"Scanned shares with usable prices span {len(letters)} starting letters: {', '.join(letters)}. Shortlist spans {top['Ticker'].astype(str).str[0].nunique()} starting letters.")
    if not (top["Evidence"]=="Passed initial holdout").any():
        st.warning("None of the available setups has passed the basic historical holdout check. These are watchlist candidates, not evidence-backed buy signals.")
    st.dataframe(top,use_container_width=True,hide_index=True)
    st.download_button("Download shortlist CSV",top.to_csv(index=False).encode(),file_name="asx_opportunity_shortlist.csv",mime="text/csv",key="opportunities_download")
    st.caption("Historical figures reflect setup-level outcomes, not ticker-specific forecasts. Check ASX announcements, spreads, liquidity and potential losses before making any decision.")

# ---------------- Independent research strategy screens ----------------
def strategy_mask(frame, kind, liquidity, volatility, vpa_min, accum_min, dist_max, momentum_min):
    def n(key):
        if key in frame.columns:
            return pd.to_numeric(frame[key], errors="coerce")
        return pd.Series(np.nan, index=frame.index)
    # Historical learning records do not contain daily dollar volume. Require a
    # measured liquidity field for live candidates; leave it untested in history.
    mask = n("Price").gt(0) if "Price" in frame else pd.Series(True, index=frame.index)
    mask &= n("20D volatility %").le(volatility)
    mask &= n("VPA Score").ge(vpa_min) & n("Accumulation Score").ge(accum_min)
    mask &= n("Distribution Risk").le(dist_max)
    if kind == "consistent":
        mask &= n("Price vs 50D %").gt(0) & n("Return 1M %").gt(0)
        mask &= n("Return 1W %").ge(-3) & n("Return 1W %").le(12)
    else:
        mask &= n("Return 1W %").ge(momentum_min)
        mask &= n("Distance from 20D High %").ge(-5)
        mask &= n("Volume Change %").ge(20)
    if "20D dollar volume AUD" in frame.columns:
        mask &= n("20D dollar volume AUD").ge(liquidity)
    return mask.fillna(False)


def render_strategy_screen(scan, kind):
    consistent = kind == "consistent"
    name = "Consistent Gains" if consistent else "Breakout Opportunities"
    st.header(("🛡️ " if consistent else "🚀 ") + name)
    st.write("Research screen for steadier trends and controlled risk." if consistent else
             "Research screen for stronger momentum, volume expansion and potential larger moves.")
    st.caption("These are separate, adjustable research screens, not predictions or guaranteed gains. Neither strategy places trades.")
    with st.expander("Strategy settings", expanded=True):
        a,b,c=st.columns(3)
        prefix="cg" if consistent else "bo"
        liquidity=a.select_slider("Minimum 20-day average daily traded value (AUD)",
            options=[0,10000,20000,50000,100000,250000,500000,1000000],
            value=100000 if consistent else 20000,key=prefix+"_liq")
        volatility=b.slider("Maximum 20-day daily volatility (%)",1.0,20.0,5.0 if consistent else 10.0,0.5,key=prefix+"_vol")
        vpa_min=c.slider("Minimum VPA score",0,100,60 if consistent else 65,key=prefix+"_vpa")
        accum_min=a.slider("Minimum accumulation score",0,100,55 if consistent else 60,key=prefix+"_acc")
        dist_max=b.slider("Maximum distribution risk",0,100,25 if consistent else 40,key=prefix+"_dist")
        momentum_min=c.slider("Minimum 1-week gain (%)",0.0,30.0,3.0,0.5,key=prefix+"_momentum") if not consistent else 0.0
    if scan is None or scan.empty:
        st.info("Run the Automatic Share Scanner first. These pages use its latest completed scan; they do not start another download.")
    else:
        data=scan.copy()
        # Scanner does not store dollar turnover directly. Derive an approximate
        # average from its recent price and its historic 20-day volume baseline
        # when possible; avoid silently treating Volume Change % as liquidity.
        if "20D dollar volume AUD" not in data.columns:
            data["20D dollar volume AUD"]=np.nan
        eligible=strategy_mask(data,kind,liquidity,volatility,vpa_min,accum_min,dist_max,momentum_min)
        result=data.loc[eligible].copy()
        st.metric("Shares matching all selected conditions",len(result))
        if data["20D dollar volume AUD"].isna().all():
            st.warning("The existing scanner does not save 20-day average dollar turnover. The liquidity condition cannot yet be verified from its results, so no shares can qualify safely. A future scan must capture that metric; existing history remains unchanged.")
        if not result.empty:
            def numeric(k):
                return pd.to_numeric(result[k],errors="coerce").fillna(50).clip(0,100) if k in result else pd.Series(50.0,index=result.index)
            ems=numeric("Early-Mover Score"); vpa=numeric("VPA Score"); acc=numeric("Accumulation Score"); risk=numeric("Distribution Risk")
            if consistent:
                result["Strategy ranking"]=(.25*ems+.25*vpa+.20*acc+.30*(100-risk)).round(1)
            else:
                result["Strategy ranking"]=(.35*ems+.30*vpa+.25*acc+.10*(100-risk)).round(1)
            result=result.sort_values("Strategy ranking",ascending=False)
            cols=[c for c in ["Ticker","Company","Price","Strategy ranking","Early-Mover Score","VPA Score","Accumulation Score","Distribution Risk","20D volatility %","20D dollar volume AUD","Return 1W %","Return 1M %","Price vs 50D %","Volume Change %","Data Date"] if c in result]
            st.dataframe(result[cols].head(100),hide_index=True,use_container_width=True)
            st.download_button("Download strategy shortlist CSV",result[cols].to_csv(index=False).encode(),file_name=prefix+"_shortlist.csv",mime="text/csv",key=prefix+"_download")
        elif not data["20D dollar volume AUD"].isna().all():
            st.info("No shares currently meet all conditions. You can relax the sliders or run a newer scan.")
    with st.expander("Historical performance check (independent for each strategy)"):
        st.caption("Uses previously stored monthly historical examples. Tests the selected technical rules on later dates, separated from the earlier sample by 45 calendar days. Historical records do not include the live liquidity measure; liquidity is NOT evaluated here. These are forward 20-session returns, not simulated portfolio profits or drawdowns.")
        if st.button("Evaluate historical 20-day outcomes",key=("cg" if consistent else "bo")+"_test"):
            con=db_connect()
            try:
                raw=pd.read_sql_query("SELECT signal_date,features_json,return_20d FROM historical_learning_monthly WHERE return_20d IS NOT NULL",con)
            finally:
                con.close()
            records=[]
            for rec in raw.itertuples(index=False):
                try: features=json.loads(rec.features_json or "{}")
                except (ValueError,TypeError): continue
                records.append({**features,"Signal date":rec.signal_date,"Return 20d %":rec.return_20d})
            hist=pd.DataFrame(records)
            if hist.empty or len(hist)<120:
                st.warning("Not enough historical monthly observations yet (at least 120 needed).")
            else:
                hist["Signal date"]=pd.to_datetime(hist["Signal date"],errors="coerce")
                hist["Return 20d %"]=pd.to_numeric(hist["Return 20d %"],errors="coerce")
                hist=hist.dropna(subset=["Signal date","Return 20d %"]).sort_values("Signal date")
                split=hist["Signal date"].iloc[int(len(hist)*.7)]
                later=hist.loc[hist["Signal date"]>=split]
                chosen=later.loc[strategy_mask(later,kind,liquidity,volatility,vpa_min,accum_min,dist_max,momentum_min)]
                st.write(f"Later sample: {len(later):,} observations; matched: {len(chosen):,}. Historical liquidity not checked.")
                if len(chosen)<20:
                    st.warning("Fewer than 20 matched later observations; evidence is too limited for a useful comparison.")
                else:
                    for label,values in [(name,chosen["Return 20d %"]),("All later signals",later["Return 20d %"])]:
                        st.write(f"**{label}:** {len(values):,} observations · {(values>0).mean()*100:.1f}% positive · {values.mean():+.2f}% average 20-session return · {values.median():+.2f}% median")
                    st.caption("Thresholds were selected interactively, so this is exploratory analysis rather than a fresh untouched validation. Returns exclude fees and may include overlapping signals.")

# ---------------- Historical two-strategy portfolio comparison ----------------
def _comparison_candidates(years, max_tickers, signal_spacing=5):
    """Build signals using only data known at each historical close."""
    con=db_connect()
    stats=con.execute("SELECT ticker,COUNT(*) AS n FROM historical_prices GROUP BY ticker HAVING COUNT(*)>=100 ORDER BY ticker").fetchall()
    codes=[r[0] for r in stats]
    if max_tickers: codes=codes[:max_tickers]
    if not codes:
        con.close(); return pd.DataFrame(),{},0
    end=pd.Timestamp.today().normalize()
    start=end-pd.DateOffset(years=years)
    warmup=start-pd.Timedelta(days=420)
    candidates=[]; histories={}
    bar=st.progress(0,text="Reconstructing historical signals from cached prices…")
    for j,ticker in enumerate(codes):
        raw=pd.read_sql_query("SELECT price_date,open,high,low,close,volume FROM historical_prices WHERE ticker=? AND price_date>=? ORDER BY price_date",con,params=(ticker,warmup.strftime('%Y-%m-%d')))
        if len(raw)<100: continue
        raw['price_date']=pd.to_datetime(raw['price_date'],errors='coerce')
        h=raw.set_index('price_date').rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}).sort_index()
        for k in ('Open','High','Low','Close','Volume'): h[k]=pd.to_numeric(h[k],errors='coerce')
        h=h.dropna(subset=['Close'])
        if len(h)<100: continue
        c=h['Close']; v=h['Volume'].fillna(0)
        h['ma50']=c.rolling(50,min_periods=50).mean()
        h['vol20']=c.pct_change().rolling(20,min_periods=20).std()*100
        h['turnover20']=(c*v).rolling(20,min_periods=20).mean()
        h['ret1w']=(c/c.shift(5)-1)*100
        h['ret1m']=(c/c.shift(21)-1)*100
        h['dist_high']=(c/c.rolling(20,min_periods=20).max()-1)*100
        h['vol_change']=(v.rolling(20,min_periods=20).mean()/v.shift(20).rolling(20,min_periods=20).mean().replace(0,np.nan)-1)*100
        histories[ticker]=h[['Open','High','Low','Close','Volume']]
        # One candidate per five cached trading sessions, starting after warmup.
        for i in range(100,len(h),signal_spacing):
            date=h.index[i]
            if date<start or date>end: continue
            row=h.iloc[i]
            if not np.isfinite(row['Close']) or row['Close']<0.05: continue
            if not np.isfinite(row['vol20']) or not np.isfinite(row['turnover20']): continue
            vol=float(row['vol20']); liq=float(row['turnover20']); w=float(row['ret1w']); m=float(row['ret1m'])
            ma=float(row['ma50']); dist=float(row['dist_high']); change=float(row['vol_change'])
            # Cheap prefilter avoids expensive VPA for clearly ineligible setups.
            cg=liq>=100000 and vol<=5 and row['Close']>ma and m>0 and -3<=w<=12
            bo=liq>=20000 and vol<=10 and w>=3 and dist>=-5 and change>=20
            if not cg and not bo: continue
            vpa=calculate_vpa(h.iloc[max(0,i-119):i+1][['Open','High','Low','Close','Volume']])
            vp=num(vpa.get('VPA Score')); ac=num(vpa.get('Accumulation Score')); dr=num(vpa.get('Distribution Risk'))
            if any(x is None for x in (vp,ac,dr)): continue
            if cg and vp>=60 and ac>=55 and dr<=25:
                rank=.25*min(100,max(0,50+w*3))+.25*vp+.20*ac+.30*(100-dr)
                candidates.append((date,ticker,'Consistent Gains',rank,liq,vol))
            if bo and vp>=65 and ac>=60 and dr<=40:
                rank=.35*min(100,max(0,50+w*3))+.30*vp+.25*ac+.10*(100-dr)
                candidates.append((date,ticker,'Breakout Opportunities',rank,liq,vol))
        if j%10==0 or j==len(codes)-1: bar.progress((j+1)/len(codes),text=f"Reconstructing signals: {j+1:,}/{len(codes):,} shares")
    con.close(); bar.empty()
    return pd.DataFrame(candidates,columns=['date','ticker','strategy','score','liquidity','volatility']),histories,len(codes)


def _comparison_simulate(signals, histories, start, end, capital, hold_days, brokerage, slip_pct,
                         max_positions=2, min_first_buy=500.0, participation_pct=1.0):
    """Cash-constrained ASX paper portfolio; whole shares and Stake first-buy minimum.

    A signal is known after its close and entered at the next available open.
    First buys of a ticker must be at least min_first_buy in share consideration,
    excluding brokerage. No leverage, and daily volume participation is capped.
    """
    sessions=sorted({d for h in histories.values() for d in h.index if start<=d<=end})
    if not sessions: return pd.DataFrame(),pd.DataFrame(),{}
    orders={}
    for row in signals.itertuples(index=False):
        h=histories[row.ticker]
        idx=h.index.searchsorted(row.date,side='right')
        if idx>=len(h): continue
        entry_date=h.index[idx]
        if entry_date>end: continue
        orders.setdefault(entry_date,[]).append(row)
    cash=float(capital); positions={}; trades=[]; curve=[]
    first_bought=set()
    diagnostics={'Signals':len(signals),'Orders considered':0,'Skipped minimum purchase':0,
                 'Skipped insufficient cash':0,'Skipped liquidity cap':0,
                 'Skipped missing open/volume':0,'Skipped already held':0,
                 'Skipped position limit':0,'Buys executed':0,'Brokerage paid (AUD)':0.0,
                 'Unspent cash at end (AUD)':0.0}
    for day in sessions:
        for ticker,p in list(positions.items()):
            h=histories[ticker]
            if day not in h.index: continue
            bar=h.loc[day]; op=num(bar['Open']); lo=num(bar['Low']); hi=num(bar['High']); cl=num(bar['Close'])
            if cl is None or cl<=0: continue
            # Entry was at today's open, so no exit before the next trading session.
            if day<=p['date']: continue
            p['days']+=1
            stop=p['entry']*(.94 if p['strategy']=='Consistent Gains' else .90)
            target=p['entry']*(1.12 if p['strategy']=='Consistent Gains' else 1.25)
            reason=None; exit_price=cl
            if op is not None and op>0 and op<=stop: reason='Stop gap'; exit_price=op
            elif lo is not None and lo<=stop: reason='Stop'; exit_price=stop
            elif hi is not None and hi>=target: reason='Target'; exit_price=target
            elif p['days']>=hold_days: reason='Time exit'
            elif day==sessions[-1]: reason='End of test'
            if reason:
                exit_price=max(0,exit_price*(1-slip_pct/100))
                proceeds=p['shares']*exit_price-brokerage
                cash+=proceeds
                diagnostics['Brokerage paid (AUD)']+=brokerage
                trades.append({'Ticker':ticker,'Entry date':p['date'].date().isoformat(),
                    'Exit date':day.date().isoformat(),'Shares':p['shares'],
                    'Entry price':round(p['entry'],4),'Exit price':round(exit_price,4),
                    'Buy consideration (AUD)':round(p['shares']*p['entry'],2),
                    'Net P/L (AUD)':round(proceeds-p['cost'],2),
                    'Return %':round((proceeds/p['cost']-1)*100,2),'Exit reason':reason})
                del positions[ticker]
        picks=sorted(orders.get(day,[]),key=lambda r:(-r.score,r.ticker))
        for s in picks:
            diagnostics['Orders considered']+=1
            if len(positions)>=max_positions:
                diagnostics['Skipped position limit']+=1; continue
            if s.ticker in positions:
                diagnostics['Skipped already held']+=1; continue
            h=histories[s.ticker]; bar=h.loc[day]
            op=num(bar['Open']); vol=num(bar['Volume'])
            if op is None or op<=0 or vol is None or vol<=0 or not np.isfinite(op*vol):
                diagnostics['Skipped missing open/volume']+=1; continue
            fill=op*(1+slip_pct/100)
            # Allocate up to an equal slot of initial capital, but never borrow.
            # If an existing position leaves less than the first-buy minimum,
            # wait for it to close instead of making a non-compliant $200 buy.
            cash_budget=min(float(capital)/max_positions,cash-brokerage)
            if cash_budget<min_first_buy:
                diagnostics['Skipped insufficient cash']+=1; continue
            liquidity_budget=op*vol*(participation_pct/100.0)
            if liquidity_budget<min_first_buy:
                diagnostics['Skipped liquidity cap']+=1; continue
            budget=min(cash_budget,liquidity_budget)
            shares=int(budget/fill)
            consideration=shares*fill
            if shares<1 or consideration+1e-8<min_first_buy:
                diagnostics['Skipped minimum purchase']+=1; continue
            cost=consideration+brokerage
            if cost>cash+1e-8:
                diagnostics['Skipped insufficient cash']+=1; continue
            cash-=cost
            diagnostics['Brokerage paid (AUD)']+=brokerage
            diagnostics['Buys executed']+=1
            first_bought.add(s.ticker)
            positions[s.ticker]={'date':day,'entry':fill,'shares':shares,'cost':cost,
                                 'days':0,'strategy':s.strategy}
        equity=cash
        for ticker,p in positions.items():
            hist_close=histories[ticker].loc[:day,'Close'].dropna()
            if not hist_close.empty: equity+=p['shares']*float(hist_close.iloc[-1])
        curve.append({'Date':day,'Equity':equity,'Cash':cash,'Open positions':len(positions)})
    diagnostics['Unspent cash at end (AUD)']=round(cash,2)
    diagnostics['Open positions at end']=len(positions)
    return pd.DataFrame(curve),pd.DataFrame(trades),diagnostics


def render_strategy_comparison():
    st.header('⚖️ Strategy Backtest Comparison')
    st.write('Simulate two independent historical portfolios using the cached ASX OHLCV data. Both start with the same capital and are evaluated over the same dates.')
    st.warning('Research simulation only. This is NOT a live-trading result. The cached universe generally excludes delisted shares (survivorship bias), adjusted historical prices may change, and spread/market impact estimates are approximate.')
    a,b,c=st.columns(3)
    years=a.selectbox('Historical period (years)',[1,2,3,5],index=2,key='compare_years')
    capital=b.number_input('Starting capital per strategy (AUD)',min_value=1000,max_value=1000000,value=1000,step=500,key='compare_capital')
    hold_days=c.selectbox('Maximum holding period (trading sessions)',[10,20,30],index=1,key='compare_hold')
    a,b,c=st.columns(3)
    brokerage=a.number_input('Brokerage per trade (AUD)',0.0,100.0,3.0,1.0,key='compare_broker')
    slippage=b.number_input('Estimated slippage per side (%)',0.0,5.0,0.25,0.05,key='compare_slip')
    max_tickers=c.selectbox('Cached shares to analyse',[100,250,500,1000,'All'],index=1,key='compare_universe')
    st.markdown('**Stake ASX small-account settings**')
    x,y,z=st.columns(3)
    min_buy=x.number_input('Minimum first purchase (AUD, excluding brokerage)',min_value=0.0,max_value=5000.0,value=500.0,step=50.0,key='compare_min_buy')
    max_positions=y.selectbox('Maximum open positions',[1,2,3,4,5],index=0,key='compare_max_positions')
    participation=z.number_input('Maximum share of daily dollar volume (%)',min_value=0.1,max_value=10.0,value=1.0,step=0.5,key='compare_participation')
    st.caption('Stake ASX standard brokerage defaults to $3 per order. The $500 first-purchase minimum is applied to every buy (conservative: no smaller top-ups). Whole shares only. Slippage is estimated, not a verified market fill. The minimum may differ for some instruments or account circumstances; check Stake terms.')
    if capital < max_positions * (min_buy + brokerage):
        st.info('The selected capital cannot fund all position slots at the minimum first-purchase amount. Some slots may stay unused; the simulation will not borrow or place undersized buys.')
    st.caption('Initial speed setting: 250 alphabetically ordered cached tickers. Select All for broader coverage; it may take several minutes. Use the same universe for both strategies. Signals sampled every 5 trading sessions; no fundamental/news replay.')
    if st.button('Run historical portfolio comparison',type='primary',key='compare_run'):
        try:
            signals,histories,count=_comparison_candidates(years,None if max_tickers=='All' else int(max_tickers))
            if not histories:
                st.error('No usable cached OHLCV histories. Run the Phase 4 historical data scan first.')
            else:
                end=pd.Timestamp.today().normalize(); start=end-pd.DateOffset(years=years)
                results={}
                for strategy in ('Consistent Gains','Breakout Opportunities'):
                    selected=signals[signals['strategy']==strategy] if not signals.empty else pd.DataFrame()
                    curve,trades,diagnostics=_comparison_simulate(selected,histories,start,end,capital,hold_days,brokerage,slippage,max_positions,min_buy,participation)
                    results[strategy]=(curve,trades,len(selected),diagnostics)
                st.session_state['compare_result']=(results,count,years,capital,hold_days,brokerage,slippage,min_buy,max_positions,participation)
        except Exception as exc:
            st.error(f'Backtest could not finish: {exc}')
    if 'compare_result' not in st.session_state: return
    results,count,used_years,used_capital,used_hold,used_broker,used_slip,used_min_buy,used_slots,used_participation=st.session_state['compare_result']
    st.caption(f'Last completed run: {count:,} cached tickers · {used_years} year(s) · ${used_capital:,.0f} per strategy · {used_hold}-session maximum holding period. Change settings and rerun to update.')
    summary=[]; plot=[]
    for strategy,(curve,trades,signals_n,diagnostics) in results.items():
        if curve.empty: continue
        values=curve['Equity'].astype(float)
        dd=(values/values.cummax()-1)*100
        wins=trades['Net P/L (AUD)'].gt(0) if not trades.empty else pd.Series(dtype=bool)
        profits=trades['Net P/L (AUD)'] if not trades.empty else pd.Series(dtype=float)
        losses=-profits[profits<0].sum(); gains=profits[profits>0].sum()
        summary.append({'Strategy':strategy,'Signals':signals_n,'Trades':len(trades),
            'Final equity (AUD)':round(values.iloc[-1],2),'Return %':round((values.iloc[-1]/used_capital-1)*100,2),
            'Max drawdown %':round(dd.min(),2),'Win rate %':round(wins.mean()*100,1) if len(wins) else np.nan,
            'Profit factor':round(gains/losses,2) if losses>0 else np.nan,
            'Brokerage paid (AUD)':round(diagnostics['Brokerage paid (AUD)'],2),
            'Final cash (AUD)':diagnostics['Unspent cash at end (AUD)']})
        plot.append(curve[['Date','Equity']].set_index('Date').rename(columns={'Equity':strategy}))
    if not summary: st.warning('No trading sessions available for this test.'); return
    st.caption(f'Applied: ${used_broker:.2f} per buy/sell · {used_slip:.2f}% slippage per side · ${used_min_buy:.0f} minimum purchase · {used_slots} positions · {used_participation:.1f}% daily-volume cap.')
    st.subheader('Portfolio results')
    st.dataframe(pd.DataFrame(summary),hide_index=True,use_container_width=True)
    st.line_chart(pd.concat(plot,axis=1).ffill())
    for strategy,(curve,trades,signals_n,diagnostics) in results.items():
        with st.expander(f'{strategy}: {len(trades)} completed trades'):
            st.write('Order diagnostics (why candidate trades were skipped):')
            st.dataframe(pd.DataFrame([diagnostics]),hide_index=True,use_container_width=True)
            if trades.empty: st.info('No trades qualified. Try a broader cached universe or check that sufficient price histories are saved.')
            else:
                st.dataframe(trades,hide_index=True,use_container_width=True)
                st.download_button(f'Download {strategy} trades',trades.to_csv(index=False).encode(),file_name=f"{strategy.lower().replace(' ','_')}_backtest_trades.csv",mime='text/csv',key='compare_download_'+strategy)
    st.subheader('Date-based holdout')
    st.caption('The final 30% of the period is shown separately as a descriptive holdout. The rules were chosen before this run, but adjusting them after viewing these results makes the holdout exploratory rather than untouched.')
    for strategy,(curve,trades,_) in results.items():
        if curve.empty: continue
        split=curve['Date'].iloc[int(len(curve)*.7)]
        earlier=curve[curve['Date']<split]; later=curve[curve['Date']>=split]
        if earlier.empty or later.empty: continue
        base=float(earlier['Equity'].iloc[-1]); finish=float(later['Equity'].iloc[-1])
        st.write(f'**{strategy}:** last 30% ({split.date()} onward): {(finish/base-1)*100:+.2f}% portfolio return; {len(trades[trades["Entry date"]>=split.date().isoformat()])} new entries.')
    st.caption('Entries use the next available open; stops and targets are evaluated from subsequent OHLC bars, with stops assumed first if both touched. The test caps new positions at 1% of that day’s reported traded value. No short selling, borrowing or reinvested dividends. Signals use only past data; actual results may differ significantly.')

with st.sidebar:
    page=st.radio("Page",["Today’s Opportunities","Consistent Gains","Breakout Opportunities","Strategy Backtest Comparison","Single-Share Analysis","Automatic Share Scanner","Early-Mover Tracker","VPA Scanner","Prediction Lab","Phase 4 Backtest","Historical Signal Explorer","Sector Performance","Sector Prediction Accuracy","Lower-Risk Investing","Value Investing","Momentum Trading","High-Growth Shares","Long-Term Investing","Portfolio Planner"])
    st.divider()
    ticker_text=st.text_area("ASX tickers (commas or new lines)",", ".join(MASTER_TICKERS),height=160,
        help="Defaults to the bundled ASX company list. Edit or paste your own tickers if needed.")
    tickers=list(dict.fromkeys(x.strip().upper().replace(".AX","") for x in ticker_text.replace("\n",",").split(",") if x.strip()))
    scan_mode=st.selectbox("Scan mode",["Full-market scan","Quick promising-shares scan"])
    # Full-market mode must cover the whole supplied universe, not just its alphabetic prefix.
    universe_count=len(tickers)
    if scan_mode == "Full-market scan":
        st.caption(f"Full-market coverage: all {universe_count:,} tickers in the supplied list (subject to data availability).")
    else:
        st.caption(f"Quick mode samples across the entire {universe_count:,}-ticker list; use Full-market for comprehensive coverage.")
    if scan_mode == "Quick promising-shares scan":
        prior=st.session_state.get("scan_results",pd.DataFrame())
        if isinstance(prior,pd.DataFrame) and not prior.empty and "Overall Score" in prior.columns:
            promising=prior[prior["Price"].notna()].sort_values("Overall Score",ascending=False,na_position="last")["Ticker"].head(150).tolist()
            if promising:
                tickers=promising
        else:
            # Sample evenly across the entire list rather than always choosing A/B tickers.
            if len(tickers)>150:
                positions=np.linspace(0,len(tickers)-1,150,dtype=int)
                tickers=[tickers[i] for i in positions]
            # Otherwise use every supplied ticker.
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

if page=="Today’s Opportunities":
    render_opportunities(scan)
elif page=="Single-Share Analysis":
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
        movers["Sector basket"]=movers["Ticker"].map(basket_sector_for_ticker)
        movers["Signal Tier"]=np.select([movers["Early-Mover Score"]>=80,movers["Early-Mover Score"]>=65],["Priority review","Watch closely"],default="Lower priority")
        st.caption("The score combines momentum, trend, volume, short-term extension, VPA, accumulation, distribution safety, effort/result, daily range, close position, opening gap and volatility. Prediction Lab shows the normalized base and learned weights. Missing inputs are excluded and available weights are renormalised.")
        st.caption(f"Latest scan: {st.session_state.get('scan_time','unknown')}. Use each share's Data Date to check freshness.")
        movers=apply_custom_score_filters(movers,"em_filter")
        cols=["Ticker","Company","Price","Early-Mover Score","Signal Tier","Sector basket","Return 1W %","Return 1M %","Return 3M %","Price vs 50D %","Above 200D %","Volume Change %","Distance from 20D High %","VPA Score","Accumulation Score","Distribution Risk","VPA Signal","Market Cap","Data Date","Scan Status"]
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
        missing_vpa = (scan["Price"].notna() if "VPA Score" not in scan.columns
                       else (scan["Price"].notna() & pd.to_numeric(scan["VPA Score"], errors="coerce").isna()))
        missing_20d = (scan["Price"].notna() if "20D Distribution Risk" not in scan.columns
                       else (scan["Price"].notna() & pd.to_numeric(scan["20D Distribution Risk"], errors="coerce").isna()))
        missing_count = int((missing_vpa | missing_20d).sum())
        if missing_count:
            st.warning(f"{missing_count} shares need updated VPA or 20-day distribution scores. Existing scan data may predate this feature.")
        if st.button("Refresh VPA + 20-day distribution scores", type="primary"):
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
            "20D Distribution Risk": np.nan,
            "Effort Result Score": np.nan,
            "VPA Signal": "VPA data not available in this saved scan",
            "VPA Notes": "Run a fresh scan to calculate VPA from OHLCV history."
        }
        for column, default_value in vpa_defaults.items():
            if column not in vpa.columns:
                vpa[column] = default_value
        learn_w,_,_,_=learned_weights()
        vpa["Early-Mover Score"]=vpa.apply(lambda row: score_with_weights(row,learn_w),axis=1)
        vpa=vpa.sort_values(["VPA Score","Accumulation Score"],ascending=False,na_position="last")
        vpa=apply_custom_score_filters(vpa,"vpa_filter")
        a,b,c=st.columns(3)
        a.metric("Shares with VPA score",int(vpa["VPA Score"].notna().sum()))
        b.metric("Possible accumulation (60+)",int((vpa["Accumulation Score"]>=60).sum()))
        c.metric("20-day distribution warnings",int((vpa["20D Distribution Risk"]>=35).sum()))
        st.caption("Distribution Risk is the latest-day warning score; 20D Distribution Risk measures recency-weighted warnings over 20 sessions with a persistence penalty. VPA score combines a 20-session directional-volume/range assessment, effort-versus-result and latest-bar event flags. It is a research ranking, not a probability or buy/sell signal.")
        cols=["Ticker","Company","Price","Early-Mover Score","VPA Score","Accumulation Score","Distribution Risk","20D Distribution Risk","Effort Result Score","VPA Signal","VPA Notes","Volume Change %","Return 1W %","Return 1M %","Data Date"]
        st.dataframe(vpa[[k for k in cols if k in vpa]].round(2),use_container_width=True,hide_index=True)
        st.download_button("Download VPA rankings CSV",vpa.to_csv(index=False).encode(),file_name="asx_vpa_rankings.csv",mime="text/csv")
        st.subheader("How to interpret the signals")
        st.markdown("- **Possible accumulation:** directional volume pressure and firm closes suggest demand may be absorbing supply.\n- **No supply:** a low-volume decline with a relatively narrow spread may indicate reduced selling interest.\n- **Stopping volume:** heavy volume during a decline with a recovery from the lows may indicate demand absorbing selling.\n- **Volume-confirmed breakout:** close above the prior 20-session high with volume at least 1.5× its 20-day average.\n- **Distribution / upthrust:** heavy volume with weak closes or a failed move above recent highs can warn of supply.")
        st.info("The Early-Mover Score now includes VPA features. The Prediction Lab learns their historical relationship with subsequent 20-day returns alongside the original indicators. Existing Phase 3 prediction history remains readable; older records simply have no VPA feature values.")

elif page=="Prediction Lab":
    st.header("🧠 Prediction Lab — accuracy and learning")
    st.write("Every saved prediction is measured against its entry close after 5, 10, 20, 30 and 60 subsequent trading sessions. The learning engine combines historical monthly examples with live completed 20-day outcomes to estimate which signals have been more useful, including the new VPA features. VPA values are saved with each new prediction so their contribution can be evaluated.")
    st.warning("This is a research experiment, not a proven forecasting model. It needs a meaningful number of saved predictions across different market conditions. It does not place trades.")
    a,b=st.columns([1,3])
    if a.button("Refresh outcomes from Yahoo Finance",type="primary"):
        with st.spinner("Checking saved predictions against later prices…"):
            count=refresh_prediction_outcomes()
        st.success(f"Updated outcomes for {count} prediction(s).")
    render_learning_dashboard()
    render_database_backup("prediction_lab")
    render_database_restore()
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
            try:
                saved_features=json.loads(pr.get("features_json") or "{}")
                parts=component_scores(saved_features) if any(num(saved_features.get(k)) is not None for k in FEATURES[8:12]) else []
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
        for h in (5,10,20,30,60):
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

elif page=="Historical Signal Explorer":
    st.header("🔎 Historical Signal Explorer")
    st.write("Reconstruct technical signals using only daily bars available at each selected historical close, then inspect subsequent returns.")
    st.caption("Uses the app's existing fixed-weight early-mover model, VPA and accumulation calculations. It does not replay past news or point-in-time fundamentals. Adjusted historical prices may be revised after corporate actions.")
    symbol=st.text_input("ASX ticker",value="LNQ",key="explorer_symbol").strip().upper().replace(".AX","")
    dates_text=st.text_input("Historical dates (comma-separated)",value="2026-10-09, 2026-10-14, 2026-10-16, 2026-10-19",key="explorer_dates")
    if st.button("Analyse historical signals",type="primary",key="explorer_run"):
        import re
        if not re.fullmatch(r"[A-Z0-9]{1,6}",symbol):
            st.error("Enter a valid ASX ticker, such as LNQ.")
        else:
            try:
                requested=[pd.Timestamp(part.strip()).normalize() for part in dates_text.split(",") if part.strip()]
                if not requested:
                    raise ValueError("Enter at least one date.")
                if len(requested)>30:
                    raise ValueError("Select no more than 30 dates at once.")
                h=yf.Ticker(symbol+".AX").history(period="max",auto_adjust=True)
                if h is None or h.empty or not all(c in h for c in ("Open","High","Low","Close","Volume")):
                    raise ValueError("No complete OHLCV price history returned by Yahoo Finance.")
                h=h.dropna(subset=["Close"]).sort_index().copy()
                h.index=pd.to_datetime(h.index).tz_localize(None).normalize()
                h=h[~h.index.duplicated(keep="last")]
                close=h["Close"].astype(float)
                rows=[]
                for requested_date in requested:
                    indices=np.flatnonzero(h.index<=requested_date)
                    if not len(indices):
                        rows.append({"Requested date":requested_date.date().isoformat(),"Status":"No earlier price history"})
                        continue
                    i=int(indices[-1])
                    if i<200:
                        rows.append({"Requested date":requested_date.date().isoformat(),"Signal date":h.index[i].date().isoformat(),"Status":"Needs 201 historical trading sessions"})
                        continue
                    past=h.iloc[:i+1]
                    pc=past["Close"].astype(float); pv=past["Volume"].fillna(0).astype(float)
                    price=float(pc.iloc[-1]); ma50=float(pc.tail(50).mean()); ma200=float(pc.tail(200).mean())
                    v20=float(pv.tail(20).mean()); vprev=float(pv.iloc[-40:-20].mean())
                    day=past.iloc[-1]; high=num(day["High"]); low=num(day["Low"]); opn=num(day["Open"])
                    inputs={"Return 1W %":pct(pc,5),"Return 1M %":pct(pc,21),"Return 3M %":pct(pc,63),
                        "Price vs 50D %":(price/ma50-1)*100 if ma50 else np.nan,
                        "Above 200D %":(price/ma200-1)*100 if ma200 else np.nan,
                        "Volume Change %":(v20/vprev-1)*100 if vprev else np.nan,
                        "Distance from 20D High %":(price/float(pc.tail(20).max())-1)*100,
                        "Daily range %":(high-low)/price*100 if high is not None and low is not None and price else np.nan,
                        "Close in daily range %":(price-low)/(high-low)*100 if high is not None and low is not None and high>low else np.nan,
                        "Gap %":(opn/float(pc.iloc[-2])-1)*100 if opn is not None and pc.iloc[-2] else np.nan,
                        "20D volatility %":pc.pct_change().tail(20).std()*100}
                    inputs.update(calculate_vpa(past.tail(120)))
                    inputs["Distribution Safety"]=100-inputs["Distribution Risk"] if pd.notna(inputs["Distribution Risk"]) else np.nan
                    result={"Requested date":requested_date.date().isoformat(),"Signal date":h.index[i].date().isoformat(),
                        "Status":"Exact trading date" if h.index[i]==requested_date else "Previous trading session",
                        "Closing price":price,"Early-Mover Score":score_with_weights(inputs,BASE_WEIGHTS),**inputs}
                    for horizon in (1,5,10,20,30):
                        result[f"Return {horizon}d %"]=(float(close.iloc[i+horizon])/price-1)*100 if i+horizon<len(close) else np.nan
                    rows.append(result)
                output=pd.DataFrame(rows)
                st.dataframe(output,use_container_width=True,hide_index=True)
                st.download_button("Download historical signal results (CSV)",output.to_csv(index=False).encode("utf-8"),file_name=f"{symbol}_historical_signals.csv",mime="text/csv")
                st.caption("Signals use the close of the signal date; future returns start after that close. Missing future returns mean the horizon has not yet completed. Scores are screening heuristics, not probabilities. The displayed early-mover score uses original fixed weights, not weights learned from later observations.")
            except Exception as exc:
                st.error(f"Could not analyse {symbol}: {exc}")

elif page=="Strategy Backtest Comparison":
    render_strategy_comparison()

elif page=="Phase 4 Backtest":
    st.header("🧪 Phase 4 — Historical Backtest & Diagnostics")
    st.write("Replay the combined Phase 4 technical and Phase 5 VPA scoring rules at past dates, compare them with later prices, and add completed historical observations to the learning engine. Those past results then inform the adaptive weights used for future live Early-Mover scores.")
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
    cached_codes={row[0] for row in con.execute("SELECT DISTINCT ticker FROM historical_prices").fetchall()}
    con.close()
    cached_in_universe=len(set(backtest_codes)&cached_codes)
    st.caption(f"This run has attempted {done:,}/{len(backtest_codes):,} codes; {cached_in_universe:,} have cached history. The local cache contains {cached_rows:,} daily OHLCV records across {cached_tickers:,} tickers. Progress counts attempted codes, so check the unavailable-share list too. Data and progress are in SQLite; back it up before replacing the app.")
    st.caption("Auto-continue processes one checkpointed batch at a time. Keep this browser tab open; if the Streamlit server sleeps or restarts, return here and resume. Progress is saved in the local SQLite database, which may be lost on cloud redeployment.")
    if "p4_auto_enabled" not in st.session_state:
        st.session_state["p4_auto_enabled"] = False
    col_auto,col_single,col_reset=st.columns(3)
    if col_auto.button("Stop auto-continue" if st.session_state["p4_auto_enabled"] else "Start auto-continue", type="primary", key="p4_toggle_auto", disabled=(done>=len(backtest_codes) or not backtest_codes) and not st.session_state["p4_auto_enabled"]):
        st.session_state["p4_auto_enabled"] = not st.session_state["p4_auto_enabled"]
        st.rerun()
    run_batch=col_single.button(f"Run one batch ({min(batch_size,max(0,len(backtest_codes)-done))})",key="p4_run_batch",disabled=(done>=len(backtest_codes) or not backtest_codes or st.session_state["p4_auto_enabled"]))
    reset_batch=col_reset.button("Restart this backtest run",key="p4_reset_batch",disabled=(done==0 or st.session_state["p4_auto_enabled"]))
    if reset_batch:
        set_batch_job_index(job_key,backtest_codes,interval,0)
        st.rerun()
    if run_batch:
        batch_codes=backtest_codes[done:min(done+batch_size,len(backtest_codes))]
        process_historical_batch(batch_codes,job_key,interval)
        set_batch_job_index(job_key,backtest_codes,interval,min(done+len(batch_codes),len(backtest_codes)))
        st.rerun()

    @st.fragment(run_every="10s")
    def p4_auto_worker():
        if not st.session_state.get("p4_auto_enabled"):
            return
        # Read the durable checkpoint on every tick instead of trusting a stale page variable.
        con=db_connect()
        record=con.execute("SELECT next_index FROM historical_batch_jobs WHERE job_key=?",(job_key,)).fetchone()
        con.close()
        current=min(int(record[0]) if record else 0,len(backtest_codes))
        if current>=len(backtest_codes):
            st.session_state["p4_auto_enabled"]=False
            st.success("All shares attempted. Auto-continue finished.")
            return
        chunk=backtest_codes[current:min(current+batch_size,len(backtest_codes))]
        st.info(f"Auto-continue: processing shares {current+1:,}–{current+len(chunk):,} of {len(backtest_codes):,}.")
        try:
            process_historical_batch(chunk,job_key,interval)
        except Exception as exc:
            # Preserve checkpoint and retry the same batch at the next interval.
            st.error(f"Batch interrupted ({type(exc).__name__}: {exc}). Will retry on the next refresh.")
            return
        set_batch_job_index(job_key,backtest_codes,interval,current+len(chunk))
        st.success(f"Saved checkpoint: {current+len(chunk):,}/{len(backtest_codes):,} shares. Next batch will start automatically.")

    p4_auto_worker()
    bt=load_historical_learning(backtest_codes)
    if not bt.empty:
        cutoff=signal_threshold
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
        render_strategy_optimiser(bt)
        render_portfolio_simulator(bt)
        render_automatic_strategy_testing(bt)
        # Independent of automatic strategy comparison / holdout completion.
        st.divider()
        st.subheader("50-day trend filter — accepted vs rejected")
        fe_saved = st.session_state.get("opt_result")
        if (fe_saved and not fe_saved.get("error") and fe_saved.get("result")
                and fe_saved.get("n") == len(bt)
                and fe_saved.get("last") == str(bt["Signal date"].max())):
            fe_top = fe_saved["result"]["table"].iloc[0]
            fe_rule = (int(fe_top["Score min"]),
                       None if pd.isna(fe_top["VPA min"]) else float(fe_top["VPA min"]),
                       None if pd.isna(fe_top["Accumulation min"]) else float(fe_top["Accumulation min"]),
                       None if pd.isna(fe_top["Distribution risk max"]) else 100 - float(fe_top["Distribution risk max"]))
            fe_hold = int(fe_top["Hold days"])
            fe_config = fe_saved
            st.caption("Using the current optimiser's score rules and holding period.")
        else:
            st.caption("No current optimiser results: use independent settings below to test the cached history.")
            fe_score = st.slider("Minimum signal score for filter comparison", 0, 100, int(cutoff), key="fe_independent_score")
            fe_hold = st.selectbox("Strategy holding period (trading sessions)", [5, 10, 20, 30], index=2, key="fe_independent_hold")
            fe_cost = st.number_input("Estimated round-trip trading costs (%)", min_value=0.0, max_value=10.0, value=0.8, step=0.1, key="fe_independent_cost")
            fe_price = st.number_input("Minimum share price (AUD)", min_value=0.0, max_value=100.0, value=0.01, step=0.01, key="fe_independent_price")
            fe_rule = (fe_score, None, None, None)
            fe_config = {"costs": fe_cost, "price": fe_price}
        fe_liquidity = st.number_input("Minimum average daily dollar volume (AUD) — filter comparison", min_value=0.0, value=20000.0, step=1000.0, key="fe_independent_liquidity")
        render_filter_effectiveness_analyser(bt, fe_rule, fe_hold, fe_config, fe_liquidity)
        st.subheader("Did learned weights help on later, unseen history?")
        st.write("This check trains weights only on matured outcomes from earlier dates, leaves a 45-day gap, then compares fixed and learned scores on the later holdout period. It tests whether the indicators ranked the later sample better; it cannot guarantee future performance.")
        holdout_key=(tuple(sorted(backtest_codes)),int(interval),int(cutoff))
        if st.button("Run learned-versus-fixed holdout check",key="p4_holdout_check"):
            holdout_result,holdout_error=evaluate_learned_holdout(bt,cutoff)
            st.session_state["p4_holdout_result"]={"key":holdout_key,"result":holdout_result,"error":holdout_error}
        saved_holdout=st.session_state.get("p4_holdout_result")
        if saved_holdout and saved_holdout.get("key")==holdout_key:
            if saved_holdout.get("error"):
                st.info(saved_holdout["error"])
            else:
                holdout=saved_holdout["result"]
                st.caption(f"Trained on {holdout['train_count']:,} effective examples dated before {holdout['train_cutoff']}; evaluated {holdout['test_rows']:,} displayed historical rows from {holdout['test_start']} onward.")
                st.dataframe(holdout["metrics"].round(2),use_container_width=True,hide_index=True)
                st.markdown("**Weights learned from the earlier period**")
                st.dataframe(holdout["weights"].round(3),use_container_width=True,hide_index=True)
                st.caption("Look for learned weights to improve positive-return rates and average returns on the later test rows, especially under the equal-sized Top 20% comparison. If results are mixed or worse, the learning has not shown a reliable improvement in this test.")
        elif not saved_holdout:
            st.caption("Run this check after the universe has enough saved history. It uses a time split and does not train on the later holdout outcomes.")
        else:
            st.caption("The selected universe, spacing, or threshold changed. Run the holdout check again for the current selection.")
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
    render_database_restore()

elif page=="Sector Performance":
    sector_performance_page()
elif page=="Sector Prediction Accuracy":
    sector_accuracy_page()

elif page=="Consistent Gains":
    render_strategy_screen(scan,"consistent")
elif page=="Breakout Opportunities":
    render_strategy_screen(scan,"breakout")
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

"""ASX Early-Mover Scanner — Phase 4 reconstruction.

Free-data research tool. Prices and historical backtests are not investment advice.
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except ImportError:
    yf = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path("data")
PREDICTIONS_FILE = DATA_DIR / "predictions.csv"
ASX_LIST_URLS = [
    "https://www.asx.com.au/asx/research/ASXListedCompanies.csv",
    "https://raw.githubusercontent.com/mbahar/ASX-listed-companies/master/ASXListedCompanies.csv",
]
HORIZONS = (5, 10, 20, 30)
st.set_page_config(page_title="ASX Early-Mover Scanner · Phase 4", page_icon="📈", layout="wide")


def load_universe(upload=None):
    if upload is not None:
        df = pd.read_csv(upload)
    else:
        local = APP_DIR / "ASX_Listed_Companies.csv"
        if local.exists():
            df = pd.read_csv(local)
        else:
            df = None
            for url in ASX_LIST_URLS:
                try:
                    df = pd.read_csv(url, encoding_errors="replace")
                    break
                except Exception:
                    continue
            if df is None:
                return pd.DataFrame({"Code": ["BHP", "CBA", "FMG", "WDS", "TLS", "NAB", "PLS", "LTR", "DRO", "4DX"]})
    code_col = next((c for c in df.columns if c.lower() in ("code", "ticker", "asx code", "symbol")), df.columns[0])
    return pd.DataFrame({"Code": df[code_col].astype(str).str.strip().str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)}).query("Code != ''").drop_duplicates()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_prices(symbol: str, period="1y"):
    if yf is None:
        raise RuntimeError("yfinance isn't installed. Check requirements.txt and redeploy.")
    df = yf.download(symbol + ".AX", period=period, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        raise ValueError("No price history returned by Yahoo Finance.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).title() for c in df.columns]
    if "Close" not in df:
        raise ValueError("Price feed did not include Close values.")
    return df.dropna(subset=["Close"])


def indicator_snapshot(df):
    close = df.Close.astype(float)
    vol = df.Volume.astype(float) if "Volume" in df else pd.Series(index=df.index, dtype=float)
    last = float(close.iloc[-1]); prev = float(close.iloc[-2]) if len(close) > 1 else last
    ma20 = float(close.tail(20).mean()); ma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
    ret5 = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0.0
    ret20 = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
    avgvol = float(vol.tail(20).mean()) if len(vol.dropna()) >= 5 else 0.0
    volratio = float(vol.iloc[-1] / avgvol) if avgvol > 0 and pd.notna(vol.iloc[-1]) else 1.0
    delta = close.diff(); gain = delta.clip(lower=0).rolling(14).mean(); loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, float("nan")); rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if len(close) >= 15 else 50.0
    if pd.isna(rsi): rsi = 100.0 if loss.iloc[-1] == 0 else 50.0
    score = 50
    reasons = []
    def add(points, good, bad):
        nonlocal score
        score += points
        if points > 0: reasons.append(good)
        elif points < 0: reasons.append(bad)
    add(15 if last > ma20 else -10, "Price is above its 20-day average", "Price is below its 20-day average")
    add(12 if ma20 > ma50 else -8, "Short trend is above the 50-day trend", "Short trend is below the 50-day trend")
    add(12 if ret20 > 5 else (5 if ret20 > 0 else -10), f"20-day price change is {ret20:+.1f}%", f"20-day price change is {ret20:+.1f}%")
    add(10 if 50 <= rsi <= 70 else (-8 if rsi > 80 or rsi < 30 else 0), f"RSI is {rsi:.0f}, within a moderate momentum range", f"RSI is {rsi:.0f}, outside the preferred range")
    add(8 if volratio >= 1.5 else (-3 if volratio < .5 else 0), f"Latest volume is {volratio:.1f}× its 20-day average", f"Latest volume is subdued at {volratio:.1f}× average")
    score = int(max(0, min(100, score)))
    risk = []
    if rsi > 80: risk.append("RSI is very high; momentum may be overextended")
    if last < ma20: risk.append("Price is below its 20-day average")
    if volratio < .5: risk.append("Recent trading volume is thin")
    if len(close) < 50: risk.append("Limited price history for stable trend comparison")
    return {"Price": last, "Score": score, "5d %": ret5, "20d %": ret20, "RSI": rsi, "Volume x": volratio,
            "As of": pd.Timestamp(df.index[-1]).strftime("%Y-%m-%d"), "Reasons": "; ".join(reasons),
            "Risks": "; ".join(risk) or "Price signals can reverse; company news is not assessed"}


def evaluate_history(df, ticker, horizons=HORIZONS, threshold=60):
    close = df.Close.astype(float).dropna()
    rows = []
    # Use only information available at each historical signal date. Forward returns are outcomes, not features.
    for i in range(55, len(close) - max(horizons), 5):
        hist = df.loc[:close.index[i]]
        snap = indicator_snapshot(hist)
        future = close.iloc[i + max(horizons)]
        row = {"Code": ticker, "Signal date": close.index[i], "Score": snap["Score"], "Signal": snap["Score"] >= threshold,
               "Entry": float(close.iloc[i]), "Future end": float(future), "Max horizon": max(horizons)}
        # Independent horizon outcome columns use the close after that number of trading sessions.
        for h in horizons:
            row[f"Return {h}d %"] = (float(close.iloc[i+h]) / float(close.iloc[i]) - 1) * 100
            row[f"Win {h}d"] = row[f"Return {h}d %"] > 0
        rows.append(row)
    return pd.DataFrame(rows)


def metric_summary(df, horizon):
    col = f"Return {horizon}d %"; win = f"Win {horizon}d"
    if df.empty: return {"Signals": 0, "Win rate": None, "Mean return": None, "Median return": None, "False positive": None}
    return {"Signals": int(len(df)), "Win rate": float(df[win].mean()*100), "Mean return": float(df[col].mean()),
            "Median return": float(df[col].median()), "False positive": float((df[col] < 0).mean()*100)}


def read_saved():
    if PREDICTIONS_FILE.exists():
        try: return pd.read_csv(PREDICTIONS_FILE)
        except Exception: pass
    return pd.DataFrame(columns=["Saved at", "Code", "Price", "Score", "As of", "Reasons", "Risks"])


def save_predictions(df):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    old = read_saved()
    new = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    new.insert(0, "Saved at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    all_rows = pd.concat([old, new], ignore_index=True).drop_duplicates(subset=["Saved at", "Code"])
    all_rows.to_csv(PREDICTIONS_FILE, index=False)
    return all_rows


st.title("📈 ASX Early-Mover Scanner · Phase 4")
st.caption("Research and measurement tool. Signals are estimates, not recommendations. Free price feeds may be delayed, incomplete, or unavailable for some ASX codes.")
with st.sidebar:
    st.header("Scan settings")
    upload = st.file_uploader("Optional ASX company list (CSV)", type="csv")
    universe = load_universe(upload)
    codes = universe.Code.tolist()
    mode = st.radio("Scan type", ["Quick scan", "Full market scan", "Choose shares"])
    if mode == "Quick scan":
        selected = codes[:min(40, len(codes))]
        st.caption(f"Scans up to {len(selected)} codes from the available list.")
    elif mode == "Full market scan":
        selected = codes
    else:
        selected = st.multiselect("ASX codes", codes, default=codes[:10])
    selected = st.text_input("Add codes (comma separated)", "").upper().replace(" ", "").split(",") + selected
    selected = list(dict.fromkeys(c for c in selected if c and c != ""))
    period = st.selectbox("Price history", ["6mo", "1y", "2y", "5y"], index=1)
    threshold = st.slider("High-score threshold", 40, 90, 65)
    run_scan = st.button("Run scan", type="primary", use_container_width=True)

tabs = st.tabs(["Scanner", "Prediction accuracy", "Historical backtest", "Saved predictions", "Help & data"])
if "scan" not in st.session_state: st.session_state.scan = pd.DataFrame()
if "failures" not in st.session_state: st.session_state.failures = []
if run_scan:
    results=[]; failures=[]
    progress=st.progress(0, text="Starting scan…")
    for i, code in enumerate(selected):
        try:
            hist=fetch_prices(code, period)
            snap=indicator_snapshot(hist)
            results.append({"Code":code, **snap})
        except Exception as exc:
            failures.append({"Code":code,"Reason":str(exc)[:240]})
        progress.progress((i+1)/max(1,len(selected)), text=f"Checked {i+1} of {len(selected)} codes")
        time.sleep(.02)
    st.session_state.scan=pd.DataFrame(results).sort_values("Score",ascending=False) if results else pd.DataFrame()
    st.session_state.failures=failures

with tabs[0]:
    st.subheader("Market scan")
    if not st.session_state.scan.empty:
        data=st.session_state.scan
        c1,c2,c3=st.columns(3); c1.metric("Shares with data",len(data)); c2.metric("High-score signals",int((data.Score>=threshold).sum())); c3.metric("Failed downloads",len(st.session_state.failures))
        st.dataframe(data, use_container_width=True, hide_index=True, column_config={"Price":st.column_config.NumberColumn(format="$%.4f"),"Score":st.column_config.ProgressColumn(min_value=0,max_value=100),"5d %":st.column_config.NumberColumn(format="%.2f%%"),"20d %":st.column_config.NumberColumn(format="%.2f%%")})
        st.download_button("Download scan CSV",data.to_csv(index=False),"asx_scan.csv","text/csv")
        if st.button("Save this scan to prediction history"):
            saved=save_predictions(data); st.success(f"Saved {len(data)} predictions. Total saved: {len(saved)}")
        st.markdown("#### Why these scores?")
        choice=st.selectbox("Choose a share to inspect",data.Code.tolist())
        row=data.loc[data.Code==choice].iloc[0]
        st.write("**Supporting factors:**",row.get("Reasons","—")); st.write("**Risks and limitations:**",row.get("Risks","—"))
    else: st.info("Choose a scan size in the sidebar and select **Run scan**.")
    if st.session_state.failures:
        with st.expander(f"Failed downloads ({len(st.session_state.failures)})"):
            st.dataframe(pd.DataFrame(st.session_state.failures),use_container_width=True,hide_index=True)

with tabs[1]:
    st.subheader("Prediction accuracy")
    st.markdown("Accuracy uses historical walk-forward simulations: each signal score is calculated only from prices before the signal date, then compared with later closes. It does not pretend that today's saved predictions already have future outcomes.")
    st.caption("The backtest is approximate: delisted shares, corporate actions, spread, slippage, dividends and historical ASX membership can bias results. Treat small samples cautiously.")
    st.info("Run a historical backtest below to produce accuracy metrics. Saved live predictions are listed separately until their future outcome can be observed.")

with tabs[2]:
    st.subheader("Historical walk-forward backtest")
    st.write("The app evaluates repeated historical signal dates using the same transparent indicator rules as the live scanner.")
    bt_codes=st.multiselect("Shares for backtest",codes,default=codes[:min(5,len(codes))],key="bt_codes")
    if st.button("Run backtest",key="run_bt"):
        rows=[]; failures=[]; p=st.progress(0)
        for i,code in enumerate(bt_codes):
            try:
                hist=fetch_prices(code,"5y"); one=evaluate_history(hist,code,threshold=threshold)
                if not one.empty: rows.append(one)
                else: failures.append({"Code":code,"Reason":"Not enough history for backtest."})
            except Exception as exc: failures.append({"Code":code,"Reason":str(exc)[:240]})
            p.progress((i+1)/max(1,len(bt_codes)))
        st.session_state.backtest=pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()
        st.session_state.bt_failures=failures
    bt=st.session_state.get("backtest",pd.DataFrame())
    if not bt.empty:
        for h in HORIZONS:
            metric=metric_summary(bt[bt.Signal],h)
            st.markdown(f"**{h} trading days**")
            cols=st.columns(5)
            for col,label,key,fmt in zip(cols,["Signals","Win rate","Average return","Median return","Negative-return rate"],["Signals","Win rate","Mean return","Median return","False positive"],["%d","%.1f%%","%+.2f%%","%+.2f%%","%.1f%%"]):
                val=metric[key]; col.metric(label,"—" if val is None else fmt % val)
        horizon=st.selectbox("Analysis timeframe",HORIZONS,index=2)
        retcol=f"Return {horizon}d %"
        bt=bt.copy(); bt["Score band"]=pd.cut(bt.Score,[0,49,64,79,100],labels=["0–49","50–64","65–79","80–100"],include_lowest=True)
        st.markdown("#### Calibration and false-positive analysis")
        st.caption("Positive-return rate by score band. A false positive here means the share ended below its entry close at the selected horizon.")
        summary=bt.groupby("Score band",observed=False).agg(Samples=(retcol,"count"),Positive_return_rate=(f"Win {horizon}d",lambda x:100*x.mean()),Average_return=(retcol,"mean"),Negative_return_rate=(retcol,lambda x:100*(x<0).mean())).reset_index()
        st.dataframe(summary,use_container_width=True,hide_index=True)
        st.markdown("#### Signal diagnostics")
        signals=bt[bt.Signal].sort_values(retcol)
        st.write("**Weakest outcomes (false positives):**")
        st.dataframe(signals.head(10)[["Code","Signal date","Score",retcol]],use_container_width=True,hide_index=True)
        st.write("**Strongest outcomes:**")
        st.dataframe(signals.tail(10).sort_values(retcol,ascending=False)[["Code","Signal date","Score",retcol]],use_container_width=True,hide_index=True)
        st.download_button("Download backtest CSV",bt.to_csv(index=False),"asx_backtest.csv","text/csv")
    else: st.info("Select a manageable sample and run the backtest. Large scans may take time because free data is fetched share by share.")
    if st.session_state.get("bt_failures"):
        with st.expander(f"Backtest download issues ({len(st.session_state.bt_failures)})"):
            st.dataframe(pd.DataFrame(st.session_state.bt_failures),use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("Saved prediction history")
    saved=read_saved()
    if saved.empty: st.info("No saved predictions yet. Run a scan and choose **Save this scan to prediction history**.")
    else:
        st.caption(f"{len(saved)} saved ticker observations. CSV storage persists only while the Streamlit instance's local filesystem persists; connect a durable database or repository storage for guaranteed retention.")
        st.dataframe(saved,use_container_width=True,hide_index=True)
        st.download_button("Download prediction history",saved.to_csv(index=False),"asx_prediction_history.csv","text/csv")
        st.markdown("Outcomes are not fabricated: each saved entry needs a later price observation to be marked complete. This build keeps saved signals distinct from historical simulations.")

with tabs[4]:
    st.subheader("What this version measures")
    st.markdown("""
- **Score (0–100):** a visible rule-based blend of price vs 20-day average, 20/50-day trend, 20-day return, RSI and volume. It is not a probability.
- **Historical evaluation:** walk-forward signals sampled every five trading sessions, measured after 5, 10, 20 and 30 sessions.
- **False positives:** high-score historical signals with negative forward returns; score bands show calibration without calling the score a probability.
- **Data dates:** each live row shows the last date supplied by the feed.
- **Limitations:** this build does not include company announcements, delisted-stock history, survivorship-free historical universes, or automatic model-weight changes. Metrics are descriptive and not evidence of future returns.

The Phase 3 source project was not attached to the referenced conversation or present in the workspace. This is a compatible reconstruction from the feature description, not a byte-for-byte continuation of the missing code. Review it before deployment.
""")

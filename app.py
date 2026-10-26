import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="ASX Share Analyser", page_icon="📈", layout="wide")
st.title("📈 ASX Share Analyser")
st.caption("Personal research prototype for Australian shares")
st.warning("Not financial advice. Free data can be incomplete, delayed or incorrect. Verify figures against company reports and ASX announcements. This is not an official QAV product.")

with st.sidebar:
    raw = st.text_input("ASX ticker", "BHP")
    period = st.selectbox("Price history", ["1y", "2y", "5y", "10y"], index=2)
    if st.button("Refresh data"):
        st.cache_data.clear()

ticker = raw.strip().upper()
if ticker and "." not in ticker:
    ticker += ".AX"
if not ticker:
    st.info("Enter an ASX ticker.")
    st.stop()

@st.cache_data(ttl=900, show_spinner=False)
def load(symbol, period):
    t = yf.Ticker(symbol)
    h = t.history(period=period, auto_adjust=True)
    try: info = t.info or {}
    except Exception: info = {}
    try: income = t.financials
    except Exception: income = pd.DataFrame()
    try: balance = t.balance_sheet
    except Exception: balance = pd.DataFrame()
    try: cash = t.cashflow
    except Exception: cash = pd.DataFrame()
    return h, info, income, balance, cash

def number(x):
    try:
        return None if x is None or pd.isna(x) else float(x)
    except Exception:
        return None

def fmt(x, kind="num"):
    x = number(x)
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
            if len(vals): return number(vals.iloc[0])
    return None

try:
    with st.spinner(f"Loading {ticker}..."):
        hist, info, income, balance, cash = load(ticker, period)
except Exception as e:
    st.error("Could not retrieve data. Check the ticker or try again later.")
    st.caption(str(e))
    st.stop()

if hist is None or hist.empty:
    st.error("No price history returned. Check the ticker or try another company.")
    st.stop()

close = hist["Close"].dropna()
price = number(close.iloc[-1]) if len(close) else None
prev = number(close.iloc[-2]) if len(close) > 1 else None
change = price / prev - 1 if price is not None and prev not in (None, 0) else None
name = info.get("longName") or info.get("shortName") or ticker

st.subheader(f"{name} ({ticker})")
a,b,c,d = st.columns(4)
a.metric("Latest available close", fmt(price, "money"), fmt(change, "pct") if change is not None else None)
b.metric("Market capitalisation", fmt(info.get("marketCap"), "large"))
c.metric("Sector", info.get("sector") or "N/A")
d.metric("Currency", info.get("currency") or "N/A")

ocf = stmt(cash, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
shares = number(info.get("sharesOutstanding")) or number(info.get("impliedSharesOutstanding"))
cash_ps = ocf / shares if ocf is not None and shares and shares > 0 else None
price_cash = price / cash_ps if price is not None and cash_ps not in (None, 0) else None
yield_raw = number(info.get("dividendYield"))
yield_pct = yield_raw * 100 if yield_raw is not None else None
pe = number(info.get("trailingPE"))
equity = stmt(balance, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"])
equity_ps = equity / shares if equity is not None and shares and shares > 0 else None
eps_auto = number(info.get("trailingEps"))
forward_eps_auto = number(info.get("forwardEps"))

price_tab, inputs_tab, score_tab, data_tab = st.tabs(["Price & technicals", "Checklist inputs", "Rule scorecard", "Raw data"])
with price_tab:
    chart = hist[["Close"]].copy()
    chart["50-day average"] = chart["Close"].rolling(50).mean()
    chart["200-day average"] = chart["Close"].rolling(200).mean()
    st.line_chart(chart, use_container_width=True)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    x,y,z,w = st.columns(4)
    x.metric("RSI (14)", fmt(rsi.iloc[-1] if len(rsi) else None))
    y.metric("50-day average", fmt(chart["50-day average"].iloc[-1], "money"))
    z.metric("200-day average", fmt(chart["200-day average"].iloc[-1], "money"))
    w.metric("MACD vs signal", "Positive" if len(macd) and macd.iloc[-1] > signal.iloc[-1] else "Negative")
    st.info("3-point trend-line (3PTL) sentiment must be assessed manually in this version; the app does not draw or validate 3PTL lines.")

with inputs_tab:
    st.markdown("### Available automatic data")
    auto = [
        ("Operating cash flow", fmt(ocf, "large")), ("Shares outstanding", fmt(shares)),
        ("Operating cash flow per share", fmt(cash_ps, "money")), ("Price / cash ratio", fmt(price_cash)),
        ("Dividend yield", fmt(yield_raw, "pct")), ("Trailing P/E", fmt(pe)),
        ("Total equity", fmt(equity, "large")), ("Equity per share", fmt(equity_ps, "money")),
        ("Trailing EPS", fmt(eps_auto, "money")), ("Forward EPS estimate", fmt(forward_eps_auto, "money"))
    ]
    st.dataframe(pd.DataFrame(auto, columns=["Metric", "Value"]), hide_index=True, use_container_width=True)
    st.markdown("### Manual checklist inputs")
    st.caption("Use current company reports or a reliable provider. Leave unverified items unassessed.")
    l,r = st.columns(2)
    with l:
        sentiment = st.selectbox("3PTL sentiment", ["Not assessed", "Positive / above buy line", "Negative / sell line breached", "Wait / needs uptick"])
        upturn = st.selectbox("Positive upturn since latest financial report?", ["Not assessed", "Yes", "No"])
        mortgage = st.number_input("Mortgage reference rate (%)", 0.0, 30.0, 6.0, 0.25)
        hist_pe = st.selectbox("P/E vs last 3 years / 6 half-year periods", ["Not assessed", "Lowest", "Neither / unknown", "Highest", "Unprofitable / unavailable"])
        equity_trend = st.selectbox("Equity trend", ["Not assessed", "Consistently increasing", "Not consistently increasing", "Insufficient history"])
        audit = st.selectbox("Latest audit opinion", ["Not checked", "Clean / unmodified", "Qualified / adverse / disclaimer", "Emphasis of matter / going concern concern", "Other / unclear"])
        founder = st.selectbox("Founder/board member owns at least 10%?", ["Not assessed", "Yes", "No"])
    with r:
        eps = st.number_input("Current EPS (override if needed)", value=float(eps_auto or 0), step=0.01, format="%.4f")
        future_eps = st.number_input("Future EPS estimate (0 if unknown)", value=float(forward_eps_auto or 0), step=0.01, format="%.4f")
        hurdle = st.number_input("Market hurdle rate (%)", 0.1, 50.0, 6.1, 0.1)
        star = st.selectbox("Verified external Star Stock score", ["Not assessed", "0", "0.5", "1", "1.5", "2"])
        external_iv = st.number_input("External intrinsic value per share (0 if unknown)", min_value=0.0, value=0.0, step=0.1)
        health_trend = st.selectbox("Verified financial-health trend", ["Not assessed", "Increasing", "Stable", "Deteriorating", "Other / unclear"])
        health_rating = st.selectbox("Verified financial-health rating", ["Not assessed", "Strong", "Satisfactory", "Other / weak"])
        commodity = st.selectbox("Relevant commodity in 3PTL sell? (if applicable)", ["Not assessed / not applicable", "Yes", "No"])

with score_tab:
    rows = []
    def add(rule, result, score=None, notes=""):
        rows.append({"Rule": rule, "Result": result, "Score": score if score is not None else "—", "Notes": notes})
    add("3PTL sentiment gate", sentiment, 1 if sentiment == "Positive / above buy line" else (0 if sentiment == "Negative / sell line breached" else None), "Manual chart assessment")
    add("Positive upturn since latest report", upturn, 1 if upturn == "Yes" else (0 if upturn == "No" else None))
    add("Operating cash flow positive", "Positive" if ocf is not None and ocf > 0 else ("Non-positive" if ocf is not None else "Missing"), None, fmt(ocf, "large"))
    add("Price / cash ratio below 7", "Pass" if price_cash is not None and 0 < price_cash < 7 else ("Fail" if price_cash is not None else "Missing"), 2 if price_cash is not None and 0 < price_cash < 7 else (0 if price_cash is not None else None))
    add("Dividend yield above mortgage rate", "Pass" if yield_pct is not None and yield_pct > mortgage else ("Fail" if yield_pct is not None else "Missing"), 1 if yield_pct is not None and yield_pct > mortgage else (0 if yield_pct is not None else None))
    add("P/E below dividend yield", "Pass" if pe is not None and yield_pct is not None and pe < yield_pct else ("Fail" if pe is not None and yield_pct is not None else "Missing"), 1 if pe is not None and yield_pct is not None and pe < yield_pct else (0 if pe is not None and yield_pct is not None else None))
    add("P/E lowest in historical period", hist_pe, {"Lowest": 2, "Highest": -1, "Unprofitable / unavailable": 0}.get(hist_pe), "Manual historical check")
    add("Equity consistently increasing", equity_trend, 1 if equity_trend == "Consistently increasing" else (0 if equity_trend == "Not consistently increasing" else None))
    add("Price below equity per share", "Pass" if price is not None and equity_ps is not None and price < equity_ps else ("Fail" if price is not None and equity_ps is not None else "Missing"), 1 if price is not None and equity_ps is not None and price < equity_ps else (0 if price is not None and equity_ps is not None else None))
    add("Price <= 1.3 x equity per share", "Pass" if price is not None and equity_ps is not None and price <= 1.3*equity_ps else ("Fail" if price is not None and equity_ps is not None else "Missing"), 1 if price is not None and equity_ps is not None and price <= 1.3*equity_ps else (0 if price is not None and equity_ps is not None and price <= 1.3*equity_ps else None))
    growth_pe = ((future_eps-eps)/eps)/pe if eps > 0 and future_eps > 0 and pe is not None and pe > 0 else None
    growth_score = 2 if growth_pe is not None and growth_pe > 1.5 else (-1 if growth_pe is not None and growth_pe < 0 else (0 if growth_pe is not None else None))
    add("EPS growth / P-E ratio", fmt(growth_pe), growth_score, "Estimate-based; verify inputs")
    iv1 = eps/0.195 if eps > 0 else None
    iv2 = future_eps/(hurdle/100) if future_eps > 0 and hurdle > 0 else None
    add("Intrinsic value #1: current EPS / 19.5%", fmt(iv1, "money"))
    add("Price below intrinsic value #1", "Pass" if price is not None and iv1 is not None and price < iv1 else ("Fail" if price is not None and iv1 is not None else "Missing"), 1 if price is not None and iv1 is not None and price < iv1 else (0 if price is not None and iv1 is not None else None))
    add(f"Intrinsic value #2: future EPS / {hurdle:.1f}%", fmt(iv2, "money"), notes="Check hurdle rate against a current authoritative source")
    add("Price below intrinsic value #2", "Pass" if price is not None and iv2 is not None and price < iv2 else ("Fail" if price is not None and iv2 is not None else "Missing"), 1 if price is not None and iv2 is not None and price < iv2 else (0 if price is not None and iv2 is not None and price < iv2 else None))
    add("Intrinsic value #2 > 2 x share price", "Pass" if price is not None and iv2 is not None and iv2 > 2*price else ("Fail" if price is not None and iv2 is not None else "Missing"), 1 if price is not None and iv2 is not None and iv2 > 2*price else (0 if price is not None and iv2 is not None else None))
    add("External Star Stock score", star, float(star) if star != "Not assessed" else None, "Manual verified input")
    add("Price below external intrinsic value", "Pass" if external_iv > 0 and price is not None and price < external_iv else ("Fail" if external_iv > 0 and price is not None else "Not assessed"), 1 if external_iv > 0 and price is not None and price < external_iv else (0 if external_iv > 0 and price is not None else None))
    add("External financial-health trend", health_trend, {"Increasing": 2, "Stable": 1, "Deteriorating": -1}.get(health_trend))
    add("External financial-health rating", health_rating, 1 if health_rating in ["Strong", "Satisfactory"] else (0 if health_rating == "Other / weak" else None))
    add("Founder/board ownership >= 10%", founder, 2 if founder == "Yes" else (0 if founder == "No" else None))
    audit_result = {"Not checked":"Not checked", "Clean / unmodified":"No warning recorded", "Qualified / adverse / disclaimer":"FAIL — review carefully", "Emphasis of matter / going concern concern":"WARNING — manual review", "Other / unclear":"Manual review required"}[audit]
    add("Audit opinion", audit_result, notes="Read the latest audit report")
    add("Commodity trend", commodity, notes="Manual check where relevant")
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    scores = [float(v) for v in df["Score"] if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v)]
    if scores:
        m1, m2 = st.columns(2)
        m1.metric("Sum of assessed points", f"{sum(scores):.1f}")
        m2.metric("Number of scored rules", len(scores))
    if sentiment == "Negative / sell line breached":
        st.error("Sentiment gate failed. Treat this as a screening flag, not an automatic trade.")
    elif sentiment == "Wait / needs uptick":
        st.warning("Wait state: verify an uptick before treating sentiment as positive.")
    if audit == "Qualified / adverse / disclaimer":
        st.error("Audit warning selected. Read the latest report before proceeding.")
    elif audit == "Emphasis of matter / going concern concern":
        st.warning("Review audit wording and going-concern disclosures carefully.")
    st.download_button("Download checklist CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{ticker.replace('.', '_')}_checklist.csv", mime="text/csv")

with data_tab:
    for label, frame in [("Income statement", income), ("Balance sheet", balance), ("Cash flow", cash)]:
        st.markdown(f"### {label}")
        if frame is not None and not frame.empty:
            st.dataframe(frame, use_container_width=True)
        else:
            st.info("No statement data returned.")
    st.markdown("### Recent prices")
    st.dataframe(hist.tail(30).sort_index(ascending=False), use_container_width=True)
    st.download_button("Download price history CSV", hist.to_csv().encode("utf-8"), file_name=f"{ticker.replace('.', '_')}_prices.csv", mime="text/csv")

st.divider()
st.caption("Personal-use prototype based on the document supplied by the user. Not affiliated with QAV, Stock Doctor, Yahoo Finance or ASX. Verify data and calculations independently.")

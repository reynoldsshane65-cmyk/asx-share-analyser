import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(page_title="ASX Share Analyser", page_icon="📈", layout="wide")

st.title("📈 ASX Share Analyser")
st.caption("Free research prototype · Australian shares · Data supplied by Yahoo Finance via yfinance")

st.warning(
    "Research tool only, not personal financial advice. Market data may be delayed, incomplete, or incorrect. "
    "Always verify important figures against company reports and official ASX announcements."
)

def asx_symbol(raw):
    symbol = raw.strip().upper()
    if not symbol:
        return ""
    if "." not in symbol:
        symbol += ".AX"
    return symbol

def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None

def fmt_num(value, style="number"):
    value = safe_float(value)
    if value is None:
        return "N/A"
    if style == "money":
        return f"${value:,.2f}"
    if style == "percent":
        return f"{value * 100:.1f}%"
    if style == "large":
        sign = "-" if value < 0 else ""
        v = abs(value)
        if v >= 1e12:
            return f"{sign}${v/1e12:.2f}T"
        if v >= 1e9:
            return f"{sign}${v/1e9:.2f}B"
        if v >= 1e6:
            return f"{sign}${v/1e6:.2f}M"
        return f"{sign}${v:,.0f}"
    return f"{value:,.2f}"

@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period, auto_adjust=True)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    try:
        financials = ticker.financials
    except Exception:
        financials = pd.DataFrame()
    try:
        balance = ticker.balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cashflow = ticker.cashflow
    except Exception:
        cashflow = pd.DataFrame()
    return hist, info, financials, balance, cashflow

def latest_statement_value(frame, possible_names):
    if frame is None or frame.empty:
        return None
    for name in possible_names:
        if name in frame.index:
            vals = frame.loc[name].dropna()
            if len(vals):
                return safe_float(vals.iloc[0])
    return None

def score_metric(value, good, bad, higher_is_better=True):
    if value is None or not np.isfinite(value):
        return None
    if higher_is_better:
        if value >= good: return 100
        if value <= bad: return 0
        return 100 * (value - bad) / (good - bad)
    else:
        if value <= good: return 100
        if value >= bad: return 0
        return 100 * (bad - value) / (bad - good)

def main():
    with st.sidebar:
        st.header("Analysis settings")
        raw_symbol = st.text_input("ASX ticker", value="BHP", help="Enter BHP or BHP.AX")
        period = st.selectbox("Price history", ["6mo", "1y", "2y", "5y", "10y"], index=2)
        st.caption("For ASX-listed shares, the app adds .AX automatically.")
        st.divider()
        st.markdown("**Example tickers**")
        st.write("BHP · CBA · CSL · WES · WOW · FMG · TLS · LKE")
        refresh = st.button("Refresh data")
        if refresh:
            st.cache_data.clear()

    symbol = asx_symbol(raw_symbol)
    if not symbol:
        st.info("Enter an ASX ticker in the sidebar to begin.")
        return

    try:
        with st.spinner(f"Loading {symbol} data..."):
            hist, info, financials, balance, cashflow = load_data(symbol, period)
    except Exception as e:
        st.error("Could not load data right now. Check the ticker and try again.")
        st.caption(f"Technical detail: {e}")
        return

    if hist is None or hist.empty:
        st.error(f"No price history was returned for {symbol}. Check the ticker; some ASX symbols may not be covered.")
        return

    company = info.get("longName") or info.get("shortName") or symbol
    latest_price = safe_float(hist["Close"].dropna().iloc[-1]) if hist["Close"].dropna().size else None
    prev_price = safe_float(hist["Close"].dropna().iloc[-2]) if hist["Close"].dropna().size > 1 else None
    day_change = (latest_price / prev_price - 1) if latest_price and prev_price else None
    first_price = safe_float(hist["Close"].dropna().iloc[0]) if hist["Close"].dropna().size else None
    period_change = (latest_price / first_price - 1) if latest_price and first_price else None

    st.subheader(company)
    st.caption(f"{symbol} · Sector: {info.get('sector', 'Not available')} · Industry: {info.get('industry', 'Not available')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Latest close", fmt_num(latest_price, "money"), fmt_num(day_change, "percent") if day_change is not None else None)
    m2.metric(f"Return over {period}", fmt_num(period_change, "percent"))
    m3.metric("Market cap", fmt_num(info.get("marketCap"), "large"))
    m4.metric("Currency", info.get("currency", "N/A"))

    tab1, tab2, tab3, tab4 = st.tabs(["Overview & chart", "Fundamentals", "Technical signals", "Scorecard"])

    with tab1:
        st.markdown("### Price history")
        chart_df = hist[["Close"]].copy()
        chart_df["50-day average"] = chart_df["Close"].rolling(50).mean()
        chart_df["200-day average"] = chart_df["Close"].rolling(200).mean()
        st.line_chart(chart_df, use_container_width=True)
        st.caption("Prices are adjusted historical prices where supplied by the data source; this is not a live trading feed.")
        st.markdown("### Company overview")
        st.write(info.get("longBusinessSummary", "No company description is available from the data source."))
        st.markdown("### Recent price data")
        st.dataframe(hist.tail(10).sort_index(ascending=False), use_container_width=True)

    with tab2:
        st.markdown("### Valuation and financial metrics")
        metrics = [
            ("Trailing P/E", info.get("trailingPE"), "number"),
            ("Forward P/E", info.get("forwardPE"), "number"),
            ("Price / book", info.get("priceToBook"), "number"),
            ("Price / sales", info.get("priceToSalesTrailing12Months"), "number"),
            ("Revenue growth", info.get("revenueGrowth"), "percent"),
            ("Earnings growth", info.get("earningsGrowth"), "percent"),
            ("Profit margin", info.get("profitMargins"), "percent"),
            ("Operating margin", info.get("operatingMargins"), "percent"),
            ("Return on equity", info.get("returnOnEquity"), "percent"),
            ("Debt / equity", info.get("debtToEquity"), "number"),
            ("Free cash flow", info.get("freeCashflow"), "large"),
            ("Dividend yield", info.get("dividendYield"), "percent"),
            ("52-week high", info.get("fiftyTwoWeekHigh"), "money"),
            ("52-week low", info.get("fiftyTwoWeekLow"), "money"),
        ]
        rows = [{"Metric": name, "Value": fmt_num(val, style)} for name, val, style in metrics]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("N/A means the data source did not provide that metric. Check the company's annual/half-year reports before relying on a figure.")

        st.markdown("### Latest available statement figures")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Income statement**")
            if financials is not None and not financials.empty:
                st.dataframe(financials.iloc[:, :4], use_container_width=True)
            else:
                st.info("Statement data unavailable.")
        with col2:
            st.markdown("**Balance sheet**")
            if balance is not None and not balance.empty:
                st.dataframe(balance.iloc[:, :4], use_container_width=True)
            else:
                st.info("Statement data unavailable.")
        with col3:
            st.markdown("**Cash flow**")
            if cashflow is not None and not cashflow.empty:
                st.dataframe(cashflow.iloc[:, :4], use_container_width=True)
            else:
                st.info("Statement data unavailable.")

    close = hist["Close"].dropna()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    latest_rsi = safe_float(rsi.iloc[-1]) if len(rsi) else None
    latest_sma50 = safe_float(sma50.iloc[-1]) if len(sma50) else None
    latest_sma200 = safe_float(sma200.iloc[-1]) if len(sma200) else None
    macd_fast = close.ewm(span=12, adjust=False).mean()
    macd_slow = close.ewm(span=26, adjust=False).mean()
    macd = macd_fast - macd_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    latest_macd_diff = safe_float((macd - signal).iloc[-1])

    with tab3:
        st.markdown("### Technical indicators")
        t1, t2, t3 = st.columns(3)
        t1.metric("RSI (14)", fmt_num(latest_rsi))
        t2.metric("50-day average", fmt_num(latest_sma50, "money"))
        t3.metric("200-day average", fmt_num(latest_sma200, "money"))
        signals = []
        if latest_price is not None and latest_sma50 is not None:
            signals.append(("Price vs 50-day average", "Above" if latest_price > latest_sma50 else "Below"))
        else:
            signals.append(("Price vs 50-day average", "Insufficient history"))
        if latest_sma50 is not None and latest_sma200 is not None:
            signals.append(("50-day vs 200-day average", "50-day above 200-day" if latest_sma50 > latest_sma200 else "50-day below 200-day"))
        else:
            signals.append(("Moving-average trend", "Insufficient history"))
        if latest_rsi is not None:
            rsi_text = "Potentially overbought (>70)" if latest_rsi > 70 else ("Potentially oversold (<30)" if latest_rsi < 30 else "Between 30 and 70")
        else:
            rsi_text = "Insufficient history"
        signals.append(("RSI interpretation", rsi_text))
        if latest_macd_diff is not None:
            signals.append(("MACD momentum", "MACD above signal" if latest_macd_diff > 0 else "MACD below signal"))
        else:
            signals.append(("MACD momentum", "Insufficient history"))
        st.dataframe(pd.DataFrame(signals, columns=["Indicator", "Current reading"]), hide_index=True, use_container_width=True)
        st.caption("Technical indicators describe past price behaviour. They do not reliably predict future prices.")

    with tab4:
        st.markdown("### Transparent research score")
        st.write("Scores are rough screening signals, not recommendations. Missing values are excluded rather than treated as good or bad.")
        rev_growth = safe_float(info.get("revenueGrowth"))
        earnings_growth = safe_float(info.get("earningsGrowth"))
        margin = safe_float(info.get("profitMargins"))
        roe = safe_float(info.get("returnOnEquity"))
        pe = safe_float(info.get("trailingPE"))
        debt_equity = safe_float(info.get("debtToEquity"))

        growth_scores = [score_metric(rev_growth, 0.20, -0.10), score_metric(earnings_growth, 0.20, -0.20)]
        growth_scores = [x for x in growth_scores if x is not None]
        quality_scores = [score_metric(margin, 0.20, -0.05), score_metric(roe, 0.20, 0.0)]
        quality_scores = [x for x in quality_scores if x is not None]
        value_score = score_metric(pe, 12, 45, higher_is_better=False) if pe is not None and pe > 0 else None
        debt_score = score_metric(debt_equity, 50, 200, higher_is_better=False) if debt_equity is not None and debt_equity >= 0 else None
        quality_combined = [x for x in quality_scores + ([debt_score] if debt_score is not None else []) if x is not None]
        momentum_score = None
        if latest_price is not None and latest_sma50 is not None and latest_sma200 is not None:
            momentum_score = (50 if latest_price >= latest_sma50 else 0) * 0.5 + (50 if latest_sma50 >= latest_sma200 else 0.5 * 50)
        components = {
            "Growth": (float(np.mean(growth_scores)) if growth_scores else None, 35),
            "Valuation": (value_score, 25),
            "Quality & balance sheet": (float(np.mean(quality_combined)) if quality_combined else None, 25),
            "Price trend": (momentum_score, 15),
        }
        available = [(score, weight) for score, weight in components.values() if score is not None]
        overall = sum(score * weight for score, weight in available) / sum(weight for score, weight in available) if available else None
        if overall is None:
            st.info("Not enough financial data is available to calculate a score for this company.")
        else:
            st.metric("Available-data score", f"{overall:.0f}/100")
            st.progress(int(max(0, min(100, overall))))
        score_rows = []
        for label, (score, weight) in components.items():
            score_rows.append({
                "Category": label,
                "Score": f"{score:.0f}/100" if score is not None else "N/A",
                "Model weight": f"{weight}%",
                "Status": "Calculated" if score is not None else "Missing data"
            })
        st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
        st.caption("This simple first-version model uses fixed thresholds. It does not account for industry differences, commodity cycles, one-off earnings, dilution, or the full quality of company disclosures.")
        st.markdown("### Research checklist")
        st.checkbox("I checked the latest ASX announcements")
        st.checkbox("I reviewed annual and half-year reports")
        st.checkbox("I checked cash flow, debt, and possible share dilution")
        st.checkbox("I compared the company with peers in the same industry")

    st.divider()
    st.markdown("### Export price history")
    export_df = hist.copy()
    export_df.index = export_df.index.strftime("%Y-%m-%d")
    st.download_button(
        "Download price history as CSV",
        data=export_df.to_csv().encode("utf-8"),
        file_name=f"{symbol.replace('.', '_')}_price_history.csv",
        mime="text/csv"
    )
    st.caption(f"Last refreshed in this session: {date.today().isoformat()}. Confirm price timestamps and company fundamentals independently.")

main()

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="ASX Share Analyser", page_icon="📈", layout="wide")
st.title("📈 ASX Share Analyser")
st.caption("Research prototype for Australian shares — no automatic trading")
st.warning("Not financial advice. Free data can be incomplete, delayed or incorrect. Verify company figures against ASX announcements and official reports. Scores are screening aids, not predictions.")

STARTER_TICKERS = ["BHP","CBA","CSL","WES","WOW","RIO","NAB","WBC","ANZ","FMG","MQG","GMG","REA","CAR","XRO","ALL","COL","TLS","STO","WDS","ORG","PLS","MIN","IGO","LYC","DRO","DMP","ZIP","TLX","PDN","BOE","LTR","SYR","CXO","A2M","BRG","IEL","ARB","CPU","QAN"]

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

@st.cache_data(ttl=900, show_spinner=False)
def load_prices(symbol, period):
    try:
        h = yf.Ticker(symbol).history(period=period, auto_adjust=True)
        if h is None or h.empty or "Close" not in h: return pd.DataFrame()
        return h[[c for c in ["Close", "Volume"] if c in h.columns]].dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_info(symbol):
    try: return yf.Ticker(symbol).info or {}
    except Exception: return {}

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

def run_scan(tickers, period):
    rows=[]
    progress=st.progress(0, text="Preparing scan…")
    for i,sym in enumerate(tickers):
        sym=sym.strip().upper().replace(".AX","")
        ysym=sym+".AX"
        h=load_prices(ysym,period)
        if h.empty or len(h)<30:
            progress.progress((i+1)/len(tickers),text=f"Checking {sym}…")
            continue
        close=h["Close"].dropna()
        vol=h["Volume"].fillna(0) if "Volume" in h else pd.Series(dtype=float)
        ma200=close.tail(200).mean() if len(close)>=200 else np.nan
        v20=num(vol.tail(20).mean()) if len(vol)>=20 else None
        vprev=num(vol.iloc[-40:-20].mean()) if len(vol)>=40 else None
        info=load_info(ysym)
        rows.append({
            "Ticker":sym,"Company":info.get("longName") or info.get("shortName") or sym,
            "Price":num(close.iloc[-1]),"Return 1W %":pct(close,5),"Return 1M %":pct(close,21),
            "Return 3M %":pct(close,63),"Return 6M %":pct(close,126),"Return 12M %":pct(close,252),
            "50D Average":num(close.tail(50).mean()) if len(close)>=50 else np.nan,
            "200D Average":num(ma200),"Above 200D %":(close.iloc[-1]/ma200-1)*100 if pd.notna(ma200) and ma200 else np.nan,
            "Volume Change %":(v20/vprev-1)*100 if v20 is not None and vprev not in (None,0) else np.nan,
            "Market Cap":info.get("marketCap"),"P/E":info.get("trailingPE"),
            "Revenue Growth %":info.get("revenueGrowth")*100 if num(info.get("revenueGrowth")) is not None else np.nan,
            "Profit Margin %":info.get("profitMargins")*100 if num(info.get("profitMargins")) is not None else np.nan,
            "Data Date":close.index[-1].strftime("%Y-%m-%d")
        })
        progress.progress((i+1)/len(tickers),text=f"Scanning {sym}…")
    progress.empty()
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    return pd.concat([df,df.apply(score_row,axis=1)],axis=1)

with st.sidebar:
    page=st.radio("Page",["Single-Share Analysis","Automatic Share Scanner","Lower-Risk Investing","Value Investing","Momentum Trading","High-Growth Shares","Long-Term Investing","Portfolio Planner"])
    st.divider()
    ticker_text=st.text_area("ASX tickers (commas or new lines)",", ".join(STARTER_TICKERS),height=130)
    tickers=list(dict.fromkeys(x.strip().upper().replace(".AX","") for x in ticker_text.replace("\n",",").split(",") if x.strip()))
    max_tickers=st.slider("Maximum tickers per scan",5,150,40,5)
    tickers=tickers[:max_tickers]
    scan_label=st.selectbox("Scanner price history",["1 year","2 years","5 years"])
    scan_period={"1 year":"1y","2 years":"2y","5 years":"5y"}[scan_label]
    if st.button("Clear cached data"): st.cache_data.clear()
    if st.button("Run / refresh multi-share scan",type="primary",use_container_width=True):
        if not tickers: st.error("Enter at least one ticker.")
        else:
            with st.spinner("Scanning shares; larger lists may take several minutes…"):
                st.session_state["scan_results"]=run_scan(tickers,scan_period)
                st.session_state["scan_time"]=datetime.now().strftime("%Y-%m-%d %H:%M:%S")

scan=st.session_state.get("scan_results",pd.DataFrame())

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
    cols=["Ticker","Company","Price","Return 1W %","Return 1M %","Return 3M %","Return 6M %","Return 12M %","Momentum Score","Value Score","Growth Score","Overall Score","Market Cap","P/E","Revenue Growth %","Profit Margin %","Data Date"]
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

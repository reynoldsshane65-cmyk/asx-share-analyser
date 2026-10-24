# ASX Share Analyser — Free Starter App

This is a beginner-friendly Streamlit web app for researching ASX-listed shares using free, unofficial market data.

## What it does
- Looks up share prices and historical data
- Displays moving averages, RSI and MACD momentum
- Displays available fundamental metrics and financial statements
- Calculates a simple, transparent screening score
- Exports price history to CSV

## Important limitations
- Data is supplied through `yfinance`, which accesses Yahoo Finance data. It is not an official ASX feed and can be delayed, incomplete, or unavailable.
- Fundamentals may be missing or inconsistent for ASX securities.
- Scores are rough screening tools, not buy/sell recommendations or financial advice.
- Verify information with official company reports and ASX announcements.
- This app is a prototype. It does not save a persistent watchlist, user account, or alerts yet.

## Run on your own computer
1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/
2. Extract this folder.
3. Open a terminal / Command Prompt in this folder.
4. Run:
   `python -m pip install -r requirements.txt`
5. Run:
   `python -m streamlit run app.py`
6. A browser tab should open. Enter an ASX ticker such as `BHP` or `BHP.AX`.

On Windows, if `python` is not recognised, try `py` instead.

## Publish free online with Streamlit Community Cloud
1. Create a free GitHub account at https://github.com/
2. Create a new repository, for example `asx-share-analyser`.
3. Upload `app.py` and `requirements.txt` from this folder to the repository root.
4. Go to https://share.streamlit.io/ and sign in with GitHub.
5. Choose **Create app**, select the repository, branch (usually `main`) and main file path `app.py`.
6. Deploy. Streamlit will install the requirements and provide a web link.
7. Open that link on your iPhone and use Safari's **Share → Add to Home Screen** if you want an app-like shortcut.

## If something goes wrong
- Confirm the ticker is valid and try the `.AX` suffix.
- Wait and retry if Yahoo Finance returns a rate limit or temporary error.
- If a company has no financial figures, that is a data coverage limitation; check the company's latest reports.
- Do not enter brokerage passwords, API secrets, or other sensitive information into this app.

## Free tier caveat
Hosting providers and data sources can change their free-tier limits and terms. Check current terms before relying on the app or distributing it publicly.

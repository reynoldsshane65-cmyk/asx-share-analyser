# ASX Share Analyser — GitHub starter files

Personal-use Streamlit prototype based on the investment checklist document you supplied. Not an official QAV product and not affiliated with QAV, Stock Doctor, Yahoo Finance or ASX.

## Files to upload
- `app.py` — the main app
- `requirements.txt` — Python packages Streamlit installs
- `README.md` — this guide

Upload all three files into the **root/top level** of your GitHub repository, not into a subfolder.

## Streamlit deployment fields
- **Repository:** your GitHub username/repository name
- **Branch:** `main` (or the branch name shown in GitHub)
- **Main file path:** `app.py`

## Included in this version
- ASX ticker lookup and historical prices using the free, unofficial `yfinance` library
- Historical price chart, 50-day/200-day moving averages, RSI and MACD
- Available income statement, balance sheet and cash-flow data
- Cash-flow-per-share, price/cash, equity-per-share, EPS growth/P-E and intrinsic-value calculations
- Manual checklist inputs for 3PTL sentiment, recent upturn, historical P/E, equity trend, audit opinion, ownership, external ratings and commodity trend
- Rule scorecard and CSV downloads

## Limitations
- Free data coverage for ASX shares is inconsistent; unavailable values display as `N/A`.
- `yfinance` is unofficial, not an official ASX feed. Data may be delayed, incomplete or unavailable.
- This version analyses one ticker at a time; it does not automatically scan all ASX shares.
- 3PTL interpretation, audit opinions, founder/board ownership, historic P/E comparisons and external ratings need manual verification.
- The scoring and valuation formulas are a prototype. Verify each formula and input against your original document and current company reports before relying on results.
- This is a research aid, not a validated predictor of returns or personal financial advice.

## Deploy for free
1. Sign in to GitHub and create your repository if you have not already.
2. Open the repository and choose **Add file → Upload files**.
3. Upload `app.py`, `requirements.txt` and `README.md`.
4. Click **Commit changes**.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Select your repository and branch, set **Main file path** to `app.py`, then deploy.
7. If deployment fails, open the Streamlit logs and check the error message.

## Run locally (optional)
Install Python, open a terminal in this folder, then run:
```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Personal-use / copyright note
Your supplied document states that its content is for personal use and restricts commercial use without written permission. Keep this project for personal use unless you have confirmed the rights required for distribution or commercial use.

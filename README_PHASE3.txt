ASX SHARE ANALYSER — PHASE 3: SELF-LEARNING PREDICTION SYSTEM

FILES
- app.py: complete updated Streamlit app, based on the current Early-Mover Scanner version.
- ASX_Listed_Companies.csv: ASX company universe used by the scanner.
- requirements.txt: Python dependencies.

INSTALL / UPDATE
1. Download and extract this ZIP.
2. In the GitHub repository connected to Streamlit Community Cloud, replace the existing app.py with this app.py.
3. Upload ASX_Listed_Companies.csv to the same folder as app.py (replace the old copy if present).
4. Replace/update requirements.txt with the supplied requirements.txt.
5. Commit the changes and let Streamlit Community Cloud redeploy.
6. Open the app and run a full-market or quick scan.
7. Open Early-Mover Tracker, select the shares you want to test, then click “Save selected predictions to history”.
8. Open Prediction Lab and use “Refresh outcomes from Yahoo Finance” when enough trading sessions have passed.

HOW THE LEARNING WORKS
- Each saved signal records ticker, signal date, entry closing price, indicator values and the score/weights used.
- The app checks the first 5th, 20th and 60th subsequent trading-session closes and records returns from the entry close.
- After at least 30 completed 20-trading-day outcomes, the adaptive weighting turns on. It blends the original weights with measured correlations between each indicator and subsequent 20-day returns.
- Until that minimum sample exists, the original transparent scoring weights remain in use.
- Duplicate ticker + signal-date predictions are ignored.

IMPORTANT LIMITATIONS
- The prediction history is stored in a local SQLite database beside app.py. Streamlit Community Cloud local storage is not guaranteed to survive rebuilds, restarts or instance changes. Download the complete prediction-history CSV from Prediction Lab regularly and keep a backup. For dependable permanent history, a later upgrade should connect a persistent database (for example Supabase/Postgres).
- Outcome checks require Yahoo Finance to return price history. Missing/suspended/delisted tickers or Yahoo throttling may delay or prevent an outcome update.
- Returns are measured from the recorded closing price, excluding brokerage, spreads, slippage, dividends and tax. They are research measurements, not executable trade returns.
- The adaptive model is experimental and can overfit. Correlation is not causation. It is not a guarantee of improved future performance and does not place trades.
- This app is for research/education, not personal financial advice.

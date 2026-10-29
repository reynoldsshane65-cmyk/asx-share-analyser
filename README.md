# ASX Share Analyser — Early-Mover Phase

## Files
- `app.py` — updated Streamlit app, based on the working `appy_fixed.py` scanner.
- `ASX_Listed_Companies.csv` — bundled 1,810-code ASX company list.
- `requirements.txt` — Python dependencies.

## Deploy/update on Streamlit Community Cloud
1. Unzip this package.
2. Upload/replace `app.py`, `requirements.txt`, and `ASX_Listed_Companies.csv` in the same GitHub repository folder.
3. In Streamlit Community Cloud, make sure the app entry point is `app.py`.
4. Reboot the app if it does not refresh automatically.

## New in this phase
- Full-market scan defaults to the bundled ASX list (1,810 codes).
- Quick promising-shares scan uses the best-scoring available results from the previous scan when present; otherwise it falls back to a smaller starter list.
- Early-Mover Tracker ranks scanned shares using a transparent technical score from momentum, volume expansion, moving-average strength, and distance from recent highs.
- Downloadable rankings and tracker CSVs.
- Add/remove tracked candidates and compare their latest price/score with the values when added.

## Important limitations
- This remains a research/screening tool, not an automatic trading system or a reliable prediction of 20–50% returns.
- The early-mover score is a heuristic, not a probability.
- Tracker entries live in Streamlit session state and may reset when the app session restarts. Download the tracker CSV to retain a copy.
- The app currently uses Yahoo Finance data via `yfinance`; missing or delayed data is possible. Verify material company news and announcements directly with ASX.

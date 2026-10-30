# ASX Early-Mover Scanner — Phase 4

## Included
- `app.py`: reconstructed Streamlit app with quick, full-list and selected-share scans; scoring explanations; failed-download reporting; CSV exports; historical walk-forward evaluation at 5, 10, 20 and 30 trading days; score-band calibration and false-positive analysis; and saved signal history.
- `requirements.txt`: app dependencies.
- `README.txt`: setup and deployment notes.

## Data and limitations
The app uses Yahoo Finance through `yfinance` for adjusted ASX price history and tries a free ASX company CSV source, with a small fallback list if it cannot fetch one. Yahoo Finance coverage may omit small or illiquid shares. Upload your own current ASX code CSV in the sidebar to control coverage. Historical backtesting uses today's available ticker set and does not contain delisted shares, so survivorship bias is possible. It omits fees, spreads, slippage and announcements. A score is a rule-based ranking, not a probability or a recommendation.

Live saved predictions are separate from simulated historical outcomes. CSV files on Streamlit's local filesystem may not persist across restarts or redeployments; download them regularly or configure durable storage.

## Deploy on Streamlit Community Cloud
1. Download and extract the ZIP.
2. In your existing GitHub repository, replace `app.py` and `requirements.txt` with these files. Keep your existing ASX list CSV if you have one; upload it to GitHub as `ASX_Listed_Companies.csv` if desired.
3. Commit the replacements to the branch used by your Streamlit app.
4. Wait for Streamlit to install the requirements and redeploy.
5. Open the app, choose a small scan first, and review the data dates and failed-download list.
6. Download prediction-history CSV regularly. Do not delete any existing saved CSV before backing it up.

This package is a reconstruction because the referenced chat did not include the Phase 3 source files. Review and run in your own Streamlit environment before relying on it.

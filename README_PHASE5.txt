ASX SHARE ANALYSER — PHASE 5: VOLUME PRICE ANALYSIS (VPA)

THIS VERSION BUILDS ON THE WORKING PHASE 3 BASELINE.
It retains the ASX scanner, early-mover tracker, prediction history, 5/20/60-session outcome checks, accuracy dashboard, adaptive learning and CSV exports.

NEW IN PHASE 5
- Downloads daily Open, High, Low, Close and Volume data for VPA calculations.
- Adds a VPA Scanner page with rankings and downloadable CSV.
- Calculates a 0–100 VPA score, accumulation score, distribution-risk score and effort-versus-result score.
- Flags possible no-supply, no-demand, stopping-volume, volume-confirmed breakout, upthrust and distribution events.
- Adds VPA features to the Early-Mover Score and to the adaptive learner's feature set.
- Saves VPA feature values with new predictions so later 20-day outcomes can help evaluate their usefulness.
- Existing Phase 3 SQLite prediction records remain compatible. Older predictions have no historical VPA readings and are not retroactively filled.

INSTALL / UPDATE
1. Download and extract this ZIP.
2. Replace the existing app.py in the GitHub repository with the supplied app.py.
3. Keep the existing ASX_Listed_Companies.csv in the same folder. The supplied copy is included for convenience; use your latest company list if it is newer.
4. Update requirements.txt with the supplied file (the dependencies are unchanged).
5. Do not delete your existing asx_prediction_history.sqlite file if it is present in the deployed app environment. Streamlit Community Cloud local storage may not persist across rebuilds/restarts; keep downloading the prediction-history CSV backup from Prediction Lab.
6. Commit changes and allow Streamlit Community Cloud to redeploy.
7. Run a fresh scan so OHLCV and VPA fields are calculated for the shares. Existing scan results held in the current session may not contain the new columns; refresh the scan if necessary.
8. Open VPA Scanner to inspect the rankings. Open Early-Mover Tracker to see the VPA-enhanced score and record new predictions.

SCORING / LEARNING NOTES
- The original eight early-mover features retain 80% combined baseline weight. VPA score, accumulation, distribution safety and effort/result account for the other 20% (12%, 4%, 2%, 2%).
- Once at least 30 completed 20-trading-day outcomes exist, the existing adaptive weighting method can adjust the combined 12 feature weights using historical correlations.
- This is a transparent experimental heuristic, not a trained price-prediction model or proof of institutional trading. Correlation is not causation. Thinly traded shares, corporate actions, announcements and stale/missing data can create misleading patterns.
- A VPA score is not a probability, forecast, or buy/sell recommendation. Validate against charts and ASX announcements.

FILES
- app.py — updated Streamlit app
- ASX_Listed_Companies.csv — ASX company list
- requirements.txt — dependencies
- README_PHASE5.txt — these instructions


PHASE 5 FIX: This build includes backward compatibility for older cached scan results that do not yet contain VPA columns. If you see blank VPA scores on old results, run a fresh scan to calculate them from OHLCV price history.

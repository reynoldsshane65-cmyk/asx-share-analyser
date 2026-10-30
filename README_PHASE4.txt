ASX Share Analyser — Phase 4

This package preserves the supplied Phase 3 app and adds a Phase 4 historical backtest page.

Preserved Phase 3 features include single-share analysis, automatic full-market and quick scans, the ASX company list, Early-Mover Tracker and watchlist, saved SQLite prediction history, outcome refresh, 5/20/60-day prediction metrics, the learning engine, investing screens, portfolio planner, failure reporting and CSV exports.

Phase 4 adds a historical walk-forward replay of the existing Early-Mover score, measured at 5, 10, 20 and 30 trading days. It includes accuracy and average/median return by timeframe, positive/negative outcome rates by score bucket, false-positive and strongest/weakest signal examples, indicator/outcome correlations, failed ticker reporting, and CSV export. Backtest weights are fixed to the original Phase 3 weights to avoid applying today's learned weights to past observations. The historical simulation is separate from saved live predictions.

Install/replace in the existing GitHub repository:
1. Download and extract `asx_phase4_replacement.zip`.
2. Back up your repository's existing `asx_prediction_history.sqlite` if you have one. Do not replace or delete it; the app creates/uses this file beside `app.py`.
3. Replace `app.py`, `requirements.txt`, and `ASX_Listed_Companies.csv` with the package files. Keep any other repository files and Streamlit settings.
4. Commit the replacements to the branch connected to Streamlit Community Cloud.
5. Wait for redeployment, then open the app and first try a small sample in **Phase 4 Backtest**.
6. Review the listed data dates and unavailable tickers. Download CSV backups of prediction history and backtest results regularly.

The historical test uses Yahoo Finance price history through yfinance and the current listed-share universe. Yahoo coverage may omit small, illiquid or delisted shares. The backtest has survivorship bias and omits dividends, transaction costs, spreads, slippage and market impact. It is a research diagnostic, not investment advice, a prediction, or evidence of future performance. Backtest output and successful syntax checks do not verify your live Streamlit deployment.

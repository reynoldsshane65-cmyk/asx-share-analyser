ASX Share Analyser — Phase 4

This package preserves the supplied Phase 3 app and adds resumable maximum-history backtesting and broader OHLCV-based learning.

Preserved Phase 3 features include single-share analysis, full-market and quick scans, the bundled ASX company list, Early-Mover Tracker and watchlist, saved SQLite prediction history, outcome refresh, adaptive scoring, investing screens, portfolio planner, failed-download reporting, and CSV exports.

PHASE 4 HISTORICAL LEARNING
- Select Entire listed universe or a subset. The bundled current list contains 1,810 company codes; it is not the historical membership of the ASX.
- Each batch requests the maximum daily adjusted OHLCV history Yahoo Finance makes available for each code. Batch size is adjustable, and the SQLite database saves progress, downloaded OHLCV, and evaluated outcomes between batches. Click Run next batch until the progress counter reaches the end.
- Default signal spacing is five trading sessions. Historical samples use trailing data only and measure later 5, 10, 20, and 30 trading-day returns.
- Phase 4 adds daily range, close position in the daily range, opening gap, and 20-day volatility to Phase 3 momentum, trend and volume factors. Early-Mover scores display observed factors, weights, contributions, and technical risks.
- Historical completed outcomes combine with completed live predictions to adapt future Early-Mover weights after at least 30 learning observations. The learning pool samples about one signal per ticker per calendar month to limit the effect of overlapping 20-day returns.
- Dashboard tables use a representative sample (up to about 30 rows per share) to keep memory use manageable. Full downloaded OHLCV and evaluation rows remain in SQLite. CSV export from the dashboard is the displayed sample.

INSTALL / UPDATE
1. Download and extract `asx_phase4_replacement.zip`.
2. Back up your existing `asx_prediction_history.sqlite`. Keep it in the same folder as `app.py`; the updated app adds tables/columns while retaining saved predictions.
3. Replace `app.py`, `requirements.txt`, and `ASX_Listed_Companies.csv` in the GitHub repository connected to Streamlit Community Cloud. Keep other project files.
4. Commit the replacements and wait for Streamlit to redeploy.
5. Open Phase 4 Backtest, select Entire listed universe, keep 5-day spacing initially, and use a cautious batch size such as 20.
6. Click Run next batch repeatedly until the progress counter reaches the end. At the default 20 codes per batch, the 1,810-code list takes about 91 batch clicks; a larger batch setting reduces clicks but may be more likely to be throttled. The run resumes from its saved cursor; avoid changing the universe or spacing mid-run. A failed code is listed; Restart this backtest run retries the universe.
7. Review unavailable codes and the learning status in Prediction Lab. Back up both prediction-history CSV and `asx_prediction_history.sqlite` regularly; Streamlit local storage can reset after redeployment.

LIMITATIONS
History length differs by company; newer listings cannot have 50 years of prices. Yahoo may omit small, illiquid or delisted shares, and historical adjustments can be revised. The test uses today's listed companies, so it has survivorship bias and cannot recreate what the scanner would have seen across the whole market at each past date. It uses OHLCV-derived technical factors, not historical announcements or financial statements. Fees, spreads, slippage and market impact are excluded. Correlation shows association in this sample, not the cause of a price rise or proof of future performance. This is a research tool, not investment advice. Successful packaging does not verify live Streamlit deployment.

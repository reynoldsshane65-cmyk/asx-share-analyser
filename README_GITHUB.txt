ASX Share Analyser - Phase 5 VPA with restored Phase 4 Backtest
Prepared 30 October 2026

INSTALL
1. Back up your current asx_prediction_history.sqlite before redeploying.
2. Replace app.py, requirements.txt and ASX_Listed_Companies.csv in the GitHub
   repository connected to your Streamlit app. Keep your other project files.
3. Commit the replacements and let Streamlit redeploy.
4. Keep the database filename and location unchanged. This app extends existing
   tables without deleting saved predictions. The ZIP contains no database.
   Community Cloud's local storage is not durable; keep an external backup.

RESTORED
The sidebar includes Phase 4 Backtest with the complete original interface:
entire listed universe or selected shares, signal spacing, batch size, score
threshold, automatic remaining batches, one batch, restart, saved progress,
OHLCV cache, unavailable-share list, timeframe accuracy, calibration, weakest
and best outcomes, indicator diagnostics, CSV export and learned/fixed holdout.
Prediction Lab includes historical learning counts and a SQLite backup download.
Live prediction outcome refresh and tables support 5, 10, 20, 30 and 60 sessions.

PRESERVED / INTEGRATED
Phase 5 VPA calculations, VPA Scanner and missing-VPA repair, same-prediction VPA
comparison, automatic/incremental scanner, Early-Mover Tracker, watchlist,
prediction saving, investment screens, single-share analysis, portfolio planner,
CSV exports and the database filename remain present.

The scoring engine retains Phase 5's twelve inputs, then adds Phase 4's daily
range, close position, opening gap and volatility inputs. Base weights are
normalized together. Available inputs are renormalized when others are missing.
Historical VPA uses only bars through each signal date. Completed historical
monthly examples and live outcomes feed the same adaptive learner; about one
example per ticker/month limits overlap. Existing feature JSON is read by name;
old Phase 4 component positions are not mistaken for saved VPA data.

USE
Open Phase 4 Backtest, choose your universe and batch size, then select Run one
batch or Run all remaining batches automatically. Progress is saved after each
batch. If interrupted, reopen the same universe/spacing and resume. Failed codes
count as attempted and appear in the unavailable list. Restart retries a run.
Old historical records remain usable. Restarting an older completed Phase 4 run
recomputes its history with VPA; changing the app does not silently erase it.
Changing the score threshold updates diagnostics without another download.

TESTING
See TEST_REPORT.txt for passed offline checks and the remaining live-test limit.

LIMITS
The bundled current-company list is inherited from Phase 4, not refreshed here.
History depends on Yahoo availability and omits historical delisted membership.
Diagnostics use the original representative display sample; full stored monthly
examples feed learning. Holdout retains the Phase 4 time split and 45-calendar-day
gap. These checks do not establish profitability. Automatic scans can be stopped
by Streamlit timeouts or Yahoo throttling; resume from the last completed batch.

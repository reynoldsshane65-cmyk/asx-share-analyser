ASX Share Analyser - Phase 6 (VPA + sector-aware accuracy)

INSTALL
1. Before replacing app.py, open the currently running app and use Prediction Lab > Prepare SQLite database backup > Download SQLite backup. Save it somewhere safe.
2. Back up your current GitHub app.py.
3. Replace GitHub app.py with the app.py in this ZIP. Keep your existing ASX_Listed_Companies.csv and other files.
4. Let Streamlit Cloud redeploy.
5. Open Sector Prediction Accuracy to compare completed saved signals with the ASX 200 and sector conditions at signal dates.
6. In Prediction Lab, use Restore prediction database if a previous SQLite backup needs to be restored. Restoration REPLACES all current local database contents.

ADDITIONS
- Sector basket classification in Early-Mover Tracker (only representative constituent stocks)
- Sector Prediction Accuracy page, showing benchmark-relative forward outcomes at 5/10/20/30/60 trading sessions and sector leadership based on prior 63 sessions
- SQLite backup restoration, with integrity check and explicit confirmation

LIMITATIONS
- Sector baskets are illustrative, NOT full GICS/ASX sector classifications.
- Analysis uses the last 2 years of Yahoo price history and only completed, previously saved predictions.
- Streamlit Community Cloud local SQLite storage is ephemeral; download backups regularly. Backup/restore is NOT persistent cloud storage.
- Benchmark is the ASX 200 price index, while individual prices use adjusted prices. Returns are not perfectly like-for-like.
- No live integration tests were performed; Python syntax was checked.
- No automated trading; not financial advice.

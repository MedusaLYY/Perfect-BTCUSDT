# Data Artifacts

Large data files are intentionally excluded from Git.

Expected local paths for full reproduction include:

- `data/raw/binance/BTCUSDT_spot_klines_1m.csv`
- `data/raw/binance/BTCUSDT_spot_agg_trades.csv`
- `data/model/BTCUSDT_60m_direction_dataset_v1.parquet`
- `data/splits/BTCUSDT_60m_split_assignments_v1.parquet`
- `data/predictions/BTCUSDT_stage7_cv_xgboost_fixed_predictions.parquet`
- `data/predictions/BTCUSDT_stage6_cv_baseline_predictions.parquet`

Share these through an artifact store or dataset release, not normal Git commits.

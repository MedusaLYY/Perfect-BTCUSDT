# BTCUSDT 60m Direction Model

This repository contains a reproducible research pipeline for a BTCUSDT 60-minute direction classification model.

The current frozen final model is included under `models/final/`.

## Current Status

- Final stage completed: Stage 9
- Selected development config: `xgb_fixed_v1_reference`
- Final calibration method: `UNCALIBRATED`
- Final XGBoost trees: `57`
- FINAL_TEST base model `predict_proba` calls: `1`
- Final assessment: `READY_FOR_RESEARCH_DEPLOYMENT`

Important: this project is for research, coursework, demos, and paper trading signals only. It is not a live trading system and does not guarantee profitability.

## Included

- Pipeline scripts in `scripts/`
- Reproducibility configs in `config/`
- Unit/regression tests in `tests/`
- Final model artifacts in `models/final/`
- Stage 9 final reports and manifests in `reports/`
- Project operating rules in `AGENT.md`

## Not Included

Large local data files are intentionally excluded from Git:

- raw Binance CSV files
- processed/interim parquet files
- full model training dataset parquet
- generated prediction parquet files
- historical stage model folders

These files are large and should be shared through object storage, a dataset release, or an internal artifact store instead of normal Git history.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verify

Stage 9 unit tests:

```powershell
python -m pytest tests/test_stage9_final_and_evaluate.py -q
```

Full regression suite:

```powershell
python -m pytest tests/test_stage0_data_audit.py tests/test_stage1_clean_data.py tests/test_stage2a_kline_model_base.py tests/test_stage2b_agg_kline_analysis.py tests/test_stage3a_label_proxy.py tests/test_stage3b_full_labels.py tests/test_stage4_features.py tests/test_stage5_dataset_and_splits.py tests/test_stage6_baselines.py tests/test_stage7_xgboost_fixed.py tests/test_stage8_limited_xgboost_tuning.py tests/test_stage9_final_and_evaluate.py -q
```

## Final Model Files

- `models/final/btcusdt_xgboost_direction_v1.json`
- `models/final/btcusdt_probability_calibrator_v1.json`
- `models/final/btcusdt_logistic_baseline_v1.joblib`
- `models/final/btcusdt_prior_baseline_v1.json`
- `models/final/btcusdt_momentum_60m_baseline_v1.json`
- `models/final/inference_manifest.json`

The inference manifest records ordered features, expected dtypes, model hash, calibration method, model limitations, and runtime versions.

## Collaboration Notes

Use branches and pull requests for changes. Do not commit raw data, regenerated large parquet files, or experimental model folders unless the team explicitly decides to use Git LFS or release assets.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_stage9_final_and_evaluate import (  # noqa: E402
    BASE_DATASET_COLUMNS,
    BASE_SPLIT_COLUMNS,
    CALIBRATED_MODEL_NAME,
    Stage9ValidationError,
    evaluate_final_predictions,
    final_refit_training_frame,
    input_hashes,
    joined_dataset,
    prepare_feature_matrix,
    read_development_dataset,
    read_json,
    resolve_paths,
    select_calibration_method,
    sha256_file,
    validate_calibration_candidates,
    validate_feature_manifest,
    validate_final_test_predictions,
    verify_stage8_selection,
)


class Stage9OutputValidationError(ValueError):
    pass


def validate_stage9_outputs(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    paths = resolve_paths(config, root)
    report_paths = paths["report_paths"]
    errors: list[str] = []

    required_reports = [
        "calibration_selection_report",
        "calibration_metrics",
        "calibration_forward_predictions",
        "final_training_report",
        "final_test_report",
        "final_test_metrics",
        "final_test_metrics_by_offset",
        "final_test_metrics_by_margin",
        "final_protocol_lock",
        "final_model_manifest",
    ]
    for key in required_reports:
        if not report_paths[key].exists():
            errors.append(f"missing report output: {key}")

    if errors:
        raise Stage9OutputValidationError("\n".join(errors))

    dataset_manifest = read_json(paths["dataset_manifest_path"])
    feature_manifest = read_json(paths["feature_manifest_path"])
    stage8_selection = read_json(paths["stage8_selection_audit_path"])
    protocol = read_json(report_paths["final_protocol_lock"])
    manifest = read_json(report_paths["final_model_manifest"])
    inference_manifest = read_json(paths["output_model_dir"] / "inference_manifest.json")
    calibration_metrics = pd.read_csv(report_paths["calibration_metrics"])
    forward_predictions = pd.read_parquet(report_paths["calibration_forward_predictions"])
    final_predictions = pd.read_parquet(paths["final_prediction_path"])
    final_metrics = pd.read_csv(report_paths["final_test_metrics"])

    try:
        verify_stage8_selection(stage8_selection)
    except Stage9ValidationError as exc:
        errors.extend(exc.errors)
    try:
        validate_calibration_candidates(config["calibration_candidates"])
    except Stage9ValidationError as exc:
        errors.extend(exc.errors)
    try:
        feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest)
    except Stage9ValidationError as exc:
        errors.extend(exc.errors)
        feature_columns = list(dataset_manifest.get("feature_columns", []))

    replay = select_calibration_method(calibration_metrics, config["calibration_selection_rules"])
    if replay["selected_calibration_method"] != protocol.get("selected_calibration_method"):
        errors.append("calibration selection is not reproducible from metrics")
    if replay["selected_calibration_method"] != manifest.get("selected_calibration_method"):
        errors.append("manifest calibration method mismatch")

    expected_fit = {
        "calibration_2021": "fold_2020",
        "calibration_2022": "fold_2020,fold_2021",
        "calibration_2023": "fold_2020,fold_2021,fold_2022",
        "calibration_2024": "fold_2020,fold_2021,fold_2022,fold_2023",
    }
    for fold_name, fit_folds in expected_fit.items():
        rows = forward_predictions[forward_predictions["calibration_fold"].eq(fold_name)]
        if rows.empty:
            errors.append(f"missing forward calibration predictions for {fold_name}")
        elif set(rows["fit_folds"]) != {fit_folds}:
            errors.append(f"{fold_name} fit folds are not forward-only")
    if "fold_2024" in set(forward_predictions.loc[forward_predictions["calibration_fold"].eq("calibration_2024"), "fit_folds"].astype(str).str.split(",").explode()):
        errors.append("calibration_2024 used fold_2024 for fit")

    tree_meta = manifest.get("final_tree_selection", {})
    if tree_meta.get("final_test_used_for_tree_count_selection") is not False:
        errors.append("final tree count selection used FINAL_TEST")
    final_model = manifest.get("final_model", {})
    if final_model.get("training_role") != "DEVELOPMENT" or final_model.get("final_test_rows_used"):
        errors.append("final model was not trained strictly on DEVELOPMENT")
    if protocol.get("final_test_accessed") is not False:
        errors.append("protocol lock must have final_test_accessed=false")
    timestamps = manifest.get("stage_timestamps", {})
    if timestamps.get("protocol_locked_at_utc") != protocol.get("protocol_locked_at_utc"):
        errors.append("manifest/protocol lock timestamp mismatch")
    if timestamps.get("final_test_feature_matrix_created_at_utc") and protocol.get("protocol_locked_at_utc"):
        if pd.Timestamp(timestamps["final_test_feature_matrix_created_at_utc"]) <= pd.Timestamp(protocol["protocol_locked_at_utc"]):
            errors.append("FINAL_TEST feature matrix timestamp is not after protocol lock")

    split_columns = BASE_SPLIT_COLUMNS
    splits = pd.read_parquet(paths["split_path"], columns=split_columns)
    expected_final_count = int(splits["final_split_role"].eq("FINAL_TEST").sum())
    try:
        validate_final_test_predictions(final_predictions, expected_final_count)
    except Stage9ValidationError as exc:
        errors.extend(exc.errors)
    if int(manifest.get("final_test_base_model_predict_proba_call_count", 0)) != 1:
        errors.append("FINAL_TEST base model predict_proba call count is not one")
    if manifest.get("quality_gates", {}).get("final_test_predict_proba_call_count_is_one") is not True:
        errors.append("predict_proba call count gate failed")

    recomputed = evaluate_final_predictions(final_predictions, int(config["calibration_bins"]))
    key_cols = ["model_name", "subset_name", "sample_count", "roc_auc", "average_precision", "log_loss", "brier_score", "mcc"]
    expected = recomputed[key_cols].sort_values(["model_name", "subset_name"], kind="mergesort").reset_index(drop=True)
    actual = final_metrics[key_cols].sort_values(["model_name", "subset_name"], kind="mergesort").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(expected, actual, check_exact=False, atol=1e-12)
    except AssertionError as exc:
        errors.append(f"final metrics do not recompute: {exc}")

    required_models = {"prior_baseline_final", "momentum_60m_baseline_final", "logistic_regression_final", "xgboost_final_raw", CALIBRATED_MODEL_NAME}
    required_subsets = {"DENSE", "NONOVERLAP_OFFSET_00", *{f"OFFSET_{i:02d}" for i in range(0, 60, 5)}, "ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"}
    if not required_models.issubset(set(final_metrics["model_name"])):
        errors.append("final metrics missing required models")
    if not required_subsets.issubset(set(final_metrics["subset_name"])):
        errors.append("final metrics missing required subsets")

    if protocol.get("ordered_feature_names") != feature_columns:
        errors.append("protocol feature order mismatch")
    if inference_manifest.get("ordered_feature_names") != feature_columns:
        errors.append("inference manifest feature order mismatch")
    for field in ["model_name", "model_version", "model_file", "model_sha256", "calibration_method", "calibrator_file", "calibrator_sha256", "fixed_classification_threshold", "model_limitations"]:
        if field not in inference_manifest:
            errors.append(f"inference manifest missing {field}")

    # Reload final model and verify it can score DEVELOPMENT sample only. This intentionally avoids FINAL_TEST.
    dev_dataset = read_development_dataset(paths["dataset_path"], splits, feature_columns)
    dev_joined = joined_dataset(dev_dataset, splits[splits["final_split_role"].ne("FINAL_TEST")].copy())
    train = final_refit_training_frame(dev_joined)
    sample = train.head(min(512, len(train))).copy()
    X_dev = prepare_feature_matrix(sample, feature_columns).to_numpy(dtype=np.float32, copy=True)
    model = xgb.XGBClassifier()
    model.load_model(paths["output_model_dir"] / "btcusdt_xgboost_direction_v1.json")
    p_dev = model.predict_proba(X_dev)[:, 1]
    if not np.isfinite(p_dev).all() or not ((p_dev >= 0.0) & (p_dev <= 1.0)).all():
        errors.append("reloaded final model produced invalid DEVELOPMENT probabilities")

    current_input_hashes = input_hashes(paths, config_path)
    for key, value in manifest.get("input_hashes", {}).items():
        if key in current_input_hashes and current_input_hashes[key] != value:
            errors.append(f"input hash mismatch for {key}")

    model_path = paths["output_model_dir"] / "btcusdt_xgboost_direction_v1.json"
    calibrator_path = Path(protocol["calibrator_file"])
    logistic_path = paths["output_model_dir"] / "btcusdt_logistic_baseline_v1.joblib"
    for path in [model_path, calibrator_path, logistic_path, paths["final_prediction_path"], paths["output_model_dir"] / "inference_manifest.json"]:
        if not path.exists():
            errors.append(f"missing artifact {path}")
    if model_path.exists() and protocol.get("xgboost_model_sha256") != sha256_file(model_path):
        errors.append("XGBoost model SHA-256 mismatch")
    if calibrator_path.exists() and protocol.get("calibrator_sha256") != sha256_file(calibrator_path):
        errors.append("calibrator SHA-256 mismatch")
    if paths["final_prediction_path"].exists() and manifest.get("final_prediction_sha256") != sha256_file(paths["final_prediction_path"]):
        errors.append("final prediction SHA-256 mismatch")

    gates = manifest.get("quality_gates", {})
    if gates.get("stage9_engineering_gate_passed") is not True:
        errors.append("stage9_engineering_gate_passed is not true")

    if errors:
        raise Stage9OutputValidationError("\n".join(errors))
    return {
        "validated": True,
        "selected_calibration_method": protocol.get("selected_calibration_method"),
        "final_test_prediction_count": int(len(final_predictions)),
        "final_test_base_model_predict_proba_call_count": int(manifest.get("final_test_base_model_predict_proba_call_count")),
        "final_model_assessment": manifest.get("final_model_assessment"),
        "protocol_lock_sha256": sha256_file(report_paths["final_protocol_lock"]),
        "final_prediction_sha256": sha256_file(paths["final_prediction_path"]),
        "final_model_manifest_sha256": sha256_file(report_paths["final_model_manifest"]),
        "inference_manifest_sha256": sha256_file(paths["output_model_dir"] / "inference_manifest.json"),
        "development_reload_sample_count": int(len(sample)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Stage9 final model outputs.")
    parser.add_argument("--config", default="config/stage9_train_final_and_evaluate.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = validate_stage9_outputs((root / args.config).resolve(), root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

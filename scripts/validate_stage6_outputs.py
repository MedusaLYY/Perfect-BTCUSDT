from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class Stage6OutputValidationError(ValueError):
    pass


ALLOWED_MODELS = {"prior_baseline", "momentum_60m_baseline", "logistic_regression_l2"}
TARGET_COLUMN = "label_up_60m"
BASE_DATASET_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "settlement_minute_open_time",
    TARGET_COLUMN,
    "absolute_future_return_bps",
]
BASE_SPLIT_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "settlement_minute_open_time",
    "final_split_role",
    "evaluation_offset_minutes",
    "is_primary_nonoverlap_evaluation",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage6_outputs(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    dataset_manifest = read_json((root / config["dataset_manifest_path"]).resolve())
    feature_manifest = read_json((root / config["feature_manifest_path"]).resolve())
    model_manifest_path = (root / config["report_paths"]["model_manifest"]).resolve()
    model_manifest = read_json(model_manifest_path)
    feature_columns = list(dataset_manifest["feature_columns"])
    if feature_columns != list(feature_manifest["ordered_feature_names"]) or len(feature_columns) != 63:
        raise Stage6OutputValidationError("63 feature order does not match manifests")

    prediction_path = (root / config["prediction_output_path"]).resolve()
    metrics_subset_path = (root / config["report_paths"]["metrics_by_subset"]).resolve()
    oof_path = (root / config["report_paths"]["oof_summary"]).resolve()
    preprocessing_audit_path = (root / config["report_paths"]["preprocessing_audit"]).resolve()
    dataset_path = (root / config["dataset_path"]).resolve()
    split_path = (root / config["split_path"]).resolve()
    predictions = pd.read_parquet(prediction_path)
    metrics = pd.read_csv(metrics_subset_path)
    oof = pd.read_csv(oof_path)
    preprocessing_audit = read_json(preprocessing_audit_path)
    split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold}_role" for fold in config["fold_names"]]]
    dataset_columns = [*BASE_DATASET_COLUMNS, *feature_columns]
    splits = pd.read_parquet(split_path, columns=split_columns)
    dataset = pd.read_parquet(dataset_path, columns=dataset_columns)
    merged = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"], how="inner", validate="one_to_one")

    errors: list[str] = []
    if set(predictions["model_name"].unique()) != ALLOWED_MODELS:
        errors.append("unexpected model names in predictions")
    if not np.isfinite(predictions["p_up"].to_numpy(dtype=np.float64)).all():
        errors.append("non-finite probabilities")
    if not predictions["p_up"].between(0.0, 1.0).all():
        errors.append("probabilities outside [0, 1]")
    logistic_thresholds = predictions.loc[predictions["model_name"].eq("logistic_regression_l2"), "prediction_threshold"].dropna().unique()
    if len(logistic_thresholds) != 1 or float(logistic_thresholds[0]) != float(config["fixed_prediction_threshold"]):
        errors.append("logistic fixed threshold mismatch")
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"])
    if predictions["dataset_row_id"].isin(final_ids).any():
        errors.append("FINAL_TEST predictions found")

    expected_total = 0
    for fold_name in config["fold_names"]:
        role_col = f"{fold_name}_role"
        valid_ids = set(splits.loc[splits[role_col].eq("VALIDATION"), "dataset_row_id"])
        train = merged.loc[merged[role_col].eq("TRAIN")]
        valid = merged.loc[merged[role_col].eq("VALIDATION")]
        expected_total += len(valid_ids) * len(ALLOWED_MODELS)
        fold_predictions = predictions[predictions["fold_name"].eq(fold_name)]
        if set(fold_predictions["dataset_row_id"].unique()) != valid_ids:
            errors.append(f"{fold_name} prediction ids do not match validation ids")
        if not fold_predictions.groupby(["model_name", "dataset_row_id"]).size().eq(1).all():
            errors.append(f"{fold_name} duplicate or missing model predictions")
        if int(train["settlement_minute_open_time"].max()) >= int(valid["decision_time"].min()):
            errors.append(f"{fold_name} time isolation failed")

        model_path = (root / config["model_output_dir"] / fold_name / "logistic_regression_pipeline.joblib").resolve()
        pipeline = joblib.load(model_path)
        scaler = pipeline.named_steps["scaler"]
        X_train = train[feature_columns].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(scaler.mean_, X_train.mean(axis=0), atol=float(config["numeric_tolerances"]["scaler_atol"]))
        np.testing.assert_allclose(scaler.var_, X_train.var(axis=0, ddof=0), atol=float(config["numeric_tolerances"]["scaler_atol"]))
        X_valid = valid[feature_columns].to_numpy(dtype=np.float64)
        reloaded_p = pipeline.predict_proba(X_valid)[:, 1]
        saved_p = (
            predictions[(predictions["fold_name"].eq(fold_name)) & (predictions["model_name"].eq("logistic_regression_l2"))]
            .sort_values(["decision_time", "dataset_row_id"], kind="mergesort")["p_up"]
            .to_numpy(dtype=np.float64)
        )
        np.testing.assert_allclose(reloaded_p, saved_p, atol=float(config["numeric_tolerances"]["prediction_atol"]))
        audit = preprocessing_audit["folds"][fold_name]
        if int(audit["scaler_n_samples_seen"]) != len(train):
            errors.append(f"{fold_name} scaler n_samples_seen mismatch")

    if len(predictions) != expected_total:
        errors.append("total prediction count mismatch")
    required_subsets = {"DENSE", "NONOVERLAP_OFFSET_00", *{f"OFFSET_{i:02d}" for i in range(0, 60, 5)}}
    if not required_subsets.issubset(set(metrics["subset_name"])):
        errors.append("metrics missing dense/non-overlap/offset subsets")
    pooled_dense = oof[(oof["summary_type"].eq("pooled")) & (oof["subset_name"].eq("DENSE"))]
    if set(pooled_dense["model_name"]) != ALLOWED_MODELS:
        errors.append("pooled dense OOF rows missing models")
    if model_manifest["feature_columns"] != feature_columns:
        errors.append("model manifest feature order mismatch")
    if model_manifest["quality_gates"]["final_test_prediction_count"] != 0:
        errors.append("model manifest final_test_prediction_count is not zero")
    actual_prediction_hash = sha256_file(prediction_path)
    manifest_prediction_hash = model_manifest["output_files"].get("prediction_output_sha256")
    if manifest_prediction_hash and actual_prediction_hash != manifest_prediction_hash:
        errors.append("prediction output hash mismatch")
    if errors:
        raise Stage6OutputValidationError("\n".join(errors))
    return {
        "prediction_count": int(len(predictions)),
        "prediction_sha256": actual_prediction_hash,
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "metrics_by_subset_rows": int(len(metrics)),
        "oof_summary_rows": int(len(oof)),
        "final_test_prediction_count": 0,
        "validated_folds": config["fold_names"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Stage 6 baseline outputs.")
    parser.add_argument("--config", default="config/stage6_train_baselines.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = validate_stage6_outputs((root / args.config).resolve(), root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

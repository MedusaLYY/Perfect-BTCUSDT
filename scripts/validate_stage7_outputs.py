from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from scripts.train_stage7_xgboost_fixed import (
    BASE_SPLIT_COLUMNS,
    MODEL_NAME,
    Stage7ValidationError,
    compare_with_stage6,
    read_stage7_dataset,
    validate_feature_manifest,
)


class Stage7OutputValidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stage7_outputs(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    dataset_manifest = read_json((root / config["dataset_manifest_path"]).resolve())
    feature_manifest = read_json((root / config["feature_manifest_path"]).resolve())
    fold_manifest = read_json((root / config["fold_manifest_path"]).resolve())
    model_manifest_path = (root / config["report_paths"]["model_manifest"]).resolve()
    model_manifest = read_json(model_manifest_path)
    feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest, int(config["feature_count"]))
    if model_manifest["feature_columns"] != feature_columns:
        raise Stage7OutputValidationError("model manifest feature order mismatch")

    split_path = (root / config["split_path"]).resolve()
    dataset_path = (root / config["dataset_path"]).resolve()
    prediction_path = (root / config["prediction_output_path"]).resolve()
    stage6_path = (root / config["stage6_prediction_path"]).resolve()
    comparison_path = (root / config["report_paths"]["model_comparison"]).resolve()
    metrics_path = (root / config["report_paths"]["metrics_by_subset"]).resolve()
    oof_path = (root / config["report_paths"]["oof_summary"]).resolve()
    split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold}_role" for fold in config["fold_names"]]]
    splits = pd.read_parquet(split_path, columns=split_columns)
    dataset = read_stage7_dataset(dataset_path, splits, feature_columns, config["fold_names"])
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"], validate="one_to_one")
    predictions = pd.read_parquet(prediction_path)
    stage6_predictions = pd.read_parquet(stage6_path)
    comparison = pd.read_csv(comparison_path)
    metrics = pd.read_csv(metrics_path)
    oof = pd.read_csv(oof_path)
    errors: list[str] = []

    if set(predictions["model_name"].unique()) != {MODEL_NAME}:
        errors.append("unexpected model_name in Stage7 predictions")
    if not np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all() or not predictions["p_up"].between(0, 1).all():
        errors.append("invalid prediction probabilities")
    if not (predictions["prediction_threshold"] == float(config["fixed_prediction_threshold"])).all():
        errors.append("prediction threshold mismatch")
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"])
    if set(predictions["dataset_row_id"]).intersection(final_ids):
        errors.append("FINAL_TEST predictions detected")
    if int(model_manifest["quality_gates"]["final_test_prediction_count"]) != 0:
        errors.append("manifest final_test_prediction_count is not zero")

    expected_total = 0
    for fold in config["fold_names"]:
        role_col = f"{fold}_role"
        valid = joined[joined[role_col].eq("VALIDATION")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort")
        train = joined[joined[role_col].eq("TRAIN")]
        expected_total += len(valid)
        fold_predictions = predictions[predictions["fold_name"].eq(fold)].sort_values(["decision_time", "dataset_row_id"], kind="mergesort")
        if set(fold_predictions["dataset_row_id"]) != set(valid["dataset_row_id"]):
            errors.append(f"{fold} prediction ids mismatch")
        if set(fold_predictions["dataset_row_id"]).intersection(set(train["dataset_row_id"])):
            errors.append(f"{fold} TRAIN predictions detected")
        if int(train["settlement_minute_open_time"].max()) >= int(valid["decision_time"].min()):
            errors.append(f"{fold} outer time isolation failed")
        fold_meta = model_manifest["folds"][fold]
        inner = fold_meta["inner_split_audit"]
        if int(inner["inner_fit_max_settlement_minute_open_time"]) >= int(inner["inner_early_stop_min_decision_time"]):
            errors.append(f"{fold} inner time isolation failed")
        if fold_meta["selector_metadata"]["outer_validation_used_for_early_stopping"]:
            errors.append(f"{fold} outer validation used for early stopping")
        best_n = int(fold_meta["selector_metadata"]["best_n_estimators"])
        if best_n != int(fold_meta["selector_metadata"]["best_iteration"]) + 1:
            errors.append(f"{fold} best_n_estimators mismatch")
        model_path = (root / config["model_output_dir"] / fold / f"{MODEL_NAME}.json").resolve()
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        if model.get_booster().num_boosted_rounds() != best_n:
            errors.append(f"{fold} refit model tree count mismatch")
        refit_params = fold_meta["refit_metadata"]["refit_parameters"]
        for key in ["objective", "booster", "tree_method", "device", "max_depth", "min_child_weight", "scale_pos_weight", "learning_rate", "reg_alpha", "reg_lambda"]:
            if refit_params.get(key) != config["xgboost_parameters"].get(key):
                errors.append(f"{fold} recorded parameter {key} mismatch")
        model_payload = read_json(model_path)
        learner = model_payload["learner"]
        if learner["objective"]["name"] != "binary:logistic":
            errors.append(f"{fold} objective mismatch in model file")
        if float(learner["objective"]["reg_loss_param"]["scale_pos_weight"]) != 1.0:
            errors.append(f"{fold} scale_pos_weight mismatch in model file")
        if int(learner["learner_model_param"]["num_feature"]) != 63:
            errors.append(f"{fold} model feature count mismatch")
        max_nodes = (2 ** (int(config["xgboost_parameters"]["max_depth"]) + 1)) - 1
        for tree in learner["gradient_booster"]["model"]["trees"]:
            if int(tree["tree_param"]["num_nodes"]) > max_nodes:
                errors.append(f"{fold} tree exceeds configured max_depth structure")
                break
        reloaded = model.predict_proba(valid[feature_columns].to_numpy(dtype=np.float32))[:, 1]
        np.testing.assert_allclose(reloaded, fold_predictions["p_up"].to_numpy(dtype=float), atol=float(config["numeric_tolerances"]["prediction_atol"]))

    if len(predictions) != expected_total:
        errors.append("prediction count mismatch")
    try:
        recomputed = compare_with_stage6(predictions, stage6_predictions)
        key_cols = ["fold_name", "subset_name", "baseline_model_name", "delta_roc_auc", "delta_log_loss", "delta_brier"]
        expected = recomputed[key_cols].sort_values(["fold_name", "subset_name", "baseline_model_name"], kind="mergesort").reset_index(drop=True)
        actual = comparison[key_cols].sort_values(["fold_name", "subset_name", "baseline_model_name"], kind="mergesort").reset_index(drop=True)
        pd.testing.assert_frame_equal(expected, actual, check_exact=False, atol=1e-12)
    except Stage7ValidationError as exc:
        errors.extend(exc.errors)
    required_subsets = {"DENSE", "NONOVERLAP_OFFSET_00", *{f"OFFSET_{i:02d}" for i in range(0, 60, 5)}}
    if not required_subsets.issubset(set(metrics["subset_name"])):
        errors.append("metrics missing required subsets")
    if set(oof[oof["summary_type"].eq("pooled")]["model_name"].unique()) != {MODEL_NAME}:
        errors.append("pooled OOF missing XGBoost model")
    actual_prediction_hash = sha256_file(prediction_path)
    manifest_hash = model_manifest["output_files"].get("prediction_output_sha256")
    if manifest_hash != actual_prediction_hash:
        errors.append("prediction SHA-256 mismatch")
    if errors:
        raise Stage7OutputValidationError("\n".join(errors))
    return {
        "prediction_count": int(len(predictions)),
        "prediction_sha256": actual_prediction_hash,
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "metrics_by_subset_rows": int(len(metrics)),
        "oof_summary_rows": int(len(oof)),
        "comparison_rows": int(len(comparison)),
        "final_test_prediction_count": 0,
        "validated_folds": config["fold_names"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Stage 7 fixed XGBoost outputs.")
    parser.add_argument("--config", default="config/stage7_train_xgboost_fixed.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = validate_stage7_outputs((root / args.config).resolve(), root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

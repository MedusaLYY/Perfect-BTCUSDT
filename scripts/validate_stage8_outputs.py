from __future__ import annotations

import argparse
import hashlib
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

from scripts.train_stage8_limited_xgboost_tuning import (  # noqa: E402
    CANDIDATE_NAMES,
    REFERENCE_CANDIDATE_NAME,
    BASE_SPLIT_COLUMNS,
    candidate_definitions_sha256,
    compare_candidate_to_stage7_reference,
    freeze_candidate_definitions,
    read_stage7_dataset,
    select_development_config,
    validate_feature_manifest,
    validate_stage8_predictions,
)


class Stage8OutputValidationError(ValueError):
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


def validate_stage8_outputs(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    report_paths = {name: (root / path).resolve() for name, path in config["report_paths"].items()}
    manifest = read_json(report_paths["model_manifest"])
    selection_audit = read_json(report_paths["selection_audit"])
    dataset_manifest = read_json((root / config["dataset_manifest_path"]).resolve())
    feature_manifest = read_json((root / config["feature_manifest_path"]).resolve())
    stage7_manifest = read_json((root / config["stage7_model_manifest_path"]).resolve())
    feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest, int(config["feature_count"]))
    split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold}_role" for fold in config["fold_names"]]]
    splits = pd.read_parquet((root / config["split_path"]).resolve(), columns=split_columns)
    dataset = read_stage7_dataset((root / config["dataset_path"]).resolve(), splits, feature_columns, config["fold_names"])
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"], validate="one_to_one")
    predictions = pd.read_parquet((root / config["prediction_output_path"]).resolve())
    stage7_predictions = pd.read_parquet((root / config["stage7_prediction_path"]).resolve())
    candidate_summary = pd.read_csv(report_paths["candidate_summary"])
    candidate_comparison = pd.read_csv(report_paths["candidate_comparison"])
    metrics_by_subset = pd.read_csv(report_paths["metrics_by_candidate_subset"])
    metrics_by_offset = pd.read_csv(report_paths["metrics_by_candidate_offset"])
    errors: list[str] = []

    frozen = freeze_candidate_definitions(config["candidate_definitions"])
    candidate_hash = candidate_definitions_sha256(frozen)
    if manifest.get("candidate_definitions_sha256") != candidate_hash:
        errors.append("candidate hash mismatch in manifest")
    if selection_audit.get("candidate_definitions_sha256") != candidate_hash:
        errors.append("candidate hash mismatch in selection audit")
    if manifest.get("candidate_count") != 6 or [c["model_name"] for c in manifest.get("candidate_definitions", [])] != CANDIDATE_NAMES:
        errors.append("candidate definitions are not the six expected candidates")
    if manifest.get("feature_columns") != feature_columns:
        errors.append("manifest feature order mismatch")
    try:
        validate_stage8_predictions(predictions, splits, config["fold_names"], CANDIDATE_NAMES)
    except Exception as exc:
        errors.append(str(exc))
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"])
    if set(predictions["dataset_row_id"]).intersection(final_ids):
        errors.append("FINAL_TEST predictions detected")
    if not np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all() or not predictions["p_up"].between(0, 1).all():
        errors.append("invalid probabilities")
    if not (predictions["prediction_threshold"] == float(config["fixed_prediction_threshold"])).all():
        errors.append("threshold is not fixed")

    reference_check = compare_candidate_to_stage7_reference(predictions, stage7_predictions, config["reference_reproduction_tolerances"])
    if not reference_check["reference_config_reproduced"]:
        errors.append("Candidate 1 did not reproduce Stage7 predictions")

    expected_total = 0
    for fold in config["fold_names"]:
        valid = joined[joined[f"{fold}_role"].eq("VALIDATION")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort")
        train = joined[joined[f"{fold}_role"].eq("TRAIN")]
        expected_total += len(valid)
        if int(train["settlement_minute_open_time"].max()) >= int(valid["decision_time"].min()):
            errors.append(f"{fold} outer time isolation failed")
        for candidate in CANDIDATE_NAMES:
            model_path = (root / config["model_output_dir"] / candidate / fold / "refit_model.json").resolve()
            if not model_path.exists():
                errors.append(f"missing model {candidate}/{fold}")
                continue
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            fold_predictions = predictions[predictions["model_name"].eq(candidate) & predictions["fold_name"].eq(fold)].sort_values(["decision_time", "dataset_row_id"], kind="mergesort")
            sample = valid[feature_columns].to_numpy(dtype=np.float32)[: min(1000, len(valid))]
            reloaded = model.predict_proba(sample)[:, 1]
            np.testing.assert_allclose(reloaded, fold_predictions["p_up"].to_numpy(dtype=float)[: len(reloaded)], atol=float(config["numeric_tolerances"]["prediction_atol"]))
            meta = read_json((root / config["model_output_dir"] / candidate / fold / "fold_metadata.json").resolve())
            if meta["selector_metadata"]["best_n_estimators"] != meta["selector_metadata"]["best_iteration"] + 1:
                errors.append(f"{candidate}/{fold} best_n_estimators mismatch")
            if meta["selector_metadata"]["outer_validation_used_for_early_stopping"]:
                errors.append(f"{candidate}/{fold} outer validation used for early stopping")
            if meta["refit_metadata"]["refit_used_early_stopping"]:
                errors.append(f"{candidate}/{fold} refit used early stopping")
            if meta["refit_metadata"]["scale_pos_weight"] != 1.0:
                errors.append(f"{candidate}/{fold} scale_pos_weight changed")

    if len(predictions) != expected_total * 6:
        errors.append("prediction count mismatch")
    required_subsets = {"DENSE", "NONOVERLAP_OFFSET_00", *{f"OFFSET_{i:02d}" for i in range(0, 60, 5)}}
    if not required_subsets.issubset(set(metrics_by_subset["subset_name"])):
        errors.append("metrics missing required subsets")
    if not required_subsets.intersection(set(metrics_by_offset["subset_name"])):
        errors.append("offset metrics missing")
    replay = select_development_config(
        candidate_summary,
        candidate_comparison[candidate_comparison["comparison_type"].eq("vs_candidate1_reference")],
        config["selection_rules"],
        bool(selection_audit["reference_reproduction"]["reference_config_reproduced"]),
        bool(manifest["quality_gates"]["stage8_engineering_gate_passed"]),
    )
    if replay["selected_development_config"] != selection_audit["selected_development_config"]:
        errors.append("selection audit is not reproducible")
    actual_prediction_hash = sha256_file((root / config["prediction_output_path"]).resolve())
    if manifest.get("output_files", {}).get("prediction_output_sha256") != actual_prediction_hash:
        errors.append("prediction SHA-256 mismatch")
    if errors:
        raise Stage8OutputValidationError("\n".join(errors))
    return {
        "prediction_count": int(len(predictions)),
        "prediction_sha256": actual_prediction_hash,
        "candidate_definitions_sha256": candidate_hash,
        "model_manifest_sha256": sha256_file(report_paths["model_manifest"]),
        "selection_audit_sha256": sha256_file(report_paths["selection_audit"]),
        "validated_candidates": CANDIDATE_NAMES,
        "validated_folds": config["fold_names"],
        "selected_development_config": selection_audit["selected_development_config"],
        "final_test_prediction_count": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently validate Stage 8 limited XGBoost tuning outputs.")
    parser.add_argument("--config", default="config/stage8_limited_xgboost_tuning.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = validate_stage8_outputs((root / args.config).resolve(), root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

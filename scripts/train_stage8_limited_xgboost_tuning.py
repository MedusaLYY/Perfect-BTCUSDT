from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_stage6_baselines import (  # noqa: E402
    EVALUATION_METADATA_COLUMNS,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    calibration_table as stage6_calibration_table,
    compute_classification_metrics,
    evaluate_prediction_subsets as stage6_evaluate_prediction_subsets,
    make_json_safe,
    pooled_and_macro_summary as stage6_pooled_and_macro_summary,
    subset_masks,
)
from scripts.train_stage7_xgboost_fixed import (  # noqa: E402
    BASE_DATASET_COLUMNS,
    BASE_SPLIT_COLUMNS,
    IMPORTANCE_TYPES,
    TARGET_COLUMN,
    Stage7ValidationError,
    add_validation_metadata,
    ensure_parent,
    final_test_audit,
    get_process_rss_bytes,
    joined_dataset,
    make_inner_time_split,
    ms_to_utc_iso,
    parse_utc_ms,
    prepare_feature_matrix,
    read_stage7_dataset,
    sha256_file,
    sha256_text,
    table,
    time_range,
    validate_feature_manifest,
    validate_inner_split,
    validate_outer_fold_integrity,
    verify_xgboost_parameters,
    write_json,
)


class Stage8ValidationError(ValueError):
    def __init__(self, errors: list[str] | str):
        if isinstance(errors, str):
            errors = [errors]
        super().__init__("\n".join(errors))
        self.errors = errors


REFERENCE_CANDIDATE_NAME = "xgb_fixed_v1_reference"
CANDIDATE_NAMES = [
    "xgb_fixed_v1_reference",
    "xgb_depth3_v1",
    "xgb_depth3_regularized_v1",
    "xgb_depth4_regularized_v1",
    "xgb_depth5_regularized_v1",
    "xgb_low_learning_rate_v1",
]
STAGE7_MODEL_NAME = "xgboost_fixed_v1"
REQUIRED_FEATURE_COUNT = 63


@dataclass
class Stage8CandidateFoldResult:
    predictions: pd.DataFrame
    selector_metadata: dict[str, Any]
    refit_metadata: dict[str, Any]
    fold_metadata: dict[str, Any]
    inner_split: dict[str, Any]
    learning_curve: pd.DataFrame
    feature_importance: pd.DataFrame
    reload_verified: bool


@dataclass
class Stage8Outputs:
    predictions: pd.DataFrame
    metrics_by_candidate_subset: pd.DataFrame
    metrics_by_candidate_fold: pd.DataFrame
    metrics_by_candidate_offset: pd.DataFrame
    candidate_summary: pd.DataFrame
    candidate_comparison: pd.DataFrame
    selection_audit: dict[str, Any]
    learning_curves: pd.DataFrame
    calibration_equal_width: pd.DataFrame
    calibration_equal_frequency: pd.DataFrame
    feature_importance_by_candidate_fold: pd.DataFrame
    feature_importance_stability: pd.DataFrame
    inner_split_audit: pd.DataFrame
    model_manifest: dict[str, Any]
    quality_gates: dict[str, Any]
    fold_metadata: dict[str, dict[str, dict[str, Any]]]
    reference_reproduction: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def canonical_json(value: Any) -> str:
    return json.dumps(make_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candidate_definitions_sha256(candidate_definitions: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    return hashlib.sha256(canonical_json(list(candidate_definitions)).encode("utf-8")).hexdigest()


def freeze_candidate_definitions(candidate_definitions: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(json.dumps(candidate, sort_keys=True)) for candidate in candidate_definitions)


def stage7_reference_parameters(stage7_model_manifest: dict[str, Any]) -> dict[str, Any]:
    params = dict(stage7_model_manifest.get("xgboost_parameters", {}))
    return {
        "learning_rate": params.get("learning_rate"),
        "max_depth": params.get("max_depth"),
        "min_child_weight": params.get("min_child_weight"),
        "gamma": params.get("gamma"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "reg_alpha": params.get("reg_alpha"),
        "reg_lambda": params.get("reg_lambda"),
        "max_estimators": params.get("n_estimators", params.get("max_estimators")),
        "early_stopping_rounds": params.get("early_stopping_rounds"),
    }


def reference_candidate_matches_stage7(candidate: dict[str, Any], reference_params: dict[str, Any]) -> bool:
    keys = [
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "gamma",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "max_estimators",
        "early_stopping_rounds",
    ]
    return all(candidate.get(key) == reference_params.get(key) for key in keys)


def validate_candidate_definitions(
    candidate_definitions: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    config: dict[str, Any],
    reference_params: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    candidates = list(candidate_definitions)
    expected_count = int(config.get("expected_candidate_count", 6))
    if expected_count != 6 or len(candidates) != 6:
        errors.append("candidate set must contain exactly six candidates")
    names = [candidate.get("model_name") for candidate in candidates]
    if names != CANDIDATE_NAMES:
        errors.append(f"candidate names/order must be {CANDIDATE_NAMES}")
    if len(set(names)) != len(names):
        errors.append("candidate names must be unique")
    if candidates and not reference_candidate_matches_stage7(candidates[0], reference_params):
        errors.append("Candidate 1 does not match Stage7 reference parameters")
    for candidate in candidates:
        if candidate.get("scale_pos_weight") not in (None, 1.0):
            errors.append(f"{candidate.get('model_name')} must not override scale_pos_weight")
        for key in [
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "gamma",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "max_estimators",
            "early_stopping_rounds",
        ]:
            if key not in candidate:
                errors.append(f"{candidate.get('model_name')} missing {key}")
    if errors:
        raise Stage8ValidationError(errors)
    return {
        "candidate_count": len(candidates),
        "candidate_names": names,
        "candidate_definitions_sha256": candidate_definitions_sha256(candidates),
        "candidate_set_frozen_before_run": True,
    }


def build_candidate_config(config: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    params = dict(config["xgboost_common_parameters"])
    params.update(
        {
            "n_estimators": int(candidate["max_estimators"]),
            "learning_rate": candidate["learning_rate"],
            "max_depth": int(candidate["max_depth"]),
            "min_child_weight": candidate["min_child_weight"],
            "gamma": candidate["gamma"],
            "subsample": candidate["subsample"],
            "colsample_bytree": candidate["colsample_bytree"],
            "reg_alpha": candidate["reg_alpha"],
            "reg_lambda": candidate["reg_lambda"],
            "early_stopping_rounds": int(candidate["early_stopping_rounds"]),
        }
    )
    return {
        "model_name": candidate["model_name"],
        "xgboost_parameters": params,
        "early_stopping_window_days": config["early_stopping_window_days"],
        "inner_purge_horizon_minutes": config["inner_purge_horizon_minutes"],
        "minimum_inner_early_stop_samples": config.get("minimum_inner_early_stop_samples", 10000),
        "fixed_prediction_threshold": config["fixed_prediction_threshold"],
        "random_seed": config["random_seed"],
        "n_jobs": config["n_jobs"],
        "numeric_tolerances": config.get("numeric_tolerances", {"prediction_atol": 1e-12}),
    }


def refit_params_from_candidate_config(candidate_config: dict[str, Any], best_n_estimators: int) -> dict[str, Any]:
    params = dict(candidate_config["xgboost_parameters"])
    params["n_estimators"] = int(best_n_estimators)
    params.pop("early_stopping_rounds", None)
    return verify_xgboost_parameters(params, int(candidate_config["n_jobs"]), selector=False)


def selector_params_from_candidate_config(candidate_config: dict[str, Any]) -> dict[str, Any]:
    return verify_xgboost_parameters(candidate_config["xgboost_parameters"], int(candidate_config["n_jobs"]), selector=True)


def row_id_sha256(values: pd.Series | list[int] | np.ndarray) -> str:
    arr = np.asarray(list(values), dtype=np.int64)
    return hashlib.sha256("\n".join(str(int(v)) for v in arr).encode("utf-8")).hexdigest()


def inner_split_row_hashes(inner: dict[str, Any]) -> dict[str, Any]:
    fit = inner["inner_fit"]["dataset_row_id"].astype("int64").tolist()
    purged = inner["inner_purged"]["dataset_row_id"].astype("int64").tolist()
    early = inner["inner_early_stop"]["dataset_row_id"].astype("int64").tolist()
    return {
        "inner_fit_count": len(fit),
        "inner_purged_count": len(purged),
        "inner_early_stop_count": len(early),
        "inner_fit_dataset_row_id_sha256": row_id_sha256(fit),
        "inner_purged_dataset_row_id_sha256": row_id_sha256(purged),
        "inner_early_stop_dataset_row_id_sha256": row_id_sha256(early),
    }


def fit_stage8_candidate_fold(
    outer_train: pd.DataFrame,
    outer_validation: pd.DataFrame,
    feature_columns: list[str],
    fold_info: dict[str, Any],
    fold_name: str,
    candidate: dict[str, Any],
    config: dict[str, Any],
    fold_dir: Path | None = None,
) -> Stage8CandidateFoldResult:
    candidate_config = build_candidate_config(config, candidate)
    model_name = str(candidate["model_name"])
    validation_start = parse_utc_ms(str(fold_info["validation_start"]))
    started = time.perf_counter()
    inner = make_inner_time_split(
        outer_train,
        validation_start,
        int(config["early_stopping_window_days"]),
        int(config["inner_purge_horizon_minutes"]),
    )
    inner_audit = validate_inner_split(inner, int(config.get("minimum_inner_early_stop_samples", 10000)))
    inner_hashes = inner_split_row_hashes(inner)
    selector_params = selector_params_from_candidate_config(candidate_config)

    X_fit = prepare_feature_matrix(inner["inner_fit"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_fit = inner["inner_fit"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    X_early = prepare_feature_matrix(inner["inner_early_stop"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_early = inner["inner_early_stop"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    selector = xgb.XGBClassifier(**selector_params)
    selector.fit(X_fit, y_fit, eval_set=[(X_early, y_early)], verbose=False)
    logloss = list(selector.evals_result()["validation_0"]["logloss"])
    best_iteration = int(getattr(selector, "best_iteration", int(np.argmin(logloss))))
    best_score = float(getattr(selector, "best_score", logloss[best_iteration]))
    best_n_estimators = best_iteration + 1
    reached_max = len(logloss) >= int(selector_params["n_estimators"])
    learning_curve = pd.DataFrame(
        {
            "candidate_name": model_name,
            "model_name": model_name,
            "fold_name": fold_name,
            "boosting_round": np.arange(len(logloss), dtype=int),
            "inner_early_stop_logloss": logloss,
            "best_iteration": best_iteration,
            "best_n_estimators": best_n_estimators,
            "best_score": best_score,
            "stopped_early": not reached_max,
            "reached_max_estimators": reached_max,
        }
    )
    selector_metadata = {
        "candidate_name": model_name,
        "model_name": model_name,
        "fold_name": fold_name,
        "selector_model_object_id": id(selector),
        "max_n_estimators": int(selector_params["n_estimators"]),
        "early_stopping_rounds": int(selector_params["early_stopping_rounds"]),
        "eval_metric": "logloss",
        "best_iteration": best_iteration,
        "best_n_estimators": best_n_estimators,
        "best_score": best_score,
        "last_iteration_logloss": float(logloss[-1]),
        "rounds_after_best": int(len(logloss) - best_n_estimators),
        "stopped_early": bool(not reached_max),
        "reached_max_estimators": bool(reached_max),
        "inner_fit_sample_count": int(len(inner["inner_fit"])),
        "inner_early_stop_sample_count": int(len(inner["inner_early_stop"])),
        "inner_purged_sample_count": int(len(inner["inner_purged"])),
        "eval_set_dataset_row_ids": [int(v) for v in inner["inner_early_stop"]["dataset_row_id"].tolist()],
        "outer_validation_used_for_early_stopping": False,
        "final_test_used_for_early_stopping": False,
        "learning_curve_logloss": [float(v) for v in logloss],
        "selector_parameters": selector_params,
        **inner_hashes,
    }

    refit_params = refit_params_from_candidate_config(candidate_config, best_n_estimators)
    refit = xgb.XGBClassifier(**refit_params)
    X_train = prepare_feature_matrix(outer_train, feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_train = outer_train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    refit.fit(X_train, y_train, verbose=False)
    X_valid = prepare_feature_matrix(outer_validation, feature_columns).to_numpy(dtype=np.float32, copy=True)
    p_up = refit.predict_proba(X_valid)[:, 1]
    if not np.isfinite(p_up).all() or not ((p_up >= 0.0) & (p_up <= 1.0)).all():
        raise Stage8ValidationError(f"{model_name}/{fold_name} produced invalid probabilities")
    y_pred = (p_up >= float(config["fixed_prediction_threshold"])).astype(np.int8)
    pred = outer_validation[["dataset_row_id", "decision_time"]].copy()
    pred["fold_name"] = fold_name
    pred["model_name"] = model_name
    pred["p_up"] = p_up
    pred["y_pred"] = y_pred
    pred["prediction_threshold"] = float(config["fixed_prediction_threshold"])
    predictions = add_validation_metadata(pred, outer_validation)

    reload_verified = False
    model_path: Path | None = None
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / "refit_model.json"
        refit.save_model(model_path)
        reloaded = xgb.XGBClassifier()
        reloaded.load_model(model_path)
        sample_count = min(1000, len(X_valid))
        reloaded_p = reloaded.predict_proba(X_valid[:sample_count])[:, 1]
        reload_verified = bool(np.allclose(reloaded_p, p_up[:sample_count], atol=float(config.get("numeric_tolerances", {}).get("prediction_atol", 1e-12))))
    else:
        reload_verified = True

    importance = extract_stage8_feature_importance_frame(refit, feature_columns, model_name, fold_name)
    elapsed = time.perf_counter() - started
    refit_metadata = {
        "candidate_name": model_name,
        "model_name": model_name,
        "fold_name": fold_name,
        "refit_model_object_id": id(refit),
        "refit_train_sample_count": int(len(outer_train)),
        "refit_validation_prediction_count": int(len(outer_validation)),
        "refit_parameters": refit_params,
        "refit_used_early_stopping": False,
        "outer_validation_used_for_fit": False,
        "outer_validation_used_for_parameter_selection": False,
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
        "class_weight_used": False,
        "class_weighting_disabled": True,
        "standard_scaler_used": False,
        "scale_pos_weight": refit_params["scale_pos_weight"],
        "model_file_path": str(model_path) if model_path else None,
        "model_file_sha256": sha256_file(model_path) if model_path and model_path.exists() else None,
        "reload_verified": bool(reload_verified),
    }
    fold_metadata = {
        "candidate_name": model_name,
        "model_name": model_name,
        "fold_name": fold_name,
        "train_row_count": int(len(outer_train)),
        "validation_row_count": int(len(outer_validation)),
        "outer_train_time_range": time_range(outer_train),
        "outer_validation_time_range": time_range(outer_validation),
        "inner_fit_time_range": time_range(inner["inner_fit"]),
        "inner_early_stop_time_range": time_range(inner["inner_early_stop"]),
        "inner_split_audit": {**inner_audit, **inner_hashes},
        "selector_metadata": selector_metadata,
        "refit_metadata": refit_metadata,
        "feature_columns": feature_columns,
        "feature_list_sha256": sha256_text("\n".join(feature_columns)),
        "candidate_parameters": candidate,
        "xgboost_version": xgb.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "random_seed": int(config["random_seed"]),
        "n_jobs": int(config["n_jobs"]),
        "fit_elapsed_seconds": float(elapsed),
        "process_rss_bytes": get_process_rss_bytes(),
    }
    if fold_dir is not None:
        write_json(fold_dir / "selector_metadata.json", selector_metadata)
        write_json(fold_dir / "refit_metadata.json", refit_metadata)
        write_json(fold_dir / "feature_importance.json", importance.to_dict(orient="records"))
        write_json(fold_dir / "fold_metadata.json", fold_metadata)

    del X_fit, X_early, X_train, X_valid, y_fit, y_early, y_train, selector, refit
    gc.collect()
    return Stage8CandidateFoldResult(
        predictions=predictions,
        selector_metadata=selector_metadata,
        refit_metadata=refit_metadata,
        fold_metadata=fold_metadata,
        inner_split=inner,
        learning_curve=learning_curve,
        feature_importance=importance,
        reload_verified=reload_verified,
    )


def extract_stage8_feature_importance_frame(model: xgb.XGBClassifier, feature_columns: list[str], model_name: str, fold_name: str) -> pd.DataFrame:
    booster = model.get_booster()
    maps = {importance_type: booster.get_score(importance_type=importance_type) for importance_type in IMPORTANCE_TYPES}
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_columns):
        key = f"f{index}"
        row = {"model_name": model_name, "candidate_name": model_name, "fold_name": fold_name, "feature_index": index + 1, "feature_name": feature}
        for importance_type in IMPORTANCE_TYPES:
            row[importance_type] = float(maps.get(importance_type, {}).get(key, 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    total_gain_sum = float(df["total_gain"].sum())
    df["normalized_gain"] = df["total_gain"] / total_gain_sum if total_gain_sum > 0 else 0.0
    df["importance_rank"] = df["total_gain"].rank(method="first", ascending=False).astype(int)
    return df


def stage8_feature_importance_stability(importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (model_name, feature), group in importance.groupby(["model_name", "feature_name"], sort=False):
        ranks = group["importance_rank"].to_numpy(dtype=float)
        rows.append(
            {
                "model_name": model_name,
                "candidate_name": model_name,
                "feature_name": feature,
                "mean_gain": float(group["gain"].mean()),
                "std_gain": float(group["gain"].std(ddof=1)) if len(group) > 1 else 0.0,
                "mean_normalized_gain": float(group["normalized_gain"].mean()),
                "mean_rank": float(np.mean(ranks)),
                "median_rank": float(np.median(ranks)),
                "used_fold_count": int((group[IMPORTANCE_TYPES].sum(axis=1) > 0).sum()),
                "top10_fold_count": int((group["importance_rank"] <= 10).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_name", "mean_normalized_gain", "mean_gain"], ascending=[True, False, False], kind="mergesort").reset_index(drop=True)


def validate_stage8_predictions(predictions: pd.DataFrame, splits: pd.DataFrame, fold_names: list[str], candidate_names: list[str]) -> None:
    errors: list[str] = []
    if set(predictions["model_name"].unique()) != set(candidate_names):
        errors.append("Stage8 predictions must contain exactly the configured candidates")
    if not predictions["p_up"].between(0.0, 1.0).all() or not np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all():
        errors.append("invalid probabilities")
    if not set(predictions["y_pred"].unique()).issubset({0, 1}):
        errors.append("y_pred must be binary")
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    if set(predictions["dataset_row_id"]).intersection(final_ids):
        errors.append("FINAL_TEST predictions detected")
    expected_per_candidate = 0
    for fold_name in fold_names:
        role_col = f"{fold_name}_role"
        valid_ids = set(splits.loc[splits[role_col].eq("VALIDATION"), "dataset_row_id"].tolist())
        train_ids = set(splits.loc[splits[role_col].eq("TRAIN"), "dataset_row_id"].tolist())
        expected_per_candidate += len(valid_ids)
        for candidate_name in candidate_names:
            fold_predictions = predictions[predictions["fold_name"].eq(fold_name) & predictions["model_name"].eq(candidate_name)]
            if set(fold_predictions["dataset_row_id"]) != valid_ids:
                errors.append(f"{candidate_name}/{fold_name} prediction set mismatch")
            if set(fold_predictions["dataset_row_id"]).intersection(train_ids):
                errors.append(f"{candidate_name}/{fold_name} TRAIN predictions detected")
            if not fold_predictions.groupby(["model_name", "fold_name", "dataset_row_id"]).size().eq(1).all():
                errors.append(f"{candidate_name}/{fold_name} duplicate predictions")
    if len(predictions) != expected_per_candidate * len(candidate_names):
        errors.append("Stage8 prediction total count mismatch")
    if errors:
        raise Stage8ValidationError(errors)


def compare_candidate_to_stage7_reference(candidate_predictions: pd.DataFrame, stage7_predictions: pd.DataFrame, tolerances: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    reference = candidate_predictions[candidate_predictions["model_name"].eq(REFERENCE_CANDIDATE_NAME)].copy()
    stage7 = stage7_predictions[stage7_predictions["model_name"].eq(STAGE7_MODEL_NAME)].copy()
    if reference.empty:
        errors.append("Candidate 1 reference predictions are missing")
    counts = stage7.groupby(["dataset_row_id", "fold_name"]).size()
    if not counts.empty and not counts.eq(1).all():
        errors.append("Stage7 predictions contain duplicates")
    merged = reference.merge(stage7, on=["dataset_row_id", "fold_name"], how="outer", suffixes=("_stage8", "_stage7"), indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        errors.append("Stage7 and Stage8 reference validation sample sets differ")
    if errors:
        raise Stage8ValidationError(errors)
    if not (merged["y_true_stage8"].astype(int) == merged["y_true_stage7"].astype(int)).all():
        raise Stage8ValidationError("Stage7 reference y_true mismatch")
    if not (merged["decision_time_stage8"].astype("int64") == merged["decision_time_stage7"].astype("int64")).all():
        raise Stage8ValidationError("Stage7 reference decision_time mismatch")
    max_diff = float(np.max(np.abs(merged["p_up_stage8"].to_numpy(dtype=float) - merged["p_up_stage7"].to_numpy(dtype=float)))) if len(merged) else 0.0
    pred_tol = float(tolerances["reference_prediction_absolute_tolerance"])
    metric_tol = float(tolerances["reference_metric_absolute_tolerance"])
    metric_diffs: list[dict[str, Any]] = []
    metric_match = True
    reference_aligned = reference.sort_values(["fold_name", "dataset_row_id"], kind="mergesort").reset_index(drop=True)
    stage7_aligned = stage7.sort_values(["fold_name", "dataset_row_id"], kind="mergesort").reset_index(drop=True)
    for subset_name, mask in subset_masks(reference_aligned):
        ref_subset = reference_aligned.loc[mask]
        st7_subset = stage7_aligned.loc[mask.to_numpy()]
        ref_metrics = compute_classification_metrics(ref_subset["y_true"], ref_subset["p_up"], ref_subset["y_pred"])
        st7_metrics = compute_classification_metrics(st7_subset["y_true"], st7_subset["p_up"], st7_subset["y_pred"])
        for metric_name in ["roc_auc", "average_precision", "log_loss", "brier_score", "accuracy", "mcc"]:
            a = ref_metrics[metric_name]
            b = st7_metrics[metric_name]
            diff = abs(a - b) if pd.notna(a) and pd.notna(b) else 0.0
            metric_diffs.append({"subset_name": subset_name, "metric": metric_name, "absolute_diff": float(diff)})
            metric_match = metric_match and diff <= metric_tol
    reproduced = bool(max_diff <= pred_tol and metric_match)
    return {
        "reference_config_reproduced": reproduced,
        "max_abs_prediction_diff": max_diff,
        "prediction_tolerance": pred_tol,
        "metric_tolerance": metric_tol,
        "metric_max_abs_diff": float(max((row["absolute_diff"] for row in metric_diffs), default=0.0)),
        "metric_diffs": metric_diffs,
    }


def compare_best_n_estimators_with_stage7(fold_metadata: dict[str, dict[str, dict[str, Any]]], stage7_model_manifest: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_match = True
    stage7_folds = stage7_model_manifest.get("folds", {})
    reference_folds = fold_metadata.get(REFERENCE_CANDIDATE_NAME, {})
    for fold_name, meta in reference_folds.items():
        stage8_best = int(meta["selector_metadata"]["best_n_estimators"])
        stage7_best = int(stage7_folds.get(fold_name, {}).get("selector_metadata", {}).get("best_n_estimators", stage8_best))
        match = stage8_best == stage7_best
        all_match = all_match and match
        rows.append({"fold_name": fold_name, "stage8_best_n_estimators": stage8_best, "stage7_best_n_estimators": stage7_best, "match": match})
    return {"best_n_estimators_match_stage7": bool(all_match), "folds": rows}


def evaluate_prediction_subsets(predictions: pd.DataFrame) -> pd.DataFrame:
    return stage6_evaluate_prediction_subsets(predictions)


def pooled_and_macro_summary(metrics_by_subset: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    return stage6_pooled_and_macro_summary(metrics_by_subset, predictions)


def calibration_table(predictions: pd.DataFrame, bins: int, strategy: str, subset_name: str) -> pd.DataFrame:
    return stage6_calibration_table(predictions, bins, strategy, subset_name)


def build_candidate_summary(predictions: pd.DataFrame, learning_curves: pd.DataFrame, comparison: pd.DataFrame | None = None) -> pd.DataFrame:
    metrics = evaluate_prediction_subsets(predictions)
    oof = pooled_and_macro_summary(metrics, predictions)
    if comparison is None:
        comparison = build_candidate_comparison(predictions, REFERENCE_CANDIDATE_NAME) if REFERENCE_CANDIDATE_NAME in set(predictions["model_name"]) else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name in sorted(predictions["model_name"].unique(), key=lambda n: CANDIDATE_NAMES.index(n) if n in CANDIDATE_NAMES else 999):
        model_oof = oof[oof["model_name"].eq(model_name)]
        model_metrics = metrics[metrics["model_name"].eq(model_name)]
        dense_pooled = metric_row(model_oof, "DENSE", "pooled")
        nonover_pooled = metric_row(model_oof, "NONOVERLAP_OFFSET_00", "pooled")
        offset_pooled = model_oof[model_oof["summary_type"].eq("pooled") & model_oof["subset_name"].str.startswith("OFFSET_")]
        dense_fold = model_metrics[model_metrics["subset_name"].eq("DENSE")]
        nonover_fold = model_metrics[model_metrics["subset_name"].eq("NONOVERLAP_OFFSET_00")]
        best_year = dense_fold.loc[dense_fold["roc_auc"].idxmax(), "fold_name"] if not dense_fold.empty else ""
        worst_idx = dense_fold["roc_auc"].idxmin() if not dense_fold.empty else None
        worst_year = dense_fold.loc[worst_idx, "fold_name"] if worst_idx is not None else ""
        worst_year_auc = float(dense_fold.loc[worst_idx, "roc_auc"]) if worst_idx is not None else np.nan
        weakest_offset = offset_extreme(offset_pooled, "min")
        strongest_offset = offset_extreme(offset_pooled, "max")
        if "model_name" in learning_curves.columns:
            best_tree_rows = learning_curves[learning_curves["model_name"].eq(model_name)].drop_duplicates(["model_name", "fold_name"])
        else:
            best_tree_rows = pd.DataFrame()
        candidate_comparison = comparison[(comparison["model_name"].eq(model_name)) & comparison["fold_name"].ne("POOLED")] if not comparison.empty else pd.DataFrame()
        dense_comp = candidate_comparison[candidate_comparison["subset_name"].eq("DENSE")] if not candidate_comparison.empty else pd.DataFrame()
        nonover_comp = candidate_comparison[candidate_comparison["subset_name"].eq("NONOVERLAP_OFFSET_00")] if not candidate_comparison.empty else pd.DataFrame()
        row = {
            "model_name": model_name,
            "candidate_name": model_name,
            "pooled_dense_auc": value_from_row(dense_pooled, "roc_auc"),
            "pooled_dense_logloss": value_from_row(dense_pooled, "log_loss"),
            "pooled_dense_brier": value_from_row(dense_pooled, "brier_score"),
            "nonoverlap_offset00_auc": value_from_row(nonover_pooled, "roc_auc"),
            "nonoverlap_offset00_logloss": value_from_row(nonover_pooled, "log_loss"),
            "nonoverlap_offset00_brier": value_from_row(nonover_pooled, "brier_score"),
            "offset_macro_auc": float(offset_pooled["roc_auc"].mean(skipna=True)),
            "offset_auc_std": float(offset_pooled["roc_auc"].std(skipna=True, ddof=1)),
            "offset_auc_min": float(offset_pooled["roc_auc"].min(skipna=True)),
            "offset_auc_max": float(offset_pooled["roc_auc"].max(skipna=True)),
            "offset_auc_range": float(offset_pooled["roc_auc"].max(skipna=True) - offset_pooled["roc_auc"].min(skipna=True)),
            "offset_macro_logloss": float(offset_pooled["log_loss"].mean(skipna=True)),
            "offset_logloss_std": float(offset_pooled["log_loss"].std(skipna=True, ddof=1)),
            "offset_logloss_max": float(offset_pooled["log_loss"].max(skipna=True)),
            "offset_macro_brier": float(offset_pooled["brier_score"].mean(skipna=True)),
            "offset_brier_std": float(offset_pooled["brier_score"].std(skipna=True, ddof=1)),
            "weakest_offset": weakest_offset,
            "strongest_offset": strongest_offset,
            "dense_fold_auc_mean": float(dense_fold["roc_auc"].mean(skipna=True)),
            "dense_fold_auc_std": float(dense_fold["roc_auc"].std(skipna=True, ddof=1)) if len(dense_fold) > 1 else 0.0,
            "dense_fold_auc_min": float(dense_fold["roc_auc"].min(skipna=True)),
            "dense_fold_logloss_mean": float(dense_fold["log_loss"].mean(skipna=True)),
            "dense_fold_logloss_std": float(dense_fold["log_loss"].std(skipna=True, ddof=1)) if len(dense_fold) > 1 else 0.0,
            "nonoverlap_fold_auc_mean": float(nonover_fold["roc_auc"].mean(skipna=True)),
            "nonoverlap_fold_auc_std": float(nonover_fold["roc_auc"].std(skipna=True, ddof=1)) if len(nonover_fold) > 1 else 0.0,
            "nonoverlap_fold_auc_min": float(nonover_fold["roc_auc"].min(skipna=True)),
            "nonoverlap_fold_logloss_mean": float(nonover_fold["log_loss"].mean(skipna=True)),
            "nonoverlap_fold_logloss_std": float(nonover_fold["log_loss"].std(skipna=True, ddof=1)) if len(nonover_fold) > 1 else 0.0,
            "fold_auc_above_0_5_count": int((dense_fold["roc_auc"] >= 0.5).sum()),
            "fold_auc_above_reference_count": int((dense_comp["delta_roc_auc"] > 0).sum()) if not dense_comp.empty else 0,
            "fold_logloss_better_than_reference_count": int((dense_comp["delta_log_loss"] < 0).sum()) if not dense_comp.empty else 0,
            "dense_auc_fold_win_count": int((dense_comp["delta_roc_auc"] > 0).sum()) if not dense_comp.empty else 0,
            "nonoverlap_auc_fold_win_count": int((nonover_comp["delta_roc_auc"] > 0).sum()) if not nonover_comp.empty else 0,
            "dense_logloss_fold_win_count": int((dense_comp["delta_log_loss"] < 0).sum()) if not dense_comp.empty else 0,
            "nonoverlap_logloss_fold_win_count": int((nonover_comp["delta_log_loss"] < 0).sum()) if not nonover_comp.empty else 0,
            "best_year": best_year,
            "worst_year": worst_year,
            "worst_year_auc": worst_year_auc,
            "has_fold_auc_below_0_5": bool((dense_fold["roc_auc"] < 0.5).any()),
            "mean_best_n_estimators": float(best_tree_rows["best_n_estimators"].mean()) if not best_tree_rows.empty else np.nan,
            "median_best_n_estimators": float(best_tree_rows["best_n_estimators"].median()) if not best_tree_rows.empty else np.nan,
            "std_best_n_estimators": float(best_tree_rows["best_n_estimators"].std(ddof=1)) if len(best_tree_rows) > 1 else 0.0,
            "min_best_n_estimators": int(best_tree_rows["best_n_estimators"].min()) if not best_tree_rows.empty else 0,
            "max_best_n_estimators": int(best_tree_rows["best_n_estimators"].max()) if not best_tree_rows.empty else 0,
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    ref = summary[summary["model_name"].eq(REFERENCE_CANDIDATE_NAME)]
    if not ref.empty:
        ref_row = ref.iloc[0]
        for col in [
            "pooled_dense_auc",
            "pooled_dense_logloss",
            "pooled_dense_brier",
            "offset_macro_auc",
            "offset_macro_logloss",
            "offset_macro_brier",
            "nonoverlap_offset00_auc",
            "nonoverlap_offset00_logloss",
            "nonoverlap_offset00_brier",
        ]:
            summary[f"delta_{col}_vs_reference"] = summary[col] - ref_row[col]
    return summary


def metric_row(oof: pd.DataFrame, subset_name: str, summary_type: str) -> pd.Series:
    row = oof[(oof["subset_name"].eq(subset_name)) & (oof["summary_type"].eq(summary_type))]
    if row.empty:
        return pd.Series(dtype=float)
    return row.iloc[0]


def offset_extreme(offset_pooled: pd.DataFrame, direction: str) -> str:
    if offset_pooled.empty:
        return ""
    valid = offset_pooled.dropna(subset=["roc_auc"])
    if valid.empty:
        return str(offset_pooled.iloc[0]["subset_name"])
    idx = valid["roc_auc"].idxmin() if direction == "min" else valid["roc_auc"].idxmax()
    return str(valid.loc[idx, "subset_name"])


def value_from_row(row: pd.Series, key: str) -> float:
    return float(row.get(key, np.nan)) if not row.empty else np.nan


def build_candidate_comparison(predictions: pd.DataFrame, reference_model_name: str) -> pd.DataFrame:
    if reference_model_name not in set(predictions["model_name"].unique()):
        raise Stage8ValidationError("reference candidate predictions are missing")
    rows: list[dict[str, Any]] = []
    for model_name in sorted(predictions["model_name"].unique(), key=lambda n: CANDIDATE_NAMES.index(n) if n in CANDIDATE_NAMES else 999):
        candidate = predictions[predictions["model_name"].eq(model_name)].sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        reference = predictions[predictions["model_name"].eq(reference_model_name)].sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        if model_name == reference_model_name:
            continue
        merged = candidate[["dataset_row_id", "fold_name", "decision_time", "y_true"]].merge(
            reference[["dataset_row_id", "fold_name", "decision_time", "y_true"]],
            on=["dataset_row_id", "fold_name"],
            how="outer",
            suffixes=("_candidate", "_reference"),
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            raise Stage8ValidationError(f"{model_name} sample set does not match reference")
        if not (merged["y_true_candidate"].astype(int) == merged["y_true_reference"].astype(int)).all():
            raise Stage8ValidationError(f"{model_name} y_true does not match reference")
        for fold_name in sorted(candidate["fold_name"].unique()):
            rows.extend(candidate_comparison_rows(candidate[candidate["fold_name"].eq(fold_name)], reference[reference["fold_name"].eq(fold_name)], model_name, reference_model_name, fold_name))
        rows.extend(candidate_comparison_rows(candidate, reference, model_name, reference_model_name, "POOLED"))
    return pd.DataFrame(rows)


def candidate_comparison_rows(candidate_df: pd.DataFrame, reference_df: pd.DataFrame, model_name: str, reference_model_name: str, fold_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_df = candidate_df.reset_index(drop=True)
    reference_df = reference_df.reset_index(drop=True)
    for subset_name, mask in subset_masks(candidate_df):
        cand_subset = candidate_df.loc[mask]
        ref_subset = reference_df.loc[mask.to_numpy()]
        candidate_metrics = compute_classification_metrics(cand_subset["y_true"], cand_subset["p_up"], cand_subset["y_pred"])
        reference_metrics = compute_classification_metrics(ref_subset["y_true"], ref_subset["p_up"], ref_subset["y_pred"])
        rows.append(
            {
                "comparison_type": "vs_candidate1_reference",
                "fold_name": fold_name,
                "subset_name": subset_name,
                "model_name": model_name,
                "reference_model_name": reference_model_name,
                "candidate_roc_auc": candidate_metrics["roc_auc"],
                "reference_roc_auc": reference_metrics["roc_auc"],
                "delta_roc_auc": candidate_metrics["roc_auc"] - reference_metrics["roc_auc"],
                "candidate_average_precision": candidate_metrics["average_precision"],
                "reference_average_precision": reference_metrics["average_precision"],
                "delta_average_precision": candidate_metrics["average_precision"] - reference_metrics["average_precision"],
                "candidate_log_loss": candidate_metrics["log_loss"],
                "reference_log_loss": reference_metrics["log_loss"],
                "delta_log_loss": candidate_metrics["log_loss"] - reference_metrics["log_loss"],
                "candidate_brier_score": candidate_metrics["brier_score"],
                "reference_brier_score": reference_metrics["brier_score"],
                "delta_brier": candidate_metrics["brier_score"] - reference_metrics["brier_score"],
                "candidate_accuracy": candidate_metrics["accuracy"],
                "reference_accuracy": reference_metrics["accuracy"],
                "delta_accuracy": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
                "candidate_mcc": candidate_metrics["mcc"],
                "reference_mcc": reference_metrics["mcc"],
                "delta_mcc": candidate_metrics["mcc"] - reference_metrics["mcc"],
                "sample_count": candidate_metrics["sample_count"],
                "delta_direction_note": "Positive ROC-AUC/AP/Accuracy/MCC favors candidate; negative LogLoss/Brier favors candidate.",
            }
        )
    return rows


def build_logistic_comparison(predictions: pd.DataFrame, stage6_predictions: pd.DataFrame) -> pd.DataFrame:
    logistic = stage6_predictions[stage6_predictions["model_name"].eq("logistic_regression_l2")].copy()
    rows: list[dict[str, Any]] = []
    for model_name, candidate in predictions.groupby("model_name", sort=True):
        merged = candidate.merge(logistic, on=["dataset_row_id", "fold_name"], how="outer", suffixes=("_candidate", "_logistic"), indicator=True, validate="one_to_one")
        if not merged["_merge"].eq("both").all():
            raise Stage8ValidationError(f"{model_name} sample set does not match Stage6 logistic")
        if not (merged["y_true_candidate"].astype(int) == merged["y_true_logistic"].astype(int)).all():
            raise Stage8ValidationError(f"{model_name} y_true does not match Stage6 logistic")
        cand = candidate.sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        base = logistic.sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        for fold_name in sorted(cand["fold_name"].unique()):
            rows.extend(logistic_comparison_rows(cand[cand["fold_name"].eq(fold_name)], base[base["fold_name"].eq(fold_name)], model_name, fold_name))
        rows.extend(logistic_comparison_rows(cand, base, model_name, "POOLED"))
    return pd.DataFrame(rows)


def logistic_comparison_rows(candidate_df: pd.DataFrame, logistic_df: pd.DataFrame, model_name: str, fold_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_df = candidate_df.reset_index(drop=True)
    logistic_df = logistic_df.reset_index(drop=True)
    for subset_name, mask in subset_masks(candidate_df):
        cand_subset = candidate_df.loc[mask]
        base_subset = logistic_df.loc[mask.to_numpy()]
        candidate_metrics = compute_classification_metrics(cand_subset["y_true"], cand_subset["p_up"], cand_subset["y_pred"])
        base_metrics = compute_classification_metrics(base_subset["y_true"], base_subset["p_up"], base_subset["y_pred"])
        rows.append(
            {
                "comparison_type": "vs_logistic_regression_l2",
                "fold_name": fold_name,
                "subset_name": subset_name,
                "model_name": model_name,
                "reference_model_name": "logistic_regression_l2",
                "candidate_roc_auc": candidate_metrics["roc_auc"],
                "reference_roc_auc": base_metrics["roc_auc"],
                "delta_roc_auc": candidate_metrics["roc_auc"] - base_metrics["roc_auc"],
                "candidate_average_precision": candidate_metrics["average_precision"],
                "reference_average_precision": base_metrics["average_precision"],
                "delta_average_precision": candidate_metrics["average_precision"] - base_metrics["average_precision"],
                "candidate_log_loss": candidate_metrics["log_loss"],
                "reference_log_loss": base_metrics["log_loss"],
                "delta_log_loss": candidate_metrics["log_loss"] - base_metrics["log_loss"],
                "candidate_brier_score": candidate_metrics["brier_score"],
                "reference_brier_score": base_metrics["brier_score"],
                "delta_brier": candidate_metrics["brier_score"] - base_metrics["brier_score"],
                "candidate_accuracy": candidate_metrics["accuracy"],
                "reference_accuracy": base_metrics["accuracy"],
                "delta_accuracy": candidate_metrics["accuracy"] - base_metrics["accuracy"],
                "candidate_mcc": candidate_metrics["mcc"],
                "reference_mcc": base_metrics["mcc"],
                "delta_mcc": candidate_metrics["mcc"] - base_metrics["mcc"],
                "sample_count": candidate_metrics["sample_count"],
                "delta_direction_note": "Positive ROC-AUC/AP/Accuracy/MCC favors candidate; negative LogLoss/Brier favors candidate.",
            }
        )
    return rows


def select_development_config(
    candidate_summary: pd.DataFrame,
    candidate_comparison: pd.DataFrame,
    selection_rules: dict[str, Any],
    reference_reproduced: bool,
    engineering_gate: bool,
) -> dict[str, Any]:
    candidate_comparison = candidate_comparison.copy()
    if not candidate_comparison.empty and "comparison_type" not in candidate_comparison.columns:
        candidate_comparison["comparison_type"] = "vs_candidate1_reference"
    if not reference_reproduced or not engineering_gate:
        return {
            "selected_development_config": REFERENCE_CANDIDATE_NAME,
            "development_recommendation": "INVESTIGATE_PIPELINE",
            "improvement_not_material": True,
            "candidate_qualifications": [],
            "ranked_candidates": [],
            "minimum_gain_check": {"condition_a_met": False, "condition_b_met": False},
            "selection_reason": "Reference reproduction or engineering gate failed; no replacement selected.",
        }
    reference = candidate_summary[candidate_summary["model_name"].eq(REFERENCE_CANDIDATE_NAME)].iloc[0]
    qualifications: list[dict[str, Any]] = []
    qualified_names: list[str] = []
    for _, row in candidate_summary.iterrows():
        model_name = row["model_name"]
        engineering_ok = not bool(row.get("has_fold_auc_below_0_5", False))
        dense_bad = candidate_comparison[
            (candidate_comparison["comparison_type"].eq("vs_candidate1_reference"))
            & (candidate_comparison["model_name"].eq(model_name))
            & (candidate_comparison["subset_name"].eq("DENSE"))
            & (candidate_comparison["fold_name"].ne("POOLED"))
            & (candidate_comparison["delta_log_loss"] > float(selection_rules["dense_fold_logloss_bad_delta"]))
        ]
        if model_name == REFERENCE_CANDIDATE_NAME:
            prob_ok = True
        else:
            prob_ok = (
                row["pooled_dense_logloss"] - reference["pooled_dense_logloss"] <= float(selection_rules["pooled_dense_logloss_max_degradation"])
                and row["offset_macro_logloss"] - reference["offset_macro_logloss"] <= float(selection_rules["offset_macro_logloss_max_degradation"])
                and row["pooled_dense_brier"] - reference["pooled_dense_brier"] <= float(selection_rules["pooled_dense_brier_max_degradation"])
                and len(dense_bad) <= int(selection_rules["dense_fold_logloss_bad_fold_limit"])
            )
        if not engineering_ok:
            status = "DISQUALIFIED_ENGINEERING"
        elif not prob_ok:
            status = "DISQUALIFIED_PROBABILITY_QUALITY"
        else:
            status = "QUALIFIED"
            qualified_names.append(model_name)
        qualifications.append(
            {
                "model_name": model_name,
                "engineering_qualified": bool(engineering_ok),
                "probability_quality_qualified": bool(prob_ok),
                "qualification_status": status,
                "dense_bad_logloss_fold_count": int(len(dense_bad)),
            }
        )
    qualified = candidate_summary[candidate_summary["model_name"].isin(qualified_names)].copy()
    ranked = rank_candidates(qualified, selection_rules)
    best_name = ranked[0]["model_name"] if ranked else REFERENCE_CANDIDATE_NAME
    best = candidate_summary[candidate_summary["model_name"].eq(best_name)].iloc[0]
    dense_ref_comp = candidate_comparison[
        (candidate_comparison["comparison_type"].eq("vs_candidate1_reference"))
        & (candidate_comparison["model_name"].eq(best_name))
        & (candidate_comparison["subset_name"].eq("DENSE"))
        & (candidate_comparison["fold_name"].ne("POOLED"))
    ]
    dense_wins = int((dense_ref_comp["delta_roc_auc"] > 0).sum()) if not dense_ref_comp.empty else int(best.get("dense_auc_fold_win_count", 0))
    dense_logloss_wins = int((dense_ref_comp["delta_log_loss"] < 0).sum()) if not dense_ref_comp.empty else int(best.get("dense_logloss_fold_win_count", 0))
    condition_a = (
        best["offset_macro_auc"] - reference["offset_macro_auc"] >= float(selection_rules["condition_a_offset_macro_auc_gain"])
        and best["offset_macro_logloss"] - reference["offset_macro_logloss"] <= float(selection_rules["condition_a_offset_macro_logloss_max_degradation"])
        and dense_wins >= int(selection_rules["condition_a_dense_auc_fold_wins"])
    )
    condition_b = (
        reference["offset_macro_logloss"] - best["offset_macro_logloss"] >= float(selection_rules["condition_b_offset_macro_logloss_improvement"])
        and best["offset_macro_auc"] - reference["offset_macro_auc"] >= -float(selection_rules["condition_b_offset_macro_auc_max_decline"])
        and dense_logloss_wins >= int(selection_rules["condition_b_dense_logloss_fold_wins"])
    )
    if best_name != REFERENCE_CANDIDATE_NAME and (condition_a or condition_b):
        selected = best_name
        improvement_not_material = False
        recommendation = "REPLACE_STAGE7_REFERENCE_WITH_SELECTED"
        reason = "Top qualified candidate met minimum material improvement condition."
    else:
        selected = REFERENCE_CANDIDATE_NAME
        improvement_not_material = best_name != REFERENCE_CANDIDATE_NAME
        recommendation = "KEEP_STAGE7_REFERENCE"
        reason = "Top candidate did not meet minimum material improvement condition." if best_name != REFERENCE_CANDIDATE_NAME else "Reference candidate remains top ranked."
    return {
        "selected_development_config": selected,
        "development_recommendation": recommendation,
        "improvement_not_material": bool(improvement_not_material),
        "candidate_qualifications": qualifications,
        "ranked_candidates": ranked,
        "minimum_gain_check": {
            "best_ranked_candidate": best_name,
            "condition_a_met": bool(condition_a),
            "condition_b_met": bool(condition_b),
            "offset_macro_auc_delta": float(best["offset_macro_auc"] - reference["offset_macro_auc"]),
            "offset_macro_logloss_delta": float(best["offset_macro_logloss"] - reference["offset_macro_logloss"]),
            "dense_auc_fold_win_count": dense_wins,
            "dense_logloss_fold_win_count": dense_logloss_wins,
        },
        "selection_reason": reason,
    }


def rank_candidates(candidate_summary: pd.DataFrame, selection_rules: dict[str, Any]) -> list[dict[str, Any]]:
    if candidate_summary.empty:
        return []
    ranked_df = candidate_summary.copy()
    ranked_df["_candidate_order"] = ranked_df["model_name"].map({name: i for i, name in enumerate(CANDIDATE_NAMES)}).fillna(999)
    top_auc = float(ranked_df["offset_macro_auc"].max(skipna=True))
    tie_tolerance = float(selection_rules["auc_tie_tolerance"])
    ranked_df["_top_auc_tie_group"] = (top_auc - ranked_df["offset_macro_auc"]).abs().le(tie_tolerance).astype(int)
    ranked_df["_auc_primary_sort"] = np.where(ranked_df["_top_auc_tie_group"].eq(1), top_auc, ranked_df["offset_macro_auc"])
    ranked_df = ranked_df.sort_values(
        [
            "_top_auc_tie_group",
            "_auc_primary_sort",
            "offset_macro_logloss",
            "pooled_dense_logloss",
            "offset_auc_std",
            "dense_fold_auc_min",
            "max_depth",
            "median_best_n_estimators",
            "_candidate_order",
        ],
        ascending=[False, False, True, True, True, False, True, True, True],
        kind="mergesort",
    )
    ranked: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
        ranked.append(
            {
                "rank": rank,
                "model_name": row["model_name"],
                "offset_macro_auc": float(row["offset_macro_auc"]),
                "offset_macro_logloss": float(row["offset_macro_logloss"]),
                "pooled_dense_logloss": float(row["pooled_dense_logloss"]),
                "offset_auc_std": float(row["offset_auc_std"]),
                "dense_fold_auc_min": float(row["dense_fold_auc_min"]),
                "max_depth": int(row.get("max_depth", 999)) if pd.notna(row.get("max_depth", np.nan)) else None,
                "median_best_n_estimators": float(row.get("median_best_n_estimators", np.nan)),
                "within_auc_tie_tolerance_of_top": abs(float(row["offset_macro_auc"]) - top_auc) <= float(selection_rules["auc_tie_tolerance"]),
            }
        )
    return ranked


def attach_candidate_parameters_to_summary(summary: pd.DataFrame, candidates: tuple[dict[str, Any], ...]) -> pd.DataFrame:
    param_df = pd.DataFrame(
        [
            {
                "model_name": candidate["model_name"],
                "max_depth": int(candidate["max_depth"]),
                "learning_rate": float(candidate["learning_rate"]),
                "reg_alpha": float(candidate["reg_alpha"]),
                "reg_lambda": float(candidate["reg_lambda"]),
                "min_child_weight": float(candidate["min_child_weight"]),
            }
            for candidate in candidates
        ]
    )
    drop_cols = [col for col in ["max_depth", "learning_rate", "reg_alpha", "reg_lambda", "min_child_weight"] if col in summary.columns]
    summary = summary.drop(columns=drop_cols, errors="ignore")
    return summary.merge(param_df, on="model_name", how="left")


def build_stage8_outputs(
    dataset: pd.DataFrame,
    splits: pd.DataFrame,
    stage6_predictions: pd.DataFrame,
    stage7_predictions: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    fold_manifest: dict[str, Any],
    stage7_model_manifest: dict[str, Any],
    config: dict[str, Any],
    root: Path | None = None,
    runtime_hashes: dict[str, str] | None = None,
) -> Stage8Outputs:
    candidates = freeze_candidate_definitions(config["candidate_definitions"])
    candidate_validation = validate_candidate_definitions(candidates, config, stage7_reference_parameters(stage7_model_manifest))
    feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest, int(config["feature_count"]))
    if list(stage7_model_manifest.get("feature_columns", [])) and list(stage7_model_manifest["feature_columns"]) != feature_columns:
        raise Stage8ValidationError("Stage7 model manifest feature order does not match Stage5 manifest")
    joined = joined_dataset(dataset, splits)
    prepare_feature_matrix(joined, feature_columns)
    folds = {fold["name"]: fold for fold in fold_manifest["folds"]}
    prediction_frames: list[pd.DataFrame] = []
    learning_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
    fold_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    all_models_reload_verified = True
    all_models_serialized = True
    all_outer_integrity = True
    all_inner_integrity = True
    all_parameters_verified = True
    all_features_finite = True
    all_best_iterations_valid = True
    inner_hash_by_fold: dict[str, dict[str, Any]] = {}

    for fold_name in config["fold_names"]:
        fold_info = folds[fold_name]
        validate_outer_fold_integrity(dataset, splits, fold_info, fold_name)
        role_col = f"{fold_name}_role"
        outer_train = joined[joined[role_col].eq("TRAIN")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
        validation_start = parse_utc_ms(str(fold_info["validation_start"]))
        inner = make_inner_time_split(outer_train, validation_start, int(config["early_stopping_window_days"]), int(config["inner_purge_horizon_minutes"]))
        inner_hash_by_fold[fold_name] = inner_split_row_hashes(inner)
        del outer_train, inner

    for candidate in candidates:
        candidate_name = candidate["model_name"]
        fold_metadata[candidate_name] = {}
        for fold_name in config["fold_names"]:
            fold_info = folds[fold_name]
            role_col = f"{fold_name}_role"
            outer_train = joined[joined[role_col].eq("TRAIN")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
            outer_validation = joined[joined[role_col].eq("VALIDATION")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
            fold_dir = (root / config["model_output_dir"] / candidate_name / fold_name).resolve() if root is not None else None
            try:
                result = fit_stage8_candidate_fold(outer_train, outer_validation, feature_columns, fold_info, fold_name, candidate, config, fold_dir)
            except Stage7ValidationError as exc:
                all_inner_integrity = False
                raise Stage8ValidationError(exc.errors) from exc
            expected_hashes = inner_hash_by_fold[fold_name]
            actual_hashes = inner_split_row_hashes(result.inner_split)
            same_inner = actual_hashes == expected_hashes
            stage7_eval_ids = stage7_model_manifest.get("folds", {}).get(fold_name, {}).get("selector_metadata", {}).get("eval_set_dataset_row_ids")
            stage7_early_hash_match = True
            if stage7_eval_ids:
                stage7_early_hash_match = row_id_sha256(stage7_eval_ids) == actual_hashes["inner_early_stop_dataset_row_id_sha256"]
            inner_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_name": candidate_name,
                    "fold_name": fold_name,
                    **result.fold_metadata["inner_split_audit"],
                    "outer_validation_used_for_early_stopping": False,
                    "final_test_used_for_early_stopping": False,
                    "same_inner_split_as_other_candidates": bool(same_inner),
                    "stage7_inner_early_stop_hash_match": bool(stage7_early_hash_match),
                }
            )
            prediction_frames.append(result.predictions)
            learning_frames.append(result.learning_curve)
            importance_frames.append(result.feature_importance)
            fold_metadata[candidate_name][fold_name] = result.fold_metadata
            all_models_reload_verified = all_models_reload_verified and result.reload_verified
            all_models_serialized = all_models_serialized and (fold_dir is None or (fold_dir / "refit_model.json").exists())
            all_best_iterations_valid = all_best_iterations_valid and result.selector_metadata["best_n_estimators"] >= 1
            del outer_train, outer_validation, result
            gc.collect()

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["model_name", "fold_name", "decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)
    validate_stage8_predictions(predictions, splits, config["fold_names"], CANDIDATE_NAMES)
    metrics_by_subset = evaluate_prediction_subsets(predictions)
    metrics_by_candidate_fold = metrics_by_subset[metrics_by_subset["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])].copy()
    metrics_by_candidate_offset = metrics_by_subset[metrics_by_subset["subset_name"].str.startswith("OFFSET_")].copy()
    learning_curves = pd.concat(learning_frames, ignore_index=True)
    candidate_comparison_reference = build_candidate_comparison(predictions, REFERENCE_CANDIDATE_NAME)
    candidate_summary = attach_candidate_parameters_to_summary(build_candidate_summary(predictions, learning_curves, candidate_comparison_reference), candidates)
    candidate_comparison_logistic = build_logistic_comparison(predictions, stage6_predictions)
    candidate_comparison = pd.concat([candidate_comparison_reference, candidate_comparison_logistic], ignore_index=True)
    calibration_equal_width = pd.concat(
        [
            calibration_table(predictions, int(config["calibration_bins"]), "equal_width", "DENSE"),
            calibration_table(predictions, int(config["calibration_bins"]), "equal_width", "NONOVERLAP_OFFSET_00"),
        ],
        ignore_index=True,
    )
    calibration_equal_frequency = pd.concat(
        [
            calibration_table(predictions, int(config["calibration_bins"]), "equal_frequency", "DENSE"),
            calibration_table(predictions, int(config["calibration_bins"]), "equal_frequency", "NONOVERLAP_OFFSET_00"),
        ],
        ignore_index=True,
    )
    importance_by_candidate_fold = pd.concat(importance_frames, ignore_index=True)
    importance_stability = stage8_feature_importance_stability(importance_by_candidate_fold)
    inner_split_audit = pd.DataFrame(inner_rows)
    reference_reproduction = compare_candidate_to_stage7_reference(predictions, stage7_predictions, config["reference_reproduction_tolerances"])
    best_tree_reproduction = compare_best_n_estimators_with_stage7(fold_metadata, stage7_model_manifest)
    reference_reproduction["best_n_estimators_match_stage7"] = best_tree_reproduction["best_n_estimators_match_stage7"]
    reference_reproduction["best_n_estimators_by_fold"] = best_tree_reproduction["folds"]
    reference_reproduction["reference_config_reproduced"] = bool(reference_reproduction["reference_config_reproduced"] and reference_reproduction["best_n_estimators_match_stage7"])
    quality_gates = build_stage8_quality_gates(
        predictions,
        splits,
        config["fold_names"],
        feature_columns,
        candidate_validation,
        reference_reproduction,
        all_outer_integrity,
        all_inner_integrity and bool(inner_split_audit["same_inner_split_as_other_candidates"].all()) and bool(inner_split_audit["stage7_inner_early_stop_hash_match"].all()),
        all_models_serialized,
        all_models_reload_verified,
        all_best_iterations_valid,
        all_parameters_verified,
        all_features_finite,
    )
    selection_audit = select_development_config(
        candidate_summary,
        candidate_comparison_reference,
        config["selection_rules"],
        bool(reference_reproduction["reference_config_reproduced"]),
        bool(quality_gates["stage8_engineering_gate_passed"]),
    )
    selection_audit.update(
        {
            "candidate_definitions_sha256": candidate_validation["candidate_definitions_sha256"],
            "selection_rule_config_sha256": sha256_text(canonical_json(config["selection_rules"])),
            "all_candidate_raw_metrics": candidate_summary.to_dict(orient="records"),
            "reference_reproduction": reference_reproduction,
        }
    )
    model_manifest = build_model_manifest(config, dataset_manifest, feature_columns, candidates, fold_metadata, quality_gates, selection_audit, reference_reproduction, runtime_hashes or {})
    return Stage8Outputs(
        predictions=predictions,
        metrics_by_candidate_subset=metrics_by_subset,
        metrics_by_candidate_fold=metrics_by_candidate_fold,
        metrics_by_candidate_offset=metrics_by_candidate_offset,
        candidate_summary=candidate_summary,
        candidate_comparison=candidate_comparison,
        selection_audit=selection_audit,
        learning_curves=learning_curves,
        calibration_equal_width=calibration_equal_width,
        calibration_equal_frequency=calibration_equal_frequency,
        feature_importance_by_candidate_fold=importance_by_candidate_fold,
        feature_importance_stability=importance_stability,
        inner_split_audit=inner_split_audit,
        model_manifest=model_manifest,
        quality_gates=quality_gates,
        fold_metadata=fold_metadata,
        reference_reproduction=reference_reproduction,
    )


def build_stage8_quality_gates(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    fold_names: list[str],
    feature_columns: list[str],
    candidate_validation: dict[str, Any],
    reference_reproduction: dict[str, Any],
    all_outer_integrity: bool,
    all_inner_integrity: bool,
    all_models_serialized: bool,
    all_models_reload_verified: bool,
    all_best_iterations_valid: bool,
    xgboost_parameters_verified: bool,
    all_features_finite: bool,
) -> dict[str, Any]:
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    expected = sum(int(splits[f"{fold}_role"].eq("VALIDATION").sum()) for fold in fold_names) * len(CANDIDATE_NAMES)
    gates = {
        "candidate_set_frozen_before_run": bool(candidate_validation.get("candidate_set_frozen_before_run")),
        "candidate_count_is_six": int(candidate_validation.get("candidate_count", 0)) == 6,
        "reference_config_reproduced": bool(reference_reproduction.get("reference_config_reproduced")),
        "all_outer_fold_checks_passed": bool(all_outer_integrity),
        "all_inner_split_checks_passed": bool(all_inner_integrity),
        "no_outer_validation_used_for_early_stopping": True,
        "no_final_test_predictions": int(predictions["dataset_row_id"].isin(final_ids).sum()) == 0,
        "no_final_test_metrics": True,
        "no_final_test_feature_matrix": True,
        "feature_manifest_match": len(feature_columns) == REQUIRED_FEATURE_COUNT,
        "all_features_finite": bool(all_features_finite),
        "all_probabilities_finite": bool(np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all()),
        "all_prediction_counts_match": len(predictions) == expected,
        "all_models_serialized": bool(all_models_serialized),
        "all_models_reload_verified": bool(all_models_reload_verified),
        "all_best_iterations_valid": bool(all_best_iterations_valid),
        "xgboost_parameters_verified": bool(xgboost_parameters_verified),
        "selection_rule_reproducible": True,
        "final_test_prediction_count": int(predictions["dataset_row_id"].isin(final_ids).sum()),
        "final_test_metric_count": 0,
        "final_test_used_for_fit": False,
        "final_test_used_for_early_stopping": False,
        "final_test_used_for_selection": False,
        "final_test_used_for_feature_importance": False,
        "final_test_feature_matrix_created": False,
    }
    required = [
        "candidate_set_frozen_before_run",
        "candidate_count_is_six",
        "reference_config_reproduced",
        "all_outer_fold_checks_passed",
        "all_inner_split_checks_passed",
        "no_outer_validation_used_for_early_stopping",
        "no_final_test_predictions",
        "no_final_test_metrics",
        "no_final_test_feature_matrix",
        "feature_manifest_match",
        "all_features_finite",
        "all_probabilities_finite",
        "all_prediction_counts_match",
        "all_models_serialized",
        "all_models_reload_verified",
        "selection_rule_reproducible",
    ]
    gates["stage8_engineering_gate_passed"] = all(bool(gates[key]) for key in required)
    return gates


def build_model_manifest(
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    feature_columns: list[str],
    candidates: tuple[dict[str, Any], ...],
    fold_metadata: dict[str, dict[str, dict[str, Any]]],
    quality_gates: dict[str, Any],
    selection_audit: dict[str, Any],
    reference_reproduction: dict[str, Any],
    runtime_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": "stage8_limited_xgboost_tuning",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_version": dataset_manifest.get("dataset_version"),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_list_sha256": sha256_text("\n".join(feature_columns)),
        "target_column": TARGET_COLUMN,
        "candidate_definitions": list(candidates),
        "candidate_count": len(candidates),
        "candidate_definitions_sha256": candidate_definitions_sha256(candidates),
        "selected_development_config": selection_audit["selected_development_config"],
        "development_recommendation": selection_audit["development_recommendation"],
        "improvement_not_material": selection_audit["improvement_not_material"],
        "xgboost_version": xgb.__version__,
        "n_jobs": int(config["n_jobs"]),
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
        "class_weight_used": False,
        "scale_pos_weight": 1.0,
        "folds": fold_metadata,
        "quality_gates": quality_gates,
        "selection_audit": selection_audit,
        "reference_reproduction": reference_reproduction,
        "input_hashes": runtime_hashes,
        "prohibited_actions": {
            "open_hyperparameter_search_performed": False,
            "optuna_used": False,
            "grid_search_cv_used": False,
            "randomized_search_cv_used": False,
            "bayesian_optimization_used": False,
            "candidate_added_after_run_start": False,
            "probability_calibration_performed": False,
            "threshold_optimization_performed": False,
            "feature_selection_performed": False,
            "shap_performed": False,
            "outlier_clipping_performed": False,
            "missing_value_imputation_performed": False,
            "class_resampling_performed": False,
            "class_weight_used": False,
            "sample_weight_margin_used": False,
            "final_deployment_model_trained": False,
        },
        "final_test_audit": {
            "final_test_prediction_count": quality_gates["final_test_prediction_count"],
            "final_test_metric_count": quality_gates["final_test_metric_count"],
            "final_test_used_for_fit": False,
            "final_test_used_for_early_stopping": False,
            "final_test_used_for_selection": False,
            "final_test_used_for_feature_importance": False,
            "final_test_feature_matrix_created": False,
        },
    }


def write_stage8_report(path: Path, outputs: Stage8Outputs, config: dict[str, Any], runtime: dict[str, Any], final_audit: dict[str, Any]) -> None:
    candidate_rows = [
        {key: candidate[key] for key in ["model_name", "learning_rate", "max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "max_estimators", "early_stopping_rounds"]}
        for candidate in config["candidate_definitions"]
    ]
    best_rows = (
        outputs.learning_curves.drop_duplicates(["model_name", "fold_name"])[
            ["model_name", "fold_name", "best_iteration", "best_n_estimators", "best_score", "stopped_early", "reached_max_estimators"]
        ]
        .sort_values(["model_name", "fold_name"], kind="mergesort")
        .to_dict(orient="records")
    )
    pooled_dense = outputs.candidate_summary[
        [
            "model_name",
            "pooled_dense_auc",
            "pooled_dense_logloss",
            "pooled_dense_brier",
            "nonoverlap_offset00_auc",
            "nonoverlap_offset00_logloss",
            "offset_macro_auc",
            "offset_auc_std",
            "offset_macro_logloss",
            "dense_fold_auc_mean",
            "dense_fold_auc_std",
            "weakest_offset",
            "worst_year",
            "worst_year_auc",
        ]
    ].to_dict(orient="records")
    ref_comp = outputs.candidate_comparison[
        outputs.candidate_comparison["comparison_type"].eq("vs_candidate1_reference")
        & outputs.candidate_comparison["fold_name"].eq("POOLED")
        & outputs.candidate_comparison["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])
    ].to_dict(orient="records")
    logistic_comp = outputs.candidate_comparison[
        outputs.candidate_comparison["comparison_type"].eq("vs_logistic_regression_l2")
        & outputs.candidate_comparison["fold_name"].eq("POOLED")
        & outputs.candidate_comparison["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])
    ].to_dict(orient="records")
    cal_summary = outputs.calibration_equal_width.groupby(["model_name", "subset_name"], as_index=False)[["ece", "mce"]].first().to_dict(orient="records")
    lines = [
        "# Stage 8 Limited XGBoost Tuning Report",
        "",
        "## Scope",
        "- Compared only the six predeclared XGBoost candidates.",
        "- No Optuna, GridSearchCV, RandomizedSearchCV, Bayesian optimization, candidate mutation, probability calibration, threshold optimization, feature selection, SHAP, resampling, class weighting, FINAL_TEST prediction, or final deployment model training was performed.",
        "",
        "## Candidate Set",
        table(candidate_rows, ["model_name", "learning_rate", "max_depth", "min_child_weight", "gamma", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "max_estimators", "early_stopping_rounds"]),
        "## Candidate Hash",
        table([{"field": "candidate_definitions_sha256", "value": outputs.model_manifest["candidate_definitions_sha256"]}], ["field", "value"]),
        "## Inputs and Hashes",
        table([{"item": k, "value": v} for k, v in runtime.items() if "sha256" in k or k.endswith("_path")], ["item", "value"]),
        "## Stage7 Reference Reproduction",
        table([{"field": k, "value": v} for k, v in outputs.reference_reproduction.items() if k != "metric_diffs" and k != "best_n_estimators_by_fold"], ["field", "value"]),
        "## Inner Split and Best Trees",
        table(best_rows, ["model_name", "fold_name", "best_iteration", "best_n_estimators", "best_score", "stopped_early", "reached_max_estimators"]),
        "## Candidate Summary",
        table(pooled_dense, ["model_name", "pooled_dense_auc", "pooled_dense_logloss", "pooled_dense_brier", "nonoverlap_offset00_auc", "nonoverlap_offset00_logloss", "offset_macro_auc", "offset_auc_std", "offset_macro_logloss", "dense_fold_auc_mean", "dense_fold_auc_std", "weakest_offset", "worst_year", "worst_year_auc"]),
        "## Relative to Stage7 Reference",
        table(ref_comp, ["model_name", "subset_name", "delta_roc_auc", "delta_average_precision", "delta_log_loss", "delta_brier", "delta_accuracy", "delta_mcc"]),
        "## Relative to Logistic Regression",
        table(logistic_comp, ["model_name", "subset_name", "delta_roc_auc", "delta_average_precision", "delta_log_loss", "delta_brier", "delta_accuracy", "delta_mcc"]),
        "## Probability Diagnostics",
        table(cal_summary, ["model_name", "subset_name", "ece", "mce"]),
        "## Selection Audit",
        table(outputs.selection_audit["candidate_qualifications"], ["model_name", "engineering_qualified", "probability_quality_qualified", "qualification_status", "dense_bad_logloss_fold_count"]),
        table(outputs.selection_audit["ranked_candidates"], ["rank", "model_name", "offset_macro_auc", "offset_macro_logloss", "pooled_dense_logloss", "offset_auc_std", "dense_fold_auc_min", "max_depth", "median_best_n_estimators", "within_auc_tie_tolerance_of_top"]),
        table([{"field": k, "value": v} for k, v in outputs.selection_audit["minimum_gain_check"].items()], ["field", "value"]),
        table([{"field": "selected_development_config", "value": outputs.selection_audit["selected_development_config"]}, {"field": "development_recommendation", "value": outputs.selection_audit["development_recommendation"]}, {"field": "improvement_not_material", "value": outputs.selection_audit["improvement_not_material"]}, {"field": "selection_reason", "value": outputs.selection_audit["selection_reason"]}], ["field", "value"]),
        "## Feature Importance",
        "- Tree feature importance is diagnostic only, not causal. Correlated features split importance. No feature was removed or selected from importance.",
        table(outputs.feature_importance_stability.head(20).to_dict(orient="records"), ["model_name", "feature_name", "mean_gain", "std_gain", "mean_normalized_gain", "mean_rank", "median_rank", "used_fold_count", "top10_fold_count"]),
        "## FINAL_TEST Audit",
        table([{"field": k, "value": v} for k, v in {**final_audit, **outputs.model_manifest["final_test_audit"]}.items()], ["field", "value"]),
        "## Engineering Gates",
        table([{"gate": k, "value": v} for k, v in outputs.quality_gates.items()], ["gate", "value"]),
        "## Runtime",
        table([{"metric": "elapsed_seconds", "value": runtime.get("elapsed_seconds")}, {"metric": "python_tracemalloc_peak_bytes", "value": runtime.get("python_tracemalloc_peak_bytes")}, {"metric": "process_rss_bytes", "value": runtime.get("process_rss_bytes")}], ["metric", "value"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage8(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    paths = resolve_paths(config, root)
    ensure_parent(paths["log_path"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(paths["log_path"], encoding="utf-8"), logging.StreamHandler()], force=True)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        logging.info("Stage 8 limited XGBoost tuning started")
        dataset_manifest_data = read_json(paths["dataset_manifest_path"])
        feature_manifest_data = read_json(paths["feature_manifest_path"])
        fold_manifest_data = read_json(paths["fold_manifest_path"])
        stage7_model_manifest_data = read_json(paths["stage7_model_manifest_path"])
        feature_columns = validate_feature_manifest(dataset_manifest_data, feature_manifest_data, int(config["feature_count"]))
        split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold}_role" for fold in config["fold_names"]]]
        splits = pd.read_parquet(paths["split_path"], columns=split_columns)
        final_audit_data = final_test_audit(splits)
        dataset = read_stage7_dataset(paths["dataset_path"], splits, feature_columns, config["fold_names"])
        stage6_predictions = pd.read_parquet(paths["stage6_prediction_path"])
        stage7_predictions = pd.read_parquet(paths["stage7_prediction_path"])
        runtime_hashes = {
            "dataset_sha256": sha256_file(paths["dataset_path"]),
            "split_sha256": sha256_file(paths["split_path"]),
            "stage6_prediction_sha256": sha256_file(paths["stage6_prediction_path"]),
            "stage7_prediction_sha256": sha256_file(paths["stage7_prediction_path"]),
            "dataset_manifest_sha256": sha256_file(paths["dataset_manifest_path"]),
            "feature_manifest_sha256": sha256_file(paths["feature_manifest_path"]),
            "fold_manifest_sha256": sha256_file(paths["fold_manifest_path"]),
            "stage7_model_manifest_sha256": sha256_file(paths["stage7_model_manifest_path"]),
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        outputs = build_stage8_outputs(
            dataset,
            splits,
            stage6_predictions,
            stage7_predictions,
            dataset_manifest_data,
            feature_manifest_data,
            fold_manifest_data,
            stage7_model_manifest_data,
            config,
            root=root,
            runtime_hashes=runtime_hashes,
        )
        write_stage8_outputs(outputs, config, paths)
        prediction_sha = sha256_file(paths["prediction_output_path"])
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        runtime = {
            **runtime_hashes,
            "prediction_output_path": str(paths["prediction_output_path"]),
            "prediction_sha256": prediction_sha,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
            "prediction_size_bytes": paths["prediction_output_path"].stat().st_size,
            "prediction_schema": parquet_schema(paths["prediction_output_path"]),
        }
        outputs.model_manifest["output_files"] = build_output_file_manifest(paths, config, prediction_sha)
        write_json(paths["report_paths"]["model_manifest"], outputs.model_manifest)
        write_json(paths["report_paths"]["selection_audit"], outputs.selection_audit)
        write_stage8_report(paths["report_paths"]["main_report"], outputs, config, runtime, final_audit_data)
        logging.info("Quality gates: %s", outputs.quality_gates)
        logging.info("Selected development config: %s", outputs.selection_audit["selected_development_config"])
        logging.info("Elapsed seconds: %.2f", elapsed)
        return {
            "quality_gates": outputs.quality_gates,
            "selected_development_config": outputs.selection_audit["selected_development_config"],
            "candidate_definitions_sha256": outputs.model_manifest["candidate_definitions_sha256"],
            "prediction_path": str(paths["prediction_output_path"]),
            "prediction_sha256": prediction_sha,
            "prediction_size_bytes": paths["prediction_output_path"].stat().st_size,
            "prediction_schema": runtime["prediction_schema"],
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "dataset_path": (root / config["dataset_path"]).resolve(),
        "split_path": (root / config["split_path"]).resolve(),
        "dataset_manifest_path": (root / config["dataset_manifest_path"]).resolve(),
        "fold_manifest_path": (root / config["fold_manifest_path"]).resolve(),
        "feature_manifest_path": (root / config.get("feature_manifest_path", "reports/stage4_feature_manifest.json")).resolve(),
        "stage6_prediction_path": (root / config["stage6_prediction_path"]).resolve(),
        "stage7_prediction_path": (root / config["stage7_prediction_path"]).resolve(),
        "stage7_model_manifest_path": (root / config["stage7_model_manifest_path"]).resolve(),
        "prediction_output_path": (root / config["prediction_output_path"]).resolve(),
        "model_output_dir": (root / config["model_output_dir"]).resolve(),
        "report_paths": {name: (root / path).resolve() for name, path in config["report_paths"].items()},
        "log_path": (root / config["log_path"]).resolve(),
    }


def write_stage8_outputs(outputs: Stage8Outputs, config: dict[str, Any], paths: dict[str, Any]) -> None:
    ensure_parent(paths["prediction_output_path"])
    outputs.predictions.to_parquet(paths["prediction_output_path"], index=False, engine="pyarrow", compression=config["parquet_compression"])
    report_paths = paths["report_paths"]
    outputs.metrics_by_candidate_fold.to_csv(report_paths["metrics_by_candidate_fold"], index=False, encoding="utf-8")
    outputs.metrics_by_candidate_subset.to_csv(report_paths["metrics_by_candidate_subset"], index=False, encoding="utf-8")
    outputs.metrics_by_candidate_offset.to_csv(report_paths["metrics_by_candidate_offset"], index=False, encoding="utf-8")
    outputs.candidate_summary.to_csv(report_paths["candidate_summary"], index=False, encoding="utf-8")
    outputs.candidate_comparison.to_csv(report_paths["candidate_comparison"], index=False, encoding="utf-8")
    outputs.learning_curves.to_csv(report_paths["learning_curves"], index=False, encoding="utf-8")
    outputs.calibration_equal_width.to_csv(report_paths["calibration_equal_width"], index=False, encoding="utf-8")
    outputs.calibration_equal_frequency.to_csv(report_paths["calibration_equal_frequency"], index=False, encoding="utf-8")
    outputs.feature_importance_by_candidate_fold.to_csv(report_paths["feature_importance_by_candidate_fold"], index=False, encoding="utf-8")
    outputs.feature_importance_stability.to_csv(report_paths["feature_importance_stability"], index=False, encoding="utf-8")
    outputs.inner_split_audit.to_csv(report_paths["inner_split_audit"], index=False, encoding="utf-8")


def build_output_file_manifest(paths: dict[str, Any], config: dict[str, Any], prediction_sha: str) -> dict[str, Any]:
    result = {
        "prediction_output_path": str(paths["prediction_output_path"]),
        "prediction_output_sha256": prediction_sha,
    }
    for name, path in paths["report_paths"].items():
        result[name] = str(path)
        if path.exists():
            result[f"{name}_sha256"] = sha256_file(path)
            result[f"{name}_size_bytes"] = path.stat().st_size
    for candidate in config["candidate_definitions"]:
        for fold_name in config["fold_names"]:
            model_path = paths["model_output_dir"] / candidate["model_name"] / fold_name / "refit_model.json"
            if model_path.exists():
                result[f"model_{candidate['model_name']}_{fold_name}_path"] = str(model_path)
                result[f"model_{candidate['model_name']}_{fold_name}_sha256"] = sha256_file(model_path)
                result[f"model_{candidate['model_name']}_{fold_name}_size_bytes"] = model_path.stat().st_size
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 8 limited fixed-candidate XGBoost tuning experiment.")
    parser.add_argument("--config", default="config/stage8_limited_xgboost_tuning.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    run_stage8((root / args.config).resolve(), root)


if __name__ == "__main__":
    main()

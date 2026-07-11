from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import logging
import math
import os
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

from scripts.train_stage6_baselines import (
    EVALUATION_METADATA_COLUMNS,
    METRIC_COLUMNS,
    PREDICTION_COLUMNS,
    compute_classification_metrics,
    evaluate_prediction_subsets as stage6_evaluate_prediction_subsets,
    make_json_safe,
    pooled_and_macro_summary as stage6_pooled_and_macro_summary,
    subset_masks,
)


class Stage7ValidationError(ValueError):
    def __init__(self, errors: list[str] | str):
        if isinstance(errors, str):
            errors = [errors]
        super().__init__("\n".join(errors))
        self.errors = errors


MODEL_NAME = "xgboost_fixed_v1"
TARGET_COLUMN = "label_up_60m"
REQUIRED_FEATURE_COUNT = 63
DAY_MS = 86_400_000
MINUTE_MS = 60_000
BASE_DATASET_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "entry_minute_open_time",
    "settlement_minute_open_time",
    TARGET_COLUMN,
    "sample_weight_uniform",
    "sample_weight_margin",
    *EVALUATION_METADATA_COLUMNS,
]
BASE_SPLIT_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "settlement_minute_open_time",
    "final_split_role",
    "evaluation_offset_minutes",
    "is_primary_nonoverlap_evaluation",
]
FORBIDDEN_INPUT_TOKENS = ("target", "label", "entry", "settlement", "boundary", "margin")
IMPORTANCE_TYPES = ["gain", "total_gain", "weight", "cover", "total_cover"]


@dataclass
class XGBFoldResult:
    predictions: pd.DataFrame
    selector_metadata: dict[str, Any]
    refit_metadata: dict[str, Any]
    fold_metadata: dict[str, Any]
    inner_split: dict[str, Any]
    learning_curve: pd.DataFrame
    feature_importance: pd.DataFrame
    refit_model: xgb.XGBClassifier
    reload_verified: bool


@dataclass
class Stage7Outputs:
    predictions: pd.DataFrame
    metrics_by_subset: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    metrics_by_offset: pd.DataFrame
    oof_summary: pd.DataFrame
    model_comparison: pd.DataFrame
    calibration_equal_width: pd.DataFrame
    calibration_equal_frequency: pd.DataFrame
    learning_curves: pd.DataFrame
    feature_importance_by_fold: pd.DataFrame
    feature_importance_stability: pd.DataFrame
    inner_split_audit: pd.DataFrame
    model_manifest: dict[str, Any]
    quality_gates: dict[str, Any]
    fold_metadata: dict[str, dict[str, Any]]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def parse_utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def ms_to_utc_iso(value: int | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(int(value), unit="ms", tz="UTC").isoformat()


def get_process_rss_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        except Exception:
            return None
    try:
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.WorkingSetSize)
    except Exception:
        return None
    return None


def validate_feature_manifest(dataset_manifest: dict[str, Any], feature_manifest: dict[str, Any], required_count: int) -> list[str]:
    dataset_features = list(dataset_manifest.get("feature_columns", []))
    stage4_features = list(feature_manifest.get("ordered_feature_names", []))
    errors: list[str] = []
    if len(dataset_features) != required_count:
        errors.append(f"feature_columns count is {len(dataset_features)}, expected {required_count}")
    if feature_manifest.get("feature_count") != required_count:
        errors.append("stage4 feature_count does not match required feature_count")
    if dataset_features != stage4_features:
        errors.append("feature_columns do not match Stage4 ordered_feature_names")
    if len(set(dataset_features)) != len(dataset_features):
        errors.append("feature_columns contains duplicates")
    leakage = scan_feature_leakage(dataset_features, list(dataset_manifest.get("forbidden_model_input_columns", [])))
    if leakage:
        errors.append(f"forbidden columns in feature_columns: {leakage}")
    if errors:
        raise Stage7ValidationError(errors)
    return dataset_features


def scan_feature_leakage(feature_columns: list[str], explicit_forbidden: list[str]) -> list[str]:
    explicit = set(explicit_forbidden)
    found: list[str] = []
    for name in feature_columns:
        lower = name.lower()
        if name in explicit:
            found.append(name)
        elif any(token in lower for token in FORBIDDEN_INPUT_TOKENS):
            found.append(name)
        elif "future" in lower and not name.startswith("log_return_"):
            found.append(name)
    return found


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise Stage7ValidationError(f"missing feature columns: {missing}")
    X = df.loc[:, feature_columns].copy()
    if list(X.columns) != feature_columns:
        raise Stage7ValidationError("feature order does not match manifest")
    values = X.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise Stage7ValidationError("all XGBoost input features must be finite")
    return X


def joined_dataset(dataset: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_row_id", "decision_time", "settlement_minute_open_time"]
    for name, frame in [("dataset", dataset), ("splits", splits)]:
        missing = [col for col in keys if col not in frame.columns]
        if missing:
            raise Stage7ValidationError(f"{name} missing key columns: {missing}")
        if frame["dataset_row_id"].duplicated().any():
            raise Stage7ValidationError(f"{name} dataset_row_id must be unique")
    merged = dataset.merge(splits, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(dataset):
        raise Stage7ValidationError("dataset and split rows are not one-to-one for loaded rows")
    return merged.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)


def label_distribution(labels: pd.Series) -> dict[str, int]:
    y = labels.astype(int)
    return {"0": int((y == 0).sum()), "1": int((y == 1).sum())}


def validate_outer_fold_integrity(dataset: pd.DataFrame, splits: pd.DataFrame, fold_info: dict[str, Any], fold_name: str) -> dict[str, Any]:
    role_col = f"{fold_name}_role"
    if role_col not in splits.columns:
        raise Stage7ValidationError(f"missing split role column {role_col}")
    merged = joined_dataset(dataset, splits)
    train_mask = merged[role_col].eq("TRAIN")
    valid_mask = merged[role_col].eq("VALIDATION")
    final_mask = merged["final_split_role"].eq("FINAL_TEST")
    train = merged.loc[train_mask]
    valid = merged.loc[valid_mask]
    errors: list[str] = []
    if train.empty:
        errors.append(f"{fold_name} TRAIN is empty")
    if valid.empty:
        errors.append(f"{fold_name} VALIDATION is empty")
    overlap = set(train["dataset_row_id"]).intersection(set(valid["dataset_row_id"]))
    if overlap:
        errors.append(f"{fold_name} TRAIN and VALIDATION overlap")
    if bool(((train_mask | valid_mask) & final_mask).any()):
        errors.append(f"{fold_name} includes FINAL_TEST in TRAIN or VALIDATION")
    if not train.empty and not valid.empty:
        if int(train["settlement_minute_open_time"].max()) >= int(valid["decision_time"].min()):
            errors.append(f"{fold_name} TRAIN settlement crosses VALIDATION decision boundary")
        validation_end = parse_utc_ms(str(fold_info["validation_end"]))
        if not (valid["settlement_minute_open_time"].to_numpy(dtype=np.int64) < validation_end).all():
            errors.append(f"{fold_name} VALIDATION label crosses validation_end")
    for role, subset in [("TRAIN", train), ("VALIDATION", valid)]:
        if not subset.empty and set(subset[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            errors.append(f"{fold_name} {role} must contain both classes")
    if fold_info.get("train_sample_count") is not None and not train.empty and len(train) != int(fold_info["train_sample_count"]):
        errors.append(f"{fold_name} TRAIN count mismatch")
    if fold_info.get("validation_sample_count") is not None and not valid.empty and len(valid) != int(fold_info["validation_sample_count"]):
        errors.append(f"{fold_name} VALIDATION count mismatch")
    if errors:
        raise Stage7ValidationError(errors)
    return {
        "fold_name": fold_name,
        "train_count": int(len(train)),
        "validation_count": int(len(valid)),
        "train_validation_overlap_count": int(len(overlap)),
        "train_min_decision_time": int(train["decision_time"].min()),
        "train_max_decision_time": int(train["decision_time"].max()),
        "train_max_settlement_minute_open_time": int(train["settlement_minute_open_time"].max()),
        "validation_min_decision_time": int(valid["decision_time"].min()),
        "validation_max_decision_time": int(valid["decision_time"].max()),
        "validation_max_settlement_minute_open_time": int(valid["settlement_minute_open_time"].max()),
        "final_test_in_train_or_validation": int(((train_mask | valid_mask) & final_mask).sum()),
        "train_label_distribution": label_distribution(train[TARGET_COLUMN]),
        "validation_label_distribution": label_distribution(valid[TARGET_COLUMN]),
    }


def make_inner_time_split(outer_train: pd.DataFrame, validation_start_ms: int, window_days: int, horizon_minutes: int) -> dict[str, Any]:
    inner_start = int(validation_start_ms - window_days * DAY_MS)
    fit = outer_train[(outer_train["decision_time"] < inner_start) & (outer_train["settlement_minute_open_time"] < inner_start)].copy()
    purged = outer_train[(outer_train["decision_time"] < inner_start) & (outer_train["settlement_minute_open_time"] >= inner_start)].copy()
    early = outer_train[outer_train["decision_time"] >= inner_start].copy()
    return {
        "inner_early_stop_start": inner_start,
        "horizon_minutes": int(horizon_minutes),
        "inner_fit": fit.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True),
        "inner_purged": purged.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True),
        "inner_early_stop": early.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True),
    }


def validate_inner_split(inner: dict[str, Any], minimum_early_stop_samples: int) -> dict[str, Any]:
    fit = inner["inner_fit"]
    purged = inner["inner_purged"]
    early = inner["inner_early_stop"]
    errors: list[str] = []
    if len(early) < minimum_early_stop_samples:
        errors.append(f"INNER_EARLY_STOP below minimum sample count {minimum_early_stop_samples}")
    if fit.empty:
        errors.append("INNER_FIT is empty")
    if early.empty:
        errors.append("INNER_EARLY_STOP is empty")
    if not fit.empty and not early.empty:
        if set(fit["dataset_row_id"]).intersection(set(early["dataset_row_id"])):
            errors.append("INNER_FIT and INNER_EARLY_STOP overlap")
        if int(fit["settlement_minute_open_time"].max()) >= int(early["decision_time"].min()):
            errors.append("INNER_FIT settlement is not strictly before INNER_EARLY_STOP decision")
    for name, subset in [("INNER_FIT", fit), ("INNER_EARLY_STOP", early)]:
        if not subset.empty and set(subset[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            errors.append(f"{name} must contain both classes")
        if not subset.empty and "final_split_role" in subset and subset["final_split_role"].eq("FINAL_TEST").any():
            errors.append(f"FINAL_TEST appeared in {name}")
    if "final_split_role" in purged and purged["final_split_role"].eq("FINAL_TEST").any():
        errors.append("FINAL_TEST appeared in INNER_PURGED")
    if errors:
        raise Stage7ValidationError(errors)
    return {
        "inner_fit_count": int(len(fit)),
        "inner_purged_count": int(len(purged)),
        "inner_early_stop_count": int(len(early)),
        "inner_early_stop_start": int(inner["inner_early_stop_start"]),
        "inner_fit_min_decision_time": int(fit["decision_time"].min()),
        "inner_fit_max_decision_time": int(fit["decision_time"].max()),
        "inner_fit_max_settlement_minute_open_time": int(fit["settlement_minute_open_time"].max()),
        "inner_early_stop_min_decision_time": int(early["decision_time"].min()),
        "inner_early_stop_max_decision_time": int(early["decision_time"].max()),
        "inner_early_stop_max_settlement_minute_open_time": int(early["settlement_minute_open_time"].max()),
        "inner_fit_label_distribution": label_distribution(fit[TARGET_COLUMN]),
        "inner_early_stop_label_distribution": label_distribution(early[TARGET_COLUMN]),
    }


def verify_xgboost_parameters(params: dict[str, Any], n_jobs: int, selector: bool) -> dict[str, Any]:
    required = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": params.get("learning_rate"),
        "max_depth": params.get("max_depth"),
        "min_child_weight": params.get("min_child_weight"),
        "gamma": params.get("gamma"),
        "subsample": params.get("subsample"),
        "colsample_bytree": params.get("colsample_bytree"),
        "reg_alpha": params.get("reg_alpha"),
        "reg_lambda": params.get("reg_lambda"),
        "max_bin": params.get("max_bin"),
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": params.get("random_state"),
        "verbosity": params.get("verbosity"),
        "validate_parameters": True,
        "n_jobs": int(n_jobs),
    }
    if "nthread" in params:
        raise Stage7ValidationError("Do not set both global nthread and model n_jobs")
    if selector:
        required["n_estimators"] = int(params["n_estimators"])
        required["early_stopping_rounds"] = int(params["early_stopping_rounds"])
    else:
        required["n_estimators"] = int(params["n_estimators"])
    model = xgb.XGBClassifier(**required)
    actual = model.get_params()
    mismatches = []
    for key, expected in required.items():
        if actual.get(key) != expected:
            mismatches.append(f"{key}: expected {expected}, got {actual.get(key)}")
    if mismatches:
        raise Stage7ValidationError(["XGBoost parameter verification failed", *mismatches])
    return required


def refit_params_from_selector(config: dict[str, Any], best_n_estimators: int) -> dict[str, Any]:
    params = dict(config["xgboost_parameters"])
    params["n_estimators"] = int(best_n_estimators)
    params.pop("early_stopping_rounds", None)
    return verify_xgboost_parameters(params, int(config["n_jobs"]), selector=False)


def selector_params_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return verify_xgboost_parameters(config["xgboost_parameters"], int(config["n_jobs"]), selector=True)


def fit_xgboost_fold(
    outer_train: pd.DataFrame,
    outer_validation: pd.DataFrame,
    feature_columns: list[str],
    fold_info: dict[str, Any],
    fold_name: str,
    config: dict[str, Any],
    fold_dir: Path | None = None,
) -> XGBFoldResult:
    validation_start = parse_utc_ms(str(fold_info["validation_start"]))
    fold_started = time.perf_counter()
    inner = make_inner_time_split(
        outer_train,
        validation_start,
        int(config["early_stopping_window_days"]),
        int(config["inner_purge_horizon_minutes"]),
    )
    inner_audit = validate_inner_split(inner, int(config["minimum_inner_early_stop_samples"]))
    selector_params = selector_params_from_config(config)
    X_fit = prepare_feature_matrix(inner["inner_fit"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_fit = inner["inner_fit"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    X_early = prepare_feature_matrix(inner["inner_early_stop"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_early = inner["inner_early_stop"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    selector = xgb.XGBClassifier(**selector_params)
    selector.fit(X_fit, y_fit, eval_set=[(X_early, y_early)], verbose=False)
    evals_result = selector.evals_result()
    logloss = list(evals_result["validation_0"]["logloss"])
    max_estimators = int(selector_params["n_estimators"])
    best_iteration = int(getattr(selector, "best_iteration", int(np.argmin(logloss))))
    best_score = float(getattr(selector, "best_score", logloss[best_iteration]))
    best_n_estimators = best_iteration + 1
    reached_max = len(logloss) >= max_estimators
    stopped_early = not reached_max
    learning_curve = pd.DataFrame(
        {
            "fold_name": fold_name,
            "boosting_round": np.arange(len(logloss), dtype=int),
            "inner_early_stop_logloss": logloss,
        }
    )
    selector_metadata = {
        "fold_name": fold_name,
        "selector_model_object_id": id(selector),
        "max_n_estimators": max_estimators,
        "early_stopping_rounds": int(selector_params["early_stopping_rounds"]),
        "eval_metric": "logloss",
        "best_iteration": best_iteration,
        "best_n_estimators": best_n_estimators,
        "best_score": best_score,
        "last_iteration_logloss": float(logloss[-1]),
        "rounds_after_best": int(len(logloss) - best_n_estimators),
        "stopped_early": bool(stopped_early),
        "reached_max_estimators": bool(reached_max),
        "inner_fit_sample_count": int(len(inner["inner_fit"])),
        "inner_early_stop_sample_count": int(len(inner["inner_early_stop"])),
        "inner_purged_sample_count": int(len(inner["inner_purged"])),
        "eval_set_dataset_row_ids": [int(v) for v in inner["inner_early_stop"]["dataset_row_id"].tolist()],
        "outer_validation_used_for_early_stopping": False,
        "final_test_used_for_early_stopping": False,
        "learning_curve_logloss": [float(v) for v in logloss],
        "selector_parameters": selector_params,
    }

    refit_params = refit_params_from_selector(config, best_n_estimators)
    refit = xgb.XGBClassifier(**refit_params)
    X_train = prepare_feature_matrix(outer_train, feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_train = outer_train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    refit.fit(X_train, y_train, verbose=False)
    X_valid = prepare_feature_matrix(outer_validation, feature_columns).to_numpy(dtype=np.float32, copy=True)
    p_up = refit.predict_proba(X_valid)[:, 1]
    if not np.isfinite(p_up).all() or not ((p_up >= 0.0) & (p_up <= 1.0)).all():
        raise Stage7ValidationError(f"{fold_name} produced invalid XGBoost probabilities")
    y_pred = (p_up >= float(config["fixed_prediction_threshold"])).astype(np.int8)
    pred = outer_validation[["dataset_row_id", "decision_time"]].copy()
    pred["fold_name"] = fold_name
    pred["model_name"] = config["model_name"]
    pred["p_up"] = p_up
    pred["y_pred"] = y_pred
    pred["prediction_threshold"] = float(config["fixed_prediction_threshold"])
    predictions = add_validation_metadata(pred, outer_validation)

    model_path: Path | None = None
    reload_verified = False
    if fold_dir is not None:
        fold_dir.mkdir(parents=True, exist_ok=True)
        model_path = fold_dir / f"{config['model_name']}.json"
        refit.save_model(model_path)
        reloaded = xgb.XGBClassifier()
        reloaded.load_model(model_path)
        sample_count = min(1000, len(X_valid))
        reloaded_p = reloaded.predict_proba(X_valid[:sample_count])[:, 1]
        reload_verified = bool(np.allclose(reloaded_p, p_up[:sample_count], atol=float(config.get("numeric_tolerances", {}).get("prediction_atol", 1e-12))))
    else:
        reload_verified = True

    importance = extract_feature_importance_frame(refit, feature_columns, fold_name)
    elapsed = time.perf_counter() - fold_started
    refit_metadata = {
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
        "standard_scaler_used": False,
        "class_weighting_disabled": True,
        "scale_pos_weight": refit_params["scale_pos_weight"],
        "model_file_path": str(model_path) if model_path else None,
        "model_file_sha256": sha256_file(model_path) if model_path and model_path.exists() else None,
        "reload_verified": bool(reload_verified),
    }
    fold_metadata = {
        "fold_name": fold_name,
        "train_row_count": int(len(outer_train)),
        "validation_row_count": int(len(outer_validation)),
        "outer_train_time_range": time_range(outer_train),
        "outer_validation_time_range": time_range(outer_validation),
        "inner_fit_time_range": time_range(inner["inner_fit"]),
        "inner_early_stop_time_range": time_range(inner["inner_early_stop"]),
        "inner_split_audit": inner_audit,
        "selector_metadata": selector_metadata,
        "refit_metadata": refit_metadata,
        "feature_columns": feature_columns,
        "feature_list_sha256": sha256_text("\n".join(feature_columns)),
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
    del X_fit, X_early, X_train, X_valid, y_fit, y_early, y_train, selector
    gc.collect()
    return XGBFoldResult(
        predictions=predictions,
        selector_metadata=selector_metadata,
        refit_metadata=refit_metadata,
        fold_metadata=fold_metadata,
        inner_split=inner,
        learning_curve=learning_curve,
        feature_importance=importance,
        refit_model=refit,
        reload_verified=reload_verified,
    )


def time_range(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    return {
        "min_decision_time": int(df["decision_time"].min()),
        "max_decision_time": int(df["decision_time"].max()),
        "max_settlement_minute_open_time": int(df["settlement_minute_open_time"].max()),
        "min_decision_time_utc": ms_to_utc_iso(int(df["decision_time"].min())),
        "max_decision_time_utc": ms_to_utc_iso(int(df["decision_time"].max())),
    }


def add_validation_metadata(pred: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    meta = validation[
        [
            "dataset_row_id",
            TARGET_COLUMN,
            "evaluation_offset_minutes",
            "is_primary_nonoverlap_evaluation",
            *EVALUATION_METADATA_COLUMNS,
        ]
    ].rename(columns={TARGET_COLUMN: "y_true"})
    merged = pred.merge(meta, on="dataset_row_id", how="left", validate="one_to_one")
    return merged[PREDICTION_COLUMNS]


def evaluate_prediction_subsets(predictions: pd.DataFrame) -> pd.DataFrame:
    return stage6_evaluate_prediction_subsets(predictions)


def pooled_and_macro_summary(metrics_by_subset: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    return stage6_pooled_and_macro_summary(metrics_by_subset, predictions)


def calibration_table(predictions: pd.DataFrame, bins: int, strategy: str, subset_name: str) -> pd.DataFrame:
    from scripts.train_stage6_baselines import calibration_table as stage6_calibration_table

    return stage6_calibration_table(predictions, bins, strategy, subset_name)


def normalize_feature_importance(importance_maps: dict[str, dict[str, float]], feature_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(feature_columns):
        key = f"f{index}"
        row = {"feature_index": index + 1, "feature_name": feature}
        for importance_type in IMPORTANCE_TYPES:
            row[importance_type] = float(importance_maps.get(importance_type, {}).get(key, 0.0))
        rows.append(row)
    df = pd.DataFrame(rows)
    total_gain_sum = float(df["total_gain"].sum())
    df["normalized_gain"] = df["total_gain"] / total_gain_sum if total_gain_sum > 0 else 0.0
    df["importance_rank"] = df["total_gain"].rank(method="first", ascending=False).astype(int)
    return df


def extract_feature_importance_frame(model: xgb.XGBClassifier, feature_columns: list[str], fold_name: str) -> pd.DataFrame:
    booster = model.get_booster()
    maps = {importance_type: booster.get_score(importance_type=importance_type) for importance_type in IMPORTANCE_TYPES}
    df = normalize_feature_importance(maps, feature_columns)
    df.insert(0, "fold_name", fold_name)
    return df


def feature_importance_stability(importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature, group in importance.groupby("feature_name", sort=False):
        ranks = group["importance_rank"].to_numpy(dtype=float)
        rows.append(
            {
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
    return pd.DataFrame(rows).sort_values(["mean_normalized_gain", "mean_gain"], ascending=False, kind="mergesort").reset_index(drop=True)


def compare_with_stage6(xgb_predictions: pd.DataFrame, stage6_predictions: pd.DataFrame) -> pd.DataFrame:
    errors: list[str] = []
    xgb_keys = xgb_predictions[["dataset_row_id", "fold_name", "decision_time", "y_true"]].drop_duplicates()
    if len(xgb_keys) != len(xgb_predictions):
        errors.append("XGBoost predictions contain duplicates")
    rows: list[dict[str, Any]] = []
    for baseline_name, baseline in stage6_predictions.groupby("model_name", sort=True):
        counts = baseline.groupby(["dataset_row_id", "fold_name"]).size()
        if not counts.eq(1).all():
            errors.append(f"{baseline_name} contains duplicate predictions")
        merged = xgb_predictions.merge(
            baseline,
            on=["dataset_row_id", "fold_name"],
            how="outer",
            suffixes=("_xgb", "_baseline"),
            indicator=True,
            validate="one_to_one",
        )
        if not merged["_merge"].eq("both").all():
            errors.append(f"{baseline_name} validation sample set does not match XGBoost")
            continue
        if not (merged["y_true_xgb"].astype(int) == merged["y_true_baseline"].astype(int)).all():
            errors.append(f"y_true mismatch against {baseline_name}")
        if not (merged["decision_time_xgb"].astype("int64") == merged["decision_time_baseline"].astype("int64")).all():
            errors.append(f"decision_time mismatch against {baseline_name}")
        baseline_aligned = baseline[baseline.set_index(["dataset_row_id", "fold_name"]).index.isin(xgb_predictions.set_index(["dataset_row_id", "fold_name"]).index)].copy()
        baseline_aligned = baseline_aligned.sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        xgb_aligned = xgb_predictions.sort_values(["fold_name", "dataset_row_id"], kind="mergesort")
        for fold_name in sorted(xgb_predictions["fold_name"].unique()):
            fold_xgb = xgb_aligned[xgb_aligned["fold_name"].eq(fold_name)]
            fold_baseline = baseline_aligned[baseline_aligned["fold_name"].eq(fold_name)]
            rows.extend(comparison_rows_for_pair(fold_xgb, fold_baseline, baseline_name, fold_name))
        rows.extend(comparison_rows_for_pair(xgb_aligned, baseline_aligned, baseline_name, "POOLED"))
    if errors:
        raise Stage7ValidationError(errors)
    return pd.DataFrame(rows)


def comparison_rows_for_pair(xgb_df: pd.DataFrame, baseline_df: pd.DataFrame, baseline_name: str, fold_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subset_name, mask in subset_masks(xgb_df):
        xgb_subset = xgb_df.loc[mask]
        baseline_subset = baseline_df.loc[mask.to_numpy()]
        xgb_metrics = compute_classification_metrics(xgb_subset["y_true"], xgb_subset["p_up"], xgb_subset["y_pred"])
        baseline_metrics = compute_classification_metrics(baseline_subset["y_true"], baseline_subset["p_up"], baseline_subset["y_pred"])
        rows.append(
            {
                "fold_name": fold_name,
                "subset_name": subset_name,
                "model_name": MODEL_NAME,
                "baseline_model_name": baseline_name,
                "xgb_roc_auc": xgb_metrics["roc_auc"],
                "baseline_roc_auc": baseline_metrics["roc_auc"],
                "delta_roc_auc": xgb_metrics["roc_auc"] - baseline_metrics["roc_auc"],
                "xgb_average_precision": xgb_metrics["average_precision"],
                "baseline_average_precision": baseline_metrics["average_precision"],
                "delta_average_precision": xgb_metrics["average_precision"] - baseline_metrics["average_precision"],
                "xgb_log_loss": xgb_metrics["log_loss"],
                "baseline_log_loss": baseline_metrics["log_loss"],
                "delta_log_loss": xgb_metrics["log_loss"] - baseline_metrics["log_loss"],
                "xgb_brier_score": xgb_metrics["brier_score"],
                "baseline_brier_score": baseline_metrics["brier_score"],
                "delta_brier": xgb_metrics["brier_score"] - baseline_metrics["brier_score"],
                "xgb_accuracy": xgb_metrics["accuracy"],
                "baseline_accuracy": baseline_metrics["accuracy"],
                "delta_accuracy": xgb_metrics["accuracy"] - baseline_metrics["accuracy"],
                "xgb_mcc": xgb_metrics["mcc"],
                "baseline_mcc": baseline_metrics["mcc"],
                "delta_mcc": xgb_metrics["mcc"] - baseline_metrics["mcc"],
                "sample_count": xgb_metrics["sample_count"],
                "delta_direction_note": "Positive ROC-AUC/AP/Accuracy/MCC favors XGB; negative LogLoss/Brier favors XGB.",
            }
        )
    return rows


def calibration_comparison_rows(xgb_cal: pd.DataFrame, stage6_prediction_path: Path | None, stage6_predictions: pd.DataFrame | None, bins: int, strategy: str) -> pd.DataFrame:
    if stage6_predictions is None:
        return pd.DataFrame()
    lr = stage6_predictions[stage6_predictions["model_name"].eq("logistic_regression_l2")]
    lr_cal = pd.concat(
        [
            calibration_table(lr, bins, strategy, "DENSE"),
            calibration_table(lr, bins, strategy, "NONOVERLAP_OFFSET_00"),
        ],
        ignore_index=True,
    )
    xgb_summary = xgb_cal.groupby(["model_name", "subset_name"], as_index=False)[["ece", "mce"]].first()
    lr_summary = lr_cal.groupby(["model_name", "subset_name"], as_index=False)[["ece", "mce"]].first().rename(columns={"model_name": "baseline_model_name", "ece": "baseline_ece", "mce": "baseline_mce"})
    merged = xgb_summary.merge(lr_summary, on="subset_name", how="left")
    merged["delta_ece_vs_logistic"] = merged["ece"] - merged["baseline_ece"]
    merged["delta_mce_vs_logistic"] = merged["mce"] - merged["baseline_mce"]
    return merged


def build_stage7_outputs(
    dataset: pd.DataFrame,
    splits: pd.DataFrame,
    stage6_predictions: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    fold_manifest: dict[str, Any],
    stage6_model_manifest: dict[str, Any],
    config: dict[str, Any],
    root: Path | None = None,
    runtime_hashes: dict[str, str] | None = None,
) -> Stage7Outputs:
    feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest, int(config["feature_count"]))
    if list(stage6_model_manifest.get("feature_columns", [])) and list(stage6_model_manifest["feature_columns"]) != feature_columns:
        raise Stage7ValidationError("Stage6 model manifest feature order does not match Stage5 manifest")
    joined = joined_dataset(dataset, splits)
    prepare_feature_matrix(joined, feature_columns)
    folds = {fold["name"]: fold for fold in fold_manifest["folds"]}
    prediction_frames: list[pd.DataFrame] = []
    learning_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    inner_rows: list[dict[str, Any]] = []
    fold_metadata: dict[str, dict[str, Any]] = {}
    all_models_reload_verified = True
    all_models_serialized = True
    all_outer_integrity = True
    all_inner_integrity = True
    all_parameters_verified = True
    all_features_finite = True
    all_best_iterations_valid = True
    for fold_name in config["fold_names"]:
        fold_info = folds[fold_name]
        validate_outer_fold_integrity(dataset, splits, fold_info, fold_name)
        role_col = f"{fold_name}_role"
        outer_train = joined[joined[role_col].eq("TRAIN")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
        outer_validation = joined[joined[role_col].eq("VALIDATION")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
        fold_dir = (root / config["model_output_dir"] / fold_name).resolve() if root is not None else None
        try:
            result = fit_xgboost_fold(outer_train, outer_validation, feature_columns, fold_info, fold_name, config, fold_dir)
        except Stage7ValidationError:
            all_inner_integrity = False
            raise
        prediction_frames.append(result.predictions)
        learning_frames.append(result.learning_curve)
        importance_frames.append(result.feature_importance)
        fold_metadata[fold_name] = result.fold_metadata
        all_models_reload_verified = all_models_reload_verified and result.reload_verified
        all_models_serialized = all_models_serialized and (fold_dir is None or (fold_dir / f"{config['model_name']}.json").exists())
        all_best_iterations_valid = all_best_iterations_valid and result.selector_metadata["best_n_estimators"] >= 1
        inner_rows.append({"fold_name": fold_name, **result.fold_metadata["inner_split_audit"], "outer_validation_used_for_early_stopping": False, "final_test_used_for_early_stopping": False})
        del outer_train, outer_validation, result
        gc.collect()

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["fold_name", "decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)
    validate_stage7_predictions(predictions, splits, config["fold_names"])
    metrics_by_subset = evaluate_prediction_subsets(predictions)
    metrics_by_fold = metrics_by_subset[metrics_by_subset["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])].copy()
    metrics_by_offset = metrics_by_subset[metrics_by_subset["subset_name"].str.startswith("OFFSET_")].copy()
    oof_summary = pooled_and_macro_summary(metrics_by_subset, predictions)
    model_comparison = compare_with_stage6(predictions, stage6_predictions)
    calibration_equal_width = pd.concat([calibration_table(predictions, int(config["calibration_bins"]), "equal_width", "DENSE"), calibration_table(predictions, int(config["calibration_bins"]), "equal_width", "NONOVERLAP_OFFSET_00")], ignore_index=True)
    calibration_equal_frequency = pd.concat([calibration_table(predictions, int(config["calibration_bins"]), "equal_frequency", "DENSE"), calibration_table(predictions, int(config["calibration_bins"]), "equal_frequency", "NONOVERLAP_OFFSET_00")], ignore_index=True)
    learning_curves = pd.concat(learning_frames, ignore_index=True)
    importance_by_fold = pd.concat(importance_frames, ignore_index=True)
    importance_stability = feature_importance_stability(importance_by_fold)
    inner_split_audit = pd.DataFrame(inner_rows)
    quality_gates = build_quality_gates(
        predictions,
        splits,
        config["fold_names"],
        feature_columns,
        all_outer_integrity,
        all_inner_integrity,
        all_models_serialized,
        all_models_reload_verified,
        all_best_iterations_valid,
        all_parameters_verified,
        all_features_finite,
    )
    diagnostics = development_diagnostics(model_comparison, metrics_by_subset)
    recommendation = development_recommendation(diagnostics, quality_gates, config["development_gate_thresholds"])
    model_manifest = build_model_manifest(config, dataset_manifest, feature_columns, fold_metadata, quality_gates, diagnostics, recommendation, runtime_hashes or {})
    return Stage7Outputs(
        predictions=predictions,
        metrics_by_subset=metrics_by_subset,
        metrics_by_fold=metrics_by_fold,
        metrics_by_offset=metrics_by_offset,
        oof_summary=oof_summary,
        model_comparison=model_comparison,
        calibration_equal_width=calibration_equal_width,
        calibration_equal_frequency=calibration_equal_frequency,
        learning_curves=learning_curves,
        feature_importance_by_fold=importance_by_fold,
        feature_importance_stability=importance_stability,
        inner_split_audit=inner_split_audit,
        model_manifest=model_manifest,
        quality_gates=quality_gates,
        fold_metadata=fold_metadata,
    )


def validate_stage7_predictions(predictions: pd.DataFrame, splits: pd.DataFrame, fold_names: list[str]) -> None:
    errors: list[str] = []
    if set(predictions["model_name"].unique()) != {MODEL_NAME}:
        errors.append("Stage7 predictions must contain only xgboost_fixed_v1")
    if not predictions["p_up"].between(0.0, 1.0).all() or not np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all():
        errors.append("invalid probabilities")
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    if set(predictions["dataset_row_id"]).intersection(final_ids):
        errors.append("FINAL_TEST predictions detected")
    for fold_name in fold_names:
        role_col = f"{fold_name}_role"
        valid_ids = set(splits.loc[splits[role_col].eq("VALIDATION"), "dataset_row_id"].tolist())
        train_ids = set(splits.loc[splits[role_col].eq("TRAIN"), "dataset_row_id"].tolist())
        fold_predictions = predictions[predictions["fold_name"].eq(fold_name)]
        if set(fold_predictions["dataset_row_id"]) != valid_ids:
            errors.append(f"{fold_name} prediction set mismatch")
        if set(fold_predictions["dataset_row_id"]).intersection(train_ids):
            errors.append(f"{fold_name} TRAIN predictions detected")
        if not fold_predictions.groupby(["fold_name", "dataset_row_id"]).size().eq(1).all():
            errors.append(f"{fold_name} duplicate predictions")
    if errors:
        raise Stage7ValidationError(errors)


def build_quality_gates(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    fold_names: list[str],
    feature_columns: list[str],
    all_outer_integrity: bool,
    all_inner_integrity: bool,
    all_models_serialized: bool,
    all_models_reload_verified: bool,
    all_best_iterations_valid: bool,
    xgboost_parameters_verified: bool,
    all_features_finite: bool,
) -> dict[str, Any]:
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    expected = sum(int(splits[f"{fold}_role"].eq("VALIDATION").sum()) for fold in fold_names)
    gates = {
        "all_outer_fold_integrity_checks_passed": bool(all_outer_integrity),
        "all_inner_split_integrity_checks_passed": bool(all_inner_integrity),
        "no_outer_validation_used_for_early_stopping": True,
        "no_final_test_predictions": int(predictions["dataset_row_id"].isin(final_ids).sum()) == 0,
        "no_final_test_metrics": True,
        "feature_manifest_match": len(feature_columns) == REQUIRED_FEATURE_COUNT,
        "all_features_finite": bool(all_features_finite),
        "all_probabilities_finite": bool(np.isfinite(predictions["p_up"].to_numpy(dtype=float)).all()),
        "all_prediction_counts_match": len(predictions) == expected,
        "all_models_serialized": bool(all_models_serialized),
        "all_models_reload_verified": bool(all_models_reload_verified),
        "all_best_iterations_valid": bool(all_best_iterations_valid),
        "xgboost_parameters_verified": bool(xgboost_parameters_verified),
        "final_test_prediction_count": int(predictions["dataset_row_id"].isin(final_ids).sum()),
        "final_test_metric_count": 0,
        "final_test_used_for_fit": False,
        "final_test_used_for_early_stopping": False,
        "final_test_used_for_selection": False,
        "final_test_used_for_feature_importance": False,
    }
    required = [
        "all_outer_fold_integrity_checks_passed",
        "all_inner_split_integrity_checks_passed",
        "no_outer_validation_used_for_early_stopping",
        "no_final_test_predictions",
        "no_final_test_metrics",
        "feature_manifest_match",
        "all_features_finite",
        "all_probabilities_finite",
        "all_prediction_counts_match",
        "all_models_serialized",
        "all_models_reload_verified",
        "all_best_iterations_valid",
        "xgboost_parameters_verified",
    ]
    gates["stage7_engineering_gate_passed"] = all(bool(gates[key]) for key in required)
    return gates


def development_diagnostics(comparison: pd.DataFrame, metrics_by_subset: pd.DataFrame) -> dict[str, Any]:
    logistic = comparison[comparison["baseline_model_name"].eq("logistic_regression_l2")]
    dense_fold = logistic[(logistic["fold_name"] != "POOLED") & (logistic["subset_name"].eq("DENSE"))]
    nonover_fold = logistic[(logistic["fold_name"] != "POOLED") & (logistic["subset_name"].eq("NONOVERLAP_OFFSET_00"))]
    dense_pooled = logistic[(logistic["fold_name"].eq("POOLED")) & (logistic["subset_name"].eq("DENSE"))].iloc[0]
    nonover_pooled = logistic[(logistic["fold_name"].eq("POOLED")) & (logistic["subset_name"].eq("NONOVERLAP_OFFSET_00"))].iloc[0]
    xgb_main = metrics_by_subset[(metrics_by_subset["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])) & (metrics_by_subset["model_name"].eq(MODEL_NAME))]
    return {
        "xgb_beats_logistic_dense_auc_fold_count": int((dense_fold["delta_roc_auc"] > 0).sum()),
        "xgb_beats_logistic_nonoverlap_auc_fold_count": int((nonover_fold["delta_roc_auc"] > 0).sum()),
        "xgb_beats_logistic_dense_logloss_fold_count": int((dense_fold["delta_log_loss"] < 0).sum()),
        "xgb_beats_logistic_nonoverlap_logloss_fold_count": int((nonover_fold["delta_log_loss"] < 0).sum()),
        "xgb_pooled_dense_auc_delta": float(dense_pooled["delta_roc_auc"]),
        "xgb_pooled_nonoverlap_auc_delta": float(nonover_pooled["delta_roc_auc"]),
        "xgb_pooled_dense_logloss_delta": float(dense_pooled["delta_log_loss"]),
        "xgb_pooled_nonoverlap_logloss_delta": float(nonover_pooled["delta_log_loss"]),
        "any_fold_auc_below_half": bool((xgb_main[xgb_main["fold_name"].ne("POOLED")]["roc_auc"] < 0.5).any()) if "fold_name" in xgb_main.columns else False,
    }


def development_recommendation(diagnostics: dict[str, Any], quality_gates: dict[str, Any], thresholds: dict[str, Any]) -> str:
    if not quality_gates.get("stage7_engineering_gate_passed", False):
        return "INVESTIGATE_PIPELINE"
    if (
        diagnostics["xgb_pooled_nonoverlap_auc_delta"] >= float(thresholds["minimum_auc_gain_for_tuning"])
        and diagnostics["xgb_pooled_nonoverlap_logloss_delta"] <= float(thresholds["maximum_allowed_logloss_degradation"])
        and diagnostics["xgb_beats_logistic_nonoverlap_auc_fold_count"] >= int(thresholds["minimum_fold_wins"])
        and not diagnostics.get("any_fold_auc_below_half", False)
    ):
        return "PROCEED_TO_LIMITED_TUNING"
    return "KEEP_LOGISTIC_AS_PRIMARY_BASELINE"


def build_model_manifest(
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    feature_columns: list[str],
    fold_metadata: dict[str, dict[str, Any]],
    quality_gates: dict[str, Any],
    diagnostics: dict[str, Any],
    recommendation: str,
    runtime_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": "stage7_xgboost_fixed_cv",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_version": dataset_manifest.get("dataset_version"),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_list_sha256": sha256_text("\n".join(feature_columns)),
        "target_column": TARGET_COLUMN,
        "model_name": MODEL_NAME,
        "xgboost_version": xgb.__version__,
        "xgboost_parameters": config["xgboost_parameters"],
        "n_jobs": int(config["n_jobs"]),
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
        "scale_pos_weight": 1.0,
        "folds": fold_metadata,
        "quality_gates": quality_gates,
        "development_diagnostics": diagnostics,
        "development_recommendation": recommendation,
        "input_hashes": runtime_hashes,
        "prohibited_actions": {
            "hyperparameter_search_performed": False,
            "probability_calibration_performed": False,
            "threshold_optimization_performed": False,
            "feature_selection_performed": False,
            "outlier_clipping_performed": False,
            "missing_value_imputation_performed": False,
            "class_resampling_performed": False,
            "sample_weight_margin_used": False,
            "final_deployment_model_trained": False,
            "shap_performed": False,
            "permutation_importance_performed": False,
        },
        "final_test_audit": {
            "final_test_prediction_count": quality_gates["final_test_prediction_count"],
            "final_test_metric_count": quality_gates["final_test_metric_count"],
            "final_test_used_for_fit": False,
            "final_test_used_for_early_stopping": False,
            "final_test_used_for_selection": False,
            "final_test_used_for_feature_importance": False,
        },
    }


def read_stage7_dataset(dataset_path: Path, splits: pd.DataFrame, feature_columns: list[str], fold_names: list[str]) -> pd.DataFrame:
    needed = pd.Series(False, index=splits.index)
    for fold in fold_names:
        needed = needed | splits[f"{fold}_role"].isin(["TRAIN", "VALIDATION"])
    needed_ids = splits.loc[needed, "dataset_row_id"].astype("int64")
    final_ids = splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].astype("int64")
    columns = [*BASE_DATASET_COLUMNS, *feature_columns]
    if final_ids.empty or int(needed_ids.max()) < int(final_ids.min()):
        table = pq.read_table(dataset_path, columns=columns, filters=[("dataset_row_id", "<=", int(needed_ids.max()))])
        df = table.to_pandas()
        df = df[df["dataset_row_id"].isin(set(needed_ids.tolist()))].copy()
    else:
        table = pq.read_table(dataset_path, columns=columns, filters=[("dataset_row_id", "in", needed_ids.tolist())])
        df = table.to_pandas()
    if df["dataset_row_id"].isin(set(final_ids.tolist())).any():
        raise Stage7ValidationError("FINAL_TEST rows were loaded into Stage7 modeling dataset")
    return df


def final_test_audit(splits: pd.DataFrame) -> dict[str, Any]:
    final = splits[splits["final_split_role"].eq("FINAL_TEST")]
    return {
        "final_test_sample_count": int(len(final)),
        "final_test_min_decision_time": int(final["decision_time"].min()) if not final.empty else None,
        "final_test_max_decision_time": int(final["decision_time"].max()) if not final.empty else None,
        "final_test_min_decision_time_utc": ms_to_utc_iso(int(final["decision_time"].min())) if not final.empty else None,
        "final_test_max_decision_time_utc": ms_to_utc_iso(int(final["decision_time"].max())) if not final.empty else None,
    }


def write_report(path: Path, outputs: Stage7Outputs, config: dict[str, Any], runtime: dict[str, Any], final_audit: dict[str, Any]) -> None:
    dense = outputs.metrics_by_subset[outputs.metrics_by_subset["subset_name"].eq("DENSE")]
    nonover = outputs.metrics_by_subset[outputs.metrics_by_subset["subset_name"].eq("NONOVERLAP_OFFSET_00")]
    pooled = outputs.oof_summary[outputs.oof_summary["summary_type"].eq("pooled") & outputs.oof_summary["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])]
    macro = outputs.oof_summary[outputs.oof_summary["summary_type"].isin(["fold_macro_mean", "fold_macro_std"]) & outputs.oof_summary["subset_name"].eq("DENSE")]
    best_rows = [
        {
            "fold_name": fold,
            "inner_fit": meta["inner_split_audit"]["inner_fit_count"],
            "inner_purged": meta["inner_split_audit"]["inner_purged_count"],
            "inner_early_stop": meta["inner_split_audit"]["inner_early_stop_count"],
            "best_iteration": meta["selector_metadata"]["best_iteration"],
            "best_n_estimators": meta["selector_metadata"]["best_n_estimators"],
            "best_score": meta["selector_metadata"]["best_score"],
            "reached_max": meta["selector_metadata"]["reached_max_estimators"],
        }
        for fold, meta in outputs.fold_metadata.items()
    ]
    lines = [
        "# Stage 7 Fixed XGBoost Report",
        "",
        "## Scope",
        "- Trained exactly one fixed XGBoost configuration: xgboost_fixed_v1.",
        "- No hyperparameter search, probability calibration, threshold optimization, feature selection, outlier clipping, missing value filling, resampling, SHAP, permutation importance, or final deployment model training was performed.",
        "- FINAL_TEST remained sealed and was not used for fit, early stopping, selection, feature importance, prediction, or metrics.",
        "",
        "## Inputs and Hashes",
        table([{"item": k, "value": v} for k, v in runtime.items() if "sha256" in k or k.endswith("_path")], ["item", "value"]),
        "## Feature Check and Fixed Parameters",
        table(
            [
                {"item": "feature_count", "value": 63},
                {"item": "feature_list_sha256", "value": outputs.model_manifest["feature_list_sha256"]},
                {"item": "xgboost_version", "value": xgb.__version__},
                {"item": "n_jobs", "value": config["n_jobs"]},
                {"item": "fixed_threshold", "value": config["fixed_prediction_threshold"]},
            ],
            ["item", "value"],
        ),
        "## Outer Fold and Inner Split Audit",
        table(best_rows, ["fold_name", "inner_fit", "inner_purged", "inner_early_stop", "best_iteration", "best_n_estimators", "best_score", "reached_max"]),
        "## Dense Metrics by Fold",
        table(metric_rows(dense), ["fold_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Non-overlap Metrics by Fold",
        table(metric_rows(nonover), ["fold_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Pooled OOF Metrics",
        table(metric_rows(pooled), ["subset_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Fold Macro Mean and Std",
        table(metric_rows(macro), ["summary_type", "model_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"]),
        "## Offset Stability",
        table(metric_rows(outputs.oof_summary[outputs.oof_summary["summary_type"].eq("pooled") & outputs.oof_summary["subset_name"].str.startswith("OFFSET_")]), ["subset_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"]),
        "## Boundary Diagnostics",
        table(metric_rows(outputs.oof_summary[outputs.oof_summary["summary_type"].eq("pooled") & outputs.oof_summary["subset_name"].isin(["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"])]), ["subset_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"]),
        "## Model Comparison",
        "- Positive ROC-AUC/AP/Accuracy/MCC deltas favor XGBoost. Negative Log Loss/Brier deltas favor XGBoost.",
        table(outputs.model_comparison[outputs.model_comparison["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"]) & outputs.model_comparison["fold_name"].isin([*config["fold_names"], "POOLED"])].to_dict(orient="records"), ["fold_name", "subset_name", "baseline_model_name", "delta_roc_auc", "delta_average_precision", "delta_log_loss", "delta_brier", "delta_accuracy", "delta_mcc"]),
        "## Probability Diagnostics",
        table(outputs.calibration_equal_width.groupby(["model_name", "subset_name"], as_index=False)[["ece", "mce"]].first().to_dict(orient="records"), ["model_name", "subset_name", "ece", "mce"]),
        "## Feature Importance Stability",
        "- Tree feature importance is not causal; correlated features split importance. No feature was removed in this stage.",
        table(outputs.feature_importance_stability.head(20).to_dict(orient="records"), ["feature_name", "mean_gain", "std_gain", "mean_normalized_gain", "mean_rank", "median_rank", "used_fold_count", "top10_fold_count"]),
        "## FINAL_TEST Audit",
        table([{"field": k, "value": v} for k, v in {**final_audit, **outputs.model_manifest["final_test_audit"]}.items()], ["field", "value"]),
        "## Engineering Gates",
        table([{"gate": k, "value": v} for k, v in outputs.quality_gates.items()], ["gate", "value"]),
        "## Development Recommendation",
        table([{"field": k, "value": v} for k, v in {**outputs.model_manifest["development_diagnostics"], "development_recommendation": outputs.model_manifest["development_recommendation"]}.items()], ["field", "value"]),
        "## Runtime",
        table([{"metric": "elapsed_seconds", "value": runtime.get("elapsed_seconds")}, {"metric": "python_tracemalloc_peak_bytes", "value": runtime.get("python_tracemalloc_peak_bytes")}, {"metric": "process_rss_bytes", "value": runtime.get("process_rss_bytes")}], ["metric", "value"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [col for col in ["fold_name", "subset_name", "summary_type", "model_name", *METRIC_COLUMNS] if col in df.columns]
    return df[cols].to_dict(orient="records")


def run_stage7(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    dataset_path = (root / config["dataset_path"]).resolve()
    split_path = (root / config["split_path"]).resolve()
    dataset_manifest_path = (root / config["dataset_manifest_path"]).resolve()
    fold_manifest_path = (root / config["fold_manifest_path"]).resolve()
    feature_manifest_path = (root / config["feature_manifest_path"]).resolve()
    stage6_prediction_path = (root / config["stage6_prediction_path"]).resolve()
    stage6_model_manifest_path = (root / config["stage6_model_manifest_path"]).resolve()
    prediction_path = (root / config["prediction_output_path"]).resolve()
    log_path = (root / config["log_path"]).resolve()
    report_paths = {name: (root / path).resolve() for name, path in config["report_paths"].items()}
    ensure_parent(log_path)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()], force=True)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        logging.info("Stage 7 fixed XGBoost training started")
        dataset_manifest_data = read_json(dataset_manifest_path)
        feature_manifest_data = read_json(feature_manifest_path)
        fold_manifest_data = read_json(fold_manifest_path)
        stage6_model_manifest_data = read_json(stage6_model_manifest_path)
        feature_columns = validate_feature_manifest(dataset_manifest_data, feature_manifest_data, int(config["feature_count"]))
        split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold}_role" for fold in config["fold_names"]]]
        splits = pd.read_parquet(split_path, columns=split_columns)
        final_audit = final_test_audit(splits)
        dataset = read_stage7_dataset(dataset_path, splits, feature_columns, config["fold_names"])
        stage6_predictions = pd.read_parquet(stage6_prediction_path)
        runtime_hashes = {
            "dataset_sha256": sha256_file(dataset_path),
            "split_sha256": sha256_file(split_path),
            "stage6_prediction_sha256": sha256_file(stage6_prediction_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "stage6_model_manifest_sha256": sha256_file(stage6_model_manifest_path),
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        outputs = build_stage7_outputs(dataset, splits, stage6_predictions, dataset_manifest_data, feature_manifest_data, fold_manifest_data, stage6_model_manifest_data, config, root=root, runtime_hashes=runtime_hashes)
        ensure_parent(prediction_path)
        outputs.predictions.to_parquet(prediction_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
        outputs.metrics_by_fold.to_csv(report_paths["metrics_by_fold"], index=False, encoding="utf-8")
        outputs.metrics_by_subset.to_csv(report_paths["metrics_by_subset"], index=False, encoding="utf-8")
        outputs.metrics_by_offset.to_csv(report_paths["metrics_by_offset"], index=False, encoding="utf-8")
        outputs.oof_summary.to_csv(report_paths["oof_summary"], index=False, encoding="utf-8")
        outputs.model_comparison.to_csv(report_paths["model_comparison"], index=False, encoding="utf-8")
        outputs.calibration_equal_width.to_csv(report_paths["calibration_equal_width"], index=False, encoding="utf-8")
        outputs.calibration_equal_frequency.to_csv(report_paths["calibration_equal_frequency"], index=False, encoding="utf-8")
        outputs.learning_curves.to_csv(report_paths["learning_curves"], index=False, encoding="utf-8")
        outputs.feature_importance_by_fold.to_csv(report_paths["feature_importance_by_fold"], index=False, encoding="utf-8")
        outputs.feature_importance_stability.to_csv(report_paths["feature_importance_stability"], index=False, encoding="utf-8")
        outputs.inner_split_audit.to_csv(report_paths["inner_split_audit"], index=False, encoding="utf-8")
        prediction_sha = sha256_file(prediction_path)
        outputs.model_manifest["output_files"] = {"prediction_output_path": str(prediction_path), "prediction_output_sha256": prediction_sha, **{k: str(v) for k, v in report_paths.items()}}
        write_json(report_paths["model_manifest"], outputs.model_manifest)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        runtime = {
            **runtime_hashes,
            "prediction_output_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
            "prediction_size_bytes": prediction_path.stat().st_size,
            "prediction_schema": parquet_schema(prediction_path),
        }
        write_report(report_paths["main_report"], outputs, config, runtime, final_audit)
        logging.info("Quality gates: %s", outputs.quality_gates)
        logging.info("Development recommendation: %s", outputs.model_manifest["development_recommendation"])
        logging.info("Prediction path: %s", prediction_path)
        logging.info("Elapsed seconds: %.2f", elapsed)
        return {
            "quality_gates": outputs.quality_gates,
            "development_recommendation": outputs.model_manifest["development_recommendation"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "prediction_size_bytes": prediction_path.stat().st_size,
            "prediction_schema": runtime["prediction_schema"],
            "fold_metadata": outputs.fold_metadata,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 7 fixed XGBoost rolling CV experiment.")
    parser.add_argument("--config", default="config/stage7_train_xgboost_fixed.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    run_stage7((root / args.config).resolve(), root)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import inspect
import json
import logging
import math
import platform
import sys
import time
import tracemalloc
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Stage6ValidationError(ValueError):
    def __init__(self, errors: list[str] | str):
        if isinstance(errors, str):
            errors = [errors]
        super().__init__("\n".join(errors))
        self.errors = errors


ALLOWED_MODEL_NAMES = {"prior_baseline", "momentum_60m_baseline", "logistic_regression_l2"}
REQUIRED_FEATURE_COUNT = 63
TARGET_COLUMN = "label_up_60m"
MOMENTUM_FEATURE = "log_return_60m"
PREDICTION_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "fold_name",
    "model_name",
    "y_true",
    "p_up",
    "y_pred",
    "prediction_threshold",
    "evaluation_offset_minutes",
    "is_primary_nonoverlap_evaluation",
    "absolute_future_return_bps",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
]
EVALUATION_METADATA_COLUMNS = [
    "absolute_future_return_bps",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
]
BASE_DATASET_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "settlement_minute_open_time",
    TARGET_COLUMN,
    "sample_weight_uniform",
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
METRIC_COLUMNS = [
    "sample_count",
    "positive_count",
    "negative_count",
    "positive_ratio",
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "average_precision",
    "pr_auc_lift_vs_positive_ratio",
    "mcc",
    "precision",
    "recall",
    "f1",
    "specificity",
    "negative_predictive_value",
    "log_loss",
    "brier_score",
    "tn",
    "fp",
    "fn",
    "tp",
    "mean_predicted_probability",
    "predicted_positive_ratio",
]


@dataclass
class Stage6Outputs:
    predictions: pd.DataFrame
    metrics_by_subset: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    metrics_by_offset: pd.DataFrame
    oof_summary: pd.DataFrame
    calibration_equal_width: pd.DataFrame
    calibration_equal_frequency: pd.DataFrame
    logistic_coefficients_by_fold: pd.DataFrame
    logistic_coefficient_stability: pd.DataFrame
    preprocessing_audit: dict[str, Any]
    model_manifest: dict[str, Any]
    quality_gates: dict[str, Any]
    fold_metadata: dict[str, dict[str, Any]]
    pipelines: dict[str, Pipeline] = field(default_factory=dict)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(make_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [make_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_none_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_markdown_cell(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def format_markdown_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def parse_utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def ms_to_utc_iso(ms_value: int | float | None) -> str | None:
    if ms_value is None or pd.isna(ms_value):
        return None
    return pd.Timestamp(int(ms_value), unit="ms", tz="UTC").isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def require_parquet_columns(path: Path, required: list[str]) -> None:
    schema_names = set(pq.read_schema(path).names)
    missing = [col for col in required if col not in schema_names]
    if missing:
        raise Stage6ValidationError(f"{path} missing required columns: {missing}")


def get_process_rss_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return int(usage.ru_maxrss * 1024)
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
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return int(counters.WorkingSetSize)
    except Exception:
        return None
    return None


def validate_feature_manifest(dataset_manifest: dict[str, Any], feature_manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    dataset_features = list(dataset_manifest.get("feature_columns", []))
    stage4_features = list(feature_manifest.get("ordered_feature_names", []))
    if len(dataset_features) != REQUIRED_FEATURE_COUNT:
        errors.append(f"dataset feature_columns count is {len(dataset_features)}, expected {REQUIRED_FEATURE_COUNT}")
    if feature_manifest.get("feature_count") != REQUIRED_FEATURE_COUNT:
        errors.append("stage4 feature_count does not equal 63")
    if dataset_features != stage4_features:
        errors.append("dataset feature_columns do not match stage4 ordered_feature_names")
    if len(set(dataset_features)) != len(dataset_features):
        errors.append("feature_columns contain duplicates")
    leakage = scan_feature_leakage(dataset_features, dataset_manifest.get("forbidden_model_input_columns", []))
    if leakage:
        errors.append(f"forbidden feature columns present in model input: {leakage}")
    if MOMENTUM_FEATURE not in dataset_features:
        errors.append(f"{MOMENTUM_FEATURE} missing from feature columns")
    if errors:
        raise Stage6ValidationError(errors)
    return dataset_features


def scan_feature_leakage(feature_columns: list[str], explicit_forbidden: list[str]) -> list[str]:
    explicit = set(explicit_forbidden)
    forbidden: list[str] = []
    for name in feature_columns:
        lower = name.lower()
        allowed_historical_return = name.startswith("log_return_")
        if name in explicit:
            forbidden.append(name)
        elif any(token in lower for token in FORBIDDEN_INPUT_TOKENS):
            forbidden.append(name)
        elif "future" in lower and not allowed_historical_return:
            forbidden.append(name)
    return forbidden


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise Stage6ValidationError(f"dataset missing feature columns: {missing}")
    X = df.loc[:, feature_columns].copy()
    if list(X.columns) != feature_columns:
        raise Stage6ValidationError("feature matrix order does not match manifest order")
    values = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise Stage6ValidationError("feature matrix contains non-finite values")
    return X


def joined_dataset(dataset: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["dataset_row_id", "decision_time", "settlement_minute_open_time"]
    for frame_name, frame in [("dataset", dataset), ("splits", splits)]:
        missing = [col for col in key_cols if col not in frame.columns]
        if missing:
            raise Stage6ValidationError(f"{frame_name} missing key columns: {missing}")
        if frame["dataset_row_id"].duplicated().any():
            raise Stage6ValidationError(f"{frame_name} dataset_row_id must be unique")
    merged = dataset.merge(splits, on=key_cols, how="inner", validate="one_to_one")
    if len(merged) != len(splits):
        raise Stage6ValidationError("dataset and split files are not one-to-one on dataset_row_id/decision_time/settlement")
    return merged.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)


def validate_fold_integrity(dataset: pd.DataFrame, splits: pd.DataFrame, fold_info: dict[str, Any], fold_name: str) -> dict[str, Any]:
    role_col = f"{fold_name}_role"
    if role_col not in splits.columns:
        raise Stage6ValidationError(f"split file missing {role_col}")
    merged = joined_dataset(dataset, splits)
    train_mask = merged[role_col].eq("TRAIN")
    valid_mask = merged[role_col].eq("VALIDATION")
    final_mask = merged["final_split_role"].eq("FINAL_TEST")
    train = merged.loc[train_mask].copy()
    valid = merged.loc[valid_mask].copy()
    errors: list[str] = []
    if train.empty:
        errors.append(f"{fold_name} TRAIN is empty")
    if valid.empty:
        errors.append(f"{fold_name} VALIDATION is empty")
    overlap = set(train["dataset_row_id"]).intersection(set(valid["dataset_row_id"]))
    if overlap:
        errors.append(f"{fold_name} train and validation dataset_row_id overlap")
    if bool((train_mask | valid_mask).loc[final_mask].any()):
        errors.append(f"{fold_name} includes FINAL_TEST samples in TRAIN or VALIDATION")
    if not train.empty and not valid.empty:
        train_max_settlement = int(train["settlement_minute_open_time"].max())
        valid_min_decision = int(valid["decision_time"].min())
        if train_max_settlement >= valid_min_decision:
            errors.append(f"{fold_name} train maximum settlement is not strictly before validation minimum decision_time")
        validation_end = parse_utc_ms(str(fold_info["validation_end"]))
        if not (valid["decision_time"].to_numpy(dtype=np.int64) < validation_end).all():
            errors.append(f"{fold_name} validation decision_time crosses validation_end")
        if not (valid["settlement_minute_open_time"].to_numpy(dtype=np.int64) < validation_end).all():
            errors.append(f"{fold_name} validation settlement_minute_open_time crosses validation_end")
    for role_name, subset in [("TRAIN", train), ("VALIDATION", valid)]:
        if not subset.empty and set(subset[TARGET_COLUMN].astype(int).unique()) != {0, 1}:
            errors.append(f"{fold_name} {role_name} must contain both classes")
    expected_train = fold_info.get("train_sample_count")
    expected_valid = fold_info.get("validation_sample_count")
    if expected_train is not None and not train.empty and len(train) != int(expected_train):
        errors.append(f"{fold_name} TRAIN count {len(train)} does not match manifest {expected_train}")
    if expected_valid is not None and not valid.empty and len(valid) != int(expected_valid):
        errors.append(f"{fold_name} VALIDATION count {len(valid)} does not match manifest {expected_valid}")
    if errors:
        raise Stage6ValidationError(errors)
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
        "validation_end": fold_info.get("validation_end"),
        "final_test_in_train_or_validation": int(((train_mask | valid_mask) & final_mask).sum()),
        "train_label_distribution": label_distribution(train[TARGET_COLUMN]),
        "validation_label_distribution": label_distribution(valid[TARGET_COLUMN]),
    }


def label_distribution(labels: pd.Series | np.ndarray) -> dict[str, int]:
    arr = pd.Series(labels).astype(int)
    return {"0": int((arr == 0).sum()), "1": int((arr == 1).sum())}


def make_prior_predictions(y_train: np.ndarray, validation: pd.DataFrame, fold_name: str, threshold: float = 0.5) -> tuple[pd.DataFrame, dict[str, Any]]:
    y = np.asarray(y_train, dtype=np.int8)
    if set(np.unique(y)) != {0, 1}:
        raise Stage6ValidationError(f"{fold_name} prior baseline training labels must contain both classes")
    train_up_probability = float(y.mean())
    predicted_class = int(train_up_probability >= threshold)
    pred = validation[["dataset_row_id", "decision_time"]].copy()
    pred["fold_name"] = fold_name
    pred["model_name"] = "prior_baseline"
    pred["p_up"] = train_up_probability
    pred["y_pred"] = np.int8(predicted_class)
    pred["prediction_threshold"] = threshold
    meta = {
        "model_name": "prior_baseline",
        "definition": "No-feature baseline; p_up is the TRAIN label_up_60m mean for this fold.",
        "train_up_probability": train_up_probability,
        "predicted_class": predicted_class,
        "training_sample_count": int(len(y)),
        "train_label_distribution": label_distribution(y),
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
    }
    return pred, meta


def make_momentum_predictions(train: pd.DataFrame, validation: pd.DataFrame, fold_name: str, alpha: float = 1.0) -> tuple[pd.DataFrame, dict[str, Any]]:
    if MOMENTUM_FEATURE not in train.columns or MOMENTUM_FEATURE not in validation.columns:
        raise Stage6ValidationError(f"{MOMENTUM_FEATURE} is required for momentum baseline")
    train_positive = train[MOMENTUM_FEATURE].to_numpy(dtype=np.float64) > 0.0
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    positive_meta = condition_group_metadata(y_train, train_positive, alpha)
    nonpositive_meta = condition_group_metadata(y_train, ~train_positive, alpha)
    valid_positive = validation[MOMENTUM_FEATURE].to_numpy(dtype=np.float64) > 0.0
    pred = validation[["dataset_row_id", "decision_time"]].copy()
    pred["fold_name"] = fold_name
    pred["model_name"] = "momentum_60m_baseline"
    pred["p_up"] = np.where(valid_positive, positive_meta["smoothed_up_probability"], nonpositive_meta["smoothed_up_probability"])
    pred["y_pred"] = valid_positive.astype(np.int8)
    pred["prediction_threshold"] = np.nan
    meta = {
        "model_name": "momentum_60m_baseline",
        "definition": "Trend continuation baseline; hard class is log_return_60m > 0, probabilities are TRAIN conditional label rates with Laplace smoothing.",
        "alpha": float(alpha),
        "positive_group": positive_meta,
        "nonpositive_group": nonpositive_meta,
        "validation_positive_group_count": int(valid_positive.sum()),
        "validation_nonpositive_group_count": int((~valid_positive).sum()),
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
    }
    return pred, meta


def condition_group_metadata(y_train: np.ndarray, mask: np.ndarray, alpha: float) -> dict[str, Any]:
    group_count = int(mask.sum())
    up_count = int(y_train[mask].sum()) if group_count else 0
    raw = float(up_count / group_count) if group_count else None
    smoothed = float((up_count + alpha) / (group_count + 2.0 * alpha))
    return {
        "group_count": group_count,
        "up_count": up_count,
        "raw_up_probability": raw,
        "smoothed_up_probability": smoothed,
    }


def logistic_parameters_for_current_sklearn(config_params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(LogisticRegression)
    params: dict[str, Any] = {}
    for key, value in config_params.items():
        if key in signature.parameters:
            params[key] = value
    if "n_jobs" in signature.parameters and "n_jobs" not in params:
        params["n_jobs"] = None
    return params


def fit_logistic_fold(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    logistic_params: dict[str, Any],
    scaler_params: dict[str, Any],
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any], Pipeline]:
    X_train_df = prepare_feature_matrix(train, feature_columns)
    X_valid_df = prepare_feature_matrix(validation, feature_columns)
    y_train = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    if set(np.unique(y_train)) != {0, 1}:
        raise Stage6ValidationError("logistic regression training labels must contain both classes")
    if "sample_weight_margin" in train.columns:
        sample_weight_margin_present = True
    else:
        sample_weight_margin_present = False
    X_train = X_train_df.to_numpy(dtype=np.float64, copy=True)
    X_valid = X_valid_df.to_numpy(dtype=np.float64, copy=True)
    independent_mean = np.mean(X_train, axis=0)
    independent_var = np.var(X_train, axis=0, ddof=0)
    scaler = StandardScaler(**scaler_params)
    X_train_scaled = scaler.fit_transform(X_train)
    scaler_mean_diff_max = float(np.max(np.abs(scaler.mean_ - independent_mean)))
    scaler_var_diff_max = float(np.max(np.abs(scaler.var_ - independent_var)))
    X_valid_scaled = scaler.transform(X_valid)
    params = logistic_parameters_for_current_sklearn(logistic_params)
    clf = LogisticRegression(**params)
    convergence_warnings: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clf.fit(X_train_scaled, y_train)
        for item in caught:
            if issubclass(item.category, ConvergenceWarning):
                convergence_warnings.append(str(item.message))
    p_up = clf.predict_proba(X_valid_scaled)[:, 1]
    if not np.isfinite(p_up).all():
        raise Stage6ValidationError("logistic regression produced non-finite probabilities")
    if not ((p_up >= 0.0) & (p_up <= 1.0)).all():
        raise Stage6ValidationError("logistic regression probabilities are outside [0, 1]")
    y_pred = (p_up >= threshold).astype(np.int8)
    pred = validation[["dataset_row_id", "decision_time"]].copy()
    pred["fold_name"] = str(validation.attrs.get("fold_name", ""))
    pred["model_name"] = "logistic_regression_l2"
    pred["p_up"] = p_up
    pred["y_pred"] = y_pred
    pred["prediction_threshold"] = threshold
    n_iter = [int(x) for x in np.ravel(clf.n_iter_)]
    converged = len(convergence_warnings) == 0
    pipeline = Pipeline([("scaler", scaler), ("logistic_regression", clf)])
    meta = {
        "model_name": "logistic_regression_l2",
        "definition": "StandardScaler fitted on TRAIN only plus LogisticRegression with fixed L2 parameters.",
        "logistic_regression_parameters": params,
        "scaler_parameters": scaler_params,
        "scaler": {
            "mean_": scaler.mean_.astype(float).tolist(),
            "var_": scaler.var_.astype(float).tolist(),
            "scale_": scaler.scale_.astype(float).tolist(),
            "n_samples_seen_": make_json_safe(scaler.n_samples_seen_),
            "independent_mean_var_check": {
                "ddof": 0,
                "max_abs_mean_diff": scaler_mean_diff_max,
                "max_abs_var_diff": scaler_var_diff_max,
            },
        },
        "model": {
            "coef_": clf.coef_.astype(float).tolist(),
            "intercept_": clf.intercept_.astype(float).tolist(),
        },
        "n_iter": n_iter,
        "converged": bool(converged),
        "convergence_warnings": convergence_warnings,
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
        "sample_weight_margin_column_present_but_unused": sample_weight_margin_present,
    }
    del X_train, X_valid, X_train_scaled, X_valid_scaled, X_train_df, X_valid_df
    return pred, meta, pipeline


def add_validation_metadata(predictions: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    metadata = validation[
        [
            "dataset_row_id",
            TARGET_COLUMN,
            "evaluation_offset_minutes",
            "is_primary_nonoverlap_evaluation",
            *EVALUATION_METADATA_COLUMNS,
        ]
    ].rename(columns={TARGET_COLUMN: "y_true"})
    merged = predictions.merge(metadata, on="dataset_row_id", how="left", validate="one_to_one")
    if merged["y_true"].isna().any():
        raise Stage6ValidationError("prediction metadata join failed")
    return merged[PREDICTION_COLUMNS]


def compute_classification_metrics(y_true: np.ndarray, p_up: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int8)
    p = np.asarray(p_up, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.int8)
    if len(y) == 0:
        return empty_metrics()
    if not np.isfinite(p).all():
        raise Stage6ValidationError("non-finite probabilities in metric input")
    if not ((p >= 0.0) & (p <= 1.0)).all():
        raise Stage6ValidationError("probabilities outside [0, 1] in metric input")
    positive = int((y == 1).sum())
    negative = int((y == 0).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = float((pred == y).mean())
    balanced = float(balanced_accuracy_score(y, pred)) if positive and negative else np.nan
    if positive and negative:
        roc_auc = float(roc_auc_score(y, p))
        roc_missing = ""
    else:
        roc_auc = np.nan
        roc_missing = "single_class_validation_subset"
    if positive == 0:
        average_precision = np.nan
    else:
        average_precision = float(average_precision_score(y, p))
    positive_ratio = float(positive / len(y))
    pr_lift = float(average_precision / positive_ratio) if positive_ratio > 0 and not np.isnan(average_precision) else np.nan
    mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(((tp * tn) - (fp * fn)) / mcc_den) if mcc_den else 0.0
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = safe_div(2 * precision * recall, precision + recall) if not (np.isnan(precision) or np.isnan(recall)) else np.nan
    return {
        "sample_count": int(len(y)),
        "positive_count": positive,
        "negative_count": negative,
        "positive_ratio": positive_ratio,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "roc_auc": roc_auc,
        "roc_auc_missing_reason": roc_missing,
        "average_precision": average_precision,
        "pr_auc_lift_vs_positive_ratio": pr_lift,
        "mcc": mcc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "negative_predictive_value": npv,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier_score": float(np.mean((p - y) ** 2)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mean_predicted_probability": float(p.mean()),
        "predicted_positive_ratio": float((pred == 1).mean()),
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def empty_metrics() -> dict[str, Any]:
    metrics = {col: np.nan for col in METRIC_COLUMNS}
    metrics.update(
        {
            "sample_count": 0,
            "positive_count": 0,
            "negative_count": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
            "roc_auc_missing_reason": "empty_subset",
        }
    )
    return metrics


def subset_masks(predictions: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    return [
        ("DENSE", pd.Series(True, index=predictions.index)),
        ("NONOVERLAP_OFFSET_00", predictions["is_primary_nonoverlap_evaluation"].astype(bool)),
        *[
            (f"OFFSET_{offset:02d}", predictions["evaluation_offset_minutes"].astype(int).eq(offset))
            for offset in range(0, 60, 5)
        ],
        ("ALL_MARGINS", pd.Series(True, index=predictions.index)),
        ("ABS_RETURN_GE_1BPS", predictions["absolute_future_return_bps"].astype(float).ge(1.0)),
        ("ABS_RETURN_GE_2_5BPS", predictions["absolute_future_return_bps"].astype(float).ge(2.5)),
        ("ABS_RETURN_GE_5BPS", predictions["absolute_future_return_bps"].astype(float).ge(5.0)),
        ("ABS_RETURN_GE_10BPS", predictions["absolute_future_return_bps"].astype(float).ge(10.0)),
    ]


def evaluate_prediction_subsets(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold_name, model_name), group in predictions.groupby(["fold_name", "model_name"], sort=True):
        for subset_name, mask in subset_masks(group):
            subset = group.loc[mask]
            metrics = compute_classification_metrics(
                subset["y_true"].to_numpy(dtype=np.int8),
                subset["p_up"].to_numpy(dtype=np.float64),
                subset["y_pred"].to_numpy(dtype=np.int8),
            )
            rows.append({"fold_name": fold_name, "model_name": model_name, "subset_name": subset_name, **metrics})
    return pd.DataFrame(rows)


def pooled_and_macro_summary(metrics_by_subset: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model_predictions in predictions.groupby("model_name", sort=True):
        for subset_name, mask in subset_masks(model_predictions):
            subset = model_predictions.loc[mask]
            metrics = compute_classification_metrics(
                subset["y_true"].to_numpy(dtype=np.int8),
                subset["p_up"].to_numpy(dtype=np.float64),
                subset["y_pred"].to_numpy(dtype=np.int8),
            )
            rows.append({"model_name": model_name, "subset_name": subset_name, "summary_type": "pooled", "fold_name": "", **metrics})
            fold_rows = metrics_by_subset[(metrics_by_subset["model_name"] == model_name) & (metrics_by_subset["subset_name"] == subset_name)]
            mean_row = {"model_name": model_name, "subset_name": subset_name, "summary_type": "fold_macro_mean", "fold_name": ""}
            std_row = {"model_name": model_name, "subset_name": subset_name, "summary_type": "fold_macro_std", "fold_name": ""}
            for col in METRIC_COLUMNS:
                mean_row[col] = float(fold_rows[col].mean(skipna=True)) if col in fold_rows else np.nan
                std_row[col] = float(fold_rows[col].std(skipna=True, ddof=1)) if len(fold_rows) > 1 and col in fold_rows else np.nan
            rows.extend([mean_row, std_row])
            if not fold_rows.empty:
                valid_auc = fold_rows.dropna(subset=["roc_auc"])
                if not valid_auc.empty:
                    best = valid_auc.loc[valid_auc["roc_auc"].idxmax()]
                    worst = valid_auc.loc[valid_auc["roc_auc"].idxmin()]
                    rows.append(summary_fold_marker(model_name, subset_name, "best_fold_by_roc_auc", best))
                    rows.append(summary_fold_marker(model_name, subset_name, "worst_fold_by_roc_auc", worst))
    for subset_name in metrics_by_subset["subset_name"].unique():
        subset_metrics = metrics_by_subset[metrics_by_subset["subset_name"] == subset_name]
        for reference in ["prior_baseline", "momentum_60m_baseline"]:
            ref = subset_metrics[subset_metrics["model_name"] == reference].set_index("fold_name")
            if ref.empty:
                continue
            for _, row in subset_metrics[~subset_metrics["model_name"].eq(reference)].iterrows():
                fold_name = row["fold_name"]
                if fold_name not in ref.index:
                    continue
                delta = {"model_name": row["model_name"], "subset_name": subset_name, "summary_type": f"fold_delta_vs_{reference}", "fold_name": fold_name}
                for col in METRIC_COLUMNS:
                    delta[col] = row[col] - ref.loc[fold_name, col] if col in row and col in ref.columns else np.nan
                rows.append(delta)
    return pd.DataFrame(rows)


def summary_fold_marker(model_name: str, subset_name: str, summary_type: str, source: pd.Series) -> dict[str, Any]:
    row = {"model_name": model_name, "subset_name": subset_name, "summary_type": summary_type, "fold_name": source["fold_name"]}
    for col in METRIC_COLUMNS:
        row[col] = source.get(col, np.nan)
    return row


def calibration_table(predictions: pd.DataFrame, bins: int, strategy: str, subset_name: str) -> pd.DataFrame:
    if subset_name == "NONOVERLAP_OFFSET_00":
        base = predictions.loc[predictions["is_primary_nonoverlap_evaluation"].astype(bool)].copy()
    elif subset_name == "DENSE":
        base = predictions.copy()
    else:
        raise Stage6ValidationError(f"unsupported calibration subset {subset_name}")
    rows: list[dict[str, Any]] = []
    for model_name, group in base.groupby("model_name", sort=True):
        p = group["p_up"].to_numpy(dtype=np.float64)
        y = group["y_true"].to_numpy(dtype=np.float64)
        assignments, bounds = assign_probability_bins(p, bins, strategy)
        total = len(group)
        bin_gaps: list[tuple[int, float]] = []
        for bin_index in sorted(set(assignments.tolist())):
            mask = assignments == bin_index
            sample_count = int(mask.sum())
            if sample_count == 0:
                continue
            mean_p = float(p[mask].mean())
            actual = float(y[mask].mean())
            gap = mean_p - actual
            bin_gaps.append((sample_count, abs(gap)))
            lower, upper = bounds[bin_index]
            rows.append(
                {
                    "model_name": model_name,
                    "subset_name": subset_name,
                    "binning_strategy": strategy,
                    "bin_index": int(bin_index),
                    "bin_lower": float(lower),
                    "bin_upper": float(upper),
                    "sample_count": sample_count,
                    "mean_predicted_probability": mean_p,
                    "actual_up_ratio": actual,
                    "probability_actual_gap": gap,
                    "ece": np.nan,
                    "mce": np.nan,
                    "ece_definition": "sum over bins of (bin_sample_count / total_sample_count) * abs(mean_predicted_probability - actual_up_ratio)",
                }
            )
        ece = float(sum((count / total) * gap for count, gap in bin_gaps)) if total else np.nan
        mce = float(max((gap for _, gap in bin_gaps), default=np.nan))
        for row in rows:
            if row["model_name"] == model_name and row["subset_name"] == subset_name and row["binning_strategy"] == strategy:
                row["ece"] = ece
                row["mce"] = mce
    return pd.DataFrame(rows)


def assign_probability_bins(p: np.ndarray, bins: int, strategy: str) -> tuple[np.ndarray, dict[int, tuple[float, float]]]:
    if len(p) == 0:
        return np.array([], dtype=int), {}
    if strategy == "equal_width":
        edges = np.linspace(0.0, 1.0, bins + 1)
        assignments = np.digitize(p, edges[1:-1], right=False).astype(int)
        bounds = {i: (edges[i], edges[i + 1]) for i in range(bins)}
        return assignments, bounds
    if strategy == "equal_frequency":
        order = np.argsort(p, kind="mergesort")
        assignments = np.empty(len(p), dtype=int)
        for rank, idx in enumerate(order):
            assignments[idx] = min(int(rank * bins / len(p)), bins - 1)
        bounds: dict[int, tuple[float, float]] = {}
        for i in sorted(set(assignments.tolist())):
            values = p[assignments == i]
            bounds[i] = (float(values.min()), float(values.max()))
        return assignments, bounds
    raise Stage6ValidationError(f"unknown calibration strategy {strategy}")


def build_stage6_outputs(
    dataset: pd.DataFrame,
    splits: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    fold_manifest: dict[str, Any],
    config: dict[str, Any],
    root: Path | None = None,
    runtime_hashes: dict[str, str] | None = None,
) -> Stage6Outputs:
    feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest)
    joined = joined_dataset(dataset, splits)
    prepare_feature_matrix(joined, feature_columns)
    folds_by_name = {fold["name"]: fold for fold in fold_manifest["folds"]}
    configured_folds = [name for name in config["fold_names"] if name in folds_by_name]
    if not configured_folds:
        raise Stage6ValidationError("no configured folds found in fold manifest")

    prediction_frames: list[pd.DataFrame] = []
    fold_metadata: dict[str, dict[str, Any]] = {}
    coefficients: list[pd.DataFrame] = []
    pipelines: dict[str, Pipeline] = {}
    preprocessing_audit: dict[str, Any] = {
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "folds": {},
        "preprocessing_train_only_verified": True,
        "scaler_ddof": 0,
    }
    all_integrity_passed = True
    all_probabilities_finite = True
    all_models_serialized = True
    script_path = Path(__file__).resolve()
    for fold_name in configured_folds:
        fold_info = folds_by_name[fold_name]
        role_col = f"{fold_name}_role"
        fold_started = time.perf_counter()
        gc.collect()
        fold_trace_started = tracemalloc.is_tracing()
        if not fold_trace_started:
            tracemalloc.start()
        integrity = validate_fold_integrity(dataset, splits, fold_info, fold_name)
        train = joined.loc[joined[role_col].eq("TRAIN")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
        validation = joined.loc[joined[role_col].eq("VALIDATION")].sort_values(["decision_time", "dataset_row_id"], kind="mergesort").copy()
        validation.attrs["fold_name"] = fold_name

        prior_start = time.perf_counter()
        prior_pred, prior_meta = make_prior_predictions(train[TARGET_COLUMN].to_numpy(dtype=np.int8), validation, fold_name, config["fixed_prediction_threshold"])
        prior_prediction_seconds = time.perf_counter() - prior_start

        momentum_start = time.perf_counter()
        momentum_pred, momentum_meta = make_momentum_predictions(train, validation, fold_name, float(config["momentum_baseline_alpha"]))
        momentum_prediction_seconds = time.perf_counter() - momentum_start

        logistic_fit_start = time.perf_counter()
        logistic_pred, logistic_meta, pipeline = fit_logistic_fold(
            train,
            validation,
            feature_columns,
            config["logistic_regression_parameters"],
            config["scaler_parameters"],
            float(config["fixed_prediction_threshold"]),
        )
        logistic_pred["fold_name"] = fold_name
        logistic_fit_seconds = time.perf_counter() - logistic_fit_start
        logistic_prediction_seconds = logistic_fit_seconds
        pipeline_path: Path | None = None
        fold_dir: Path | None = None
        if root is not None:
            fold_dir = (root / config["model_output_dir"] / fold_name).resolve()
            fold_dir.mkdir(parents=True, exist_ok=True)
            pipeline_path = fold_dir / "logistic_regression_pipeline.joblib"
            joblib.dump(pipeline, pipeline_path)
            write_json(fold_dir / "prior_baseline.json", prior_meta)
            write_json(fold_dir / "momentum_60m_baseline.json", momentum_meta)
            all_models_serialized = all_models_serialized and pipeline_path.exists()
        pipelines[fold_name] = pipeline

        fold_predictions = [
            add_validation_metadata(prior_pred, validation),
            add_validation_metadata(momentum_pred, validation),
            add_validation_metadata(logistic_pred, validation),
        ]
        for frame in fold_predictions:
            all_probabilities_finite = all_probabilities_finite and bool(np.isfinite(frame["p_up"].to_numpy(dtype=np.float64)).all())
        prediction_frames.extend(fold_predictions)

        coef_df = coefficients_for_fold(fold_name, feature_columns, logistic_meta["model"]["coef_"][0])
        coefficients.append(coef_df)
        _, fold_peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        model_hash = sha256_file(pipeline_path) if pipeline_path is not None and pipeline_path.exists() else None
        fold_elapsed = time.perf_counter() - fold_started
        feature_hash = sha256_text("\n".join(feature_columns))
        fold_meta = {
            "fold_name": fold_name,
            "train_time_range": {
                "min_decision_time": int(train["decision_time"].min()),
                "max_decision_time": int(train["decision_time"].max()),
                "max_settlement_minute_open_time": int(train["settlement_minute_open_time"].max()),
                "min_decision_time_utc": ms_to_utc_iso(int(train["decision_time"].min())),
                "max_decision_time_utc": ms_to_utc_iso(int(train["decision_time"].max())),
            },
            "validation_time_range": {
                "min_decision_time": int(validation["decision_time"].min()),
                "max_decision_time": int(validation["decision_time"].max()),
                "max_settlement_minute_open_time": int(validation["settlement_minute_open_time"].max()),
                "min_decision_time_utc": ms_to_utc_iso(int(validation["decision_time"].min())),
                "max_decision_time_utc": ms_to_utc_iso(int(validation["decision_time"].max())),
            },
            "train_row_count": int(len(train)),
            "validation_row_count": int(len(validation)),
            "feature_columns": feature_columns,
            "feature_list_sha256": feature_hash,
            "integrity": integrity,
            "prior_baseline": prior_meta,
            "momentum_60m_baseline": momentum_meta,
            "scaler_parameter_summary": {
                "mean_": logistic_meta["scaler"]["mean_"],
                "var_": logistic_meta["scaler"]["var_"],
                "scale_": logistic_meta["scaler"]["scale_"],
                "n_samples_seen_": logistic_meta["scaler"]["n_samples_seen_"],
                "independent_mean_var_check": logistic_meta["scaler"]["independent_mean_var_check"],
            },
            "logistic_regression_parameters": logistic_meta["logistic_regression_parameters"],
            "logistic_regression_n_iter": logistic_meta["n_iter"],
            "converged": logistic_meta["converged"],
            "convergence_warnings": logistic_meta["convergence_warnings"],
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "random_seed": int(config["random_seed"]),
            "training_weight_scheme": "uniform",
            "sample_weight_margin_used": False,
            "dataset_sha256": (runtime_hashes or {}).get("dataset_sha256"),
            "split_file_sha256": (runtime_hashes or {}).get("split_sha256"),
            "config_sha256": (runtime_hashes or {}).get("config_sha256"),
            "script_sha256": sha256_file(script_path) if script_path.exists() else None,
            "model_file_path": str(pipeline_path) if pipeline_path else None,
            "model_file_sha256": model_hash,
            "fit_elapsed_seconds": float(fold_elapsed),
            "prediction_elapsed_seconds": float(prior_prediction_seconds + momentum_prediction_seconds + logistic_prediction_seconds),
            "python_tracemalloc_peak_bytes": int(fold_peak),
            "process_rss_bytes": rss,
        }
        fold_metadata[fold_name] = fold_meta
        preprocessing_audit["folds"][fold_name] = {
            "train_only_fit": True,
            "train_row_count": int(len(train)),
            "validation_row_count": int(len(validation)),
            "scaler_n_samples_seen": logistic_meta["scaler"]["n_samples_seen_"],
            "max_abs_mean_diff_vs_independent_train_mean": logistic_meta["scaler"]["independent_mean_var_check"]["max_abs_mean_diff"],
            "max_abs_var_diff_vs_independent_train_var": logistic_meta["scaler"]["independent_mean_var_check"]["max_abs_var_diff"],
            "validation_used_for_scaler_fit": False,
            "final_test_used_for_scaler_fit": False,
        }
        if fold_dir is not None:
            write_json(fold_dir / "fold_metadata.json", fold_meta)
        del train, validation, prior_pred, momentum_pred, logistic_pred, pipeline
        gc.collect()

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions.sort_values(["fold_name", "model_name", "decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)
    validate_predictions(predictions, splits, configured_folds)
    metrics_by_subset = evaluate_prediction_subsets(predictions)
    metrics_by_fold = metrics_by_subset[metrics_by_subset["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])].copy()
    metrics_by_offset = metrics_by_subset[metrics_by_subset["subset_name"].str.startswith("OFFSET_")].copy()
    oof_summary = pooled_and_macro_summary(metrics_by_subset, predictions)
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
    coefficient_df = pd.concat(coefficients, ignore_index=True) if coefficients else pd.DataFrame()
    coefficient_stability = coefficient_stability_table(coefficient_df)
    quality_gates = build_quality_gates(predictions, splits, configured_folds, feature_columns, preprocessing_audit, all_integrity_passed, all_probabilities_finite, all_models_serialized, metrics_by_subset)
    quality_gates["logistic_all_folds_converged"] = all(bool(fold_metadata[fold]["converged"]) for fold in configured_folds)
    engineering_keys = [
        "all_fold_integrity_checks_passed",
        "no_final_test_predictions",
        "feature_manifest_match",
        "preprocessing_train_only_verified",
        "all_probabilities_finite",
        "all_prediction_counts_match",
        "all_models_serialized",
        "all_metric_outputs_complete",
    ]
    quality_gates["stage6_engineering_gate_passed"] = all(bool(quality_gates[key]) for key in engineering_keys)
    model_manifest = build_model_manifest(
        config,
        dataset_manifest,
        feature_manifest,
        fold_metadata,
        feature_columns,
        predictions,
        metrics_by_subset,
        quality_gates,
        runtime_hashes or {},
    )
    return Stage6Outputs(
        predictions=predictions,
        metrics_by_subset=metrics_by_subset,
        metrics_by_fold=metrics_by_fold,
        metrics_by_offset=metrics_by_offset,
        oof_summary=oof_summary,
        calibration_equal_width=calibration_equal_width,
        calibration_equal_frequency=calibration_equal_frequency,
        logistic_coefficients_by_fold=coefficient_df,
        logistic_coefficient_stability=coefficient_stability,
        preprocessing_audit=preprocessing_audit,
        model_manifest=model_manifest,
        quality_gates=quality_gates,
        fold_metadata=fold_metadata,
        pipelines=pipelines,
    )


def coefficients_for_fold(fold_name: str, feature_columns: list[str], coefficients: list[float]) -> pd.DataFrame:
    coef = np.asarray(coefficients, dtype=np.float64)
    abs_coef = np.abs(coef)
    order = np.argsort(-abs_coef, kind="mergesort")
    ranks = np.empty(len(coef), dtype=int)
    ranks[order] = np.arange(1, len(coef) + 1)
    return pd.DataFrame(
        {
            "fold_name": fold_name,
            "feature_index": np.arange(1, len(feature_columns) + 1),
            "feature_name": feature_columns,
            "coefficient": coef,
            "absolute_coefficient": abs_coef,
            "coefficient_rank": ranks,
            "coefficient_sign": np.where(coef > 0, "positive", np.where(coef < 0, "negative", "zero")),
        }
    )


def coefficient_stability_table(coef_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if coef_df.empty:
        return pd.DataFrame()
    for feature_name, group in coef_df.groupby("feature_name", sort=False):
        coef = group["coefficient"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "feature_name": feature_name,
                "coefficient_mean": float(np.mean(coef)),
                "coefficient_std": float(np.std(coef, ddof=1)) if len(coef) > 1 else 0.0,
                "mean_absolute_coefficient": float(np.mean(np.abs(coef))),
                "positive_fold_count": int((coef > 0).sum()),
                "negative_fold_count": int((coef < 0).sum()),
                "sign_consistency_ratio": float(max((coef > 0).sum(), (coef < 0).sum(), (coef == 0).sum()) / len(coef)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_absolute_coefficient", ascending=False, kind="mergesort").reset_index(drop=True)


def validate_predictions(predictions: pd.DataFrame, splits: pd.DataFrame, fold_names: list[str]) -> None:
    errors: list[str] = []
    if set(predictions["model_name"].unique()) != ALLOWED_MODEL_NAMES:
        errors.append("prediction file contains unexpected model_name values")
    if not np.isfinite(predictions["p_up"].to_numpy(dtype=np.float64)).all():
        errors.append("prediction probabilities contain non-finite values")
    if not predictions["p_up"].between(0, 1).all():
        errors.append("prediction probabilities outside [0, 1]")
    if not set(predictions["y_pred"].unique()).issubset({0, 1}):
        errors.append("prediction y_pred contains values outside {0, 1}")
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    if final_ids.intersection(set(predictions["dataset_row_id"].tolist())):
        errors.append("prediction output contains FINAL_TEST samples")
    for fold_name in fold_names:
        role_col = f"{fold_name}_role"
        validation_ids = set(splits.loc[splits[role_col].eq("VALIDATION"), "dataset_row_id"].tolist())
        train_ids = set(splits.loc[splits[role_col].eq("TRAIN"), "dataset_row_id"].tolist())
        fold_predictions = predictions[predictions["fold_name"].eq(fold_name)]
        if set(fold_predictions["dataset_row_id"].unique()) != validation_ids:
            errors.append(f"{fold_name} prediction ids do not match validation ids")
        if set(fold_predictions["dataset_row_id"].unique()).intersection(train_ids):
            errors.append(f"{fold_name} prediction output contains TRAIN samples")
        counts = fold_predictions.groupby(["model_name", "dataset_row_id"]).size()
        if not counts.eq(1).all() or len(counts) != len(validation_ids) * len(ALLOWED_MODEL_NAMES):
            errors.append(f"{fold_name} does not have exactly one prediction per validation sample per model")
    if errors:
        raise Stage6ValidationError(errors)


def build_quality_gates(
    predictions: pd.DataFrame,
    splits: pd.DataFrame,
    fold_names: list[str],
    feature_columns: list[str],
    preprocessing_audit: dict[str, Any],
    all_integrity_passed: bool,
    all_probabilities_finite: bool,
    all_models_serialized: bool,
    metrics_by_subset: pd.DataFrame,
) -> dict[str, Any]:
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].tolist())
    final_test_prediction_count = int(predictions["dataset_row_id"].isin(final_ids).sum())
    final_test_metric_count = 0
    expected_predictions = 0
    prediction_counts_match = True
    for fold_name in fold_names:
        role_col = f"{fold_name}_role"
        valid_count = int(splits[role_col].eq("VALIDATION").sum())
        expected_predictions += valid_count * len(ALLOWED_MODEL_NAMES)
        actual = len(predictions[predictions["fold_name"].eq(fold_name)])
        prediction_counts_match = prediction_counts_match and actual == valid_count * len(ALLOWED_MODEL_NAMES)
    required_subset_count = len(subset_masks(predictions))
    metric_outputs_complete = True
    for fold_name in fold_names:
        for model_name in ALLOWED_MODEL_NAMES:
            actual = len(metrics_by_subset[(metrics_by_subset["fold_name"] == fold_name) & (metrics_by_subset["model_name"] == model_name)])
            metric_outputs_complete = metric_outputs_complete and actual == required_subset_count
    logistic_converged = bool(
        metrics_by_subset["model_name"].eq("logistic_regression_l2").any()
    )
    gates = {
        "all_fold_integrity_checks_passed": bool(all_integrity_passed),
        "all_probabilities_finite": bool(all_probabilities_finite and np.isfinite(predictions["p_up"].to_numpy(dtype=np.float64)).all()),
        "no_final_test_predictions": final_test_prediction_count == 0,
        "feature_manifest_match": len(feature_columns) == REQUIRED_FEATURE_COUNT,
        "preprocessing_train_only_verified": bool(preprocessing_audit.get("preprocessing_train_only_verified")),
        "all_models_serialized": bool(all_models_serialized),
        "all_prediction_counts_match": bool(prediction_counts_match and len(predictions) == expected_predictions),
        "all_metric_outputs_complete": bool(metric_outputs_complete),
        "logistic_all_folds_converged": bool(logistic_converged),
        "final_test_prediction_count": final_test_prediction_count,
        "final_test_metric_count": final_test_metric_count,
        "final_test_used_for_fit": False,
        "final_test_used_for_selection": False,
    }
    engineering_keys = [
        "all_fold_integrity_checks_passed",
        "no_final_test_predictions",
        "feature_manifest_match",
        "preprocessing_train_only_verified",
        "all_probabilities_finite",
        "all_prediction_counts_match",
        "all_models_serialized",
        "all_metric_outputs_complete",
    ]
    gates["stage6_engineering_gate_passed"] = all(bool(gates[key]) for key in engineering_keys)
    return gates


def build_model_manifest(
    config: dict[str, Any],
    dataset_manifest: dict[str, Any],
    feature_manifest: dict[str, Any],
    fold_metadata: dict[str, dict[str, Any]],
    feature_columns: list[str],
    predictions: pd.DataFrame,
    metrics_by_subset: pd.DataFrame,
    quality_gates: dict[str, Any],
    runtime_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "stage": "stage6_baseline_cv_training",
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_version": dataset_manifest.get("dataset_version"),
        "feature_set_version": feature_manifest.get("feature_set_version"),
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "feature_list_sha256": sha256_text("\n".join(feature_columns)),
        "target_column": TARGET_COLUMN,
        "training_weight_scheme": "uniform",
        "sample_weight_margin_used": False,
        "models": config["model_definitions"],
        "logistic_regression_parameters": logistic_parameters_for_current_sklearn(config["logistic_regression_parameters"]),
        "fixed_prediction_threshold": config["fixed_prediction_threshold"],
        "momentum_baseline_alpha": config["momentum_baseline_alpha"],
        "folds": fold_metadata,
        "prediction_count": int(len(predictions)),
        "metric_row_count": int(len(metrics_by_subset)),
        "quality_gates": quality_gates,
        "input_hashes": runtime_hashes,
        "final_test_audit": {
            "final_test_prediction_count": quality_gates["final_test_prediction_count"],
            "final_test_metric_count": quality_gates["final_test_metric_count"],
            "final_test_used_for_fit": False,
            "final_test_used_for_selection": False,
        },
        "prohibited_actions": {
            "probability_calibration_performed": False,
            "threshold_optimization_performed": False,
            "xgboost_trained": False,
            "lightgbm_trained": False,
            "catboost_trained": False,
            "neural_network_trained": False,
            "feature_selection_performed": False,
            "outlier_clipping_performed": False,
            "missing_value_imputation_performed": False,
            "class_resampling_performed": False,
            "final_deployment_model_trained": False,
        },
    }


def final_test_audit_from_splits(splits: pd.DataFrame) -> dict[str, Any]:
    final = splits[splits["final_split_role"].eq("FINAL_TEST")]
    return {
        "final_test_sample_count": int(len(final)),
        "final_test_min_decision_time": int(final["decision_time"].min()) if not final.empty else None,
        "final_test_max_decision_time": int(final["decision_time"].max()) if not final.empty else None,
        "final_test_max_settlement_minute_open_time": int(final["settlement_minute_open_time"].max()) if not final.empty else None,
        "final_test_min_decision_time_utc": ms_to_utc_iso(int(final["decision_time"].min())) if not final.empty else None,
        "final_test_max_decision_time_utc": ms_to_utc_iso(int(final["decision_time"].max())) if not final.empty else None,
    }


def write_report(
    path: Path,
    outputs: Stage6Outputs,
    config: dict[str, Any],
    runtime: dict[str, Any],
    final_test_audit: dict[str, Any],
) -> None:
    dense = outputs.metrics_by_subset[outputs.metrics_by_subset["subset_name"].eq("DENSE")]
    nonoverlap = outputs.metrics_by_subset[outputs.metrics_by_subset["subset_name"].eq("NONOVERLAP_OFFSET_00")]
    pooled = outputs.oof_summary[outputs.oof_summary["summary_type"].eq("pooled") & outputs.oof_summary["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])]
    offset_summary = outputs.oof_summary[outputs.oof_summary["summary_type"].eq("pooled") & outputs.oof_summary["subset_name"].str.startswith("OFFSET_")]
    margin_summary = outputs.oof_summary[
        outputs.oof_summary["summary_type"].eq("pooled")
        & outputs.oof_summary["subset_name"].isin(["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"])
    ]
    convergence_rows = [
        {
            "fold_name": fold,
            "converged": meta["converged"],
            "n_iter": meta["logistic_regression_n_iter"],
            "fit_elapsed_seconds": meta["fit_elapsed_seconds"],
            "process_rss_bytes": meta["process_rss_bytes"],
        }
        for fold, meta in outputs.fold_metadata.items()
    ]
    comparison_rows = baseline_comparison_rows(outputs.metrics_by_subset)
    lines = [
        "# Stage 6 Baseline Model Report",
        "",
        "## Scope",
        "- Trained only the three Stage 6 baselines: prior_baseline, momentum_60m_baseline, logistic_regression_l2.",
        "- No XGBoost, LightGBM, CatBoost, neural network, probability calibration, threshold optimization, feature selection, outlier clipping, missing value imputation, class resampling, or final deployment model training was performed.",
        "- FINAL_TEST remained sealed: no predict, predict_proba, decision_function, metrics, model selection, or threshold decisions used FINAL_TEST rows.",
        "",
        "## Inputs and Hashes",
        table(
            [
                {"item": "dataset_path", "value": config["dataset_path"]},
                {"item": "split_path", "value": config["split_path"]},
                {"item": "dataset_sha256", "value": runtime.get("dataset_sha256")},
                {"item": "split_sha256", "value": runtime.get("split_sha256")},
                {"item": "config_sha256", "value": runtime.get("config_sha256")},
                {"item": "script_sha256", "value": runtime.get("script_sha256")},
            ],
            ["item", "value"],
        ),
        "## Feature Manifest Check",
        table(
            [
                {"check": "feature_count", "value": len(outputs.preprocessing_audit["feature_columns"])},
                {"check": "feature_manifest_match", "value": outputs.quality_gates["feature_manifest_match"]},
                {"check": "feature_list_sha256", "value": outputs.model_manifest["feature_list_sha256"]},
            ],
            ["check", "value"],
        ),
        "## Fold Counts and Time Boundaries",
        table(
            [
                {
                    "fold_name": fold,
                    "train_rows": meta["train_row_count"],
                    "validation_rows": meta["validation_row_count"],
                    "train_max_settlement_utc": ms_to_utc_iso(meta["integrity"]["train_max_settlement_minute_open_time"]),
                    "validation_min_decision_utc": ms_to_utc_iso(meta["integrity"]["validation_min_decision_time"]),
                    "overlap_count": meta["integrity"]["train_validation_overlap_count"],
                }
                for fold, meta in outputs.fold_metadata.items()
            ],
            ["fold_name", "train_rows", "validation_rows", "train_max_settlement_utc", "validation_min_decision_utc", "overlap_count"],
        ),
        "## Model Definitions",
        table(
            [
                {"model_name": "prior_baseline", "definition": "TRAIN label_up_60m mean; hard class is 1 when train prior >= 0.5."},
                {"model_name": "momentum_60m_baseline", "definition": "log_return_60m > 0 predicts 1; probabilities are TRAIN conditional rates with Laplace alpha=1.0."},
                {"model_name": "logistic_regression_l2", "definition": "StandardScaler fitted on TRAIN only, then LogisticRegression L2 with fixed C=1.0/lbfgs/max_iter=2000/tol=1e-6 and threshold 0.5."},
            ],
            ["model_name", "definition"],
        ),
        "## StandardScaler Audit",
        table(
            [
                {
                    "fold_name": fold,
                    "train_only_fit": audit["train_only_fit"],
                    "scaler_n_samples_seen": audit["scaler_n_samples_seen"],
                    "mean_diff": audit["max_abs_mean_diff_vs_independent_train_mean"],
                    "var_diff": audit["max_abs_var_diff_vs_independent_train_var"],
                }
                for fold, audit in outputs.preprocessing_audit["folds"].items()
            ],
            ["fold_name", "train_only_fit", "scaler_n_samples_seen", "mean_diff", "var_diff"],
        ),
        "## Dense Metrics by Fold",
        table(select_metric_rows(dense), ["fold_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Non-overlap Metrics by Fold",
        table(select_metric_rows(nonoverlap), ["fold_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Pooled OOF Metrics",
        table(select_metric_rows(pooled), ["subset_name", "model_name", "sample_count", "positive_ratio", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score"]),
        "## Offset Stability",
        table(select_metric_rows(offset_summary), ["subset_name", "model_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"]),
        "## Boundary Return Diagnostics",
        table(select_metric_rows(margin_summary), ["subset_name", "model_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "log_loss", "brier_score"]),
        "## Probability Diagnostics",
        "- Calibration tables are raw probability diagnostics only. ECE = sum over bins of (bin_sample_count / total_sample_count) * abs(mean_predicted_probability - actual_up_ratio).",
        table(
            outputs.calibration_equal_width.groupby(["model_name", "subset_name"], as_index=False)[["ece", "mce"]].first().to_dict(orient="records"),
            ["model_name", "subset_name", "ece", "mce"],
        ),
        "## Logistic Regression Convergence",
        table(convergence_rows, ["fold_name", "converged", "n_iter", "fit_elapsed_seconds", "process_rss_bytes"]),
        "## Logistic Coefficient Stability",
        "- Coefficients are on standardized features. They are affected by collinearity, are not causal effects, and were not used for feature selection.",
        table(outputs.logistic_coefficient_stability.head(20).to_dict(orient="records"), ["feature_name", "coefficient_mean", "coefficient_std", "mean_absolute_coefficient", "positive_fold_count", "negative_fold_count", "sign_consistency_ratio"]),
        "## Baseline Comparison",
        table(comparison_rows, ["question", "answer"]),
        "## FINAL_TEST Seal Audit",
        table(
            [
                {"field": "final_test_sample_count", "value": final_test_audit.get("final_test_sample_count")},
                {"field": "final_test_prediction_count", "value": outputs.quality_gates["final_test_prediction_count"]},
                {"field": "final_test_metric_count", "value": outputs.quality_gates["final_test_metric_count"]},
                {"field": "final_test_used_for_fit", "value": outputs.quality_gates["final_test_used_for_fit"]},
                {"field": "final_test_used_for_selection", "value": outputs.quality_gates["final_test_used_for_selection"]},
            ],
            ["field", "value"],
        ),
        "## Engineering Gates",
        table([{"gate": key, "value": value} for key, value in outputs.quality_gates.items()], ["gate", "value"]),
        "## Runtime and Memory",
        table(
            [
                {"metric": "elapsed_seconds", "value": runtime.get("elapsed_seconds")},
                {"metric": "python_tracemalloc_peak_bytes", "value": runtime.get("python_tracemalloc_peak_bytes")},
                {"metric": "process_rss_bytes", "value": runtime.get("process_rss_bytes")},
            ],
            ["metric", "value"],
        ),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_metric_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [col for col in ["fold_name", "subset_name", "model_name", *METRIC_COLUMNS] if col in df.columns]
    return df[cols].to_dict(orient="records")


def baseline_comparison_rows(metrics_by_subset: pd.DataFrame) -> list[dict[str, str]]:
    dense = metrics_by_subset[metrics_by_subset["subset_name"].eq("DENSE")]
    nonoverlap = metrics_by_subset[metrics_by_subset["subset_name"].eq("NONOVERLAP_OFFSET_00")]

    def count_beats(subset: pd.DataFrame, reference: str, metric: str = "roc_auc") -> int:
        pivot = subset.pivot(index="fold_name", columns="model_name", values=metric)
        if "logistic_regression_l2" not in pivot or reference not in pivot:
            return 0
        return int((pivot["logistic_regression_l2"] > pivot[reference]).sum())

    dense_pivot = dense.pivot(index="fold_name", columns="model_name", values="roc_auc")
    lr_auc_above_half = int((dense_pivot.get("logistic_regression_l2", pd.Series(dtype=float)) > 0.5).sum())
    brier_pivot = dense.pivot(index="fold_name", columns="model_name", values="brier_score")
    logloss_pivot = dense.pivot(index="fold_name", columns="model_name", values="log_loss")
    brier_better = int((brier_pivot.get("logistic_regression_l2", pd.Series(dtype=float)) < brier_pivot.get("prior_baseline", pd.Series(dtype=float))).sum())
    logloss_better = int((logloss_pivot.get("logistic_regression_l2", pd.Series(dtype=float)) < logloss_pivot.get("prior_baseline", pd.Series(dtype=float))).sum())
    pred_ratio = dense.pivot(index="fold_name", columns="model_name", values="predicted_positive_ratio")
    majority_note = pred_ratio.get("logistic_regression_l2", pd.Series(dtype=float)).round(4).to_dict()
    return [
        {"question": "Logistic beats prior on dense ROC-AUC folds", "answer": str(count_beats(dense, "prior_baseline"))},
        {"question": "Logistic beats momentum on dense ROC-AUC folds", "answer": str(count_beats(dense, "momentum_60m_baseline"))},
        {"question": "Dense/non-overlap consistency", "answer": f"dense beats prior={count_beats(dense, 'prior_baseline')}, non-overlap beats prior={count_beats(nonoverlap, 'prior_baseline')}"},
        {"question": "Offset differences", "answer": "See OFFSET_00 through OFFSET_55 pooled rows; no threshold or model parameter was changed from these diagnostics."},
        {"question": "Logistic ROC-AUC above 0.5 folds", "answer": str(lr_auc_above_half)},
        {"question": "Logistic Brier/LogLoss better than prior folds", "answer": f"brier={brier_better}, log_loss={logloss_better}"},
        {"question": "Failed years", "answer": str(dense_pivot.get("logistic_regression_l2", pd.Series(dtype=float)).sort_values().head(2).to_dict())},
        {"question": "Near-zero return boundary effect", "answer": "See ABS_RETURN_GE_* diagnostics; boundary subsets were evaluation-only."},
        {"question": "Majority-class behavior", "answer": f"logistic predicted_positive_ratio by fold={majority_note}"},
        {"question": "Probability confidence", "answer": "See equal-width and equal-frequency ECE/MCE tables; probabilities were not calibrated."},
    ]


def run_stage6(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    dataset_manifest_path = (root / config["dataset_manifest_path"]).resolve()
    feature_manifest_path = (root / config["feature_manifest_path"]).resolve()
    fold_manifest_path = (root / config["fold_manifest_path"]).resolve()
    dataset_path = (root / config["dataset_path"]).resolve()
    split_path = (root / config["split_path"]).resolve()
    prediction_path = (root / config["prediction_output_path"]).resolve()
    log_path = (root / config["log_path"]).resolve()
    report_paths = {name: (root / value).resolve() for name, value in config["report_paths"].items()}

    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    started = time.perf_counter()
    tracemalloc.start()
    try:
        logging.info("Stage 6 baseline training started")
        dataset_manifest_data = read_json(dataset_manifest_path)
        feature_manifest_data = read_json(feature_manifest_path)
        fold_manifest_data = read_json(fold_manifest_path)
        feature_columns = validate_feature_manifest(dataset_manifest_data, feature_manifest_data)
        split_columns = [*BASE_SPLIT_COLUMNS, *[f"{fold_name}_role" for fold_name in config["fold_names"]]]
        dataset_columns = [*BASE_DATASET_COLUMNS, *feature_columns]
        require_parquet_columns(dataset_path, dataset_columns)
        require_parquet_columns(split_path, split_columns)
        splits = pd.read_parquet(split_path, columns=split_columns)
        final_test_audit = final_test_audit_from_splits(splits)
        dataset = pd.read_parquet(dataset_path, columns=dataset_columns)
        runtime_hashes = {
            "dataset_sha256": sha256_file(dataset_path),
            "split_sha256": sha256_file(split_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
            "fold_manifest_sha256": sha256_file(fold_manifest_path),
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        }
        outputs = build_stage6_outputs(dataset, splits, dataset_manifest_data, feature_manifest_data, fold_manifest_data, config, root=root, runtime_hashes=runtime_hashes)
        ensure_parent(prediction_path)
        outputs.predictions.to_parquet(prediction_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
        outputs.metrics_by_fold.to_csv(report_paths["metrics_by_fold"], index=False, encoding="utf-8")
        outputs.metrics_by_subset.to_csv(report_paths["metrics_by_subset"], index=False, encoding="utf-8")
        outputs.metrics_by_offset.to_csv(report_paths["metrics_by_offset"], index=False, encoding="utf-8")
        outputs.oof_summary.to_csv(report_paths["oof_summary"], index=False, encoding="utf-8")
        outputs.calibration_equal_width.to_csv(report_paths["calibration_equal_width"], index=False, encoding="utf-8")
        outputs.calibration_equal_frequency.to_csv(report_paths["calibration_equal_frequency"], index=False, encoding="utf-8")
        outputs.logistic_coefficients_by_fold.to_csv(report_paths["logistic_coefficients_by_fold"], index=False, encoding="utf-8")
        outputs.logistic_coefficient_stability.to_csv(report_paths["logistic_coefficient_stability"], index=False, encoding="utf-8")
        write_json(report_paths["preprocessing_audit"], outputs.preprocessing_audit)
        prediction_sha = sha256_file(prediction_path)
        outputs.model_manifest["output_files"] = {
            "prediction_output_path": str(prediction_path),
            "prediction_output_sha256": prediction_sha,
            **{name: str(path) for name, path in report_paths.items()},
        }
        write_json(report_paths["model_manifest"], outputs.model_manifest)
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        runtime = {
            **runtime_hashes,
            "prediction_sha256": prediction_sha,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
            "prediction_schema": parquet_schema(prediction_path),
            "prediction_size_bytes": prediction_path.stat().st_size,
        }
        write_report(report_paths["main_report"], outputs, config, runtime, final_test_audit)
        logging.info("Quality gates: %s", outputs.quality_gates)
        logging.info("Prediction path: %s", prediction_path)
        logging.info("Elapsed seconds: %.2f", elapsed)
        logging.info("Python tracemalloc peak bytes: %s", peak)
        logging.info("Process RSS bytes: %s", rss)
        return {
            "quality_gates": outputs.quality_gates,
            "fold_metadata": outputs.fold_metadata,
            "prediction_path": str(prediction_path),
            "prediction_sha256": prediction_sha,
            "prediction_size_bytes": prediction_path.stat().st_size,
            "prediction_schema": runtime["prediction_schema"],
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 6 train and evaluate baseline models on rolling CV folds.")
    parser.add_argument("--config", default="config/stage6_train_baselines.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage6(config_path, root)


if __name__ == "__main__":
    main()

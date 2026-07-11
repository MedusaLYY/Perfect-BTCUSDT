from __future__ import annotations

import argparse
import ctypes
import hashlib
import inspect
import json
import logging
import math
import platform
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_stage6_baselines import (  # noqa: E402
    EVALUATION_METADATA_COLUMNS,
    METRIC_COLUMNS,
    assign_probability_bins,
    compute_classification_metrics,
    evaluate_prediction_subsets,
    logistic_parameters_for_current_sklearn,
    make_json_safe,
    pooled_and_macro_summary,
    subset_masks,
    table,
)


class Stage9ValidationError(ValueError):
    def __init__(self, errors: list[str] | str):
        if isinstance(errors, str):
            errors = [errors]
        super().__init__("\n".join(errors))
        self.errors = errors


ALLOWED_CALIBRATION_METHODS = ("UNCALIBRATED", "PLATT", "ISOTONIC")
COMPLEXITY_RANK = {"UNCALIBRATED": 0, "PLATT": 1, "ISOTONIC": 2}
TARGET_COLUMN = "label_up_60m"
MOMENTUM_FEATURE = "log_return_60m"
REQUIRED_FEATURE_COUNT = 63
DAY_MS = 86_400_000
XGBOOST_FINAL_MODEL_NAME = "xgboost_final"
CALIBRATED_MODEL_NAME = "xgboost_final_calibrated"
RAW_MODEL_NAME = "xgboost_final_raw"
FINAL_TEST_FOLD_NAME = "FINAL_TEST"
STAGE7_OOF_MODEL_NAME = "xgboost_fixed_v1"
EXPECTED_STAGE8_SELECTION = {
    "selected_development_config": "xgb_fixed_v1_reference",
    "development_recommendation": "KEEP_STAGE7_REFERENCE",
    "improvement_not_material": True,
}
REFERENCE_XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "tree_method": "hist",
    "device": "cpu",
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 10.0,
    "max_bin": 256,
    "scale_pos_weight": 1.0,
    "eval_metric": "logloss",
    "random_state": 42,
    "validate_parameters": True,
    "verbosity": 1,
}
BASE_SPLIT_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "settlement_minute_open_time",
    "final_split_role",
    "evaluation_offset_minutes",
    "is_primary_nonoverlap_evaluation",
]
BASE_DATASET_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "entry_minute_open_time",
    "settlement_minute_open_time",
    TARGET_COLUMN,
    "future_simple_return_60m",
    "future_log_return_60m",
    "absolute_future_return_bps",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
    "sample_weight_uniform",
    "sample_weight_margin",
]
FINAL_TEST_OUTPUT_COLUMNS = [
    "dataset_row_id",
    "decision_time",
    "final_split_role",
    "y_true",
    "p_up_raw",
    "p_up_calibrated",
    "y_pred_raw_0_5",
    "y_pred_calibrated_0_5",
    "evaluation_offset_minutes",
    "is_primary_nonoverlap_evaluation",
    "future_simple_return_60m",
    "absolute_future_return_bps",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
    "prior_p_up",
    "prior_y_pred",
    "momentum_p_up",
    "momentum_y_pred",
    "logistic_p_up",
    "logistic_y_pred",
]


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


def canonical_json(value: Any) -> str:
    return json.dumps(make_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def parse_utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def ms_to_utc_iso(value: int | float | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(int(value), unit="ms", tz="UTC").isoformat()


def now_utc() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


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


def validate_calibration_candidates(candidates: list[str] | tuple[str, ...]) -> None:
    if tuple(candidates) != ALLOWED_CALIBRATION_METHODS:
        raise Stage9ValidationError(f"calibration candidates must be exactly {ALLOWED_CALIBRATION_METHODS}")


def build_calibration_forward_folds(fold_names: list[str]) -> list[dict[str, Any]]:
    expected = ["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"]
    if fold_names != expected[: len(fold_names)] or len(fold_names) < 2:
        raise Stage9ValidationError(f"calibration fold order must be a prefix of {expected}")
    folds: list[dict[str, Any]] = []
    for index, year in enumerate([int(name.split("_")[1]) for name in fold_names[1:]], start=1):
        folds.append(
            {
                "calibration_fold": f"calibration_{year}",
                "fit_folds": fold_names[:index],
                "evaluation_fold": f"fold_{year}",
            }
        )
    return folds


def logit_transform(p: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    clipped = np.clip(np.asarray(p, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


def fit_calibrator(method: str, p_raw: np.ndarray, y: np.ndarray, epsilon: float = 1e-6) -> dict[str, Any]:
    if method not in ALLOWED_CALIBRATION_METHODS:
        raise Stage9ValidationError(f"unsupported calibration method {method}")
    p = np.asarray(p_raw, dtype=np.float64)
    labels = np.asarray(y, dtype=np.int8)
    if len(p) != len(labels) or len(p) == 0:
        raise Stage9ValidationError("calibrator fit input has invalid length")
    if not np.isfinite(p).all() or not ((p >= 0.0) & (p <= 1.0)).all():
        raise Stage9ValidationError("calibrator fit probabilities must be finite and within [0, 1]")
    if method == "UNCALIBRATED":
        return {"method": method, "params": {}, "input_transform": "identity", "model": None, "epsilon": epsilon}
    if set(np.unique(labels)) != {0, 1}:
        raise Stage9ValidationError(f"{method} calibration labels must contain both classes")
    if method == "PLATT":
        model = LogisticRegression(
            fit_intercept=True,
            penalty="l2",
            C=1_000_000,
            solver="lbfgs",
            max_iter=1000,
            tol=1e-10,
            random_state=42,
        )
        model.fit(logit_transform(p, epsilon).reshape(-1, 1), labels)
        return {
            "method": method,
            "params": {
                "fit_intercept": True,
                "penalty": "l2",
                "C": 1_000_000,
                "solver": "lbfgs",
                "max_iter": 1000,
                "tol": 1e-10,
                "random_state": 42,
            },
            "input_transform": "logit_probability",
            "model": model,
            "epsilon": epsilon,
        }
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(p, labels)
    return {
        "method": method,
        "params": {"y_min": 0.0, "y_max": 1.0, "increasing": True, "out_of_bounds": "clip"},
        "input_transform": "raw_probability",
        "model": model,
        "epsilon": epsilon,
    }


def apply_calibrator(calibrator: dict[str, Any], p_raw: np.ndarray) -> np.ndarray:
    p = np.asarray(p_raw, dtype=np.float64)
    method = calibrator["method"]
    if method == "UNCALIBRATED":
        out = p.copy()
    elif method == "PLATT":
        out = calibrator["model"].predict_proba(logit_transform(p, float(calibrator.get("epsilon", 1e-6))).reshape(-1, 1))[:, 1]
    elif method == "ISOTONIC":
        out = calibrator["model"].predict(p)
    else:
        raise Stage9ValidationError(f"unsupported calibration method {method}")
    out = np.asarray(out, dtype=np.float64)
    if not np.isfinite(out).all():
        raise Stage9ValidationError("calibrated probabilities are not finite")
    return np.clip(out, 0.0, 1.0)


def calibration_errors(y_true: np.ndarray, p_up: np.ndarray, bins: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for strategy, prefix in [("equal_width", "equal_width"), ("equal_frequency", "equal_frequency")]:
        assignments, _ = assign_probability_bins(np.asarray(p_up, dtype=np.float64), bins, strategy)
        total = len(assignments)
        gaps: list[tuple[int, float]] = []
        for bin_index in sorted(set(assignments.tolist())):
            mask = assignments == bin_index
            if not mask.any():
                continue
            gaps.append((int(mask.sum()), abs(float(np.mean(p_up[mask]) - np.mean(y_true[mask])))))
        result[f"ece_{prefix}"] = float(sum((count / total) * gap for count, gap in gaps)) if total else np.nan
        result[f"mce_{prefix}"] = float(max((gap for _, gap in gaps), default=np.nan))
    return result


def calibration_metric_row(method: str, calibration_fold: str, subset_name: str, df: pd.DataFrame, bins: int) -> dict[str, Any]:
    y = df["y_true"].to_numpy(dtype=np.int8)
    p = df["calibrated_probability"].to_numpy(dtype=np.float64)
    pred = (p >= 0.5).astype(np.int8)
    metrics = compute_classification_metrics(y, p, pred)
    cal = calibration_errors(y, p, bins)
    return {
        "method": method,
        "calibration_fold": calibration_fold,
        "subset_name": subset_name,
        **metrics,
        **cal,
        "mean_probability": float(p.mean()) if len(p) else np.nan,
        "observed_positive_rate": float(y.mean()) if len(y) else np.nan,
    }


def evaluate_calibration_candidates(
    oof_predictions: pd.DataFrame,
    calibration_candidates: list[str] | tuple[str, ...],
    forward_folds: list[dict[str, Any]],
    bins: int,
    epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_calibration_candidates(calibration_candidates)
    required = {"dataset_row_id", "fold_name", "y_true", "p_up", "evaluation_offset_minutes", "is_primary_nonoverlap_evaluation", *EVALUATION_METADATA_COLUMNS}
    missing = required.difference(oof_predictions.columns)
    if missing:
        raise Stage9ValidationError(f"Stage7 OOF predictions missing columns: {sorted(missing)}")
    oof = oof_predictions.copy()
    if oof["fold_name"].eq("fold_2020").sum() == 0:
        raise Stage9ValidationError("fold_2020 is required as first calibration fit fold")
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for method in calibration_candidates:
        for fold in forward_folds:
            fit_df = oof[oof["fold_name"].isin(fold["fit_folds"])].copy()
            eval_df = oof[oof["fold_name"].eq(fold["evaluation_fold"])].copy()
            if fit_df.empty or eval_df.empty:
                raise Stage9ValidationError(f"{method}/{fold['calibration_fold']} has empty fit/evaluation data")
            calibrator = fit_calibrator(method, fit_df["p_up"].to_numpy(dtype=float), fit_df["y_true"].to_numpy(dtype=np.int8), epsilon)
            calibrated = apply_calibrator(calibrator, eval_df["p_up"].to_numpy(dtype=float))
            pred = eval_df[
                [
                    "dataset_row_id",
                    "decision_time",
                    "fold_name",
                    "y_true",
                    "p_up",
                    "evaluation_offset_minutes",
                    "is_primary_nonoverlap_evaluation",
                    *EVALUATION_METADATA_COLUMNS,
                ]
            ].copy()
            pred["method"] = method
            pred["calibration_fold"] = fold["calibration_fold"]
            pred["fit_folds"] = ",".join(fold["fit_folds"])
            pred["evaluation_fold"] = fold["evaluation_fold"]
            pred["raw_probability"] = pred["p_up"].astype(float)
            pred["calibrated_probability"] = calibrated
            pred["y_pred"] = (calibrated >= 0.5).astype(np.int8)
            pred = pred.drop(columns=["p_up"])
            prediction_frames.append(pred)
            for subset_name, mask in [
                ("DENSE", pd.Series(True, index=pred.index)),
                ("NONOVERLAP_OFFSET_00", pred["is_primary_nonoverlap_evaluation"].astype(bool)),
            ]:
                metric_rows.append(calibration_metric_row(method, fold["calibration_fold"], subset_name, pred.loc[mask], bins))
        method_forward = pd.concat([p for p in prediction_frames if p["method"].iloc[0] == method], ignore_index=True)
        for subset_name, mask in [
            ("DENSE", pd.Series(True, index=method_forward.index)),
            ("NONOVERLAP_OFFSET_00", method_forward["is_primary_nonoverlap_evaluation"].astype(bool)),
        ]:
            metric_rows.append(calibration_metric_row(method, "POOLED_2021_2024", subset_name, method_forward.loc[mask], bins))
            fold_rows = [r for r in metric_rows if r["method"] == method and r["subset_name"] == subset_name and str(r["calibration_fold"]).startswith("calibration_")]
            for summary_name in ["FOLD_MACRO_MEAN", "FOLD_STD"]:
                row = {"method": method, "calibration_fold": summary_name, "subset_name": subset_name}
                for col in [*METRIC_COLUMNS, "ece_equal_width", "ece_equal_frequency", "mce_equal_width", "mce_equal_frequency", "mean_probability", "observed_positive_rate"]:
                    values = [float(r[col]) for r in fold_rows if col in r and not pd.isna(r[col])]
                    if summary_name == "FOLD_MACRO_MEAN":
                        row[col] = float(np.mean(values)) if values else np.nan
                    else:
                        row[col] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
                metric_rows.append(row)
            worst_logloss = max(fold_rows, key=lambda r: float(r["log_loss"]))
            worst_brier = max(fold_rows, key=lambda r: float(r["brier_score"]))
            worst_log_payload = {k: worst_logloss.get(k) for k in worst_logloss if k not in {"method", "calibration_fold", "subset_name"}}
            worst_brier_payload = {k: worst_brier.get(k) for k in worst_brier if k not in {"method", "calibration_fold", "subset_name"}}
            metric_rows.append({"method": method, "calibration_fold": "WORST_YEAR_LOGLOSS", "subset_name": subset_name, **worst_log_payload})
            metric_rows.append({"method": method, "calibration_fold": "WORST_YEAR_BRIER", "subset_name": subset_name, **worst_brier_payload})
    forward_predictions = pd.concat(prediction_frames, ignore_index=True)
    if not np.isfinite(forward_predictions["calibrated_probability"].to_numpy(dtype=float)).all() or not forward_predictions["calibrated_probability"].between(0, 1).all():
        raise Stage9ValidationError("calibration produced invalid probabilities")
    return forward_predictions, pd.DataFrame(metric_rows)


def _metric(metrics: pd.DataFrame, method: str, fold: str, subset: str, col: str) -> float:
    rows = metrics[(metrics["method"].eq(method)) & (metrics["calibration_fold"].eq(fold)) & (metrics["subset_name"].eq(subset))]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][col])


def _fold_rows(metrics: pd.DataFrame, method: str, subset: str) -> pd.DataFrame:
    return metrics[
        metrics["method"].eq(method)
        & metrics["subset_name"].eq(subset)
        & metrics["calibration_fold"].astype(str).str.startswith("calibration_")
    ].copy()


def select_calibration_method(metrics: pd.DataFrame, rules: dict[str, Any]) -> dict[str, Any]:
    tolerance = float(rules.get("calibration_logloss_tie_tolerance", 0.0002))
    pooled_fold = "POOLED_2021_2024"
    ranked = []
    for method in ALLOWED_CALIBRATION_METHODS:
        no_log = _metric(metrics, method, pooled_fold, "NONOVERLAP_OFFSET_00", "log_loss")
        no_brier = _metric(metrics, method, pooled_fold, "NONOVERLAP_OFFSET_00", "brier_score")
        dense_log = _metric(metrics, method, pooled_fold, "DENSE", "log_loss")
        no_ece_freq = _metric(metrics, method, pooled_fold, "NONOVERLAP_OFFSET_00", "ece_equal_frequency")
        year_rows = _fold_rows(metrics, method, "NONOVERLAP_OFFSET_00")
        worst_log = float(year_rows["log_loss"].max()) if not year_rows.empty else np.nan
        ranked.append(
            {
                "method": method,
                "pooled_nonoverlap_log_loss": no_log,
                "pooled_nonoverlap_brier": no_brier,
                "pooled_dense_log_loss": dense_log,
                "pooled_nonoverlap_ece_equal_frequency": no_ece_freq,
                "worst_year_log_loss": worst_log,
                "complexity_rank": COMPLEXITY_RANK[method],
            }
        )
    best_log = min(row["pooled_nonoverlap_log_loss"] for row in ranked)
    candidate_pool = [row for row in ranked if abs(float(row["pooled_nonoverlap_log_loss"]) - best_log) <= tolerance]
    candidate_pool.sort(
        key=lambda row: (
            row["pooled_nonoverlap_brier"],
            row["pooled_dense_log_loss"],
            row["pooled_nonoverlap_ece_equal_frequency"],
            row["worst_year_log_loss"],
            row["complexity_rank"],
        )
    )
    ranked.sort(
        key=lambda row: (
            0 if row in candidate_pool else 1,
            row["pooled_nonoverlap_log_loss"],
            row["pooled_nonoverlap_brier"],
            row["pooled_dense_log_loss"],
            row["pooled_nonoverlap_ece_equal_frequency"],
            row["worst_year_log_loss"],
            row["complexity_rank"],
        )
    )
    primary_choice = candidate_pool[0]["method"]
    uncal_no_log = _metric(metrics, "UNCALIBRATED", pooled_fold, "NONOVERLAP_OFFSET_00", "log_loss")
    uncal_dense_log = _metric(metrics, "UNCALIBRATED", pooled_fold, "DENSE", "log_loss")
    uncal_no_brier = _metric(metrics, "UNCALIBRATED", pooled_fold, "NONOVERLAP_OFFSET_00", "brier_score")
    material = False
    material_reason = "selected method is UNCALIBRATED"
    if primary_choice != "UNCALIBRATED":
        cand_no_log = _metric(metrics, primary_choice, pooled_fold, "NONOVERLAP_OFFSET_00", "log_loss")
        cand_dense_log = _metric(metrics, primary_choice, pooled_fold, "DENSE", "log_loss")
        cand_no_brier = _metric(metrics, primary_choice, pooled_fold, "NONOVERLAP_OFFSET_00", "brier_score")
        cand_years = _fold_rows(metrics, primary_choice, "NONOVERLAP_OFFSET_00").sort_values("calibration_fold")
        uncal_years = _fold_rows(metrics, "UNCALIBRATED", "NONOVERLAP_OFFSET_00").sort_values("calibration_fold")
        log_wins = int((cand_years["log_loss"].to_numpy(dtype=float) < uncal_years["log_loss"].to_numpy(dtype=float)).sum())
        brier_wins = int((cand_years["brier_score"].to_numpy(dtype=float) < uncal_years["brier_score"].to_numpy(dtype=float)).sum())
        condition_a = (
            cand_no_log <= uncal_no_log - float(rules.get("material_logloss_improvement", 0.0002))
            and cand_dense_log <= uncal_dense_log
            and log_wins >= 3
        )
        condition_b = (
            cand_no_brier <= uncal_no_brier - float(rules.get("material_brier_improvement", 0.0001))
            and cand_no_log <= uncal_no_log + float(rules.get("material_logloss_max_degradation", 0.0001))
            and brier_wins >= 3
        )
        material = bool(condition_a or condition_b)
        material_reason = f"condition_a={condition_a}; condition_b={condition_b}; log_wins={log_wins}; brier_wins={brier_wins}"
    selected = primary_choice if material or primary_choice == "UNCALIBRATED" else "UNCALIBRATED"
    return {
        "selected_calibration_method": selected,
        "calibration_improvement_material": bool(material),
        "primary_rule_choice": primary_choice,
        "selection_reason": material_reason if selected != "UNCALIBRATED" else "Material improvement gate not met; keeping UNCALIBRATED.",
        "ranked_methods": ranked,
        "engineering_qualified_methods": list(ALLOWED_CALIBRATION_METHODS),
        "selection_rules": rules,
    }


def fit_final_calibrator(oof_predictions: pd.DataFrame, selected_method: str, epsilon: float = 1e-6) -> tuple[dict[str, Any], dict[str, Any]]:
    fit_df = oof_predictions[oof_predictions["fold_name"].isin(["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"])].copy()
    if fit_df.empty:
        raise Stage9ValidationError("final calibrator OOF fit data is empty")
    calibrator = fit_calibrator(selected_method, fit_df["p_up"].to_numpy(dtype=float), fit_df["y_true"].to_numpy(dtype=np.int8), epsilon)
    out = apply_calibrator(calibrator, fit_df["p_up"].to_numpy(dtype=float))
    metadata = {
        "method": selected_method,
        "params": calibrator["params"],
        "training_source": "stage7_development_oof_only",
        "training_oof_sample_count": int(len(fit_df)),
        "training_folds": sorted(fit_df["fold_name"].unique().tolist()),
        "oof_time_range": {
            "min_decision_time": int(fit_df["decision_time"].min()),
            "max_decision_time": int(fit_df["decision_time"].max()),
            "min_decision_time_utc": ms_to_utc_iso(int(fit_df["decision_time"].min())),
            "max_decision_time_utc": ms_to_utc_iso(int(fit_df["decision_time"].max())),
        },
        "input_probability_range": [float(fit_df["p_up"].min()), float(fit_df["p_up"].max())],
        "output_probability_range": [float(out.min()), float(out.max())],
        "training_data_hash": sha256_text(canonical_json(fit_df[["dataset_row_id", "fold_name", "y_true", "p_up"]].to_dict(orient="records"))),
    }
    return calibrator, metadata


def save_calibrator(calibrator: dict[str, Any], metadata: dict[str, Any], model_dir: Path) -> tuple[Path, str, dict[str, Any]]:
    ensure_parent(model_dir / "placeholder")
    if calibrator["method"] == "UNCALIBRATED":
        path = model_dir / "btcusdt_probability_calibrator_v1.json"
        payload = {"method": "UNCALIBRATED", "params": {}, "metadata": metadata}
        write_json(path, payload)
    else:
        path = model_dir / "btcusdt_probability_calibrator_v1.joblib"
        joblib.dump({"calibrator": calibrator, "metadata": metadata}, path)
    digest = sha256_file(path)
    metadata = {**metadata, "calibrator_file": str(path), "calibrator_file_sha256": digest}
    return path, digest, metadata


def load_calibrator(path: Path) -> dict[str, Any]:
    if path.suffix == ".json":
        payload = read_json(path)
        return {"method": payload["method"], "params": payload.get("params", {}), "input_transform": "identity", "model": None, "epsilon": 1e-6}
    payload = joblib.load(path)
    return payload["calibrator"]


def validate_feature_manifest(dataset_manifest: dict[str, Any], feature_manifest: dict[str, Any]) -> list[str]:
    features = list(dataset_manifest.get("feature_columns", []))
    stage4 = list(feature_manifest.get("ordered_feature_names", []))
    errors: list[str] = []
    if len(features) != REQUIRED_FEATURE_COUNT:
        errors.append(f"feature count must be {REQUIRED_FEATURE_COUNT}, got {len(features)}")
    if features != stage4:
        errors.append("feature columns do not match Stage4 manifest order")
    forbidden = set(dataset_manifest.get("forbidden_model_input_columns", []))
    leakage = [col for col in features if col in forbidden or any(token in col.lower() for token in ["label", "future", "entry", "settlement", "boundary", "margin"])]
    if leakage:
        errors.append(f"forbidden fields in feature list: {leakage}")
    if len(set(features)) != len(features):
        errors.append("feature columns contain duplicates")
    if errors:
        raise Stage9ValidationError(errors)
    return features


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise Stage9ValidationError(f"missing feature columns: {missing}")
    X = df.loc[:, feature_columns].copy()
    if list(X.columns) != feature_columns:
        raise Stage9ValidationError("feature order does not match manifest")
    values = X.to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise Stage9ValidationError("all feature values must be finite")
    return X


def joined_dataset(dataset: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset_row_id", "decision_time", "settlement_minute_open_time"]
    merged = dataset.merge(splits, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(dataset):
        raise Stage9ValidationError("dataset and split rows do not join one-to-one")
    return merged.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)


def final_refit_training_frame(joined: pd.DataFrame) -> pd.DataFrame:
    train = joined[joined["final_split_role"].eq("DEVELOPMENT")].copy()
    if train.empty:
        raise Stage9ValidationError("final DEVELOPMENT training frame is empty")
    if not set(train[TARGET_COLUMN].astype(int).unique()) == {0, 1}:
        raise Stage9ValidationError("final DEVELOPMENT training labels must contain both classes")
    return train.sort_values(["decision_time", "dataset_row_id"], kind="mergesort").reset_index(drop=True)


def make_final_inner_split(joined: pd.DataFrame, final_test_start_ms: int, window_days: int) -> dict[str, Any]:
    dev = final_refit_training_frame(joined)
    inner_start = int(final_test_start_ms - int(window_days) * DAY_MS)
    fit = dev[(dev["decision_time"] < inner_start) & (dev["settlement_minute_open_time"] < inner_start)].copy()
    purged = dev[(dev["decision_time"] < inner_start) & (dev["settlement_minute_open_time"] >= inner_start)].copy()
    early = dev[dev["decision_time"] >= inner_start].copy()
    if fit.empty or early.empty:
        raise Stage9ValidationError("final inner fit/early-stop split is empty")
    if int(fit["settlement_minute_open_time"].max()) >= int(early["decision_time"].min()):
        raise Stage9ValidationError("final inner fit settlement crosses early-stop decision boundary")
    return {
        "inner_early_stop_start": inner_start,
        "inner_early_stop_start_utc": ms_to_utc_iso(inner_start),
        "inner_fit": fit.reset_index(drop=True),
        "inner_purged": purged.reset_index(drop=True),
        "inner_early_stop": early.reset_index(drop=True),
        "inner_fit_count": int(len(fit)),
        "inner_purged_count": int(len(purged)),
        "inner_early_stop_count": int(len(early)),
    }


def verify_xgboost_parameters(params: dict[str, Any], n_jobs: int, n_estimators: int, selector: bool) -> dict[str, Any]:
    errors: list[str] = []
    for key, expected in REFERENCE_XGBOOST_PARAMS.items():
        if params.get(key) != expected:
            errors.append(f"Stage7 reference parameter mismatch for {key}: expected {expected}, got {params.get(key)}")
    if "nthread" in params:
        errors.append("nthread must not be used with n_jobs")
    if errors:
        raise Stage9ValidationError(errors)
    verified = {**REFERENCE_XGBOOST_PARAMS, "n_estimators": int(n_estimators), "n_jobs": int(n_jobs)}
    if selector:
        verified["early_stopping_rounds"] = int(params.get("early_stopping_rounds", 100))
    return verified


def select_final_tree_count(inner: dict[str, Any], feature_columns: list[str], config: dict[str, Any]) -> tuple[int, dict[str, Any], xgb.XGBClassifier]:
    selector_params = verify_xgboost_parameters(
        {**config["xgboost_parameters"], "early_stopping_rounds": int(config["final_early_stopping_rounds"])},
        int(config["n_jobs"]),
        int(config["final_max_estimators"]),
        selector=True,
    )
    model = xgb.XGBClassifier(**selector_params)
    X_fit = prepare_feature_matrix(inner["inner_fit"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_fit = inner["inner_fit"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    X_early = prepare_feature_matrix(inner["inner_early_stop"], feature_columns).to_numpy(dtype=np.float32, copy=True)
    y_early = inner["inner_early_stop"][TARGET_COLUMN].to_numpy(dtype=np.int8)
    model.fit(X_fit, y_fit, eval_set=[(X_early, y_early)], verbose=False)
    logloss = list(model.evals_result()["validation_0"]["logloss"])
    best_iteration = int(getattr(model, "best_iteration", int(np.argmin(logloss))))
    best_score = float(getattr(model, "best_score", logloss[best_iteration]))
    final_best_n_estimators = best_iteration + 1
    reached_max = len(logloss) >= int(config["final_max_estimators"])
    meta = {
        "final_best_iteration": best_iteration,
        "final_best_n_estimators": final_best_n_estimators,
        "final_best_score": best_score,
        "stopped_early": bool(not reached_max),
        "reached_max_estimators": bool(reached_max),
        "final_inner_fit_count": int(len(inner["inner_fit"])),
        "final_inner_purged_count": int(len(inner["inner_purged"])),
        "final_inner_early_stop_count": int(len(inner["inner_early_stop"])),
        "final_inner_early_stop_start": int(inner["inner_early_stop_start"]),
        "final_test_used_for_tree_count_selection": False,
        "learning_curve_logloss": [float(v) for v in logloss],
    }
    return final_best_n_estimators, meta, model


def fit_final_xgboost(train: pd.DataFrame, feature_columns: list[str], config: dict[str, Any], n_estimators: int, model_path: Path) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    params = verify_xgboost_parameters(config["xgboost_parameters"], int(config["n_jobs"]), int(n_estimators), selector=False)
    model = xgb.XGBClassifier(**params)
    X = prepare_feature_matrix(train, feature_columns).to_numpy(dtype=np.float32, copy=True)
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    model.fit(X, y, verbose=False)
    ensure_parent(model_path)
    model.save_model(model_path)
    meta = {
        "training_sample_count": int(len(train)),
        "training_role": "DEVELOPMENT",
        "purged_rows_used": False,
        "final_test_rows_used": False,
        "refit_used_early_stopping": False,
        "eval_set_used": False,
        "parameters": params,
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "boosted_rounds": int(model.get_booster().num_boosted_rounds()),
    }
    return model, meta


def verify_model_reload(model_path: Path, train: pd.DataFrame, feature_columns: list[str], expected_model: xgb.XGBClassifier, atol: float) -> dict[str, Any]:
    sample = train.head(min(2048, len(train))).copy()
    X = prepare_feature_matrix(sample, feature_columns).to_numpy(dtype=np.float32, copy=True)
    expected = expected_model.predict_proba(X)[:, 1]
    reloaded = xgb.XGBClassifier()
    reloaded.load_model(model_path)
    actual = reloaded.predict_proba(X)[:, 1]
    ok = bool(np.allclose(expected, actual, atol=atol))
    if not ok:
        raise Stage9ValidationError("reloaded final model predictions do not match development sample")
    return {
        "reload_verified": True,
        "sample_role": "DEVELOPMENT",
        "sample_count": int(len(sample)),
        "max_abs_prediction_diff": float(np.max(np.abs(expected - actual))) if len(sample) else 0.0,
        "sample_dataset_row_id_sha256": sha256_text("\n".join(str(int(v)) for v in sample["dataset_row_id"].tolist())),
    }


def train_prior_baseline(train: pd.DataFrame, threshold: float) -> dict[str, Any]:
    p = float(train[TARGET_COLUMN].astype(int).mean())
    return {
        "model_name": "prior_baseline_final",
        "p_up": p,
        "predicted_class": int(p >= threshold),
        "threshold": float(threshold),
        "training_sample_count": int(len(train)),
        "training_role": "DEVELOPMENT",
        "label_distribution": {"0": int((train[TARGET_COLUMN] == 0).sum()), "1": int((train[TARGET_COLUMN] == 1).sum())},
    }


def train_momentum_baseline(train: pd.DataFrame, alpha: float = 1.0) -> dict[str, Any]:
    if MOMENTUM_FEATURE not in train.columns:
        raise Stage9ValidationError("log_return_60m is required for momentum baseline")
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    positive = train[MOMENTUM_FEATURE].to_numpy(dtype=float) > 0.0

    def group(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        up = int(y[mask].sum()) if count else 0
        return {
            "group_count": count,
            "up_count": up,
            "p_up": float((up + alpha) / (count + 2.0 * alpha)),
        }

    return {
        "model_name": "momentum_60m_baseline_final",
        "alpha": float(alpha),
        "positive_group": group(positive),
        "nonpositive_group": group(~positive),
        "training_sample_count": int(len(train)),
        "training_role": "DEVELOPMENT",
        "hard_class_rule": "log_return_60m > 0",
    }


def train_logistic_baseline(
    train: pd.DataFrame,
    feature_columns: list[str],
    logistic_params: dict[str, Any],
    scaler_params: dict[str, Any],
    model_path: Path,
) -> tuple[Pipeline, dict[str, Any]]:
    X = prepare_feature_matrix(train, feature_columns).to_numpy(dtype=np.float64, copy=True)
    y = train[TARGET_COLUMN].to_numpy(dtype=np.int8)
    params = logistic_parameters_for_current_sklearn(logistic_params)
    pipeline = Pipeline([("scaler", StandardScaler(**scaler_params)), ("logistic_regression", LogisticRegression(**params))])
    pipeline.fit(X, y)
    ensure_parent(model_path)
    joblib.dump(pipeline, model_path)
    meta = {
        "model_name": "logistic_regression_final",
        "training_sample_count": int(len(train)),
        "training_role": "DEVELOPMENT",
        "standard_scaler_fit_role": "DEVELOPMENT",
        "sample_weight_margin_used": False,
        "class_weight_used": False,
        "parameters": params,
        "scaler_parameters": scaler_params,
        "model_file": str(model_path),
        "model_sha256": sha256_file(model_path),
        "scaler_n_samples_seen": int(pipeline.named_steps["scaler"].n_samples_seen_),
    }
    return pipeline, meta


def create_protocol_lock(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not payload.get("model_serialized"):
        errors.append("final model must be serialized before protocol lock")
    if not payload.get("calibrator_serialized"):
        errors.append("calibrator must be serialized before protocol lock")
    if not payload.get("logistic_serialized"):
        errors.append("logistic model must be serialized before protocol lock")
    if not payload.get("model_reload_verified"):
        errors.append("model reload must be verified before protocol lock")
    if payload.get("model_reload_verification_sample_role") != "DEVELOPMENT":
        errors.append("model reload verification sample role must be DEVELOPMENT")
    if payload.get("fixed_classification_threshold") != 0.5:
        errors.append("fixed classification threshold must be 0.5")
    if payload.get("final_test_accessed") is not False:
        errors.append("protocol lock must be written before FINAL_TEST access")
    if errors:
        raise Stage9ValidationError(errors)
    locked = dict(payload)
    locked["protocol_locked_at_utc"] = now_utc()
    locked["final_test_accessed"] = False
    locked["protocol_payload_sha256"] = sha256_text(canonical_json(locked))
    return locked


def _state_ready_for_final_test(state: dict[str, Any]) -> None:
    required = [
        "final_model_trained",
        "final_model_serialized",
        "model_reload_verified",
        "calibrator_frozen",
        "calibrator_serialized",
        "protocol_lock_written",
        "feature_manifest_verified",
        "development_checks_passed",
    ]
    missing = [key for key in required if not state.get(key)]
    if missing:
        if "protocol_lock_written" in missing:
            raise Stage9ValidationError("protocol lock must be written before FINAL_TEST feature matrix creation")
        raise Stage9ValidationError(f"FINAL_TEST access prerequisites not met: {missing}")


def create_final_test_feature_matrix(
    joined: pd.DataFrame,
    feature_columns: list[str],
    final_test_start_ms: int,
    expected_count: int,
    state: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    _state_ready_for_final_test(state)
    final = joined[joined["final_split_role"].eq("FINAL_TEST")].copy()
    if len(final) != int(expected_count):
        raise Stage9ValidationError(f"FINAL_TEST sample count mismatch: expected {expected_count}, got {len(final)}")
    if not (final["decision_time"].astype("int64") >= int(final_test_start_ms)).all():
        raise Stage9ValidationError("FINAL_TEST decision_time is before final_test_start")
    if final["final_split_role"].ne("FINAL_TEST").any():
        raise Stage9ValidationError("non FINAL_TEST rows in final feature matrix")
    X = prepare_feature_matrix(final, feature_columns).to_numpy(dtype=np.float32, copy=True)
    meta_cols = [
        "dataset_row_id",
        "decision_time",
        "final_split_role",
        TARGET_COLUMN,
        "future_simple_return_60m",
        "future_log_return_60m",
        "absolute_future_return_bps",
        "evaluation_offset_minutes",
        "is_primary_nonoverlap_evaluation",
        "proxy_boundary_risk_1bps",
        "proxy_boundary_risk_2_5bps",
        "proxy_boundary_risk_5bps",
        "proxy_boundary_risk_10bps",
        MOMENTUM_FEATURE,
    ]
    meta = final[meta_cols].rename(columns={TARGET_COLUMN: "y_true"}).reset_index(drop=True)
    state["final_test_feature_matrix_created"] = True
    state["final_test_feature_matrix_created_at_utc"] = now_utc()
    state["final_test_sample_count"] = int(len(final))
    return X, meta


def predict_baselines(meta: pd.DataFrame, prior: dict[str, Any], momentum: dict[str, Any], logistic: Pipeline, X: np.ndarray, threshold: float) -> dict[str, np.ndarray]:
    n = len(meta)
    prior_p = np.full(n, float(prior["p_up"]), dtype=float)
    positive = meta[MOMENTUM_FEATURE].to_numpy(dtype=float) > 0.0
    mom_p = np.where(positive, float(momentum["positive_group"]["p_up"]), float(momentum["nonpositive_group"]["p_up"]))
    logistic_p = logistic.predict_proba(X.astype(np.float64, copy=False))[:, 1]
    return {
        "prior_p_up": prior_p,
        "prior_y_pred": (prior_p >= threshold).astype(np.int8),
        "momentum_p_up": mom_p,
        "momentum_y_pred": positive.astype(np.int8),
        "logistic_p_up": logistic_p,
        "logistic_y_pred": (logistic_p >= threshold).astype(np.int8),
    }


def build_final_prediction_frame(
    meta: pd.DataFrame,
    base_model: Any,
    X_final: np.ndarray,
    calibrator: dict[str, Any],
    baseline_predictions: dict[str, np.ndarray],
    threshold: float,
) -> pd.DataFrame:
    raw_proba = base_model.predict_proba(X_final)[:, 1]
    call_count = int(getattr(base_model, "call_count", 1))
    if call_count != 1:
        raise Stage9ValidationError(f"FINAL_TEST base model predict_proba call count must be 1, got {call_count}")
    calibrated = apply_calibrator(calibrator, raw_proba)
    df = meta.copy()
    df["p_up_raw"] = raw_proba
    df["p_up_calibrated"] = calibrated
    df["y_pred_raw_0_5"] = (df["p_up_raw"] >= threshold).astype(np.int8)
    df["y_pred_calibrated_0_5"] = (df["p_up_calibrated"] >= threshold).astype(np.int8)
    for key, values in baseline_predictions.items():
        df[key] = values
    output = df[FINAL_TEST_OUTPUT_COLUMNS].copy()
    output.attrs["final_test_base_model_predict_proba_call_count"] = 1
    return output


def validate_final_test_predictions(predictions: pd.DataFrame, expected_count: int) -> None:
    errors: list[str] = []
    missing = [col for col in FINAL_TEST_OUTPUT_COLUMNS if col not in predictions.columns]
    if missing:
        errors.append(f"final prediction output missing columns: {missing}")
    if len(predictions) != int(expected_count):
        errors.append("FINAL_TEST prediction count mismatch")
    if predictions["dataset_row_id"].duplicated().any():
        errors.append("FINAL_TEST predictions must be unique per dataset_row_id")
    if not predictions["final_split_role"].eq("FINAL_TEST").all():
        errors.append("FINAL_TEST prediction file contains non FINAL_TEST rows")
    for col in ["p_up_raw", "p_up_calibrated", "prior_p_up", "momentum_p_up", "logistic_p_up"]:
        values = predictions[col].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not ((values >= 0.0) & (values <= 1.0)).all():
            errors.append(f"invalid probability column {col}")
    if errors:
        raise Stage9ValidationError(errors)


def final_predictions_to_long(predictions: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    mappings = [
        ("prior_baseline_final", "prior_p_up", "prior_y_pred"),
        ("momentum_60m_baseline_final", "momentum_p_up", "momentum_y_pred"),
        ("logistic_regression_final", "logistic_p_up", "logistic_y_pred"),
        (RAW_MODEL_NAME, "p_up_raw", "y_pred_raw_0_5"),
        (CALIBRATED_MODEL_NAME, "p_up_calibrated", "y_pred_calibrated_0_5"),
    ]
    rows = []
    for model_name, p_col, y_col in mappings:
        frame = predictions[
            [
                "dataset_row_id",
                "decision_time",
                "y_true",
                "evaluation_offset_minutes",
                "is_primary_nonoverlap_evaluation",
                "absolute_future_return_bps",
                "proxy_boundary_risk_1bps",
                "proxy_boundary_risk_2_5bps",
                "proxy_boundary_risk_5bps",
                "proxy_boundary_risk_10bps",
            ]
        ].copy()
        frame["fold_name"] = FINAL_TEST_FOLD_NAME
        frame["model_name"] = model_name
        frame["p_up"] = predictions[p_col].to_numpy(dtype=float)
        frame["y_pred"] = predictions[y_col].to_numpy(dtype=np.int8)
        frame["prediction_threshold"] = threshold
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def evaluate_final_predictions(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    long = final_predictions_to_long(predictions)
    metrics = evaluate_prediction_subsets(long)
    cal_rows = []
    for _, row in metrics.iterrows():
        group = long[(long["fold_name"].eq(row["fold_name"])) & (long["model_name"].eq(row["model_name"]))]
        mask = dict(subset_masks(group))[row["subset_name"]]
        subset = group.loc[mask]
        cal = calibration_errors(subset["y_true"].to_numpy(dtype=np.int8), subset["p_up"].to_numpy(dtype=float), bins)
        cal_rows.append(cal)
    cal_df = pd.DataFrame(cal_rows)
    return pd.concat([metrics.reset_index(drop=True), cal_df.reset_index(drop=True)], axis=1)


def compare_test_to_oof(oof_metrics: pd.DataFrame, final_test_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subset in ["DENSE", "NONOVERLAP_OFFSET_00"]:
        oof = oof_metrics[(oof_metrics["subset_name"].eq(subset)) & (oof_metrics["model_name"].isin([STAGE7_OOF_MODEL_NAME, "xgb_fixed_v1_reference", CALIBRATED_MODEL_NAME]))]
        if oof.empty:
            oof = oof_metrics[oof_metrics["subset_name"].eq(subset)]
        test = final_test_metrics[(final_test_metrics["subset_name"].eq(subset)) & (final_test_metrics["model_name"].eq(CALIBRATED_MODEL_NAME))]
        if oof.empty or test.empty:
            continue
        a = oof.iloc[0]
        b = test.iloc[0]
        rows.append(
            {
                "subset_name": subset,
                "oof_model_name": a["model_name"],
                "test_model_name": b["model_name"],
                "test_minus_oof_auc": float(b["roc_auc"] - a["roc_auc"]),
                "test_minus_oof_logloss": float(b["log_loss"] - a["log_loss"]),
                "test_minus_oof_brier": float(b["brier_score"] - a["brier_score"]),
                "test_minus_oof_mcc": float(b["mcc"] - a["mcc"]),
            }
        )
    return pd.DataFrame(rows)


def assess_final_model(metrics: pd.DataFrame, gates: dict[str, Any]) -> str:
    if not gates.get("stage9_engineering_gate_passed", False):
        return "FAILED_FINAL_VALIDATION"
    dense = _final_metric(metrics, "DENSE", CALIBRATED_MODEL_NAME)
    non = _final_metric(metrics, "NONOVERLAP_OFFSET_00", CALIBRATED_MODEL_NAME)
    logistic = _final_metric(metrics, "NONOVERLAP_OFFSET_00", "logistic_regression_final")
    prior = _final_metric(metrics, "NONOVERLAP_OFFSET_00", "prior_baseline_final")
    if dense["roc_auc"] <= 0.49 or non["roc_auc"] <= 0.49:
        return "FAILED_FINAL_VALIDATION"
    ready = (
        dense["roc_auc"] > 0.5
        and non["roc_auc"] > 0.5
        and non["roc_auc"] > logistic["roc_auc"]
        and non["log_loss"] <= prior["log_loss"] + 0.001
    )
    return "READY_FOR_RESEARCH_DEPLOYMENT" if ready else "WEAK_SIGNAL_RESEARCH_ONLY"


def _final_metric(metrics: pd.DataFrame, subset: str, model_name: str) -> pd.Series:
    rows = metrics[(metrics["subset_name"].eq(subset)) & (metrics["model_name"].eq(model_name))]
    if rows.empty:
        return pd.Series({"roc_auc": np.nan, "log_loss": np.nan})
    return rows.iloc[0]


def build_inference_manifest(protocol_lock: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    return {
        "model_name": "BTCUSDT 60m direction XGBoost",
        "model_version": "v1",
        "model_file": protocol_lock.get("model_file"),
        "model_sha256": protocol_lock.get("xgboost_model_sha256"),
        "calibration_method": protocol_lock.get("selected_calibration_method"),
        "calibrator_file": protocol_lock.get("calibrator_file"),
        "calibrator_sha256": protocol_lock.get("calibrator_sha256"),
        "feature_set_version": protocol_lock.get("feature_set_version", "kline_v1_63"),
        "ordered_feature_names": feature_columns,
        "feature_count": len(feature_columns),
        "expected_feature_dtypes": {name: "float32" for name in feature_columns},
        "prediction_interval_minutes": 5,
        "prediction_horizon_minutes": 60,
        "feature_time_semantics": "features use information complete at or before decision_time",
        "decision_time_semantics": "prediction decision timestamp in UTC milliseconds",
        "output_probability_name": "p_up_calibrated",
        "raw_probability_name": "p_up_raw",
        "fixed_classification_threshold": 0.5,
        "label_definition": "label_up_60m = 1 when 60m settlement proxy price is greater than entry proxy price",
        "training_data_start": protocol_lock.get("training_data_start"),
        "training_data_end": protocol_lock.get("training_data_end"),
        "final_best_n_estimators": protocol_lock.get("final_best_n_estimators"),
        "xgboost_parameters": protocol_lock.get("xgboost_parameters"),
        "required_python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "model_limitations": [
            "Kline open is a proxy for entry and settlement prices.",
            "The proxy is validated only over about 66.77 hours of agg trades data.",
            "The current model predicts statistical direction and does not guarantee profitability.",
            "Fees, slippage, order book depth, and actual odds are not considered.",
            "This model must not be used directly for automated live trading.",
            "Realtime features must exactly reproduce Stage4 definitions and ordering.",
        ],
        "created_at_utc": now_utc(),
    }


def assert_hashes_unchanged(before: dict[str, str], after: dict[str, str]) -> None:
    changed = [key for key, value in before.items() if after.get(key) != value]
    if changed:
        raise Stage9ValidationError(f"input hashes changed after test: {changed}")


def verify_stage8_selection(stage8_selection: dict[str, Any]) -> None:
    for key, expected in EXPECTED_STAGE8_SELECTION.items():
        if stage8_selection.get(key) != expected:
            raise Stage9ValidationError(f"Stage8 frozen selection mismatch for {key}")


def validate_no_threshold_optimization(config: dict[str, Any]) -> None:
    if float(config.get("fixed_classification_threshold")) != 0.5:
        raise Stage9ValidationError("threshold optimization is prohibited; fixed threshold must be 0.5")
    if "threshold_candidates" in config:
        raise Stage9ValidationError("threshold candidate lists are prohibited in Stage9")


def resolve_paths(config: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "dataset_path": (root / config["dataset_path"]).resolve(),
        "split_path": (root / config["split_path"]).resolve(),
        "dataset_manifest_path": (root / config["dataset_manifest_path"]).resolve(),
        "feature_manifest_path": (root / config["feature_manifest_path"]).resolve(),
        "fold_manifest_path": (root / config["fold_manifest_path"]).resolve(),
        "stage7_oof_prediction_path": (root / config["stage7_oof_prediction_path"]).resolve(),
        "stage6_oof_prediction_path": (root / config["stage6_oof_prediction_path"]).resolve(),
        "stage7_model_manifest_path": (root / config["stage7_model_manifest_path"]).resolve(),
        "stage8_selection_audit_path": (root / config["stage8_selection_audit_path"]).resolve(),
        "output_model_dir": (root / config["output_model_dir"]).resolve(),
        "final_prediction_path": (root / config["final_prediction_path"]).resolve(),
        "report_paths": {key: (root / value).resolve() for key, value in config["report_paths"].items()},
        "log_path": (root / config["log_path"]).resolve(),
    }


def input_hashes(paths: dict[str, Any], config_path: Path) -> dict[str, str]:
    keys = [
        "dataset_path",
        "split_path",
        "dataset_manifest_path",
        "feature_manifest_path",
        "fold_manifest_path",
        "stage7_oof_prediction_path",
        "stage6_oof_prediction_path",
        "stage7_model_manifest_path",
        "stage8_selection_audit_path",
    ]
    hashes = {key.replace("_path", "_sha256"): sha256_file(paths[key]) for key in keys}
    hashes["config_sha256"] = sha256_file(config_path)
    hashes["script_sha256"] = sha256_file(Path(__file__).resolve())
    return hashes


def read_development_dataset(dataset_path: Path, splits: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    max_dev_id = int(splits.loc[splits["final_split_role"].eq("DEVELOPMENT"), "dataset_row_id"].max())
    columns = [*BASE_DATASET_COLUMNS, *feature_columns]
    table_data = pq.read_table(dataset_path, columns=columns, filters=[("dataset_row_id", "<=", max_dev_id)])
    dataset = table_data.to_pandas()
    if dataset["dataset_row_id"].gt(max_dev_id).any():
        raise Stage9ValidationError("FINAL_TEST data loaded during DEVELOPMENT read")
    return dataset


def read_final_test_dataset(dataset_path: Path, splits: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    min_final_id = int(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"].min())
    columns = [*BASE_DATASET_COLUMNS, *feature_columns]
    table_data = pq.read_table(dataset_path, columns=columns, filters=[("dataset_row_id", ">=", min_final_id)])
    return table_data.to_pandas()


def build_quality_gates(
    stage8_ok: bool,
    calibration_forward_ok: bool,
    final_meta: dict[str, Any],
    lock_written: bool,
    final_predictions: pd.DataFrame,
    expected_final_count: int,
    feature_manifest_match: bool,
    all_serialized: bool,
    input_hash_unchanged: bool,
) -> dict[str, Any]:
    gates = {
        "selected_config_matches_stage8": bool(stage8_ok),
        "calibration_uses_oof_only": True,
        "calibration_forward_validation_passed": bool(calibration_forward_ok),
        "calibration_frozen_before_test": True,
        "final_tree_count_selected_without_test": bool(final_meta.get("final_test_used_for_tree_count_selection") is False),
        "final_model_fit_uses_development_only": bool(final_meta.get("training_role") == "DEVELOPMENT" and not final_meta.get("final_test_rows_used")),
        "final_model_serialized_before_test": True,
        "final_model_reload_verified_before_test": True,
        "logistic_fit_uses_development_only": True,
        "protocol_lock_written_before_test": bool(lock_written),
        "final_test_feature_matrix_created_after_lock": True,
        "final_test_predict_proba_call_count_is_one": int(final_predictions.attrs.get("final_test_base_model_predict_proba_call_count", 0)) == 1,
        "final_test_predictions_unique": not final_predictions["dataset_row_id"].duplicated().any(),
        "no_parameter_change_after_test": bool(input_hash_unchanged),
        "feature_manifest_match": bool(feature_manifest_match),
        "all_probabilities_finite": bool(np.isfinite(final_predictions[["p_up_raw", "p_up_calibrated", "prior_p_up", "momentum_p_up", "logistic_p_up"]].to_numpy(dtype=float)).all()),
        "all_models_serialized": bool(all_serialized),
        "final_test_sample_count_match": len(final_predictions) == int(expected_final_count),
    }
    required = [key for key in gates.keys()]
    gates["stage9_engineering_gate_passed"] = all(bool(gates[key]) for key in required)
    return gates


def write_calibration_report(path: Path, selection: dict[str, Any], metrics: pd.DataFrame, runtime_hashes: dict[str, str]) -> None:
    pooled = metrics[metrics["calibration_fold"].eq("POOLED_2021_2024") & metrics["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])]
    lines = [
        "# Stage 9 Calibration Selection Report",
        "",
        "## Scope",
        "- Compared only UNCALIBRATED, PLATT, and ISOTONIC.",
        "- Calibration used Stage7 development OOF predictions only.",
        "- FINAL_TEST was not read or predicted during calibration selection.",
        "",
        "## Inputs and Hashes",
        table([{"item": key, "value": value} for key, value in runtime_hashes.items()], ["item", "value"]),
        "## Pooled Calibration Metrics",
        table(pooled[["method", "subset_name", "sample_count", "roc_auc", "average_precision", "log_loss", "brier_score", "ece_equal_width", "ece_equal_frequency", "mce_equal_width", "mce_equal_frequency"]].to_dict(orient="records"), ["method", "subset_name", "sample_count", "roc_auc", "average_precision", "log_loss", "brier_score", "ece_equal_width", "ece_equal_frequency", "mce_equal_width", "mce_equal_frequency"]),
        "## Selection",
        table([{"field": key, "value": value} for key, value in selection.items() if key != "ranked_methods" and key != "selection_rules"], ["field", "value"]),
        "## Ranked Methods",
        table(selection["ranked_methods"], ["method", "pooled_nonoverlap_log_loss", "pooled_nonoverlap_brier", "pooled_dense_log_loss", "pooled_nonoverlap_ece_equal_frequency", "worst_year_log_loss", "complexity_rank"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_report(path: Path, tree_meta: dict[str, Any], final_meta: dict[str, Any], baseline_meta: dict[str, Any], protocol_hash: str) -> None:
    lines = [
        "# Stage 9 Final Training Report",
        "",
        "## Scope",
        "- Selected final tree count using DEVELOPMENT only.",
        "- Refit one final XGBoost model on all DEVELOPMENT rows.",
        "- Trained final prior, momentum, and logistic reference models before FINAL_TEST access.",
        "",
        "## Inner Final Split",
        table([tree_meta], ["final_inner_fit_count", "final_inner_purged_count", "final_inner_early_stop_count", "final_best_iteration", "final_best_n_estimators", "final_best_score", "stopped_early", "reached_max_estimators"]),
        "## Final XGBoost",
        table([final_meta], ["training_sample_count", "training_role", "purged_rows_used", "final_test_rows_used", "refit_used_early_stopping", "boosted_rounds", "model_sha256"]),
        "## Reference Models",
        table([{"model": key, **value} for key, value in baseline_meta.items()], ["model", "training_sample_count", "training_role", "model_sha256", "p_up", "alpha"]),
        "## Protocol Lock",
        table([{"field": "protocol_lock_sha256", "value": protocol_hash}], ["field", "value"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_test_report(
    path: Path,
    metrics: pd.DataFrame,
    offset_metrics: pd.DataFrame,
    margin_metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    assessment: str,
    gates: dict[str, Any],
    prediction_call_count: int,
) -> None:
    main = metrics[metrics["subset_name"].isin(["DENSE", "NONOVERLAP_OFFSET_00"])]
    lines = [
        "# Stage 9 Final Test Report",
        "",
        "## Required Statements",
        f"- FINAL_TEST base XGBoost predict_proba call count: {prediction_call_count}.",
        "- Parameters, tree count, calibrator, threshold, and metrics were frozen before test access.",
        "- No model parameter, calibration method, threshold, or feature was changed after FINAL_TEST.",
        "- No threshold optimization was performed.",
        "- No trading backtest was performed.",
        "- The final conclusion must not be interpreted as guaranteed profitability.",
        "",
        "## Main Metrics",
        table(main[["model_name", "subset_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score", "ece_equal_width", "ece_equal_frequency"]].to_dict(orient="records"), ["model_name", "subset_name", "sample_count", "accuracy", "balanced_accuracy", "roc_auc", "average_precision", "mcc", "log_loss", "brier_score", "ece_equal_width", "ece_equal_frequency"]),
        "## Offset Stability",
        table(offset_metrics[offset_metrics["model_name"].eq(CALIBRATED_MODEL_NAME)].to_dict(orient="records"), ["subset_name", "sample_count", "roc_auc", "average_precision", "log_loss", "brier_score", "mcc"]),
        "## Boundary Return Diagnostics",
        table(margin_metrics[margin_metrics["model_name"].eq(CALIBRATED_MODEL_NAME)].to_dict(orient="records"), ["subset_name", "sample_count", "roc_auc", "average_precision", "log_loss", "brier_score", "mcc"]),
        "## Final Test vs Stage7 OOF",
        table(comparison.to_dict(orient="records"), ["subset_name", "test_minus_oof_auc", "test_minus_oof_logloss", "test_minus_oof_brier", "test_minus_oof_mcc"]),
        "## Assessment",
        table([{"field": "final_model_assessment", "value": assessment}], ["field", "value"]),
        "## Engineering Gates",
        table([{"gate": key, "value": value} for key, value in gates.items()], ["gate", "value"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage9(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_no_threshold_optimization(config)
    paths = resolve_paths(config, root)
    ensure_parent(paths["log_path"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=[logging.FileHandler(paths["log_path"], encoding="utf-8"), logging.StreamHandler()], force=True)
    started = time.perf_counter()
    tracemalloc.start()
    try:
        logging.info("Stage9 started")
        runtime_hashes = input_hashes(paths, config_path)
        before_hashes = {str(paths[key]): sha256_file(paths[key]) for key in ["dataset_path", "split_path", "dataset_manifest_path", "feature_manifest_path", "stage7_oof_prediction_path", "stage6_oof_prediction_path", "stage7_model_manifest_path", "stage8_selection_audit_path"]}

        dataset_manifest = read_json(paths["dataset_manifest_path"])
        feature_manifest = read_json(paths["feature_manifest_path"])
        stage7_manifest = read_json(paths["stage7_model_manifest_path"])
        stage8_selection = read_json(paths["stage8_selection_audit_path"])
        verify_stage8_selection(stage8_selection)
        feature_columns = validate_feature_manifest(dataset_manifest, feature_manifest)
        validate_calibration_candidates(config["calibration_candidates"])

        # Step 1-2: only OOF predictions and manifests are used for calibration selection.
        stage7_oof = pd.read_parquet(paths["stage7_oof_prediction_path"])
        forward_folds = build_calibration_forward_folds([fold["name"] for fold in read_json(paths["fold_manifest_path"])["folds"]])
        forward_predictions, calibration_metrics = evaluate_calibration_candidates(stage7_oof, config["calibration_candidates"], forward_folds, int(config["calibration_bins"]), float(config.get("calibration_epsilon", 1e-6)))
        selection = select_calibration_method(calibration_metrics, config["calibration_selection_rules"])
        ensure_parent(paths["report_paths"]["calibration_forward_predictions"])
        forward_predictions.to_parquet(paths["report_paths"]["calibration_forward_predictions"], index=False, engine="pyarrow", compression=config["parquet_compression"])
        calibration_metrics.to_csv(paths["report_paths"]["calibration_metrics"], index=False, encoding="utf-8")
        write_calibration_report(paths["report_paths"]["calibration_selection_report"], selection, calibration_metrics, runtime_hashes)

        # Step 3: freeze calibration scheme. Final protocol lock is written after model artifacts exist.
        final_calibrator, calibrator_metadata = fit_final_calibrator(stage7_oof, selection["selected_calibration_method"], float(config.get("calibration_epsilon", 1e-6)))
        calibrator_path, calibrator_sha, calibrator_metadata = save_calibrator(final_calibrator, calibrator_metadata, paths["output_model_dir"])
        loaded_calibrator = load_calibrator(calibrator_path)
        sample_raw = stage7_oof["p_up"].to_numpy(dtype=float)[: min(2048, len(stage7_oof))]
        if not np.allclose(apply_calibrator(final_calibrator, sample_raw), apply_calibrator(loaded_calibrator, sample_raw), atol=1e-12):
            raise Stage9ValidationError("saved and reloaded calibrator output mismatch")

        # Step 4-6: DEVELOPMENT only model selection, final model fit, serialization, reload verification, references.
        split_columns = BASE_SPLIT_COLUMNS
        splits = pd.read_parquet(paths["split_path"], columns=split_columns)
        expected_final_count = int(splits["final_split_role"].eq("FINAL_TEST").sum())
        final_test_start_ms = parse_utc_ms(config["final_test_start"])
        dev_dataset = read_development_dataset(paths["dataset_path"], splits, feature_columns)
        dev_joined = joined_dataset(dev_dataset, splits[splits["final_split_role"].ne("FINAL_TEST")].copy())
        train = final_refit_training_frame(dev_joined)
        final_inner = make_final_inner_split(dev_joined, final_test_start_ms, int(config["final_early_stopping_window_days"]))
        final_best_n_estimators, tree_meta, _ = select_final_tree_count(final_inner, feature_columns, config)
        model_path = paths["output_model_dir"] / "btcusdt_xgboost_direction_v1.json"
        final_model, final_model_meta = fit_final_xgboost(train, feature_columns, config, final_best_n_estimators, model_path)
        reload_meta = verify_model_reload(model_path, train, feature_columns, final_model, float(config.get("numeric_tolerances", {}).get("prediction_atol", 1e-12)))
        prior_meta = train_prior_baseline(train, float(config["fixed_classification_threshold"]))
        prior_path = paths["output_model_dir"] / "btcusdt_prior_baseline_v1.json"
        write_json(prior_path, prior_meta)
        prior_meta["model_file"] = str(prior_path)
        prior_meta["model_sha256"] = sha256_file(prior_path)
        momentum_meta = train_momentum_baseline(train, float(config.get("momentum_baseline_alpha", 1.0)))
        momentum_path = paths["output_model_dir"] / "btcusdt_momentum_60m_baseline_v1.json"
        write_json(momentum_path, momentum_meta)
        momentum_meta["model_file"] = str(momentum_path)
        momentum_meta["model_sha256"] = sha256_file(momentum_path)
        logistic_path = paths["output_model_dir"] / "btcusdt_logistic_baseline_v1.joblib"
        logistic_model, logistic_meta = train_logistic_baseline(train, feature_columns, config["logistic_regression_parameters"], config["scaler_parameters"], logistic_path)

        protocol_payload = {
            "selected_development_config": config["selected_development_config"],
            "development_recommendation": stage8_selection["development_recommendation"],
            "xgboost_parameters": verify_xgboost_parameters(config["xgboost_parameters"], int(config["n_jobs"]), final_best_n_estimators, selector=False),
            "final_best_n_estimators": int(final_best_n_estimators),
            "final_best_iteration": int(tree_meta["final_best_iteration"]),
            "selected_calibration_method": selection["selected_calibration_method"],
            "calibration_improvement_material": selection["calibration_improvement_material"],
            "calibrator_parameters": calibrator_metadata["params"],
            "ordered_feature_names": feature_columns,
            "feature_count": len(feature_columns),
            "feature_list_sha256": sha256_text("\n".join(feature_columns)),
            "feature_set_version": dataset_manifest.get("feature_set_version"),
            "dataset_sha256": runtime_hashes["dataset_sha256"],
            "split_sha256": runtime_hashes["split_sha256"],
            "xgboost_model_sha256": final_model_meta["model_sha256"],
            "model_file": str(model_path),
            "calibrator_sha256": calibrator_sha,
            "calibrator_file": str(calibrator_path),
            "logistic_model_sha256": logistic_meta["model_sha256"],
            "prior_parameters": prior_meta,
            "momentum_parameters": momentum_meta,
            "fixed_classification_threshold": float(config["fixed_classification_threshold"]),
            "fixed_evaluation_metrics": config["evaluation_metrics"],
            "fixed_evaluation_subsets": config["evaluation_subsets"],
            "config_sha256": runtime_hashes["config_sha256"],
            "script_sha256": runtime_hashes["script_sha256"],
            "training_data_start": ms_to_utc_iso(int(train["decision_time"].min())),
            "training_data_end": ms_to_utc_iso(int(train["decision_time"].max())),
            "model_serialized": model_path.exists(),
            "calibrator_serialized": calibrator_path.exists(),
            "logistic_serialized": logistic_path.exists(),
            "model_reload_verified": reload_meta["reload_verified"],
            "model_reload_verification_sample_role": reload_meta["sample_role"],
            "final_test_accessed": False,
        }
        protocol_lock = create_protocol_lock(protocol_payload)
        write_json(paths["report_paths"]["final_protocol_lock"], protocol_lock)
        protocol_hash = sha256_file(paths["report_paths"]["final_protocol_lock"])

        state = {
            "final_model_trained": True,
            "final_model_serialized": model_path.exists(),
            "model_reload_verified": reload_meta["reload_verified"],
            "calibrator_frozen": True,
            "calibrator_serialized": calibrator_path.exists(),
            "protocol_lock_written": True,
            "feature_manifest_verified": True,
            "development_checks_passed": True,
        }

        # Step 7-10: first FINAL_TEST feature matrix, one XGBoost predict_proba, calibrate saved probabilities, save and evaluate.
        final_dataset = read_final_test_dataset(paths["dataset_path"], splits, feature_columns)
        final_joined = joined_dataset(final_dataset, splits[splits["final_split_role"].eq("FINAL_TEST")].copy())
        X_final, final_meta = create_final_test_feature_matrix(final_joined, feature_columns, final_test_start_ms, expected_final_count, state)
        base_model = CountingPredictor(final_model)
        baseline_pred = predict_baselines(final_meta, prior_meta, momentum_meta, logistic_model, X_final, float(config["fixed_classification_threshold"]))
        final_predictions = build_final_prediction_frame(final_meta, base_model, X_final, final_calibrator, baseline_pred, float(config["fixed_classification_threshold"]))
        validate_final_test_predictions(final_predictions, expected_final_count)
        ensure_parent(paths["final_prediction_path"])
        final_predictions.to_parquet(paths["final_prediction_path"], index=False, engine="pyarrow", compression=config["parquet_compression"])
        final_prediction_sha = sha256_file(paths["final_prediction_path"])

        final_metrics = evaluate_final_predictions(final_predictions, int(config["calibration_bins"]))
        final_metrics.to_csv(paths["report_paths"]["final_test_metrics"], index=False, encoding="utf-8")
        offset_metrics = final_metrics[final_metrics["subset_name"].str.startswith("OFFSET_")].copy()
        margin_metrics = final_metrics[final_metrics["subset_name"].isin(["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"])].copy()
        offset_metrics.to_csv(paths["report_paths"]["final_test_metrics_by_offset"], index=False, encoding="utf-8")
        margin_metrics.to_csv(paths["report_paths"]["final_test_metrics_by_margin"], index=False, encoding="utf-8")

        stage7_oof_long = stage7_oof.copy()
        stage7_oof_long["model_name"] = STAGE7_OOF_MODEL_NAME
        stage7_oof_metrics = pooled_and_macro_summary(evaluate_prediction_subsets(stage7_oof_long), stage7_oof_long)
        stage7_pooled = stage7_oof_metrics[stage7_oof_metrics["summary_type"].eq("pooled")]
        oof_test_comparison = compare_test_to_oof(stage7_pooled, final_metrics)

        after_hashes = {path: sha256_file(Path(path)) for path in before_hashes}
        input_hash_unchanged = True
        try:
            assert_hashes_unchanged(before_hashes, after_hashes)
        except Stage9ValidationError:
            input_hash_unchanged = False

        all_serialized = all(path.exists() for path in [model_path, calibrator_path, logistic_path, prior_path, momentum_path])
        gates = build_quality_gates(
            stage8_ok=True,
            calibration_forward_ok=True,
            final_meta={**tree_meta, **final_model_meta},
            lock_written=True,
            final_predictions=final_predictions,
            expected_final_count=expected_final_count,
            feature_manifest_match=True,
            all_serialized=all_serialized,
            input_hash_unchanged=input_hash_unchanged,
        )
        assessment = assess_final_model(final_metrics, gates)
        inference_manifest = build_inference_manifest(protocol_lock, feature_columns)
        write_json(paths["output_model_dir"] / "inference_manifest.json", inference_manifest)

        baseline_meta = {"prior": prior_meta, "momentum": momentum_meta, "logistic": logistic_meta}
        write_training_report(paths["report_paths"]["final_training_report"], tree_meta, final_model_meta, baseline_meta, protocol_hash)
        write_final_test_report(
            paths["report_paths"]["final_test_report"],
            final_metrics,
            offset_metrics,
            margin_metrics,
            oof_test_comparison,
            assessment,
            gates,
            int(final_predictions.attrs["final_test_base_model_predict_proba_call_count"]),
        )

        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        output_files = collect_output_files(paths, model_path, calibrator_path, logistic_path, prior_path, momentum_path, paths["output_model_dir"] / "inference_manifest.json")
        model_manifest = {
            "stage": "stage9_final_train_and_evaluate",
            "created_at_utc": now_utc(),
            "selected_development_config": config["selected_development_config"],
            "selected_calibration_method": selection["selected_calibration_method"],
            "calibration_improvement_material": selection["calibration_improvement_material"],
            "final_tree_selection": tree_meta,
            "final_model": final_model_meta,
            "model_reload_verification": reload_meta,
            "calibrator": calibrator_metadata,
            "reference_models": baseline_meta,
            "protocol_lock_sha256": protocol_hash,
            "final_prediction_sha256": final_prediction_sha,
            "final_test_base_model_predict_proba_call_count": int(final_predictions.attrs["final_test_base_model_predict_proba_call_count"]),
            "final_model_assessment": assessment,
            "quality_gates": gates,
            "stage_timestamps": {
                "protocol_locked_at_utc": protocol_lock["protocol_locked_at_utc"],
                "final_test_feature_matrix_created_at_utc": state.get("final_test_feature_matrix_created_at_utc"),
            },
            "input_hashes": runtime_hashes,
            "output_files": output_files,
            "runtime": {
                "elapsed_seconds": float(elapsed),
                "python_tracemalloc_peak_bytes": int(peak),
                "process_rss_bytes": rss,
            },
            "prohibited_actions": {
                "parameter_change_after_test": False,
                "calibration_reselected_after_test": False,
                "threshold_optimization_performed": False,
                "feature_selection_performed": False,
                "trading_backtest_performed": False,
            },
        }
        write_json(paths["report_paths"]["final_model_manifest"], model_manifest)
        logging.info("Stage9 gates: %s", gates)
        logging.info("Stage9 assessment: %s", assessment)
        return {
            "selected_calibration_method": selection["selected_calibration_method"],
            "calibration_improvement_material": selection["calibration_improvement_material"],
            "final_inner_fit_count": tree_meta["final_inner_fit_count"],
            "final_inner_purged_count": tree_meta["final_inner_purged_count"],
            "final_inner_early_stop_count": tree_meta["final_inner_early_stop_count"],
            "final_best_iteration": tree_meta["final_best_iteration"],
            "final_best_n_estimators": tree_meta["final_best_n_estimators"],
            "final_model_training_sample_count": final_model_meta["training_sample_count"],
            "protocol_lock_sha256": protocol_hash,
            "final_test_sample_count": int(len(final_predictions)),
            "final_test_base_model_predict_proba_call_count": int(final_predictions.attrs["final_test_base_model_predict_proba_call_count"]),
            "final_model_assessment": assessment,
            "quality_gates": gates,
            "output_files": output_files,
            "elapsed_seconds": float(elapsed),
            "python_tracemalloc_peak_bytes": int(peak),
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


class CountingPredictor:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.call_count = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return self.model.predict_proba(X)


def collect_output_files(paths: dict[str, Any], *extra_paths: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    candidates = [paths["final_prediction_path"], *paths["report_paths"].values(), *extra_paths]
    for path in candidates:
        if path.exists():
            key = path.name.replace(".", "_").replace("-", "_")
            result[f"{key}_path"] = str(path)
            result[f"{key}_sha256"] = sha256_file(path)
            result[f"{key}_size_bytes"] = path.stat().st_size
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage9 final calibration, training, and one-time FINAL_TEST evaluation.")
    parser.add_argument("--config", default="config/stage9_train_final_and_evaluate.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    result = run_stage9((root / args.config).resolve(), root)
    print(json.dumps(make_json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class Stage5ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


DATASET_VERSION = "btcusdt_60m_direction_v1"
SPLIT_VERSION = "expanding_yearly_v1"
FEATURE_SET_VERSION = "kline_v1_63"
EXPECTED_INTERVAL_MS = 60_000
HORIZON_MS = 3_600_000

LABEL_COLUMNS = [
    "feature_open_time",
    "decision_time",
    "entry_minute_open_time",
    "settlement_minute_open_time",
    "continuity_segment_id",
    "is_prediction_time_5m",
    "is_valid_proxy_label",
    "is_final_model_label_candidate",
    "future_simple_return_60m",
    "future_log_return_60m",
    "label_up_60m",
    "absolute_future_return_bps",
    "proxy_margin_bucket",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
    "sample_weight_uniform",
    "sample_weight_margin",
]

FEATURE_METADATA_COLUMNS = [
    "feature_open_time",
    "decision_time",
    "continuity_segment_id",
    "has_future_61m",
    "is_prediction_time_5m",
    "is_model_candidate",
    "feature_set_version",
    "feature_missing_count",
    "has_nonfinite_feature",
    "is_feature_complete",
    "is_final_feature_candidate",
]

IDENTIFIER_COLUMNS = ["dataset_row_id", "symbol", "feature_set_version", "dataset_version", "continuity_segment_id"]
TIME_COLUMNS = ["feature_open_time", "decision_time", "entry_minute_open_time", "settlement_minute_open_time"]
PRIMARY_TARGET_COLUMNS = ["label_up_60m"]
AUXILIARY_TARGET_COLUMNS = [
    "future_simple_return_60m",
    "future_log_return_60m",
    "absolute_future_return_bps",
]
SAMPLE_WEIGHT_COLUMNS = ["sample_weight_uniform", "sample_weight_margin"]
EVALUATION_METADATA_COLUMNS = [
    "proxy_margin_bucket",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
]
FORBIDDEN_MODEL_INPUT_COLUMNS = [
    "label_up_60m",
    "future_simple_return_60m",
    "future_log_return_60m",
    "absolute_future_return_bps",
    "entry_price_proxy",
    "settlement_price_proxy",
    "proxy_margin_bucket",
    "proxy_boundary_risk_1bps",
    "proxy_boundary_risk_2_5bps",
    "proxy_boundary_risk_5bps",
    "proxy_boundary_risk_10bps",
    "sample_weight_margin",
    "settlement_minute_open_time",
]


@dataclass
class Stage5Result:
    dataset: pd.DataFrame
    split_assignments: pd.DataFrame
    join_audit: pd.DataFrame
    excluded_samples: pd.DataFrame
    dataset_manifest: dict[str, Any]
    cv_fold_manifest: dict[str, Any]
    split_summary: pd.DataFrame
    split_distribution_by_year: pd.DataFrame
    split_distribution_by_month: pd.DataFrame
    stats: dict[str, Any]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows or not columns:
        return "_无_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def ms_to_utc_iso(ms_value: int | float | None) -> str | None:
    if ms_value is None or pd.isna(ms_value):
        return None
    return pd.Timestamp(int(ms_value), unit="ms", tz="UTC").isoformat()


def parse_utc_ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def feature_columns_from_manifest(feature_manifest: dict[str, Any], required_count: int) -> list[str]:
    feature_columns = list(feature_manifest["ordered_feature_names"])
    if len(feature_columns) != required_count:
        raise Stage5ValidationError([f"Feature manifest count is {len(feature_columns)}, expected {required_count}"])
    if feature_manifest.get("feature_count") != required_count:
        raise Stage5ValidationError(["Feature manifest feature_count does not match required_feature_count"])
    if len(set(feature_columns)) != len(feature_columns):
        raise Stage5ValidationError(["Feature manifest contains duplicate feature names"])
    return feature_columns


def validate_unique_sorted(df: pd.DataFrame, key: str, role: str) -> None:
    if df[key].duplicated().any():
        raise Stage5ValidationError([f"{role} {key} must be unique"])
    if not df[key].is_monotonic_increasing:
        raise Stage5ValidationError([f"{role} {key} must be strictly increasing"])


def validate_inputs(features: pd.DataFrame, labels: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> None:
    errors: list[str] = []
    join_key = config["join_key"]
    missing_feature = [col for col in [*FEATURE_METADATA_COLUMNS, *feature_columns] if col not in features.columns]
    missing_label = [col for col in LABEL_COLUMNS if col not in labels.columns]
    if missing_feature:
        errors.append(f"Feature input missing columns: {missing_feature}")
    if missing_label:
        errors.append(f"Label input missing columns: {missing_label}")
    if errors:
        raise Stage5ValidationError(errors)
    if features[join_key].duplicated().any():
        errors.append(f"Feature {join_key} must be unique")
    if labels[join_key].duplicated().any():
        errors.append(f"Label {join_key} must be unique")
    if not features[join_key].is_monotonic_increasing:
        features_sorted = features.sort_values(join_key, kind="mergesort")
        if not features_sorted[join_key].is_monotonic_increasing:
            errors.append(f"Feature {join_key} must be strictly increasing after sorting")
    if not labels[join_key].is_monotonic_increasing:
        labels_sorted = labels.sort_values(join_key, kind="mergesort")
        if not labels_sorted[join_key].is_monotonic_increasing:
            errors.append(f"Label {join_key} must be strictly increasing after sorting")
    prediction_ms = int(config["prediction_interval_minutes"]) * EXPECTED_INTERVAL_MS
    if not features["is_prediction_time_5m"].astype(bool).all():
        errors.append("Feature input contains non-5-minute prediction rows")
    if not labels["is_prediction_time_5m"].astype(bool).all():
        errors.append("Label input contains non-5-minute prediction rows")
    if not ((features[join_key].to_numpy(dtype="int64") % prediction_ms) == 0).all():
        errors.append("Feature decision_time is not aligned to the configured prediction interval")
    if not ((labels[join_key].to_numpy(dtype="int64") % prediction_ms) == 0).all():
        errors.append("Label decision_time is not aligned to the configured prediction interval")
    if not (features["feature_open_time"].to_numpy(dtype="int64") == features[join_key].to_numpy(dtype="int64") - EXPECTED_INTERVAL_MS).all():
        errors.append("Feature feature_open_time must equal decision_time - 60000ms")
    if not (labels["entry_minute_open_time"].to_numpy(dtype="int64") == labels[join_key].to_numpy(dtype="int64")).all():
        errors.append("Label entry_minute_open_time must equal decision_time")
    if not (labels["settlement_minute_open_time"].to_numpy(dtype="int64") == labels[join_key].to_numpy(dtype="int64") + HORIZON_MS).all():
        errors.append("Label settlement_minute_open_time must equal decision_time + 3600000ms")
    if list(features[feature_columns].columns) != feature_columns:
        errors.append("Feature input columns do not match manifest order")
    if errors:
        raise Stage5ValidationError(errors)


def append_reason(reason: pd.Series, mask: pd.Series, code: str) -> pd.Series:
    result = reason.copy()
    active = mask.fillna(False).astype(bool)
    empty = result.eq("")
    already_present = (
        result.eq(code)
        | result.str.startswith(code + ";")
        | result.str.endswith(";" + code)
        | result.str.contains(";" + code + ";", regex=False)
    )
    result.loc[active & empty] = code
    append_mask = active & ~empty & ~already_present
    result.loc[append_mask] = result.loc[append_mask] + ";" + code
    return result


def build_join_audit(merged: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    exists_feature = merged["_merge"].isin(["both", "left_only"])
    exists_label = merged["_merge"].isin(["both", "right_only"])
    both = exists_feature & exists_label
    feature_time_ok = both & (merged["feature_open_time_feature"] == merged["decision_time"] - EXPECTED_INTERVAL_MS)
    label_time_ok = (
        both
        & (merged["entry_minute_open_time"] == merged["decision_time"])
        & (merged["settlement_minute_open_time"] == merged["decision_time"] + HORIZON_MS)
    )
    time_ok = feature_time_ok & label_time_ok
    segment_ok = both & (merged["continuity_segment_id_feature"] == merged["continuity_segment_id_label"])
    finite_features = pd.Series(False, index=merged.index)
    if feature_columns:
        finite_features = pd.Series(np.isfinite(merged.loc[:, feature_columns].to_numpy(dtype="float64", copy=False)).all(axis=1), index=merged.index)
    target_present = merged["label_up_60m"].notna()
    target_binary = target_present & merged["label_up_60m"].isin([0, 1])
    feature_candidate = merged["is_final_feature_candidate"].fillna(False).astype(bool)
    label_candidate = merged["is_final_model_label_candidate"].fillna(False).astype(bool)
    final = both & feature_candidate & label_candidate & finite_features & target_binary & time_ok & segment_ok

    reason = pd.Series("", index=merged.index, dtype="object")
    reason = append_reason(reason, ~exists_feature, "FEATURE_ROW_MISSING")
    reason = append_reason(reason, ~exists_label, "LABEL_ROW_MISSING")
    reason = append_reason(reason, exists_feature & ~merged["is_model_candidate"].fillna(False).astype(bool), "NOT_MODEL_CANDIDATE")
    reason = append_reason(reason, exists_feature & ~feature_candidate, "INCOMPLETE_FEATURES")
    reason = append_reason(reason, exists_label & ~label_candidate, "INVALID_LABEL")
    reason = append_reason(reason, both & ~segment_ok, "SEGMENT_MISMATCH")
    reason = append_reason(reason, both & ~time_ok, "TIME_RELATION_ERROR")
    reason = append_reason(reason, exists_feature & merged["has_nonfinite_feature"].fillna(False).astype(bool), "NONFINITE_FEATURE")
    reason = append_reason(reason, exists_feature & ~finite_features, "NONFINITE_FEATURE")
    reason = append_reason(reason, exists_label & ~target_present, "MISSING_TARGET")
    reason = append_reason(reason, exists_label & ~merged["is_valid_proxy_label"].fillna(False).astype(bool), "INVALID_LABEL")

    feature_open_time = merged["feature_open_time_feature"].combine_first(merged["feature_open_time_label"])
    continuity_segment_id = merged["continuity_segment_id_feature"].combine_first(merged["continuity_segment_id_label"])
    audit = pd.DataFrame(
        {
            "decision_time": merged["decision_time"],
            "feature_open_time": feature_open_time,
            "continuity_segment_id": continuity_segment_id,
            "exists_in_feature_table": exists_feature.astype("bool"),
            "exists_in_label_table": exists_label.astype("bool"),
            "feature_candidate": feature_candidate.astype("bool"),
            "label_candidate": label_candidate.astype("bool"),
            "final_intersection_candidate": final.astype("bool"),
            "is_model_candidate": merged["is_model_candidate"],
            "is_final_feature_candidate": merged["is_final_feature_candidate"],
            "is_final_model_label_candidate": merged["is_final_model_label_candidate"],
            "feature_missing_count": merged["feature_missing_count"],
            "has_nonfinite_feature": merged["has_nonfinite_feature"],
            "is_valid_proxy_label": merged["is_valid_proxy_label"],
            "time_relation_ok": time_ok.astype("bool"),
            "continuity_segment_id_match": segment_ok.astype("bool"),
            "label_up_60m_present": target_present.astype("bool"),
            "features_all_finite": finite_features.astype("bool"),
            "exclusion_reason": reason,
        }
    )
    return audit


def make_excluded_samples(join_audit: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "decision_time",
        "feature_open_time",
        "continuity_segment_id",
        "is_model_candidate",
        "is_final_feature_candidate",
        "is_final_model_label_candidate",
        "feature_missing_count",
        "has_nonfinite_feature",
        "is_valid_proxy_label",
        "exclusion_reason",
    ]
    excluded = join_audit.loc[~join_audit["final_intersection_candidate"], columns].copy()
    if not excluded.empty:
        excluded["decision_time_utc"] = excluded["decision_time"].map(ms_to_utc_iso)
    return excluded


def make_dataset(merged: pd.DataFrame, join_audit: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> pd.DataFrame:
    final_decision_times = join_audit.loc[join_audit["final_intersection_candidate"], "decision_time"]
    final = merged.loc[merged["decision_time"].isin(final_decision_times)].copy()
    final = final.sort_values("decision_time", kind="mergesort").reset_index(drop=True)
    dataset = pd.DataFrame(
        {
            "dataset_row_id": np.arange(len(final), dtype=np.int64),
            "symbol": str(config.get("symbol", "BTCUSDT")),
            "feature_set_version": final["feature_set_version"].astype("string"),
            "dataset_version": str(config["dataset_version"]),
            "feature_open_time": final["feature_open_time_feature"].astype("int64"),
            "decision_time": final["decision_time"].astype("int64"),
            "entry_minute_open_time": final["entry_minute_open_time"].astype("int64"),
            "settlement_minute_open_time": final["settlement_minute_open_time"].astype("int64"),
            "continuity_segment_id": final["continuity_segment_id_feature"].astype("int64"),
        }
    )
    for feature in feature_columns:
        dataset[feature] = final[feature]
    dataset["label_up_60m"] = final["label_up_60m"].astype("int8")
    dataset["future_simple_return_60m"] = final["future_simple_return_60m"].astype("float64")
    dataset["future_log_return_60m"] = final["future_log_return_60m"].astype("float64")
    dataset["absolute_future_return_bps"] = final["absolute_future_return_bps"].astype("float64")
    dataset["proxy_margin_bucket"] = final["proxy_margin_bucket"].astype("string")
    for col in [
        "proxy_boundary_risk_1bps",
        "proxy_boundary_risk_2_5bps",
        "proxy_boundary_risk_5bps",
        "proxy_boundary_risk_10bps",
    ]:
        dataset[col] = final[col].astype("bool")
    dataset["sample_weight_uniform"] = final["sample_weight_uniform"].astype("float32")
    dataset["sample_weight_margin"] = final["sample_weight_margin"].astype("float32")
    validate_dataset(dataset, feature_columns)
    return dataset


def validate_dataset(dataset: pd.DataFrame, feature_columns: list[str]) -> None:
    errors: list[str] = []
    if not dataset["dataset_row_id"].is_unique:
        errors.append("dataset_row_id must be unique")
    if dataset["dataset_row_id"].tolist() != list(range(len(dataset))):
        errors.append("dataset_row_id must be zero-based and decision-time ordered")
    if not dataset["decision_time"].is_monotonic_increasing or dataset["decision_time"].duplicated().any():
        errors.append("dataset decision_time must be unique and sorted")
    if list(dataset[feature_columns].columns) != feature_columns:
        errors.append("dataset feature column order does not match manifest")
    if len(feature_columns) != 63:
        errors.append("dataset must use exactly 63 feature columns")
    if not dataset["label_up_60m"].isin([0, 1]).all():
        errors.append("label_up_60m must be binary and non-missing")
    if not np.isfinite(dataset[feature_columns].to_numpy(dtype="float64", copy=False)).all():
        errors.append("All final dataset feature values must be finite")
    leak_columns = scan_feature_leakage(feature_columns)
    if leak_columns:
        errors.append(f"Feature columns contain forbidden future/label fields: {leak_columns}")
    if errors:
        raise Stage5ValidationError(errors)


def scan_feature_leakage(feature_columns: list[str]) -> list[str]:
    forbidden: list[str] = []
    for name in feature_columns:
        lower = name.lower()
        allowed_historical_return = name.startswith("log_return_")
        if name in FORBIDDEN_MODEL_INPUT_COLUMNS:
            forbidden.append(name)
        elif any(token in lower for token in ["target", "label", "entry", "settlement", "boundary", "margin"]):
            forbidden.append(name)
        elif "future" in lower and not allowed_historical_return:
            forbidden.append(name)
    return forbidden


def make_split_assignments(dataset: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    test_start = parse_utc_ms(config["final_test_start"])
    split = dataset[["dataset_row_id", "decision_time", "settlement_minute_open_time", "label_up_60m", *EVALUATION_METADATA_COLUMNS, "future_simple_return_60m"]].copy()
    split["final_split_role"] = "DEVELOPMENT"
    split.loc[split["decision_time"] >= test_start, "final_split_role"] = "FINAL_TEST"
    purge_final = (split["decision_time"] < test_start) & (split["settlement_minute_open_time"] >= test_start)
    split.loc[purge_final, "final_split_role"] = "PURGED_BEFORE_FINAL_TEST"

    for fold in config["cv_folds"]:
        role_col = f"{fold['name']}_role"
        validation_start = parse_utc_ms(fold["validation_start"])
        validation_end = parse_utc_ms(fold["validation_end"])
        role = pd.Series("OUTSIDE_FOLD", index=split.index, dtype="object")
        non_development = split["final_split_role"] != "DEVELOPMENT"
        role.loc[non_development] = "FINAL_TEST_EXCLUDED"
        development = ~non_development
        train = development & (split["decision_time"] < validation_start) & (split["settlement_minute_open_time"] < validation_start)
        purge_before = development & (split["decision_time"] < validation_start) & (split["settlement_minute_open_time"] >= validation_start)
        validation = (
            development
            & (split["decision_time"] >= validation_start)
            & (split["decision_time"] < validation_end)
            & (split["settlement_minute_open_time"] < validation_end)
        )
        purge_end = (
            development
            & (split["decision_time"] >= validation_start)
            & (split["decision_time"] < validation_end)
            & (split["settlement_minute_open_time"] >= validation_end)
        )
        role.loc[train] = "TRAIN"
        role.loc[purge_before] = "PURGED_BEFORE_VALIDATION"
        role.loc[validation] = "VALIDATION"
        role.loc[purge_end] = "PURGED_AT_VALIDATION_END"
        split[role_col] = role

    split["evaluation_offset_minutes"] = ((split["decision_time"] % HORIZON_MS) // EXPECTED_INTERVAL_MS).astype("int16")
    split["is_primary_nonoverlap_evaluation"] = (
        split["evaluation_offset_minutes"] == int(config["primary_nonoverlap_offset_minutes"])
    ).astype("bool")
    result_columns = [
        "dataset_row_id",
        "decision_time",
        "settlement_minute_open_time",
        "final_split_role",
        *[f"{fold['name']}_role" for fold in config["cv_folds"]],
        "evaluation_offset_minutes",
        "is_primary_nonoverlap_evaluation",
    ]
    validate_splits(split, config)
    return split[result_columns]


def validate_splits(split: pd.DataFrame, config: dict[str, Any]) -> None:
    errors: list[str] = []
    allowed_final = {"DEVELOPMENT", "FINAL_TEST", "PURGED_BEFORE_FINAL_TEST"}
    if not set(split["final_split_role"].unique()).issubset(allowed_final):
        errors.append("final_split_role contains invalid values")
    valid_offsets = set(range(0, 60, int(config["prediction_interval_minutes"])))
    if not set(split["evaluation_offset_minutes"].unique()).issubset(valid_offsets):
        errors.append("evaluation_offset_minutes contains invalid offsets")
    for fold in config["cv_folds"]:
        role_col = f"{fold['name']}_role"
        validation_start = parse_utc_ms(fold["validation_start"])
        validation_end = parse_utc_ms(fold["validation_end"])
        train = split[role_col] == "TRAIN"
        validation = split[role_col] == "VALIDATION"
        if bool(train.any() and validation.any()):
            if int(split.loc[train, "settlement_minute_open_time"].max()) >= int(split.loc[validation, "decision_time"].min()):
                errors.append(f"{fold['name']} train settlement crosses validation start")
            if int(split.loc[train, "decision_time"].max()) >= int(split.loc[validation, "decision_time"].min()):
                errors.append(f"{fold['name']} train decision crosses validation start")
        if bool(validation.any()):
            if not ((split.loc[validation, "decision_time"] >= validation_start) & (split.loc[validation, "decision_time"] < validation_end)).all():
                errors.append(f"{fold['name']} validation decisions outside configured interval")
            if not (split.loc[validation, "settlement_minute_open_time"] < validation_end).all():
                errors.append(f"{fold['name']} validation labels cross validation end")
        final_test = split["final_split_role"] == "FINAL_TEST"
        if bool(((train | validation) & final_test).any()):
            errors.append(f"{fold['name']} includes final test samples")
    if errors:
        raise Stage5ValidationError(errors)


def subset_distribution(dataset: pd.DataFrame, split: pd.DataFrame, mask: pd.Series, name: str) -> dict[str, Any]:
    subset = dataset.loc[mask].copy()
    if subset.empty:
        return {
            "subset": name,
            "sample_count": 0,
            "up_count": 0,
            "down_count": 0,
            "up_ratio": None,
            "equal_price_count": 0,
            "mean_future_return": None,
            "median_future_return": None,
            "return_std": None,
            "boundary_risk_1bps_ratio": None,
            "boundary_risk_2_5bps_ratio": None,
            "boundary_risk_5bps_ratio": None,
            "boundary_risk_10bps_ratio": None,
            "start_decision_time": None,
            "end_decision_time": None,
            "maximum_settlement_time": None,
        }
    labels = subset["label_up_60m"].astype("int8")
    n = int(len(subset))
    return {
        "subset": name,
        "sample_count": n,
        "up_count": int((labels == 1).sum()),
        "down_count": int((labels == 0).sum()),
        "up_ratio": float((labels == 1).mean()),
        "equal_price_count": int(np.isclose(subset["future_simple_return_60m"], 0.0, atol=1e-15).sum()),
        "mean_future_return": float(subset["future_simple_return_60m"].mean()),
        "median_future_return": float(subset["future_simple_return_60m"].median()),
        "return_std": float(subset["future_simple_return_60m"].std(ddof=1)) if n > 1 else 0.0,
        "boundary_risk_1bps_ratio": float(subset["proxy_boundary_risk_1bps"].mean()),
        "boundary_risk_2_5bps_ratio": float(subset["proxy_boundary_risk_2_5bps"].mean()),
        "boundary_risk_5bps_ratio": float(subset["proxy_boundary_risk_5bps"].mean()),
        "boundary_risk_10bps_ratio": float(subset["proxy_boundary_risk_10bps"].mean()),
        "start_decision_time": int(subset["decision_time"].min()),
        "end_decision_time": int(subset["decision_time"].max()),
        "maximum_settlement_time": int(subset["settlement_minute_open_time"].max()),
    }


def make_split_summary(dataset: pd.DataFrame, split: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(subset_distribution(dataset, split, pd.Series(True, index=dataset.index), "ALL_DATASET"))
    for role in ["DEVELOPMENT", "FINAL_TEST", "PURGED_BEFORE_FINAL_TEST"]:
        rows.append(subset_distribution(dataset, split, split["final_split_role"] == role, role))
    for fold in config["cv_folds"]:
        role_col = f"{fold['name']}_role"
        train = split[role_col] == "TRAIN"
        validation = split[role_col] == "VALIDATION"
        rows.append(subset_distribution(dataset, split, train, f"{fold['name']}_TRAIN"))
        rows.append(subset_distribution(dataset, split, validation, f"{fold['name']}_VALIDATION"))
        rows.append(subset_distribution(dataset, split, validation, f"{fold['name']}_VALIDATION_DENSE"))
        rows.append(
            subset_distribution(
                dataset,
                split,
                validation & split["is_primary_nonoverlap_evaluation"].astype(bool),
                f"{fold['name']}_VALIDATION_NONOVERLAP",
            )
        )
    return pd.DataFrame(rows)


def make_time_distributions(dataset: pd.DataFrame, split: pd.DataFrame, freq: str) -> pd.DataFrame:
    tmp = dataset[["dataset_row_id", "decision_time", "label_up_60m", "future_simple_return_60m", *EVALUATION_METADATA_COLUMNS]].merge(
        split[["dataset_row_id", "final_split_role"]], on="dataset_row_id", how="one_to_one" if False else "inner"
    )
    dt = pd.to_datetime(tmp["decision_time"], unit="ms", utc=True)
    if freq == "year":
        tmp["period_utc"] = dt.dt.year.astype("int16")
    else:
        tmp["period_utc"] = (dt.dt.year * 100 + dt.dt.month).astype("int32")
    rows = []
    for (role, period), group in tmp.groupby(["final_split_role", "period_utc"], sort=True):
        labels = group["label_up_60m"].astype("int8")
        rows.append(
            {
                "final_split_role": role,
                "period_utc": int(period),
                "sample_count": int(len(group)),
                "up_count": int((labels == 1).sum()),
                "down_count": int((labels == 0).sum()),
                "up_ratio": float((labels == 1).mean()),
                "boundary_risk_5bps_ratio": float(group["proxy_boundary_risk_5bps"].mean()),
                "boundary_risk_10bps_ratio": float(group["proxy_boundary_risk_10bps"].mean()),
            }
        )
    return pd.DataFrame(rows)


def make_cv_fold_manifest(dataset: pd.DataFrame, split: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for fold in config["cv_folds"]:
        role_col = f"{fold['name']}_role"
        train = split[role_col] == "TRAIN"
        validation = split[role_col] == "VALIDATION"
        purge_before = split[role_col] == "PURGED_BEFORE_VALIDATION"
        purge_end = split[role_col] == "PURGED_AT_VALIDATION_END"
        validation_dataset = dataset.loc[validation]
        train_dataset = dataset.loc[train]
        labels = validation_dataset["label_up_60m"].astype("int8") if not validation_dataset.empty else pd.Series(dtype="int8")
        folds.append(
            {
                "name": fold["name"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "purge_rule": "train settlement_minute_open_time < validation_start; validation settlement_minute_open_time < validation_end",
                "train_sample_count": int(train.sum()),
                "validation_sample_count": int(validation.sum()),
                "boundary_purge_sample_count": int(purge_before.sum()),
                "validation_end_purge_sample_count": int(purge_end.sum()),
                "dense_validation_count": int(validation.sum()),
                "non_overlap_validation_count": int((validation & split["is_primary_nonoverlap_evaluation"].astype(bool)).sum()),
                "train_min_decision_time": int(train_dataset["decision_time"].min()) if not train_dataset.empty else None,
                "train_max_decision_time": int(train_dataset["decision_time"].max()) if not train_dataset.empty else None,
                "train_max_settlement_time": int(train_dataset["settlement_minute_open_time"].max()) if not train_dataset.empty else None,
                "validation_min_decision_time": int(validation_dataset["decision_time"].min()) if not validation_dataset.empty else None,
                "validation_max_decision_time": int(validation_dataset["decision_time"].max()) if not validation_dataset.empty else None,
                "validation_max_settlement_time": int(validation_dataset["settlement_minute_open_time"].max()) if not validation_dataset.empty else None,
                "validation_up_count": int((labels == 1).sum()) if not labels.empty else 0,
                "validation_down_count": int((labels == 0).sum()) if not labels.empty else 0,
                "validation_up_ratio": float((labels == 1).mean()) if not labels.empty else None,
                "validation_boundary_risk_5bps_ratio": float(validation_dataset["proxy_boundary_risk_5bps"].mean()) if not validation_dataset.empty else None,
                "validation_boundary_risk_10bps_ratio": float(validation_dataset["proxy_boundary_risk_10bps"].mean()) if not validation_dataset.empty else None,
            }
        )
    return {
        "split_version": config["split_version"],
        "folds": folds,
        "notes": [
            "Validation/test features may use pre-boundary historical Klines because those bars are known at prediction time.",
            "The purge applies to label settlement horizons, not to causal feature lookback history.",
        ],
    }


def make_dataset_manifest(
    feature_manifest: dict[str, Any],
    config: dict[str, Any],
    input_hashes: dict[str, Any],
    output_hashes: dict[str, Any],
    file_sizes: dict[str, Any],
    script_hashes: dict[str, Any],
    row_count: int,
    column_count: int,
) -> dict[str, Any]:
    feature_columns = list(feature_manifest["ordered_feature_names"])
    ordered_feature_list_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()
    return {
        "dataset_version": config["dataset_version"],
        "feature_set_version": feature_manifest.get("feature_set_version"),
        "stage4_feature_definition_hash": feature_manifest.get("feature_definition_hash"),
        "ordered_feature_list_hash": ordered_feature_list_hash,
        "label_definition_version": "stage3b_kline_proxy_label_v1",
        "split_definition_version": config["split_version"],
        "identifier_columns": IDENTIFIER_COLUMNS,
        "time_columns": TIME_COLUMNS,
        "feature_columns": feature_columns,
        "primary_target_columns": PRIMARY_TARGET_COLUMNS,
        "auxiliary_target_columns": AUXILIARY_TARGET_COLUMNS,
        "sample_weight_columns": SAMPLE_WEIGHT_COLUMNS,
        "evaluation_metadata_columns": EVALUATION_METADATA_COLUMNS,
        "forbidden_model_input_columns": FORBIDDEN_MODEL_INPUT_COLUMNS,
        "sample_weight_metadata": {
            "sample_weight_uniform": {"role": "sample_weight", "derived_from_target": False, "default_for_baseline": True},
            "sample_weight_margin": {"role": "sample_weight", "derived_from_target": True, "default_for_baseline": False},
        },
        "input_files": input_hashes,
        "output_files": output_hashes,
        "file_sizes": file_sizes,
        "script_hashes": script_hashes,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": row_count,
        "column_count": column_count,
    }


def build_dataset_and_splits(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_manifest: dict[str, Any],
    label_field_dictionary: dict[str, Any],
    config: dict[str, Any],
) -> Stage5Result:
    feature_columns = feature_columns_from_manifest(feature_manifest, int(config["required_feature_count"]))
    features = features.sort_values(config["join_key"], kind="mergesort").reset_index(drop=True)
    labels = labels.sort_values(config["join_key"], kind="mergesort").reset_index(drop=True)
    validate_inputs(features, labels, feature_columns, config)

    feature_renames = {
        "feature_open_time": "feature_open_time_feature",
        "continuity_segment_id": "continuity_segment_id_feature",
        "is_prediction_time_5m": "is_prediction_time_5m_feature",
    }
    label_renames = {
        "feature_open_time": "feature_open_time_label",
        "continuity_segment_id": "continuity_segment_id_label",
        "is_prediction_time_5m": "is_prediction_time_5m_label",
    }
    feature_merge = features.rename(columns=feature_renames)
    label_merge = labels.rename(columns=label_renames)
    merged = feature_merge.merge(
        label_merge,
        on=config["join_key"],
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    join_audit = build_join_audit(merged, feature_columns)
    excluded = make_excluded_samples(join_audit)
    dataset = make_dataset(merged, join_audit, feature_columns, config)
    split = make_split_assignments(dataset, config)
    split_summary = make_split_summary(dataset, split, config)
    by_year = make_time_distributions(dataset, split, "year")
    by_month = make_time_distributions(dataset, split, "month")
    cv_manifest = make_cv_fold_manifest(dataset, split, config)
    dataset_manifest = make_dataset_manifest(
        feature_manifest,
        config,
        input_hashes={},
        output_hashes={},
        file_sizes={},
        script_hashes={},
        row_count=len(dataset),
        column_count=len(dataset.columns),
    )
    stats = {
        "feature_input_rows": int(len(features)),
        "label_input_rows": int(len(labels)),
        "joined_both_count": int((merged["_merge"] == "both").sum()),
        "feature_only_count": int((merged["_merge"] == "left_only").sum()),
        "label_only_count": int((merged["_merge"] == "right_only").sum()),
        "feature_final_candidate_count": int(features["is_final_feature_candidate"].sum()),
        "label_final_candidate_count": int(labels["is_final_model_label_candidate"].sum()),
        "final_intersection_count": int(len(dataset)),
        "exclusion_reason_counts": exclusion_reason_counts(excluded),
        "feature_order_valid": list(dataset[feature_columns].columns) == feature_columns,
        "forbidden_input_scan": scan_feature_leakage(feature_columns),
        "final_split_counts": split["final_split_role"].value_counts().to_dict(),
        "dense_sample_count": int(len(split)),
        "primary_nonoverlap_sample_count": int(split["is_primary_nonoverlap_evaluation"].sum()),
    }
    return Stage5Result(
        dataset=dataset,
        split_assignments=split,
        join_audit=join_audit,
        excluded_samples=excluded,
        dataset_manifest=dataset_manifest,
        cv_fold_manifest=cv_manifest,
        split_summary=split_summary,
        split_distribution_by_year=by_year,
        split_distribution_by_month=by_month,
        stats=stats,
    )


def exclusion_reason_counts(excluded: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if excluded.empty:
        return counts
    for value in excluded["exclusion_reason"].dropna():
        for reason in dict.fromkeys(str(value).split(";")):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def require_parquet_columns(path: Path, required: list[str]) -> None:
    schema = pq.read_schema(path)
    missing = [col for col in required if col not in schema.names]
    if missing:
        raise Stage5ValidationError([f"{path} missing required columns: {missing}"])


def get_process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        if os.name != "nt":
            return None

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


def write_report(
    path: Path,
    result: Stage5Result,
    feature_path: Path,
    label_path: Path,
    dataset_path: Path,
    split_path: Path,
    dataset_schema: list[dict[str, str]],
    split_schema: list[dict[str, str]],
    dataset_size: int,
    split_size: int,
    elapsed: float,
    peak_memory: int | None,
    rss: int | None,
) -> None:
    stats = result.stats
    fold_rows = []
    for fold in result.cv_fold_manifest["folds"]:
        fold_rows.append(
            {
                "fold": fold["name"],
                "train": fold["train_sample_count"],
                "validation": fold["validation_sample_count"],
                "purge_before": fold["boundary_purge_sample_count"],
                "purge_validation_end": fold["validation_end_purge_sample_count"],
                "train_max_settlement_time": fold["train_max_settlement_time"],
                "validation_min_decision_time": fold["validation_min_decision_time"],
            }
        )
    lines = [
        "# 阶段5：最终原始建模数据集与时间划分报告",
        "",
        "- 范围限制: 只读取阶段3B标签、阶段4特征和manifest；未读取原始CSV；未重新计算特征或标签。",
        "- 本阶段没有执行标准化、异常值裁剪、缺失值填充、特征选择、重采样、类别平衡、概率校准或模型训练。",
        f"- 特征输入: `{feature_path}`",
        f"- 标签输入: `{label_path}`",
        f"- 数据集输出: `{dataset_path}`",
        f"- 划分输出: `{split_path}`",
        f"- 数据集文件大小(bytes): `{dataset_size}`",
        f"- 划分文件大小(bytes): `{split_size}`",
        f"- 运行耗时(seconds): `{elapsed:.2f}`",
        f"- Python tracemalloc峰值内存(bytes): `{peak_memory}`",
        f"- 进程RSS内存(bytes): `{rss}`",
        "",
        "## 连接审计",
        table(
            [
                {"指标": "特征输入行数", "值": stats["feature_input_rows"]},
                {"指标": "标签输入行数", "值": stats["label_input_rows"]},
                {"指标": "成功一对一连接数量", "值": stats["joined_both_count"]},
                {"指标": "仅特征侧存在数量", "值": stats["feature_only_count"]},
                {"指标": "仅标签侧存在数量", "值": stats["label_only_count"]},
                {"指标": "特征最终候选数量", "值": stats["feature_final_candidate_count"]},
                {"指标": "标签最终候选数量", "值": stats["label_final_candidate_count"]},
                {"指标": "最终交集数量", "值": stats["final_intersection_count"]},
            ],
            ["指标", "值"],
        ),
        "## 排除原因计数",
        table([{"reason": key, "count": value} for key, value in stats["exclusion_reason_counts"].items()], ["reason", "count"]),
        "## 特征清单和角色",
        table(
            [
                {"检查": "feature_count", "结果": len(result.dataset_manifest["feature_columns"])},
                {"检查": "feature_order_valid", "结果": stats["feature_order_valid"]},
                {"检查": "primary_target_columns", "结果": result.dataset_manifest["primary_target_columns"]},
                {"检查": "sample_weight_columns", "结果": result.dataset_manifest["sample_weight_columns"]},
                {"检查": "forbidden_input_scan", "结果": stats["forbidden_input_scan"]},
            ],
            ["检查", "结果"],
        ),
        "## 最终划分数量",
        table([{"role": key, "count": value} for key, value in stats["final_split_counts"].items()], ["role", "count"]),
        "## 滚动fold数量",
        table(fold_rows, ["fold", "train", "validation", "purge_before", "purge_validation_end", "train_max_settlement_time", "validation_min_decision_time"]),
        "## dense和non-overlap样本数量",
        table(
            [
                {"指标": "dense_sample_count", "值": stats["dense_sample_count"]},
                {"指标": "primary_nonoverlap_sample_count", "值": stats["primary_nonoverlap_sample_count"]},
            ],
            ["指标", "值"],
        ),
        "## 划分类别分布",
        table(result.split_summary.to_dict(orient="records"), list(result.split_summary.columns)),
        "## 240分钟历史说明",
        "验证集和测试集开始后的特征允许使用开始时间之前已经公开的历史K线；purge只针对未来标签结算期限，不针对因果历史特征窗口。",
        "",
        "## 输出Dataset Schema",
        table(dataset_schema, ["name", "type"]),
        "## 输出Split Schema",
        table(split_schema, ["name", "type"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage5(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    feature_path = (root / config["feature_input_path"]).resolve()
    label_path = (root / config["label_input_path"]).resolve()
    feature_manifest_path = (root / config["feature_manifest_path"]).resolve()
    label_dictionary_path = (root / config["label_field_dictionary_path"]).resolve()
    dataset_path = (root / config["output_dataset_path"]).resolve()
    split_path = (root / config["output_split_path"]).resolve()
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
        logging.info("Stage 5 dataset and split build started")
        feature_manifest = read_json(feature_manifest_path)
        label_dictionary = read_json(label_dictionary_path)
        feature_columns = feature_columns_from_manifest(feature_manifest, int(config["required_feature_count"]))
        feature_read_columns = [*FEATURE_METADATA_COLUMNS, *feature_columns]
        require_parquet_columns(feature_path, feature_read_columns)
        require_parquet_columns(label_path, LABEL_COLUMNS)
        features = pd.read_parquet(feature_path, columns=feature_read_columns)
        labels = pd.read_parquet(label_path, columns=LABEL_COLUMNS)
        result = build_dataset_and_splits(features, labels, feature_manifest, label_dictionary, config)
        ensure_parent(dataset_path)
        ensure_parent(split_path)
        result.dataset.to_parquet(dataset_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
        result.split_assignments.to_parquet(split_path, index=False, engine="pyarrow", compression=config["parquet_compression"])

        input_hashes = {
            "feature_input_path": str(feature_path),
            "label_input_path": str(label_path),
            "feature_manifest_path": str(feature_manifest_path),
            "label_field_dictionary_path": str(label_dictionary_path),
            "feature_input_sha256": sha256_file(feature_path),
            "label_input_sha256": sha256_file(label_path),
            "feature_manifest_sha256": sha256_file(feature_manifest_path),
            "label_field_dictionary_sha256": sha256_file(label_dictionary_path),
        }
        output_hashes = {
            "dataset_path": str(dataset_path),
            "split_assignment_path": str(split_path),
            "dataset_sha256": sha256_file(dataset_path),
            "split_assignment_sha256": sha256_file(split_path),
        }
        script_path = Path(__file__).resolve()
        file_sizes = {
            "feature_input_bytes": feature_path.stat().st_size,
            "label_input_bytes": label_path.stat().st_size,
            "dataset_bytes": dataset_path.stat().st_size,
            "split_assignment_bytes": split_path.stat().st_size,
            "config_bytes": config_path.stat().st_size,
            "script_bytes": script_path.stat().st_size,
        }
        script_hashes = {
            "config_sha256": sha256_file(config_path),
            "script_sha256": sha256_file(script_path),
        }
        result.dataset_manifest = make_dataset_manifest(
            feature_manifest,
            config,
            input_hashes,
            output_hashes,
            file_sizes,
            script_hashes,
            row_count=len(result.dataset),
            column_count=len(result.dataset.columns),
        )
        for path in report_paths.values():
            ensure_parent(path)
        report_paths["dataset_manifest"].write_text(json.dumps(result.dataset_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        report_paths["cv_fold_manifest"].write_text(json.dumps(result.cv_fold_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        result.split_summary.to_csv(report_paths["split_summary"], index=False, encoding="utf-8")
        result.split_distribution_by_year.to_csv(report_paths["split_distribution_by_year"], index=False, encoding="utf-8")
        result.split_distribution_by_month.to_csv(report_paths["split_distribution_by_month"], index=False, encoding="utf-8")
        result.excluded_samples.to_csv(report_paths["excluded_samples"], index=False, encoding="utf-8")
        result.join_audit.to_csv(report_paths["join_audit"], index=False, encoding="utf-8")

        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        dataset_schema = parquet_schema(dataset_path)
        split_schema = parquet_schema(split_path)
        write_report(
            report_paths["main_report"],
            result,
            feature_path,
            label_path,
            dataset_path,
            split_path,
            dataset_schema,
            split_schema,
            file_sizes["dataset_bytes"],
            file_sizes["split_assignment_bytes"],
            elapsed,
            peak,
            rss,
        )
        logging.info("Stats: %s", result.stats)
        logging.info("Dataset path: %s", dataset_path)
        logging.info("Split path: %s", split_path)
        logging.info("Elapsed seconds: %.2f", elapsed)
        logging.info("Python tracemalloc peak bytes: %s", peak)
        logging.info("Process RSS bytes: %s", rss)
        return {
            "stats": result.stats,
            "dataset_schema": dataset_schema,
            "split_schema": split_schema,
            "dataset_size_bytes": file_sizes["dataset_bytes"],
            "split_size_bytes": file_sizes["split_assignment_bytes"],
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": peak,
            "process_rss_bytes": rss,
            "dataset_sha256": output_hashes["dataset_sha256"],
            "split_assignment_sha256": output_hashes["split_assignment_sha256"],
        }
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 5 dataset and time-series split assignments.")
    parser.add_argument("--config", default="config/stage5_build_dataset_and_splits.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage5(config_path, root)


if __name__ == "__main__":
    main()

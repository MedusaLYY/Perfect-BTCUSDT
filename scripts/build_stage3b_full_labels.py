from __future__ import annotations

import argparse
import json
import logging
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class Stage3BValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Stage3BResult:
    df: pd.DataFrame
    stats: dict[str, Any]
    distribution_by_year: pd.DataFrame
    distribution_by_month: pd.DataFrame
    distribution_by_segment: pd.DataFrame
    distribution_by_margin: pd.DataFrame
    invalid_label_candidates: pd.DataFrame
    extreme_return_samples: pd.DataFrame
    field_dictionary: dict[str, Any]


REQUIRED_INPUT_COLUMNS = [
    "open_time",
    "open",
    "continuity_segment_id",
    "decision_time",
    "is_prediction_time_5m",
    "has_history_240m",
    "has_future_61m",
    "is_model_candidate",
]

OUTPUT_COLUMNS = [
    "feature_open_time",
    "decision_time",
    "entry_minute_open_time",
    "settlement_minute_open_time",
    "continuity_segment_id",
    "is_prediction_time_5m",
    "has_history_240m",
    "has_future_61m",
    "is_model_candidate",
    "has_entry_price_proxy",
    "has_settlement_price_proxy",
    "same_continuity_segment",
    "exact_horizon_match",
    "is_valid_proxy_label",
    "is_final_model_label_candidate",
    "entry_price_proxy",
    "settlement_price_proxy",
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

MODEL_FEATURE_TOKENS = [
    "ema",
    "atr",
    "momentum",
    "volatility",
    "rolling",
    "zscore",
    "rsi",
    "macd",
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ms_to_utc_iso(ms: int | float | None) -> str | None:
    if ms is None or pd.isna(ms):
        return None
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_无_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def require_columns(path: Path, required: list[str]) -> None:
    names = pq.read_schema(path).names
    missing = [col for col in required if col not in names]
    if missing:
        raise Stage3BValidationError([f"Stage 3B input missing required columns: {missing}"])


def threshold_suffix(value: float) -> str:
    return f"{value:g}".replace(".", "_")


def make_margin_bucket(values_bps: pd.Series, bins: list[float]) -> pd.Series:
    edges = [float(v) for v in bins] + [np.inf]
    labels = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if np.isinf(right):
            labels.append(f"[{left:g}, +inf)")
        else:
            labels.append(f"[{left:g}, {right:g})")
    return pd.cut(values_bps, bins=edges, labels=labels, right=False, include_lowest=True)


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    return pd.Series(np.isfinite(values) & (values > 0), index=series.index)


def first_examples(df: pd.DataFrame, mask: pd.Series, columns: list[str], limit: int = 5) -> list[dict[str, Any]]:
    return df.loc[mask, columns].head(limit).to_dict(orient="records")


def validate_input_frame(df: pd.DataFrame, config: dict[str, Any]) -> None:
    errors: list[str] = []
    expected_interval_ms = int(config["expected_interval_ms"])
    prediction_interval_ms = int(config["prediction_interval_minutes"]) * expected_interval_ms

    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Input DataFrame missing required columns: {missing}")
    if errors:
        raise Stage3BValidationError(errors)
    if df.empty:
        raise Stage3BValidationError(["Input DataFrame is empty"])

    if not df["open_time"].is_monotonic_increasing:
        errors.append("open_time must be sorted by integer milliseconds in increasing order")
    if not df["open_time"].is_unique:
        errors.append("open_time must be unique for exact many-to-one time joins")

    expected_decision = pd.to_numeric(df["open_time"], errors="coerce") + expected_interval_ms
    bad_decision = pd.to_numeric(df["decision_time"], errors="coerce") != expected_decision
    if bool(bad_decision.any()):
        errors.append(
            "decision_time must equal open_time + expected_interval_ms; "
            f"examples={first_examples(df, bad_decision, ['open_time', 'decision_time'])}"
        )

    prediction_flag = df["is_prediction_time_5m"].astype("bool")
    modulo_prediction = (pd.to_numeric(df["decision_time"], errors="coerce") % prediction_interval_ms) == 0
    bad_prediction_flag = prediction_flag != modulo_prediction
    if bool(bad_prediction_flag.any()):
        errors.append(
            "is_prediction_time_5m does not match decision_time modulo prediction interval; "
            f"examples={first_examples(df, bad_prediction_flag, ['open_time', 'decision_time', 'is_prediction_time_5m'])}"
        )

    bad_price = ~finite_positive(df["open"])
    if bool(bad_price.any()):
        errors.append(f"Input open price must be finite and > 0; examples={first_examples(df, bad_price, ['open_time', 'open'])}")

    if errors:
        raise Stage3BValidationError(errors)


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if int(config["expected_interval_ms"]) <= 0:
        errors.append("expected_interval_ms must be positive")
    if int(config["prediction_interval_minutes"]) <= 0:
        errors.append("prediction_interval_minutes must be positive")
    if int(config["horizon_minutes"]) != 60:
        errors.append("Stage 3B constructs 60-minute labels; horizon_minutes must be 60")
    bins = [float(v) for v in config["return_margin_bins_bps"]]
    if bins[0] != 0 or any(right <= left for left, right in zip(bins[:-1], bins[1:], strict=True)):
        errors.append("return_margin_bins_bps must start at 0 and be strictly increasing")
    thresholds = [float(v) for v in config["proxy_boundary_thresholds_bps"]]
    if thresholds != [1.0, 2.5, 5.0, 10.0]:
        errors.append("proxy_boundary_thresholds_bps must be [1, 2.5, 5, 10]")
    weight_settings = config["sample_weight_settings"]
    if float(weight_settings["margin_full_weight_bps"]) <= 0:
        errors.append("margin_full_weight_bps must be positive")
    minimum_weight = float(weight_settings["minimum_margin_weight"])
    if not 0 <= minimum_weight <= 1:
        errors.append("minimum_margin_weight must be within [0, 1]")
    if errors:
        raise Stage3BValidationError(errors)


def build_full_labels(model_base: pd.DataFrame, config: dict[str, Any]) -> Stage3BResult:
    validate_config(config)
    validate_input_frame(model_base, config)

    expected_interval_ms = int(config["expected_interval_ms"])
    horizon_ms = int(config["horizon_minutes"]) * expected_interval_ms
    bins = [float(v) for v in config["return_margin_bins_bps"]]
    thresholds = [float(v) for v in config["proxy_boundary_thresholds_bps"]]
    weight_settings = config["sample_weight_settings"]
    full_weight_bps = float(weight_settings["margin_full_weight_bps"])
    minimum_weight = float(weight_settings["minimum_margin_weight"])

    source = model_base[REQUIRED_INPUT_COLUMNS].copy(deep=False)
    source["is_prediction_time_5m"] = source["is_prediction_time_5m"].astype("bool")
    source["has_history_240m"] = source["has_history_240m"].astype("bool")
    source["has_future_61m"] = source["has_future_61m"].astype("bool")
    source["is_model_candidate"] = source["is_model_candidate"].astype("bool")

    base = source.loc[source["is_prediction_time_5m"]].copy()
    base.rename(columns={"open_time": "feature_open_time"}, inplace=True)
    base["entry_minute_open_time"] = base["decision_time"].astype("int64")
    base["settlement_minute_open_time"] = base["entry_minute_open_time"] + horizon_ms

    minute = source[["open_time", "open", "continuity_segment_id"]].copy(deep=False)
    entry = minute.rename(
        columns={
            "open_time": "entry_minute_open_time",
            "open": "entry_price_proxy",
            "continuity_segment_id": "entry_continuity_segment_id",
        }
    )
    settlement = minute.rename(
        columns={
            "open_time": "settlement_minute_open_time",
            "open": "settlement_price_proxy",
            "continuity_segment_id": "settlement_continuity_segment_id",
        }
    )
    out = base.merge(entry, on="entry_minute_open_time", how="left", validate="many_to_one", sort=False)
    out = out.merge(settlement, on="settlement_minute_open_time", how="left", validate="many_to_one", sort=False)

    entry_row_exists = out["entry_continuity_segment_id"].notna()
    settlement_row_exists = out["settlement_continuity_segment_id"].notna()
    out["has_entry_price_proxy"] = out["entry_price_proxy"].notna()
    out["has_settlement_price_proxy"] = out["settlement_price_proxy"].notna()
    out["same_continuity_segment"] = (
        entry_row_exists
        & settlement_row_exists
        & (out["entry_continuity_segment_id"] == out["continuity_segment_id"])
        & (out["settlement_continuity_segment_id"] == out["continuity_segment_id"])
    )
    out["exact_horizon_match"] = (
        entry_row_exists
        & settlement_row_exists
        & ((out["settlement_minute_open_time"] - out["entry_minute_open_time"]) == horizon_ms)
    )
    time_segment_connection_ok = entry_row_exists & settlement_row_exists & out["same_continuity_segment"] & out["exact_horizon_match"]
    future_mismatch = out["has_future_61m"].astype("bool") != time_segment_connection_ok.astype("bool")
    if bool(future_mismatch.any()):
        examples = out.loc[
            future_mismatch,
            [
                "feature_open_time",
                "decision_time",
                "entry_minute_open_time",
                "settlement_minute_open_time",
                "continuity_segment_id",
                "entry_continuity_segment_id",
                "settlement_continuity_segment_id",
                "has_future_61m",
            ],
        ].head(20)
        raise Stage3BValidationError(
            [
                "Stage 2A has_future_61m is inconsistent with exact time/segment joins",
                f"mismatch_count={int(future_mismatch.sum())}",
                f"examples={examples.to_dict(orient='records')}",
            ]
        )

    entry_price_valid = finite_positive(out["entry_price_proxy"])
    settlement_price_valid = finite_positive(out["settlement_price_proxy"])
    out["is_valid_proxy_label"] = (
        out["has_entry_price_proxy"]
        & out["has_settlement_price_proxy"]
        & out["same_continuity_segment"]
        & out["exact_horizon_match"]
        & entry_price_valid
        & settlement_price_valid
    ).astype("bool")
    out["is_final_model_label_candidate"] = (out["is_model_candidate"].astype("bool") & out["is_valid_proxy_label"]).astype("bool")

    out["future_simple_return_60m"] = np.nan
    out["future_log_return_60m"] = np.nan
    valid = out["is_valid_proxy_label"]
    ratio = out.loc[valid, "settlement_price_proxy"] / out.loc[valid, "entry_price_proxy"]
    out.loc[valid, "future_simple_return_60m"] = ratio - 1.0
    out.loc[valid, "future_log_return_60m"] = np.log(ratio)
    out["label_up_60m"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[valid, "label_up_60m"] = (
        out.loc[valid, "settlement_price_proxy"] > out.loc[valid, "entry_price_proxy"]
    ).astype("int8")

    out["absolute_future_return_bps"] = np.round(out["future_simple_return_60m"].abs() * 10000.0, 9)
    out["proxy_margin_bucket"] = make_margin_bucket(out["absolute_future_return_bps"], bins).astype("string")
    for threshold in thresholds:
        out[f"proxy_boundary_risk_{threshold_suffix(threshold)}bps"] = (
            out["absolute_future_return_bps"] < threshold
        ).fillna(False).astype("bool")
    out["sample_weight_uniform"] = np.float32(1.0)
    margin_weight = (out["absolute_future_return_bps"] / full_weight_bps).clip(lower=minimum_weight, upper=1.0)
    out["sample_weight_margin"] = margin_weight

    validate_label_quality(out, config)

    invalid_detail = make_invalid_label_candidates(out, entry_row_exists, settlement_row_exists, entry_price_valid, settlement_price_valid)
    output = out[OUTPUT_COLUMNS].copy()
    if any(token in col.lower() for col in output.columns for token in MODEL_FEATURE_TOKENS):
        raise Stage3BValidationError(["Stage 3B output contains model feature-like columns"])

    distribution_by_year, distribution_by_month, distribution_by_segment, distribution_by_margin = make_distributions(output)
    extreme = make_extreme_return_samples(output, int(config["extreme_return_rows"]))
    stats = make_stats(output, invalid_detail, distribution_by_month, config)
    return Stage3BResult(
        df=output,
        stats=stats,
        distribution_by_year=distribution_by_year,
        distribution_by_month=distribution_by_month,
        distribution_by_segment=distribution_by_segment,
        distribution_by_margin=distribution_by_margin,
        invalid_label_candidates=invalid_detail,
        extreme_return_samples=extreme,
        field_dictionary=make_field_dictionary(config),
    )


def validate_label_quality(out: pd.DataFrame, config: dict[str, Any]) -> None:
    errors: list[str] = []
    valid = out["is_valid_proxy_label"].astype("bool")
    abs_tol = float(config["floating_absolute_tolerance"])

    if bool((valid & ~(finite_positive(out["entry_price_proxy"]) & finite_positive(out["settlement_price_proxy"]))).any()):
        errors.append("Valid labels contain non-finite or non-positive proxy prices")

    simple = out.loc[valid, "future_simple_return_60m"]
    log_return = out.loc[valid, "future_log_return_60m"]
    if bool((~np.isfinite(simple)).any()) or bool((~np.isfinite(log_return)).any()):
        errors.append("Valid labels contain non-finite future returns")

    label = out.loc[valid, "label_up_60m"].astype("int8")
    if bool(((simple > abs_tol) & (label != 1)).any()):
        errors.append("Positive simple returns must have label_up_60m=1")
    if bool(((simple <= abs_tol) & (label != 0)).any()):
        errors.append("Non-positive simple returns must have label_up_60m=0")
    sign_mismatch = np.sign(simple.to_numpy(dtype="float64", copy=False)) != np.sign(log_return.to_numpy(dtype="float64", copy=False))
    if bool(np.any(sign_mismatch)):
        errors.append("Simple return and log return directions are inconsistent")
    invalid = ~valid
    if bool(out.loc[invalid, "label_up_60m"].notna().any()):
        errors.append("Invalid proxy labels must keep label_up_60m missing")

    if errors:
        raise Stage3BValidationError(errors)


def make_invalid_label_candidates(
    out: pd.DataFrame,
    entry_row_exists: pd.Series,
    settlement_row_exists: pd.Series,
    entry_price_valid: pd.Series,
    settlement_price_valid: pd.Series,
) -> pd.DataFrame:
    invalid = out.loc[~out["is_valid_proxy_label"]].copy()
    if invalid.empty:
        return pd.DataFrame(
            columns=[
                "feature_open_time",
                "decision_time",
                "entry_minute_open_time",
                "settlement_minute_open_time",
                "is_model_candidate",
                "missing_entry_time",
                "missing_settlement_time",
                "cross_segment",
                "exact_horizon_failed",
                "entry_price_not_finite_positive",
                "settlement_price_not_finite_positive",
            ]
        )
    invalid["missing_entry_time"] = ~entry_row_exists.loc[invalid.index]
    invalid["missing_settlement_time"] = ~settlement_row_exists.loc[invalid.index]
    invalid["cross_segment"] = entry_row_exists.loc[invalid.index] & settlement_row_exists.loc[invalid.index] & ~invalid["same_continuity_segment"]
    invalid["exact_horizon_failed"] = entry_row_exists.loc[invalid.index] & settlement_row_exists.loc[invalid.index] & ~invalid["exact_horizon_match"]
    invalid["entry_price_not_finite_positive"] = invalid["has_entry_price_proxy"] & ~entry_price_valid.loc[invalid.index]
    invalid["settlement_price_not_finite_positive"] = invalid["has_settlement_price_proxy"] & ~settlement_price_valid.loc[invalid.index]
    cols = [
        "feature_open_time",
        "decision_time",
        "entry_minute_open_time",
        "settlement_minute_open_time",
        "continuity_segment_id",
        "has_history_240m",
        "has_future_61m",
        "is_model_candidate",
        "has_entry_price_proxy",
        "has_settlement_price_proxy",
        "same_continuity_segment",
        "exact_horizon_match",
        "missing_entry_time",
        "missing_settlement_time",
        "cross_segment",
        "exact_horizon_failed",
        "entry_price_not_finite_positive",
        "settlement_price_not_finite_positive",
    ]
    return add_utc_columns(invalid[cols])


def add_utc_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in ["feature_open_time", "decision_time", "entry_minute_open_time", "settlement_minute_open_time"]:
        if col in result.columns:
            result[f"{col}_utc"] = result[col].map(ms_to_utc_iso)
    return result


def make_distribution_rows(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    valid = df.loc[df["is_valid_proxy_label"]].copy()
    rows: list[dict[str, Any]] = []
    for keys, group in valid.groupby(group_columns, sort=True, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        labels = group["label_up_60m"].astype("int8")
        sample_count = int(len(group))
        row = {name: value for name, value in zip(group_columns, keys, strict=True)}
        row.update(
            {
                "sample_count": sample_count,
                "up_count": int((labels == 1).sum()),
                "down_or_equal_count": int((labels == 0).sum()),
                "up_ratio": float((labels == 1).mean()) if sample_count else None,
                "mean_return": float(group["future_simple_return_60m"].mean()) if sample_count else None,
                "median_return": float(group["future_simple_return_60m"].median()) if sample_count else None,
                "return_std": float(group["future_simple_return_60m"].std(ddof=1)) if sample_count > 1 else 0.0,
                "boundary_risk_5bps_ratio": float(group["proxy_boundary_risk_5bps"].mean()) if sample_count else None,
                "boundary_risk_10bps_ratio": float(group["proxy_boundary_risk_10bps"].mean()) if sample_count else None,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_distributions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    working = df.copy(deep=False)
    decision_dt = pd.to_datetime(working["decision_time"], unit="ms", utc=True)
    working = working.assign(
        decision_year_utc=decision_dt.dt.year.astype("int16"),
        decision_month_utc=(decision_dt.dt.year * 100 + decision_dt.dt.month).astype("int32"),
    )
    by_year = make_distribution_rows(working, ["decision_year_utc"])
    by_month = make_distribution_rows(working, ["decision_month_utc"])
    by_segment = make_distribution_rows(working, ["continuity_segment_id"])
    by_margin = make_distribution_rows(working, ["proxy_margin_bucket"])
    if not by_margin.empty:
        by_margin["proxy_margin_bucket"] = by_margin["proxy_margin_bucket"].astype(str)
    return by_year, by_month, by_segment, by_margin


def make_extreme_return_samples(df: pd.DataFrame, rows_per_side: int) -> pd.DataFrame:
    valid = df.loc[df["is_valid_proxy_label"]].copy()
    if valid.empty:
        return pd.DataFrame()
    largest_up = valid.nlargest(rows_per_side, "future_simple_return_60m").copy()
    largest_up["extreme_type"] = "largest_up"
    largest_down = valid.nsmallest(rows_per_side, "future_simple_return_60m").copy()
    largest_down["extreme_type"] = "largest_down"
    cols = [
        "extreme_type",
        "feature_open_time",
        "decision_time",
        "entry_minute_open_time",
        "settlement_minute_open_time",
        "continuity_segment_id",
        "entry_price_proxy",
        "settlement_price_proxy",
        "future_simple_return_60m",
        "future_log_return_60m",
        "label_up_60m",
        "absolute_future_return_bps",
        "proxy_margin_bucket",
        "is_final_model_label_candidate",
    ]
    return add_utc_columns(pd.concat([largest_up[cols], largest_down[cols]], ignore_index=True))


def series_distribution(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "p01": float(clean.quantile(0.01)),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def make_stats(
    df: pd.DataFrame,
    invalid_detail: pd.DataFrame,
    distribution_by_month: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    valid = df["is_valid_proxy_label"].astype("bool")
    labels = df.loc[valid, "label_up_60m"].astype("int8")
    valid_count = int(valid.sum())
    up_count = int((labels == 1).sum())
    down_count = int((labels == 0).sum())
    equal_count = int(
        (
            valid
            & np.isclose(
                df["entry_price_proxy"],
                df["settlement_price_proxy"],
                rtol=float(config["floating_relative_tolerance"]),
                atol=float(config["floating_absolute_tolerance"]),
                equal_nan=False,
            )
        ).sum()
    )
    invalid_reasons = {
        "invalid_label_rows": int((~valid).sum()),
        "missing_entry_time": int(invalid_detail.get("missing_entry_time", pd.Series(dtype=bool)).sum()),
        "missing_settlement_time": int(invalid_detail.get("missing_settlement_time", pd.Series(dtype=bool)).sum()),
        "cross_segment": int(invalid_detail.get("cross_segment", pd.Series(dtype=bool)).sum()),
        "exact_horizon_failed": int(invalid_detail.get("exact_horizon_failed", pd.Series(dtype=bool)).sum()),
        "entry_price_not_finite_positive": int(invalid_detail.get("entry_price_not_finite_positive", pd.Series(dtype=bool)).sum()),
        "settlement_price_not_finite_positive": int(invalid_detail.get("settlement_price_not_finite_positive", pd.Series(dtype=bool)).sum()),
        "model_candidate_invalid_label": int((df["is_model_candidate"] & ~valid).sum()),
    }
    boundary_counts = {}
    for threshold in [float(v) for v in config["proxy_boundary_thresholds_bps"]]:
        suffix = threshold_suffix(threshold)
        count = int((valid & (df["absolute_future_return_bps"] < threshold)).sum())
        boundary_counts[f"abs_return_lt_{suffix}bps_count"] = count
        boundary_counts[f"abs_return_lt_{suffix}bps_ratio"] = count / valid_count if valid_count else None

    stability = monthly_stability_diagnostics(distribution_by_month, config)
    return {
        "output_rows": int(len(df)),
        "prediction_time_count": int(df["is_prediction_time_5m"].sum()),
        "model_candidate_count": int(df["is_model_candidate"].sum()),
        "valid_proxy_label_count": valid_count,
        "final_model_label_candidate_count": int(df["is_final_model_label_candidate"].sum()),
        "invalid_reasons": invalid_reasons,
        "up_count": up_count,
        "down_or_equal_count": down_count,
        "up_ratio": up_count / valid_count if valid_count else None,
        "down_or_equal_ratio": down_count / valid_count if valid_count else None,
        "equal_price_count": equal_count,
        "boundary_counts": boundary_counts,
        "simple_return_distribution": series_distribution(df.loc[valid, "future_simple_return_60m"]),
        "log_return_distribution": series_distribution(df.loc[valid, "future_log_return_60m"]),
        "time_join_validation": {
            "has_future_61m_exact_join_mismatch_count": 0,
            "all_entry_and_settlement_keys_joined_by_integer_ms": True,
            "all_valid_labels_same_segment": bool(df.loc[valid, "same_continuity_segment"].all()) if valid_count else True,
            "all_valid_labels_exact_60m_horizon": bool(df.loc[valid, "exact_horizon_match"].all()) if valid_count else True,
        },
        "stage3a_proxy_validation_summary": {
            "validated_window_utc": "2022-09-02 to 2022-09-04, about 66.77 hours of agg trades overlap",
            "kline_open_vs_5s_vwap_agreement_rate": 0.9822784810126582,
            "primary_reliable_agreement_rate": 0.984873949579832,
            "agreement_when_abs_return_ge_5bps": 1.0,
            "agreement_when_abs_return_ge_10bps": 1.0,
            "pearson_return_correlation": 0.9981456895632609,
            "mean_return_difference_bps": -0.009433021543726338,
            "proxy_label_recommendation": "ACCEPT_WITH_LIMITATIONS",
        },
        "monthly_stability_diagnostics": stability,
    }


def monthly_stability_diagnostics(distribution_by_month: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    if distribution_by_month.empty:
        return {
            "largest_up_ratio_diff_month": None,
            "small_sample_months": [],
            "high_near_zero_5bps_months": [],
        }
    settings = config.get("stability_diagnostic_settings", {})
    small_ratio = float(settings.get("small_month_count_ratio_of_median", 0.5))
    high_std_multiplier = float(settings.get("near_zero_high_std_multiplier", 2.0))
    month = distribution_by_month.copy()
    total = int(month["sample_count"].sum())
    overall_up_ratio = float((month["up_count"].sum() / total)) if total else None
    if overall_up_ratio is not None:
        month["up_ratio_abs_diff_from_overall"] = (month["up_ratio"] - overall_up_ratio).abs()
        largest = month.sort_values("up_ratio_abs_diff_from_overall", ascending=False).head(1).to_dict(orient="records")[0]
    else:
        largest = None
    median_count = float(month["sample_count"].median()) if len(month) else 0.0
    small_sample = month.loc[month["sample_count"] < median_count * small_ratio].to_dict(orient="records")
    near_zero_mean = float(month["boundary_risk_5bps_ratio"].mean())
    near_zero_std = float(month["boundary_risk_5bps_ratio"].std(ddof=1)) if len(month) > 1 else 0.0
    high_near_zero = month.loc[
        month["boundary_risk_5bps_ratio"] > near_zero_mean + high_std_multiplier * near_zero_std
    ].to_dict(orient="records")
    return {
        "overall_up_ratio": overall_up_ratio,
        "largest_up_ratio_diff_month": largest,
        "small_sample_month_count_threshold": median_count * small_ratio,
        "small_sample_months": small_sample,
        "high_near_zero_5bps_threshold": near_zero_mean + high_std_multiplier * near_zero_std,
        "high_near_zero_5bps_months": high_near_zero,
    }


def make_field_dictionary(config: dict[str, Any]) -> dict[str, Any]:
    definitions: dict[str, dict[str, Any]] = {
        "feature_open_time": {
            "definition": "Current feature Kline open_time in Unix milliseconds.",
            "unit": "Unix milliseconds",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "decision_time": {
            "definition": "Prediction decision time, feature_open_time + expected_interval_ms.",
            "unit": "Unix milliseconds",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "entry_minute_open_time": {
            "definition": "Entry proxy minute open_time, equal to decision_time.",
            "unit": "Unix milliseconds",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "settlement_minute_open_time": {
            "definition": "Settlement proxy minute open_time, entry_minute_open_time + 60 minutes.",
            "unit": "Unix milliseconds",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "continuity_segment_id": {
            "definition": "Stage 2A continuous one-minute Kline segment id for the feature row.",
            "unit": "segment id",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "is_prediction_time_5m": {
            "definition": "Whether decision_time falls on the configured 5-minute prediction grid.",
            "unit": "boolean",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "has_history_240m": {
            "definition": "Stage 2A flag that the feature row has the required 240-minute history inside the segment.",
            "unit": "boolean",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "has_future_61m": {
            "definition": "Stage 2A flag that the feature row has 61 future one-minute rows inside the segment.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "is_model_candidate": {
            "definition": "Stage 2A candidate flag based on prediction grid, history, future horizon, and continuity constraints.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "has_entry_price_proxy": {
            "definition": "Whether the exact entry minute Kline open proxy price is present.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "has_settlement_price_proxy": {
            "definition": "Whether the exact settlement minute Kline open proxy price is present.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "same_continuity_segment": {
            "definition": "Whether feature, entry, and settlement minutes are in the same Stage 2A continuity segment.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "exact_horizon_match": {
            "definition": "Whether entry and settlement rows exist and settlement_minute_open_time - entry_minute_open_time equals exactly 60 minutes.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "is_valid_proxy_label": {
            "definition": "Whether exact proxy prices exist, are finite positive, share the same continuity segment, and have an exact 60-minute horizon.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "is_final_model_label_candidate": {
            "definition": "is_model_candidate and is_valid_proxy_label are both true.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "entry_price_proxy": {
            "definition": "Kline open engineering proxy price for the entry minute; not a real order fill or true execution price.",
            "unit": "USDT per BTC",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "settlement_price_proxy": {
            "definition": "Kline open engineering proxy price for the settlement minute; not a real order fill or true execution price.",
            "unit": "USDT per BTC",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "future_simple_return_60m": {
            "definition": "settlement_price_proxy / entry_price_proxy - 1 for valid proxy labels.",
            "unit": "ratio",
            "may_use_future_data": True,
            "is_model_label": True,
            "allowed_as_model_input": False,
        },
        "future_log_return_60m": {
            "definition": "Natural log of settlement_price_proxy / entry_price_proxy for valid proxy labels.",
            "unit": "log ratio",
            "may_use_future_data": True,
            "is_model_label": True,
            "allowed_as_model_input": False,
        },
        "label_up_60m": {
            "definition": "Nullable binary label: 1 when settlement_price_proxy > entry_price_proxy, otherwise 0; equal prices are 0.",
            "unit": "0/1",
            "may_use_future_data": True,
            "is_model_label": True,
            "allowed_as_model_input": False,
        },
        "absolute_future_return_bps": {
            "definition": "abs(future_simple_return_60m) * 10000 for proxy boundary diagnostics.",
            "unit": "basis points",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "proxy_margin_bucket": {
            "definition": "Configured absolute_future_return_bps bucket for proxy boundary risk diagnostics.",
            "unit": "basis point bucket",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "sample_weight_uniform": {
            "definition": "Diagnostic uniform sample weight set to 1.0; Stage 3B does not train or select weights.",
            "unit": "weight",
            "may_use_future_data": False,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
        "sample_weight_margin": {
            "definition": "Diagnostic margin-based weight clipped from absolute_future_return_bps; use in later experiments only after rolling validation.",
            "unit": "weight",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        },
    }
    for threshold in [float(v) for v in config["proxy_boundary_thresholds_bps"]]:
        suffix = threshold_suffix(threshold)
        definitions[f"proxy_boundary_risk_{suffix}bps"] = {
            "definition": f"Diagnostic flag where absolute_future_return_bps < {threshold:g}; does not modify the main binary label.",
            "unit": "boolean",
            "may_use_future_data": True,
            "is_model_label": False,
            "allowed_as_model_input": False,
        }
    fields = []
    for name in OUTPUT_COLUMNS:
        dtype = "unknown"
        if name in {
            "feature_open_time",
            "decision_time",
            "entry_minute_open_time",
            "settlement_minute_open_time",
            "continuity_segment_id",
        }:
            dtype = "int64"
        elif name.startswith("is_") or name.startswith("has_") or name.startswith("same_") or name.startswith("exact_") or name.startswith("proxy_boundary"):
            dtype = "bool"
        elif name == "label_up_60m":
            dtype = "nullable int8"
        elif name == "proxy_margin_bucket":
            dtype = "string"
        elif name.startswith("sample_weight") or "price" in name or "return" in name:
            dtype = "float64"
        info = definitions[name]
        fields.append({"name": name, "dtype": dtype, **info})
    return {
        "fields": fields,
        "notes": [
            "Stage 3B uses Kline open only as an engineering proxy price, not as a real order fill or true execution price.",
            "The proxy was validated only in the 2022-09-02 to 2022-09-04 agg trades overlap window of about 66.77 hours.",
            "Stage 3A found about 98.23% overall agreement between Kline open and first-5-second VWAP labels, with differences concentrated below 5 bps absolute 60-minute return.",
            "Do not assume 2017-2026 all have exactly the same proxy error.",
            "No field in this Stage 3B output is allowed directly into a model feature matrix.",
        ],
    }


def get_process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        import ctypes
        import os

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
    result: Stage3BResult,
    input_path: Path,
    output_path: Path,
    input_rows: int,
    output_schema: list[dict[str, str]],
    output_file_size: int,
    elapsed: float,
    peak_memory: int | None,
    rss_memory: int | None,
) -> None:
    stats = result.stats
    invalid = stats["invalid_reasons"]
    boundary = stats["boundary_counts"]
    stage3a = stats["stage3a_proxy_validation_summary"]
    simple = stats["simple_return_distribution"]
    log_return = stats["log_return_distribution"]
    largest_up = result.extreme_return_samples.loc[result.extreme_return_samples["extreme_type"] == "largest_up"].head(1)
    largest_down = result.extreme_return_samples.loc[result.extreme_return_samples["extreme_type"] == "largest_down"].head(1)
    stability = stats["monthly_stability_diagnostics"]
    lines = [
        "# 阶段3B：完整历史60分钟K线open代理标签报告",
        "",
        "- 范围限制: 只读取阶段2A K线模型基础Parquet；未读取原始CSV；未读取或合并agg trades；未生成模型输入特征、切分或模型。",
        "- 标签定位: K线open是工程代理价格，不是真实下单成交价格。",
        f"- 输入: `{input_path}`",
        f"- 输出: `{output_path}`",
        f"- 输出文件大小(bytes): `{output_file_size}`",
        f"- 运行耗时(seconds): `{elapsed:.2f}`",
        f"- Python tracemalloc峰值内存(bytes): `{peak_memory}`",
        f"- 进程RSS内存(bytes): `{rss_memory}`",
        "",
        "## 总体数量",
        table(
            [
                {"指标": "输入总行数", "值": input_rows},
                {"指标": "5分钟预测时刻数量", "值": stats["prediction_time_count"]},
                {"指标": "is_model_candidate数量", "值": stats["model_candidate_count"]},
                {"指标": "有效代理标签数量", "值": stats["valid_proxy_label_count"]},
                {"指标": "最终模型标签候选数量", "值": stats["final_model_label_candidate_count"]},
            ],
            ["指标", "值"],
        ),
        "## 无效标签原因",
        table([invalid], list(invalid.keys())),
        "## 标签分布",
        table(
            [
                {"指标": "上涨标签数量", "值": stats["up_count"]},
                {"指标": "上涨标签比例", "值": stats["up_ratio"]},
                {"指标": "下跌或持平标签数量", "值": stats["down_or_equal_count"]},
                {"指标": "下跌或持平标签比例", "值": stats["down_or_equal_ratio"]},
                {"指标": "价格完全相等样本数量", "值": stats["equal_price_count"]},
            ],
            ["指标", "值"],
        ),
        "## 近零收益样本",
        table(
            [{"指标": key, "值": value} for key, value in boundary.items()],
            ["指标", "值"],
        ),
        "## Simple Return分布",
        table([simple], list(simple.keys())),
        "## Log Return分布",
        table([log_return], list(log_return.keys())),
        "## 各年份标签分布",
        table(result.distribution_by_year.to_dict(orient="records"), list(result.distribution_by_year.columns)),
        "## 各月份标签分布",
        table(result.distribution_by_month.to_dict(orient="records"), list(result.distribution_by_month.columns)),
        "## 各连续片段标签分布",
        table(result.distribution_by_segment.to_dict(orient="records"), list(result.distribution_by_segment.columns)),
        "## 绝对收益bps区间分布",
        table(result.distribution_by_margin.to_dict(orient="records"), list(result.distribution_by_margin.columns)),
        "## 最大上涨样本",
        table(largest_up.to_dict(orient="records"), list(largest_up.columns) if not largest_up.empty else []),
        "## 最大下跌样本",
        table(largest_down.to_dict(orient="records"), list(largest_down.columns) if not largest_down.empty else []),
        "## 时间连接和segment验证",
        table([stats["time_join_validation"]], list(stats["time_join_validation"].keys())),
        "## 标签分布稳定性诊断",
        table(
            [
                {"指标": "整体up_ratio", "值": stability.get("overall_up_ratio")},
                {"指标": "up_ratio差异最大月份", "值": stability.get("largest_up_ratio_diff_month")},
                {"指标": "样本数量异常少月份阈值", "值": stability.get("small_sample_month_count_threshold")},
                {"指标": "样本数量异常少月份", "值": stability.get("small_sample_months")},
                {"指标": "近零收益比例异常高阈值", "值": stability.get("high_near_zero_5bps_threshold")},
                {"指标": "近零收益比例异常高月份", "值": stability.get("high_near_zero_5bps_months")},
            ],
            ["指标", "值"],
        ),
        "## 阶段3A代理验证摘要",
        table([stage3a], list(stage3a.keys())),
        "## 标签代理限制说明",
        "- `entry_price_proxy` 和 `settlement_price_proxy` 是K线open工程代理价格，不是真实下单成交价格。",
        "- 该代理只在2022-09-02至2022-09-04约66.77小时的agg trades窗口中进行了验证。",
        "- 阶段3A显示K线open代理标签与前5秒VWAP标签总体一致率约98.23%，primary reliable样本约98.49%。",
        "- 标签差异集中在60分钟绝对收益小于5 bps的边界样本；绝对收益大于等于5 bps和10 bps样本在阶段3A窗口中一致率为100%。",
        "- 不能假设2017-2026所有年份都具有完全相同的代理误差。",
        "- 边界风险字段和样本权重只用于诊断、分层评价、稳健性分析或后续样本权重实验；本阶段不改变主标签、不删除近零收益样本、不训练模型。",
        "",
        "## 输出Schema",
        table(output_schema, ["name", "type"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage3b(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    input_path = (root / config["input_path"]).resolve()
    output_path = (root / config["output_path"]).resolve()
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
    logging.info("Stage 3B full Kline proxy label build started")
    logging.info("Input path: %s", input_path)
    try:
        require_columns(input_path, REQUIRED_INPUT_COLUMNS)
        model_base = pd.read_parquet(input_path, columns=REQUIRED_INPUT_COLUMNS)
        input_rows = int(len(model_base))
        result = build_full_labels(model_base, config)

        ensure_parent(output_path)
        result.df.to_parquet(output_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
        for path in report_paths.values():
            ensure_parent(path)
        result.distribution_by_year.to_csv(report_paths["distribution_by_year"], index=False, encoding="utf-8")
        result.distribution_by_month.to_csv(report_paths["distribution_by_month"], index=False, encoding="utf-8")
        result.distribution_by_segment.to_csv(report_paths["distribution_by_segment"], index=False, encoding="utf-8")
        result.distribution_by_margin.to_csv(report_paths["distribution_by_margin"], index=False, encoding="utf-8")
        result.invalid_label_candidates.to_csv(report_paths["invalid_label_candidates"], index=False, encoding="utf-8")
        result.extreme_return_samples.to_csv(report_paths["extreme_return_samples"], index=False, encoding="utf-8")
        report_paths["field_dictionary"].write_text(
            json.dumps(result.field_dictionary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        schema = parquet_schema(output_path)
        output_file_size = output_path.stat().st_size
        write_report(
            report_paths["main_report"],
            result,
            input_path,
            output_path,
            input_rows,
            schema,
            output_file_size,
            elapsed,
            peak,
            rss,
        )
        logging.info("Input rows: %s", input_rows)
        logging.info("Stats: %s", result.stats)
        logging.info("Output path: %s", output_path)
        logging.info("Output file size bytes: %s", output_file_size)
        logging.info("Elapsed seconds: %.2f", elapsed)
        logging.info("Python tracemalloc peak bytes: %s", peak)
        logging.info("Process RSS bytes: %s", rss)
        return {
            "stats": result.stats,
            "schema": schema,
            "output_path": str(output_path),
            "output_size_bytes": output_file_size,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": peak,
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 3B full-history 60m Kline open proxy labels.")
    parser.add_argument("--config", default="config/stage3b_build_full_labels.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage3b(config_path, root)


if __name__ == "__main__":
    main()

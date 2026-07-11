from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class Stage3AValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Stage3AResult:
    df: pd.DataFrame
    stats: dict[str, Any]
    agreement_sets: pd.DataFrame
    agreement_by_margin: pd.DataFrame
    agreement_by_hour: pd.DataFrame
    agreement_by_date: pd.DataFrame
    flip_samples: pd.DataFrame
    largest_return_differences: pd.DataFrame
    field_dictionary: dict[str, Any]


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


def require_columns(path: Path, required: list[str], role: str) -> None:
    names = pq.read_schema(path).names
    missing = [col for col in required if col not in names]
    if missing:
        raise Stage3AValidationError([f"{role} missing required columns: {missing}"])


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float | None:
    valid = a.notna() & b.notna()
    if int(valid.sum()) == 0:
        return None
    av = a.loc[valid].astype(int)
    bv = b.loc[valid].astype(int)
    po = float((av == bv).mean())
    p1a = float((av == 1).mean())
    p1b = float((bv == 1).mean())
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    if abs(1 - pe) < 1e-15:
        return None
    return (po - pe) / (1 - pe)


def matthews_corrcoef(a: pd.Series, b: pd.Series) -> float | None:
    valid = a.notna() & b.notna()
    if int(valid.sum()) == 0:
        return None
    av = a.loc[valid].astype(int)
    bv = b.loc[valid].astype(int)
    tp = int(((av == 1) & (bv == 1)).sum())
    tn = int(((av == 0) & (bv == 0)).sum())
    fp = int(((av == 0) & (bv == 1)).sum())
    fn = int(((av == 1) & (bv == 0)).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / denom


def binary_agreement_metrics(df: pd.DataFrame, mask: pd.Series, label_a: str, label_b: str) -> dict[str, Any]:
    subset = df.loc[mask].copy()
    valid = subset[label_a].notna() & subset[label_b].notna()
    subset = subset.loc[valid]
    n = int(len(subset))
    if n == 0:
        lo, hi = wilson_interval(0, 0)
        return {
            "sample_count": 0,
            "agreement_count": 0,
            "flip_count": 0,
            "agreement_rate": None,
            "agreement_rate_ci95_low": lo,
            "agreement_rate_ci95_high": hi,
            "flip_rate": None,
            "flip_rate_ci95_low": lo,
            "flip_rate_ci95_high": hi,
            "cohen_kappa": None,
            "mcc": None,
            "kline_up_rate": None,
            "reference_up_rate": None,
            "up_to_down_count": 0,
            "down_to_up_count": 0,
        }
    a = subset[label_a].astype(int)
    b = subset[label_b].astype(int)
    agree = int((a == b).sum())
    flip = n - agree
    agree_lo, agree_hi = wilson_interval(agree, n)
    flip_lo, flip_hi = wilson_interval(flip, n)
    return {
        "sample_count": n,
        "agreement_count": agree,
        "flip_count": flip,
        "agreement_rate": agree / n,
        "agreement_rate_ci95_low": agree_lo,
        "agreement_rate_ci95_high": agree_hi,
        "flip_rate": flip / n,
        "flip_rate_ci95_low": flip_lo,
        "flip_rate_ci95_high": flip_hi,
        "cohen_kappa": cohen_kappa(a, b),
        "mcc": matthews_corrcoef(a, b),
        "kline_up_rate": float((a == 1).mean()),
        "reference_up_rate": float((b == 1).mean()),
        "up_to_down_count": int(((a == 1) & (b == 0)).sum()),
        "down_to_up_count": int(((a == 0) & (b == 1)).sum()),
    }


def make_margin_bucket(values_bps: pd.Series, bins: list[float]) -> pd.Series:
    edges = [float(v) for v in bins] + [np.inf]
    labels = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        if np.isinf(right):
            labels.append(f"[{left:g}, +inf)")
        else:
            labels.append(f"[{left:g}, {right:g})")
    return pd.cut(values_bps, bins=edges, labels=labels, right=False, include_lowest=True)


def price_bps(proxy: pd.Series, reference: pd.Series, epsilon: float) -> pd.Series:
    return (proxy / np.maximum(reference.abs(), epsilon) - 1.0) * 10000


def return_stats(diff: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(diff, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    abs_clean = clean.abs()
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "abs_p90": float(abs_clean.quantile(0.90)),
        "abs_p95": float(abs_clean.quantile(0.95)),
        "abs_p99": float(abs_clean.quantile(0.99)),
        "max_abs": float(abs_clean.max()),
    }


def validate_time_inputs(model_base: pd.DataFrame, horizon_ms: int, expected_interval_ms: int) -> None:
    errors: list[str] = []
    if not (model_base["decision_time"] == model_base["open_time"] + expected_interval_ms).all():
        errors.append("model_base decision_time must equal open_time + 60000ms")
    if errors:
        raise Stage3AValidationError(errors)


def build_label_proxy_comparison(
    model_base: pd.DataFrame,
    agg_1m: pd.DataFrame,
    overlap: pd.DataFrame,
    config: dict[str, Any],
) -> Stage3AResult:
    expected_interval_ms = 60_000
    prediction_interval_ms = int(config["prediction_interval_minutes"]) * expected_interval_ms
    horizon_ms = int(config["horizon_minutes"]) * expected_interval_ms
    windows = [int(v) for v in config["vwap_windows_seconds"]]
    primary_window = int(config["primary_vwap_window_seconds"])
    epsilon = float(config["epsilon"])

    validate_time_inputs(model_base, horizon_ms, expected_interval_ms)
    if not agg_1m["open_time"].is_unique:
        raise Stage3AValidationError(["agg_1m open_time must be unique"])
    if not overlap["open_time"].is_unique:
        raise Stage3AValidationError(["overlap open_time must be unique"])

    overlap_min = int(overlap["open_time"].min())
    overlap_max = int(overlap["open_time"].max())
    base = model_base.copy(deep=False)
    for col in ["is_prediction_time_5m", "is_model_candidate", "has_future_61m"]:
        base[col] = base[col].astype("bool")
    overlap = overlap.copy(deep=False)
    for col in ["is_reliable_overlap_minute", "is_boundary_partial_minute", "has_any_id_gap"]:
        overlap[col] = overlap[col].astype("bool")
    base = base.loc[
        base["is_prediction_time_5m"]
        & base["has_future_61m"]
        & (base["decision_time"] >= overlap_min)
        & (base["decision_time"] <= overlap_max)
    ].copy()
    base.rename(columns={"open_time": "feature_open_time"}, inplace=True)
    base["entry_minute_open_time"] = base["decision_time"].astype("int64")
    base["settlement_minute_open_time"] = base["decision_time"].astype("int64") + horizon_ms

    minute_cols = [
        "open_time",
        "kline_open",
        "vwap_first_1s",
        "vwap_first_2s",
        "vwap_first_5s",
        "vwap_first_10s",
        "is_reliable_overlap_minute",
        "is_boundary_partial_minute",
        "has_any_id_gap",
        "id_gap_event_count",
        "cross_minute_id_gap_event_count",
        "maximum_internal_id_gap",
        "kline_base_volume",
    ]
    minute = overlap[minute_cols].copy()
    entry = minute.add_prefix("entry_").rename(columns={"entry_open_time": "entry_minute_open_time"})
    settlement = minute.add_prefix("settlement_").rename(columns={"settlement_open_time": "settlement_minute_open_time"})
    out = base.merge(entry, on="entry_minute_open_time", how="left", validate="many_to_one")
    out = out.merge(settlement, on="settlement_minute_open_time", how="left", validate="many_to_one")
    out["entry_exists"] = out["entry_kline_open"].notna()
    out["settlement_exists"] = out["settlement_kline_open"].notna()
    out["is_label_only_candidate"] = (
        out["is_prediction_time_5m"]
        & out["has_future_61m"]
        & out["entry_exists"]
        & out["settlement_exists"]
        & ((out["settlement_minute_open_time"] - out["entry_minute_open_time"]) == horizon_ms)
    )
    out["is_model_aligned_candidate"] = out["is_label_only_candidate"] & out["is_model_candidate"]
    # Output only short-window label candidates with both entry and settlement minutes present.
    out = out.loc[out["is_label_only_candidate"]].copy()

    out["entry_price_kline_open"] = out["entry_kline_open"]
    out["settlement_price_kline_open"] = out["settlement_kline_open"]
    valid_kline = out["is_label_only_candidate"] & out["entry_price_kline_open"].notna() & out["settlement_price_kline_open"].notna()
    out["return_kline_open"] = np.where(valid_kline, out["settlement_price_kline_open"] / out["entry_price_kline_open"] - 1.0, np.nan)
    out["log_return_kline_open"] = np.where(valid_kline, np.log(out["settlement_price_kline_open"] / out["entry_price_kline_open"]), np.nan)
    out["label_kline_open"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out.loc[valid_kline, "label_kline_open"] = (out.loc[valid_kline, "settlement_price_kline_open"] > out.loc[valid_kline, "entry_price_kline_open"]).astype("int8")

    for window in windows:
        entry_col = f"entry_vwap_first_{window}s"
        settlement_col = f"settlement_vwap_first_{window}s"
        proxy = f"vwap_{window}s"
        out[f"entry_price_{proxy}"] = out[entry_col]
        out[f"settlement_price_{proxy}"] = out[settlement_col]
        out[f"has_entry_{proxy}"] = out[f"entry_price_{proxy}"].notna()
        out[f"has_settlement_{proxy}"] = out[f"settlement_price_{proxy}"].notna()
        valid = out["is_label_only_candidate"] & out[f"has_entry_{proxy}"] & out[f"has_settlement_{proxy}"]
        out[f"is_valid_{proxy}_label"] = valid.astype("bool")
        out[f"return_{proxy}"] = np.where(valid, out[f"settlement_price_{proxy}"] / out[f"entry_price_{proxy}"] - 1.0, np.nan)
        out[f"log_return_{proxy}"] = np.where(valid, np.log(out[f"settlement_price_{proxy}"] / out[f"entry_price_{proxy}"]), np.nan)
        out[f"label_{proxy}"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
        out.loc[valid, f"label_{proxy}"] = (out.loc[valid, f"settlement_price_{proxy}"] > out.loc[valid, f"entry_price_{proxy}"]).astype("int8")

    primary = f"vwap_{primary_window}s"
    req = config["reliability_requirements"]
    primary_reliable = out["is_label_only_candidate"].astype("bool").copy()
    if req.get("require_entry_reliable_overlap", True):
        primary_reliable &= out["entry_is_reliable_overlap_minute"].fillna(False).astype("bool")
    if req.get("require_settlement_reliable_overlap", True):
        primary_reliable &= out["settlement_is_reliable_overlap_minute"].fillna(False).astype("bool")
    if req.get("exclude_entry_boundary_partial_minute", True):
        primary_reliable &= ~out["entry_is_boundary_partial_minute"].fillna(True).astype("bool")
    if req.get("exclude_settlement_boundary_partial_minute", True):
        primary_reliable &= ~out["settlement_is_boundary_partial_minute"].fillna(True).astype("bool")
    if req.get("require_entry_primary_vwap", True):
        primary_reliable &= out[f"has_entry_{primary}"]
    if req.get("require_settlement_primary_vwap", True):
        primary_reliable &= out[f"has_settlement_{primary}"]
    if req.get("require_kline_open_prices", True):
        primary_reliable &= valid_kline
    if req.get("require_exact_horizon_minutes", True):
        primary_reliable &= (out["settlement_minute_open_time"] - out["entry_minute_open_time"]) == horizon_ms
    out["is_primary_reliable_sample"] = primary_reliable.astype("bool")

    out["absolute_return_kline_open_bps"] = out["return_kline_open"].abs() * 10000
    out["absolute_return_vwap_5s_bps"] = out["return_vwap_5s"].abs() * 10000
    bins = [float(v) for v in config["return_margin_bins_bps"]]
    out["absolute_return_kline_open_bps_bucket"] = make_margin_bucket(out["absolute_return_kline_open_bps"], bins).astype("string")
    out["absolute_return_vwap_5s_bps_bucket"] = make_margin_bucket(out["absolute_return_vwap_5s_bps"], bins).astype("string")

    out["entry_has_any_id_gap"] = out["entry_has_any_id_gap"].fillna(False).astype("bool")
    out["settlement_has_any_id_gap"] = out["settlement_has_any_id_gap"].fillna(False).astype("bool")
    out["entry_and_settlement_reliable"] = (
        out["entry_is_reliable_overlap_minute"].fillna(False)
        & out["settlement_is_reliable_overlap_minute"].fillna(False)
    ).astype("bool")

    make_comparison_fields(out, "kline_open", primary, epsilon, exact_primary_names=True)
    for other in [f"vwap_{w}s" for w in windows if w != primary_window]:
        make_comparison_fields(out, "kline_open", other, epsilon)
        make_comparison_fields(out, other, primary, epsilon)

    out["decision_hour_utc"] = pd.to_datetime(out["decision_time"], unit="ms", utc=True).dt.hour.astype("int8")
    out["decision_date_utc"] = pd.to_datetime(out["decision_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    out["entry_kline_volume_quantile_bucket"] = pd.qcut(out["entry_kline_base_volume"], q=4, duplicates="drop").astype("string")

    output_columns = make_output_columns(windows)
    out = out[output_columns].copy()

    agreement_sets = make_agreement_sets(out, config)
    agreement_by_margin = make_agreement_by_margin(out, config)
    agreement_by_hour = make_group_agreement(out, "decision_hour_utc")
    agreement_by_date = make_group_agreement(out, "decision_date_utc")
    flip_samples = make_flip_samples(out)
    largest_diffs = out.loc[out["is_label_only_candidate"] & out["is_valid_vwap_5s_label"]].copy()
    largest_diffs = largest_diffs.sort_values("absolute_return_difference_kline_vs_5s", ascending=False).head(int(config["largest_difference_rows"]))
    largest_diffs = add_iso_columns(largest_diffs)

    stats = make_stats(out, agreement_sets, config)
    field_dictionary = make_field_dictionary(windows)
    return Stage3AResult(
        df=out,
        stats=stats,
        agreement_sets=agreement_sets,
        agreement_by_margin=agreement_by_margin,
        agreement_by_hour=agreement_by_hour,
        agreement_by_date=agreement_by_date,
        flip_samples=flip_samples,
        largest_return_differences=largest_diffs,
        field_dictionary=field_dictionary,
    )


def make_comparison_fields(df: pd.DataFrame, proxy: str, reference: str, epsilon: float, exact_primary_names: bool = False) -> None:
    label_proxy = f"label_{proxy}"
    label_reference = f"label_{reference}"
    valid = df[label_proxy].notna() & df[label_reference].notna()
    key = f"{proxy}_vs_{reference}"
    agree_col = f"label_agree_{key}"
    flip_col = f"label_flip_{key}"
    df[agree_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df[flip_col] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[valid, agree_col] = df.loc[valid, label_proxy].astype(int).eq(df.loc[valid, label_reference].astype(int))
    df.loc[valid, flip_col] = ~df.loc[valid, agree_col].astype(bool)
    df[f"return_difference_bps_{key}"] = (df[f"return_{proxy}"] - df[f"return_{reference}"]) * 10000
    if exact_primary_names:
        df["label_agree_kline_vs_5s"] = df[agree_col]
        df["label_flip_kline_vs_5s"] = df[flip_col]
        df["return_difference_kline_vs_5s"] = df["return_kline_open"] - df["return_vwap_5s"]
        df["absolute_return_difference_kline_vs_5s"] = df["return_difference_kline_vs_5s"].abs()
        df["return_difference_bps_kline_vs_5s"] = df["return_difference_kline_vs_5s"] * 10000
        df["entry_price_difference_bps_kline_vs_5s"] = price_bps(df["entry_price_kline_open"], df["entry_price_vwap_5s"], epsilon)
        df["settlement_price_difference_bps_kline_vs_5s"] = price_bps(df["settlement_price_kline_open"], df["settlement_price_vwap_5s"], epsilon)


def make_output_columns(windows: list[int]) -> list[str]:
    cols = [
        "feature_open_time",
        "decision_time",
        "entry_minute_open_time",
        "settlement_minute_open_time",
        "is_prediction_time_5m",
        "is_model_candidate",
        "is_label_only_candidate",
        "is_model_aligned_candidate",
        "is_primary_reliable_sample",
        "entry_exists",
        "settlement_exists",
        "entry_is_reliable_overlap_minute",
        "settlement_is_reliable_overlap_minute",
        "entry_is_boundary_partial_minute",
        "settlement_is_boundary_partial_minute",
        "entry_has_any_id_gap",
        "settlement_has_any_id_gap",
        "entry_id_gap_event_count",
        "settlement_id_gap_event_count",
        "entry_cross_minute_id_gap_event_count",
        "settlement_cross_minute_id_gap_event_count",
        "entry_kline_base_volume",
        "settlement_kline_base_volume",
        "entry_and_settlement_reliable",
        "entry_kline_volume_quantile_bucket",
        "entry_price_kline_open",
        "settlement_price_kline_open",
        "return_kline_open",
        "log_return_kline_open",
        "label_kline_open",
    ]
    for w in windows:
        proxy = f"vwap_{w}s"
        cols.extend(
            [
                f"entry_price_{proxy}",
                f"settlement_price_{proxy}",
                f"has_entry_{proxy}",
                f"has_settlement_{proxy}",
                f"is_valid_{proxy}_label",
                f"return_{proxy}",
                f"log_return_{proxy}",
                f"label_{proxy}",
            ]
        )
    cols.extend(
        [
            "label_agree_kline_vs_5s",
            "label_flip_kline_vs_5s",
            "return_difference_kline_vs_5s",
            "absolute_return_difference_kline_vs_5s",
            "entry_price_difference_bps_kline_vs_5s",
            "settlement_price_difference_bps_kline_vs_5s",
            "return_difference_bps_kline_vs_5s",
            "absolute_return_kline_open_bps",
            "absolute_return_vwap_5s_bps",
            "absolute_return_kline_open_bps_bucket",
            "absolute_return_vwap_5s_bps_bucket",
            "decision_hour_utc",
            "decision_date_utc",
        ]
    )
    for w in [1, 2, 10]:
        cols.extend(
            [
                f"label_agree_kline_open_vs_vwap_{w}s",
                f"label_flip_kline_open_vs_vwap_{w}s",
                f"return_difference_bps_kline_open_vs_vwap_{w}s",
                f"label_agree_vwap_{w}s_vs_vwap_5s",
                f"label_flip_vwap_{w}s_vs_vwap_5s",
                f"return_difference_bps_vwap_{w}s_vs_vwap_5s",
            ]
        )
    return [col for col in cols if col not in cols[: cols.index(col)]]


def make_agreement_sets(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    comparable = df["is_label_only_candidate"] & df["is_valid_vwap_5s_label"] & df["label_kline_open"].notna()
    masks = {
        "all_comparable": comparable,
        "primary_reliable": comparable & df["is_primary_reliable_sample"],
        "model_aligned": comparable & df["is_model_aligned_candidate"],
        "label_only": comparable & df["is_label_only_candidate"],
    }
    for threshold in [1, 2.5, 5, 10]:
        masks[f"abs_return_kline_ge_{threshold:g}bps"] = comparable & (df["absolute_return_kline_open_bps"] >= threshold)
    rows = []
    for name, mask in masks.items():
        metrics = binary_agreement_metrics(df, mask, "label_kline_open", "label_vwap_5s")
        metrics["sample_set"] = name
        rows.append(metrics)
    return pd.DataFrame(rows)


def make_agreement_by_margin(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for source, bucket_col in [
        ("kline_open_abs_return", "absolute_return_kline_open_bps_bucket"),
        ("vwap_5s_abs_return", "absolute_return_vwap_5s_bps_bucket"),
    ]:
        for bucket, group in df.groupby(bucket_col, dropna=False, observed=False):
            mask = group.index
            metrics = binary_agreement_metrics(df, df.index.isin(mask), "label_kline_open", "label_vwap_5s")
            subset = df.loc[mask]
            metrics.update(
                {
                    "bucket_source": source,
                    "bucket": str(bucket),
                    "return_kline_open_mean": float(subset["return_kline_open"].mean(skipna=True)),
                    "return_kline_open_median": float(subset["return_kline_open"].median(skipna=True)),
                    "return_vwap_5s_mean": float(subset["return_vwap_5s"].mean(skipna=True)),
                    "return_vwap_5s_median": float(subset["return_vwap_5s"].median(skipna=True)),
                    "entry_price_difference_bps_mean": float(subset["entry_price_difference_bps_kline_vs_5s"].mean(skipna=True)),
                    "settlement_price_difference_bps_mean": float(subset["settlement_price_difference_bps_kline_vs_5s"].mean(skipna=True)),
                }
            )
            rows.append(metrics)
    return pd.DataFrame(rows)


def make_group_agreement(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        mask = df.index.isin(group.index)
        metrics = binary_agreement_metrics(df, mask, "label_kline_open", "label_vwap_5s")
        metrics[group_col] = value
        metrics["primary_reliable_count"] = int(group["is_primary_reliable_sample"].sum())
        metrics["entry_id_gap_count"] = int(group["entry_has_any_id_gap"].sum())
        metrics["settlement_id_gap_count"] = int(group["settlement_has_any_id_gap"].sum())
        rows.append(metrics)
    return pd.DataFrame(rows)


def add_iso_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in ["decision_time", "entry_minute_open_time", "settlement_minute_open_time", "feature_open_time"]:
        if col in result.columns:
            result[f"{col}_utc"] = result[col].map(ms_to_utc_iso)
    return result


def make_flip_samples(df: pd.DataFrame) -> pd.DataFrame:
    flips = df.loc[df["label_flip_kline_vs_5s"].fillna(False)].copy()
    cols = [
        "decision_time",
        "entry_minute_open_time",
        "settlement_minute_open_time",
        "entry_price_kline_open",
        "entry_price_vwap_5s",
        "settlement_price_kline_open",
        "settlement_price_vwap_5s",
        "return_kline_open",
        "return_vwap_5s",
        "label_kline_open",
        "label_vwap_5s",
        "entry_price_difference_bps_kline_vs_5s",
        "settlement_price_difference_bps_kline_vs_5s",
        "return_difference_bps_kline_vs_5s",
        "is_primary_reliable_sample",
        "entry_is_reliable_overlap_minute",
        "settlement_is_reliable_overlap_minute",
        "entry_has_any_id_gap",
        "settlement_has_any_id_gap",
        "absolute_return_kline_open_bps_bucket",
        "absolute_return_vwap_5s_bps_bucket",
    ]
    flips = add_iso_columns(flips[cols])
    return flips


def continuous_return_summary(df: pd.DataFrame, mask: pd.Series) -> dict[str, Any]:
    subset = df.loc[mask & df["return_kline_open"].notna() & df["return_vwap_5s"].notna()].copy()
    if subset.empty:
        return {}
    diff = subset["return_kline_open"] - subset["return_vwap_5s"]
    diff_bps = diff * 10000
    pearson = None
    spearman = None
    if len(subset) >= 2:
        pearson = float(subset["return_kline_open"].corr(subset["return_vwap_5s"], method="pearson"))
        spearman = float(subset["return_kline_open"].corr(subset["return_vwap_5s"], method="spearman"))
    return {
        "sample_count": int(len(subset)),
        "pearson": pearson,
        "spearman": spearman,
        "return_diff": return_stats(diff),
        "return_diff_bps": return_stats(diff_bps),
        "entry_price_difference_bps_mean": float(subset["entry_price_difference_bps_kline_vs_5s"].mean()),
        "entry_price_difference_bps_median": float(subset["entry_price_difference_bps_kline_vs_5s"].median()),
        "settlement_price_difference_bps_mean": float(subset["settlement_price_difference_bps_kline_vs_5s"].mean()),
        "settlement_price_difference_bps_median": float(subset["settlement_price_difference_bps_kline_vs_5s"].median()),
    }


def make_stats(df: pd.DataFrame, agreement_sets: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    comparable = df["is_label_only_candidate"] & df["is_valid_vwap_5s_label"] & df["label_kline_open"].notna()
    primary = comparable & df["is_primary_reliable_sample"]
    thresholds = config["acceptance_thresholds"]
    overall = agreement_sets.loc[agreement_sets["sample_set"] == "all_comparable"].iloc[0].to_dict()
    primary_metrics = agreement_sets.loc[agreement_sets["sample_set"] == "primary_reliable"].iloc[0].to_dict()
    ge5 = agreement_sets.loc[agreement_sets["sample_set"] == "abs_return_kline_ge_5bps"].iloc[0].to_dict()
    non_boundary_mask = comparable & ~df["entry_is_boundary_partial_minute"].fillna(True) & ~df["settlement_is_boundary_partial_minute"].fillna(True)
    non_boundary_metrics = binary_agreement_metrics(df, non_boundary_mask, "label_kline_open", "label_vwap_5s")
    missing_rate_5s = 1.0 - float(df["is_valid_vwap_5s_label"].sum() / max(int(df["is_label_only_candidate"].sum()), 1))
    continuous = continuous_return_summary(df, comparable)
    mean_bias_bps = abs(float(continuous.get("return_diff_bps", {}).get("mean", np.nan)))
    diagnostics = {
        "sample_size_check_passed": int(primary_metrics["sample_count"]) >= int(thresholds["minimum_primary_reliable_samples"]),
        "overall_agreement_check_passed": float(overall["agreement_rate"]) >= float(thresholds["minimum_overall_label_agreement"]),
        "non_boundary_agreement_check_passed": float(non_boundary_metrics["agreement_rate"]) >= float(thresholds["minimum_non_boundary_agreement"]),
        "agreement_ge_5bps_check_passed": float(ge5["agreement_rate"]) >= float(thresholds["minimum_agreement_when_abs_return_ge_5bps"]),
        "vwap_coverage_check_passed": missing_rate_5s <= float(thresholds["maximum_missing_rate_for_5s_vwap"]),
        "return_bias_check_passed": mean_bias_bps <= float(thresholds["maximum_absolute_mean_return_bias_bps"]),
    }
    if all(diagnostics.values()):
        recommendation = "ACCEPT_WITH_LIMITATIONS"
    elif not diagnostics["sample_size_check_passed"] or not diagnostics["vwap_coverage_check_passed"]:
        recommendation = "INCONCLUSIVE_NEED_MORE_AGG_DATA"
    else:
        recommendation = "REJECT_PROXY"
    diagnostics["proxy_label_recommendation"] = recommendation

    return {
        "output_rows": int(len(df)),
        "label_only_candidate_count": int(df["is_label_only_candidate"].sum()),
        "model_aligned_candidate_count": int(df["is_model_aligned_candidate"].sum()),
        "primary_reliable_sample_count": int(df["is_primary_reliable_sample"].sum()),
        "vwap_label_counts": {
            f"vwap_{w}s": {
                "valid_count": int(df[f"is_valid_vwap_{w}s_label"].sum()),
                "coverage_of_label_only": float(df[f"is_valid_vwap_{w}s_label"].sum() / max(int(df["is_label_only_candidate"].sum()), 1)),
            }
            for w in config["vwap_windows_seconds"]
        },
        "kline_vs_vwap_agreement": {
            f"vwap_{w}s": binary_agreement_metrics(df, df["is_label_only_candidate"] & df[f"is_valid_vwap_{w}s_label"], "label_kline_open", f"label_vwap_{w}s")
            for w in config["vwap_windows_seconds"]
        },
        "continuous_return_summary": continuous,
        "non_boundary_agreement": non_boundary_metrics,
        "diagnostics": diagnostics,
        "missing_rate_5s": missing_rate_5s,
        "id_gap_flip_summary": {
            "flip_count": int(df["label_flip_kline_vs_5s"].fillna(False).sum()),
            "flips_with_entry_id_gap": int((df["label_flip_kline_vs_5s"].fillna(False) & df["entry_has_any_id_gap"]).sum()),
            "flips_with_settlement_id_gap": int((df["label_flip_kline_vs_5s"].fillna(False) & df["settlement_has_any_id_gap"]).sum()),
            "nonflips_with_entry_id_gap": int((~df["label_flip_kline_vs_5s"].fillna(False) & df["entry_has_any_id_gap"] & comparable).sum()),
            "nonflips_with_settlement_id_gap": int((~df["label_flip_kline_vs_5s"].fillna(False) & df["settlement_has_any_id_gap"] & comparable).sum()),
        },
    }


def make_field_dictionary(windows: list[int]) -> dict[str, Any]:
    fields = {
        "entry_price_difference_bps_kline_vs_5s": "(entry_price_kline_open / entry_price_vwap_5s - 1) * 10000",
        "settlement_price_difference_bps_kline_vs_5s": "(settlement_price_kline_open / settlement_price_vwap_5s - 1) * 10000",
        "return_difference_bps_kline_vs_5s": "(return_kline_open - return_vwap_5s) * 10000",
        "label_kline_open": "1 if settlement_price_kline_open > entry_price_kline_open else 0",
        "label_vwap_5s": "1 if settlement_price_vwap_5s > entry_price_vwap_5s else 0; missing if either VWAP is missing",
        "is_primary_reliable_sample": "Configured reliability rule requiring reliable entry/settlement overlap minutes, non-boundary minutes, valid primary VWAP, valid kline open, and exact 60-minute horizon.",
    }
    for w in windows:
        fields[f"entry_price_vwap_{w}s"] = f"entry minute vwap_first_{w}s from Stage 2B left-closed/right-open window"
        fields[f"settlement_price_vwap_{w}s"] = f"settlement minute vwap_first_{w}s from Stage 2B left-closed/right-open window"
        fields[f"return_vwap_{w}s"] = f"settlement_price_vwap_{w}s / entry_price_vwap_{w}s - 1"
        fields[f"label_vwap_{w}s"] = f"1 if settlement_price_vwap_{w}s > entry_price_vwap_{w}s else 0; missing if either side missing"
    return {"fields": fields, "notes": ["All joins use integer millisecond timestamps.", "No model features, train/test split, or model outputs are created in Stage 3A."]}


def write_reports(result: Stage3AResult, config: dict[str, Any], report_paths: dict[str, Path], output_path: Path, output_schema: list[dict[str, str]], file_size: int, elapsed: float) -> None:
    stats = result.stats
    agreement = result.agreement_sets.to_dict(orient="records")
    diag = stats["diagnostics"]
    continuous = stats["continuous_return_summary"]
    lines = [
        "# 阶段3A：短窗口60分钟方向标签代理敏感性验证报告",
        "",
        "- 范围限制: 仅使用2022-09-02至2022-09-04短重叠窗口；未构造完整2017-2026标签；未生成模型特征、切分或模型。",
        f"- 输出: `{output_path}`",
        f"- 输出文件大小(bytes): `{file_size}`",
        f"- 运行耗时(seconds): `{elapsed:.2f}`",
        "",
        "## 样本数量",
        table(
            [
                {"指标": "输出行数", "值": stats["output_rows"]},
                {"指标": "label_only候选数", "值": stats["label_only_candidate_count"]},
                {"指标": "model_aligned候选数", "值": stats["model_aligned_candidate_count"]},
                {"指标": "primary reliable样本数", "值": stats["primary_reliable_sample_count"]},
            ],
            ["指标", "值"],
        ),
        "## VWAP窗口标签覆盖",
        table(
            [{"窗口": k, **v} for k, v in stats["vwap_label_counts"].items()],
            ["窗口", "valid_count", "coverage_of_label_only"],
        ),
        "## K线open vs 前5秒VWAP标签一致性",
        table(agreement, list(result.agreement_sets.columns)),
        "## K线open与各VWAP窗口标签一致性",
        table(
            [{"窗口": k, **v} for k, v in stats["kline_vs_vwap_agreement"].items()],
            ["窗口", "sample_count", "agreement_count", "flip_count", "agreement_rate", "agreement_rate_ci95_low", "agreement_rate_ci95_high", "cohen_kappa", "mcc", "up_to_down_count", "down_to_up_count"],
        ),
        "## 连续收益比较",
        table(
            [
                {"指标": "Pearson", "值": continuous.get("pearson")},
                {"指标": "Spearman", "值": continuous.get("spearman")},
                {"指标": "收益差均值", "值": continuous.get("return_diff", {}).get("mean")},
                {"指标": "收益差中位数", "值": continuous.get("return_diff", {}).get("median")},
                {"指标": "收益差bps均值", "值": continuous.get("return_diff_bps", {}).get("mean")},
                {"指标": "收益差bps p95绝对值", "值": continuous.get("return_diff_bps", {}).get("abs_p95")},
                {"指标": "收益差bps p99绝对值", "值": continuous.get("return_diff_bps", {}).get("abs_p99")},
                {"指标": "收益差bps最大绝对值", "值": continuous.get("return_diff_bps", {}).get("max_abs")},
                {"指标": "entry价格差bps均值", "值": continuous.get("entry_price_difference_bps_mean")},
                {"指标": "settlement价格差bps均值", "值": continuous.get("settlement_price_difference_bps_mean")},
            ],
            ["指标", "值"],
        ),
        "## ID gap与标签翻转",
        table([stats["id_gap_flip_summary"]], list(stats["id_gap_flip_summary"].keys())),
        "## 诊断阈值",
        table([diag], list(diag.keys())),
        "## 结论",
        conclusion_text(diag["proxy_label_recommendation"]),
        "",
        "## 输出Schema",
        table(output_schema, ["name", "type"]),
    ]
    report_paths["main_report"].write_text("\n".join(lines) + "\n", encoding="utf-8")


def conclusion_text(recommendation: str) -> str:
    if recommendation == "ACCEPT_WITH_LIMITATIONS":
        return (
            "在当前短重叠窗口中，K线open生成的60分钟方向标签与前5秒VWAP标签高度一致，"
            "可作为完整历史标签的工程代理，但必须保留代理误差和局限性说明。该结论不能证明2017-2026所有年份都有相同代理误差。"
        )
    if recommendation == "INCONCLUSIVE_NEED_MORE_AGG_DATA":
        return "当前短窗口样本或VWAP覆盖不足，建议补充更长时间agg trades后再决定是否使用K线open代理。"
    return "当前短窗口诊断未通过代理标签阈值，不建议直接使用K线open作为逐笔VWAP标签代理。"


def run_stage3a(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    model_base_path = (root / config["model_base_input_path"]).resolve()
    agg_1m_path = (root / config["agg_1m_input_path"]).resolve()
    overlap_path = (root / config["overlap_input_path"]).resolve()
    output_path = (root / config["output_path"]).resolve()
    log_path = (root / config["log_path"]).resolve()
    report_paths = {k: (root / v).resolve() for k, v in config["report_paths"].items()}
    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )
    started = time.perf_counter()
    logging.info("Stage 3A label proxy validation started")
    require_columns(
        model_base_path,
        ["open_time", "decision_time", "is_prediction_time_5m", "is_model_candidate", "has_future_61m"],
        "model_base",
    )
    require_columns(
        agg_1m_path,
        ["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"],
        "agg_1m",
    )
    require_columns(
        overlap_path,
        [
            "open_time",
            "kline_open",
            "vwap_first_1s",
            "vwap_first_2s",
            "vwap_first_5s",
            "vwap_first_10s",
            "is_reliable_overlap_minute",
            "is_boundary_partial_minute",
            "has_any_id_gap",
            "id_gap_event_count",
            "cross_minute_id_gap_event_count",
            "maximum_internal_id_gap",
            "kline_base_volume",
        ],
        "overlap",
    )
    model_base = pd.read_parquet(model_base_path, columns=["open_time", "decision_time", "is_prediction_time_5m", "is_model_candidate", "has_future_61m"])
    agg_1m = pd.read_parquet(agg_1m_path, columns=["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"])
    overlap = pd.read_parquet(
        overlap_path,
        columns=[
            "open_time",
            "kline_open",
            "vwap_first_1s",
            "vwap_first_2s",
            "vwap_first_5s",
            "vwap_first_10s",
            "is_reliable_overlap_minute",
            "is_boundary_partial_minute",
            "has_any_id_gap",
            "id_gap_event_count",
            "cross_minute_id_gap_event_count",
            "maximum_internal_id_gap",
            "kline_base_volume",
        ],
    )
    result = build_label_proxy_comparison(model_base, agg_1m, overlap, config)
    ensure_parent(output_path)
    result.df.to_parquet(output_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
    for path in report_paths.values():
        ensure_parent(path)
    result.agreement_by_margin.to_csv(report_paths["agreement_by_margin"], index=False, encoding="utf-8")
    result.agreement_by_hour.to_csv(report_paths["agreement_by_hour"], index=False, encoding="utf-8")
    result.agreement_by_date.to_csv(report_paths["agreement_by_date"], index=False, encoding="utf-8")
    result.flip_samples.to_csv(report_paths["label_flip_samples"], index=False, encoding="utf-8")
    result.largest_return_differences.to_csv(report_paths["largest_return_differences"], index=False, encoding="utf-8")
    report_paths["field_dictionary"].write_text(json.dumps(result.field_dictionary, ensure_ascii=False, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    output_schema = parquet_schema(output_path)
    file_size = output_path.stat().st_size
    write_reports(result, config, report_paths, output_path, output_schema, file_size, elapsed)
    logging.info("Stats: %s", result.stats)
    logging.info("Output path: %s", output_path)
    logging.info("Stage 3A completed in %.2f seconds", elapsed)
    return {"stats": result.stats, "schema": output_schema, "output_size_bytes": file_size, "elapsed_seconds": elapsed}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 3A label proxy sensitivity in overlap window.")
    parser.add_argument("--config", default="config/stage3a_label_proxy_validation.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage3a(config_path, root)


if __name__ == "__main__":
    main()

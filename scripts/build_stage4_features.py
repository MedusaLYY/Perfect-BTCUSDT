from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import logging
import math
import os
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class Stage4ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


FEATURE_SET_VERSION = "kline_v1_63"

FEATURE_NAMES = [
    "log_return_1m",
    "log_return_3m",
    "log_return_5m",
    "log_return_10m",
    "log_return_15m",
    "log_return_30m",
    "log_return_60m",
    "log_return_120m",
    "momentum_acceleration_5_30",
    "momentum_acceleration_15_60",
    "efficiency_ratio_15m",
    "efficiency_ratio_60m",
    "up_candle_ratio_15m",
    "realized_volatility_5m",
    "realized_volatility_15m",
    "realized_volatility_30m",
    "realized_volatility_60m",
    "realized_volatility_120m",
    "volatility_ratio_5_60",
    "volatility_ratio_15_120",
    "normalized_atr_14",
    "normalized_atr_60",
    "semivariance_imbalance_60m",
    "ema_distance_5",
    "ema_distance_20",
    "ema_distance_60",
    "ema_distance_120",
    "ema_spread_5_20",
    "ema_spread_20_60",
    "normalized_slope_15m",
    "normalized_slope_60m",
    "range_position_30m",
    "range_position_60m",
    "range_position_120m",
    "support_distance_60m",
    "resistance_distance_60m",
    "vwap_distance_60m",
    "signed_body_ratio",
    "absolute_body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_location",
    "signed_body_mean_5m",
    "upper_wick_mean_15m",
    "lower_wick_mean_15m",
    "is_zero_volume_1m",
    "log_quote_volume_zscore_60m",
    "volume_ratio_5_60",
    "volume_ratio_15_60",
    "taker_buy_quote_ratio_1m",
    "taker_buy_quote_ratio_5m",
    "taker_buy_quote_ratio_15m",
    "taker_buy_quote_ratio_60m",
    "buy_pressure_change_5_60",
    "buy_pressure_persistence_15m",
    "buy_pressure_std_15m",
    "log_trade_count_zscore_60m",
    "log_avg_trade_quote_size_zscore_60m",
    "time_sin",
    "time_cos",
    "weekday_sin",
    "weekday_cos",
    "is_weekend",
]

INPUT_COLUMNS = [
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "continuity_segment_id",
    "gap_before_minutes",
    "feature_time",
    "decision_time",
    "segment_row_number",
    "segment_length",
    "minutes_until_segment_end",
    "has_history_240m",
    "has_future_61m",
    "is_prediction_time_5m",
    "is_model_candidate",
]

OUTPUT_METADATA_COLUMNS = [
    "feature_open_time",
    "decision_time",
    "continuity_segment_id",
    "segment_row_number",
    "has_history_240m",
    "has_future_61m",
    "is_prediction_time_5m",
    "is_model_candidate",
    "feature_set_version",
    "feature_missing_count",
    "has_nonfinite_feature",
    "is_feature_complete",
    "is_final_feature_candidate",
]

FORBIDDEN_EXACT_FIELDS = {
    "entry_price_proxy",
    "settlement_price_proxy",
    "future_simple_return_60m",
    "future_log_return_60m",
    "label_up_60m",
    "absolute_future_return_bps",
    "proxy_margin_bucket",
    "sample_weight_margin",
}

FORBIDDEN_SUBSTRINGS = ["label", "target", "entry", "settlement", "boundary_risk", "proxy_margin_bucket"]
ALLOWED_STATUS_FIELDS = {"has_future_1m", "has_future_60m", "has_future_61m"}


@dataclass
class Stage4Result:
    df: pd.DataFrame
    stats: dict[str, Any]
    feature_summary: pd.DataFrame
    missing_by_year: pd.DataFrame
    missing_by_segment: pd.DataFrame
    incomplete_feature_samples: pd.DataFrame
    feature_dictionary: dict[str, Any]
    feature_manifest: dict[str, Any]
    float32_report: dict[str, Any]
    neutral_counts: dict[str, int]
    range_checks: dict[str, Any]
    forbidden_field_scan: list[str]


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


def ms_to_utc_iso(ms: int | float | None) -> str | None:
    if ms is None or pd.isna(ms):
        return None
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()


def parquet_schema(path: Path) -> list[dict[str, str]]:
    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def require_columns(path: Path, required: list[str]) -> None:
    schema = pq.read_schema(path)
    missing = [col for col in required if col not in schema.names]
    if missing:
        raise Stage4ValidationError([f"Stage 4 input missing required columns: {missing}"])
    forbidden = scan_for_forbidden_fields(schema.names, FEATURE_NAMES, allow_required_status=True)
    if forbidden:
        raise Stage4ValidationError([f"Stage 4 input contains forbidden label/future fields: {forbidden}"])


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if config.get("feature_set_version") != FEATURE_SET_VERSION:
        errors.append(f"feature_set_version must be {FEATURE_SET_VERSION}")
    if config.get("ordered_feature_names") != FEATURE_NAMES:
        errors.append("ordered_feature_names must exactly match the Stage 4 fixed 63-feature list")
    if int(config.get("expected_feature_count", -1)) != len(FEATURE_NAMES):
        errors.append("expected_feature_count must be 63")
    if int(config["expected_interval_ms"]) != 60000:
        errors.append("expected_interval_ms must be 60000")
    if int(config["prediction_interval_minutes"]) != 5:
        errors.append("prediction_interval_minutes must be 5")
    if config.get("output_float_dtype") != "float32":
        errors.append("output_float_dtype must be float32 for Stage 4 v1")
    if len(set(FEATURE_NAMES)) != len(FEATURE_NAMES):
        errors.append("FEATURE_NAMES contains duplicates")
    if errors:
        raise Stage4ValidationError(errors)


def scan_for_forbidden_fields(
    columns: list[str] | pd.Index,
    feature_names: list[str],
    allow_required_status: bool = True,
) -> list[str]:
    feature_name_set = set(feature_names)
    forbidden: list[str] = []
    for column in columns:
        name = str(column)
        lower = name.lower()
        if name in feature_name_set:
            continue
        if allow_required_status and name in ALLOWED_STATUS_FIELDS:
            continue
        if name in FORBIDDEN_EXACT_FIELDS:
            forbidden.append(name)
            continue
        if any(token in lower for token in FORBIDDEN_SUBSTRINGS):
            forbidden.append(name)
            continue
        if "future" in lower and name not in ALLOWED_STATUS_FIELDS:
            forbidden.append(name)
    return forbidden


def validate_input_frame(df: pd.DataFrame, config: dict[str, Any]) -> None:
    errors: list[str] = []
    expected_interval_ms = int(config["expected_interval_ms"])
    prediction_interval_ms = int(config["prediction_interval_minutes"]) * expected_interval_ms

    missing = [col for col in INPUT_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Input DataFrame missing required columns: {missing}")
    forbidden = scan_for_forbidden_fields(df.columns, FEATURE_NAMES, allow_required_status=True)
    if forbidden:
        errors.append(f"Input DataFrame contains forbidden label/future fields: {forbidden}")
    if errors:
        raise Stage4ValidationError(errors)
    if df.empty:
        raise Stage4ValidationError(["Input DataFrame is empty"])

    if not df["open_time"].is_monotonic_increasing:
        errors.append("open_time must be strictly increasing")
    if not df["open_time"].is_unique:
        errors.append("open_time must be unique")
    if not (df["decision_time"].to_numpy(dtype="int64") == df["open_time"].to_numpy(dtype="int64") + expected_interval_ms).all():
        errors.append("decision_time must equal open_time + expected_interval_ms")
    if "feature_time" in df.columns and not (
        df["feature_time"].to_numpy(dtype="int64") == df["decision_time"].to_numpy(dtype="int64")
    ).all():
        errors.append("feature_time must equal decision_time")

    open_time = df["open_time"].to_numpy(dtype="int64", copy=False)
    segment = df["continuity_segment_id"].to_numpy(dtype="int64", copy=False)
    open_diff = np.diff(open_time)
    segment_diff = np.diff(segment)
    if len(open_diff):
        internal = segment_diff == 0
        if not np.all(open_diff[internal] == expected_interval_ms):
            errors.append("segment-internal open_time intervals must be exactly 60000ms")
    expected_prediction = (df["decision_time"].to_numpy(dtype="int64", copy=False) % prediction_interval_ms) == 0
    if not (df["is_prediction_time_5m"].astype(bool).to_numpy() == expected_prediction).all():
        errors.append("is_prediction_time_5m does not match decision_time modulo the 5-minute interval")

    for col in ["open", "high", "low", "close"]:
        values = df[col].to_numpy(dtype="float64", copy=False)
        if not (np.isfinite(values) & (values > 0)).all():
            errors.append(f"{col} must be finite and > 0")
    for col in ["volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]:
        values = df[col].to_numpy(dtype="float64", copy=False)
        if not (np.isfinite(values) & (values >= 0)).all():
            errors.append(f"{col} must be finite and >= 0")
    trades = df["number_of_trades"].to_numpy(dtype="float64", copy=False)
    if not (np.isfinite(trades) & (trades >= 0)).all():
        errors.append("number_of_trades must be finite and >= 0")
    abs_tol = float(config["numeric_tolerances"]["absolute"])
    rel_tol = float(config["numeric_tolerances"]["relative"])
    if bool((df["taker_buy_base_volume"] > df["volume"] + abs_tol + rel_tol * df["volume"].abs()).any()):
        errors.append("taker_buy_base_volume exceeds volume beyond tolerance")
    if bool((df["taker_buy_quote_volume"] > df["quote_volume"] + abs_tol + rel_tol * df["quote_volume"].abs()).any()):
        errors.append("taker_buy_quote_volume exceeds quote_volume beyond tolerance")
    if errors:
        raise Stage4ValidationError(errors)


def divide_with_neutral(
    numerator: pd.Series | np.ndarray,
    denominator: pd.Series | np.ndarray,
    neutral_mask: pd.Series | np.ndarray,
    neutral_value: float,
) -> pd.Series:
    num = pd.Series(numerator, copy=False)
    den = pd.Series(denominator, copy=False)
    neutral = pd.Series(neutral_mask, index=num.index).fillna(False).astype(bool)
    result = pd.Series(np.nan, index=num.index, dtype="float64")
    valid = den.notna() & num.notna() & np.isfinite(den) & np.isfinite(num) & (den != 0)
    result.loc[valid] = num.loc[valid] / den.loc[valid]
    result.loc[neutral] = neutral_value
    return result


def rolling_zscore(values: pd.Series, window: int) -> tuple[pd.Series, int]:
    rolling = values.rolling(window=window, min_periods=window)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    neutral = std.eq(0) & mean.notna()
    z = divide_with_neutral(values - mean, std, neutral, 0.0)
    return z, int(neutral.sum())


def rolling_slope_normalized(close: pd.Series, window: int) -> pd.Series:
    n = len(close)
    pos = pd.Series(np.arange(n, dtype="float64"), index=close.index)
    y = close.astype("float64")
    sum_y = y.rolling(window=window, min_periods=window).sum()
    sum_abs_xy = (pos * y).rolling(window=window, min_periods=window).sum()
    start = pos - (window - 1)
    sum_local_xy = sum_abs_xy - start * sum_y
    sum_x = window * (window - 1) / 2.0
    sum_x2 = (window - 1) * window * (2 * window - 1) / 6.0
    denominator = window * sum_x2 - sum_x * sum_x
    slope = (window * sum_local_xy - sum_x * sum_y) / denominator
    rolling_mean = sum_y / window
    return divide_with_neutral(slope, rolling_mean, (slope == 0) & (rolling_mean == 0), 0.0)


def compute_segment_features(segment: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    validate_config(config)
    segment = segment.sort_values("open_time").reset_index(drop=True)
    counts: dict[str, int] = {}
    features = pd.DataFrame(index=segment.index)

    open_price = segment["open"].astype("float64")
    high = segment["high"].astype("float64")
    low = segment["low"].astype("float64")
    close = segment["close"].astype("float64")
    volume = segment["volume"].astype("float64")
    quote_volume = segment["quote_volume"].astype("float64")
    number_of_trades = segment["number_of_trades"].astype("float64")
    taker_buy_quote_volume = segment["taker_buy_quote_volume"].astype("float64")

    for window in config["rolling_windows"]["log_return"]:
        features[f"log_return_{window}m"] = np.log(close / close.shift(window))

    features["momentum_acceleration_5_30"] = features["log_return_5m"] / 5.0 - features["log_return_30m"] / 30.0
    features["momentum_acceleration_15_60"] = features["log_return_15m"] / 15.0 - features["log_return_60m"] / 60.0

    abs_change = close.diff().abs()
    for window in config["rolling_windows"]["efficiency"]:
        path = abs_change.rolling(window=window, min_periods=window).sum()
        direct = (close - close.shift(window)).abs()
        neutral = path.eq(0) & direct.eq(0)
        counts[f"zero_efficiency_denominator_{window}m"] = int(neutral.sum())
        features[f"efficiency_ratio_{window}m"] = divide_with_neutral(direct, path, neutral, 0.0)

    features["up_candle_ratio_15m"] = (close > open_price).astype("float64").rolling(window=15, min_periods=15).mean()

    log_return_1m = features["log_return_1m"]
    for window in config["rolling_windows"]["realized_volatility"]:
        features[f"realized_volatility_{window}m"] = log_return_1m.rolling(window=window, min_periods=window).std(ddof=0)
    for short, long in [(5, 60), (15, 120)]:
        short_vol = features[f"realized_volatility_{short}m"]
        long_vol = features[f"realized_volatility_{long}m"]
        neutral = short_vol.eq(0) & long_vol.eq(0)
        counts[f"zero_volatility_ratio_denominator_{short}_{long}"] = int(neutral.sum())
        features[f"volatility_ratio_{short}_{long}"] = divide_with_neutral(short_vol, long_vol, neutral, 1.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_abs: dict[int, pd.Series] = {}
    for window in config["atr_windows"]:
        atr_abs[window] = true_range.rolling(window=window, min_periods=window).mean()
        features[f"normalized_atr_{window}"] = divide_with_neutral(atr_abs[window], close, atr_abs[window].eq(0) & close.eq(0), 0.0)

    positive_squared = log_return_1m.clip(lower=0) ** 2
    negative_squared = log_return_1m.clip(upper=0) ** 2
    upside = positive_squared.rolling(window=60, min_periods=60).sum()
    downside = negative_squared.rolling(window=60, min_periods=60).sum()
    features["semivariance_imbalance_60m"] = divide_with_neutral(upside - downside, upside + downside, upside.eq(0) & downside.eq(0), 0.0)

    ema: dict[int, pd.Series] = {}
    atr14 = atr_abs[14]
    for span in config["ema_spans"]:
        ema[span] = close.ewm(span=span, adjust=False).mean()
        distance = divide_with_neutral(close - ema[span], atr14, (close - ema[span]).eq(0) & atr14.eq(0), 0.0)
        distance.loc[segment["segment_row_number"].to_numpy(dtype="int64") < span - 1] = np.nan
        features[f"ema_distance_{span}"] = distance
    spread_5_20 = divide_with_neutral(ema[5] - ema[20], atr14, (ema[5] - ema[20]).eq(0) & atr14.eq(0), 0.0)
    spread_5_20.loc[segment["segment_row_number"].to_numpy(dtype="int64") < 19] = np.nan
    features["ema_spread_5_20"] = spread_5_20
    spread_20_60 = divide_with_neutral(ema[20] - ema[60], atr14, (ema[20] - ema[60]).eq(0) & atr14.eq(0), 0.0)
    spread_20_60.loc[segment["segment_row_number"].to_numpy(dtype="int64") < 59] = np.nan
    features["ema_spread_20_60"] = spread_20_60

    for window in config["rolling_windows"]["slope"]:
        features[f"normalized_slope_{window}m"] = rolling_slope_normalized(close, window)

    rolling_high: dict[int, pd.Series] = {}
    rolling_low: dict[int, pd.Series] = {}
    for window in config["rolling_windows"]["range_position"]:
        rolling_high[window] = high.rolling(window=window, min_periods=window).max()
        rolling_low[window] = low.rolling(window=window, min_periods=window).min()
        denominator = rolling_high[window] - rolling_low[window]
        neutral = denominator.eq(0) & rolling_high[window].notna()
        counts[f"zero_range_position_denominator_{window}m"] = int(neutral.sum())
        features[f"range_position_{window}m"] = divide_with_neutral(close - rolling_low[window], denominator, neutral, 0.5)

    features["support_distance_60m"] = divide_with_neutral(
        close - rolling_low[60],
        atr14,
        (close - rolling_low[60]).eq(0) & atr14.eq(0),
        0.0,
    )
    features["resistance_distance_60m"] = divide_with_neutral(
        rolling_high[60] - close,
        atr14,
        (rolling_high[60] - close).eq(0) & atr14.eq(0),
        0.0,
    )

    rolling_quote_60 = quote_volume.rolling(window=60, min_periods=60).sum()
    rolling_volume_60 = volume.rolling(window=60, min_periods=60).sum()
    zero_rolling_volume_60 = rolling_volume_60.eq(0) & rolling_volume_60.notna()
    counts["rolling_vwap_zero_base_volume_60m"] = int(zero_rolling_volume_60.sum())
    rolling_vwap_60 = divide_with_neutral(rolling_quote_60, rolling_volume_60, zero_rolling_volume_60, np.nan)
    vwap_numerator = close - rolling_vwap_60
    vwap_distance = divide_with_neutral(vwap_numerator, atr14, vwap_numerator.eq(0) & atr14.eq(0), 0.0)
    vwap_distance.loc[zero_rolling_volume_60] = 0.0
    features["vwap_distance_60m"] = vwap_distance

    candle_range = high - low
    body_high = pd.Series(np.maximum(open_price, close), index=segment.index)
    body_low = pd.Series(np.minimum(open_price, close), index=segment.index)
    zero_range = candle_range.eq(0)
    counts["zero_candle_range"] = int(zero_range.sum())
    features["signed_body_ratio"] = divide_with_neutral(close - open_price, candle_range, zero_range, 0.0)
    features["absolute_body_ratio"] = divide_with_neutral((close - open_price).abs(), candle_range, zero_range, 0.0)
    features["upper_wick_ratio"] = divide_with_neutral(high - body_high, candle_range, zero_range, 0.0)
    features["lower_wick_ratio"] = divide_with_neutral(body_low - low, candle_range, zero_range, 0.0)
    features["close_location"] = divide_with_neutral(close - low, candle_range, zero_range, 0.5)
    features["signed_body_mean_5m"] = features["signed_body_ratio"].rolling(window=5, min_periods=5).mean()
    features["upper_wick_mean_15m"] = features["upper_wick_ratio"].rolling(window=15, min_periods=15).mean()
    features["lower_wick_mean_15m"] = features["lower_wick_ratio"].rolling(window=15, min_periods=15).mean()

    zero_volume = volume.eq(0) | quote_volume.eq(0)
    counts["zero_volume_1m"] = int(zero_volume.sum())
    features["is_zero_volume_1m"] = zero_volume.astype("int8")
    log_quote_volume = np.log1p(quote_volume)
    features["log_quote_volume_zscore_60m"], counts["zero_log_quote_volume_zscore_std_60m"] = rolling_zscore(log_quote_volume, 60)
    rolling_quote_mean_5 = quote_volume.rolling(window=5, min_periods=5).mean()
    rolling_quote_mean_15 = quote_volume.rolling(window=15, min_periods=15).mean()
    rolling_quote_mean_60 = quote_volume.rolling(window=60, min_periods=60).mean()
    zero_quote_mean_60 = rolling_quote_mean_60.eq(0) & rolling_quote_mean_60.notna()
    counts["zero_volume_ratio_60m_denominator"] = int(zero_quote_mean_60.sum())
    features["volume_ratio_5_60"] = divide_with_neutral(rolling_quote_mean_5, rolling_quote_mean_60, zero_quote_mean_60, 0.0)
    features["volume_ratio_15_60"] = divide_with_neutral(rolling_quote_mean_15, rolling_quote_mean_60, zero_quote_mean_60, 0.0)

    features["taker_buy_quote_ratio_1m"] = divide_with_neutral(taker_buy_quote_volume, quote_volume, quote_volume.eq(0), 0.5)
    for window in [5, 15, 60]:
        buy_sum = taker_buy_quote_volume.rolling(window=window, min_periods=window).sum()
        quote_sum = quote_volume.rolling(window=window, min_periods=window).sum()
        zero_quote_sum = quote_sum.eq(0) & quote_sum.notna()
        counts[f"zero_taker_buy_quote_ratio_denominator_{window}m"] = int(zero_quote_sum.sum())
        features[f"taker_buy_quote_ratio_{window}m"] = divide_with_neutral(buy_sum, quote_sum, zero_quote_sum, 0.5)
    features["buy_pressure_change_5_60"] = features["taker_buy_quote_ratio_5m"] - features["taker_buy_quote_ratio_60m"]
    buy_dominant = (features["taker_buy_quote_ratio_1m"] > 0.5).astype("float64")
    features["buy_pressure_persistence_15m"] = buy_dominant.rolling(window=15, min_periods=15).mean()
    features["buy_pressure_std_15m"] = features["taker_buy_quote_ratio_1m"].rolling(window=15, min_periods=15).std(ddof=0)

    log_trade_count = np.log1p(number_of_trades)
    features["log_trade_count_zscore_60m"], counts["zero_log_trade_count_zscore_std_60m"] = rolling_zscore(log_trade_count, 60)
    avg_trade_quote_size = divide_with_neutral(quote_volume, number_of_trades, number_of_trades.eq(0), 0.0)
    log_avg_trade_quote_size = np.log1p(avg_trade_quote_size)
    features["log_avg_trade_quote_size_zscore_60m"], counts["zero_log_avg_trade_quote_size_zscore_std_60m"] = rolling_zscore(
        log_avg_trade_quote_size, 60
    )

    decision_dt = pd.to_datetime(segment["decision_time"], unit="ms", utc=True)
    minute_of_day = decision_dt.dt.hour.to_numpy(dtype="float64") * 60.0 + decision_dt.dt.minute.to_numpy(dtype="float64")
    weekday = decision_dt.dt.weekday.to_numpy(dtype="float64")
    features["time_sin"] = np.sin(2 * math.pi * minute_of_day / 1440.0)
    features["time_cos"] = np.cos(2 * math.pi * minute_of_day / 1440.0)
    features["weekday_sin"] = np.sin(2 * math.pi * weekday / 7.0)
    features["weekday_cos"] = np.cos(2 * math.pi * weekday / 7.0)
    features["is_weekend"] = (weekday >= 5).astype("int8")

    return features[FEATURE_NAMES], counts


def update_float32_report(report: dict[str, Any], original: pd.DataFrame, converted: pd.DataFrame) -> None:
    orig = original.to_numpy(dtype="float64", copy=False)
    conv = converted.to_numpy(dtype="float64", copy=False)
    finite = np.isfinite(orig)
    both_finite = finite & np.isfinite(conv)
    abs_err = np.abs(conv[both_finite] - orig[both_finite])
    if abs_err.size:
        report["max_abs_error"] = max(float(report.get("max_abs_error", 0.0)), float(abs_err.max()))
        rel = abs_err / np.maximum(np.abs(orig[both_finite]), 1e-12)
        report["max_relative_error"] = max(float(report.get("max_relative_error", 0.0)), float(rel.max()))
    sign_change = both_finite & (np.sign(orig) != np.sign(conv))
    report["sign_change_count"] = int(report.get("sign_change_count", 0)) + int(sign_change.sum())
    finite_to_inf = finite & np.isinf(conv)
    report["finite_to_infinite_count"] = int(report.get("finite_to_infinite_count", 0)) + int(finite_to_inf.sum())


def merge_counts(left: dict[str, int], right: dict[str, int]) -> None:
    for key, value in right.items():
        left[key] = int(left.get(key, 0)) + int(value)


def build_features(model_base: pd.DataFrame, config: dict[str, Any]) -> Stage4Result:
    validate_config(config)
    validate_input_frame(model_base, config)

    chunks: list[pd.DataFrame] = []
    neutral_counts: dict[str, int] = {}
    float32_report: dict[str, Any] = {
        "max_abs_error": 0.0,
        "max_relative_error": 0.0,
        "sign_change_count": 0,
        "finite_to_infinite_count": 0,
    }

    for _, segment in model_base.groupby("continuity_segment_id", sort=False, observed=True):
        segment = segment.sort_values("open_time").reset_index(drop=True)
        segment_features, counts = compute_segment_features(segment, config)
        merge_counts(neutral_counts, counts)
        mask = segment["is_prediction_time_5m"].astype(bool).to_numpy()
        if not bool(mask.any()):
            continue
        meta = pd.DataFrame(
            {
                "feature_open_time": segment.loc[mask, "open_time"].to_numpy(dtype="int64"),
                "decision_time": segment.loc[mask, "decision_time"].to_numpy(dtype="int64"),
                "continuity_segment_id": segment.loc[mask, "continuity_segment_id"].to_numpy(dtype="int64"),
                "segment_row_number": segment.loc[mask, "segment_row_number"].to_numpy(dtype="int64"),
                "has_history_240m": segment.loc[mask, "has_history_240m"].astype(bool).to_numpy(),
                "has_future_61m": segment.loc[mask, "has_future_61m"].astype(bool).to_numpy(),
                "is_prediction_time_5m": segment.loc[mask, "is_prediction_time_5m"].astype(bool).to_numpy(),
                "is_model_candidate": segment.loc[mask, "is_model_candidate"].astype(bool).to_numpy(),
                "feature_set_version": FEATURE_SET_VERSION,
            }
        )
        selected_features64 = segment_features.loc[mask, FEATURE_NAMES].reset_index(drop=True)
        selected_features32 = selected_features64.astype("float32")
        update_float32_report(float32_report, selected_features64, selected_features32)
        values = selected_features32.to_numpy(dtype="float64", copy=False)
        meta["feature_missing_count"] = selected_features32.isna().sum(axis=1).astype("int16")
        meta["has_nonfinite_feature"] = np.isinf(values).any(axis=1)
        meta["is_feature_complete"] = np.isfinite(values).all(axis=1)
        meta["is_final_feature_candidate"] = meta["is_model_candidate"].astype(bool) & meta["is_feature_complete"].astype(bool)
        chunks.append(pd.concat([meta, selected_features32], axis=1))

    if chunks:
        output = pd.concat(chunks, ignore_index=True)
    else:
        output = pd.DataFrame(columns=[*OUTPUT_METADATA_COLUMNS, *FEATURE_NAMES])
    output = output.sort_values("decision_time", kind="mergesort").reset_index(drop=True)

    validate_output(output, config)
    feature_summary = make_feature_summary(output)
    missing_by_year = make_missing_by_year(output)
    missing_by_segment = make_missing_by_segment(output)
    incomplete = make_incomplete_samples(output)
    range_checks = make_range_checks(output, config)
    forbidden_scan = scan_for_forbidden_fields(output.columns, FEATURE_NAMES)
    stats = make_stats(output, model_base, neutral_counts, float32_report, range_checks, forbidden_scan)
    return Stage4Result(
        df=output[[*OUTPUT_METADATA_COLUMNS, *FEATURE_NAMES]],
        stats=stats,
        feature_summary=feature_summary,
        missing_by_year=missing_by_year,
        missing_by_segment=missing_by_segment,
        incomplete_feature_samples=incomplete,
        feature_dictionary=make_feature_dictionary(config),
        feature_manifest=make_feature_manifest(config, "config/stage4_build_features.json", "scripts/build_stage4_features.py", str(config.get("input_path", ""))),
        float32_report=float32_report,
        neutral_counts=neutral_counts,
        range_checks=range_checks,
        forbidden_field_scan=forbidden_scan,
    )


def validate_output(output: pd.DataFrame, config: dict[str, Any]) -> None:
    errors: list[str] = []
    if len([col for col in output.columns if col in FEATURE_NAMES]) != len(FEATURE_NAMES):
        errors.append("Output must contain exactly 63 model feature columns")
    if list(output[FEATURE_NAMES].columns) != FEATURE_NAMES:
        errors.append("Output feature order does not match manifest order")
    if output["decision_time"].duplicated().any():
        errors.append("decision_time must be unique")
    if not output["decision_time"].is_monotonic_increasing:
        errors.append("decision_time must be strictly increasing")
    if not output["is_prediction_time_5m"].astype(bool).all():
        errors.append("Output contains non-prediction rows")
    expected_missing = output[FEATURE_NAMES].isna().sum(axis=1).astype("int16")
    if not (expected_missing.to_numpy() == output["feature_missing_count"].to_numpy()).all():
        errors.append("feature_missing_count logic mismatch")
    values = output[FEATURE_NAMES].to_numpy(dtype="float64", copy=False)
    if not (np.isinf(values).any(axis=1) == output["has_nonfinite_feature"].to_numpy(dtype=bool)).all():
        errors.append("has_nonfinite_feature logic mismatch")
    if not (np.isfinite(values).all(axis=1) == output["is_feature_complete"].to_numpy(dtype=bool)).all():
        errors.append("is_feature_complete logic mismatch")
    expected_final = output["is_model_candidate"].astype(bool) & output["is_feature_complete"].astype(bool)
    if not (expected_final.to_numpy() == output["is_final_feature_candidate"].to_numpy(dtype=bool)).all():
        errors.append("is_final_feature_candidate logic mismatch")
    forbidden = scan_for_forbidden_fields(output.columns, FEATURE_NAMES)
    if forbidden:
        errors.append(f"Output contains forbidden label/future fields: {forbidden}")
    if errors:
        raise Stage4ValidationError(errors)


def make_feature_summary(output: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in FEATURE_NAMES:
        series = pd.to_numeric(output[feature], errors="coerce")
        finite = np.isfinite(series.to_numpy(dtype="float64", copy=False))
        clean = series.loc[finite]
        row: dict[str, Any] = {
            "feature": feature,
            "non_missing_count": int(series.notna().sum()),
            "missing_rate": float(series.isna().mean()) if len(series) else None,
            "finite_rate": float(finite.mean()) if len(series) else None,
        }
        if clean.empty:
            row.update({"mean": None, "std": None, "min": None, "p01": None, "p05": None, "median": None, "p95": None, "p99": None, "max": None})
        else:
            row.update(
                {
                    "mean": float(clean.mean()),
                    "std": float(clean.std(ddof=1)) if len(clean) > 1 else 0.0,
                    "min": float(clean.min()),
                    "p01": float(clean.quantile(0.01)),
                    "p05": float(clean.quantile(0.05)),
                    "median": float(clean.quantile(0.50)),
                    "p95": float(clean.quantile(0.95)),
                    "p99": float(clean.quantile(0.99)),
                    "max": float(clean.max()),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def make_missing_by_year(output: pd.DataFrame) -> pd.DataFrame:
    if output.empty:
        return pd.DataFrame()
    year = pd.to_datetime(output["decision_time"], unit="ms", utc=True).dt.year.astype("int16")
    tmp = output.assign(decision_year_utc=year)
    rows = []
    for value, group in tmp.groupby("decision_year_utc", sort=True):
        rows.append(
            {
                "decision_year_utc": int(value),
                "sample_count": int(len(group)),
                "feature_complete_count": int(group["is_feature_complete"].sum()),
                "feature_complete_ratio": float(group["is_feature_complete"].mean()),
                "mean_feature_missing_count": float(group["feature_missing_count"].mean()),
                "final_feature_candidate_count": int(group["is_final_feature_candidate"].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_missing_by_segment(output: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for value, group in output.groupby("continuity_segment_id", sort=True):
        rows.append(
            {
                "continuity_segment_id": int(value),
                "sample_count": int(len(group)),
                "feature_complete_count": int(group["is_feature_complete"].sum()),
                "feature_complete_ratio": float(group["is_feature_complete"].mean()) if len(group) else None,
                "mean_feature_missing_count": float(group["feature_missing_count"].mean()) if len(group) else None,
                "final_feature_candidate_count": int(group["is_final_feature_candidate"].sum()),
            }
        )
    return pd.DataFrame(rows)


def make_incomplete_samples(output: pd.DataFrame, limit: int = 200) -> pd.DataFrame:
    incomplete = output.loc[~output["is_feature_complete"]].head(limit).copy()
    if incomplete.empty:
        return pd.DataFrame(columns=["feature_open_time", "decision_time", "feature_missing_count"])
    cols = [
        "feature_open_time",
        "decision_time",
        "continuity_segment_id",
        "segment_row_number",
        "has_history_240m",
        "has_future_61m",
        "is_model_candidate",
        "feature_missing_count",
        "is_feature_complete",
        "is_final_feature_candidate",
    ]
    result = incomplete[cols].copy()
    result["feature_open_time_utc"] = result["feature_open_time"].map(ms_to_utc_iso)
    result["decision_time_utc"] = result["decision_time"].map(ms_to_utc_iso)
    return result


def make_range_checks(output: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    tol = float(config["numeric_tolerances"]["range_tolerance"])

    def count_outside(columns: list[str], low: float, high: float) -> int:
        count = 0
        for column in columns:
            series = pd.to_numeric(output[column], errors="coerce")
            count += int(((series < low - tol) | (series > high + tol)).sum())
        return count

    return {
        "efficiency_ratio_outside_0_1_count": count_outside(["efficiency_ratio_15m", "efficiency_ratio_60m"], 0.0, 1.0),
        "semivariance_imbalance_outside_minus1_1_count": count_outside(["semivariance_imbalance_60m"], -1.0, 1.0),
        "range_position_outside_0_1_count": count_outside(["range_position_30m", "range_position_60m", "range_position_120m"], 0.0, 1.0),
        "taker_buy_ratio_outside_0_1_count": count_outside(
            [
                "taker_buy_quote_ratio_1m",
                "taker_buy_quote_ratio_5m",
                "taker_buy_quote_ratio_15m",
                "taker_buy_quote_ratio_60m",
            ],
            0.0,
            1.0,
        ),
        "time_feature_outside_minus1_1_count": count_outside(["time_sin", "time_cos", "weekday_sin", "weekday_cos"], -1.0, 1.0),
    }


def make_stats(
    output: pd.DataFrame,
    input_df: pd.DataFrame,
    neutral_counts: dict[str, int],
    float32_report: dict[str, Any],
    range_checks: dict[str, Any],
    forbidden_scan: list[str],
) -> dict[str, Any]:
    prediction_count = int(input_df["is_prediction_time_5m"].sum())
    model_candidate_count = int(output["is_model_candidate"].sum()) if not output.empty else 0
    feature_complete_count = int(output["is_feature_complete"].sum()) if not output.empty else 0
    final_count = int(output["is_final_feature_candidate"].sum()) if not output.empty else 0
    history_incomplete_count = int((output["has_history_240m"].astype(bool) & ~output["is_feature_complete"].astype(bool)).sum()) if not output.empty else 0
    history_shortfall_count = int((~output["has_history_240m"].astype(bool) & ~output["is_feature_complete"].astype(bool)).sum()) if not output.empty else 0
    return {
        "input_rows": int(len(input_df)),
        "prediction_time_count": prediction_count,
        "output_rows": int(len(output)),
        "model_candidate_count": model_candidate_count,
        "feature_complete_count": feature_complete_count,
        "final_feature_candidate_count": final_count,
        "history_shortfall_incomplete_count": history_shortfall_count,
        "has_history_but_incomplete_count": history_incomplete_count,
        "feature_count": len(FEATURE_NAMES),
        "feature_groups": {
            "momentum": 13,
            "volatility": 10,
            "trend_position": 14,
            "candle_structure": 8,
            "volume_taker": 13,
            "time": 5,
        },
        "neutral_counts": neutral_counts,
        "float32_report": float32_report,
        "range_checks": range_checks,
        "forbidden_field_scan": forbidden_scan,
        "manifest_feature_order_valid": True,
    }


def feature_metadata(name: str) -> dict[str, Any]:
    groups = {
        "momentum": FEATURE_NAMES[0:13],
        "volatility": FEATURE_NAMES[13:23],
        "trend_position": FEATURE_NAMES[23:37],
        "candle_structure": FEATURE_NAMES[37:45],
        "volume_taker": FEATURE_NAMES[45:58],
        "time": FEATURE_NAMES[58:63],
    }
    group = next(k for k, names in groups.items() if name in names)
    source_columns = ["open", "high", "low", "close", "volume", "quote_volume", "number_of_trades", "taker_buy_quote_volume"]
    lookback = 1
    expected_range = "unbounded"
    unit = "ratio"
    missing_rule = "NaN until the segment-local required lookback is available."
    zero_rule = "Configured feature-specific neutral value; never global fillna(0)."
    formula = name
    description = name
    if name.startswith("log_return_"):
        lookback = int(name.removeprefix("log_return_").removesuffix("m"))
        source_columns = ["close"]
        formula = f"log(close_t / close_(t-{lookback}))"
        description = f"Segment-local historical {lookback}-minute close-to-close log return."
    elif name.startswith("realized_volatility_"):
        lookback = int(name.removeprefix("realized_volatility_").removesuffix("m"))
        source_columns = ["close"]
        formula = f"rolling std(log_return_1m, {lookback}, ddof=0)"
        description = f"Segment-local one-minute realized volatility over {lookback} minutes."
        expected_range = "[0, +inf)"
    elif "ratio" in name or "position" in name or name in {"close_location", "buy_pressure_persistence_15m"}:
        expected_range = "[0, 1] or documented neutral value"
    elif "sin" in name or "cos" in name or "semivariance" in name:
        expected_range = "[-1, 1]"
    elif name.startswith("normalized_atr"):
        source_columns = ["high", "low", "close"]
        lookback = int(name.removeprefix("normalized_atr_"))
        formula = f"rolling mean(true_range, {lookback}) / close"
        expected_range = "[0, +inf)"
    elif name.startswith("ema_"):
        source_columns = ["close", "high", "low"]
        lookback = 120 if "120" in name else 60 if "60" in name else 20 if "20" in name else 5
        formula = "EMA distance or spread normalized by segment-local ATR_14_abs"
    elif name.startswith("normalized_slope"):
        lookback = int(name.removeprefix("normalized_slope_").removesuffix("m"))
        source_columns = ["close"]
        formula = f"OLS slope over last {lookback} closes / rolling mean(close, {lookback})"
    elif name.startswith("taker_buy"):
        source_columns = ["taker_buy_quote_volume", "quote_volume"]
        lookback = 1 if name.endswith("1m") else int(name.removeprefix("taker_buy_quote_ratio_").removesuffix("m"))
        formula = "sum(taker_buy_quote_volume, window) / sum(quote_volume, window)"
        expected_range = "[0, 1]"
    elif name in {"time_sin", "time_cos", "weekday_sin", "weekday_cos", "is_weekend"}:
        source_columns = ["decision_time"]
        lookback = 0
        formula = "UTC decision_time cyclic encoding"
        unit = "cyclic numeric"
        missing_rule = "Always present when decision_time is valid."
        zero_rule = "Not applicable."
        expected_range = "[-1, 1]" if name != "is_weekend" else "{0, 1}"
    return {
        "name": name,
        "group": group,
        "description": description,
        "formula": formula,
        "source_columns": source_columns,
        "lookback_minutes": lookback,
        "output_dtype": "float32",
        "unit": unit,
        "expected_range": expected_range,
        "uses_current_closed_bar": True,
        "uses_future_data": False,
        "allowed_as_model_input": True,
        "segment_aware": True,
        "missing_value_rule": missing_rule,
        "zero_denominator_rule": zero_rule,
    }


def make_feature_dictionary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_set_version": config["feature_set_version"],
        "features": [feature_metadata(name) for name in FEATURE_NAMES],
        "safe_division_policy": {
            "general": "Normal non-zero denominators divide normally; denominator-zero cases use feature-specific neutral values only where economically defined; infinities are not allowed.",
            "no_global_fillna_zero": True,
        },
    }


def make_feature_manifest(config: dict[str, Any], config_path: str, script_path: str, input_path: str) -> dict[str, Any]:
    dictionary = make_feature_dictionary(config)
    definition_payload = {
        "feature_set_version": config["feature_set_version"],
        "ordered_feature_names": FEATURE_NAMES,
        "feature_dictionary": dictionary["features"],
    }
    definition_hash = hashlib.sha256(json.dumps(definition_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {
        "feature_set_version": config["feature_set_version"],
        "ordered_feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "config_path": config_path,
        "script_path": script_path,
        "input_path": input_path,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "feature_definition_hash": definition_hash,
    }


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
    result: Stage4Result,
    input_path: Path,
    output_path: Path,
    output_schema: list[dict[str, str]],
    file_size: int,
    elapsed: float,
    peak_memory: int | None,
    rss: int | None,
) -> None:
    stats = result.stats
    lines = [
        "# 阶段4：完整1分钟K线历史特征构造报告",
        "",
        "- 范围限制: 只读取阶段2A K线模型基础Parquet；未读取标签文件；未合并标签；未切分训练集；未做标准化、异常值裁剪、特征选择或模型训练。",
        f"- 输入: `{input_path}`",
        f"- 输出: `{output_path}`",
        f"- 输出文件大小(bytes): `{file_size}`",
        f"- 运行耗时(seconds): `{elapsed:.2f}`",
        f"- Python tracemalloc峰值内存(bytes): `{peak_memory}`",
        f"- 进程RSS内存(bytes): `{rss}`",
        "",
        "## 总体数量",
        table(
            [
                {"指标": "输入行数", "值": stats["input_rows"]},
                {"指标": "5分钟预测时刻数量", "值": stats["prediction_time_count"]},
                {"指标": "is_model_candidate数量", "值": stats["model_candidate_count"]},
                {"指标": "is_feature_complete数量", "值": stats["feature_complete_count"]},
                {"指标": "is_final_feature_candidate数量", "值": stats["final_feature_candidate_count"]},
                {"指标": "历史不足导致特征不完整数量", "值": stats["history_shortfall_incomplete_count"]},
                {"指标": "有240m历史但仍不完整数量", "值": stats["has_history_but_incomplete_count"]},
            ],
            ["指标", "值"],
        ),
        "## 特征组数量",
        table([{"group": key, "count": value} for key, value in stats["feature_groups"].items()], ["group", "count"]),
        "## 特殊中性规则触发数量",
        table([{"rule": key, "count": value} for key, value in result.neutral_counts.items()], ["rule", "count"]),
        "## 越界检查",
        table([result.range_checks], list(result.range_checks.keys())),
        "## float32转换误差",
        table([result.float32_report], list(result.float32_report.keys())),
        "## manifest和禁用字段扫描",
        table(
            [
                {"检查": "feature_count", "结果": stats["feature_count"]},
                {"检查": "manifest_feature_order_valid", "结果": stats["manifest_feature_order_valid"]},
                {"检查": "forbidden_field_scan", "结果": result.forbidden_field_scan},
            ],
            ["检查", "结果"],
        ),
        "## 特征摘要",
        table(result.feature_summary.to_dict(orient="records"), list(result.feature_summary.columns)),
        "## 各年份特征完整率",
        table(result.missing_by_year.to_dict(orient="records"), list(result.missing_by_year.columns)),
        "## 各segment特征完整率",
        table(result.missing_by_segment.to_dict(orient="records"), list(result.missing_by_segment.columns)),
        "## 当前尚未进行的数据处理",
        "- 标准化",
        "- 异常值裁剪",
        "- 特征选择",
        "- 标签合并",
        "- 数据切分",
        "- 模型训练",
        "",
        "## 输出Schema",
        table(output_schema, ["name", "type"]),
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage4(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    input_path = (root / config["input_path"]).resolve()
    output_path = (root / config["output_path"]).resolve()
    log_path = (root / config["log_path"]).resolve()
    report_paths = {name: (root / value).resolve() for name, value in config["report_paths"].items()}

    if "label" in str(input_path).lower() or "stage3a" in str(input_path).lower() or "stage3b" in str(input_path).lower():
        raise Stage4ValidationError([f"Stage 4 must not read label files: {input_path}"])

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
        logging.info("Stage 4 Kline feature build started")
        logging.info("Input path: %s", input_path)
        require_columns(input_path, INPUT_COLUMNS)
        model_base = pd.read_parquet(input_path, columns=INPUT_COLUMNS)
        result = build_features(model_base, config)
        result.feature_manifest["config_path"] = str(config_path)
        result.feature_manifest["input_path"] = str(input_path)
        ensure_parent(output_path)
        result.df.to_parquet(output_path, index=False, engine="pyarrow", compression=config["parquet_compression"])
        for path in report_paths.values():
            ensure_parent(path)
        result.feature_summary.to_csv(report_paths["feature_summary"], index=False, encoding="utf-8")
        result.missing_by_year.to_csv(report_paths["missing_by_year"], index=False, encoding="utf-8")
        result.missing_by_segment.to_csv(report_paths["missing_by_segment"], index=False, encoding="utf-8")
        result.incomplete_feature_samples.to_csv(report_paths["incomplete_feature_samples"], index=False, encoding="utf-8")
        report_paths["feature_dictionary"].write_text(json.dumps(result.feature_dictionary, ensure_ascii=False, indent=2), encoding="utf-8")
        report_paths["feature_manifest"].write_text(json.dumps(result.feature_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        rss = get_process_rss_bytes()
        schema = parquet_schema(output_path)
        file_size = output_path.stat().st_size
        write_report(report_paths["main_report"], result, input_path, output_path, schema, file_size, elapsed, peak, rss)
        logging.info("Stats: %s", result.stats)
        logging.info("Output path: %s", output_path)
        logging.info("Output file size bytes: %s", file_size)
        logging.info("Elapsed seconds: %.2f", elapsed)
        logging.info("Python tracemalloc peak bytes: %s", peak)
        logging.info("Process RSS bytes: %s", rss)
        return {
            "stats": result.stats,
            "schema": schema,
            "output_path": str(output_path),
            "output_size_bytes": file_size,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": peak,
            "process_rss_bytes": rss,
        }
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 4 Kline-only historical feature set v1.")
    parser.add_argument("--config", default="config/stage4_build_features.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage4(config_path, root)


if __name__ == "__main__":
    main()

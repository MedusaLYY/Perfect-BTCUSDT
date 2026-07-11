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


class Stage2BValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Stage2BResult:
    agg_1m: pd.DataFrame
    overlap: pd.DataFrame
    stats: dict[str, Any]
    reports: dict[str, pd.DataFrame]


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


def get_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def required_schema_columns(path: Path, required: list[str], role: str) -> list[str]:
    schema = pq.read_schema(path)
    columns = schema.names
    missing = [col for col in required if col not in columns]
    if missing:
        raise Stage2BValidationError([f"{role} input missing required columns: {missing}"])
    return required


def validate_agg_input(df: pd.DataFrame, abs_tol: float, rel_tol: float) -> None:
    errors: list[str] = []
    trade_time = df["trade_time"].to_numpy(dtype=np.int64, copy=False)
    agg_id = df["agg_trade_id"].to_numpy(dtype=np.int64, copy=False)
    if len(df) > 1:
        sorted_ok = np.all((trade_time[1:] > trade_time[:-1]) | ((trade_time[1:] == trade_time[:-1]) & (agg_id[1:] >= agg_id[:-1])))
        if not sorted_ok:
            errors.append("agg trades must be sorted by trade_time, agg_trade_id")
    if not df["agg_trade_id"].is_unique:
        errors.append("agg_trade_id must be unique")
    for col in ("price", "quantity", "quote_quantity"):
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype="float64", copy=False)
        if not np.all(np.isfinite(values) & (values > 0)):
            errors.append(f"{col} must be finite and > 0")
    buy = df["is_active_buy"].to_numpy(dtype=bool, copy=False)
    sell = df["is_active_sell"].to_numpy(dtype=bool, copy=False)
    if not np.all(buy ^ sell):
        errors.append("is_active_buy and is_active_sell must be mutually exclusive and exactly one must be True")
    expected_quote = df["price"].to_numpy(dtype="float64", copy=False) * df["quantity"].to_numpy(dtype="float64", copy=False)
    actual_quote = df["quote_quantity"].to_numpy(dtype="float64", copy=False)
    tolerance = abs_tol + rel_tol * np.abs(expected_quote)
    if not np.all(np.abs(actual_quote - expected_quote) <= tolerance):
        errors.append("quote_quantity must match price * quantity within tolerance")
    if errors:
        raise Stage2BValidationError(errors)


def validate_kline_input(df: pd.DataFrame) -> None:
    errors: list[str] = []
    open_time = df["open_time"].to_numpy(dtype=np.int64, copy=False)
    if len(df) > 1 and not np.all(np.diff(open_time) > 0):
        errors.append("Kline open_time must be strictly increasing")
    if not df["open_time"].is_unique:
        errors.append("Kline open_time must be unique")
    if errors:
        raise Stage2BValidationError(errors)


def assign_trade_minute(trade_time: pd.Series, expected_interval_ms: int) -> pd.Series:
    values = trade_time.to_numpy(dtype=np.int64, copy=False)
    return pd.Series((values // expected_interval_ms) * expected_interval_ms, index=trade_time.index, dtype="int64")


def add_window_vwap(
    agg: pd.DataFrame,
    trades: pd.DataFrame,
    mask: pd.Series,
    value_prefix: str,
) -> pd.DataFrame:
    window = trades.loc[mask, ["trade_minute", "quantity", "quote_quantity"]]
    count_col = f"{value_prefix}_trade_count"
    vwap_col = f"vwap_{value_prefix}"
    if window.empty:
        agg[count_col] = 0
        agg[vwap_col] = np.nan
        return agg
    grouped = window.groupby("trade_minute", sort=True).agg(
        window_base_volume=("quantity", "sum"),
        window_quote_volume=("quote_quantity", "sum"),
        window_trade_count=("quantity", "size"),
    )
    grouped[vwap_col] = grouped["window_quote_volume"] / grouped["window_base_volume"]
    agg = agg.join(grouped[[vwap_col, "window_trade_count"]].rename(columns={"window_trade_count": count_col}), on="open_time")
    agg[count_col] = agg[count_col].fillna(0).astype("int64")
    return agg


def aggregate_agg_trades_1m(agg: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    expected_interval_ms = int(config["expected_interval_ms"])
    trades = agg.copy(deep=False)
    trades["trade_minute"] = assign_trade_minute(trades["trade_time"], expected_interval_ms)
    trades["minute_offset_ms"] = (trades["trade_time"] - trades["trade_minute"]).astype("int64")
    trades["active_buy_base"] = np.where(trades["is_active_buy"], trades["quantity"], 0.0)
    trades["active_sell_base"] = np.where(trades["is_active_sell"], trades["quantity"], 0.0)
    trades["active_buy_quote"] = np.where(trades["is_active_buy"], trades["quote_quantity"], 0.0)
    trades["active_sell_quote"] = np.where(trades["is_active_sell"], trades["quote_quantity"], 0.0)
    trades["price_x_quantity"] = trades["price"] * trades["quantity"]
    trades["price_sq_x_quantity"] = trades["price"] * trades["price"] * trades["quantity"]

    grouped = trades.groupby("trade_minute", sort=True, observed=True)
    agg_1m = grouped.agg(
        first_trade_time=("trade_time", "first"),
        last_trade_time=("trade_time", "last"),
        agg_trade_count=("agg_trade_id", "size"),
        first_agg_trade_id=("agg_trade_id", "first"),
        last_agg_trade_id=("agg_trade_id", "last"),
        base_volume=("quantity", "sum"),
        quote_volume=("quote_quantity", "sum"),
        active_buy_base_volume=("active_buy_base", "sum"),
        active_sell_base_volume=("active_sell_base", "sum"),
        active_buy_quote_volume=("active_buy_quote", "sum"),
        active_sell_quote_volume=("active_sell_quote", "sum"),
        first_trade_price=("price", "first"),
        last_trade_price=("price", "last"),
        trade_high=("price", "max"),
        trade_low=("price", "min"),
        sum_price_x_quantity=("price_x_quantity", "sum"),
        sum_price_sq_x_quantity=("price_sq_x_quantity", "sum"),
        mean_trade_base_size=("quantity", "mean"),
        max_trade_base_size=("quantity", "max"),
        mean_trade_quote_size=("quote_quantity", "mean"),
        max_trade_quote_size=("quote_quantity", "max"),
    ).reset_index().rename(columns={"trade_minute": "open_time"})

    quantiles = grouped[["quantity", "quote_quantity"]].quantile([0.5, 0.9, 0.99]).unstack()
    quantiles.columns = [
        "median_trade_base_size",
        "p90_trade_base_size",
        "p99_trade_base_size",
        "median_trade_quote_size",
        "p90_trade_quote_size",
        "p99_trade_quote_size",
    ]
    agg_1m = agg_1m.merge(quantiles.reset_index().rename(columns={"trade_minute": "open_time"}), on="open_time", how="left", validate="one_to_one")

    buy = trades.loc[trades["is_active_buy"]].groupby("trade_minute", sort=True).agg(
        buy_quote=("quote_quantity", "sum"),
        buy_base=("quantity", "sum"),
    )
    sell = trades.loc[trades["is_active_sell"]].groupby("trade_minute", sort=True).agg(
        sell_quote=("quote_quantity", "sum"),
        sell_base=("quantity", "sum"),
    )
    buy["active_buy_vwap"] = buy["buy_quote"] / buy["buy_base"]
    sell["active_sell_vwap"] = sell["sell_quote"] / sell["sell_base"]
    agg_1m = agg_1m.join(buy[["active_buy_vwap"]], on="open_time")
    agg_1m = agg_1m.join(sell[["active_sell_vwap"]], on="open_time")

    agg_1m["minute_first_offset_ms"] = agg_1m["first_trade_time"] - agg_1m["open_time"]
    agg_1m["minute_last_offset_ms"] = agg_1m["last_trade_time"] - agg_1m["open_time"]
    agg_1m["covered_duration_ms"] = agg_1m["last_trade_time"] - agg_1m["first_trade_time"]
    agg_1m["trade_vwap"] = agg_1m["quote_volume"] / agg_1m["base_volume"]
    variance = agg_1m["sum_price_sq_x_quantity"] / agg_1m["base_volume"] - agg_1m["trade_vwap"] ** 2
    agg_1m["trade_price_std_volume_weighted"] = np.sqrt(np.maximum(variance, 0.0))
    agg_1m["trade_price_range_ratio"] = (agg_1m["trade_high"] - agg_1m["trade_low"]) / agg_1m["trade_vwap"]
    agg_1m.drop(columns=["sum_price_x_quantity", "sum_price_sq_x_quantity"], inplace=True)

    agg_1m["active_buy_base_ratio"] = agg_1m["active_buy_base_volume"] / agg_1m["base_volume"]
    agg_1m["active_buy_quote_ratio"] = agg_1m["active_buy_quote_volume"] / agg_1m["quote_volume"]
    agg_1m["trade_flow_imbalance_base"] = (agg_1m["active_buy_base_volume"] - agg_1m["active_sell_base_volume"]) / agg_1m["base_volume"]
    agg_1m["trade_flow_imbalance_quote"] = (agg_1m["active_buy_quote_volume"] - agg_1m["active_sell_quote_volume"]) / agg_1m["quote_volume"]

    first_windows = [int(v) for v in config["execution_windows_seconds"]["first"]]
    last_windows = [int(v) for v in config["execution_windows_seconds"]["last"]]
    for seconds in first_windows:
        mask = trades["minute_offset_ms"] < seconds * 1000
        agg_1m = add_window_vwap(agg_1m, trades, mask, f"first_{seconds}s")
    for seconds in last_windows:
        mask = trades["minute_offset_ms"] >= expected_interval_ms - seconds * 1000
        agg_1m = add_window_vwap(agg_1m, trades, mask, f"last_{seconds}s")

    agg_1m["has_trade_in_first_1s"] = agg_1m["first_1s_trade_count"] > 0
    agg_1m["has_trade_in_first_5s"] = agg_1m["first_5s_trade_count"] > 0
    agg_1m["has_trade_in_last_1s"] = agg_1m["last_1s_trade_count"] > 0
    agg_1m["has_trade_in_last_5s"] = agg_1m["last_5s_trade_count"] > 0

    prev_id = trades["agg_trade_id"].shift(1)
    prev_minute = trades["trade_minute"].shift(1)
    id_delta = trades["agg_trade_id"] - prev_id
    missing = (id_delta - 1).where(id_delta > 1, 0).fillna(0)
    same_minute = trades["trade_minute"].eq(prev_minute)
    internal_missing = missing.where(same_minute, 0)
    cross_missing = missing.where(~same_minute, 0)
    id_gap_frame = pd.DataFrame(
        {
            "open_time": trades["trade_minute"],
            "internal_missing": internal_missing.astype("int64"),
            "cross_missing": cross_missing.astype("int64"),
        }
    )
    id_gap_frame["internal_event"] = id_gap_frame["internal_missing"] > 0
    id_gap_frame["cross_event"] = id_gap_frame["cross_missing"] > 0
    id_stats = id_gap_frame.groupby("open_time", sort=True).agg(
        internal_missing_agg_trade_id_count=("internal_missing", "sum"),
        id_gap_event_count=("internal_event", "sum"),
        maximum_internal_id_gap=("internal_missing", "max"),
        cross_minute_missing_agg_trade_id_count=("cross_missing", "sum"),
        cross_minute_id_gap_event_count=("cross_event", "sum"),
        maximum_cross_minute_id_gap=("cross_missing", "max"),
    )
    agg_1m = agg_1m.merge(id_stats.reset_index(), on="open_time", how="left", validate="one_to_one")
    id_cols = [
        "internal_missing_agg_trade_id_count",
        "id_gap_event_count",
        "maximum_internal_id_gap",
        "cross_minute_missing_agg_trade_id_count",
        "cross_minute_id_gap_event_count",
        "maximum_cross_minute_id_gap",
    ]
    agg_1m[id_cols] = agg_1m[id_cols].fillna(0).astype("int64")
    agg_1m["has_internal_id_gap"] = agg_1m["id_gap_event_count"] > 0
    agg_1m["has_any_id_gap"] = (agg_1m["id_gap_event_count"] + agg_1m["cross_minute_id_gap_event_count"]) > 0

    first_minute = int(agg_1m["open_time"].min())
    last_minute = int(agg_1m["open_time"].max())
    agg_1m["is_file_first_minute"] = agg_1m["open_time"] == first_minute
    agg_1m["is_file_last_minute"] = agg_1m["open_time"] == last_minute
    agg_1m["is_boundary_partial_minute"] = agg_1m["is_file_first_minute"] | agg_1m["is_file_last_minute"]

    ordered_columns = [
        "open_time",
        "first_trade_time",
        "last_trade_time",
        "minute_first_offset_ms",
        "minute_last_offset_ms",
        "covered_duration_ms",
        "has_trade_in_first_1s",
        "has_trade_in_first_5s",
        "has_trade_in_last_1s",
        "has_trade_in_last_5s",
        "agg_trade_count",
        "first_agg_trade_id",
        "last_agg_trade_id",
        "base_volume",
        "quote_volume",
        "active_buy_base_volume",
        "active_sell_base_volume",
        "active_buy_quote_volume",
        "active_sell_quote_volume",
        "active_buy_base_ratio",
        "active_buy_quote_ratio",
        "trade_flow_imbalance_base",
        "trade_flow_imbalance_quote",
        "first_trade_price",
        "last_trade_price",
        "trade_high",
        "trade_low",
        "trade_vwap",
        "active_buy_vwap",
        "active_sell_vwap",
        "trade_price_std_volume_weighted",
        "trade_price_range_ratio",
        "mean_trade_base_size",
        "median_trade_base_size",
        "p90_trade_base_size",
        "p99_trade_base_size",
        "max_trade_base_size",
        "mean_trade_quote_size",
        "median_trade_quote_size",
        "p90_trade_quote_size",
        "p99_trade_quote_size",
        "max_trade_quote_size",
        "vwap_first_1s",
        "vwap_first_2s",
        "vwap_first_5s",
        "vwap_first_10s",
        "vwap_last_1s",
        "vwap_last_5s",
        "first_1s_trade_count",
        "first_5s_trade_count",
        "first_10s_trade_count",
        "last_1s_trade_count",
        "last_5s_trade_count",
        "internal_missing_agg_trade_id_count",
        "id_gap_event_count",
        "maximum_internal_id_gap",
        "has_internal_id_gap",
        "cross_minute_missing_agg_trade_id_count",
        "cross_minute_id_gap_event_count",
        "maximum_cross_minute_id_gap",
        "has_any_id_gap",
        "is_file_first_minute",
        "is_file_last_minute",
        "is_boundary_partial_minute",
    ]
    return agg_1m[ordered_columns].sort_values("open_time", kind="mergesort").reset_index(drop=True)


def validate_agg_1m_identities(agg_1m: pd.DataFrame, abs_tol: float, rel_tol: float) -> None:
    errors: list[str] = []
    for lhs, rhs_a, rhs_b in [
        ("base_volume", "active_buy_base_volume", "active_sell_base_volume"),
        ("quote_volume", "active_buy_quote_volume", "active_sell_quote_volume"),
    ]:
        expected = agg_1m[rhs_a] + agg_1m[rhs_b]
        tolerance = abs_tol + rel_tol * np.abs(agg_1m[lhs])
        ok = np.abs(agg_1m[lhs] - expected) <= tolerance
        if not ok.all():
            errors.append(f"{lhs} identity failed for {(~ok).sum()} minutes")
    if errors:
        raise Stage2BValidationError(errors)


def build_overlap(agg_1m: pd.DataFrame, kline: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    epsilon = float(config["epsilon"])
    warning_volume = float(config["warning_thresholds"]["volume_relative_error"])
    severe_volume = float(config["severe_thresholds"]["volume_relative_error"])
    warning_price = float(config["warning_thresholds"]["price_error_bps"])
    severe_price = float(config["severe_thresholds"]["price_error_bps"])

    if not agg_1m["open_time"].is_unique:
        raise Stage2BValidationError(["agg_1m open_time must be unique before join"])
    if not kline["open_time"].is_unique:
        raise Stage2BValidationError(["Kline open_time must be unique before join"])

    start = int(agg_1m["open_time"].min())
    end = int(agg_1m["open_time"].max())
    kline_overlap = kline.loc[(kline["open_time"] >= start) & (kline["open_time"] <= end)].copy()

    agg_pref = agg_1m.rename(
        columns={
            "base_volume": "agg_base_volume",
            "quote_volume": "agg_quote_volume",
            "active_buy_base_volume": "agg_active_buy_base_volume",
            "active_sell_base_volume": "agg_active_sell_base_volume",
            "active_buy_quote_volume": "agg_active_buy_quote_volume",
            "active_sell_quote_volume": "agg_active_sell_quote_volume",
        }
    )
    kline_pref = kline_overlap.rename(
        columns={
            "open": "kline_open",
            "high": "kline_high",
            "low": "kline_low",
            "close": "kline_close",
            "volume": "kline_base_volume",
            "quote_volume": "kline_quote_volume",
            "number_of_trades": "kline_number_of_trades",
            "taker_buy_base_volume": "kline_taker_buy_base_volume",
            "taker_buy_quote_volume": "kline_taker_buy_quote_volume",
            "continuity_segment_id": "kline_continuity_segment_id",
        }
    )
    overlap = agg_pref.merge(kline_pref, on="open_time", how="outer", indicator=True, validate="one_to_one")
    overlap["join_status"] = overlap["_merge"].map({"left_only": "agg_only", "right_only": "kline_only", "both": "both"}).astype("string")
    overlap.drop(columns=["_merge"], inplace=True)
    both = overlap["join_status"] == "both"

    overlap["kline_vwap"] = np.where(
        overlap["kline_base_volume"] > 0,
        overlap["kline_quote_volume"] / overlap["kline_base_volume"],
        np.nan,
    )

    def rel_abs(a: str, b: str, denom: str) -> pd.Series:
        return (overlap[a] - overlap[b]).abs() / np.maximum(overlap[denom].abs(), epsilon)

    def rel_signed(a: str, b: str, denom: str) -> pd.Series:
        return (overlap[a] - overlap[b]) / np.maximum(overlap[denom].abs(), epsilon)

    overlap["base_volume_abs_error"] = (overlap["agg_base_volume"] - overlap["kline_base_volume"]).abs()
    overlap["base_volume_relative_error"] = rel_abs("agg_base_volume", "kline_base_volume", "kline_base_volume")
    overlap["base_volume_signed_relative_error"] = rel_signed("agg_base_volume", "kline_base_volume", "kline_base_volume")
    overlap["quote_volume_abs_error"] = (overlap["agg_quote_volume"] - overlap["kline_quote_volume"]).abs()
    overlap["quote_volume_relative_error"] = rel_abs("agg_quote_volume", "kline_quote_volume", "kline_quote_volume")
    overlap["quote_volume_signed_relative_error"] = rel_signed("agg_quote_volume", "kline_quote_volume", "kline_quote_volume")
    overlap["active_buy_base_abs_error"] = (overlap["agg_active_buy_base_volume"] - overlap["kline_taker_buy_base_volume"]).abs()
    overlap["active_buy_base_relative_error"] = rel_abs("agg_active_buy_base_volume", "kline_taker_buy_base_volume", "kline_taker_buy_base_volume")
    overlap["active_buy_quote_abs_error"] = (overlap["agg_active_buy_quote_volume"] - overlap["kline_taker_buy_quote_volume"]).abs()
    overlap["active_buy_quote_relative_error"] = rel_abs("agg_active_buy_quote_volume", "kline_taker_buy_quote_volume", "kline_taker_buy_quote_volume")
    overlap["agg_to_kline_trade_count_ratio"] = overlap["agg_trade_count"] / np.maximum(overlap["kline_number_of_trades"], epsilon)
    overlap["trade_count_difference"] = overlap["agg_trade_count"] - overlap["kline_number_of_trades"]

    overlap["first_price_vs_open_relative_error"] = rel_abs("first_trade_price", "kline_open", "kline_open")
    overlap["last_price_vs_close_relative_error"] = rel_abs("last_trade_price", "kline_close", "kline_close")
    overlap["high_relative_error"] = rel_abs("trade_high", "kline_high", "kline_high")
    overlap["low_relative_error"] = rel_abs("trade_low", "kline_low", "kline_low")
    overlap["agg_vwap_vs_kline_vwap_relative_error"] = (overlap["trade_vwap"] - overlap["kline_vwap"]).abs() / np.maximum(overlap["kline_vwap"].abs(), epsilon)
    overlap["first_price_minus_open_bps"] = rel_signed("first_trade_price", "kline_open", "kline_open") * 10000
    overlap["last_price_minus_close_bps"] = rel_signed("last_trade_price", "kline_close", "kline_close") * 10000
    overlap["agg_vwap_minus_kline_vwap_bps"] = (overlap["trade_vwap"] - overlap["kline_vwap"]) / np.maximum(overlap["kline_vwap"].abs(), epsilon) * 10000
    overlap["high_minus_kline_high_bps"] = rel_signed("trade_high", "kline_high", "kline_high") * 10000
    overlap["low_minus_kline_low_bps"] = rel_signed("trade_low", "kline_low", "kline_low") * 10000

    for seconds in (1, 5, 10):
        overlap[f"vwap_first_{seconds}s_minus_open_bps"] = (overlap[f"vwap_first_{seconds}s"] - overlap["kline_open"]) / np.maximum(overlap["kline_open"].abs(), epsilon) * 10000
    overlap["vwap_last_5s_minus_close_bps"] = (overlap["vwap_last_5s"] - overlap["kline_close"]) / np.maximum(overlap["kline_close"].abs(), epsilon) * 10000

    price_abs_bps = overlap[
        [
            "first_price_minus_open_bps",
            "last_price_minus_close_bps",
            "high_minus_kline_high_bps",
            "low_minus_kline_low_bps",
            "agg_vwap_minus_kline_vwap_bps",
        ]
    ].abs()
    overlap["max_abs_price_error_bps"] = price_abs_bps.max(axis=1)
    max_volume_rel = overlap[["base_volume_relative_error", "quote_volume_relative_error"]].max(axis=1)
    overlap["volume_match_good"] = both & (max_volume_rel <= warning_volume)
    overlap["volume_match_warning"] = both & (max_volume_rel > warning_volume) & (max_volume_rel <= severe_volume)
    overlap["volume_match_severe"] = both & (max_volume_rel > severe_volume)
    overlap["price_match_good"] = both & (overlap["max_abs_price_error_bps"] <= warning_price)
    overlap["price_match_warning"] = both & (overlap["max_abs_price_error_bps"] > warning_price) & (overlap["max_abs_price_error_bps"] <= severe_price)
    overlap["price_match_severe"] = both & (overlap["max_abs_price_error_bps"] > severe_price)

    rule = config["reliability_rule"]
    reliable = pd.Series(True, index=overlap.index)
    if rule.get("require_both_sources", True):
        reliable &= both
    if rule.get("exclude_boundary_partial_minute", True):
        reliable &= ~overlap["is_boundary_partial_minute"].fillna(False)
    reliable &= overlap["base_volume_relative_error"] <= float(rule["max_base_volume_relative_error"])
    reliable &= overlap["quote_volume_relative_error"] <= float(rule["max_quote_volume_relative_error"])
    reliable &= overlap["max_abs_price_error_bps"] <= float(rule["max_price_error_bps"])
    if rule.get("require_first_5s_trade", True):
        reliable &= overlap["first_5s_trade_count"].fillna(0) > 0
    if rule.get("disallow_connection_anomaly", True):
        reliable &= overlap["join_status"] == "both"
    overlap["is_reliable_overlap_minute"] = reliable.fillna(False)

    unmatched = overlap.loc[overlap["join_status"] != "both", ["open_time", "join_status"]].copy()
    unmatched["open_time_utc"] = unmatched["open_time"].map(ms_to_utc_iso)
    return overlap.sort_values("open_time", kind="mergesort").reset_index(drop=True), unmatched


def describe(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def describe_abs_bps(series: pd.Series) -> dict[str, float | int | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "missing_rate": None, "mean_bps": None, "median_bps": None, "p90_abs_bps": None, "p95_abs_bps": None, "p99_abs_bps": None, "max_abs_bps": None, "positive_ratio": None, "negative_ratio": None}
    abs_clean = clean.abs()
    return {
        "count": int(clean.size),
        "missing_rate": None,
        "mean_bps": float(clean.mean()),
        "median_bps": float(clean.median()),
        "p90_abs_bps": float(abs_clean.quantile(0.90)),
        "p95_abs_bps": float(abs_clean.quantile(0.95)),
        "p99_abs_bps": float(abs_clean.quantile(0.99)),
        "max_abs_bps": float(abs_clean.max()),
        "positive_ratio": float((clean > 0).mean()),
        "negative_ratio": float((clean < 0).mean()),
    }


def make_reports(agg_1m: pd.DataFrame, overlap: pd.DataFrame, unmatched: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    both = overlap["join_status"] == "both"
    hour = (overlap.loc[both, "open_time"] // 3_600_000) * 3_600_000
    consistency_by_hour = overlap.loc[both].assign(hour=hour).groupby("hour", sort=True).agg(
        matched_minutes=("open_time", "size"),
        reliable_minutes=("is_reliable_overlap_minute", "sum"),
        base_volume_relative_error_mean=("base_volume_relative_error", "mean"),
        base_volume_relative_error_median=("base_volume_relative_error", "median"),
        quote_volume_relative_error_mean=("quote_volume_relative_error", "mean"),
        active_buy_base_relative_error_mean=("active_buy_base_relative_error", "mean"),
        max_abs_price_error_bps_mean=("max_abs_price_error_bps", "mean"),
        id_gap_minutes=("has_any_id_gap", "sum"),
    ).reset_index()
    consistency_by_hour["hour_utc"] = consistency_by_hour["hour"].map(ms_to_utc_iso)

    largest_n = int(config["largest_error_rows"])
    largest_volume = overlap.loc[both].copy()
    largest_volume["max_volume_relative_error"] = largest_volume[["base_volume_relative_error", "quote_volume_relative_error"]].max(axis=1)
    largest_volume = largest_volume.sort_values("max_volume_relative_error", ascending=False).head(largest_n)
    largest_volume["open_time_utc"] = largest_volume["open_time"].map(ms_to_utc_iso)

    largest_price = overlap.loc[both].sort_values("max_abs_price_error_bps", ascending=False).head(largest_n).copy()
    largest_price["open_time_utc"] = largest_price["open_time"].map(ms_to_utc_iso)

    gap_df = overlap.loc[both].copy()
    gap_df["id_gap_event_bucket"] = np.select(
        [gap_df["id_gap_event_count"] == 0, gap_df["id_gap_event_count"] == 1, gap_df["id_gap_event_count"] > 1],
        ["no_internal_gap", "one_internal_gap", "multiple_internal_gaps"],
        default="unknown",
    )
    gap_df["max_internal_gap_bucket"] = pd.cut(
        gap_df["maximum_internal_id_gap"],
        bins=[-1, 0, 5, 20, 100, np.inf],
        labels=["0", "1-5", "6-20", "21-100", ">100"],
    )
    by_gap_event = gap_df.groupby("id_gap_event_bucket", sort=True).agg(
        minute_count=("open_time", "size"),
        base_volume_relative_error_mean=("base_volume_relative_error", "mean"),
        base_volume_relative_error_median=("base_volume_relative_error", "median"),
        quote_volume_relative_error_mean=("quote_volume_relative_error", "mean"),
        active_buy_base_relative_error_mean=("active_buy_base_relative_error", "mean"),
        max_abs_price_error_bps_mean=("max_abs_price_error_bps", "mean"),
        reliable_minutes=("is_reliable_overlap_minute", "sum"),
    ).reset_index()
    by_gap_event.insert(0, "group_type", "internal_gap_event_count")
    by_gap_event.rename(columns={"id_gap_event_bucket": "group"}, inplace=True)
    by_max_gap = gap_df.groupby("max_internal_gap_bucket", observed=False).agg(
        minute_count=("open_time", "size"),
        base_volume_relative_error_mean=("base_volume_relative_error", "mean"),
        base_volume_relative_error_median=("base_volume_relative_error", "median"),
        quote_volume_relative_error_mean=("quote_volume_relative_error", "mean"),
        active_buy_base_relative_error_mean=("active_buy_base_relative_error", "mean"),
        max_abs_price_error_bps_mean=("max_abs_price_error_bps", "mean"),
        reliable_minutes=("is_reliable_overlap_minute", "sum"),
    ).reset_index()
    by_max_gap.insert(0, "group_type", "maximum_internal_id_gap")
    by_max_gap.rename(columns={"max_internal_gap_bucket": "group"}, inplace=True)
    gap_analysis = pd.concat([by_gap_event, by_max_gap], ignore_index=True)

    return {
        "consistency_by_hour": consistency_by_hour,
        "largest_volume_errors": largest_volume,
        "largest_price_errors": largest_price,
        "id_gap_error_analysis": gap_analysis,
        "unmatched_minutes": unmatched,
    }


def write_markdown_reports(
    paths: dict[str, Path],
    agg_1m: pd.DataFrame,
    overlap: pd.DataFrame,
    reports: dict[str, pd.DataFrame],
    stats: dict[str, Any],
    elapsed_seconds: float,
    peak_memory_bytes: int,
    rss_bytes: int | None,
) -> None:
    both = overlap["join_status"] == "both"
    non_boundary = ~agg_1m["is_boundary_partial_minute"]
    agg_lines = [
        "# 阶段2B：agg trades 1分钟聚合报告",
        "",
        "- 范围限制: 短窗口agg trades验证；未生成60分钟标签、正式模型特征、切分或模型。",
        f"- 输入agg行数: `{stats['agg_input_rows']}`",
        f"- 输出聚合分钟数: `{len(agg_1m)}`",
        f"- 时间范围: `{ms_to_utc_iso(int(agg_1m['open_time'].min()))}` 到 `{ms_to_utc_iso(int(agg_1m['open_time'].max()))}`",
        f"- 文件首分钟: `{ms_to_utc_iso(stats['first_minute'])}`；文件末分钟: `{ms_to_utc_iso(stats['last_minute'])}`",
        f"- 排除边界分钟后的完整候选分钟: `{int(non_boundary.sum())}`",
        f"- 分钟内ID gap事件数: `{int(agg_1m['id_gap_event_count'].sum())}`；跨分钟ID gap事件数: `{int(agg_1m['cross_minute_id_gap_event_count'].sum())}`",
        f"- 运行耗时(seconds): `{elapsed_seconds:.2f}`；tracemalloc峰值(bytes): `{peak_memory_bytes}`；RSS(bytes): `{rss_bytes}`",
        "",
        "## 分布统计",
        table(
            [
                {"指标": "每分钟agg_trade_count", **describe(agg_1m["agg_trade_count"])},
                {"指标": "每分钟base_volume", **describe(agg_1m["base_volume"])},
                {"指标": "active_buy_base_ratio", **describe(agg_1m["active_buy_base_ratio"])},
                {"指标": "trade_flow_imbalance_base", **describe(agg_1m["trade_flow_imbalance_base"])},
            ],
            ["指标", "count", "mean", "median", "p90", "p95", "p99", "max"],
        ),
        "## 执行窗口成交覆盖率",
        table(
            [
                {"窗口": "first_1s", "有成交分钟数": int((agg_1m["first_1s_trade_count"] > 0).sum()), "覆盖率": float((agg_1m["first_1s_trade_count"] > 0).mean())},
                {"窗口": "first_5s", "有成交分钟数": int((agg_1m["first_5s_trade_count"] > 0).sum()), "覆盖率": float((agg_1m["first_5s_trade_count"] > 0).mean())},
                {"窗口": "first_10s", "有成交分钟数": int((agg_1m["first_10s_trade_count"] > 0).sum()), "覆盖率": float((agg_1m["first_10s_trade_count"] > 0).mean())},
                {"窗口": "last_1s", "有成交分钟数": int((agg_1m["last_1s_trade_count"] > 0).sum()), "覆盖率": float((agg_1m["last_1s_trade_count"] > 0).mean())},
                {"窗口": "last_5s", "有成交分钟数": int((agg_1m["last_5s_trade_count"] > 0).sum()), "覆盖率": float((agg_1m["last_5s_trade_count"] > 0).mean())},
            ],
            ["窗口", "有成交分钟数", "覆盖率"],
        ),
    ]
    paths["aggregation"].write_text("\n".join(agg_lines) + "\n", encoding="utf-8")

    consistency_rows = [
        {"指标": "agg分钟数", "值": len(agg_1m)},
        {"指标": "K线重叠分钟数", "值": stats["kline_overlap_minutes"]},
        {"指标": "成功匹配分钟数", "值": int(both.sum())},
        {"指标": "仅agg分钟数", "值": int((overlap["join_status"] == "agg_only").sum())},
        {"指标": "仅K线分钟数", "值": int((overlap["join_status"] == "kline_only").sum())},
        {"指标": "volume_match_good", "值": int(overlap["volume_match_good"].sum())},
        {"指标": "volume_match_warning", "值": int(overlap["volume_match_warning"].sum())},
        {"指标": "volume_match_severe", "值": int(overlap["volume_match_severe"].sum())},
        {"指标": "price_match_good", "值": int(overlap["price_match_good"].sum())},
        {"指标": "price_match_warning", "值": int(overlap["price_match_warning"].sum())},
        {"指标": "price_match_severe", "值": int(overlap["price_match_severe"].sum())},
        {"指标": "is_reliable_overlap_minute", "值": int(overlap["is_reliable_overlap_minute"].sum())},
    ]
    consistency_lines = [
        "# 阶段2B：agg/K线一致性报告",
        "",
        table(consistency_rows, ["指标", "值"]),
        "## 误差分布",
        table(
            [
                {"指标": "base_volume_relative_error", **describe(overlap.loc[both, "base_volume_relative_error"])},
                {"指标": "quote_volume_relative_error", **describe(overlap.loc[both, "quote_volume_relative_error"])},
                {"指标": "active_buy_base_relative_error", **describe(overlap.loc[both, "active_buy_base_relative_error"])},
                {"指标": "active_buy_quote_relative_error", **describe(overlap.loc[both, "active_buy_quote_relative_error"])},
                {"指标": "first_price_vs_open_relative_error", **describe(overlap.loc[both, "first_price_vs_open_relative_error"])},
                {"指标": "last_price_vs_close_relative_error", **describe(overlap.loc[both, "last_price_vs_close_relative_error"])},
                {"指标": "high_relative_error", **describe(overlap.loc[both, "high_relative_error"])},
                {"指标": "low_relative_error", **describe(overlap.loc[both, "low_relative_error"])},
                {"指标": "agg_vwap_vs_kline_vwap_relative_error", **describe(overlap.loc[both, "agg_vwap_vs_kline_vwap_relative_error"])},
            ],
            ["指标", "count", "mean", "median", "p90", "p95", "p99", "max"],
        ),
        "## ID gap与误差关系",
        table(reports["id_gap_error_analysis"].to_dict(orient="records"), list(reports["id_gap_error_analysis"].columns)),
    ]
    paths["consistency"].write_text("\n".join(consistency_lines) + "\n", encoding="utf-8")

    proxy_specs = [
        ("kline open vs first trade", "first_price_minus_open_bps"),
        ("kline open vs first 1s VWAP", "vwap_first_1s_minus_open_bps"),
        ("kline open vs first 5s VWAP", "vwap_first_5s_minus_open_bps"),
        ("kline open vs first 10s VWAP", "vwap_first_10s_minus_open_bps"),
        ("kline close vs last trade", "last_price_minus_close_bps"),
        ("kline close vs last 5s VWAP", "vwap_last_5s_minus_close_bps"),
    ]
    proxy_rows = []
    total_matched = int(both.sum())
    for label, col in proxy_specs:
        d = describe_abs_bps(overlap.loc[both, col])
        d["missing_rate"] = 1.0 - (d["count"] / total_matched if total_matched else 0)
        proxy_rows.append({"代理": label, **d})
    first5 = next(row for row in proxy_rows if row["代理"] == "kline open vs first 5s VWAP")
    conclusion = (
        "在当前66.77小时短窗口内，K线open相对first 5s VWAP的典型偏差较小，"
        "可作为后续K线近似标签方案的候选代理；但该结论不能推广到完整2017-2026区间。"
        if first5["p95_abs_bps"] is not None and first5["p95_abs_bps"] <= 5.0
        else "在当前66.77小时短窗口内，K线open相对first 5s VWAP存在不可忽略偏差，后续应谨慎比较逐笔VWAP标签。"
    )
    proxy_lines = [
        "# 阶段2B：分钟内执行价格代理分析报告",
        "",
        table(
            proxy_rows,
            [
                "代理",
                "count",
                "missing_rate",
                "mean_bps",
                "median_bps",
                "p90_abs_bps",
                "p95_abs_bps",
                "p99_abs_bps",
                "max_abs_bps",
                "positive_ratio",
                "negative_ratio",
            ],
        ),
        "## 数据驱动结论",
        f"- {conclusion}",
        "- 本结论只基于agg trades覆盖的约66.77小时窗口。",
        "- 本阶段没有构造entry_price、settlement_price、future_return或label。",
    ]
    paths["execution_proxy"].write_text("\n".join(proxy_lines) + "\n", encoding="utf-8")


def run_stage2b(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    agg_path = (root / config["agg_input_path"]).resolve()
    kline_path = (root / config["kline_input_path"]).resolve()
    agg_out = (root / config["agg_1m_output_path"]).resolve()
    overlap_out = (root / config["overlap_output_path"]).resolve()
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
    tracemalloc.start()
    logging.info("Stage 2B agg/kline analysis started")

    agg_cols = required_schema_columns(
        agg_path,
        ["agg_trade_id", "price", "quantity", "trade_time", "is_buyer_maker", "quote_quantity", "is_active_buy", "is_active_sell", "id_gap_before"],
        "agg_trades",
    )
    kline_cols = required_schema_columns(
        kline_path,
        ["open_time", "open", "high", "low", "close", "volume", "quote_volume", "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume", "continuity_segment_id"],
        "klines",
    )
    agg = pd.read_parquet(agg_path, columns=agg_cols)
    kline = pd.read_parquet(kline_path, columns=kline_cols)
    validate_agg_input(agg, float(config["floating_absolute_tolerance"]), float(config["floating_relative_tolerance"]))
    validate_kline_input(kline)
    agg_1m = aggregate_agg_trades_1m(agg, config)
    validate_agg_1m_identities(agg_1m, float(config["floating_absolute_tolerance"]), float(config["floating_relative_tolerance"]))
    overlap, unmatched = build_overlap(agg_1m, kline, config)
    reports = make_reports(agg_1m, overlap, unmatched, config)

    for path in [agg_out, overlap_out, *report_paths.values()]:
        ensure_parent(path)
    agg_1m.to_parquet(agg_out, index=False, engine="pyarrow", compression=config["parquet_compression"])
    overlap.to_parquet(overlap_out, index=False, engine="pyarrow", compression=config["parquet_compression"])
    for key, frame in reports.items():
        csv_key = {
            "consistency_by_hour": "consistency_by_hour",
            "largest_volume_errors": "largest_volume_errors",
            "largest_price_errors": "largest_price_errors",
            "id_gap_error_analysis": "id_gap_error_analysis",
            "unmatched_minutes": "unmatched_minutes",
        }[key]
        frame.to_csv(report_paths[csv_key], index=False, encoding="utf-8")

    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    rss = get_rss_bytes()
    stats = {
        "agg_input_rows": int(len(agg)),
        "agg_1m_rows": int(len(agg_1m)),
        "kline_overlap_minutes": int(((kline["open_time"] >= agg_1m["open_time"].min()) & (kline["open_time"] <= agg_1m["open_time"].max())).sum()),
        "matched_minutes": int((overlap["join_status"] == "both").sum()),
        "agg_only_minutes": int((overlap["join_status"] == "agg_only").sum()),
        "kline_only_minutes": int((overlap["join_status"] == "kline_only").sum()),
        "first_minute": int(agg_1m["open_time"].min()),
        "last_minute": int(agg_1m["open_time"].max()),
        "reliable_overlap_minutes": int(overlap["is_reliable_overlap_minute"].sum()),
        "agg_1m_output_size_bytes": int(agg_out.stat().st_size),
        "overlap_output_size_bytes": int(overlap_out.stat().st_size),
        "elapsed_seconds": elapsed,
        "python_tracemalloc_peak_bytes": int(peak),
        "process_rss_bytes": rss,
    }
    write_markdown_reports(report_paths, agg_1m, overlap, reports, stats, elapsed, int(peak), rss)
    logging.info("Stats: %s", stats)
    logging.info("Stage 2B completed in %.2f seconds", elapsed)
    tracemalloc.stop()
    return {
        "stats": stats,
        "agg_1m_schema": parquet_schema(agg_out),
        "overlap_schema": parquet_schema(overlap_out),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2B aggregate agg trades to 1m and compare with klines.")
    parser.add_argument("--config", default="config/stage2b_agg_kline_analysis.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage2b(config_path, root)


if __name__ == "__main__":
    main()

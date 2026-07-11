from __future__ import annotations

import argparse
import ast
import csv
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object


MISSING_STRINGS = {"", "nan", "na", "n/a", "null", "none"}
BOOL_STRINGS = {"true", "false"}
MINUTE_MS = 60_000
DATETIME_STRING_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        else:
            value = value.astimezone(timezone.utc)
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def parse_datetime_utc(value: str) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_candidate_path(root: Path, candidates: list[str], role: str) -> tuple[Path, list[str]]:
    attempted = []
    for candidate in candidates:
        path = (root / candidate).resolve()
        attempted.append(str(path))
        if path.exists():
            return path, attempted
    raise FileNotFoundError(f"{role} file not found. Tried: {attempted}")


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def infer_meaning(column: str) -> str:
    name = normalize_name(column)
    meanings = {
        "symbol": "交易对符号",
        "interval": "K线周期",
        "open_time": "K线开盘时间戳",
        "open_time_utc": "K线开盘时间(UTC字符串)",
        "open": "开盘价",
        "high": "最高价",
        "low": "最低价",
        "close": "收盘价",
        "volume": "基础资产成交量",
        "close_time": "K线收盘时间戳",
        "close_time_utc": "K线收盘时间(UTC字符串)",
        "quote_volume": "计价资产成交额",
        "number_of_trades": "成交笔数",
        "taker_buy_base_volume": "主动买入基础资产成交量",
        "taker_buy_quote_volume": "主动买入计价资产成交额",
        "agg_trade_id": "聚合成交ID",
        "price": "成交价格",
        "quantity": "成交数量",
        "first_trade_id": "聚合成交包含的首个成交ID",
        "last_trade_id": "聚合成交包含的最后成交ID",
        "trade_time": "成交时间戳",
        "trade_time_utc": "成交时间(UTC字符串)",
        "is_buyer_maker": "买方是否为挂单方",
        "is_best_match": "是否最佳撮合",
    }
    if name in meanings:
        return meanings[name]
    if "time" in name or "date" in name or "utc" in name:
        return "可能的时间字段"
    if "price" in name:
        return "可能的价格字段"
    if "volume" in name or "quantity" in name or name == "qty":
        return "可能的成交量/数量字段"
    if "id" in name:
        return "可能的ID字段"
    return "含义未确认"


@dataclass
class HyperLogLog:
    precision: int = 12
    registers: np.ndarray = field(default_factory=lambda: np.zeros(1 << 12, dtype=np.uint8))

    def __post_init__(self) -> None:
        size = 1 << self.precision
        if self.registers.size != size:
            self.registers = np.zeros(size, dtype=np.uint8)

    def update_series(self, values: pd.Series) -> None:
        if values.empty:
            return
        hashes = hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64, copy=False)
        if hashes.size == 0:
            return
        mask = np.uint64((1 << self.precision) - 1)
        idx = (hashes & mask).astype(np.int64)
        shifted = hashes >> np.uint64(self.precision)
        remaining_bits = 64 - self.precision
        ranks = np.empty(shifted.shape, dtype=np.uint8)
        nonzero = shifted != 0
        if np.any(nonzero):
            logs = np.floor(np.log2(shifted[nonzero].astype(np.float64))).astype(np.int16)
            ranks[nonzero] = (remaining_bits - logs).astype(np.uint8)
        ranks[~nonzero] = remaining_bits + 1
        np.maximum.at(self.registers, idx, ranks)

    def estimate(self) -> int:
        m = float(1 << self.precision)
        if m == 16:
            alpha = 0.673
        elif m == 32:
            alpha = 0.697
        elif m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1.0 + 1.079 / m)
        indicator = np.sum(np.exp2(-self.registers.astype(np.float64)))
        raw = alpha * m * m / indicator
        zeros = int(np.count_nonzero(self.registers == 0))
        if raw <= 2.5 * m and zeros:
            raw = m * math.log(m / zeros)
        return int(round(raw))


@dataclass
class ColumnAudit:
    name: str
    unique_exact_threshold: int
    row_count: int = 0
    missing_count: int = 0
    non_missing_count: int = 0
    bool_count: int = 0
    integer_string_count: int = 0
    numeric_count: int = 0
    datetime_count: int = 0
    numeric_min: float | None = None
    numeric_max: float | None = None
    datetime_min: pd.Timestamp | None = None
    datetime_max: pd.Timestamp | None = None
    string_min: str | None = None
    string_max: str | None = None
    exact_values: set[str] | None = field(default_factory=set)
    hll: HyperLogLog = field(default_factory=HyperLogLog)

    def update(self, series: pd.Series) -> None:
        values = series.astype(str)
        self.row_count += int(values.size)
        lowered = values.str.strip().str.lower()
        missing_mask = lowered.isin(MISSING_STRINGS)
        missing = int(missing_mask.sum())
        self.missing_count += missing
        nonmissing = values[~missing_mask]
        self.non_missing_count += int(nonmissing.size)
        if nonmissing.empty:
            return

        nonmissing_lower = lowered[~missing_mask]
        self.bool_count += int(nonmissing_lower.isin(BOOL_STRINGS).sum())
        self.integer_string_count += int(nonmissing.str.match(r"^[+-]?\d+$", na=False).sum())

        numeric = pd.to_numeric(nonmissing, errors="coerce")
        numeric_ok = numeric.notna()
        numeric_count = int(numeric_ok.sum())
        self.numeric_count += numeric_count
        if numeric_count:
            numeric_values = numeric[numeric_ok].astype(float)
            chunk_min = float(numeric_values.min())
            chunk_max = float(numeric_values.max())
            self.numeric_min = chunk_min if self.numeric_min is None else min(self.numeric_min, chunk_min)
            self.numeric_max = chunk_max if self.numeric_max is None else max(self.numeric_max, chunk_max)

        if is_datetime_string_candidate(self.name, nonmissing):
            datetime_like = nonmissing.str.match(DATETIME_STRING_RE, na=False)
            parsed_count = int(datetime_like.sum())
            self.datetime_count += parsed_count
            if parsed_count:
                datetime_strings = nonmissing[datetime_like]
                chunk_min_dt = parse_datetime_utc(str(datetime_strings.min()))
                chunk_max_dt = parse_datetime_utc(str(datetime_strings.max()))
                if chunk_min_dt is not None:
                    self.datetime_min = chunk_min_dt if self.datetime_min is None else min(self.datetime_min, chunk_min_dt)
                if chunk_max_dt is not None:
                    self.datetime_max = chunk_max_dt if self.datetime_max is None else max(self.datetime_max, chunk_max_dt)

        chunk_min_s = str(nonmissing.min())
        chunk_max_s = str(nonmissing.max())
        self.string_min = chunk_min_s if self.string_min is None else min(self.string_min, chunk_min_s)
        self.string_max = chunk_max_s if self.string_max is None else max(self.string_max, chunk_max_s)

        self.hll.update_series(nonmissing)
        if self.exact_values is not None:
            self.exact_values.update(str(v) for v in pd.unique(nonmissing))
            if len(self.exact_values) > self.unique_exact_threshold:
                self.exact_values = None

    def data_type(self) -> str:
        if self.non_missing_count == 0:
            return "empty"
        if self.bool_count == self.non_missing_count:
            return "boolean"
        if self.numeric_count == self.non_missing_count:
            if self.integer_string_count == self.non_missing_count:
                return "integer"
            return "decimal"
        if self.datetime_count == self.non_missing_count:
            return "datetime_string"
        return "string"

    def min_max(self) -> tuple[Any, Any]:
        dtype = self.data_type()
        if dtype in {"integer", "decimal"}:
            if dtype == "integer" and self.numeric_min is not None and self.numeric_max is not None:
                return int(self.numeric_min), int(self.numeric_max)
            return self.numeric_min, self.numeric_max
        if dtype == "datetime_string":
            return to_jsonable(self.datetime_min), to_jsonable(self.datetime_max)
        return self.string_min, self.string_max

    def unique_summary(self) -> dict[str, Any]:
        estimate = self.hll.estimate()
        if self.exact_values is not None:
            return {"count": len(self.exact_values), "is_exact": True, "estimate": estimate}
        return {"count": estimate, "is_exact": False, "estimate": estimate}

    def as_dict(self) -> dict[str, Any]:
        min_value, max_value = self.min_max()
        unique = self.unique_summary()
        return {
            "original_name": self.name,
            "inferred_meaning": infer_meaning(self.name),
            "data_type": self.data_type(),
            "missing_count": self.missing_count,
            "unique_count": unique["count"],
            "unique_count_is_exact": unique["is_exact"],
            "min": to_jsonable(min_value),
            "max": to_jsonable(max_value),
        }


def is_datetime_string_candidate(name: str, values: pd.Series) -> bool:
    normalized = normalize_name(name)
    if "utc" in normalized or "date" in normalized or normalized.endswith("_dt"):
        return True
    sample = values.head(20).astype(str)
    return bool(sample.str.contains(r"\d{4}-\d{2}-\d{2}|T\d{2}:", regex=True).any())


def infer_timestamp_unit_from_range(min_value: float | int | None, max_value: float | int | None) -> str | None:
    if min_value is None or max_value is None:
        return None
    magnitude = max(abs(float(min_value)), abs(float(max_value)))
    if 1e8 <= magnitude < 1e11:
        return "seconds"
    if 1e11 <= magnitude < 1e14:
        return "milliseconds"
    if 1e14 <= magnitude < 1e17:
        return "microseconds"
    return None


def numeric_timestamp_to_utc(value: float | int, unit: str) -> str:
    divisor = {"seconds": 1, "milliseconds": 1_000, "microseconds": 1_000_000}[unit]
    return datetime.fromtimestamp(float(value) / divisor, tz=timezone.utc).isoformat()


def detect_time_fields(field_audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for field_info in field_audits:
        name = field_info["original_name"]
        normalized = normalize_name(name)
        dtype = field_info["data_type"]
        looks_time_named = any(token in normalized for token in ("time", "timestamp", "date", "utc"))
        if dtype == "datetime_string" and looks_time_named:
            results.append(
                {
                    "field": name,
                    "unit": "formatted_time",
                    "min_utc": field_info["min"],
                    "max_utc": field_info["max"],
                    "basis": "字段名包含时间含义且内容可解析为格式化时间",
                }
            )
            continue
        if dtype == "integer" and looks_time_named:
            unit = infer_timestamp_unit_from_range(field_info["min"], field_info["max"])
            min_utc = max_utc = None
            if unit is not None:
                min_utc = numeric_timestamp_to_utc(field_info["min"], unit)
                max_utc = numeric_timestamp_to_utc(field_info["max"], unit)
            results.append(
                {
                    "field": name,
                    "unit": unit or "unknown_numeric_time",
                    "min_utc": min_utc,
                    "max_utc": max_utc,
                    "basis": "字段名包含时间含义，单位按数值量级推断",
                }
            )
    return results


def find_column(columns: list[str], candidates: list[str], contains_all: list[str] | None = None) -> str | None:
    by_name = {normalize_name(c): c for c in columns}
    for candidate in candidates:
        if candidate in by_name:
            return by_name[candidate]
    if contains_all:
        for column in columns:
            normalized = normalize_name(column)
            if all(token in normalized for token in contains_all):
                return column
    return None


def infer_field_mapping(role: str, columns: list[str]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def add(logical: str, column: str | None, meaning: str, confidence: str) -> None:
        mapping[logical] = {
            "column": column,
            "meaning": meaning,
            "confidence": confidence if column else "missing",
        }

    if role == "klines":
        add("symbol", find_column(columns, ["symbol"]), "交易对符号", "high")
        add("interval", find_column(columns, ["interval"]), "K线周期", "high")
        add("open_time", find_column(columns, ["open_time"], ["open", "time"]), "K线开盘时间", "high")
        add("open_time_utc", find_column(columns, ["open_time_utc"], ["open", "utc"]), "K线开盘UTC时间", "high")
        add("close_time", find_column(columns, ["close_time"], ["close", "time"]), "K线收盘时间", "high")
        add("open", find_column(columns, ["open"]), "开盘价", "high")
        add("high", find_column(columns, ["high"]), "最高价", "high")
        add("low", find_column(columns, ["low"]), "最低价", "high")
        add("close", find_column(columns, ["close"]), "收盘价", "high")
        add("volume", find_column(columns, ["volume", "base_volume"]), "基础资产成交量", "high")
        add("quote_volume", find_column(columns, ["quote_volume"], ["quote", "volume"]), "计价资产成交额", "high")
        add("number_of_trades", find_column(columns, ["number_of_trades", "trade_count"], ["trades"]), "成交笔数", "high")
        add(
            "taker_buy_base_volume",
            find_column(columns, ["taker_buy_base_volume"], ["taker", "buy", "base"]),
            "主动买入基础资产成交量",
            "high",
        )
    elif role == "agg_trades":
        add("symbol", find_column(columns, ["symbol"]), "交易对符号", "high")
        add("agg_trade_id", find_column(columns, ["agg_trade_id", "aggregate_trade_id", "a"], ["agg", "trade", "id"]), "聚合成交ID", "high")
        add("price", find_column(columns, ["price", "p"]), "成交价格", "high")
        add("quantity", find_column(columns, ["quantity", "qty", "q"]), "成交数量", "high")
        add("first_trade_id", find_column(columns, ["first_trade_id", "f"], ["first", "trade", "id"]), "首个成交ID", "high")
        add("last_trade_id", find_column(columns, ["last_trade_id", "l"], ["last", "trade", "id"]), "最后成交ID", "high")
        add("trade_time", find_column(columns, ["trade_time", "transact_time", "t"], ["trade", "time"]), "成交时间", "high")
        add("trade_time_utc", find_column(columns, ["trade_time_utc"], ["trade", "utc"]), "成交UTC时间", "high")
        add("is_buyer_maker", find_column(columns, ["is_buyer_maker", "m"], ["buyer", "maker"]), "买方是否为挂单方", "high")
        add("is_best_match", find_column(columns, ["is_best_match", "M"], ["best", "match"]), "是否最佳撮合", "high")
    return mapping


def series_to_epoch_ms(series: pd.Series, unit_hint: str | None = None) -> tuple[pd.Series, str | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any() and numeric.notna().sum() == len(series):
        unit = unit_hint or infer_timestamp_unit_from_range(float(numeric.min()), float(numeric.max()))
        if unit == "seconds":
            return (numeric * 1_000).round().astype("int64"), unit
        if unit == "milliseconds":
            return numeric.round().astype("int64"), unit
        if unit == "microseconds":
            return (numeric / 1_000).round().astype("int64"), unit
        return pd.Series([pd.NA] * len(series), index=series.index, dtype="Int64"), unit
    parsed_ms = series.astype(str).map(lambda value: int(parse_datetime_utc(value).timestamp() * 1000) if parse_datetime_utc(value) else pd.NA)
    if parsed_ms.notna().any():
        return parsed_ms.astype("Int64"), "formatted_time"
    return pd.Series([pd.NA] * len(series), index=series.index, dtype="Int64"), None


@dataclass
class KlineChecks:
    mapping: dict[str, dict[str, Any]]
    max_examples: int
    row_count: int = 0
    time_unit: str | None = None
    min_open_time_ms: int | None = None
    max_open_time_ms: int | None = None
    previous_open_time_ms: int | None = None
    sorted_violations: int = 0
    duplicate_open_time_count: int = 0
    missing_minutes: int = 0
    gap_examples: list[dict[str, Any]] = field(default_factory=list)
    interval_counts: Counter = field(default_factory=Counter)
    non_1m_intervals: int = 0
    time_parse_errors: int = 0
    nonpositive_price_counts: Counter = field(default_factory=Counter)
    negative_volume_counts: Counter = field(default_factory=Counter)
    ohlc_violation_count: int = 0
    close_time_span_violation_count: int = 0

    def column(self, logical: str) -> str | None:
        return self.mapping.get(logical, {}).get("column")

    def process(self, chunk: pd.DataFrame) -> None:
        self.row_count += len(chunk)
        open_time_col = self.column("open_time") or self.column("open_time_utc")
        if open_time_col:
            open_ms, unit = series_to_epoch_ms(chunk[open_time_col], self.time_unit)
            self.time_unit = self.time_unit or unit
            valid = open_ms.dropna().astype("int64").to_numpy()
            self.time_parse_errors += len(open_ms) - len(valid)
            if valid.size:
                chunk_min = int(valid.min())
                chunk_max = int(valid.max())
                self.min_open_time_ms = chunk_min if self.min_open_time_ms is None else min(self.min_open_time_ms, chunk_min)
                self.max_open_time_ms = chunk_max if self.max_open_time_ms is None else max(self.max_open_time_ms, chunk_max)
                if self.previous_open_time_ms is not None:
                    valid = np.insert(valid, 0, self.previous_open_time_ms)
                diffs = np.diff(valid)
                for diff_index, diff in enumerate(diffs):
                    diff_int = int(diff)
                    self.interval_counts[diff_int] += 1
                    if diff_int != MINUTE_MS:
                        self.non_1m_intervals += 1
                    if diff_int < 0:
                        self.sorted_violations += 1
                    elif diff_int == 0:
                        self.duplicate_open_time_count += 1
                    elif diff_int > MINUTE_MS:
                        missing = diff_int // MINUTE_MS - 1
                        if diff_int % MINUTE_MS:
                            missing = max(missing, 0)
                        self.missing_minutes += int(missing)
                        if len(self.gap_examples) < self.max_examples:
                            prev_ms = int(valid[diff_index])
                            curr_ms = prev_ms + diff_int
                            self.gap_examples.append(
                                {
                                    "previous_open_time_utc": ms_to_utc(prev_ms),
                                    "next_open_time_utc": ms_to_utc(curr_ms),
                                    "interval_ms": diff_int,
                                    "missing_minutes": int(missing),
                                }
                            )
                self.previous_open_time_ms = int(valid[-1])

        for logical in ("open", "high", "low", "close"):
            col = self.column(logical)
            if col:
                values = pd.to_numeric(chunk[col], errors="coerce")
                self.nonpositive_price_counts[col] += int((values <= 0).sum())

        for logical in ("volume", "quote_volume", "taker_buy_base_volume"):
            col = self.column(logical)
            if col:
                values = pd.to_numeric(chunk[col], errors="coerce")
                self.negative_volume_counts[col] += int((values < 0).sum())

        ohlc_cols = [self.column(x) for x in ("open", "high", "low", "close")]
        if all(ohlc_cols):
            open_v, high_v, low_v, close_v = [pd.to_numeric(chunk[col], errors="coerce") for col in ohlc_cols]
            violation = (high_v < low_v) | (open_v > high_v) | (open_v < low_v) | (close_v > high_v) | (close_v < low_v)
            self.ohlc_violation_count += int(violation.sum())

        close_col = self.column("close_time")
        open_col = self.column("open_time")
        if close_col and open_col:
            open_ms, _ = series_to_epoch_ms(chunk[open_col], self.time_unit)
            close_ms, _ = series_to_epoch_ms(chunk[close_col], self.time_unit)
            span = close_ms - open_ms
            self.close_time_span_violation_count += int((span != 59_999).sum())

    def as_dict(self) -> dict[str, Any]:
        top_intervals = [
            {"interval_ms": int(interval), "count": int(count)}
            for interval, count in self.interval_counts.most_common(10)
        ]
        return {
            "row_count": self.row_count,
            "primary_time_unit": self.time_unit,
            "time_range_utc": {
                "start": ms_to_utc(self.min_open_time_ms) if self.min_open_time_ms is not None else None,
                "end": ms_to_utc(self.max_open_time_ms) if self.max_open_time_ms is not None else None,
            },
            "is_sorted_by_open_time": self.sorted_violations == 0,
            "sorted_violations": self.sorted_violations,
            "duplicate_open_time_count": self.duplicate_open_time_count,
            "missing_minutes": self.missing_minutes,
            "gap_examples": self.gap_examples,
            "is_interval_stable_1m": self.non_1m_intervals == 0,
            "non_1m_interval_count": self.non_1m_intervals,
            "interval_counts_top": top_intervals,
            "time_parse_errors": self.time_parse_errors,
            "nonpositive_price_counts": dict(self.nonpositive_price_counts),
            "has_nonpositive_prices": any(v > 0 for v in self.nonpositive_price_counts.values()),
            "negative_volume_counts": dict(self.negative_volume_counts),
            "has_negative_volume": any(v > 0 for v in self.negative_volume_counts.values()),
            "ohlc_violation_count": self.ohlc_violation_count,
            "has_ohlc_violations": self.ohlc_violation_count > 0,
            "close_time_span_violation_count": self.close_time_span_violation_count,
        }


@dataclass
class AggTradeChecks:
    mapping: dict[str, dict[str, Any]]
    row_count: int = 0
    time_unit: str | None = None
    min_trade_time_ms: int | None = None
    max_trade_time_ms: int | None = None
    previous_trade_time_ms: int | None = None
    previous_agg_trade_id: int | None = None
    trade_time_sorted_violations: int = 0
    duplicate_agg_trade_id_count: int = 0
    agg_trade_id_decrease_count: int = 0
    agg_trade_id_gap_count: int = 0
    agg_trade_id_max_gap: int = 0
    nonpositive_price_count: int = 0
    nonpositive_quantity_count: int = 0
    is_buyer_maker_values: Counter = field(default_factory=Counter)
    is_best_match_values: Counter = field(default_factory=Counter)
    minute_base_volume: defaultdict[int, float] = field(default_factory=lambda: defaultdict(float))

    def column(self, logical: str) -> str | None:
        return self.mapping.get(logical, {}).get("column")

    def process(self, chunk: pd.DataFrame) -> None:
        self.row_count += len(chunk)
        time_col = self.column("trade_time") or self.column("trade_time_utc")
        quantity_col = self.column("quantity")
        if time_col:
            trade_ms, unit = series_to_epoch_ms(chunk[time_col], self.time_unit)
            self.time_unit = self.time_unit or unit
            valid = trade_ms.dropna().astype("int64").to_numpy()
            if valid.size:
                chunk_min = int(valid.min())
                chunk_max = int(valid.max())
                self.min_trade_time_ms = chunk_min if self.min_trade_time_ms is None else min(self.min_trade_time_ms, chunk_min)
                self.max_trade_time_ms = chunk_max if self.max_trade_time_ms is None else max(self.max_trade_time_ms, chunk_max)
                sorted_values = valid
                if self.previous_trade_time_ms is not None:
                    sorted_values = np.insert(sorted_values, 0, self.previous_trade_time_ms)
                self.trade_time_sorted_violations += int((np.diff(sorted_values) < 0).sum())
                self.previous_trade_time_ms = int(valid[-1])

            if quantity_col and valid.size:
                quantities = pd.to_numeric(chunk.loc[trade_ms.notna(), quantity_col], errors="coerce")
                minutes = (trade_ms.dropna().astype("int64") // MINUTE_MS) * MINUTE_MS
                grouped = pd.DataFrame({"minute": minutes.to_numpy(), "quantity": quantities.to_numpy()}).dropna()
                if not grouped.empty:
                    sums = grouped.groupby("minute", sort=False)["quantity"].sum()
                    for minute, volume in sums.items():
                        self.minute_base_volume[int(minute)] += float(volume)

        id_col = self.column("agg_trade_id")
        if id_col:
            ids = pd.to_numeric(chunk[id_col], errors="coerce").dropna().astype("int64").to_numpy()
            if ids.size:
                id_values = ids
                if self.previous_agg_trade_id is not None:
                    id_values = np.insert(id_values, 0, self.previous_agg_trade_id)
                diffs = np.diff(id_values)
                self.duplicate_agg_trade_id_count += int((diffs == 0).sum())
                self.agg_trade_id_decrease_count += int((diffs < 0).sum())
                gaps = diffs[diffs > 1]
                self.agg_trade_id_gap_count += int(gaps.size)
                if gaps.size:
                    self.agg_trade_id_max_gap = max(self.agg_trade_id_max_gap, int(gaps.max()))
                self.previous_agg_trade_id = int(ids[-1])

        price_col = self.column("price")
        if price_col:
            prices = pd.to_numeric(chunk[price_col], errors="coerce")
            self.nonpositive_price_count += int((prices <= 0).sum())

        quantity_col = self.column("quantity")
        if quantity_col:
            quantities = pd.to_numeric(chunk[quantity_col], errors="coerce")
            self.nonpositive_quantity_count += int((quantities <= 0).sum())

        buyer_maker_col = self.column("is_buyer_maker")
        if buyer_maker_col:
            self.is_buyer_maker_values.update(chunk[buyer_maker_col].astype(str).value_counts().to_dict())

        best_match_col = self.column("is_best_match")
        if best_match_col:
            self.is_best_match_values.update(chunk[best_match_col].astype(str).value_counts().to_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "primary_time_unit": self.time_unit,
            "time_range_utc": {
                "start": ms_to_utc(self.min_trade_time_ms) if self.min_trade_time_ms is not None else None,
                "end": ms_to_utc(self.max_trade_time_ms) if self.max_trade_time_ms is not None else None,
            },
            "is_sorted_by_trade_time": self.trade_time_sorted_violations == 0,
            "trade_time_sorted_violations": self.trade_time_sorted_violations,
            "duplicate_agg_trade_id_count": self.duplicate_agg_trade_id_count,
            "is_agg_trade_id_non_decreasing": self.agg_trade_id_decrease_count == 0,
            "agg_trade_id_decrease_count": self.agg_trade_id_decrease_count,
            "agg_trade_id_gap_count": self.agg_trade_id_gap_count,
            "agg_trade_id_max_gap": self.agg_trade_id_max_gap,
            "nonpositive_price_count": self.nonpositive_price_count,
            "nonpositive_quantity_count": self.nonpositive_quantity_count,
            "is_buyer_maker_values": dict(self.is_buyer_maker_values),
            "is_best_match_values": dict(self.is_best_match_values),
            "aggregated_minute_count": len(self.minute_base_volume),
        }


def ms_to_utc(ms: int | None) -> str | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(int(ms) / 1_000, tz=timezone.utc).isoformat()


def audit_csv(path: Path, role: str, config: dict[str, Any]) -> dict[str, Any]:
    chunk_size = int(config["audit"]["chunk_size"])
    threshold = int(config["audit"]["unique_exact_threshold"])
    max_examples = int(config["audit"]["max_examples"])
    logging.info("Auditing %s: %s", role, path)
    reader = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, chunksize=chunk_size)
    columns: list[str] | None = None
    column_stats: dict[str, ColumnAudit] = {}
    head_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    checker: KlineChecks | AggTradeChecks | None = None
    row_count = 0

    for chunk_index, chunk in enumerate(reader, start=1):
        if columns is None:
            columns = list(chunk.columns)
            mapping = infer_field_mapping(role, columns)
            column_stats = {col: ColumnAudit(col, threshold) for col in columns}
            checker = KlineChecks(mapping, max_examples) if role == "klines" else AggTradeChecks(mapping)
            head_rows = chunk.head(5).to_dict(orient="records")
            logging.info("%s columns: %s", role, columns)

        row_count += len(chunk)
        tail_rows = chunk.tail(5).to_dict(orient="records")
        for col in columns:
            column_stats[col].update(chunk[col])
        if checker is not None:
            checker.process(chunk)
        if chunk_index % 10 == 0:
            logging.info("%s processed rows: %s", role, row_count)

    if columns is None:
        raise ValueError(f"CSV has no readable header or rows: {path}")

    fields = [column_stats[col].as_dict() for col in columns]
    time_fields = detect_time_fields(fields)
    return {
        "role": role,
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "row_count": row_count,
        "columns": columns,
        "head_rows": head_rows,
        "tail_rows": tail_rows,
        "fields": fields,
        "time_fields": time_fields,
        "field_mapping": mapping,
        "checks": checker.as_dict() if checker else {},
        "_checker": checker,
    }


def compare_agg_volume_to_kline(
    klines_path: Path,
    kline_mapping: dict[str, dict[str, Any]],
    agg_minute_volume: dict[int, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    open_time_col = kline_mapping.get("open_time", {}).get("column") or kline_mapping.get("open_time_utc", {}).get("column")
    volume_col = kline_mapping.get("volume", {}).get("column")
    if not open_time_col or not volume_col or not agg_minute_volume:
        return {
            "status": "skipped",
            "reason": "缺少K线open_time/volume映射或agg trades分钟成交量为空",
        }

    chunk_size = int(config["audit"]["chunk_size"])
    agg_minutes = set(int(k) for k in agg_minute_volume.keys())
    kline_volumes: dict[int, float] = {}
    duplicate_kline_minutes = 0
    time_unit: str | None = None
    reader = pd.read_csv(klines_path, dtype=str, keep_default_na=False, na_filter=False, chunksize=chunk_size)
    for chunk in reader:
        open_ms, unit = series_to_epoch_ms(chunk[open_time_col], time_unit)
        time_unit = time_unit or unit
        minutes = ((open_ms.dropna().astype("int64") // MINUTE_MS) * MINUTE_MS).to_numpy()
        volumes = pd.to_numeric(chunk.loc[open_ms.notna(), volume_col], errors="coerce").to_numpy()
        for minute, volume in zip(minutes, volumes, strict=False):
            minute_int = int(minute)
            if minute_int not in agg_minutes or pd.isna(volume):
                continue
            if minute_int in kline_volumes:
                duplicate_kline_minutes += 1
            kline_volumes[minute_int] = float(volume)

    matched_minutes = sorted(agg_minutes & set(kline_volumes.keys()))
    if not matched_minutes:
        return {
            "status": "no_overlap",
            "agg_minute_count": len(agg_minutes),
            "kline_matched_minute_count": len(kline_volumes),
        }

    records = []
    for minute in matched_minutes:
        agg_volume = float(agg_minute_volume[minute])
        kline_volume = float(kline_volumes[minute])
        signed_diff = agg_volume - kline_volume
        abs_diff = abs(signed_diff)
        rel_diff = abs_diff / abs(kline_volume) if kline_volume else None
        records.append(
            {
                "minute_utc": ms_to_utc(minute),
                "agg_base_volume": agg_volume,
                "kline_volume": kline_volume,
                "signed_diff": signed_diff,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
            }
        )

    def describe(values: list[float]) -> dict[str, float | None]:
        clean = np.array([v for v in values if v is not None and not math.isnan(v)], dtype=float)
        if clean.size == 0:
            return {"min": None, "p50": None, "mean": None, "p90": None, "p99": None, "max": None}
        return {
            "min": float(np.min(clean)),
            "p50": float(np.percentile(clean, 50)),
            "mean": float(np.mean(clean)),
            "p90": float(np.percentile(clean, 90)),
            "p99": float(np.percentile(clean, 99)),
            "max": float(np.max(clean)),
        }

    sample_n = int(config["audit"]["volume_compare_sample_rows"])
    largest = sorted(records, key=lambda item: item["abs_diff"], reverse=True)[:sample_n]
    first = records[:sample_n]
    return {
        "status": "completed",
        "assumption": "按币安标准语义，agg_trades.quantity 与 K线 volume 均视为基础资产成交量。",
        "agg_minute_count": len(agg_minutes),
        "matched_minute_count": len(matched_minutes),
        "agg_minutes_without_kline": len(agg_minutes - set(kline_volumes.keys())),
        "duplicate_kline_minutes_in_overlap": duplicate_kline_minutes,
        "signed_diff_distribution": describe([r["signed_diff"] for r in records]),
        "abs_diff_distribution": describe([r["abs_diff"] for r in records]),
        "rel_diff_distribution": describe([r["rel_diff"] for r in records if r["rel_diff"] is not None]),
        "first_samples": first,
        "largest_abs_diff_samples": largest,
    }


def compute_overlap(klines: dict[str, Any], agg: dict[str, Any]) -> dict[str, Any]:
    k_range = klines["checks"].get("time_range_utc", {})
    a_range = agg["checks"].get("time_range_utc", {})
    if not all([k_range.get("start"), k_range.get("end"), a_range.get("start"), a_range.get("end")]):
        return {"status": "unknown"}

    k_start = pd.Timestamp(k_range["start"]).to_pydatetime()
    k_end = pd.Timestamp(k_range["end"]).to_pydatetime()
    a_start = pd.Timestamp(a_range["start"]).to_pydatetime()
    a_end = pd.Timestamp(a_range["end"]).to_pydatetime()
    overlap_start = max(k_start, a_start)
    overlap_end = min(k_end, a_end)
    has_overlap = overlap_start <= overlap_end
    seconds = (overlap_end - overlap_start).total_seconds() if has_overlap else 0
    return {
        "status": "completed",
        "has_overlap": has_overlap,
        "overlap_start_utc": overlap_start.isoformat() if has_overlap else None,
        "overlap_end_utc": overlap_end.isoformat() if has_overlap else None,
        "overlap_duration_seconds": seconds,
        "overlap_duration_hours": seconds / 3600 if has_overlap else 0,
    }


def collect_project_info(root: Path) -> dict[str, Any]:
    top_level = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        top_level.append(
            {
                "path": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size_bytes": child.stat().st_size if child.is_file() else None,
            }
        )

    dependency_patterns = {
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "environment.yml",
        "Pipfile",
        "poetry.lock",
    }
    dependency_files = []
    for path in root.rglob("*"):
        if path.name in dependency_patterns:
            dependency_files.append(str(path.relative_to(root)))

    imports: dict[str, list[str]] = {}
    for py_file in root.rglob("*.py"):
        if any(part in {".venv", "__pycache__"} for part in py_file.parts):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - report-only fallback
            imports[str(py_file.relative_to(root))] = [f"parse_error: {exc}"]
            continue
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module.split(".")[0])
        imports[str(py_file.relative_to(root))] = sorted(modules)

    return {
        "top_level": top_level,
        "dependency_files": dependency_files,
        "python_imports": imports,
        "dependency_assessment": (
            "未发现requirements.txt/pyproject.toml等依赖清单；当前只能从现有.py文件推断依赖。"
            if not dependency_files
            else "发现依赖清单文件。"
        ),
    }


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_无数据_\n"
    escaped_rows = []
    for row in rows:
        escaped = {}
        for col in columns:
            value = row.get(col, "")
            text = "" if value is None else str(value)
            escaped[col] = text.replace("|", "\\|").replace("\n", " ")
        escaped_rows.append(escaped)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(row[col] for col in columns) + " |" for row in escaped_rows]
    return "\n".join([header, separator, *body]) + "\n"


def field_table(fields: list[dict[str, Any]]) -> str:
    rows = []
    for field_info in fields:
        unique = field_info["unique_count"]
        if not field_info["unique_count_is_exact"]:
            unique = f"~{unique}"
        rows.append(
            {
                "原始列名": field_info["original_name"],
                "推测含义": field_info["inferred_meaning"],
                "数据类型": field_info["data_type"],
                "缺失值数量": field_info["missing_count"],
                "唯一值数量": unique,
                "最小值": field_info["min"],
                "最大值": field_info["max"],
            }
        )
    return markdown_table(rows, ["原始列名", "推测含义", "数据类型", "缺失值数量", "唯一值数量", "最小值", "最大值"])


def mapping_table(mapping: dict[str, dict[str, Any]]) -> str:
    rows = [
        {
            "逻辑字段": logical,
            "原始列": info.get("column"),
            "含义": info.get("meaning"),
            "置信度": info.get("confidence"),
        }
        for logical, info in mapping.items()
    ]
    return markdown_table(rows, ["逻辑字段", "原始列", "含义", "置信度"])


def rows_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_无数据_\n"
    columns = list(rows[0].keys())
    return markdown_table(rows, columns)


def write_markdown_report(schema: dict[str, Any], path: Path) -> None:
    klines = schema["datasets"]["klines"]
    agg = schema["datasets"]["agg_trades"]
    volume_cmp = schema["quality_checks"]["volume_comparison"]
    overlap = schema["quality_checks"]["time_overlap"]
    assumptions = schema["assumptions"]

    lines: list[str] = []
    lines.append("# 阶段0：原始数据审计报告\n")
    lines.append(f"- 生成时间(UTC): `{schema['generated_at_utc']}`")
    lines.append("- 范围限制: 只审计原始CSV；未清洗数据、未生成特征、未创建标签、未训练模型。")
    lines.append("- 唯一值说明: 表中带 `~` 的唯一值为HyperLogLog近似值；未带 `~` 的为精确值。")
    lines.append("")

    lines.append("## 输入文件\n")
    lines.append(
        markdown_table(
            [
                {
                    "数据集": "klines",
                    "路径": klines["path"],
                    "行数": klines["row_count"],
                    "大小(bytes)": klines["file_size_bytes"],
                },
                {
                    "数据集": "agg_trades",
                    "路径": agg["path"],
                    "行数": agg["row_count"],
                    "大小(bytes)": agg["file_size_bytes"],
                },
            ],
            ["数据集", "路径", "行数", "大小(bytes)"],
        )
    )

    for dataset in (klines, agg):
        lines.append(f"## {dataset['role']} 表头与样本\n")
        lines.append(f"- 表头: `{', '.join(dataset['columns'])}`")
        lines.append("\n前5行:\n")
        lines.append(rows_table(dataset["head_rows"]))
        lines.append("\n后5行:\n")
        lines.append(rows_table(dataset["tail_rows"]))
        lines.append(f"\n## {dataset['role']} 字段审计\n")
        lines.append(field_table(dataset["fields"]))
        lines.append(f"\n## {dataset['role']} 字段映射\n")
        lines.append(mapping_table(dataset["field_mapping"]))
        lines.append(f"\n## {dataset['role']} 时间字段识别\n")
        lines.append(markdown_table(dataset["time_fields"], ["field", "unit", "min_utc", "max_utc", "basis"]))

    k_checks = klines["checks"]
    lines.append("## K线数据质量检查\n")
    lines.append(
        markdown_table(
            [
                {"检查项": "是否按open_time排序", "结果": k_checks["is_sorted_by_open_time"], "细节": f"违规数={k_checks['sorted_violations']}"},
                {"检查项": "重复open_time", "结果": k_checks["duplicate_open_time_count"] == 0, "细节": f"重复数={k_checks['duplicate_open_time_count']}"},
                {"检查项": "缺失分钟", "结果": k_checks["missing_minutes"] == 0, "细节": f"缺失分钟数={k_checks['missing_minutes']}"},
                {"检查项": "非正价格", "结果": not k_checks["has_nonpositive_prices"], "细节": k_checks["nonpositive_price_counts"]},
                {"检查项": "负成交量", "结果": not k_checks["has_negative_volume"], "细节": k_checks["negative_volume_counts"]},
                {"检查项": "OHLC关系", "结果": not k_checks["has_ohlc_violations"], "细节": f"违规行数={k_checks['ohlc_violation_count']}"},
                {"检查项": "1分钟间隔稳定", "结果": k_checks["is_interval_stable_1m"], "细节": f"非1分钟间隔数={k_checks['non_1m_interval_count']}"},
                {"检查项": "close_time-open_time=59999ms", "结果": k_checks["close_time_span_violation_count"] == 0, "细节": f"违规数={k_checks['close_time_span_violation_count']}"},
            ],
            ["检查项", "结果", "细节"],
        )
    )
    lines.append("K线时间范围(UTC): ")
    lines.append(f"- start: `{k_checks['time_range_utc']['start']}`")
    lines.append(f"- end: `{k_checks['time_range_utc']['end']}`")
    lines.append("\nK线时间间隔Top值:\n")
    lines.append(markdown_table(k_checks["interval_counts_top"], ["interval_ms", "count"]))
    if k_checks["gap_examples"]:
        lines.append("\n缺失分钟示例:\n")
        lines.append(markdown_table(k_checks["gap_examples"], ["previous_open_time_utc", "next_open_time_utc", "interval_ms", "missing_minutes"]))

    a_checks = agg["checks"]
    lines.append("## agg trades 数据质量检查\n")
    lines.append(
        markdown_table(
            [
                {"检查项": "是否按trade_time排序", "结果": a_checks["is_sorted_by_trade_time"], "细节": f"违规数={a_checks['trade_time_sorted_violations']}"},
                {"检查项": "重复agg_trade_id", "结果": a_checks["duplicate_agg_trade_id_count"] == 0, "细节": f"重复数={a_checks['duplicate_agg_trade_id_count']}"},
                {"检查项": "价格为正", "结果": a_checks["nonpositive_price_count"] == 0, "细节": f"非正价格数={a_checks['nonpositive_price_count']}"},
                {"检查项": "数量为正", "结果": a_checks["nonpositive_quantity_count"] == 0, "细节": f"非正数量数={a_checks['nonpositive_quantity_count']}"},
                {"检查项": "agg_trade_id基本单调", "结果": a_checks["is_agg_trade_id_non_decreasing"], "细节": f"下降数={a_checks['agg_trade_id_decrease_count']}; gap数={a_checks['agg_trade_id_gap_count']}; 最大gap={a_checks['agg_trade_id_max_gap']}"},
                {"检查项": "is_buyer_maker取值", "结果": "见细节", "细节": a_checks["is_buyer_maker_values"]},
            ],
            ["检查项", "结果", "细节"],
        )
    )
    lines.append("agg trades 时间范围(UTC): ")
    lines.append(f"- start: `{a_checks['time_range_utc']['start']}`")
    lines.append(f"- end: `{a_checks['time_range_utc']['end']}`")

    lines.append("\n## 两份数据时间范围重叠\n")
    lines.append(
        markdown_table(
            [
                {
                    "是否重叠": overlap.get("has_overlap"),
                    "重叠开始UTC": overlap.get("overlap_start_utc"),
                    "重叠结束UTC": overlap.get("overlap_end_utc"),
                    "重叠小时数": overlap.get("overlap_duration_hours"),
                }
            ],
            ["是否重叠", "重叠开始UTC", "重叠结束UTC", "重叠小时数"],
        )
    )

    lines.append("## agg trades按分钟汇总成交量 vs K线成交量\n")
    lines.append(f"- 状态: `{volume_cmp.get('status')}`")
    if volume_cmp.get("assumption"):
        lines.append(f"- 字段假设: {volume_cmp['assumption']}")
    lines.append(
        markdown_table(
            [
                {"指标": "agg分钟数", "值": volume_cmp.get("agg_minute_count")},
                {"指标": "匹配分钟数", "值": volume_cmp.get("matched_minute_count")},
                {"指标": "agg有但K线未匹配分钟数", "值": volume_cmp.get("agg_minutes_without_kline")},
                {"指标": "重叠K线重复分钟数", "值": volume_cmp.get("duplicate_kline_minutes_in_overlap")},
            ],
            ["指标", "值"],
        )
    )
    if volume_cmp.get("status") == "completed":
        lines.append("\n差异分布:\n")
        rows = []
        for name in ("signed_diff_distribution", "abs_diff_distribution", "rel_diff_distribution"):
            row = {"分布": name}
            row.update(volume_cmp[name])
            rows.append(row)
        lines.append(markdown_table(rows, ["分布", "min", "p50", "mean", "p90", "p99", "max"]))
        lines.append("\n前若干分钟抽样:\n")
        lines.append(markdown_table(volume_cmp["first_samples"], ["minute_utc", "agg_base_volume", "kline_volume", "signed_diff", "abs_diff", "rel_diff"]))
        lines.append("\n绝对差异最大的抽样:\n")
        lines.append(markdown_table(volume_cmp["largest_abs_diff_samples"], ["minute_utc", "agg_base_volume", "kline_volume", "signed_diff", "abs_diff", "rel_diff"]))

    lines.append("## 项目依赖和目录结构\n")
    project = schema["project"]
    lines.append("顶层目录:\n")
    lines.append(markdown_table(project["top_level"], ["path", "type", "size_bytes"]))
    lines.append(f"- 依赖清单: `{project['dependency_files']}`")
    lines.append(f"- 依赖判断: {project['dependency_assessment']}")
    lines.append("- Python import推断:")
    for file_name, imports in project["python_imports"].items():
        lines.append(f"  - `{file_name}`: {', '.join(imports) if imports else '(无)'}")

    lines.append("\n## 尚未确认的假设\n")
    for item in assumptions:
        lines.append(f"- {item}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_assumptions(root: Path, path_attempts: dict[str, list[str]], schema: dict[str, Any]) -> list[str]:
    assumptions = []
    for role, attempts in path_attempts.items():
        requested = root / f"data/raw/BTCUSDT_spot_{'klines_1m' if role == 'klines' else 'agg_trades'}.csv"
        actual = Path(schema["datasets"][role]["path"])
        if requested.resolve() != actual.resolve():
            assumptions.append(f"{role} 用户给出的直接路径不存在，实际使用候选路径 `{actual}`；尝试路径包括: {attempts}")
    assumptions.append("唯一值统计对高基数字段使用HyperLogLog近似，报告中以 `~` 标注。")
    assumptions.append("agg trades 的 `quantity` 按币安标准语义假设为基础资产成交量，用于和K线 `volume` 做分钟级审计对比。")
    assumptions.append("`is_buyer_maker` 按币安标准语义解释：False表示买方主动成交，True表示卖方主动成交；本阶段仅记录取值，不生成特征。")
    return assumptions


def strip_internal(schema: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(schema)
    serializable["datasets"] = {
        role: {key: value for key, value in dataset.items() if key != "_checker"}
        for role, dataset in schema.get("datasets", {}).items()
    }
    clean = json.loads(json.dumps(serializable, default=to_jsonable, ensure_ascii=False))
    for dataset in clean.get("datasets", {}).values():
        dataset.pop("_checker", None)
    return clean


def run_audit(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    output_cfg = config["outputs"]
    report_md = (root / output_cfg["report_md"]).resolve()
    schema_json = (root / output_cfg["schema_json"]).resolve()
    log_file = (root / output_cfg["log_file"]).resolve()
    report_md.parent.mkdir(parents=True, exist_ok=True)
    schema_json.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    logging.info("Stage 0 raw data audit started")
    path_attempts: dict[str, list[str]] = {}
    kline_path, path_attempts["klines"] = resolve_candidate_path(root, config["datasets"]["klines"]["candidate_paths"], "klines")
    agg_path, path_attempts["agg_trades"] = resolve_candidate_path(root, config["datasets"]["agg_trades"]["candidate_paths"], "agg_trades")

    klines = audit_csv(kline_path, "klines", config)
    agg = audit_csv(agg_path, "agg_trades", config)
    agg_checker = agg["_checker"]
    if not isinstance(agg_checker, AggTradeChecks):
        raise TypeError("Internal error: agg checker not available")

    volume_comparison = compare_agg_volume_to_kline(kline_path, klines["field_mapping"], dict(agg_checker.minute_base_volume), config)
    schema = {
        "generated_at_utc": utc_now_iso(),
        "scope": "stage0_raw_data_audit_only",
        "config_path": str(config_path.resolve()),
        "datasets": {
            "klines": klines,
            "agg_trades": agg,
        },
        "quality_checks": {
            "time_overlap": compute_overlap(klines, agg),
            "volume_comparison": volume_comparison,
        },
        "project": collect_project_info(root),
        "assumptions": [],
        "outputs": {
            "report_md": str(report_md),
            "schema_json": str(schema_json),
            "log_file": str(log_file),
        },
    }
    schema["assumptions"] = build_assumptions(root, path_attempts, schema)
    clean_schema = strip_internal(schema)
    schema_json.write_text(json.dumps(clean_schema, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(clean_schema, report_md)
    logging.info("Stage 0 raw data audit completed")
    return clean_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 0 raw CSV data audit for BTCUSDT project.")
    parser.add_argument("--config", default="config/stage0_data_audit.json", help="Path to stage0 audit config JSON.")
    parser.add_argument("--root", default=".", help="Project root path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_audit(config_path=config_path, root=root)


if __name__ == "__main__":
    main()

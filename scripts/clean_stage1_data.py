from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ms_to_utc_iso(ms: int | float | None) -> str | None:
    if ms is None or pd.isna(ms):
        return None
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").isoformat()


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def finite_mask(series: pd.Series) -> pd.Series:
    numeric = to_numeric(series)
    return pd.Series(np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan)), index=series.index)


def to_utc_iso_series(series: pd.Series) -> pd.Series:
    return pd.Series([ms_to_utc_iso(value) for value in series.tolist()], index=series.index, dtype="string")


def mapping_column(mapping: dict[str, Any], logical_name: str) -> str:
    column = mapping.get(logical_name, {}).get("column")
    if not column:
        raise KeyError(f"Missing required field mapping: {logical_name}")
    return column


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_无_\n"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def parse_bool_series(series: pd.Series) -> tuple[pd.Series, pd.Series, dict[str, int], list[str]]:
    text = series.astype("string").str.strip().str.lower()
    true_values = {"true", "1"}
    false_values = {"false", "0"}
    valid = text.isin(true_values | false_values)
    parsed = pd.Series(pd.NA, index=series.index, dtype="boolean")
    parsed.loc[text.isin(true_values)] = True
    parsed.loc[text.isin(false_values)] = False
    invalid_values = text[~valid].value_counts(dropna=False).head(20)
    invalid_counts = {str(k): int(v) for k, v in invalid_values.items()}
    examples = [str(v) for v in text[~valid].drop_duplicates().head(20).tolist()]
    return parsed, valid.fillna(False), invalid_counts, examples


def read_csv_arrow(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype_backend="pyarrow")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    df.to_parquet(path, engine="pyarrow", index=False)


def output_schema(path: Path) -> list[dict[str, str]]:
    import pyarrow.parquet as pq

    schema = pq.read_schema(path)
    return [{"name": field.name, "type": str(field.type)} for field in schema]


@dataclass
class KlineCleanResult:
    df: pd.DataFrame
    stats: dict[str, Any]
    gaps: pd.DataFrame
    close_time_repairs: pd.DataFrame


@dataclass
class AggCleanResult:
    df: pd.DataFrame
    stats: dict[str, Any]
    id_gaps: pd.DataFrame


def clean_klines_dataframe(
    raw: pd.DataFrame,
    mapping: dict[str, Any],
    expected_interval_ms: int,
    expected_duration_ms: int,
    repair_close_time: bool,
) -> KlineCleanResult:
    df = raw.copy(deep=False)
    original_hashable_columns = list(df.columns)
    df["_original_order"] = np.arange(len(df), dtype=np.int64)

    symbol_col = mapping.get("symbol", {}).get("column")
    interval_col = mapping.get("interval", {}).get("column")
    open_time_col = mapping_column(mapping, "open_time")
    close_time_col = mapping_column(mapping, "close_time")
    price_cols = [mapping_column(mapping, name) for name in ("open", "high", "low", "close")]
    open_col, high_col, low_col, close_col = price_cols
    volume_cols = [
        mapping_column(mapping, "volume"),
        mapping_column(mapping, "quote_volume"),
        mapping_column(mapping, "taker_buy_base_volume"),
        mapping_column(mapping, "number_of_trades"),
    ]
    optional_taker_quote = mapping.get("taker_buy_quote_volume", {}).get("column")
    if optional_taker_quote and optional_taker_quote in df.columns:
        volume_cols.append(optional_taker_quote)

    original_rows = len(df)
    df[open_time_col] = to_numeric(df[open_time_col])
    df[close_time_col] = to_numeric(df[close_time_col])
    missing_open_time_mask = df[open_time_col].isna()
    missing_open_time_deleted = int(missing_open_time_mask.sum())
    df = df.loc[~missing_open_time_mask].copy()

    duplicate_mask = df.duplicated(subset=[open_time_col], keep="last")
    duplicate_open_time_deleted = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    for col in price_cols + volume_cols:
        df[col] = to_numeric(df[col])

    open_v = df[open_col]
    high_v = df[high_col]
    low_v = df[low_col]
    close_v = df[close_col]

    nonfinite_price_mask = ~(
        finite_mask(open_v) & finite_mask(high_v) & finite_mask(low_v) & finite_mask(close_v)
    )
    nonpositive_price_mask = (open_v <= 0) | (high_v <= 0) | (low_v <= 0) | (close_v <= 0)
    nonfinite_volume_mask = pd.Series(False, index=df.index)
    negative_volume_mask = pd.Series(False, index=df.index)
    for col in volume_cols:
        col_numeric = df[col]
        nonfinite_volume_mask |= ~finite_mask(col_numeric)
        negative_volume_mask |= col_numeric < 0

    ohlc_violation_mask = (
        (high_v < np.maximum(open_v, close_v))
        | (low_v > np.minimum(open_v, close_v))
        | (high_v < low_v)
    )
    invalid_mask = nonfinite_price_mask | nonpositive_price_mask | nonfinite_volume_mask | negative_volume_mask | ohlc_violation_mask
    invalid_deleted = int(invalid_mask.sum())
    deletion_reasons = {
        "missing_open_time": missing_open_time_deleted,
        "duplicate_open_time_keep_last_original_order": duplicate_open_time_deleted,
        "nonfinite_price": int(nonfinite_price_mask.sum()),
        "nonpositive_price": int(nonpositive_price_mask.sum()),
        "nonfinite_volume_or_trade_count": int(nonfinite_volume_mask.sum()),
        "negative_volume_or_trade_count": int(negative_volume_mask.sum()),
        "ohlc_relation_violation": int(ohlc_violation_mask.sum()),
        "invalid_rows_after_dedup": invalid_deleted,
    }
    df = df.loc[~invalid_mask].copy()

    df[open_time_col] = df[open_time_col].astype("int64")
    df[close_time_col] = df[close_time_col].astype("Int64")
    for col in (open_col, high_col, low_col, close_col, *volume_cols):
        if col == mapping_column(mapping, "number_of_trades"):
            df[col] = df[col].astype("int64")
        else:
            df[col] = df[col].astype("float64")

    df.sort_values([open_time_col, "_original_order"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    expected_close = df[open_time_col] + expected_duration_ms
    close_time_before = df[close_time_col].astype("int64")
    repaired_mask = close_time_before != expected_close
    repair_records = pd.DataFrame(
        {
            "open_time": df.loc[repaired_mask, open_time_col].astype("int64"),
            "open_time_utc": [ms_to_utc_iso(v) for v in df.loc[repaired_mask, open_time_col].tolist()],
            "original_close_time": close_time_before.loc[repaired_mask].astype("int64"),
            "original_close_time_utc": [ms_to_utc_iso(v) for v in close_time_before.loc[repaired_mask].tolist()],
            "expected_close_time": expected_close.loc[repaired_mask].astype("int64"),
            "expected_close_time_utc": [ms_to_utc_iso(v) for v in expected_close.loc[repaired_mask].tolist()],
        }
    )
    if repair_close_time:
        df[close_time_col] = expected_close.astype("int64")
    df["expected_close_time"] = expected_close.astype("int64")
    df["close_time_repaired"] = repaired_mask.astype("bool")

    open_time_utc_col = mapping.get("open_time_utc", {}).get("column")
    close_time_utc_col = mapping.get("close_time_utc", {}).get("column")
    if open_time_utc_col and open_time_utc_col in df.columns:
        df[open_time_utc_col] = df[open_time_utc_col].astype("string")
    else:
        df["open_time_utc"] = to_utc_iso_series(df[open_time_col])
        open_time_utc_col = "open_time_utc"
    if close_time_utc_col and close_time_utc_col in df.columns:
        df[close_time_utc_col] = df[close_time_utc_col].astype("string")
        df.loc[repaired_mask, close_time_utc_col] = [ms_to_utc_iso(v) for v in df.loc[repaired_mask, close_time_col].tolist()]
    else:
        df["close_time_utc"] = to_utc_iso_series(df[close_time_col])
        close_time_utc_col = "close_time_utc"
    df["expected_close_time_utc"] = to_utc_iso_series(df["expected_close_time"])

    delta_ms = df[open_time_col].diff()
    new_segment = delta_ms.ne(expected_interval_ms)
    if len(new_segment):
        new_segment.iloc[0] = False
    df["continuity_segment_id"] = new_segment.cumsum().astype("int64")
    gap_before = np.where(delta_ms > expected_interval_ms, (delta_ms // expected_interval_ms) - 1, 0)
    df["gap_before_minutes"] = pd.Series(gap_before, index=df.index).fillna(0).astype("int64")

    gap_mask = new_segment
    previous_open = df[open_time_col].shift(1)
    previous_segment = df["continuity_segment_id"].shift(1)
    gaps = pd.DataFrame(
        {
            "previous_open_time": [ms_to_utc_iso(v) for v in previous_open.loc[gap_mask].tolist()],
            "current_open_time": [ms_to_utc_iso(v) for v in df.loc[gap_mask, open_time_col].tolist()],
            "delta_minutes": (delta_ms.loc[gap_mask] / expected_interval_ms).astype("float64"),
            "missing_minutes": df.loc[gap_mask, "gap_before_minutes"].astype("int64"),
            "previous_segment_id": previous_segment.loc[gap_mask].astype("int64"),
            "current_segment_id": df.loc[gap_mask, "continuity_segment_id"].astype("int64"),
        }
    )

    segment_sizes = df.groupby("continuity_segment_id", sort=True).size()
    longest_segment_id = int(segment_sizes.idxmax()) if not segment_sizes.empty else None
    longest_segment_rows = int(segment_sizes.max()) if not segment_sizes.empty else 0

    output_columns = [
        col for col in original_hashable_columns if col in df.columns
    ] + [
        "expected_close_time",
        "expected_close_time_utc",
        "close_time_repaired",
        "continuity_segment_id",
        "gap_before_minutes",
    ]
    output_columns = [col for i, col in enumerate(output_columns) if col not in output_columns[:i]]
    df = df[output_columns].copy()

    stats = {
        "input_rows": original_rows,
        "output_rows": len(df),
        "deleted_rows_total": original_rows - len(df),
        "deletion_reasons": deletion_reasons,
        "close_time_repaired_count": int(repaired_mask.sum()) if repair_close_time else 0,
        "close_time_repair_enabled": repair_close_time,
        "time_range_utc": {
            "start": ms_to_utc_iso(df[open_time_col].min()) if len(df) else None,
            "end": ms_to_utc_iso(df[open_time_col].max()) if len(df) else None,
        },
        "continuity_segment_count": int(df["continuity_segment_id"].nunique()) if len(df) else 0,
        "longest_segment_id": longest_segment_id,
        "longest_segment_rows": longest_segment_rows,
        "missing_minutes_total": int(df["gap_before_minutes"].sum()) if len(df) else 0,
        "gap_row_count": int(len(gaps)),
    }
    return KlineCleanResult(df=df, stats=stats, gaps=gaps, close_time_repairs=repair_records)


def clean_agg_trades_dataframe(
    raw: pd.DataFrame,
    mapping: dict[str, Any],
    long_no_trade_threshold_ms: int,
) -> AggCleanResult:
    df = raw.copy(deep=False)
    original_columns = list(df.columns)
    df["_original_order"] = np.arange(len(df), dtype=np.int64)

    agg_id_col = mapping_column(mapping, "agg_trade_id")
    price_col = mapping_column(mapping, "price")
    quantity_col = mapping_column(mapping, "quantity")
    trade_time_col = mapping_column(mapping, "trade_time")
    is_buyer_maker_col = mapping_column(mapping, "is_buyer_maker")

    original_rows = len(df)
    df[agg_id_col] = to_numeric(df[agg_id_col])
    df[trade_time_col] = to_numeric(df[trade_time_col])
    raw_valid_time = df[trade_time_col].dropna()
    raw_trade_time_backward_count = int((raw_valid_time.diff() < 0).sum())

    missing_agg_id_mask = df[agg_id_col].isna()
    missing_agg_id_deleted = int(missing_agg_id_mask.sum())
    df = df.loc[~missing_agg_id_mask].copy()

    duplicate_mask = df.duplicated(subset=[agg_id_col], keep="last")
    duplicate_id_deleted = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()

    df[price_col] = to_numeric(df[price_col])
    df[quantity_col] = to_numeric(df[quantity_col])
    parsed_bool, valid_bool_mask, invalid_bool_counts, invalid_bool_examples = parse_bool_series(df[is_buyer_maker_col])

    missing_trade_time_mask = df[trade_time_col].isna()
    nonfinite_price_mask = ~finite_mask(df[price_col])
    nonfinite_quantity_mask = ~finite_mask(df[quantity_col])
    nonpositive_price_mask = df[price_col] <= 0
    nonpositive_quantity_mask = df[quantity_col] <= 0
    invalid_bool_mask = ~valid_bool_mask
    invalid_mask = (
        missing_trade_time_mask
        | nonfinite_price_mask
        | nonfinite_quantity_mask
        | nonpositive_price_mask
        | nonpositive_quantity_mask
        | invalid_bool_mask
    )
    deletion_reasons = {
        "missing_agg_trade_id": missing_agg_id_deleted,
        "duplicate_agg_trade_id_keep_last_original_order": duplicate_id_deleted,
        "missing_trade_time": int(missing_trade_time_mask.sum()),
        "nonfinite_price": int(nonfinite_price_mask.sum()),
        "nonfinite_quantity": int(nonfinite_quantity_mask.sum()),
        "nonpositive_price": int(nonpositive_price_mask.sum()),
        "nonpositive_quantity": int(nonpositive_quantity_mask.sum()),
        "invalid_is_buyer_maker": int(invalid_bool_mask.sum()),
        "invalid_rows_after_dedup": int(invalid_mask.sum()),
    }
    df = df.loc[~invalid_mask].copy()
    parsed_bool = parsed_bool.loc[df.index]

    df[agg_id_col] = df[agg_id_col].astype("int64")
    df[trade_time_col] = df[trade_time_col].astype("int64")
    df[price_col] = df[price_col].astype("float64")
    df[quantity_col] = df[quantity_col].astype("float64")
    for logical in ("first_trade_id", "last_trade_id"):
        column = mapping.get(logical, {}).get("column")
        if column and column in df.columns:
            df[column] = to_numeric(df[column]).astype("Int64")

    df[is_buyer_maker_col] = parsed_bool.astype("bool")
    best_match_col = mapping.get("is_best_match", {}).get("column")
    if best_match_col and best_match_col in df.columns:
        best_match_parsed, best_match_valid, _, _ = parse_bool_series(df[best_match_col])
        df[best_match_col] = best_match_parsed.where(best_match_valid, pd.NA).astype("boolean")

    df.sort_values([trade_time_col, agg_id_col, "_original_order"], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    trade_time_utc_col = mapping.get("trade_time_utc", {}).get("column")
    if trade_time_utc_col and trade_time_utc_col in df.columns:
        df[trade_time_utc_col] = df[trade_time_utc_col].astype("string")
    else:
        df["trade_time_utc"] = to_utc_iso_series(df[trade_time_col])
        trade_time_utc_col = "trade_time_utc"

    df["quote_quantity"] = df[price_col] * df[quantity_col]
    df["is_active_buy"] = (~df[is_buyer_maker_col]).astype("bool")
    df["is_active_sell"] = df[is_buyer_maker_col].astype("bool")

    id_delta = df[agg_id_col].diff()
    id_gap_before = np.where(id_delta > 1, id_delta - 1, 0)
    df["id_gap_before"] = pd.Series(id_gap_before, index=df.index).fillna(0).astype("int64")
    id_gap_mask = df["id_gap_before"] > 0
    time_delta_ms = df[trade_time_col].diff()
    id_gaps = pd.DataFrame(
        {
            "previous_agg_trade_id": df[agg_id_col].shift(1).loc[id_gap_mask].astype("int64"),
            "current_agg_trade_id": df.loc[id_gap_mask, agg_id_col].astype("int64"),
            "missing_id_count": df.loc[id_gap_mask, "id_gap_before"].astype("int64"),
            "previous_trade_time": [ms_to_utc_iso(v) for v in df[trade_time_col].shift(1).loc[id_gap_mask].tolist()],
            "current_trade_time": [ms_to_utc_iso(v) for v in df.loc[id_gap_mask, trade_time_col].tolist()],
            "time_delta_ms": time_delta_ms.loc[id_gap_mask].astype("float64"),
        }
    )

    time_counts = df[trade_time_col].value_counts(sort=False)
    same_ms_counts = time_counts[time_counts > 1]
    long_no_trade_mask = time_delta_ms > long_no_trade_threshold_ms

    output_columns = [
        col for col in original_columns if col in df.columns
    ] + [
        "quote_quantity",
        "is_active_buy",
        "is_active_sell",
        "id_gap_before",
    ]
    output_columns = [col for i, col in enumerate(output_columns) if col not in output_columns[:i]]
    df = df[output_columns].copy()

    stats = {
        "input_rows": original_rows,
        "output_rows": len(df),
        "deleted_rows_total": original_rows - len(df),
        "deletion_reasons": deletion_reasons,
        "invalid_is_buyer_maker_values": invalid_bool_counts,
        "invalid_is_buyer_maker_examples": invalid_bool_examples,
        "time_range_utc": {
            "start": ms_to_utc_iso(df[trade_time_col].min()) if len(df) else None,
            "end": ms_to_utc_iso(df[trade_time_col].max()) if len(df) else None,
        },
        "raw_trade_time_backward_count": raw_trade_time_backward_count,
        "post_sort_trade_time_backward_count": int((df[trade_time_col].diff() < 0).sum()) if len(df) else 0,
        "same_millisecond_group_count": int(len(same_ms_counts)),
        "same_millisecond_row_count": int(same_ms_counts.sum()) if len(same_ms_counts) else 0,
        "same_millisecond_max_trades": int(same_ms_counts.max()) if len(same_ms_counts) else 0,
        "long_no_trade_threshold_ms": long_no_trade_threshold_ms,
        "long_no_trade_interval_count": int(long_no_trade_mask.sum()),
        "long_no_trade_max_ms": float(time_delta_ms[long_no_trade_mask].max()) if long_no_trade_mask.any() else 0,
        "agg_trade_id_gap_count": int(len(id_gaps)),
        "agg_trade_id_missing_total": int(df["id_gap_before"].sum()) if len(df) else 0,
        "agg_trade_id_max_gap": int(df["id_gap_before"].max()) if len(df) else 0,
        "agg_trade_id_decrease_count": int((id_delta < 0).sum()),
    }
    return AggCleanResult(df=df, stats=stats, id_gaps=id_gaps)


def write_kline_report(
    path: Path,
    result: KlineCleanResult,
    output_path: Path,
    parquet_schema: list[dict[str, str]],
    file_size: int,
) -> None:
    stats = result.stats
    repair_rows = result.close_time_repairs.to_dict(orient="records")
    lines = [
        "# 阶段1：K线独立清洗报告",
        "",
        "- 范围限制: 只清洗K线，不合并agg trades，不生成特征、标签或切分。",
        f"- 输出Parquet: `{output_path}`",
        f"- 输出文件大小(bytes): `{file_size}`",
        "",
        "## 行数与时间范围",
        table(
            [
                {"指标": "输入行数", "值": stats["input_rows"]},
                {"指标": "输出行数", "值": stats["output_rows"]},
                {"指标": "总删除行数", "值": stats["deleted_rows_total"]},
                {"指标": "开始时间UTC", "值": stats["time_range_utc"]["start"]},
                {"指标": "结束时间UTC", "值": stats["time_range_utc"]["end"]},
            ],
            ["指标", "值"],
        ),
        "## 删除与修复统计",
        table([{"原因": k, "数量": v} for k, v in stats["deletion_reasons"].items()], ["原因", "数量"]),
        table(
            [
                {"指标": "close_time修复数量", "值": stats["close_time_repaired_count"]},
                {"指标": "连续片段数量", "值": stats["continuity_segment_count"]},
                {"指标": "最长连续片段ID", "值": stats["longest_segment_id"]},
                {"指标": "最长连续片段行数", "值": stats["longest_segment_rows"]},
                {"指标": "缺失分钟总数", "值": stats["missing_minutes_total"]},
                {"指标": "gap记录行数", "值": stats["gap_row_count"]},
            ],
            ["指标", "值"],
        ),
        "## close_time显式修复明细",
        table(
            repair_rows,
            [
                "open_time_utc",
                "original_close_time",
                "original_close_time_utc",
                "expected_close_time",
                "expected_close_time_utc",
            ],
        ),
        "## 输出Parquet Schema",
        table(parquet_schema, ["name", "type"]),
        "## 说明",
        "- 未插入缺失分钟K线。",
        "- 未对价格或成交量做前向填充、后向填充或插值。",
        "- `continuity_segment_id` 在open_time间隔不等于60秒时开启新片段。",
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_agg_report(
    path: Path,
    result: AggCleanResult,
    output_path: Path,
    parquet_schema: list[dict[str, str]],
    file_size: int,
) -> None:
    stats = result.stats
    lines = [
        "# 阶段1：agg trades独立清洗报告",
        "",
        "- 范围限制: 只清洗agg trades，不聚合到分钟，不合并K线，不生成滚动订单流特征、标签或模型。",
        f"- 输出Parquet: `{output_path}`",
        f"- 输出文件大小(bytes): `{file_size}`",
        "",
        "## 行数与时间范围",
        table(
            [
                {"指标": "输入行数", "值": stats["input_rows"]},
                {"指标": "输出行数", "值": stats["output_rows"]},
                {"指标": "总删除行数", "值": stats["deleted_rows_total"]},
                {"指标": "开始时间UTC", "值": stats["time_range_utc"]["start"]},
                {"指标": "结束时间UTC", "值": stats["time_range_utc"]["end"]},
            ],
            ["指标", "值"],
        ),
        "## 删除统计",
        table([{"原因": k, "数量": v} for k, v in stats["deletion_reasons"].items()], ["原因", "数量"]),
        "## is_buyer_maker解析",
        table(
            [
                {"指标": "非法值计数", "值": stats["invalid_is_buyer_maker_values"]},
                {"指标": "非法值样例", "值": stats["invalid_is_buyer_maker_examples"]},
            ],
            ["指标", "值"],
        ),
        "## 时间与ID连续性",
        table(
            [
                {"指标": "原始顺序成交时间倒退数", "值": stats["raw_trade_time_backward_count"]},
                {"指标": "排序后成交时间倒退数", "值": stats["post_sort_trade_time_backward_count"]},
                {"指标": "同一毫秒多笔成交的毫秒数", "值": stats["same_millisecond_group_count"]},
                {"指标": "同一毫秒多笔成交涉及行数", "值": stats["same_millisecond_row_count"]},
                {"指标": "同一毫秒最大成交笔数", "值": stats["same_millisecond_max_trades"]},
                {"指标": "超长无成交阈值(ms)", "值": stats["long_no_trade_threshold_ms"]},
                {"指标": "超长无成交间隔数", "值": stats["long_no_trade_interval_count"]},
                {"指标": "最大无成交间隔(ms)", "值": stats["long_no_trade_max_ms"]},
                {"指标": "ID gap记录数", "值": stats["agg_trade_id_gap_count"]},
                {"指标": "缺失ID总数", "值": stats["agg_trade_id_missing_total"]},
                {"指标": "最大ID gap缺失数", "值": stats["agg_trade_id_max_gap"]},
                {"指标": "ID倒退数", "值": stats["agg_trade_id_decrease_count"]},
            ],
            ["指标", "值"],
        ),
        "## 输出Parquet Schema",
        table(parquet_schema, ["name", "type"]),
        "## 说明",
        "- 未补造缺失成交ID。",
        "- 未聚合agg trades到分钟。",
        "- 未构造滚动订单流特征，未使用OFI命名。",
        "- `is_active_buy = not is_buyer_maker`; `is_active_sell = is_buyer_maker`。",
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_mappings(schema_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    schema = read_json(schema_path)
    return (
        schema["datasets"]["klines"]["field_mapping"],
        schema["datasets"]["agg_trades"]["field_mapping"],
    )


def run_stage1(config_path: Path, root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = read_json(config_path)
    cleaning_cfg = config["cleaning"]
    outputs = config["outputs"]

    log_path = (root / outputs["log_file"]).resolve()
    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    schema_path = (root / config["schema_path"]).resolve()
    kline_mapping, agg_mapping = load_mappings(schema_path)
    kline_input = (root / config["inputs"]["klines"]).resolve()
    agg_input = (root / config["inputs"]["agg_trades"]).resolve()
    kline_output = (root / outputs["clean_klines_parquet"]).resolve()
    agg_output = (root / outputs["clean_agg_trades_parquet"]).resolve()
    kline_gaps_output = (root / outputs["kline_gaps_csv"]).resolve()
    agg_gaps_output = (root / outputs["agg_trade_id_gaps_csv"]).resolve()
    kline_report_output = (root / outputs["kline_report"]).resolve()
    agg_report_output = (root / outputs["agg_trades_report"]).resolve()

    logging.info("Stage 1 cleaning started")
    logging.info("Reading klines input: %s", kline_input)
    raw_klines = read_csv_arrow(kline_input)
    kline_result = clean_klines_dataframe(
        raw=raw_klines,
        mapping=kline_mapping,
        expected_interval_ms=int(cleaning_cfg["expected_kline_interval_ms"]),
        expected_duration_ms=int(cleaning_cfg["expected_kline_duration_ms"]),
        repair_close_time=bool(cleaning_cfg["repair_close_time"]),
    )
    write_parquet(kline_result.df, kline_output)
    ensure_parent(kline_gaps_output)
    kline_result.gaps.to_csv(kline_gaps_output, index=False, encoding="utf-8")
    kline_schema = output_schema(kline_output)
    write_kline_report(
        kline_report_output,
        kline_result,
        kline_output,
        kline_schema,
        kline_output.stat().st_size,
    )
    logging.info("Klines input rows: %s", kline_result.stats["input_rows"])
    logging.info("Klines output rows: %s", kline_result.stats["output_rows"])
    logging.info("Klines deletions: %s", kline_result.stats["deletion_reasons"])
    logging.info("Kline time range: %s", kline_result.stats["time_range_utc"])
    logging.info("Kline continuity segment count: %s", kline_result.stats["continuity_segment_count"])
    logging.info("Kline missing minutes: %s", kline_result.stats["missing_minutes_total"])
    logging.info("Kline close_time repaired: %s", kline_result.stats["close_time_repaired_count"])
    logging.info("Kline output path: %s", kline_output)

    kline_stats = kline_result.stats
    kline_file_size = kline_output.stat().st_size
    del raw_klines, kline_result
    gc.collect()

    logging.info("Reading agg trades input: %s", agg_input)
    raw_agg = read_csv_arrow(agg_input)
    agg_result = clean_agg_trades_dataframe(
        raw=raw_agg,
        mapping=agg_mapping,
        long_no_trade_threshold_ms=int(cleaning_cfg["agg_long_no_trade_threshold_ms"]),
    )
    write_parquet(agg_result.df, agg_output)
    ensure_parent(agg_gaps_output)
    agg_result.id_gaps.to_csv(agg_gaps_output, index=False, encoding="utf-8")
    agg_schema = output_schema(agg_output)
    write_agg_report(
        agg_report_output,
        agg_result,
        agg_output,
        agg_schema,
        agg_output.stat().st_size,
    )
    logging.info("Agg trades input rows: %s", agg_result.stats["input_rows"])
    logging.info("Agg trades output rows: %s", agg_result.stats["output_rows"])
    logging.info("Agg trades deletions: %s", agg_result.stats["deletion_reasons"])
    logging.info("Agg trades time range: %s", agg_result.stats["time_range_utc"])
    logging.info("Agg trade ID gap count: %s", agg_result.stats["agg_trade_id_gap_count"])
    logging.info("Agg trade max ID gap missing count: %s", agg_result.stats["agg_trade_id_max_gap"])
    logging.info("Agg trades output path: %s", agg_output)

    agg_stats = agg_result.stats
    agg_file_size = agg_output.stat().st_size
    del raw_agg, agg_result
    gc.collect()

    elapsed = time.perf_counter() - started
    logging.info("Stage 1 cleaning completed in %.2f seconds", elapsed)
    return {
        "kline": {
            "stats": kline_stats,
            "output": str(kline_output),
            "gaps_csv": str(kline_gaps_output),
            "report": str(kline_report_output),
            "schema": kline_schema,
            "file_size_bytes": kline_file_size,
        },
        "agg_trades": {
            "stats": agg_stats,
            "output": str(agg_output),
            "id_gaps_csv": str(agg_gaps_output),
            "report": str(agg_report_output),
            "schema": agg_schema,
            "file_size_bytes": agg_file_size,
        },
        "elapsed_seconds": elapsed,
        "log_file": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 independent cleaning for raw BTCUSDT klines and agg trades.")
    parser.add_argument("--config", default="config/stage1_clean_data.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage1(config_path=config_path, root=root)


if __name__ == "__main__":
    main()

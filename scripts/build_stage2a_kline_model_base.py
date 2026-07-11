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


class Stage2AValidationError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("\n".join(errors))
        self.errors = errors


@dataclass
class Stage2AResult:
    df: pd.DataFrame
    stats: dict[str, Any]
    segment_summary: pd.DataFrame
    candidate_summary_by_month: pd.DataFrame
    candidate_summary_by_year: pd.DataFrame


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


def validate_input_schema(input_path: Path) -> tuple[list[str], list[str]]:
    schema = pq.read_schema(input_path)
    columns = schema.names
    required = [
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
        "continuity_segment_id",
        "gap_before_minutes",
        "close_time_repaired",
    ]
    missing = [col for col in required if col not in columns]
    if missing:
        raise Stage2AValidationError([f"Input Parquet missing required columns: {missing}"])
    selected = list(required)
    optional = []
    if "taker_buy_quote_volume" in columns:
        selected.insert(selected.index("continuity_segment_id"), "taker_buy_quote_volume")
        optional.append("taker_buy_quote_volume")
    return selected, optional


def read_kline_input(input_path: Path) -> pd.DataFrame:
    selected, _ = validate_input_schema(input_path)
    return pd.read_parquet(input_path, columns=selected)


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    return pd.Series(np.isfinite(values) & (values > 0), index=series.index)


def finite_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", copy=False)
    return pd.Series(np.isfinite(values) & (values >= 0), index=series.index)


def within_upper_bound(lhs: pd.Series, rhs: pd.Series, abs_tol: float, rel_tol: float) -> pd.Series:
    lhs_values = pd.to_numeric(lhs, errors="coerce").to_numpy(dtype="float64", copy=False)
    rhs_values = pd.to_numeric(rhs, errors="coerce").to_numpy(dtype="float64", copy=False)
    tolerance = abs_tol + rel_tol * np.abs(rhs_values)
    return pd.Series(lhs_values <= rhs_values + tolerance, index=lhs.index)


def first_bad_indices(mask: pd.Series, limit: int = 5) -> list[int]:
    return [int(i) for i in mask[~mask].index[:limit].tolist()]


def validate_clean_kline_input(
    df: pd.DataFrame,
    expected_interval_ms: int,
    abs_tol: float,
    rel_tol: float,
) -> None:
    errors: list[str] = []
    if df.empty:
        errors.append("Input Kline DataFrame is empty")
        raise Stage2AValidationError(errors)

    open_time = pd.to_numeric(df["open_time"], errors="coerce")
    if open_time.isna().any():
        errors.append(f"open_time contains null/non-numeric values; examples={first_bad_indices(open_time.notna())}")
    open_values = open_time.to_numpy(dtype="int64", copy=False)
    open_diffs = np.diff(open_values)
    if not np.all(open_diffs > 0):
        errors.append("open_time is not strictly increasing")
    if not df["open_time"].is_unique:
        errors.append("open_time is not unique")

    segment = pd.to_numeric(df["continuity_segment_id"], errors="coerce")
    if segment.isna().any():
        errors.append("continuity_segment_id contains null/non-numeric values")
    segment_values = segment.to_numpy(dtype="int64", copy=False)
    segment_diffs = np.diff(segment_values)
    if not np.all(segment_diffs >= 0):
        errors.append("continuity_segment_id is not non-decreasing")
    segment_changes = segment_diffs != 0
    if len(open_diffs):
        internal_bad = (~segment_changes) & (open_diffs != expected_interval_ms)
        boundary_bad = segment_changes & (open_diffs == expected_interval_ms)
        if np.any(internal_bad):
            first = int(np.flatnonzero(internal_bad)[0] + 1)
            errors.append(
                f"Segment internal open_time interval is not {expected_interval_ms} ms; first_bad_row={first}"
            )
        if np.any(boundary_bad):
            first = int(np.flatnonzero(boundary_bad)[0] + 1)
            errors.append("Segment boundary has a 60000 ms delta; first_bad_row={first}".format(first=first))

    for col in ("open", "high", "low", "close"):
        ok = finite_positive(df[col])
        if not ok.all():
            errors.append(f"{col} must be finite and > 0; bad_indices={first_bad_indices(ok)}")
    for col in ("volume", "quote_volume", "taker_buy_base_volume"):
        ok = finite_nonnegative(df[col])
        if not ok.all():
            errors.append(f"{col} must be finite and >= 0; bad_indices={first_bad_indices(ok)}")
    trades_ok = finite_nonnegative(df["number_of_trades"])
    if not trades_ok.all():
        errors.append(f"number_of_trades must be finite and >= 0; bad_indices={first_bad_indices(trades_ok)}")

    taker_base_ok = within_upper_bound(df["taker_buy_base_volume"], df["volume"], abs_tol, rel_tol)
    if not taker_base_ok.all():
        errors.append(
            "taker_buy_base_volume is greater than volume beyond tolerance; "
            f"bad_indices={first_bad_indices(taker_base_ok)}"
        )
    if "taker_buy_quote_volume" in df.columns:
        taker_quote_nonnegative = finite_nonnegative(df["taker_buy_quote_volume"])
        if not taker_quote_nonnegative.all():
            errors.append(
                "taker_buy_quote_volume must be finite and >= 0; "
                f"bad_indices={first_bad_indices(taker_quote_nonnegative)}"
            )
        taker_quote_ok = within_upper_bound(df["taker_buy_quote_volume"], df["quote_volume"], abs_tol, rel_tol)
        if not taker_quote_ok.all():
            errors.append(
                "taker_buy_quote_volume is greater than quote_volume beyond tolerance; "
                f"bad_indices={first_bad_indices(taker_quote_ok)}"
            )

    if errors:
        raise Stage2AValidationError(errors)


def build_model_base_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> Stage2AResult:
    expected_interval_ms = int(config["expected_interval_ms"])
    prediction_interval_minutes = int(config["prediction_interval_minutes"])
    history_windows = [int(v) for v in config["history_windows_minutes"]]
    max_history = int(config["maximum_history_minutes"])
    future_windows = [int(v) for v in config["future_windows_minutes"]]
    future_required = int(config["future_required_minutes"])
    abs_tol = float(config["floating_absolute_tolerance"])
    rel_tol = float(config["floating_relative_tolerance"])

    validate_clean_kline_input(df, expected_interval_ms, abs_tol, rel_tol)

    result = df.copy(deep=False)
    open_time = result["open_time"].to_numpy(dtype="int64", copy=False)
    segment = result["continuity_segment_id"].to_numpy(dtype="int64", copy=False)
    feature_time = open_time + expected_interval_ms

    segment_group = result.groupby("continuity_segment_id", sort=False, observed=True)
    result["segment_row_number"] = segment_group.cumcount().astype("int64")
    result["segment_length"] = segment_group["open_time"].transform("size").astype("int64")
    result["minutes_since_segment_start"] = result["segment_row_number"].astype("int64")
    result["minutes_until_segment_end"] = (result["segment_length"] - result["segment_row_number"] - 1).astype("int64")
    result["is_segment_start"] = (result["segment_row_number"] == 0).astype("bool")
    result["is_segment_end"] = (result["minutes_until_segment_end"] == 0).astype("bool")
    result["feature_time"] = feature_time.astype("int64")
    result["decision_time"] = feature_time.astype("int64")

    for window in history_windows:
        result[f"has_history_{window}m"] = (result["segment_row_number"] >= window - 1).astype("bool")
    for window in future_windows:
        result[f"has_future_{window}m"] = (result["minutes_until_segment_end"] >= window).astype("bool")

    prediction_interval_ms = prediction_interval_minutes * expected_interval_ms
    result["is_prediction_time_5m"] = ((result["decision_time"] % prediction_interval_ms) == 0).astype("bool")
    result["is_model_candidate"] = (
        result["is_prediction_time_5m"]
        & result[f"has_history_{max_history}m"]
        & result[f"has_future_{future_required}m"]
    ).astype("bool")

    output_columns = [
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
    ]
    if "taker_buy_quote_volume" in result.columns:
        output_columns.append("taker_buy_quote_volume")
    output_columns.extend(
        [
            "continuity_segment_id",
            "gap_before_minutes",
            "close_time_repaired",
            "feature_time",
            "decision_time",
            "segment_row_number",
            "segment_length",
            "minutes_since_segment_start",
            "minutes_until_segment_end",
            "is_segment_start",
            "is_segment_end",
            *[f"has_history_{window}m" for window in history_windows],
            *[f"has_future_{window}m" for window in future_windows],
            "is_prediction_time_5m",
            "is_model_candidate",
        ]
    )
    result = result[output_columns].copy()

    segment_summary = make_segment_summary(result, expected_interval_ms)
    by_month, by_year = make_candidate_summaries(result, max_history, future_required)
    stats = make_stats(result, segment_summary, by_year, max_history, future_required)
    return Stage2AResult(
        df=result,
        stats=stats,
        segment_summary=segment_summary,
        candidate_summary_by_month=by_month,
        candidate_summary_by_year=by_year,
    )


def make_segment_summary(df: pd.DataFrame, expected_interval_ms: int) -> pd.DataFrame:
    grouped = df.groupby("continuity_segment_id", sort=True, observed=True)
    summary = grouped.agg(
        segment_start_open_time=("open_time", "min"),
        segment_end_open_time=("open_time", "max"),
        segment_length=("open_time", "size"),
        repaired_close_time_count=("close_time_repaired", "sum"),
        prediction_time_count=("is_prediction_time_5m", "sum"),
        model_candidate_count=("is_model_candidate", "sum"),
    ).reset_index()
    summary["duration_minutes"] = (
        (summary["segment_end_open_time"] - summary["segment_start_open_time"]) // expected_interval_ms + 1
    ).astype("int64")
    columns = [
        "continuity_segment_id",
        "segment_start_open_time",
        "segment_end_open_time",
        "segment_length",
        "duration_minutes",
        "repaired_close_time_count",
        "prediction_time_count",
        "model_candidate_count",
    ]
    return summary[columns]


def make_candidate_summaries(
    df: pd.DataFrame,
    max_history: int,
    future_required: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_dt = pd.to_datetime(df["decision_time"], unit="ms", utc=True)
    tmp = pd.DataFrame(
        {
            "year": decision_dt.dt.year.astype("int64"),
            "month": decision_dt.dt.month.astype("int64"),
            "is_prediction_time_5m": df["is_prediction_time_5m"].to_numpy(dtype=bool, copy=False),
            "is_model_candidate": df["is_model_candidate"].to_numpy(dtype=bool, copy=False),
            "history_shortfall": (
                df["is_prediction_time_5m"] & ~df[f"has_history_{max_history}m"]
            ).to_numpy(dtype=bool, copy=False),
            "future_shortfall": (
                df["is_prediction_time_5m"] & ~df[f"has_future_{future_required}m"]
            ).to_numpy(dtype=bool, copy=False),
        }
    )
    value_columns = [
        "is_prediction_time_5m",
        "is_model_candidate",
        "history_shortfall",
        "future_shortfall",
    ]
    by_month = tmp.groupby(["year", "month"], sort=True)[value_columns].sum().reset_index()
    by_year = tmp.groupby(["year"], sort=True)[value_columns].sum().reset_index()
    return by_month, by_year


def make_stats(
    df: pd.DataFrame,
    segment_summary: pd.DataFrame,
    by_year: pd.DataFrame,
    max_history: int,
    future_required: int,
) -> dict[str, Any]:
    prediction = df["is_prediction_time_5m"]
    has_history = df[f"has_history_{max_history}m"]
    has_future = df[f"has_future_{future_required}m"]
    global_history = pd.Series(np.arange(len(df)) >= (max_history - 1), index=df.index)
    global_future = pd.Series((len(df) - 1 - np.arange(len(df))) >= future_required, index=df.index)
    segment_gap_excluded = prediction & global_history & global_future & (~has_history | ~has_future)

    history_counts = {
        col: int(df[col].sum())
        for col in df.columns
        if col.startswith("has_history_")
    }
    future_counts = {
        col: int(df[col].sum())
        for col in df.columns
        if col.startswith("has_future_")
    }
    return {
        "row_count": int(len(df)),
        "time_range_utc": {
            "open_time_start": ms_to_utc_iso(int(df["open_time"].min())),
            "open_time_end": ms_to_utc_iso(int(df["open_time"].max())),
            "decision_time_start": ms_to_utc_iso(int(df["decision_time"].min())),
            "decision_time_end": ms_to_utc_iso(int(df["decision_time"].max())),
        },
        "segment_count": int(df["continuity_segment_id"].nunique()),
        "history_available_counts": history_counts,
        "future_available_counts": future_counts,
        "prediction_time_count": int(prediction.sum()),
        "model_candidate_count": int(df["is_model_candidate"].sum()),
        "excluded_by_history_shortfall_prediction_times": int((prediction & ~has_history).sum()),
        "excluded_by_future_shortfall_prediction_times": int((prediction & ~has_future).sum()),
        "excluded_by_segment_gap_prediction_times": int(segment_gap_excluded.sum()),
        "segment_summary_rows": int(len(segment_summary)),
        "candidate_count_by_year": by_year.to_dict(orient="records"),
    }


def write_error_report(path: Path, errors: list[str]) -> None:
    ensure_parent(path)
    lines = [
        "# 阶段2A：K线建模基础表构建失败",
        "",
        "本阶段不得修复或删除输入记录；发现阶段1输出仍不满足约束，因此已停止。",
        "",
        "## 错误",
        *[f"- {error}" for error in errors],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    result: Stage2AResult,
    input_path: Path,
    output_path: Path,
    input_rows: int,
    output_schema: list[dict[str, str]],
    output_file_size: int,
    elapsed_seconds: float,
    python_peak_memory_bytes: int | None,
) -> None:
    stats = result.stats
    segment_report_rows = result.segment_summary.copy()
    segment_report_rows["segment_start_open_time_utc"] = segment_report_rows["segment_start_open_time"].map(ms_to_utc_iso)
    segment_report_rows["segment_end_open_time_utc"] = segment_report_rows["segment_end_open_time"].map(ms_to_utc_iso)
    lines = [
        "# 阶段2A：完整K线建模基础表与预测候选时刻报告",
        "",
        "- 范围限制: 只读取清洗后K线Parquet；未读取或处理agg trades；未生成技术指标、标签、切分或模型。",
        f"- 输入: `{input_path}`",
        f"- 输出: `{output_path}`",
        f"- 输出文件大小(bytes): `{output_file_size}`",
        f"- 运行耗时(seconds): `{elapsed_seconds:.2f}`",
        f"- Python tracemalloc峰值内存(bytes): `{python_peak_memory_bytes}`",
        "",
        "## 行数与时间范围",
        table(
            [
                {"指标": "输入行数", "值": input_rows},
                {"指标": "输出行数", "值": stats["row_count"]},
                {"指标": "输入open_time开始UTC", "值": stats["time_range_utc"]["open_time_start"]},
                {"指标": "输入open_time结束UTC", "值": stats["time_range_utc"]["open_time_end"]},
                {"指标": "decision_time开始UTC", "值": stats["time_range_utc"]["decision_time_start"]},
                {"指标": "decision_time结束UTC", "值": stats["time_range_utc"]["decision_time_end"]},
            ],
            ["指标", "值"],
        ),
        "## 总体统计",
        table(
            [
                {"指标": "连续片段数量", "值": stats["segment_count"]},
                {"指标": "5分钟预测时刻数量", "值": stats["prediction_time_count"]},
                {"指标": "is_model_candidate数量", "值": stats["model_candidate_count"]},
                {"指标": "因历史不足排除的预测时刻", "值": stats["excluded_by_history_shortfall_prediction_times"]},
                {"指标": "因未来不足排除的预测时刻", "值": stats["excluded_by_future_shortfall_prediction_times"]},
                {"指标": "因segment缺口影响排除的预测时刻", "值": stats["excluded_by_segment_gap_prediction_times"]},
            ],
            ["指标", "值"],
        ),
        "## 历史窗口可用行数",
        table([{"字段": k, "行数": v} for k, v in stats["history_available_counts"].items()], ["字段", "行数"]),
        "## 未来窗口可用行数",
        table([{"字段": k, "行数": v} for k, v in stats["future_available_counts"].items()], ["字段", "行数"]),
        "## Segment 汇总",
        table(
            segment_report_rows[
                [
                    "continuity_segment_id",
                    "segment_start_open_time_utc",
                    "segment_end_open_time_utc",
                    "segment_length",
                    "duration_minutes",
                    "repaired_close_time_count",
                    "prediction_time_count",
                    "model_candidate_count",
                ]
            ].to_dict(orient="records"),
            [
                "continuity_segment_id",
                "segment_start_open_time_utc",
                "segment_end_open_time_utc",
                "segment_length",
                "duration_minutes",
                "repaired_close_time_count",
                "prediction_time_count",
                "model_candidate_count",
            ],
        ),
        "## 按年份候选统计",
        table(result.candidate_summary_by_year.to_dict(orient="records"), list(result.candidate_summary_by_year.columns)),
        "## 输出Parquet Schema",
        table(output_schema, ["name", "type"]),
        "## 说明",
        "- `feature_time = open_time + 60000ms`，`decision_time = feature_time`。",
        "- `is_prediction_time_5m` 基于 decision_time 的毫秒整数模运算。",
        "- `is_model_candidate` 只表示时间结构、历史窗口和未来连续性候选，不代表标签有效或训练集样本。",
        "- 本阶段未统计上涨/下跌，因为没有生成标签。",
    ]
    ensure_parent(path)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stage2a(config_path: Path, root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    input_path = (root / config["input_path"]).resolve()
    output_path = (root / config["output_path"]).resolve()
    report_path = (root / config["report_path"]).resolve()
    log_path = (root / config["log_path"]).resolve()
    segment_summary_path = (root / config["segment_summary_path"]).resolve()
    by_month_path = (root / config["candidate_summary_by_month_path"]).resolve()
    by_year_path = (root / config["candidate_summary_by_year_path"]).resolve()

    ensure_parent(log_path)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    started = time.perf_counter()
    tracemalloc.start()
    logging.info("Stage 2A Kline model base build started")
    logging.info("Input path: %s", input_path)
    try:
        selected_columns, optional_columns = validate_input_schema(input_path)
        logging.info("Selected input columns: %s", selected_columns)
        logging.info("Optional columns retained: %s", optional_columns)
        raw = pd.read_parquet(input_path, columns=selected_columns)
        input_rows = int(len(raw))
        result = build_model_base_dataframe(raw, config)
        ensure_parent(output_path)
        result.df.to_parquet(output_path, engine="pyarrow", index=False, compression=config["parquet_compression"])
        ensure_parent(segment_summary_path)
        result.segment_summary.to_csv(segment_summary_path, index=False, encoding="utf-8")
        ensure_parent(by_month_path)
        result.candidate_summary_by_month.to_csv(by_month_path, index=False, encoding="utf-8")
        ensure_parent(by_year_path)
        result.candidate_summary_by_year.to_csv(by_year_path, index=False, encoding="utf-8")
        elapsed = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        output_file_size = output_path.stat().st_size
        schema = parquet_schema(output_path)
        write_report(
            report_path,
            result,
            input_path,
            output_path,
            input_rows,
            schema,
            output_file_size,
            elapsed,
            peak,
        )
        logging.info("Input rows: %s", input_rows)
        logging.info("Output rows: %s", result.stats["row_count"])
        logging.info("Time range: %s", result.stats["time_range_utc"])
        logging.info("Segment count: %s", result.stats["segment_count"])
        logging.info("Prediction time count: %s", result.stats["prediction_time_count"])
        logging.info("Model candidate count: %s", result.stats["model_candidate_count"])
        logging.info("History counts: %s", result.stats["history_available_counts"])
        logging.info("Future counts: %s", result.stats["future_available_counts"])
        logging.info("Output file: %s", output_path)
        logging.info("Output file size bytes: %s", output_file_size)
        logging.info("Elapsed seconds: %.2f", elapsed)
        logging.info("Python tracemalloc peak bytes: %s", peak)
        return {
            "stats": result.stats,
            "output_path": str(output_path),
            "report_path": str(report_path),
            "segment_summary_path": str(segment_summary_path),
            "candidate_summary_by_month_path": str(by_month_path),
            "candidate_summary_by_year_path": str(by_year_path),
            "schema": schema,
            "output_file_size_bytes": output_file_size,
            "elapsed_seconds": elapsed,
            "python_tracemalloc_peak_bytes": peak,
        }
    except Stage2AValidationError as exc:
        write_error_report(report_path, exc.errors)
        logging.error("Stage 2A validation failed: %s", exc.errors)
        raise
    finally:
        tracemalloc.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 2A Kline model base table.")
    parser.add_argument("--config", default="config/stage2a_kline_model_base.json")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    config_path = (root / args.config).resolve()
    run_stage2a(config_path=config_path, root=root)


if __name__ == "__main__":
    main()

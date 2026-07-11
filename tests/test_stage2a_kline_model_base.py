from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_stage2a_kline_model_base import (  # noqa: E402
    Stage2AValidationError,
    build_model_base_dataframe,
    run_stage2a,
)


BASE_CONFIG = {
    "expected_interval_ms": 60000,
    "prediction_interval_minutes": 5,
    "history_windows_minutes": [5, 15, 30, 60, 120, 240],
    "maximum_history_minutes": 240,
    "future_windows_minutes": [1, 60, 61],
    "future_required_minutes": 61,
    "floating_absolute_tolerance": 1e-10,
    "floating_relative_tolerance": 1e-9,
    "parquet_compression": "snappy",
}


def test_feature_and_decision_time_and_prediction_modulo() -> None:
    base = int(pd.Timestamp("2022-01-01T10:04:00Z").timestamp() * 1000)
    df = make_kline_df([base, base + 60_000], [0, 0])
    result = build_model_base_dataframe(df, BASE_CONFIG).df

    assert result.loc[0, "feature_time"] == base + 60_000
    assert result.loc[0, "decision_time"] == base + 60_000
    assert bool(result.loc[0, "is_prediction_time_5m"]) is True
    assert bool(result.loc[1, "is_prediction_time_5m"]) is False


def test_history_future_and_model_candidate_boundaries() -> None:
    base = int(pd.Timestamp("2022-01-01T00:00:00Z").timestamp() * 1000)
    open_times = [base + i * 60_000 for i in range(305)]
    df = make_kline_df(open_times, [0] * len(open_times))
    result = build_model_base_dataframe(df, BASE_CONFIG).df

    assert bool(result.loc[238, "has_history_240m"]) is False
    assert bool(result.loc[239, "has_history_240m"]) is True
    assert bool(result.loc[243, "has_future_61m"]) is True
    assert bool(result.loc[244, "has_future_61m"]) is False
    assert bool(result.loc[239, "is_prediction_time_5m"]) is True
    assert bool(result.loc[239, "is_model_candidate"]) is True
    assert bool(result.loc[244, "is_prediction_time_5m"]) is True
    assert bool(result.loc[244, "is_model_candidate"]) is False


def test_segment_fields_reset_and_future_does_not_cross_segment() -> None:
    base = int(pd.Timestamp("2022-01-01T00:00:00Z").timestamp() * 1000)
    first = [base + i * 60_000 for i in range(10)]
    second_base = base + 20 * 60_000
    second = [second_base + i * 60_000 for i in range(300)]
    df = make_kline_df(first + second, [0] * len(first) + [1] * len(second))
    result = build_model_base_dataframe(df, BASE_CONFIG).df

    assert result.loc[0, "segment_row_number"] == 0
    assert result.loc[9, "segment_row_number"] == 9
    assert result.loc[10, "segment_row_number"] == 0
    assert result.loc[0, "segment_length"] == 10
    assert result.loc[10, "segment_length"] == 300
    assert result.loc[0, "minutes_until_segment_end"] == 9
    assert bool(result.loc[0, "is_segment_start"]) is True
    assert bool(result.loc[9, "is_segment_end"]) is True
    assert bool(result.loc[9, "has_future_1m"]) is False
    assert bool(result.loc[9, "has_future_61m"]) is False
    assert bool(result.loc[15, "has_history_15m"]) is False


def test_invalid_segment_internal_interval_fails() -> None:
    base = int(pd.Timestamp("2022-01-01T00:00:00Z").timestamp() * 1000)
    df = make_kline_df([base, base + 120_000], [0, 0])
    with pytest.raises(Stage2AValidationError, match="Segment internal"):
        build_model_base_dataframe(df, BASE_CONFIG)


def test_taker_buy_base_volume_greater_than_volume_fails() -> None:
    base = int(pd.Timestamp("2022-01-01T00:00:00Z").timestamp() * 1000)
    df = make_kline_df([base, base + 60_000], [0, 0])
    df.loc[1, "taker_buy_base_volume"] = 2.0
    with pytest.raises(Stage2AValidationError, match="taker_buy_base_volume"):
        build_model_base_dataframe(df, BASE_CONFIG)


def test_run_stage2a_output_excludes_iso_columns_and_input_unchanged(tmp_path: Path) -> None:
    base = int(pd.Timestamp("2022-01-01T00:00:00Z").timestamp() * 1000)
    open_times = [base + i * 60_000 for i in range(305)]
    input_path = tmp_path / "input.parquet"
    make_kline_df(open_times, [0] * len(open_times)).to_parquet(input_path, index=False)
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()

    config = {
        **BASE_CONFIG,
        "input_path": "input.parquet",
        "output_path": "out/model_base.parquet",
        "report_path": "reports/report.md",
        "log_path": "reports/run.log",
        "segment_summary_path": "reports/segments.csv",
        "candidate_summary_by_month_path": "reports/month.csv",
        "candidate_summary_by_year_path": "reports/year.csv",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_stage2a(config_path=config_path, root=tmp_path)
    after = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output = pd.read_parquet(tmp_path / "out/model_base.parquet")

    assert before == after
    assert "open_time_utc" not in output.columns
    assert "close_time_utc" not in output.columns
    assert not any(col.endswith("_utc") for col in output.columns)
    assert output["open_time"].is_monotonic_increasing
    assert output["open_time"].is_unique


def make_kline_df(open_times: list[int], segment_ids: list[int]) -> pd.DataFrame:
    n = len(open_times)
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "interval": ["1m"] * n,
            "open_time": np.array(open_times, dtype=np.int64),
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.5),
            "volume": np.full(n, 1.0),
            "quote_volume": np.full(n, 100.0),
            "number_of_trades": np.ones(n, dtype=np.int64),
            "taker_buy_base_volume": np.full(n, 0.5),
            "taker_buy_quote_volume": np.full(n, 50.0),
            "continuity_segment_id": np.array(segment_ids, dtype=np.int64),
            "gap_before_minutes": np.zeros(n, dtype=np.int64),
            "close_time_repaired": np.zeros(n, dtype=bool),
        }
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_stage3b_full_labels import (  # noqa: E402
    Stage3BValidationError,
    build_full_labels,
    make_field_dictionary,
    make_margin_bucket,
    run_stage3b,
)


CONFIG = {
    "expected_interval_ms": 60000,
    "prediction_interval_minutes": 5,
    "horizon_minutes": 60,
    "return_margin_bins_bps": [0, 1, 2.5, 5, 10, 25, 50],
    "proxy_boundary_thresholds_bps": [1, 2.5, 5, 10],
    "sample_weight_settings": {
        "margin_full_weight_bps": 10,
        "minimum_margin_weight": 0.25,
    },
    "floating_absolute_tolerance": 1e-10,
    "floating_relative_tolerance": 1e-9,
    "parquet_compression": "snappy",
    "extreme_return_rows": 10,
}


def test_time_mapping_uses_exact_time_keys_not_row_offsets() -> None:
    base = 1_662_076_800_000
    frame = pd.DataFrame(
        [
            row(base - 60_000, 50, 0, True, True, True, True),
            row(base, 100, 0, False, True, True, True),
            row(base + 120_000, 999, 0, False, True, True, True),
            row(base + 3_600_000, 101, 0, False, True, True, True),
        ]
    )

    result = build_full_labels(frame, CONFIG).df
    first = result.iloc[0]

    assert len(result) == 1
    assert first["feature_open_time"] == base - 60_000
    assert first["decision_time"] == base
    assert first["entry_minute_open_time"] == base
    assert first["settlement_minute_open_time"] == base + 3_600_000
    assert first["entry_price_proxy"] == 100
    assert first["settlement_price_proxy"] == 101
    assert first["future_simple_return_60m"] == pytest.approx(0.01)
    assert first["future_log_return_60m"] == pytest.approx(math.log(1.01))
    assert int(first["label_up_60m"]) == 1
    assert bool(first["is_valid_proxy_label"]) is True
    assert bool(first["is_final_model_label_candidate"]) is True


def test_invalid_labels_stay_nullable_for_missing_price_and_cross_segment() -> None:
    base = 1_662_076_800_000
    missing_settlement = base + 300_000
    cross_segment = base + 600_000
    frame = pd.DataFrame(
        [
            row(missing_settlement - 60_000, 50, 0, True, True, False, False),
            row(missing_settlement, 100, 0, False, True, True, False),
            row(cross_segment - 60_000, 50, 0, True, True, False, False),
            row(cross_segment, 100, 0, False, True, True, False),
            row(cross_segment + 3_600_000, 101, 1, False, True, True, False),
        ]
    )

    result = build_full_labels(frame, CONFIG).df.sort_values("decision_time").reset_index(drop=True)

    assert len(result) == 2
    assert bool(result.loc[0, "has_entry_price_proxy"]) is True
    assert bool(result.loc[0, "has_settlement_price_proxy"]) is False
    assert bool(result.loc[0, "is_valid_proxy_label"]) is False
    assert pd.isna(result.loc[0, "label_up_60m"])
    assert bool(result.loc[1, "has_settlement_price_proxy"]) is True
    assert bool(result.loc[1, "same_continuity_segment"]) is False
    assert bool(result.loc[1, "is_valid_proxy_label"]) is False
    assert pd.isna(result.loc[1, "future_simple_return_60m"])
    assert pd.isna(result.loc[1, "label_up_60m"])


def test_has_future_61m_mismatch_fails_instead_of_silent_handling() -> None:
    base = 1_662_076_800_000
    frame = pd.DataFrame(
        [
            row(base - 60_000, 50, 0, True, True, True, True),
            row(base, 100, 0, False, True, True, True),
        ]
    )

    with pytest.raises(Stage3BValidationError, match="has_future_61m"):
        build_full_labels(frame, CONFIG)


def test_equal_down_up_returns_bps_buckets_and_sample_weights() -> None:
    base = 1_662_076_800_000
    exact_5bps = base + 300_000
    exact_2_5bps_down = base + 600_000
    equal_price = base + 900_000
    frame = pd.DataFrame(
        [
            row(exact_5bps - 60_000, 50, 0, True, True, True, True),
            row(exact_5bps, 100, 0, False, True, True, True),
            row(exact_5bps + 3_600_000, 100.05, 0, False, True, True, True),
            row(exact_2_5bps_down - 60_000, 50, 0, True, True, True, True),
            row(exact_2_5bps_down, 100, 0, False, True, True, True),
            row(exact_2_5bps_down + 3_600_000, 99.975, 0, False, True, True, True),
            row(equal_price - 60_000, 50, 0, True, True, True, True),
            row(equal_price, 100, 0, False, True, True, True),
            row(equal_price + 3_600_000, 100, 0, False, True, True, True),
        ]
    ).sort_values("open_time", ignore_index=True)

    result = build_full_labels(frame, CONFIG).df.sort_values("decision_time").reset_index(drop=True)

    assert result["label_up_60m"].astype("int8").tolist() == [1, 0, 0]
    assert result.loc[0, "absolute_future_return_bps"] == pytest.approx(5.0)
    assert result.loc[0, "proxy_margin_bucket"] == "[5, 10)"
    assert bool(result.loc[0, "proxy_boundary_risk_5bps"]) is False
    assert bool(result.loc[0, "proxy_boundary_risk_10bps"]) is True
    assert result.loc[0, "sample_weight_margin"] == pytest.approx(0.5)
    assert result.loc[1, "absolute_future_return_bps"] == pytest.approx(2.5)
    assert result.loc[1, "proxy_margin_bucket"] == "[2.5, 5)"
    assert bool(result.loc[1, "proxy_boundary_risk_2_5bps"]) is False
    assert bool(result.loc[1, "proxy_boundary_risk_5bps"]) is True
    assert result.loc[2, "proxy_margin_bucket"] == "[0, 1)"
    assert result.loc[2, "sample_weight_margin"] == pytest.approx(0.25)
    assert (result["sample_weight_margin"].between(0.25, 1.0)).all()


def test_margin_bucket_boundaries() -> None:
    buckets = make_margin_bucket(pd.Series([0, 0.999, 1, 2.5, 5, 10, 25, 50]), [0, 1, 2.5, 5, 10, 25, 50])
    assert buckets.astype(str).tolist() == [
        "[0, 1)",
        "[0, 1)",
        "[1, 2.5)",
        "[2.5, 5)",
        "[5, 10)",
        "[10, 25)",
        "[25, 50)",
        "[50, +inf)",
    ]


def test_field_dictionary_marks_label_and_future_metadata_as_not_model_inputs() -> None:
    field_dictionary = make_field_dictionary(CONFIG)
    fields = {item["name"]: item for item in field_dictionary["fields"]}
    forbidden = [
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
        "sample_weight_margin",
    ]
    for name in forbidden:
        assert fields[name]["allowed_as_model_input"] is False
    assert fields["label_up_60m"]["is_model_label"] is True
    assert fields["entry_price_proxy"]["definition"].startswith("Kline open engineering proxy")
    assert fields["sample_weight_margin"]["may_use_future_data"] is True


def test_run_stage3b_outputs_only_prediction_times_no_model_features_and_preserves_input(tmp_path: Path) -> None:
    base = 1_662_076_800_000
    frame = pd.DataFrame(
        [
            row(base - 60_000, 50, 0, True, True, True, True),
            row(base, 100, 0, False, True, True, True),
            row(base + 60_000, 500, 0, False, True, True, True),
            row(base + 3_600_000, 101, 0, False, True, True, True),
        ]
    )
    input_path = tmp_path / "model_base.parquet"
    frame.to_parquet(input_path, index=False)
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    config = {
        **CONFIG,
        "input_path": "model_base.parquet",
        "output_path": "out/labels.parquet",
        "log_path": "reports/log.txt",
        "report_paths": {
            "main_report": "reports/main.md",
            "distribution_by_year": "reports/year.csv",
            "distribution_by_month": "reports/month.csv",
            "distribution_by_segment": "reports/segment.csv",
            "distribution_by_margin": "reports/margin.csv",
            "invalid_label_candidates": "reports/invalid.csv",
            "extreme_return_samples": "reports/extreme.csv",
            "field_dictionary": "reports/fields.json",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_stage3b(config_path, tmp_path)
    output = pd.read_parquet(tmp_path / "out/labels.parquet")

    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == before
    assert len(output) == 1
    assert output["is_prediction_time_5m"].all()
    assert not any(token in col.lower() for col in output.columns for token in ["ema", "atr", "momentum", "volatility"])
    assert (tmp_path / "reports/main.md").exists()
    assert (tmp_path / "reports/fields.json").exists()


def row(
    open_time: int,
    open_price: float,
    segment_id: int,
    prediction_time: bool,
    has_history: bool,
    has_future: bool,
    model_candidate: bool,
) -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": open_price,
        "continuity_segment_id": segment_id,
        "decision_time": open_time + 60_000,
        "is_prediction_time_5m": prediction_time,
        "has_history_240m": has_history,
        "has_future_61m": has_future,
        "is_model_candidate": model_candidate,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

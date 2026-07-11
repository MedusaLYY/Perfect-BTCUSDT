from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_stage3a_label_proxy import (  # noqa: E402
    build_label_proxy_comparison,
    make_margin_bucket,
    run_stage3a,
    wilson_interval,
)


CONFIG = {
    "prediction_interval_minutes": 5,
    "horizon_minutes": 60,
    "vwap_windows_seconds": [1, 2, 5, 10],
    "primary_vwap_window_seconds": 5,
    "return_margin_bins_bps": [0, 1, 2.5, 5, 10, 25, 50],
    "reliability_requirements": {
        "require_entry_reliable_overlap": True,
        "require_settlement_reliable_overlap": True,
        "exclude_entry_boundary_partial_minute": True,
        "exclude_settlement_boundary_partial_minute": True,
        "require_entry_primary_vwap": True,
        "require_settlement_primary_vwap": True,
        "require_kline_open_prices": True,
        "require_exact_horizon_minutes": True,
    },
    "acceptance_thresholds": {
        "minimum_primary_reliable_samples": 1,
        "minimum_overall_label_agreement": 0.0,
        "minimum_non_boundary_agreement": 0.0,
        "minimum_agreement_when_abs_return_ge_5bps": 0.0,
        "maximum_missing_rate_for_5s_vwap": 1.0,
        "maximum_absolute_mean_return_bias_bps": 999.0,
    },
    "epsilon": 1e-12,
    "floating_absolute_tolerance": 1e-10,
    "floating_relative_tolerance": 1e-9,
    "parquet_compression": "snappy",
    "largest_difference_rows": 10,
}


def test_time_mapping_exact_join_and_labels() -> None:
    base = 1_662_076_800_000
    model = pd.DataFrame(
        [
            model_row(base - 60_000, base, True, True),
            model_row(base + 60_000, base + 120_000, True, True),
        ]
    )
    overlap = pd.DataFrame(
        [
            minute_row(base, 100, 100, 100, 100, 100),
            minute_row(base + 120_000, 999, 999, 999, 999, 999),
            minute_row(base + 3_600_000, 101, 101, 101, 101, 102),
            minute_row(base + 3_720_000, 998, 998, 998, 998, 998),
        ]
    )
    result = build_label_proxy_comparison(model, overlap[["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"]], overlap, CONFIG).df
    first = result.iloc[0]

    assert first["feature_open_time"] == base - 60_000
    assert first["decision_time"] == base
    assert first["entry_minute_open_time"] == base
    assert first["settlement_minute_open_time"] == base + 3_600_000
    assert first["settlement_minute_open_time"] - first["entry_minute_open_time"] == 3_600_000
    assert first["entry_price_kline_open"] == 100
    assert first["settlement_price_kline_open"] == 101
    assert first["return_kline_open"] == pytest.approx(0.01)
    assert int(first["label_kline_open"]) == 1
    assert int(first["label_vwap_5s"]) == 1
    assert result.iloc[1]["entry_price_kline_open"] == 999


def test_missing_intermediate_minutes_do_not_break_time_join() -> None:
    base = 1_662_076_800_000
    model = pd.DataFrame([model_row(base - 60_000, base, True, True)])
    overlap = pd.DataFrame([minute_row(base, 100, 100, 100, 100, 100), minute_row(base + 3_600_000, 101, 101, 101, 101, 101)])
    result = build_label_proxy_comparison(model, overlap[["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"]], overlap, CONFIG).df
    assert len(result) == 1
    assert bool(result.loc[result.index[0], "is_label_only_candidate"]) is True


def test_missing_vwap_stays_missing_and_equal_price_label_zero() -> None:
    base = 1_662_076_800_000
    model = pd.DataFrame([model_row(base - 60_000, base, True, True)])
    overlap = pd.DataFrame(
        [
            minute_row(base, 100, None, 100, 100, 100),
            minute_row(base + 3_600_000, 100, 100, None, 100, 100),
        ]
    )
    result = build_label_proxy_comparison(model, overlap[["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"]], overlap, CONFIG).df.iloc[0]
    assert int(result["label_kline_open"]) == 0
    assert bool(result["is_valid_vwap_1s_label"]) is False
    assert bool(result["is_valid_vwap_2s_label"]) is False
    assert pd.isna(result["label_vwap_1s"])
    assert pd.isna(result["label_vwap_2s"])


def test_flip_fields_and_bps_direction_and_primary_reliable() -> None:
    base = 1_662_076_800_000
    model = pd.DataFrame([model_row(base - 60_000, base, True, True)])
    overlap = pd.DataFrame(
        [
            minute_row(base, 100, 100, 100, 110, 100),
            minute_row(base + 3_600_000, 101, 101, 101, 100, 101),
        ]
    )
    result = build_label_proxy_comparison(model, overlap[["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"]], overlap, CONFIG).df.iloc[0]
    assert int(result["label_kline_open"]) == 1
    assert int(result["label_vwap_5s"]) == 0
    assert bool(result["label_flip_kline_vs_5s"]) is True
    assert result["entry_price_difference_bps_kline_vs_5s"] == pytest.approx((100 / 110 - 1) * 10000)
    assert bool(result["is_primary_reliable_sample"]) is True


def test_margin_bucket_boundaries_and_wilson() -> None:
    buckets = make_margin_bucket(pd.Series([0, 0.999, 1, 2.5, 5, 10, 50]), [0, 1, 2.5, 5, 10, 25, 50]).astype(str).tolist()
    assert buckets == ["[0, 1)", "[0, 1)", "[1, 2.5)", "[2.5, 5)", "[5, 10)", "[10, 25)", "[50, +inf)"]
    lo, hi = wilson_interval(50, 100)
    assert lo == pytest.approx(0.4038315303659956)
    assert hi == pytest.approx(0.5961684696340044)


def test_run_stage3a_outputs_no_model_features_and_inputs_unchanged(tmp_path: Path) -> None:
    base = 1_662_076_800_000
    model = pd.DataFrame([model_row(base - 60_000, base, True, True)])
    overlap = pd.DataFrame([minute_row(base, 100, 100, 100, 100, 100), minute_row(base + 3_600_000, 101, 101, 101, 101, 101)])
    agg = overlap[["open_time", "vwap_first_1s", "vwap_first_2s", "vwap_first_5s", "vwap_first_10s"]]
    model_path = tmp_path / "model.parquet"
    agg_path = tmp_path / "agg.parquet"
    overlap_path = tmp_path / "overlap.parquet"
    model.to_parquet(model_path, index=False)
    agg.to_parquet(agg_path, index=False)
    overlap.to_parquet(overlap_path, index=False)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [model_path, agg_path, overlap_path]}
    cfg = {
        **CONFIG,
        "model_base_input_path": "model.parquet",
        "agg_1m_input_path": "agg.parquet",
        "overlap_input_path": "overlap.parquet",
        "output_path": "out/result.parquet",
        "log_path": "reports/log.txt",
        "report_paths": {
            "main_report": "reports/main.md",
            "agreement_by_margin": "reports/margin.csv",
            "agreement_by_hour": "reports/hour.csv",
            "agreement_by_date": "reports/date.csv",
            "label_flip_samples": "reports/flips.csv",
            "largest_return_differences": "reports/diffs.csv",
            "field_dictionary": "reports/fields.json",
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_stage3a(cfg_path, tmp_path)
    output = pd.read_parquet(tmp_path / "out/result.parquet")
    assert not any(token in c.lower() for c in output.columns for token in ["ema", "atr", "momentum", "volatility"])
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[p] for p in before)


def model_row(feature_open: int, decision: int, prediction: bool, model_candidate: bool) -> dict[str, object]:
    return {
        "open_time": feature_open,
        "decision_time": decision,
        "is_prediction_time_5m": prediction,
        "is_model_candidate": model_candidate,
        "has_future_61m": True,
    }


def minute_row(open_time: int, kline_open: float, v1: float | None, v2: float | None, v5: float | None, v10: float | None) -> dict[str, object]:
    return {
        "open_time": open_time,
        "kline_open": kline_open,
        "vwap_first_1s": v1,
        "vwap_first_2s": v2,
        "vwap_first_5s": v5,
        "vwap_first_10s": v10,
        "is_reliable_overlap_minute": True,
        "is_boundary_partial_minute": False,
        "has_any_id_gap": False,
        "id_gap_event_count": 0,
        "cross_minute_id_gap_event_count": 0,
        "maximum_internal_id_gap": 0,
        "kline_base_volume": 10.0,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

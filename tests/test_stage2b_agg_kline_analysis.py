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

from scripts.build_stage2b_agg_kline_analysis import (  # noqa: E402
    aggregate_agg_trades_1m,
    assign_trade_minute,
    build_overlap,
    run_stage2b,
)


CONFIG = {
    "expected_interval_ms": 60000,
    "execution_windows_seconds": {"first": [1, 2, 5, 10], "last": [1, 5]},
    "floating_absolute_tolerance": 1e-10,
    "floating_relative_tolerance": 1e-9,
    "epsilon": 1e-12,
    "warning_thresholds": {"volume_relative_error": 0.01, "price_error_bps": 1.0},
    "severe_thresholds": {"volume_relative_error": 0.05, "price_error_bps": 5.0},
    "reliability_rule": {
        "exclude_boundary_partial_minute": True,
        "require_both_sources": True,
        "max_base_volume_relative_error": 0.01,
        "max_quote_volume_relative_error": 0.01,
        "max_price_error_bps": 5.0,
        "require_first_5s_trade": True,
        "disallow_connection_anomaly": True,
    },
    "parquet_compression": "snappy",
    "largest_error_rows": 10,
}


def test_trade_minute_floor_and_boundary() -> None:
    base = (1_000_000_000_000 // 60_000) * 60_000
    s = pd.Series([base, base + 59_999, base + 60_000])
    out = assign_trade_minute(s, 60_000)
    assert out.iloc[0] == (base // 60_000) * 60_000
    assert out.iloc[1] == (base // 60_000) * 60_000
    assert out.iloc[2] == ((base + 60_000) // 60_000) * 60_000


def test_aggregation_values_windows_and_id_gaps() -> None:
    m0 = 1_662_076_800_000
    rows = [
        agg_row(1, 100, 1, m0 + 0, True),
        agg_row(2, 102, 2, m0 + 999, False),
        agg_row(4, 104, 1, m0 + 5_000, False),
        agg_row(5, 106, 1, m0 + 55_000, True),
        agg_row(8, 110, 1, m0 + 60_000, True),
        agg_row(9, 112, 1, m0 + 60_000 + 59_999, True),
    ]
    agg = pd.DataFrame(rows)
    out = aggregate_agg_trades_1m(agg, CONFIG)
    first = out.iloc[0]
    second = out.iloc[1]

    assert first["agg_trade_count"] == 4
    assert first["base_volume"] == 5
    assert first["quote_volume"] == 514
    assert first["active_buy_base_volume"] == 3
    assert first["active_sell_base_volume"] == 2
    assert first["trade_flow_imbalance_base"] == pytest.approx(0.2)
    assert first["trade_vwap"] == pytest.approx(514 / 5)
    assert pd.notna(first["active_buy_vwap"])
    assert first["first_1s_trade_count"] == 2
    assert first["first_5s_trade_count"] == 2
    assert first["first_10s_trade_count"] == 3
    assert first["last_5s_trade_count"] == 1
    assert first["id_gap_event_count"] == 1
    assert first["internal_missing_agg_trade_id_count"] == 1
    assert second["cross_minute_id_gap_event_count"] == 1
    assert second["cross_minute_missing_agg_trade_id_count"] == 2
    assert bool(first["is_file_first_minute"]) is True
    assert bool(second["is_file_last_minute"]) is True


def test_active_buy_vwap_missing_when_no_active_buy_and_same_ms_kept() -> None:
    m0 = 1_662_076_800_000
    agg = pd.DataFrame(
        [
            agg_row(1, 100, 1, m0, True),
            agg_row(2, 101, 1, m0, True),
        ]
    )
    out = aggregate_agg_trades_1m(agg, CONFIG)
    assert out.loc[0, "agg_trade_count"] == 2
    assert pd.isna(out.loc[0, "active_buy_vwap"])
    assert out.loc[0, "active_sell_vwap"] == pytest.approx(100.5)


def test_overlap_prefixes_errors_and_reliability() -> None:
    m0 = 1_662_076_800_000
    agg_1m = aggregate_agg_trades_1m(
        pd.DataFrame(
            [
                agg_row(1, 100, 1, m0, False),
                agg_row(2, 101, 1, m0 + 1_000, True),
                agg_row(3, 102, 1, m0 + 55_000, False),
                agg_row(4, 200, 1, m0 + 60_000, False),
            ]
        ),
        CONFIG,
    )
    kline = pd.DataFrame(
        [
            kline_row(m0, 100, 102, 100, 102, 3, 303, 3, 2, 202, 0),
            kline_row(m0 + 60_000, 200, 200, 200, 200, 1, 200, 1, 1, 200, 0),
        ]
    )
    overlap, unmatched = build_overlap(agg_1m, kline, CONFIG)
    assert unmatched.empty
    assert "agg_base_volume" in overlap.columns
    assert "kline_base_volume" in overlap.columns
    assert "volume_x" not in overlap.columns
    assert overlap.loc[0, "base_volume_relative_error"] == pytest.approx(0)
    assert overlap.loc[0, "first_price_minus_open_bps"] == pytest.approx(0)
    assert bool(overlap.loc[0, "is_reliable_overlap_minute"]) is False
    assert bool(overlap.loc[1, "is_boundary_partial_minute"]) is True


def test_run_stage2b_no_label_fields_and_inputs_unchanged(tmp_path: Path) -> None:
    m0 = 1_662_076_800_000
    agg = pd.DataFrame(
        [
            agg_row(1, 100, 1, m0, False),
            agg_row(2, 101, 1, m0 + 1_000, True),
            agg_row(3, 102, 1, m0 + 55_000, False),
        ]
    )
    kline = pd.DataFrame([kline_row(m0, 100, 102, 100, 102, 3, 303, 3, 2, 202, 0)])
    agg_path = tmp_path / "agg.parquet"
    kline_path = tmp_path / "kline.parquet"
    agg.to_parquet(agg_path, index=False)
    kline.to_parquet(kline_path, index=False)
    before_agg = hashlib.sha256(agg_path.read_bytes()).hexdigest()
    before_kline = hashlib.sha256(kline_path.read_bytes()).hexdigest()
    cfg = {
        **CONFIG,
        "agg_input_path": "agg.parquet",
        "kline_input_path": "kline.parquet",
        "agg_1m_output_path": "out/agg_1m.parquet",
        "overlap_output_path": "out/overlap.parquet",
        "log_path": "reports/log.txt",
        "report_paths": {
            "aggregation": "reports/agg.md",
            "consistency": "reports/consistency.md",
            "execution_proxy": "reports/proxy.md",
            "consistency_by_hour": "reports/hour.csv",
            "largest_volume_errors": "reports/vol.csv",
            "largest_price_errors": "reports/price.csv",
            "id_gap_error_analysis": "reports/gap.csv",
            "unmatched_minutes": "reports/unmatched.csv",
        },
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    run_stage2b(cfg_path, tmp_path)
    overlap = pd.read_parquet(tmp_path / "out/overlap.parquet")
    bad = [c for c in overlap.columns if "label" in c or "future_return" in c]
    assert bad == []
    assert hashlib.sha256(agg_path.read_bytes()).hexdigest() == before_agg
    assert hashlib.sha256(kline_path.read_bytes()).hexdigest() == before_kline


def agg_row(agg_id: int, price: float, qty: float, trade_time: int, is_buyer_maker: bool) -> dict[str, object]:
    return {
        "agg_trade_id": agg_id,
        "price": price,
        "quantity": qty,
        "trade_time": trade_time,
        "is_buyer_maker": is_buyer_maker,
        "quote_quantity": price * qty,
        "is_active_buy": not is_buyer_maker,
        "is_active_sell": is_buyer_maker,
        "id_gap_before": 0,
    }


def kline_row(
    open_time: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float,
    quote_volume: float,
    trades: int,
    taker_base: float,
    taker_quote: float,
    segment: int,
) -> dict[str, object]:
    return {
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "quote_volume": quote_volume,
        "number_of_trades": trades,
        "taker_buy_base_volume": taker_base,
        "taker_buy_quote_volume": taker_quote,
        "continuity_segment_id": segment,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

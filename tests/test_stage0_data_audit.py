from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_raw_data import infer_timestamp_unit_from_range, run_audit  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row[col]) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_timestamp_unit_inference() -> None:
    assert infer_timestamp_unit_from_range(1_662_076_800, 1_662_076_801) == "seconds"
    assert infer_timestamp_unit_from_range(1_662_076_800_000, 1_662_076_801_000) == "milliseconds"
    assert infer_timestamp_unit_from_range(1_662_076_800_000_000, 1_662_076_801_000_000) == "microseconds"


def test_stage0_audit_on_synthetic_data(tmp_path: Path) -> None:
    kline_path = tmp_path / "data/raw/binance/BTCUSDT_spot_klines_1m.csv"
    agg_path = tmp_path / "data/raw/binance/BTCUSDT_spot_agg_trades.csv"
    config_path = tmp_path / "config/stage0_data_audit.json"

    write_csv(
        kline_path,
        [
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": 1_662_076_800_000,
                "open_time_utc": "2022-09-02T00:00:00+00:00",
                "open": 100,
                "high": 105,
                "low": 99,
                "close": 102,
                "volume": 1.5,
                "close_time": 1_662_076_859_999,
                "close_time_utc": "2022-09-02T00:00:59.999000+00:00",
                "quote_volume": 150,
                "number_of_trades": 2,
                "taker_buy_base_volume": 1.0,
                "taker_buy_quote_volume": 100,
            },
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": 1_662_076_860_000,
                "open_time_utc": "2022-09-02T00:01:00+00:00",
                "open": 102,
                "high": 106,
                "low": 101,
                "close": 103,
                "volume": 2.0,
                "close_time": 1_662_076_919_999,
                "close_time_utc": "2022-09-02T00:01:59.999000+00:00",
                "quote_volume": 204,
                "number_of_trades": 2,
                "taker_buy_base_volume": 1.0,
                "taker_buy_quote_volume": 102,
            },
            {
                "symbol": "BTCUSDT",
                "interval": "1m",
                "open_time": 1_662_076_980_000,
                "open_time_utc": "2022-09-02T00:03:00+00:00",
                "open": 103,
                "high": 104,
                "low": 102,
                "close": 103,
                "volume": 0.5,
                "close_time": 1_662_077_039_999,
                "close_time_utc": "2022-09-02T00:03:59.999000+00:00",
                "quote_volume": 51.5,
                "number_of_trades": 1,
                "taker_buy_base_volume": 0.0,
                "taker_buy_quote_volume": 0.0,
            },
        ],
    )
    write_csv(
        agg_path,
        [
            {
                "symbol": "BTCUSDT",
                "agg_trade_id": 10,
                "price": 100,
                "quantity": 1.0,
                "first_trade_id": 20,
                "last_trade_id": 20,
                "trade_time": 1_662_076_800_000,
                "trade_time_utc": "2022-09-02T00:00:00+00:00",
                "is_buyer_maker": "False",
                "is_best_match": "True",
            },
            {
                "symbol": "BTCUSDT",
                "agg_trade_id": 11,
                "price": 101,
                "quantity": 0.5,
                "first_trade_id": 21,
                "last_trade_id": 21,
                "trade_time": 1_662_076_830_000,
                "trade_time_utc": "2022-09-02T00:00:30+00:00",
                "is_buyer_maker": "True",
                "is_best_match": "True",
            },
            {
                "symbol": "BTCUSDT",
                "agg_trade_id": 12,
                "price": 102,
                "quantity": 1.8,
                "first_trade_id": 22,
                "last_trade_id": 22,
                "trade_time": 1_662_076_860_000,
                "trade_time_utc": "2022-09-02T00:01:00+00:00",
                "is_buyer_maker": "False",
                "is_best_match": "True",
            },
        ],
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(
            {
                "random_seed": 42,
                "timezone": "UTC",
                "datasets": {
                    "klines": {"candidate_paths": [str(kline_path.relative_to(tmp_path))]},
                    "agg_trades": {"candidate_paths": [str(agg_path.relative_to(tmp_path))]},
                },
                "outputs": {
                    "report_md": "reports/data_audit.md",
                    "schema_json": "reports/data_schema.json",
                    "log_file": "reports/data_audit.log",
                },
                "audit": {
                    "chunk_size": 2,
                    "unique_exact_threshold": 10,
                    "max_examples": 5,
                    "volume_compare_sample_rows": 5,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    schema = run_audit(config_path=config_path, root=tmp_path)

    assert schema["datasets"]["klines"]["row_count"] == 3
    assert schema["datasets"]["agg_trades"]["row_count"] == 3
    assert schema["datasets"]["klines"]["checks"]["missing_minutes"] == 1
    assert schema["datasets"]["agg_trades"]["checks"]["is_buyer_maker_values"] == {"False": 2, "True": 1}
    assert schema["quality_checks"]["time_overlap"]["has_overlap"] is True
    assert schema["quality_checks"]["volume_comparison"]["matched_minute_count"] == 2
    assert (tmp_path / "reports/data_audit.md").exists()
    assert (tmp_path / "reports/data_schema.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

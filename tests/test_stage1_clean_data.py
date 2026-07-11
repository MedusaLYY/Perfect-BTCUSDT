from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.clean_stage1_data import clean_agg_trades_dataframe, clean_klines_dataframe  # noqa: E402


KLINE_MAPPING = {
    "symbol": {"column": "symbol"},
    "interval": {"column": "interval"},
    "open_time": {"column": "open_time"},
    "open_time_utc": {"column": "open_time_utc"},
    "close_time": {"column": "close_time"},
    "close_time_utc": {"column": "close_time_utc"},
    "open": {"column": "open"},
    "high": {"column": "high"},
    "low": {"column": "low"},
    "close": {"column": "close"},
    "volume": {"column": "volume"},
    "quote_volume": {"column": "quote_volume"},
    "number_of_trades": {"column": "number_of_trades"},
    "taker_buy_base_volume": {"column": "taker_buy_base_volume"},
    "taker_buy_quote_volume": {"column": "taker_buy_quote_volume"},
}


AGG_MAPPING = {
    "symbol": {"column": "symbol"},
    "agg_trade_id": {"column": "agg_trade_id"},
    "price": {"column": "price"},
    "quantity": {"column": "quantity"},
    "first_trade_id": {"column": "first_trade_id"},
    "last_trade_id": {"column": "last_trade_id"},
    "trade_time": {"column": "trade_time"},
    "trade_time_utc": {"column": "trade_time_utc"},
    "is_buyer_maker": {"column": "is_buyer_maker"},
    "is_best_match": {"column": "is_best_match"},
}


def test_kline_cleaning_rules() -> None:
    raw = pd.DataFrame(
        [
            _kline(0, 100, 105, 99, 101, 1, close_time=59_999),
            _kline(60_000, 101, 106, 100, 102, 1, close_time=119_999),
            _kline(60_000, 102, 107, 101, 103, 2, close_time=119_999),
            _kline(120_000, 0, 105, 0, 101, 1, close_time=179_999),
            _kline(180_000, 101, 105, 99, 102, -1, close_time=239_999),
            _kline(240_000, 101, 100, 99, 102, 1, close_time=299_999),
            _kline(300_000, 103, 104, 102, 103, 1, close_time=333_333),
            _kline(360_000, 103, 105, 102, 104, 1, close_time=419_999),
            _kline(540_000, 104, 106, 103, 105, 1, close_time=599_999),
        ]
    )

    result = clean_klines_dataframe(raw, KLINE_MAPPING, 60_000, 59_999, True)
    clean = result.df

    assert result.stats["deletion_reasons"]["duplicate_open_time_keep_last_original_order"] == 1
    assert result.stats["deletion_reasons"]["nonpositive_price"] == 1
    assert result.stats["deletion_reasons"]["negative_volume_or_trade_count"] == 1
    assert result.stats["deletion_reasons"]["ohlc_relation_violation"] == 1
    assert clean.loc[clean["open_time"] == 60_000, "close"].iloc[0] == 103
    assert clean["open_time"].tolist() == [0, 60_000, 300_000, 360_000, 540_000]
    assert result.stats["close_time_repaired_count"] == 1
    repaired = clean.loc[clean["open_time"] == 300_000].iloc[0]
    assert repaired["close_time"] == 359_999
    assert bool(repaired["close_time_repaired"]) is True
    assert clean.loc[clean["open_time"] == 300_000, "continuity_segment_id"].iloc[0] == clean.loc[
        clean["open_time"] == 360_000, "continuity_segment_id"
    ].iloc[0]
    assert clean.loc[clean["open_time"] == 540_000, "continuity_segment_id"].iloc[0] != clean.loc[
        clean["open_time"] == 360_000, "continuity_segment_id"
    ].iloc[0]
    assert clean.loc[clean["open_time"] == 540_000, "gap_before_minutes"].iloc[0] == 2
    assert len(clean) == 5


def test_agg_trade_cleaning_rules() -> None:
    raw = pd.DataFrame(
        [
            _agg(1, 100, 1, 1_000, False),
            _agg(2, 101, 1, 1_001, "True"),
            _agg(2, 102, 2, 1_002, "0"),
            _agg(4, 103, 1, 1_002, "1"),
            _agg(5, 0, 1, 1_003, "false"),
            _agg(6, 104, 0, 1_004, "TRUE"),
            _agg(7, 105, 1, 1_005, "not_bool"),
            _agg(8, 106, 1, 70_000, "FALSE"),
        ]
    )

    result = clean_agg_trades_dataframe(raw, AGG_MAPPING, long_no_trade_threshold_ms=60_000)
    clean = result.df

    assert result.stats["deletion_reasons"]["duplicate_agg_trade_id_keep_last_original_order"] == 1
    assert result.stats["deletion_reasons"]["nonpositive_price"] == 1
    assert result.stats["deletion_reasons"]["nonpositive_quantity"] == 1
    assert result.stats["deletion_reasons"]["invalid_is_buyer_maker"] == 1
    assert "not_bool" in result.stats["invalid_is_buyer_maker_values"]
    assert clean["agg_trade_id"].tolist() == [1, 2, 4, 8]
    assert clean.loc[clean["agg_trade_id"] == 2, "price"].iloc[0] == 102
    assert bool(clean.loc[clean["agg_trade_id"] == 2, "is_buyer_maker"].iloc[0]) is False
    assert bool(clean.loc[clean["agg_trade_id"] == 2, "is_active_buy"].iloc[0]) is True
    assert bool(clean.loc[clean["agg_trade_id"] == 4, "is_active_sell"].iloc[0]) is True
    assert clean.loc[clean["agg_trade_id"] == 4, "id_gap_before"].iloc[0] == 1
    assert result.id_gaps.loc[result.id_gaps["current_agg_trade_id"] == 4, "missing_id_count"].iloc[0] == 1
    assert result.stats["same_millisecond_group_count"] == 1
    assert result.stats["long_no_trade_interval_count"] == 1


def test_cleaning_does_not_modify_input_file(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    raw = pd.DataFrame([_kline(0, 100, 101, 99, 100, 1, close_time=12_345)])
    raw.to_csv(path, index=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = pd.read_csv(path)
    clean_klines_dataframe(loaded, KLINE_MAPPING, 60_000, 59_999, True)

    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after


def _kline(open_time: int, open_: float, high: float, low: float, close: float, volume: float, close_time: int) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "open_time": open_time,
        "open_time_utc": pd.Timestamp(open_time, unit="ms", tz="UTC").isoformat(),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "close_time": close_time,
        "close_time_utc": pd.Timestamp(close_time, unit="ms", tz="UTC").isoformat(),
        "quote_volume": max(volume, 0) * close if close > 0 else 0,
        "number_of_trades": 1,
        "taker_buy_base_volume": max(volume, 0) / 2,
        "taker_buy_quote_volume": max(volume, 0) * close / 2 if close > 0 else 0,
    }


def _agg(agg_id: int, price: float, quantity: float, trade_time: int, is_buyer_maker: object) -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "agg_trade_id": agg_id,
        "price": price,
        "quantity": quantity,
        "first_trade_id": agg_id * 10,
        "last_trade_id": agg_id * 10,
        "trade_time": trade_time,
        "trade_time_utc": pd.Timestamp(trade_time, unit="ms", tz="UTC").isoformat(),
        "is_buyer_maker": is_buyer_maker,
        "is_best_match": True,
    }


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

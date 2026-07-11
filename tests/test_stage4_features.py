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

from scripts.build_stage4_features import (  # noqa: E402
    FEATURE_NAMES,
    Stage4ValidationError,
    build_features,
    compute_segment_features,
    make_feature_dictionary,
    make_feature_manifest,
    run_stage4,
    scan_for_forbidden_fields,
)


CONFIG = {
    "feature_set_version": "kline_v1_63",
    "ordered_feature_names": [
        "log_return_1m",
        "log_return_3m",
        "log_return_5m",
        "log_return_10m",
        "log_return_15m",
        "log_return_30m",
        "log_return_60m",
        "log_return_120m",
        "momentum_acceleration_5_30",
        "momentum_acceleration_15_60",
        "efficiency_ratio_15m",
        "efficiency_ratio_60m",
        "up_candle_ratio_15m",
        "realized_volatility_5m",
        "realized_volatility_15m",
        "realized_volatility_30m",
        "realized_volatility_60m",
        "realized_volatility_120m",
        "volatility_ratio_5_60",
        "volatility_ratio_15_120",
        "normalized_atr_14",
        "normalized_atr_60",
        "semivariance_imbalance_60m",
        "ema_distance_5",
        "ema_distance_20",
        "ema_distance_60",
        "ema_distance_120",
        "ema_spread_5_20",
        "ema_spread_20_60",
        "normalized_slope_15m",
        "normalized_slope_60m",
        "range_position_30m",
        "range_position_60m",
        "range_position_120m",
        "support_distance_60m",
        "resistance_distance_60m",
        "vwap_distance_60m",
        "signed_body_ratio",
        "absolute_body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "close_location",
        "signed_body_mean_5m",
        "upper_wick_mean_15m",
        "lower_wick_mean_15m",
        "is_zero_volume_1m",
        "log_quote_volume_zscore_60m",
        "volume_ratio_5_60",
        "volume_ratio_15_60",
        "taker_buy_quote_ratio_1m",
        "taker_buy_quote_ratio_5m",
        "taker_buy_quote_ratio_15m",
        "taker_buy_quote_ratio_60m",
        "buy_pressure_change_5_60",
        "buy_pressure_persistence_15m",
        "buy_pressure_std_15m",
        "log_trade_count_zscore_60m",
        "log_avg_trade_quote_size_zscore_60m",
        "time_sin",
        "time_cos",
        "weekday_sin",
        "weekday_cos",
        "is_weekend",
    ],
    "expected_feature_count": 63,
    "expected_interval_ms": 60000,
    "prediction_interval_minutes": 5,
    "rolling_windows": {
        "log_return": [1, 3, 5, 10, 15, 30, 60, 120],
        "efficiency": [15, 60],
        "realized_volatility": [5, 15, 30, 60, 120],
        "range_position": [30, 60, 120],
        "vwap": [60],
        "volume_zscore": [60],
        "buy_pressure": [5, 15, 60],
        "slope": [15, 60],
    },
    "ema_spans": [5, 20, 60, 120],
    "atr_windows": [14, 60],
    "safe_division_rules": {},
    "numeric_tolerances": {
        "absolute": 1e-10,
        "relative": 1e-9,
        "range_tolerance": 1e-6,
        "float32_max_abs_error": 1e-4,
        "float32_max_relative_error": 1e-5,
    },
    "output_float_dtype": "float32",
    "parquet_compression": "snappy",
    "processing_mode": "segment",
    "output_batch_size": 200000,
    "summary_quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
}


def test_log_returns_momentum_efficiency_volatility_atr_and_slope() -> None:
    df = make_model_base(180)
    df["close"] = 100.0 + np.arange(len(df), dtype=float) * 0.5
    df["open"] = df["close"] - 0.1
    df["high"] = df["close"] + 0.2
    df["low"] = df["close"] - 0.3

    features, _ = compute_segment_features(df, CONFIG)
    i = 150

    assert features.loc[i, "log_return_5m"] == pytest.approx(math.log(df.loc[i, "close"] / df.loc[i - 5, "close"]))
    assert features.loc[i, "momentum_acceleration_5_30"] == pytest.approx(
        features.loc[i, "log_return_5m"] / 5 - features.loc[i, "log_return_30m"] / 30
    )
    path = df["close"].diff().abs().iloc[i - 14 : i + 1].sum()
    direct = abs(df.loc[i, "close"] - df.loc[i - 15, "close"])
    assert features.loc[i, "efficiency_ratio_15m"] == pytest.approx(direct / path)
    last_5 = np.log(df["close"] / df["close"].shift(1)).iloc[i - 4 : i + 1]
    assert features.loc[i, "realized_volatility_5m"] == pytest.approx(float(np.std(last_5, ddof=0)))
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    assert features.loc[i, "normalized_atr_14"] == pytest.approx(true_range.iloc[i - 13 : i + 1].mean() / df.loc[i, "close"])
    y = df["close"].iloc[i - 14 : i + 1].to_numpy()
    slope = np.polyfit(np.arange(15), y, 1)[0] / y.mean()
    assert features.loc[i, "normalized_slope_15m"] == pytest.approx(slope)


def test_neutral_rules_for_flat_prices_zero_ranges_and_zero_volume() -> None:
    df = make_model_base(150, close_base=100.0)
    df[["open", "high", "low", "close"]] = 100.0
    df[["volume", "quote_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]] = 0.0
    df["number_of_trades"] = 0

    features, counts = compute_segment_features(df, CONFIG)
    i = 130

    assert features.loc[i, "efficiency_ratio_15m"] == 0
    assert features.loc[i, "efficiency_ratio_60m"] == 0
    assert features.loc[i, "volatility_ratio_5_60"] == 1
    assert features.loc[i, "volatility_ratio_15_120"] == 1
    assert features.loc[i, "normalized_atr_14"] == 0
    assert features.loc[i, "normalized_atr_60"] == 0
    assert features.loc[i, "semivariance_imbalance_60m"] == 0
    assert features.loc[i, "range_position_30m"] == 0.5
    assert features.loc[i, "support_distance_60m"] == 0
    assert features.loc[i, "resistance_distance_60m"] == 0
    assert features.loc[i, "vwap_distance_60m"] == 0
    assert features.loc[i, "signed_body_ratio"] == 0
    assert features.loc[i, "absolute_body_ratio"] == 0
    assert features.loc[i, "upper_wick_ratio"] == 0
    assert features.loc[i, "lower_wick_ratio"] == 0
    assert features.loc[i, "close_location"] == 0.5
    assert features.loc[i, "is_zero_volume_1m"] == 1
    assert features.loc[i, "volume_ratio_5_60"] == 0
    assert features.loc[i, "taker_buy_quote_ratio_1m"] == 0.5
    assert features.loc[i, "taker_buy_quote_ratio_60m"] == 0.5
    assert features.loc[i, "log_quote_volume_zscore_60m"] == 0
    assert features.loc[i, "log_trade_count_zscore_60m"] == 0
    assert counts["zero_volume_1m"] == 150
    assert counts["zero_candle_range"] == 150
    assert counts["rolling_vwap_zero_base_volume_60m"] > 0


def test_taker_buy_ratios_are_volume_weighted_and_buy_pressure_features() -> None:
    df = make_model_base(80)
    df["quote_volume"] = np.where(np.arange(80) % 2 == 0, 100.0, 300.0)
    df["volume"] = df["quote_volume"] / 100.0
    df["taker_buy_quote_volume"] = np.where(np.arange(80) % 2 == 0, 25.0, 240.0)
    df["taker_buy_base_volume"] = df["taker_buy_quote_volume"] / 100.0

    features, _ = compute_segment_features(df, CONFIG)
    i = 70

    expected_5 = df["taker_buy_quote_volume"].iloc[i - 4 : i + 1].sum() / df["quote_volume"].iloc[i - 4 : i + 1].sum()
    simple_average = (df["taker_buy_quote_volume"] / df["quote_volume"]).iloc[i - 4 : i + 1].mean()
    expected_60 = df["taker_buy_quote_volume"].iloc[i - 59 : i + 1].sum() / df["quote_volume"].iloc[i - 59 : i + 1].sum()
    assert features.loc[i, "taker_buy_quote_ratio_5m"] == pytest.approx(expected_5)
    assert features.loc[i, "taker_buy_quote_ratio_5m"] != pytest.approx(simple_average)
    assert features.loc[i, "buy_pressure_change_5_60"] == pytest.approx(expected_5 - expected_60)
    ratio_1m = df["taker_buy_quote_volume"] / df["quote_volume"]
    assert features.loc[i, "buy_pressure_persistence_15m"] == pytest.approx((ratio_1m.iloc[i - 14 : i + 1] > 0.5).mean())
    assert features.loc[i, "buy_pressure_std_15m"] == pytest.approx(float(np.std(ratio_1m.iloc[i - 14 : i + 1], ddof=0)))


def test_range_support_resistance_vwap_and_candle_structure() -> None:
    df = make_model_base(130)
    df["close"] = 100 + np.sin(np.arange(130) / 7)
    df["open"] = df["close"] - 0.2
    df["high"] = df["close"] + 0.8
    df["low"] = df["close"] - 0.4
    df["quote_volume"] = 1000 + np.arange(130)
    df["volume"] = 10 + np.arange(130) / 100
    df["taker_buy_quote_volume"] = df["quote_volume"] * 0.6
    df["taker_buy_base_volume"] = df["volume"] * 0.6

    features, _ = compute_segment_features(df, CONFIG)
    i = 100
    rolling_low = df["low"].iloc[i - 59 : i + 1].min()
    rolling_high = df["high"].iloc[i - 59 : i + 1].max()
    atr14 = true_range(df).iloc[i - 13 : i + 1].mean()
    vwap60 = df["quote_volume"].iloc[i - 59 : i + 1].sum() / df["volume"].iloc[i - 59 : i + 1].sum()
    candle_range = df.loc[i, "high"] - df.loc[i, "low"]

    assert features.loc[i, "range_position_60m"] == pytest.approx((df.loc[i, "close"] - rolling_low) / (rolling_high - rolling_low))
    assert features.loc[i, "support_distance_60m"] == pytest.approx((df.loc[i, "close"] - rolling_low) / atr14)
    assert features.loc[i, "resistance_distance_60m"] == pytest.approx((rolling_high - df.loc[i, "close"]) / atr14)
    assert features.loc[i, "vwap_distance_60m"] == pytest.approx((df.loc[i, "close"] - vwap60) / atr14)
    assert features.loc[i, "signed_body_ratio"] == pytest.approx((df.loc[i, "close"] - df.loc[i, "open"]) / candle_range)
    assert features.loc[i, "upper_wick_ratio"] == pytest.approx((df.loc[i, "high"] - max(df.loc[i, "open"], df.loc[i, "close"])) / candle_range)
    assert features.loc[i, "lower_wick_mean_15m"] == pytest.approx(features["lower_wick_ratio"].iloc[i - 14 : i + 1].mean())


def test_segment_boundaries_reset_returns_rolling_slope_and_ema() -> None:
    first = make_model_base(80, segment_id=0)
    second = make_model_base(130, base_open_time=6_000_000, segment_id=1)
    first["close"] = np.linspace(100, 180, len(first))
    second["close"] = np.linspace(50, 70, len(second))
    for df in [first, second]:
        df["open"] = df["close"]
        df["high"] = df["close"] + 1
        df["low"] = df["close"] - 1
    model = pd.concat([first, second], ignore_index=True)

    result = build_features(model, CONFIG).df
    second_start_rows = result[result["continuity_segment_id"] == 1].sort_values("segment_row_number")
    early = second_start_rows.iloc[0]

    assert pd.isna(early["log_return_120m"])
    assert pd.isna(early["normalized_slope_60m"])
    assert early["feature_missing_count"] > 0
    assert bool(early["is_feature_complete"]) is False
    assert bool(early["is_final_feature_candidate"]) is False


def test_output_only_prediction_times_feature_count_manifest_dictionary_and_time_features() -> None:
    base = int(pd.Timestamp("2022-01-08T00:00:00Z").timestamp() * 1000)
    df = make_model_base(310, base_open_time=base)
    result = build_features(df, CONFIG)
    output = result.df
    manifest = make_feature_manifest(CONFIG, "config/stage4_build_features.json", "scripts/build_stage4_features.py", "input.parquet")
    dictionary = make_feature_dictionary(CONFIG)

    assert len(FEATURE_NAMES) == 63
    assert FEATURE_NAMES == CONFIG["ordered_feature_names"]
    assert output["is_prediction_time_5m"].all()
    assert not output["decision_time"].duplicated().any()
    assert list(output[FEATURE_NAMES].columns) == manifest["ordered_feature_names"]
    assert manifest["feature_count"] == 63
    assert len(dictionary["features"]) == 63
    assert all(item["uses_future_data"] is False for item in dictionary["features"])
    assert all(item["allowed_as_model_input"] is True for item in dictionary["features"])
    saturday = output.loc[output["decision_time"] == base + 5 * 60_000].iloc[0]
    assert saturday["is_weekend"] == 1
    assert saturday["time_sin"] == pytest.approx(math.sin(2 * math.pi * 5 / 1440))
    assert saturday["time_cos"] == pytest.approx(math.cos(2 * math.pi * 5 / 1440))


def test_feature_completeness_final_candidate_float32_and_forbidden_scan() -> None:
    df = make_model_base(310)
    result = build_features(df, CONFIG)
    output = result.df
    final = output[output["is_final_feature_candidate"]]

    assert not final.empty
    assert (final["feature_missing_count"] == 0).all()
    assert final["is_feature_complete"].all()
    assert np.isfinite(final[FEATURE_NAMES].to_numpy(dtype="float64")).all()
    assert result.float32_report["finite_to_infinite_count"] == 0
    assert result.float32_report["sign_change_count"] == 0
    assert scan_for_forbidden_fields(output.columns, FEATURE_NAMES) == []
    assert scan_for_forbidden_fields(["label_up_60m"], FEATURE_NAMES) == ["label_up_60m"]


def test_prefix_invariance_and_future_mutation_do_not_change_past_features() -> None:
    df = make_model_base(320)
    t_open = df.loc[254, "open_time"]
    full = build_features(df, CONFIG).df
    mutated = df.copy()
    mutated.loc[mutated["open_time"] > t_open, ["open", "high", "low", "close", "quote_volume"]] *= 10
    changed_future = build_features(mutated, CONFIG).df
    prefix = build_features(df.loc[df["open_time"] <= t_open].copy(), CONFIG).df

    full_past = full[full["feature_open_time"] <= t_open].reset_index(drop=True)
    changed_past = changed_future[changed_future["feature_open_time"] <= t_open].reset_index(drop=True)
    prefix_rows = prefix.reset_index(drop=True)

    pd.testing.assert_frame_equal(full_past[FEATURE_NAMES], changed_past[FEATURE_NAMES], check_dtype=False)
    pd.testing.assert_frame_equal(full_past[FEATURE_NAMES], prefix_rows[FEATURE_NAMES], check_dtype=False)


def test_run_stage4_preserves_input_and_writes_reports(tmp_path: Path) -> None:
    df = make_model_base(310)
    input_path = tmp_path / "model_base.parquet"
    df.to_parquet(input_path, index=False)
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()
    config = {
        **CONFIG,
        "input_path": "model_base.parquet",
        "output_path": "out/features.parquet",
        "log_path": "reports/stage4.log",
        "report_paths": {
            "main_report": "reports/stage4.md",
            "feature_summary": "reports/summary.csv",
            "missing_by_year": "reports/year.csv",
            "missing_by_segment": "reports/segment.csv",
            "incomplete_feature_samples": "reports/incomplete.csv",
            "feature_dictionary": "reports/dictionary.json",
            "feature_manifest": "reports/manifest.json",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_stage4(config_path, tmp_path)

    assert hashlib.sha256(input_path.read_bytes()).hexdigest() == before
    output = pd.read_parquet(tmp_path / "out/features.parquet")
    assert len([col for col in output.columns if col in FEATURE_NAMES]) == 63
    assert output["is_prediction_time_5m"].all()
    assert (tmp_path / "reports/stage4.md").exists()
    assert (tmp_path / "reports/dictionary.json").exists()
    assert (tmp_path / "reports/manifest.json").exists()


def make_model_base(n: int, base_open_time: int = 0, segment_id: int = 0, close_base: float = 100.0) -> pd.DataFrame:
    idx = np.arange(n)
    open_time = base_open_time + idx * 60_000
    close = close_base + idx * 0.1
    open_price = close - 0.02
    high = np.maximum(open_price, close) + 0.05
    low = np.minimum(open_price, close) - 0.05
    volume = np.full(n, 10.0)
    quote_volume = volume * close
    number_of_trades = np.full(n, 20, dtype=np.int64)
    taker_buy_quote_volume = quote_volume * 0.55
    taker_buy_base_volume = volume * 0.55
    segment_row_number = idx
    segment_length = np.full(n, n, dtype=np.int64)
    minutes_until_segment_end = n - idx - 1
    decision_time = open_time + 60_000
    prediction = (decision_time % (5 * 60_000)) == 0
    has_history = segment_row_number >= 239
    has_future = minutes_until_segment_end >= 61
    return pd.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "interval": ["1m"] * n,
            "open_time": open_time.astype("int64"),
            "open": open_price.astype(float),
            "high": high.astype(float),
            "low": low.astype(float),
            "close": close.astype(float),
            "volume": volume.astype(float),
            "quote_volume": quote_volume.astype(float),
            "number_of_trades": number_of_trades,
            "taker_buy_base_volume": taker_buy_base_volume.astype(float),
            "taker_buy_quote_volume": taker_buy_quote_volume.astype(float),
            "continuity_segment_id": np.full(n, segment_id, dtype=np.int64),
            "gap_before_minutes": np.zeros(n, dtype=np.int64),
            "feature_time": decision_time.astype("int64"),
            "decision_time": decision_time.astype("int64"),
            "segment_row_number": segment_row_number.astype("int64"),
            "segment_length": segment_length,
            "minutes_until_segment_end": minutes_until_segment_end.astype("int64"),
            "has_history_240m": has_history.astype(bool),
            "has_future_61m": has_future.astype(bool),
            "is_prediction_time_5m": prediction.astype(bool),
            "is_model_candidate": (prediction & has_history & has_future).astype(bool),
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

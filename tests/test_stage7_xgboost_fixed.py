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

from scripts.build_stage4_features import FEATURE_NAMES  # noqa: E402
from scripts.train_stage6_baselines import compute_classification_metrics  # noqa: E402
from scripts.train_stage7_xgboost_fixed import (  # noqa: E402
    MODEL_NAME,
    Stage7ValidationError,
    build_stage7_outputs,
    calibration_table,
    compare_with_stage6,
    development_recommendation,
    evaluate_prediction_subsets,
    extract_feature_importance_frame,
    fit_xgboost_fold,
    make_inner_time_split,
    normalize_feature_importance,
    pooled_and_macro_summary,
    prepare_feature_matrix,
    run_stage7,
    validate_feature_manifest,
    validate_inner_split,
    validate_outer_fold_integrity,
    verify_xgboost_parameters,
)


CONFIG = {
    "dataset_path": "dataset.parquet",
    "split_path": "splits.parquet",
    "dataset_manifest_path": "dataset_manifest.json",
    "fold_manifest_path": "fold_manifest.json",
    "feature_manifest_path": "feature_manifest.json",
    "stage6_prediction_path": "stage6_predictions.parquet",
    "stage6_model_manifest_path": "stage6_model_manifest.json",
    "prediction_output_path": "out/predictions.parquet",
    "model_output_dir": "models",
    "report_paths": {
        "main_report": "reports/report.md",
        "metrics_by_fold": "reports/metrics_by_fold.csv",
        "metrics_by_subset": "reports/metrics_by_subset.csv",
        "metrics_by_offset": "reports/metrics_by_offset.csv",
        "oof_summary": "reports/oof_summary.csv",
        "model_comparison": "reports/model_comparison.csv",
        "calibration_equal_width": "reports/calibration_equal_width.csv",
        "calibration_equal_frequency": "reports/calibration_equal_frequency.csv",
        "learning_curves": "reports/learning_curves.csv",
        "feature_importance_by_fold": "reports/feature_importance_by_fold.csv",
        "feature_importance_stability": "reports/feature_importance_stability.csv",
        "inner_split_audit": "reports/inner_split_audit.csv",
        "model_manifest": "reports/model_manifest.json",
    },
    "log_path": "reports/stage7.log",
    "fold_names": ["fold_2020"],
    "feature_count": 63,
    "model_name": MODEL_NAME,
    "xgboost_parameters": {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 8,
        "learning_rate": 0.3,
        "max_depth": 2,
        "min_child_weight": 1,
        "gamma": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "early_stopping_rounds": 2,
        "random_state": 42,
        "verbosity": 0,
        "validate_parameters": True,
    },
    "early_stopping_window_days": 1,
    "inner_purge_horizon_minutes": 60,
    "minimum_inner_early_stop_samples": 4,
    "fixed_prediction_threshold": 0.5,
    "evaluation_subsets": {
        "dense": "DENSE",
        "nonoverlap": "NONOVERLAP_OFFSET_00",
        "offsets": list(range(0, 60, 5)),
        "margin_subsets": ["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"],
    },
    "calibration_bins": 2,
    "development_gate_thresholds": {
        "minimum_auc_gain_for_tuning": 0.002,
        "maximum_allowed_logloss_degradation": 0.001,
        "minimum_fold_wins": 3,
    },
    "random_seed": 42,
    "n_jobs": 1,
    "parquet_compression": "snappy",
    "hash_algorithm": "sha256",
    "numeric_tolerances": {"prediction_atol": 1e-12},
}


def test_feature_columns_strictly_from_manifest_and_ordered() -> None:
    features = validate_feature_manifest(dataset_manifest(), feature_manifest(), 63)

    assert features == FEATURE_NAMES
    assert len(features) == 63


def test_feature_manifest_mismatch_fails() -> None:
    bad = dataset_manifest()
    bad["feature_columns"] = FEATURE_NAMES[:-1] + ["bad_feature"]

    with pytest.raises(Stage7ValidationError, match="feature_columns"):
        validate_feature_manifest(bad, feature_manifest(), 63)


def test_target_future_weight_and_boundary_fields_never_enter_x() -> None:
    dataset, _ = make_stage7_frames()
    X = prepare_feature_matrix(dataset, FEATURE_NAMES)

    assert list(X.columns) == FEATURE_NAMES
    forbidden = {
        "label_up_60m",
        "future_simple_return_60m",
        "future_log_return_60m",
        "absolute_future_return_bps",
        "sample_weight_margin",
        "entry_minute_open_time",
        "settlement_minute_open_time",
        "proxy_boundary_risk_5bps",
    }
    assert forbidden.isdisjoint(X.columns)


def test_nonfinite_feature_fails() -> None:
    dataset, _ = make_stage7_frames()
    dataset.loc[0, FEATURE_NAMES[0]] = np.nan

    with pytest.raises(Stage7ValidationError, match="finite"):
        prepare_feature_matrix(dataset, FEATURE_NAMES)


def test_outer_fold_integrity_and_final_test_exclusion() -> None:
    dataset, splits = make_stage7_frames()
    integrity = validate_outer_fold_integrity(dataset, splits, fold_manifest()["folds"][0], "fold_2020")

    assert integrity["train_count"] == 18
    assert integrity["validation_count"] == 12
    assert integrity["train_validation_overlap_count"] == 0
    assert integrity["final_test_in_train_or_validation"] == 0
    assert integrity["train_max_settlement_minute_open_time"] < integrity["validation_min_decision_time"]


def test_final_test_in_outer_role_fails_before_fit() -> None:
    dataset, splits = make_stage7_frames()
    splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "fold_2020_role"] = "TRAIN"

    with pytest.raises(Stage7ValidationError, match="FINAL_TEST"):
        validate_outer_fold_integrity(dataset, splits, fold_manifest()["folds"][0], "fold_2020")


def test_inner_split_rules_and_start_time() -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    outer_train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    inner = make_inner_time_split(outer_train, ms("2020-01-01T00:00:00Z"), window_days=1, horizon_minutes=60)

    assert inner["inner_early_stop_start"] == ms("2019-12-31T00:00:00Z")
    assert len(inner["inner_fit"]) == 8
    assert len(inner["inner_purged"]) == 2
    assert len(inner["inner_early_stop"]) == 8
    assert inner["inner_fit"]["settlement_minute_open_time"].max() < inner["inner_early_stop"]["decision_time"].min()
    assert set(inner["inner_fit"]["dataset_row_id"]).isdisjoint(set(inner["inner_early_stop"]["dataset_row_id"]))


def test_inner_split_requires_minimum_samples_and_two_classes() -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    outer_train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    inner = make_inner_time_split(outer_train, ms("2020-01-01T00:00:00Z"), window_days=1, horizon_minutes=60)

    validate_inner_split(inner, minimum_early_stop_samples=4)
    with pytest.raises(Stage7ValidationError, match="minimum"):
        validate_inner_split(inner, minimum_early_stop_samples=100)
    inner["inner_fit"].loc[:, "label_up_60m"] = 1
    with pytest.raises(Stage7ValidationError, match="both classes"):
        validate_inner_split(inner, minimum_early_stop_samples=4)


def test_xgboost_fixed_parameters_are_verified() -> None:
    params = verify_xgboost_parameters(CONFIG["xgboost_parameters"], n_jobs=1, selector=True)

    assert params["objective"] == "binary:logistic"
    assert params["tree_method"] == "hist"
    assert params["device"] == "cpu"
    assert params["scale_pos_weight"] == 1.0
    assert params["n_estimators"] == 8
    assert params["early_stopping_rounds"] == 2


def test_two_stage_fit_uses_inner_eval_and_refits_new_model_on_outer_train(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    result = fit_xgboost_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", CONFIG, tmp_path)

    assert set(result.selector_metadata["eval_set_dataset_row_ids"]) == set(result.inner_split["inner_early_stop"]["dataset_row_id"])
    assert set(result.selector_metadata["eval_set_dataset_row_ids"]).isdisjoint(set(valid["dataset_row_id"]))
    assert result.selector_metadata["best_n_estimators"] == result.selector_metadata["best_iteration"] + 1
    assert result.refit_metadata["refit_train_sample_count"] == len(train)
    assert result.refit_metadata["refit_used_early_stopping"] is False
    assert result.selector_metadata["selector_model_object_id"] != result.refit_metadata["refit_model_object_id"]
    assert result.refit_metadata["sample_weight_margin_used"] is False
    assert result.refit_metadata["standard_scaler_used"] is False
    assert (tmp_path / "xgboost_fixed_v1.json").exists()


def test_validation_labels_do_not_change_refit_predictions(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    flipped = valid.copy()
    flipped["label_up_60m"] = 1 - flipped["label_up_60m"]

    a = fit_xgboost_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", CONFIG, tmp_path / "a")
    b = fit_xgboost_fold(train, flipped, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", CONFIG, tmp_path / "b")

    np.testing.assert_allclose(a.predictions["p_up"], b.predictions["p_up"])


def test_prediction_contract_no_train_or_final_and_threshold() -> None:
    dataset, splits = make_stage7_frames()
    stage6 = make_stage6_predictions(dataset, splits)
    output = build_stage7_outputs(dataset, splits, stage6, dataset_manifest(), feature_manifest(), fold_manifest(), stage6_model_manifest(), CONFIG)
    predictions = output.predictions

    valid_ids = set(splits.loc[splits["fold_2020_role"].eq("VALIDATION"), "dataset_row_id"])
    train_ids = set(splits.loc[splits["fold_2020_role"].eq("TRAIN"), "dataset_row_id"])
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"])
    assert set(predictions["dataset_row_id"]) == valid_ids
    assert set(predictions["dataset_row_id"]).isdisjoint(train_ids | final_ids)
    assert predictions["model_name"].unique().tolist() == [MODEL_NAME]
    assert predictions.groupby(["fold_name", "dataset_row_id"]).size().eq(1).all()
    assert predictions["p_up"].between(0, 1).all()
    assert (predictions["prediction_threshold"] == 0.5).all()
    assert predictions["y_pred"].equals((predictions["p_up"] >= 0.5).astype("int8"))


def test_metrics_subsets_match_stage6_definitions() -> None:
    dataset, splits = make_stage7_frames()
    output = build_stage7_outputs(dataset, splits, make_stage6_predictions(dataset, splits), dataset_manifest(), feature_manifest(), fold_manifest(), stage6_model_manifest(), CONFIG)
    metrics = evaluate_prediction_subsets(output.predictions)

    assert "DENSE" in set(metrics["subset_name"])
    assert "NONOVERLAP_OFFSET_00" in set(metrics["subset_name"])
    for offset in range(0, 60, 5):
        assert f"OFFSET_{offset:02d}" in set(metrics["subset_name"])
    assert "ABS_RETURN_GE_5BPS" in set(metrics["subset_name"])
    dense = metrics[metrics["subset_name"].eq("DENSE")].iloc[0]
    expected = compute_classification_metrics(output.predictions["y_true"], output.predictions["p_up"], output.predictions["y_pred"])
    assert dense["sample_count"] == expected["sample_count"]
    assert dense["log_loss"] == expected["log_loss"]


def test_stage6_prediction_join_and_delta_direction() -> None:
    dataset, splits = make_stage7_frames()
    output = build_stage7_outputs(dataset, splits, make_stage6_predictions(dataset, splits), dataset_manifest(), feature_manifest(), fold_manifest(), stage6_model_manifest(), CONFIG)
    comparison = compare_with_stage6(output.predictions, make_stage6_predictions(dataset, splits))

    assert {"logistic_regression_l2", "momentum_60m_baseline", "prior_baseline"} == set(comparison["baseline_model_name"])
    row = comparison[(comparison["subset_name"].eq("DENSE")) & (comparison["baseline_model_name"].eq("logistic_regression_l2"))].iloc[0]
    assert row["delta_roc_auc"] == row["xgb_roc_auc"] - row["baseline_roc_auc"]
    assert row["delta_log_loss"] == row["xgb_log_loss"] - row["baseline_log_loss"]


def test_stage6_y_true_mismatch_fails() -> None:
    dataset, splits = make_stage7_frames()
    output = build_stage7_outputs(dataset, splits, make_stage6_predictions(dataset, splits), dataset_manifest(), feature_manifest(), fold_manifest(), stage6_model_manifest(), CONFIG)
    bad = make_stage6_predictions(dataset, splits)
    bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]

    with pytest.raises(Stage7ValidationError, match="y_true"):
        compare_with_stage6(output.predictions, bad)


def test_pooled_oof_and_macro_are_distinct() -> None:
    rows = []
    for fold, y_true in [("fold_a", [1, 0]), ("fold_b", [1, 1, 1, 0, 0, 0])]:
        for y in y_true:
            p = 0.8 if (fold == "fold_a" and y == 1) else 0.2 if fold == "fold_a" else 0.4
            rows.append(prediction_row(len(rows), fold, y, p))
    predictions = pd.DataFrame(rows)
    summary = pooled_and_macro_summary(evaluate_prediction_subsets(predictions), predictions)
    dense = summary[summary["subset_name"].eq("DENSE")]

    assert {"pooled", "fold_macro_mean", "fold_macro_std"}.issubset(set(dense["summary_type"]))
    assert dense.loc[dense["summary_type"].eq("pooled"), "accuracy"].iloc[0] != dense.loc[dense["summary_type"].eq("fold_macro_mean"), "accuracy"].iloc[0]


def test_calibration_ece_mce() -> None:
    predictions = pd.DataFrame([prediction_row(0, "fold", 0, 0.1), prediction_row(1, "fold", 0, 0.2), prediction_row(2, "fold", 1, 0.8), prediction_row(3, "fold", 1, 0.9)])
    table = calibration_table(predictions, bins=2, strategy="equal_width", subset_name="DENSE")

    assert table["sample_count"].sum() == 4
    assert np.isclose(table["ece"].iloc[0], 0.15)
    assert np.isclose(table["mce"].iloc[0], 0.15)


def test_unused_feature_importance_is_zero() -> None:
    importance = normalize_feature_importance({"gain": {"f0": 2.0}, "weight": {"f0": 3.0}}, ["a", "b"])

    assert importance.loc[importance["feature_name"].eq("a"), "gain"].iloc[0] == 2.0
    assert importance.loc[importance["feature_name"].eq("b"), "gain"].iloc[0] == 0.0
    assert importance.loc[importance["feature_name"].eq("b"), "weight"].iloc[0] == 0.0


def test_feature_importance_extracts_all_features(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    result = fit_xgboost_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", CONFIG, tmp_path)
    importance = extract_feature_importance_frame(result.refit_model, FEATURE_NAMES, "fold_2020")

    assert len(importance) == 63
    assert set(["gain", "total_gain", "weight", "cover", "total_cover"]).issubset(importance.columns)


def test_model_reload_predictions_match(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    result = fit_xgboost_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", CONFIG, tmp_path)

    assert result.reload_verified is True


def test_input_order_shuffle_is_stable() -> None:
    dataset, splits = make_stage7_frames()
    stage6 = make_stage6_predictions(dataset, splits)
    a = build_stage7_outputs(dataset, splits, stage6, dataset_manifest(), feature_manifest(), fold_manifest(), stage6_model_manifest(), CONFIG)
    b = build_stage7_outputs(
        dataset.sample(frac=1, random_state=4).reset_index(drop=True),
        splits.sample(frac=1, random_state=5).reset_index(drop=True),
        stage6.sample(frac=1, random_state=6).reset_index(drop=True),
        dataset_manifest(),
        feature_manifest(),
        fold_manifest(),
        stage6_model_manifest(),
        CONFIG,
    )

    pd.testing.assert_frame_equal(
        a.predictions.sort_values("dataset_row_id").reset_index(drop=True),
        b.predictions.sort_values("dataset_row_id").reset_index(drop=True),
    )


def test_reached_max_estimators_marked(tmp_path: Path) -> None:
    config = {**CONFIG, "xgboost_parameters": {**CONFIG["xgboost_parameters"], "n_estimators": 1, "early_stopping_rounds": 2}}
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    result = fit_xgboost_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", config, tmp_path)

    assert result.selector_metadata["reached_max_estimators"] is True
    assert result.selector_metadata["stopped_early"] is False


def test_development_recommendation_rules() -> None:
    gates = {"stage7_engineering_gate_passed": True}
    thresholds = CONFIG["development_gate_thresholds"]
    proceed = {
        "xgb_pooled_nonoverlap_auc_delta": 0.003,
        "xgb_pooled_nonoverlap_logloss_delta": 0.0005,
        "xgb_beats_logistic_nonoverlap_auc_fold_count": 3,
        "any_fold_auc_below_half": False,
    }
    keep = {**proceed, "xgb_pooled_nonoverlap_auc_delta": 0.001}

    assert development_recommendation(proceed, gates, thresholds) == "PROCEED_TO_LIMITED_TUNING"
    assert development_recommendation(keep, gates, thresholds) == "KEEP_LOGISTIC_AS_PRIMARY_BASELINE"
    assert development_recommendation(proceed, {"stage7_engineering_gate_passed": False}, thresholds) == "INVESTIGATE_PIPELINE"


def test_run_stage7_writes_outputs_and_preserves_inputs(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames(all_folds=True)
    stage6 = make_stage6_predictions(dataset, splits, fold_names=["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"])
    dataset_path = tmp_path / "dataset.parquet"
    split_path = tmp_path / "splits.parquet"
    stage6_path = tmp_path / "stage6_predictions.parquet"
    dataset.to_parquet(dataset_path, index=False)
    splits.to_parquet(split_path, index=False)
    stage6.to_parquet(stage6_path, index=False)
    write_json(tmp_path / "dataset_manifest.json", dataset_manifest())
    write_json(tmp_path / "feature_manifest.json", feature_manifest())
    write_json(tmp_path / "fold_manifest.json", fold_manifest(all_folds=True))
    write_json(tmp_path / "stage6_model_manifest.json", stage6_model_manifest())
    config = {**CONFIG, "fold_names": ["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"]}
    write_json(tmp_path / "config.json", config)
    before = {p: sha256(p) for p in [dataset_path, split_path, stage6_path]}

    result = run_stage7(tmp_path / "config.json", tmp_path)

    assert all(sha256(p) == before[p] for p in before)
    assert result["quality_gates"]["no_final_test_predictions"] is True
    assert result["quality_gates"]["stage7_engineering_gate_passed"] is True
    assert (tmp_path / "out/predictions.parquet").exists()
    assert (tmp_path / "reports/model_manifest.json").exists()
    assert (tmp_path / "models/fold_2020/xgboost_fixed_v1.json").exists()


def prediction_row(row_id: int, fold: str, y: int, p: float) -> dict[str, object]:
    return {
        "dataset_row_id": row_id,
        "decision_time": row_id,
        "fold_name": fold,
        "model_name": MODEL_NAME,
        "y_true": y,
        "p_up": p,
        "y_pred": int(p >= 0.5),
        "prediction_threshold": 0.5,
        "evaluation_offset_minutes": 0,
        "is_primary_nonoverlap_evaluation": True,
        "absolute_future_return_bps": 10.0,
        "proxy_boundary_risk_1bps": False,
        "proxy_boundary_risk_2_5bps": False,
        "proxy_boundary_risk_5bps": False,
        "proxy_boundary_risk_10bps": False,
    }


def ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def dataset_manifest() -> dict[str, object]:
    return {
        "dataset_version": "btcusdt_60m_direction_v1",
        "feature_set_version": "kline_v1_63",
        "feature_columns": FEATURE_NAMES,
        "primary_target_columns": ["label_up_60m"],
        "sample_weight_columns": ["sample_weight_uniform", "sample_weight_margin"],
        "forbidden_model_input_columns": [
            "label_up_60m",
            "future_simple_return_60m",
            "future_log_return_60m",
            "absolute_future_return_bps",
            "proxy_margin_bucket",
            "sample_weight_margin",
            "entry_minute_open_time",
            "settlement_minute_open_time",
        ],
        "output_files": {"dataset_sha256": "dataset_sha", "split_assignment_sha256": "split_sha"},
    }


def feature_manifest() -> dict[str, object]:
    return {"feature_set_version": "kline_v1_63", "ordered_feature_names": FEATURE_NAMES, "feature_count": 63}


def stage6_model_manifest() -> dict[str, object]:
    return {
        "feature_columns": FEATURE_NAMES,
        "quality_gates": {"stage6_engineering_gate_passed": True, "final_test_prediction_count": 0},
        "output_files": {"prediction_output_sha256": "stage6_sha"},
    }


def fold_manifest(all_folds: bool = False) -> dict[str, object]:
    years = [2020, 2021, 2022, 2023, 2024] if all_folds else [2020]
    return {
        "split_version": "expanding_yearly_v1",
        "folds": [
            {
                "name": f"fold_{year}",
                "validation_start": f"{year}-01-01T00:00:00Z",
                "validation_end": f"{year + 1}-01-01T00:00:00Z",
                "train_sample_count": 18,
                "validation_sample_count": 12,
            }
            for year in years
        ],
    }


def make_stage7_frames(all_folds: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = [2020, 2021, 2022, 2023, 2024] if all_folds else [2020]
    rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    row_id = 0
    for year in years:
        validation_start = ms(f"{year}-01-01T00:00:00Z")
        inner_start = validation_start - 24 * 60 * 60 * 1000
        times_roles = [
            *[(inner_start - (120 - i * 5) * 60_000, "TRAIN") for i in range(8)],
            *[(inner_start - (55 - i * 5) * 60_000, "TRAIN") for i in range(2)],
            *[(inner_start + i * 5 * 60_000, "TRAIN") for i in range(8)],
            *[(validation_start + i * 5 * 60_000, "VALIDATION") for i in range(12)],
            (ms("2025-01-01T00:00:00Z") + row_id * 60_000, "FINAL_TEST"),
        ]
        for local_index, (decision_time, role) in enumerate(times_roles):
            label = int((local_index + year) % 2 == 0)
            row = {
                "dataset_row_id": row_id,
                "symbol": "BTCUSDT",
                "feature_set_version": "kline_v1_63",
                "dataset_version": "btcusdt_60m_direction_v1",
                "feature_open_time": decision_time - 60_000,
                "decision_time": decision_time,
                "entry_minute_open_time": decision_time,
                "settlement_minute_open_time": decision_time + 3_600_000,
                "continuity_segment_id": 1,
                "label_up_60m": label,
                "future_simple_return_60m": 0.001 if label else -0.001,
                "future_log_return_60m": np.log1p(0.001 if label else -0.001),
                "absolute_future_return_bps": float(local_index % 12),
                "proxy_margin_bucket": "test",
                "proxy_boundary_risk_1bps": (local_index % 12) < 1,
                "proxy_boundary_risk_2_5bps": (local_index % 12) < 3,
                "proxy_boundary_risk_5bps": (local_index % 12) < 5,
                "proxy_boundary_risk_10bps": (local_index % 12) < 10,
                "sample_weight_uniform": 1.0,
                "sample_weight_margin": 2.0,
            }
            signal = 1.0 if label else -1.0
            for feature_index, feature in enumerate(FEATURE_NAMES):
                row[feature] = signal * ((feature_index % 5) + 1) + (local_index * 0.01) + (year - 2020) * 0.1
            split = {
                "dataset_row_id": row_id,
                "decision_time": decision_time,
                "settlement_minute_open_time": decision_time + 3_600_000,
                "final_split_role": "FINAL_TEST" if role == "FINAL_TEST" else "DEVELOPMENT",
                "evaluation_offset_minutes": 0 if role != "VALIDATION" else (local_index - 18) * 5,
                "is_primary_nonoverlap_evaluation": role == "VALIDATION" and (local_index - 18) == 0,
            }
            for fold_year in [2020, 2021, 2022, 2023, 2024]:
                if fold_year == year and role in {"TRAIN", "VALIDATION"}:
                    split[f"fold_{fold_year}_role"] = role
                elif role == "FINAL_TEST":
                    split[f"fold_{fold_year}_role"] = "FINAL_TEST_EXCLUDED"
                else:
                    split[f"fold_{fold_year}_role"] = "OUTSIDE_FOLD"
            rows.append(row)
            split_rows.append(split)
            row_id += 1
    return pd.DataFrame(rows), pd.DataFrame(split_rows)


def make_stage6_predictions(dataset: pd.DataFrame, splits: pd.DataFrame, fold_names: list[str] | None = None) -> pd.DataFrame:
    fold_names = fold_names or ["fold_2020"]
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    rows: list[dict[str, object]] = []
    for fold_name in fold_names:
        valid = joined[joined[f"{fold_name}_role"].eq("VALIDATION")].copy()
        for model_name, offset in [("prior_baseline", 0.0), ("momentum_60m_baseline", 0.02), ("logistic_regression_l2", 0.04)]:
            for _, row in valid.iterrows():
                y = int(row["label_up_60m"])
                p = float(np.clip(0.45 + 0.1 * y + offset, 0.01, 0.99))
                rows.append(
                    {
                        "dataset_row_id": int(row["dataset_row_id"]),
                        "decision_time": int(row["decision_time"]),
                        "fold_name": fold_name,
                        "model_name": model_name,
                        "y_true": y,
                        "p_up": p,
                        "y_pred": int(p >= 0.5),
                        "prediction_threshold": 0.5,
                        "evaluation_offset_minutes": int(row["evaluation_offset_minutes"]),
                        "is_primary_nonoverlap_evaluation": bool(row["is_primary_nonoverlap_evaluation"]),
                        "absolute_future_return_bps": float(row["absolute_future_return_bps"]),
                        "proxy_boundary_risk_1bps": bool(row["proxy_boundary_risk_1bps"]),
                        "proxy_boundary_risk_2_5bps": bool(row["proxy_boundary_risk_2_5bps"]),
                        "proxy_boundary_risk_5bps": bool(row["proxy_boundary_risk_5bps"]),
                        "proxy_boundary_risk_10bps": bool(row["proxy_boundary_risk_10bps"]),
                    }
                )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

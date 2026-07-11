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
from scripts.train_stage8_limited_xgboost_tuning import (  # noqa: E402
    CANDIDATE_NAMES,
    REFERENCE_CANDIDATE_NAME,
    Stage8ValidationError,
    build_candidate_comparison,
    build_candidate_config,
    build_candidate_summary,
    build_stage8_outputs,
    candidate_definitions_sha256,
    compare_candidate_to_stage7_reference,
    fit_stage8_candidate_fold,
    freeze_candidate_definitions,
    inner_split_row_hashes,
    reference_candidate_matches_stage7,
    run_stage8,
    select_development_config,
    validate_candidate_definitions,
    validate_stage8_predictions,
)
from scripts.train_stage7_xgboost_fixed import (  # noqa: E402
    make_inner_time_split,
    read_stage7_dataset,
    validate_feature_manifest,
)
from tests.test_stage7_xgboost_fixed import (  # noqa: E402
    CONFIG as STAGE7_TEST_CONFIG,
    dataset_manifest,
    feature_manifest,
    fold_manifest,
    make_stage6_predictions,
    make_stage7_frames,
    ms,
    prediction_row,
    sha256,
    stage6_model_manifest,
    write_json,
)


BASE_CANDIDATES = [
    {
        "model_name": "xgb_fixed_v1_reference",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 20,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    },
    {
        "model_name": "xgb_depth3_v1",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 20,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    },
    {
        "model_name": "xgb_depth3_regularized_v1",
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 50,
        "gamma": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 20.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    },
    {
        "model_name": "xgb_depth4_regularized_v1",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 50,
        "gamma": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 20.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    },
    {
        "model_name": "xgb_depth5_regularized_v1",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 50,
        "gamma": 0.1,
        "subsample": 0.75,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.5,
        "reg_lambda": 20.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    },
    {
        "model_name": "xgb_low_learning_rate_v1",
        "learning_rate": 0.015,
        "max_depth": 4,
        "min_child_weight": 20,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_estimators": 10,
        "early_stopping_rounds": 3,
    },
]


CONFIG = {
    "dataset_path": "dataset.parquet",
    "split_path": "splits.parquet",
    "dataset_manifest_path": "dataset_manifest.json",
    "fold_manifest_path": "fold_manifest.json",
    "feature_manifest_path": "feature_manifest.json",
    "stage6_prediction_path": "stage6_predictions.parquet",
    "stage7_prediction_path": "stage7_predictions.parquet",
    "stage7_model_manifest_path": "stage7_model_manifest.json",
    "prediction_output_path": "out/predictions.parquet",
    "model_output_dir": "models",
    "report_paths": {
        "main_report": "reports/report.md",
        "metrics_by_candidate_fold": "reports/metrics_by_candidate_fold.csv",
        "metrics_by_candidate_subset": "reports/metrics_by_candidate_subset.csv",
        "metrics_by_candidate_offset": "reports/metrics_by_candidate_offset.csv",
        "candidate_summary": "reports/candidate_summary.csv",
        "candidate_comparison": "reports/candidate_comparison.csv",
        "selection_audit": "reports/selection_audit.json",
        "learning_curves": "reports/learning_curves.csv",
        "calibration_equal_width": "reports/calibration_equal_width.csv",
        "calibration_equal_frequency": "reports/calibration_equal_frequency.csv",
        "feature_importance_by_candidate_fold": "reports/feature_importance_by_candidate_fold.csv",
        "feature_importance_stability": "reports/feature_importance_stability.csv",
        "inner_split_audit": "reports/inner_split_audit.csv",
        "model_manifest": "reports/model_manifest.json",
    },
    "log_path": "reports/stage8.log",
    "candidate_definitions": BASE_CANDIDATES,
    "expected_candidate_count": 6,
    "fold_names": ["fold_2020"],
    "feature_count": 63,
    "xgboost_common_parameters": {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "max_bin": 256,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "verbosity": 0,
        "validate_parameters": True,
    },
    "early_stopping_window_days": 1,
    "inner_purge_horizon_minutes": 60,
    "minimum_inner_early_stop_samples": 4,
    "fixed_prediction_threshold": 0.5,
    "evaluation_subsets": STAGE7_TEST_CONFIG["evaluation_subsets"],
    "calibration_bins": 2,
    "selection_rules": {
        "pooled_dense_logloss_max_degradation": 0.0005,
        "offset_macro_logloss_max_degradation": 0.0005,
        "pooled_dense_brier_max_degradation": 0.00025,
        "dense_fold_logloss_bad_delta": 0.001,
        "dense_fold_logloss_bad_fold_limit": 2,
        "auc_tie_tolerance": 0.001,
        "condition_a_offset_macro_auc_gain": 0.0015,
        "condition_a_offset_macro_logloss_max_degradation": 0.0003,
        "condition_a_dense_auc_fold_wins": 3,
        "condition_b_offset_macro_logloss_improvement": 0.0005,
        "condition_b_offset_macro_auc_max_decline": 0.0005,
        "condition_b_dense_logloss_fold_wins": 3,
    },
    "reference_reproduction_tolerances": {
        "reference_prediction_absolute_tolerance": 1e-7,
        "reference_metric_absolute_tolerance": 1e-10,
    },
    "random_seed": 42,
    "n_jobs": 1,
    "parquet_compression": "snappy",
    "hash_algorithm": "sha256",
    "numeric_tolerances": {"prediction_atol": 1e-12},
}


def test_candidate_definitions_are_exactly_six_unique_and_frozen() -> None:
    frozen = freeze_candidate_definitions(CONFIG["candidate_definitions"])
    result = validate_candidate_definitions(frozen, CONFIG, stage7_reference_parameters())

    assert result["candidate_count"] == 6
    assert [c["model_name"] for c in frozen] == CANDIDATE_NAMES
    assert len({c["model_name"] for c in frozen}) == 6
    assert candidate_definitions_sha256(frozen) == result["candidate_definitions_sha256"]
    CONFIG["candidate_definitions"][0]["max_depth"] = 99
    assert frozen[0]["max_depth"] == 4
    CONFIG["candidate_definitions"][0]["max_depth"] = 4


def test_candidate_count_or_duplicate_name_fails() -> None:
    bad = BASE_CANDIDATES[:-1]
    with pytest.raises(Stage8ValidationError, match="six"):
        validate_candidate_definitions(bad, CONFIG, stage7_reference_parameters())

    duplicate = [dict(c) for c in BASE_CANDIDATES]
    duplicate[-1]["model_name"] = duplicate[0]["model_name"]
    with pytest.raises(Stage8ValidationError, match="unique"):
        validate_candidate_definitions(duplicate, CONFIG, stage7_reference_parameters())


def test_candidate_one_matches_stage7_and_candidate_six_limits() -> None:
    frozen = freeze_candidate_definitions(BASE_CANDIDATES)

    assert reference_candidate_matches_stage7(frozen[0], stage7_reference_parameters())
    c1 = build_candidate_config(CONFIG, frozen[0])
    c6 = build_candidate_config(CONFIG, frozen[-1])
    assert c1["xgboost_parameters"]["n_estimators"] == 8
    assert c1["xgboost_parameters"]["early_stopping_rounds"] == 2
    assert c6["xgboost_parameters"]["n_estimators"] == 10
    assert c6["xgboost_parameters"]["early_stopping_rounds"] == 3
    assert c6["xgboost_parameters"]["learning_rate"] == 0.015
    assert c6["xgboost_parameters"]["scale_pos_weight"] == 1.0


def test_feature_manifest_and_final_test_dataset_loading_rules(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    dataset_path = tmp_path / "dataset.parquet"
    dataset.to_parquet(dataset_path, index=False)
    split_columns = [
        "dataset_row_id",
        "decision_time",
        "settlement_minute_open_time",
        "final_split_role",
        "evaluation_offset_minutes",
        "is_primary_nonoverlap_evaluation",
        "fold_2020_role",
    ]
    loaded = read_stage7_dataset(dataset_path, splits[split_columns], FEATURE_NAMES, ["fold_2020"])
    features = validate_feature_manifest(dataset_manifest(), feature_manifest(), 63)

    assert features == FEATURE_NAMES
    final_ids = set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"])
    assert set(loaded["dataset_row_id"]).isdisjoint(final_ids)
    assert "label_up_60m" not in features
    assert "sample_weight_margin" not in features
    assert "settlement_minute_open_time" not in features


def test_inner_split_hashes_are_reused_for_all_candidates() -> None:
    dataset, splits = make_stage7_frames()
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    outer_train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    inner = make_inner_time_split(outer_train, ms("2020-01-01T00:00:00Z"), 1, 60)
    expected = inner_split_row_hashes(inner)
    by_candidate = {candidate["model_name"]: inner_split_row_hashes(inner) for candidate in BASE_CANDIDATES}

    assert len(set(tuple(v.items()) for v in by_candidate.values())) == 1
    assert expected["inner_fit_count"] == 8
    assert expected["inner_early_stop_count"] == 8
    assert expected["inner_fit_dataset_row_id_sha256"] != expected["inner_early_stop_dataset_row_id_sha256"]


def test_build_outputs_prediction_contract_and_stage7_reference_reproduction(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    stage6 = make_stage6_predictions(dataset, splits)
    stage7_reference = make_stage7_reference_predictions(dataset, splits)
    stage7_manifest = stage7_model_manifest_for_test()

    outputs = build_stage8_outputs(
        dataset,
        splits,
        stage6,
        stage7_reference,
        dataset_manifest(),
        feature_manifest(),
        fold_manifest(),
        stage7_manifest,
        CONFIG,
        root=tmp_path,
    )

    predictions = outputs.predictions
    validate_stage8_predictions(predictions, splits, CONFIG["fold_names"], CANDIDATE_NAMES)
    assert set(predictions["model_name"]) == set(CANDIDATE_NAMES)
    assert predictions.groupby(["model_name", "fold_name", "dataset_row_id"]).size().eq(1).all()
    assert predictions["p_up"].between(0, 1).all()
    assert (predictions["prediction_threshold"] == 0.5).all()
    assert set(predictions["dataset_row_id"]).isdisjoint(set(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"]))
    reference_check = outputs.reference_reproduction
    assert reference_check["reference_config_reproduced"] is True
    assert reference_check["max_abs_prediction_diff"] <= CONFIG["reference_reproduction_tolerances"]["reference_prediction_absolute_tolerance"]
    assert (tmp_path / "models/xgb_fixed_v1_reference/fold_2020/refit_model.json").exists()


def test_no_validation_labels_used_for_refit_predictions(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames()
    flipped = dataset.copy()
    valid_ids = set(splits.loc[splits["fold_2020_role"].eq("VALIDATION"), "dataset_row_id"])
    flipped.loc[flipped["dataset_row_id"].isin(valid_ids), "label_up_60m"] = 1 - flipped.loc[flipped["dataset_row_id"].isin(valid_ids), "label_up_60m"]
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    flipped_joined = flipped.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    train = joined[joined["fold_2020_role"].eq("TRAIN")].copy()
    valid = joined[joined["fold_2020_role"].eq("VALIDATION")].copy()
    flipped_valid = flipped_joined[flipped_joined["fold_2020_role"].eq("VALIDATION")].copy()

    a = fit_stage8_candidate_fold(train, valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", BASE_CANDIDATES[0], CONFIG, tmp_path / "a")
    b = fit_stage8_candidate_fold(train, flipped_valid, FEATURE_NAMES, fold_manifest()["folds"][0], "fold_2020", BASE_CANDIDATES[0], CONFIG, tmp_path / "b")

    np.testing.assert_allclose(a.predictions["p_up"], b.predictions["p_up"])


def test_metrics_offset_macro_fold_macro_and_calibration() -> None:
    rows = []
    for model_name, shift in [(REFERENCE_CANDIDATE_NAME, 0.0), ("xgb_depth3_v1", 0.05)]:
        for fold in ["fold_2020", "fold_2021"]:
            for offset in range(0, 60, 5):
                y = int(((offset // 5) + (1 if fold == "fold_2021" else 0)) % 2 == 0)
                p = float(np.clip(0.45 + 0.2 * y + shift, 0.01, 0.99))
                row_id = offset + (0 if fold == "fold_2020" else 1000)
                row = prediction_row(row_id, fold, y, p)
                row["model_name"] = model_name
                row["evaluation_offset_minutes"] = offset
                row["is_primary_nonoverlap_evaluation"] = offset == 0
                rows.append(row)
    predictions = pd.DataFrame(rows)
    metrics = build_candidate_summary(predictions, pd.DataFrame())
    ref = metrics[metrics["model_name"].eq(REFERENCE_CANDIDATE_NAME)].iloc[0]

    assert np.isclose(ref["offset_macro_auc"], 1.0)
    assert ref["offset_auc_std"] == 0.0
    assert ref["dense_fold_auc_mean"] == 1.0
    assert ref["fold_auc_above_0_5_count"] == 2
    assert "weakest_offset" in metrics.columns


def test_candidate_comparison_delta_direction_and_stage6_join() -> None:
    predictions = pd.DataFrame(
        [
            {**prediction_row(0, "fold_2020", 0, 0.2), "model_name": REFERENCE_CANDIDATE_NAME},
            {**prediction_row(1, "fold_2020", 1, 0.8), "model_name": REFERENCE_CANDIDATE_NAME},
            {**prediction_row(0, "fold_2020", 0, 0.1), "model_name": "xgb_depth3_v1"},
            {**prediction_row(1, "fold_2020", 1, 0.9), "model_name": "xgb_depth3_v1"},
        ]
    )
    comparison = build_candidate_comparison(predictions, REFERENCE_CANDIDATE_NAME)

    row = comparison[(comparison["model_name"].eq("xgb_depth3_v1")) & (comparison["subset_name"].eq("DENSE"))].iloc[0]
    assert row["delta_roc_auc"] == row["candidate_roc_auc"] - row["reference_roc_auc"]
    assert row["delta_log_loss"] == row["candidate_log_loss"] - row["reference_log_loss"]
    assert row["delta_log_loss"] < 0


def test_stage7_y_true_or_decision_mismatch_fails() -> None:
    dataset, splits = make_stage7_frames()
    stage7 = make_stage7_reference_predictions(dataset, splits)
    candidate = stage7.copy()
    candidate["model_name"] = REFERENCE_CANDIDATE_NAME
    bad = stage7.copy()
    bad.loc[0, "y_true"] = 1 - bad.loc[0, "y_true"]

    with pytest.raises(Stage8ValidationError, match="y_true"):
        compare_candidate_to_stage7_reference(candidate, bad, CONFIG["reference_reproduction_tolerances"])


def test_probability_quality_and_engineering_disqualifications() -> None:
    summary = pd.DataFrame(
        [
            candidate_summary_row(REFERENCE_CANDIDATE_NAME, 0.550, 0.6900, 0.2480, dense_min=0.53),
            candidate_summary_row("xgb_depth3_v1", 0.552, 0.6904, 0.2481, dense_min=0.52),
            candidate_summary_row("xgb_depth3_regularized_v1", 0.553, 0.6910, 0.2484, dense_min=0.52),
            candidate_summary_row("xgb_depth4_regularized_v1", 0.540, 0.6890, 0.2477, dense_min=0.49),
        ]
    )
    comparison = pd.DataFrame(
        [
            candidate_comparison_row("xgb_depth3_v1", "fold_2020", 0.001, -0.0001, 0.0),
            candidate_comparison_row("xgb_depth3_v1", "fold_2021", 0.001, 0.0002, 0.0),
            candidate_comparison_row("xgb_depth3_regularized_v1", "fold_2020", 0.002, 0.002, 0.0),
        ]
    )
    audit = select_development_config(summary, comparison, CONFIG["selection_rules"], reference_reproduced=True, engineering_gate=True)

    statuses = {row["model_name"]: row["qualification_status"] for row in audit["candidate_qualifications"]}
    assert statuses["xgb_depth3_regularized_v1"] == "DISQUALIFIED_PROBABILITY_QUALITY"
    assert statuses["xgb_depth4_regularized_v1"] == "DISQUALIFIED_ENGINEERING"


def test_tie_rules_and_complexity_decision() -> None:
    summary = pd.DataFrame(
        [
            candidate_summary_row(REFERENCE_CANDIDATE_NAME, 0.5500, 0.6900, 0.2480, max_depth=4, median_trees=20),
            candidate_summary_row("xgb_depth3_v1", 0.5505, 0.6900, 0.2480, max_depth=3, median_trees=20),
        ]
    )
    comparison = pd.DataFrame([candidate_comparison_row("xgb_depth3_v1", f"fold_{i}", 0.0001, 0.0, 0.0) for i in range(5)])
    audit = select_development_config(summary, comparison, CONFIG["selection_rules"], reference_reproduced=True, engineering_gate=True)

    assert audit["ranked_candidates"][0]["model_name"] == "xgb_depth3_v1"
    assert audit["selected_development_config"] == REFERENCE_CANDIDATE_NAME
    assert audit["improvement_not_material"] is True


def test_material_gain_condition_a_and_b() -> None:
    summary_a = pd.DataFrame(
        [
            candidate_summary_row(REFERENCE_CANDIDATE_NAME, 0.5500, 0.6900, 0.2480),
            candidate_summary_row("xgb_depth3_v1", 0.5520, 0.6901, 0.2480),
        ]
    )
    comp_a = pd.DataFrame([candidate_comparison_row("xgb_depth3_v1", f"fold_{i}", 0.001, 0.0, 0.0) for i in range(5)])
    audit_a = select_development_config(summary_a, comp_a, CONFIG["selection_rules"], reference_reproduced=True, engineering_gate=True)

    summary_b = pd.DataFrame(
        [
            candidate_summary_row(REFERENCE_CANDIDATE_NAME, 0.5500, 0.6900, 0.2480),
            candidate_summary_row("xgb_depth3_v1", 0.5498, 0.6894, 0.2479),
        ]
    )
    comp_b = pd.DataFrame([candidate_comparison_row("xgb_depth3_v1", f"fold_{i}", 0.0, -0.001, -0.0001) for i in range(5)])
    audit_b = select_development_config(summary_b, comp_b, CONFIG["selection_rules"], reference_reproduced=True, engineering_gate=True)

    assert audit_a["selected_development_config"] == "xgb_depth3_v1"
    assert audit_a["minimum_gain_check"]["condition_a_met"] is True
    assert audit_b["selected_development_config"] == "xgb_depth3_v1"
    assert audit_b["minimum_gain_check"]["condition_b_met"] is True


def test_reference_reproduction_failure_blocks_selection() -> None:
    summary = pd.DataFrame(
        [
            candidate_summary_row(REFERENCE_CANDIDATE_NAME, 0.55, 0.69, 0.248),
            candidate_summary_row("xgb_depth3_v1", 0.60, 0.68, 0.24),
        ]
    )
    audit = select_development_config(summary, pd.DataFrame(), CONFIG["selection_rules"], reference_reproduced=False, engineering_gate=True)

    assert audit["selected_development_config"] == REFERENCE_CANDIDATE_NAME
    assert audit["development_recommendation"] == "INVESTIGATE_PIPELINE"


def test_run_stage8_writes_outputs_and_preserves_stage5_to_stage7_inputs(tmp_path: Path) -> None:
    dataset, splits = make_stage7_frames(all_folds=True)
    stage6 = make_stage6_predictions(dataset, splits, fold_names=CONFIG_ALL_FOLDS["fold_names"])
    stage7 = make_stage7_reference_predictions(dataset, splits, fold_names=CONFIG_ALL_FOLDS["fold_names"])
    paths = write_stage8_fixture_files(tmp_path, dataset, splits, stage6, stage7, CONFIG_ALL_FOLDS)
    before = {p: sha256(p) for p in paths}

    result = run_stage8(tmp_path / "config.json", tmp_path)

    assert all(sha256(p) == before[p] for p in before)
    assert result["quality_gates"]["candidate_set_frozen_before_run"] is True
    assert result["quality_gates"]["candidate_count_is_six"] is True
    assert result["quality_gates"]["no_final_test_feature_matrix"] is True
    assert result["quality_gates"]["stage8_engineering_gate_passed"] is True
    assert result["selected_development_config"] in CANDIDATE_NAMES
    assert (tmp_path / "out/predictions.parquet").exists()
    assert (tmp_path / "reports/selection_audit.json").exists()
    assert (tmp_path / "models/xgb_low_learning_rate_v1/fold_2024/refit_model.json").exists()


CONFIG_ALL_FOLDS = {**CONFIG, "fold_names": ["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"]}


def stage7_reference_parameters() -> dict[str, object]:
    return {
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 20,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_estimators": 8,
        "early_stopping_rounds": 2,
    }


def stage7_model_manifest_for_test() -> dict[str, object]:
    return {
        "feature_columns": FEATURE_NAMES,
        "xgboost_parameters": {
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 20,
            "gamma": 0.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 10.0,
            "n_estimators": 8,
            "early_stopping_rounds": 2,
        },
        "folds": {},
        "quality_gates": {"stage7_engineering_gate_passed": True, "final_test_prediction_count": 0},
    }


def make_stage7_reference_predictions(dataset: pd.DataFrame, splits: pd.DataFrame, fold_names: list[str] | None = None) -> pd.DataFrame:
    fold_names = fold_names or ["fold_2020"]
    joined = dataset.merge(splits, on=["dataset_row_id", "decision_time", "settlement_minute_open_time"])
    fold_info = {fold["name"]: fold for fold in fold_manifest(all_folds=len(fold_names) > 1)["folds"]}
    frames = []
    for fold in fold_names:
        train = joined[joined[f"{fold}_role"].eq("TRAIN")].copy()
        valid = joined[joined[f"{fold}_role"].eq("VALIDATION")].copy()
        result = fit_stage8_candidate_fold(train, valid, FEATURE_NAMES, fold_info[fold], fold, BASE_CANDIDATES[0], CONFIG, None)
        pred = result.predictions.copy()
        pred["model_name"] = "xgboost_fixed_v1"
        frames.append(pred)
    return pd.concat(frames, ignore_index=True)


def candidate_summary_row(
    model_name: str,
    offset_auc: float,
    offset_logloss: float,
    offset_brier: float,
    *,
    dense_min: float = 0.52,
    max_depth: int = 4,
    median_trees: int = 10,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "pooled_dense_auc": offset_auc,
        "pooled_dense_logloss": offset_logloss,
        "pooled_dense_brier": offset_brier,
        "nonoverlap_offset00_auc": offset_auc,
        "nonoverlap_offset00_logloss": offset_logloss,
        "nonoverlap_offset00_brier": offset_brier,
        "offset_macro_auc": offset_auc,
        "offset_auc_std": 0.001,
        "offset_auc_min": offset_auc - 0.01,
        "offset_auc_max": offset_auc + 0.01,
        "offset_auc_range": 0.02,
        "offset_macro_logloss": offset_logloss,
        "offset_logloss_std": 0.001,
        "offset_logloss_max": offset_logloss + 0.002,
        "offset_macro_brier": offset_brier,
        "offset_brier_std": 0.0001,
        "dense_fold_auc_mean": offset_auc,
        "dense_fold_auc_std": 0.001,
        "dense_fold_auc_min": dense_min,
        "dense_fold_logloss_mean": offset_logloss,
        "dense_fold_logloss_std": 0.001,
        "nonoverlap_fold_auc_mean": offset_auc,
        "nonoverlap_fold_auc_std": 0.001,
        "nonoverlap_fold_auc_min": dense_min,
        "nonoverlap_fold_logloss_mean": offset_logloss,
        "nonoverlap_fold_logloss_std": 0.001,
        "fold_auc_above_0_5_count": 5 if dense_min >= 0.5 else 4,
        "max_depth": max_depth,
        "median_best_n_estimators": median_trees,
        "best_year": "fold_2020",
        "worst_year": "fold_2021",
        "worst_year_auc": dense_min,
        "has_fold_auc_below_0_5": dense_min < 0.5,
    }


def candidate_comparison_row(model_name: str, fold: str, auc_delta: float, logloss_delta: float, brier_delta: float) -> dict[str, object]:
    return {
        "model_name": model_name,
        "reference_model_name": REFERENCE_CANDIDATE_NAME,
        "fold_name": fold,
        "subset_name": "DENSE",
        "delta_roc_auc": auc_delta,
        "delta_log_loss": logloss_delta,
        "delta_brier": brier_delta,
        "delta_accuracy": 0.0,
        "delta_mcc": 0.0,
    }


def write_stage8_fixture_files(
    tmp_path: Path,
    dataset: pd.DataFrame,
    splits: pd.DataFrame,
    stage6: pd.DataFrame,
    stage7: pd.DataFrame,
    config: dict[str, object],
) -> list[Path]:
    dataset_path = tmp_path / "dataset.parquet"
    split_path = tmp_path / "splits.parquet"
    stage6_path = tmp_path / "stage6_predictions.parquet"
    stage7_path = tmp_path / "stage7_predictions.parquet"
    dataset.to_parquet(dataset_path, index=False)
    splits.to_parquet(split_path, index=False)
    stage6.to_parquet(stage6_path, index=False)
    stage7.to_parquet(stage7_path, index=False)
    write_json(tmp_path / "dataset_manifest.json", dataset_manifest())
    write_json(tmp_path / "feature_manifest.json", feature_manifest())
    write_json(tmp_path / "fold_manifest.json", fold_manifest(all_folds=True))
    write_json(tmp_path / "stage7_model_manifest.json", stage7_model_manifest_for_test())
    write_json(tmp_path / "config.json", config)
    return [dataset_path, split_path, stage6_path, stage7_path, tmp_path / "dataset_manifest.json", tmp_path / "feature_manifest.json", tmp_path / "fold_manifest.json", tmp_path / "stage7_model_manifest.json"]

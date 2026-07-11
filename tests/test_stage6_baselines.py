from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, balanced_accuracy_score, log_loss, matthews_corrcoef, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_stage4_features import FEATURE_NAMES  # noqa: E402
from scripts.train_stage6_baselines import (  # noqa: E402
    ALLOWED_MODEL_NAMES,
    Stage6ValidationError,
    build_stage6_outputs,
    calibration_table,
    compute_classification_metrics,
    evaluate_prediction_subsets,
    fit_logistic_fold,
    make_momentum_predictions,
    make_prior_predictions,
    pooled_and_macro_summary,
    prepare_feature_matrix,
    run_stage6,
    validate_feature_manifest,
    validate_fold_integrity,
)


CONFIG = {
    "dataset_path": "dataset.parquet",
    "split_path": "splits.parquet",
    "dataset_manifest_path": "dataset_manifest.json",
    "fold_manifest_path": "fold_manifest.json",
    "feature_manifest_path": "feature_manifest.json",
    "prediction_output_path": "out/predictions.parquet",
    "model_output_dir": "models",
    "report_paths": {
        "main_report": "reports/report.md",
        "metrics_by_fold": "reports/metrics_by_fold.csv",
        "metrics_by_subset": "reports/metrics_by_subset.csv",
        "metrics_by_offset": "reports/metrics_by_offset.csv",
        "oof_summary": "reports/oof_summary.csv",
        "calibration_equal_width": "reports/calibration_equal_width.csv",
        "calibration_equal_frequency": "reports/calibration_equal_frequency.csv",
        "logistic_coefficients_by_fold": "reports/coefficients_by_fold.csv",
        "logistic_coefficient_stability": "reports/coefficient_stability.csv",
        "preprocessing_audit": "reports/preprocessing_audit.json",
        "model_manifest": "reports/model_manifest.json",
    },
    "log_path": "reports/stage6.log",
    "fold_names": ["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"],
    "model_definitions": {
        "prior_baseline": "training label prior probability",
        "momentum_60m_baseline": "log_return_60m direction continuation with train conditional probabilities",
        "logistic_regression_l2": "StandardScaler plus LogisticRegression L2",
    },
    "logistic_regression_parameters": {
        "penalty": "l2",
        "C": 1.0,
        "solver": "lbfgs",
        "fit_intercept": True,
        "class_weight": None,
        "max_iter": 2000,
        "tol": 1e-6,
        "random_state": 42,
    },
    "scaler_parameters": {"with_mean": True, "with_std": True},
    "fixed_prediction_threshold": 0.5,
    "momentum_baseline_alpha": 1.0,
    "evaluation_subsets": {
        "dense": "DENSE",
        "nonoverlap": "NONOVERLAP_OFFSET_00",
        "offsets": list(range(0, 60, 5)),
        "margin_subsets": ["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"],
    },
    "calibration_bins": 10,
    "random_seed": 42,
    "numeric_tolerances": {"scaler_atol": 1e-12, "prediction_atol": 1e-12},
    "parquet_compression": "snappy",
    "hash_algorithm": "sha256",
}


def test_feature_columns_strictly_come_from_manifest_and_keep_order() -> None:
    features = validate_feature_manifest(dataset_manifest(), feature_manifest())

    assert features == FEATURE_NAMES
    assert len(features) == 63


def test_feature_manifest_mismatch_fails() -> None:
    bad_dataset_manifest = dataset_manifest()
    bad_dataset_manifest["feature_columns"] = FEATURE_NAMES[:-1] + ["unexpected_feature"]

    with pytest.raises(Stage6ValidationError, match="feature_columns do not match"):
        validate_feature_manifest(bad_dataset_manifest, feature_manifest())


def test_target_future_and_weight_fields_never_enter_x() -> None:
    dataset, _ = make_stage6_frames()
    X = prepare_feature_matrix(dataset, FEATURE_NAMES)

    assert list(X.columns) == FEATURE_NAMES
    forbidden = {
        "label_up_60m",
        "future_simple_return_60m",
        "future_log_return_60m",
        "absolute_future_return_bps",
        "proxy_margin_bucket",
        "sample_weight_margin",
        "decision_time",
        "settlement_minute_open_time",
    }
    assert forbidden.isdisjoint(X.columns)


def test_nonfinite_feature_fails() -> None:
    dataset, _ = make_stage6_frames()
    dataset.loc[0, FEATURE_NAMES[0]] = np.inf

    with pytest.raises(Stage6ValidationError, match="non-finite"):
        prepare_feature_matrix(dataset, FEATURE_NAMES)


def test_fold_integrity_excludes_final_test_and_checks_class_presence() -> None:
    dataset, splits = make_stage6_frames()
    fold_manifest_data = fold_manifest()
    integrity = validate_fold_integrity(dataset, splits, fold_manifest_data["folds"][0], "fold_2020")

    assert integrity["train_count"] == 6
    assert integrity["validation_count"] == 12
    assert integrity["final_test_in_train_or_validation"] == 0
    assert integrity["train_validation_overlap_count"] == 0
    assert integrity["train_max_settlement_minute_open_time"] < integrity["validation_min_decision_time"]


def test_final_test_in_fit_or_prediction_roles_fails() -> None:
    dataset, splits = make_stage6_frames()
    splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "fold_2020_role"] = "VALIDATION"

    with pytest.raises(Stage6ValidationError, match="FINAL_TEST"):
        validate_fold_integrity(dataset, splits, fold_manifest()["folds"][0], "fold_2020")


def test_single_class_fold_fails_before_training() -> None:
    dataset, splits = make_stage6_frames()
    dataset.loc[splits["fold_2020_role"].eq("TRAIN").to_numpy(), "label_up_60m"] = 1

    with pytest.raises(Stage6ValidationError, match="both classes"):
        validate_fold_integrity(dataset, splits, fold_manifest()["folds"][0], "fold_2020")


def test_prior_baseline_uses_only_training_labels_and_tie_predicts_one() -> None:
    y_train = np.array([1, 0, 1, 0])
    validation = pd.DataFrame({"dataset_row_id": [10, 11], "decision_time": [1, 2]})
    pred, meta = make_prior_predictions(y_train, validation, "fold_x", threshold=0.5)

    assert meta["train_up_probability"] == 0.5
    assert meta["predicted_class"] == 1
    assert pred["p_up"].tolist() == [0.5, 0.5]
    assert pred["y_pred"].tolist() == [1, 1]
    assert meta["train_label_distribution"] == {"0": 2, "1": 2}


def test_prior_baseline_ignores_validation_labels() -> None:
    y_train = np.array([1, 1, 1, 0])
    validation_a = pd.DataFrame({"dataset_row_id": [10, 11], "decision_time": [1, 2], "label_up_60m": [0, 0]})
    validation_b = validation_a.assign(label_up_60m=[1, 1])

    pred_a, meta_a = make_prior_predictions(y_train, validation_a, "fold_x", threshold=0.5)
    pred_b, meta_b = make_prior_predictions(y_train, validation_b, "fold_x", threshold=0.5)

    pd.testing.assert_frame_equal(pred_a, pred_b)
    assert meta_a == meta_b


def test_momentum_baseline_hard_rule_and_laplace_probabilities_use_train_only() -> None:
    train = pd.DataFrame({"log_return_60m": [0.1, 0.2, -0.1, 0.0], "label_up_60m": [1, 0, 0, 0]})
    validation = pd.DataFrame({"dataset_row_id": [1, 2], "decision_time": [1, 2], "log_return_60m": [0.01, 0.0]})
    pred, meta = make_momentum_predictions(train, validation, "fold_x", alpha=1.0)

    assert meta["positive_group"]["group_count"] == 2
    assert meta["positive_group"]["up_count"] == 1
    assert meta["positive_group"]["raw_up_probability"] == 0.5
    assert meta["positive_group"]["smoothed_up_probability"] == 0.5
    assert meta["nonpositive_group"]["group_count"] == 2
    assert meta["nonpositive_group"]["up_count"] == 0
    assert meta["nonpositive_group"]["raw_up_probability"] == 0.0
    assert meta["nonpositive_group"]["smoothed_up_probability"] == 0.25
    assert pred["p_up"].tolist() == [0.5, 0.25]
    assert pred["y_pred"].tolist() == [1, 0]


def test_momentum_validation_labels_do_not_change_probabilities() -> None:
    train = pd.DataFrame({"log_return_60m": [0.1, -0.1], "label_up_60m": [1, 0]})
    validation_a = pd.DataFrame({"dataset_row_id": [1, 2], "decision_time": [1, 2], "log_return_60m": [0.1, -0.1], "label_up_60m": [0, 0]})
    validation_b = validation_a.assign(label_up_60m=[1, 1])

    pred_a, meta_a = make_momentum_predictions(train, validation_a, "fold_x", alpha=1.0)
    pred_b, meta_b = make_momentum_predictions(train, validation_b, "fold_x", alpha=1.0)

    pd.testing.assert_frame_equal(pred_a, pred_b)
    assert meta_a == meta_b


def test_standard_scaler_is_fit_on_train_only_and_validation_extreme_does_not_change_parameters() -> None:
    dataset, splits = make_stage6_frames()
    features = FEATURE_NAMES
    train = dataset.loc[splits["fold_2020_role"].eq("TRAIN")].copy()
    valid = dataset.loc[splits["fold_2020_role"].eq("VALIDATION")].copy()
    valid_extreme = valid.copy()
    valid_extreme[features[0]] = valid_extreme[features[0]] + 1_000_000.0

    _, meta_a, _ = fit_logistic_fold(train, valid, features, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)
    _, meta_b, _ = fit_logistic_fold(train, valid_extreme, features, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)

    np.testing.assert_allclose(meta_a["scaler"]["mean_"], meta_b["scaler"]["mean_"])
    np.testing.assert_allclose(meta_a["scaler"]["var_"], meta_b["scaler"]["var_"])
    np.testing.assert_allclose(meta_a["scaler"]["scale_"], meta_b["scaler"]["scale_"])


def test_validation_label_change_does_not_change_fitted_logistic_model() -> None:
    dataset, splits = make_stage6_frames()
    train = dataset.loc[splits["fold_2020_role"].eq("TRAIN")].copy()
    valid = dataset.loc[splits["fold_2020_role"].eq("VALIDATION")].copy()
    valid_flipped = valid.copy()
    valid_flipped["label_up_60m"] = 1 - valid_flipped["label_up_60m"]

    pred_a, meta_a, _ = fit_logistic_fold(train, valid, FEATURE_NAMES, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)
    pred_b, meta_b, _ = fit_logistic_fold(train, valid_flipped, FEATURE_NAMES, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)

    np.testing.assert_allclose(pred_a["p_up"], pred_b["p_up"])
    np.testing.assert_allclose(meta_a["model"]["coef_"], meta_b["model"]["coef_"])


def test_logistic_parameters_fixed_threshold_and_no_sample_weight_margin() -> None:
    dataset, splits = make_stage6_frames()
    train = dataset.loc[splits["fold_2020_role"].eq("TRAIN")].copy()
    valid = dataset.loc[splits["fold_2020_role"].eq("VALIDATION")].copy()
    pred, meta, _ = fit_logistic_fold(train, valid, FEATURE_NAMES, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)

    params = meta["logistic_regression_parameters"]
    assert params["penalty"] == "l2"
    assert params["C"] == 1.0
    assert params["solver"] == "lbfgs"
    assert params["class_weight"] is None
    assert params["max_iter"] == 2000
    assert meta["training_weight_scheme"] == "uniform"
    assert meta["sample_weight_margin_used"] is False
    assert (pred["prediction_threshold"] == 0.5).all()
    assert pred["y_pred"].equals((pred["p_up"] >= 0.5).astype("int8"))
    assert pred["p_up"].between(0, 1).all()


def test_prediction_output_has_one_validation_row_per_model_and_no_train_or_final_test() -> None:
    dataset, splits = make_stage6_frames()
    output = build_stage6_outputs(dataset, splits, dataset_manifest(), feature_manifest(), fold_manifest(), CONFIG)
    predictions = output.predictions

    assert set(predictions["model_name"]) == ALLOWED_MODEL_NAMES
    validation_ids = set(splits.loc[splits["fold_2020_role"].eq("VALIDATION"), "dataset_row_id"])
    assert set(predictions["dataset_row_id"]) == validation_ids
    assert predictions.groupby(["fold_name", "model_name", "dataset_row_id"]).size().eq(1).all()
    assert len(predictions) == len(validation_ids) * 3
    assert not predictions["dataset_row_id"].isin(splits.loc[splits["fold_2020_role"].eq("TRAIN"), "dataset_row_id"]).any()
    assert not predictions["dataset_row_id"].isin(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"]).any()


def test_classification_metrics_match_sklearn_and_confusion_fields() -> None:
    y_true = np.array([1, 0, 1, 0])
    p_up = np.array([0.9, 0.2, 0.4, 0.8])
    y_pred = np.array([1, 0, 0, 1])
    metrics = compute_classification_metrics(y_true, p_up, y_pred)

    assert metrics["sample_count"] == 4
    assert metrics["positive_count"] == 2
    assert metrics["negative_count"] == 2
    assert metrics["accuracy"] == 0.5
    assert metrics["balanced_accuracy"] == balanced_accuracy_score(y_true, y_pred)
    assert metrics["roc_auc"] == roc_auc_score(y_true, p_up)
    assert metrics["average_precision"] == average_precision_score(y_true, p_up)
    assert metrics["mcc"] == matthews_corrcoef(y_true, y_pred)
    assert metrics["log_loss"] == log_loss(y_true, p_up, labels=[0, 1])
    assert metrics["brier_score"] == float(np.mean((p_up - y_true) ** 2))
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["specificity"] == 0.5
    assert metrics["negative_predictive_value"] == 0.5


def test_single_class_roc_auc_is_missing_not_fabricated() -> None:
    metrics = compute_classification_metrics(np.array([1, 1]), np.array([0.2, 0.8]), np.array([0, 1]))

    assert pd.isna(metrics["roc_auc"])
    assert metrics["roc_auc_missing_reason"] == "single_class_validation_subset"


def test_dense_nonoverlap_offsets_and_margin_subsets_are_evaluated() -> None:
    dataset, splits = make_stage6_frames()
    output = build_stage6_outputs(dataset, splits, dataset_manifest(), feature_manifest(), fold_manifest(), CONFIG)
    metrics = evaluate_prediction_subsets(output.predictions)
    subset_names = set(metrics["subset_name"])

    assert "DENSE" in subset_names
    assert "NONOVERLAP_OFFSET_00" in subset_names
    for offset in range(0, 60, 5):
        assert f"OFFSET_{offset:02d}" in subset_names
    for name in ["ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"]:
        assert name in subset_names


def test_pooled_oof_and_fold_macro_are_distinct() -> None:
    rows = []
    for fold, y_true in [("fold_a", [1, 0]), ("fold_b", [1, 1, 1, 0, 0, 0])]:
        for i, y in enumerate(y_true):
            p_up = 0.8 if (fold == "fold_a" and y == 1) else 0.2 if fold == "fold_a" else 0.4
            y_pred = y if fold == "fold_a" else 0
            rows.append(
                {
                    "dataset_row_id": len(rows),
                    "decision_time": len(rows),
                    "fold_name": fold,
                    "model_name": "prior_baseline",
                    "y_true": y,
                    "p_up": p_up,
                    "y_pred": y_pred,
                    "prediction_threshold": 0.5,
                    "evaluation_offset_minutes": 0,
                    "is_primary_nonoverlap_evaluation": True,
                    "absolute_future_return_bps": 10.0,
                    "proxy_boundary_risk_1bps": False,
                    "proxy_boundary_risk_2_5bps": False,
                    "proxy_boundary_risk_5bps": False,
                    "proxy_boundary_risk_10bps": False,
                }
            )
    predictions = pd.DataFrame(rows)
    summary = pooled_and_macro_summary(evaluate_prediction_subsets(predictions), predictions)

    dense = summary[(summary["model_name"] == "prior_baseline") & (summary["subset_name"] == "DENSE")]
    assert set(dense["summary_type"]) >= {"pooled", "fold_macro_mean", "fold_macro_std"}
    pooled_acc = dense.loc[dense["summary_type"].eq("pooled"), "accuracy"].iloc[0]
    macro_acc = dense.loc[dense["summary_type"].eq("fold_macro_mean"), "accuracy"].iloc[0]
    assert pooled_acc != macro_acc


def test_calibration_equal_width_ece_mce() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 4,
            "fold_name": ["fold"] * 4,
            "y_true": [0, 0, 1, 1],
            "p_up": [0.1, 0.2, 0.8, 0.9],
            "is_primary_nonoverlap_evaluation": [True] * 4,
        }
    )
    table = calibration_table(predictions, bins=2, strategy="equal_width", subset_name="DENSE")

    assert table["sample_count"].sum() == 4
    assert np.isclose(table["ece"].iloc[0], 0.15)
    assert np.isclose(table["mce"].iloc[0], 0.15)
    assert {"mean_predicted_probability", "actual_up_ratio", "probability_actual_gap"}.issubset(table.columns)


def test_calibration_equal_frequency_bins() -> None:
    predictions = pd.DataFrame(
        {
            "model_name": ["m"] * 5,
            "fold_name": ["fold"] * 5,
            "y_true": [0, 0, 1, 1, 1],
            "p_up": [0.1, 0.2, 0.3, 0.8, 0.9],
            "is_primary_nonoverlap_evaluation": [True] * 5,
        }
    )
    table = calibration_table(predictions, bins=2, strategy="equal_frequency", subset_name="DENSE")

    assert table["sample_count"].sum() == 5
    assert table["bin_index"].nunique() == 2


def test_logistic_coefficients_follow_feature_order() -> None:
    dataset, splits = make_stage6_frames()
    output = build_stage6_outputs(dataset, splits, dataset_manifest(), feature_manifest(), fold_manifest(), CONFIG)
    coefficients = output.logistic_coefficients_by_fold

    assert coefficients.sort_values("feature_index").head(63)["feature_name"].tolist() == FEATURE_NAMES
    assert {"coefficient", "absolute_coefficient", "coefficient_rank", "coefficient_sign"}.issubset(coefficients.columns)


def test_input_row_order_shuffle_is_stable() -> None:
    dataset, splits = make_stage6_frames()
    output_a = build_stage6_outputs(dataset, splits, dataset_manifest(), feature_manifest(), fold_manifest(), CONFIG)
    shuffled_dataset = dataset.sample(frac=1.0, random_state=9).reset_index(drop=True)
    shuffled_splits = splits.sample(frac=1.0, random_state=2).reset_index(drop=True)
    output_b = build_stage6_outputs(shuffled_dataset, shuffled_splits, dataset_manifest(), feature_manifest(), fold_manifest(), CONFIG)

    left = output_a.predictions.sort_values(["model_name", "fold_name", "dataset_row_id"]).reset_index(drop=True)
    right = output_b.predictions.sort_values(["model_name", "fold_name", "dataset_row_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_model_serialization_roundtrip_matches_predictions(tmp_path: Path) -> None:
    dataset, splits = make_stage6_frames()
    train = dataset.loc[splits["fold_2020_role"].eq("TRAIN")].copy()
    valid = dataset.loc[splits["fold_2020_role"].eq("VALIDATION")].copy()
    pred, _, pipeline = fit_logistic_fold(train, valid, FEATURE_NAMES, CONFIG["logistic_regression_parameters"], CONFIG["scaler_parameters"], 0.5)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)

    loaded = joblib.load(path)
    reloaded_p = loaded.predict_proba(valid[FEATURE_NAMES].to_numpy(dtype=np.float64))[:, 1]
    np.testing.assert_allclose(reloaded_p, pred["p_up"].to_numpy())


def test_run_stage6_writes_outputs_and_preserves_stage5_files(tmp_path: Path) -> None:
    dataset, splits = make_stage6_frames(all_folds=True)
    dataset_path = tmp_path / "dataset.parquet"
    split_path = tmp_path / "splits.parquet"
    dataset.to_parquet(dataset_path, index=False)
    splits.to_parquet(split_path, index=False)
    write_json(tmp_path / "dataset_manifest.json", dataset_manifest())
    write_json(tmp_path / "feature_manifest.json", feature_manifest())
    write_json(tmp_path / "fold_manifest.json", fold_manifest(all_folds=True))
    config = {**CONFIG, "dataset_path": "dataset.parquet", "split_path": "splits.parquet"}
    write_json(tmp_path / "config.json", config)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [dataset_path, split_path]}

    result = run_stage6(tmp_path / "config.json", tmp_path)

    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[p] for p in before)
    assert result["quality_gates"]["no_final_test_predictions"] is True
    assert result["quality_gates"]["stage6_engineering_gate_passed"] is True
    predictions = pd.read_parquet(tmp_path / "out/predictions.parquet")
    manifest_out = json.loads((tmp_path / "reports/model_manifest.json").read_text(encoding="utf-8"))
    assert not predictions["dataset_row_id"].isin(splits.loc[splits["final_split_role"].eq("FINAL_TEST"), "dataset_row_id"]).any()
    assert manifest_out["feature_columns"] == FEATURE_NAMES
    assert (tmp_path / "models/fold_2020/logistic_regression_pipeline.joblib").exists()
    assert (tmp_path / "reports/report.md").exists()


def ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def dataset_manifest() -> dict[str, object]:
    return {
        "dataset_version": "btcusdt_60m_direction_v1",
        "feature_set_version": "kline_v1_63",
        "ordered_feature_list_hash": hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest(),
        "feature_columns": FEATURE_NAMES,
        "primary_target_columns": ["label_up_60m"],
        "sample_weight_columns": ["sample_weight_uniform", "sample_weight_margin"],
        "evaluation_metadata_columns": [
            "proxy_margin_bucket",
            "proxy_boundary_risk_1bps",
            "proxy_boundary_risk_2_5bps",
            "proxy_boundary_risk_5bps",
            "proxy_boundary_risk_10bps",
        ],
        "forbidden_model_input_columns": [
            "label_up_60m",
            "future_simple_return_60m",
            "future_log_return_60m",
            "absolute_future_return_bps",
            "proxy_margin_bucket",
            "proxy_boundary_risk_1bps",
            "proxy_boundary_risk_2_5bps",
            "proxy_boundary_risk_5bps",
            "proxy_boundary_risk_10bps",
            "sample_weight_margin",
            "settlement_minute_open_time",
        ],
        "output_files": {
            "dataset_path": "dataset.parquet",
            "split_assignment_path": "splits.parquet",
            "dataset_sha256": "dataset_sha",
            "split_assignment_sha256": "split_sha",
        },
    }


def feature_manifest() -> dict[str, object]:
    return {
        "feature_set_version": "kline_v1_63",
        "ordered_feature_names": FEATURE_NAMES,
        "feature_count": 63,
        "feature_definition_hash": "feature_hash",
    }


def fold_manifest(all_folds: bool = False) -> dict[str, object]:
    folds = []
    years = [2020, 2021, 2022, 2023, 2024] if all_folds else [2020]
    for year in years:
        folds.append(
            {
                "name": f"fold_{year}",
                "validation_start": f"{year}-01-01T00:00:00Z",
                "validation_end": f"{year + 1}-01-01T00:00:00Z",
                "train_sample_count": 6,
                "validation_sample_count": 12,
                "train_max_settlement_time": ms(f"{year}-01-01T00:00:00Z") - 300_000,
                "validation_min_decision_time": ms(f"{year}-01-01T00:00:00Z"),
                "validation_max_settlement_time": ms(f"{year}-01-01T00:55:00Z") + 3_600_000,
            }
        )
    return {"split_version": "expanding_yearly_v1", "folds": folds}


def make_stage6_frames(all_folds: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = [2020, 2021, 2022, 2023, 2024] if all_folds else [2020]
    dataset_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    row_id = 0
    for year_index, year in enumerate(years):
        validation_start = ms(f"{year}-01-01T00:00:00Z")
        train_times = [validation_start - (18 - i) * 5 * 60_000 for i in range(6)]
        valid_times = [validation_start + i * 5 * 60_000 for i in range(12)]
        final_times = [ms("2025-01-01T00:00:00Z") + year_index * 60_000] if all_folds else [ms("2025-01-01T00:00:00Z")]
        for role, times in [("TRAIN", train_times), ("VALIDATION", valid_times), ("FINAL_TEST", final_times)]:
            for local_index, decision_time in enumerate(times):
                label = int((local_index + (0 if role == "TRAIN" else 1)) % 2 == 0)
                if role == "FINAL_TEST":
                    label = local_index % 2
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
                for feature_index, feature in enumerate(FEATURE_NAMES):
                    if feature == "log_return_60m":
                        row[feature] = 0.01 if local_index % 2 == 0 else -0.01
                    else:
                        row[feature] = float((row_id + 1) * (feature_index + 1)) / 1000.0
                split = {
                    "dataset_row_id": row_id,
                    "decision_time": decision_time,
                    "settlement_minute_open_time": decision_time + 3_600_000,
                    "final_split_role": "FINAL_TEST" if role == "FINAL_TEST" else "DEVELOPMENT",
                    "evaluation_offset_minutes": 0 if role != "VALIDATION" else local_index * 5,
                    "is_primary_nonoverlap_evaluation": role == "VALIDATION" and local_index == 0,
                }
                for fold_year in [2020, 2021, 2022, 2023, 2024]:
                    if fold_year == year and role in {"TRAIN", "VALIDATION"}:
                        split[f"fold_{fold_year}_role"] = role
                    elif role == "FINAL_TEST":
                        split[f"fold_{fold_year}_role"] = "FINAL_TEST_EXCLUDED"
                    else:
                        split[f"fold_{fold_year}_role"] = "OUTSIDE_FOLD"
                dataset_rows.append(row)
                split_rows.append(split)
                row_id += 1
    return pd.DataFrame(dataset_rows), pd.DataFrame(split_rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

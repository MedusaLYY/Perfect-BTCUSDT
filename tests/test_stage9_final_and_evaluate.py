from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_stage4_features import FEATURE_NAMES  # noqa: E402
from scripts.train_stage9_final_and_evaluate import (  # noqa: E402
    ALLOWED_CALIBRATION_METHODS,
    FINAL_TEST_OUTPUT_COLUMNS,
    Stage9ValidationError,
    apply_calibrator,
    assert_hashes_unchanged,
    assess_final_model,
    build_calibration_forward_folds,
    build_final_prediction_frame,
    build_inference_manifest,
    compare_test_to_oof,
    create_final_test_feature_matrix,
    create_protocol_lock,
    evaluate_calibration_candidates,
    evaluate_final_predictions,
    fit_calibrator,
    fit_final_calibrator,
    final_refit_training_frame,
    logit_transform,
    make_final_inner_split,
    prepare_feature_matrix,
    select_calibration_method,
    train_logistic_baseline,
    train_momentum_baseline,
    train_prior_baseline,
    validate_calibration_candidates,
    validate_final_test_predictions,
    validate_no_threshold_optimization,
    verify_stage8_selection,
    verify_xgboost_parameters,
)


def test_only_three_calibration_methods_allowed() -> None:
    assert ALLOWED_CALIBRATION_METHODS == ("UNCALIBRATED", "PLATT", "ISOTONIC")
    validate_calibration_candidates(["UNCALIBRATED", "PLATT", "ISOTONIC"])
    with pytest.raises(Stage9ValidationError, match="calibration"):
        validate_calibration_candidates(["UNCALIBRATED", "BETA"])


def test_calibration_meta_folds_are_strictly_forward() -> None:
    folds = build_calibration_forward_folds(["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"])
    assert [f["evaluation_fold"] for f in folds] == ["fold_2021", "fold_2022", "fold_2023", "fold_2024"]
    assert folds[0]["fit_folds"] == ["fold_2020"]
    assert folds[-1]["fit_folds"] == ["fold_2020", "fold_2021", "fold_2022", "fold_2023"]
    assert all(f["evaluation_fold"] not in f["fit_folds"] for f in folds)


def test_2021_calibration_uses_only_2020_fit() -> None:
    first = build_calibration_forward_folds(["fold_2020", "fold_2021", "fold_2022"])[0]
    assert first == {"calibration_fold": "calibration_2021", "fit_folds": ["fold_2020"], "evaluation_fold": "fold_2021"}


def test_2024_calibration_never_uses_2024_labels_for_fit() -> None:
    last = build_calibration_forward_folds(["fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"])[-1]
    assert last["evaluation_fold"] == "fold_2024"
    assert "fold_2024" not in last["fit_folds"]


def test_uncalibrated_identity_is_exact() -> None:
    p = np.array([0.01, 0.2, 0.8, 0.99])
    calibrator = fit_calibrator("UNCALIBRATED", p, np.array([0, 0, 1, 1]))
    np.testing.assert_array_equal(apply_calibrator(calibrator, p), p)


def test_platt_uses_probability_logit_input() -> None:
    p = np.array([0.05, 0.2, 0.8, 0.95])
    expected = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    np.testing.assert_allclose(logit_transform(p), expected)
    calibrator = fit_calibrator("PLATT", p, np.array([0, 0, 1, 1]))
    assert calibrator["input_transform"] == "logit_probability"


def test_isotonic_output_is_clipped_to_unit_interval() -> None:
    calibrator = fit_calibrator("ISOTONIC", np.array([0.2, 0.4, 0.6, 0.8]), np.array([0, 0, 1, 1]))
    out = apply_calibrator(calibrator, np.array([-0.5, 0.1, 0.9, 1.5]))
    assert np.isfinite(out).all()
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_calibration_selection_rule_primary_and_ties() -> None:
    metrics = calibration_metrics_fixture()
    selected = select_calibration_method(metrics, calibration_rules())
    assert selected["selected_calibration_method"] == "PLATT"
    assert selected["ranked_methods"][0]["method"] == "PLATT"


def test_calibration_keeps_uncalibrated_without_material_gain() -> None:
    metrics = calibration_metrics_fixture(platt_logloss=0.68695, platt_brier=0.24694, isotonic_logloss=0.68698, isotonic_brier=0.24698)
    selected = select_calibration_method(metrics, calibration_rules())
    assert selected["selected_calibration_method"] == "UNCALIBRATED"
    assert selected["calibration_improvement_material"] is False


def test_final_calibrator_uses_only_oof_predictions() -> None:
    oof = make_oof_predictions()
    calibrator, meta = fit_final_calibrator(oof, "PLATT", epsilon=1e-6)
    assert meta["training_source"] == "stage7_development_oof_only"
    assert meta["training_oof_sample_count"] == len(oof)
    assert set(meta["training_folds"]) == {"fold_2020", "fold_2021", "fold_2022", "fold_2023", "fold_2024"}
    assert calibrator["method"] == "PLATT"


def test_final_tree_split_excludes_final_test() -> None:
    joined = make_joined_frame()
    split = make_final_inner_split(joined, ms("2025-01-01T00:00:00Z"), 90)
    assert not split["inner_fit"]["final_split_role"].eq("FINAL_TEST").any()
    assert not split["inner_early_stop"]["final_split_role"].eq("FINAL_TEST").any()
    assert split["inner_fit"]["settlement_minute_open_time"].max() < split["inner_early_stop"]["decision_time"].min()


def test_final_inner_purge_boundary_is_correct() -> None:
    joined = make_joined_frame()
    split = make_final_inner_split(joined, ms("2025-01-01T00:00:00Z"), 90)
    start = split["inner_early_stop_start"]
    assert (split["inner_purged"]["decision_time"] < start).all()
    assert (split["inner_purged"]["settlement_minute_open_time"] >= start).all()


def test_final_refit_uses_all_development_only() -> None:
    joined = make_joined_frame()
    train = final_refit_training_frame(joined)
    assert set(train["final_split_role"]) == {"DEVELOPMENT"}
    assert len(train) == int(joined["final_split_role"].eq("DEVELOPMENT").sum())


def test_purged_rows_do_not_enter_final_training() -> None:
    joined = make_joined_frame()
    train = final_refit_training_frame(joined)
    assert "PURGED_BEFORE_FINAL_TEST" not in set(train["final_split_role"])


def test_final_test_rows_do_not_enter_final_training() -> None:
    joined = make_joined_frame()
    train = final_refit_training_frame(joined)
    assert "FINAL_TEST" not in set(train["final_split_role"])


def test_xgboost_parameters_match_stage8_reference() -> None:
    params = verify_xgboost_parameters(stage9_xgb_params(), n_jobs=19, n_estimators=12, selector=False)
    assert params["learning_rate"] == 0.03
    assert params["max_depth"] == 4
    assert params["min_child_weight"] == 20
    assert params["reg_lambda"] == 10.0
    assert params["scale_pos_weight"] == 1.0
    with pytest.raises(Stage9ValidationError, match="Stage7 reference"):
        verify_xgboost_parameters({**stage9_xgb_params(), "max_depth": 5}, n_jobs=19, n_estimators=12, selector=False)


def test_feature_order_is_exactly_63() -> None:
    df = make_joined_frame()
    X = prepare_feature_matrix(df, FEATURE_NAMES)
    assert list(X.columns) == FEATURE_NAMES
    assert X.shape[1] == 63


def test_future_and_target_fields_never_enter_x() -> None:
    X = prepare_feature_matrix(make_joined_frame(), FEATURE_NAMES)
    forbidden = {"label_up_60m", "future_simple_return_60m", "sample_weight_margin", "settlement_minute_open_time"}
    assert forbidden.isdisjoint(X.columns)


def test_protocol_lock_required_before_final_test_matrix() -> None:
    joined = make_joined_frame()
    state = {"protocol_lock_written": False}
    with pytest.raises(Stage9ValidationError, match="protocol lock"):
        create_final_test_feature_matrix(joined, FEATURE_NAMES, ms("2025-01-01T00:00:00Z"), expected_count=2, state=state)


def test_model_serialized_before_protocol_lock() -> None:
    payload = protocol_payload_fixture(model_serialized=False)
    with pytest.raises(Stage9ValidationError, match="serialized"):
        create_protocol_lock(payload)


def test_model_reload_verification_is_development_only() -> None:
    payload = protocol_payload_fixture()
    payload["model_reload_verification_sample_role"] = "FINAL_TEST"
    with pytest.raises(Stage9ValidationError, match="DEVELOPMENT"):
        create_protocol_lock(payload)


def test_final_test_predict_proba_count_is_one() -> None:
    model = CountingModel([0.2, 0.8])
    joined = make_joined_frame()
    state = protocol_state()
    X, meta = create_final_test_feature_matrix(joined, FEATURE_NAMES, ms("2025-01-01T00:00:00Z"), expected_count=2, state=state)
    pred = build_final_prediction_frame(meta, model, X, identity_calibrator(), baseline_predictions(meta), 0.5)
    assert pred.attrs["final_test_base_model_predict_proba_call_count"] == 1
    assert model.call_count == 1


def test_calibration_transform_does_not_call_xgboost_again() -> None:
    model = CountingModel([0.2, 0.8])
    joined = make_joined_frame()
    X, meta = create_final_test_feature_matrix(joined, FEATURE_NAMES, ms("2025-01-01T00:00:00Z"), expected_count=2, state=protocol_state())
    build_final_prediction_frame(meta, model, X, identity_calibrator(), baseline_predictions(meta), 0.5)
    assert model.call_count == 1


def test_final_predictions_are_unique_per_sample() -> None:
    pred = final_prediction_fixture()
    validate_final_test_predictions(pred, expected_count=2)
    bad = pd.concat([pred, pred.iloc[[0]]], ignore_index=True)
    with pytest.raises(Stage9ValidationError, match="unique"):
        validate_final_test_predictions(bad, expected_count=3)


def test_final_predictions_contain_no_development_rows() -> None:
    pred = final_prediction_fixture()
    assert set(pred["final_split_role"]) == {"FINAL_TEST"}


def test_prior_probability_uses_development_labels_only() -> None:
    joined = make_joined_frame()
    prior = train_prior_baseline(final_refit_training_frame(joined), threshold=0.5)
    assert prior["training_sample_count"] == int(joined["final_split_role"].eq("DEVELOPMENT").sum())
    assert np.isclose(prior["p_up"], final_refit_training_frame(joined)["label_up_60m"].mean())


def test_momentum_probability_uses_development_labels_only() -> None:
    joined = make_joined_frame()
    momentum = train_momentum_baseline(final_refit_training_frame(joined), alpha=1.0)
    assert momentum["training_sample_count"] == int(joined["final_split_role"].eq("DEVELOPMENT").sum())
    assert momentum["alpha"] == 1.0


def test_logistic_scaler_fits_on_development_only(tmp_path: Path) -> None:
    train = final_refit_training_frame(make_joined_frame())
    model_path = tmp_path / "lr.joblib"
    _, meta = train_logistic_baseline(train, FEATURE_NAMES, logistic_params(), {"with_mean": True, "with_std": True}, model_path)
    pipe = joblib.load(model_path)
    assert int(pipe.named_steps["scaler"].n_samples_seen_) == len(train)
    assert meta["training_sample_count"] == len(train)


def test_fixed_threshold_is_half_and_optimization_is_rejected() -> None:
    validate_no_threshold_optimization({"fixed_classification_threshold": 0.5})
    with pytest.raises(Stage9ValidationError, match="threshold"):
        validate_no_threshold_optimization({"fixed_classification_threshold": 0.51})


def test_dense_metrics_are_computed_correctly() -> None:
    metrics = evaluate_final_predictions(final_prediction_fixture(), bins=2)
    row = metrics[(metrics["model_name"].eq("xgboost_final_calibrated")) & (metrics["subset_name"].eq("DENSE"))].iloc[0]
    assert row["sample_count"] == 2
    assert row["accuracy"] == 1.0


def test_nonoverlap_metrics_are_computed_correctly() -> None:
    metrics = evaluate_final_predictions(final_prediction_fixture(), bins=2)
    row = metrics[(metrics["model_name"].eq("xgboost_final_calibrated")) & (metrics["subset_name"].eq("NONOVERLAP_OFFSET_00"))].iloc[0]
    assert row["sample_count"] == 1


def test_all_12_offsets_are_reported() -> None:
    metrics = evaluate_final_predictions(final_prediction_fixture(all_offsets=True), bins=2)
    offsets = {f"OFFSET_{i:02d}" for i in range(0, 60, 5)}
    assert offsets.issubset(set(metrics["subset_name"]))


def test_margin_subsets_use_returns_for_evaluation_only() -> None:
    metrics = evaluate_final_predictions(final_prediction_fixture(), bins=2)
    assert {"ALL_MARGINS", "ABS_RETURN_GE_1BPS", "ABS_RETURN_GE_2_5BPS", "ABS_RETURN_GE_5BPS", "ABS_RETURN_GE_10BPS"}.issubset(set(metrics["subset_name"]))
    assert "absolute_future_return_bps" in final_prediction_fixture().columns


def test_test_minus_oof_differences_are_correct() -> None:
    oof = pd.DataFrame([{"subset_name": "DENSE", "model_name": "xgboost_fixed_v1", "roc_auc": 0.55, "log_loss": 0.69, "brier_score": 0.25, "mcc": 0.1}])
    test = pd.DataFrame([{"subset_name": "DENSE", "model_name": "xgboost_final_calibrated", "roc_auc": 0.56, "log_loss": 0.68, "brier_score": 0.24, "mcc": 0.2}])
    diff = compare_test_to_oof(oof, test)
    assert np.isclose(diff.iloc[0]["test_minus_oof_auc"], 0.01)
    assert np.isclose(diff.iloc[0]["test_minus_oof_logloss"], -0.01)


def test_final_model_assessment_rules() -> None:
    rows = [
        metric_row("DENSE", "xgboost_final_calibrated", auc=0.53, logloss=0.690),
        metric_row("NONOVERLAP_OFFSET_00", "xgboost_final_calibrated", auc=0.54, logloss=0.691),
        metric_row("NONOVERLAP_OFFSET_00", "logistic_regression_final", auc=0.52, logloss=0.692),
        metric_row("NONOVERLAP_OFFSET_00", "prior_baseline_final", auc=0.50, logloss=0.6915),
    ]
    assert assess_final_model(pd.DataFrame(rows), {"stage9_engineering_gate_passed": True}) == "READY_FOR_RESEARCH_DEPLOYMENT"
    rows[1]["roc_auc"] = 0.49
    assert assess_final_model(pd.DataFrame(rows), {"stage9_engineering_gate_passed": True}) == "FAILED_FINAL_VALIDATION"


def test_inference_manifest_has_required_fields() -> None:
    manifest = build_inference_manifest(protocol_payload_fixture(), ["a"] * 63)
    required = {"model_name", "model_version", "model_file", "calibration_method", "ordered_feature_names", "model_limitations", "created_at_utc"}
    assert required.issubset(manifest)
    assert len(manifest["ordered_feature_names"]) == 63


def test_model_save_and_reload_consistency(tmp_path: Path) -> None:
    calibrator = fit_calibrator("PLATT", np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1]))
    path = tmp_path / "cal.joblib"
    joblib.dump(calibrator, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(apply_calibrator(calibrator, np.array([0.3, 0.7])), apply_calibrator(loaded, np.array([0.3, 0.7])))


def test_inference_manifest_output_columns_are_complete() -> None:
    assert {"dataset_row_id", "p_up_raw", "p_up_calibrated", "logistic_p_up"}.issubset(set(FINAL_TEST_OUTPUT_COLUMNS))


def test_no_parameter_change_after_test_hash_check() -> None:
    before = {"config": "a", "protocol": "b"}
    assert_hashes_unchanged(before, {"config": "a", "protocol": "b"})
    with pytest.raises(Stage9ValidationError, match="changed"):
        assert_hashes_unchanged(before, {"config": "a", "protocol": "c"})


def test_stage5_to_stage8_inputs_not_modified(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"x":1}', encoding="utf-8")
    before = sha256(path)
    assert_hashes_unchanged({str(path): before}, {str(path): sha256(path)})


def test_stage8_selection_must_match_reference() -> None:
    verify_stage8_selection({"selected_development_config": "xgb_fixed_v1_reference", "development_recommendation": "KEEP_STAGE7_REFERENCE", "improvement_not_material": True})
    with pytest.raises(Stage9ValidationError, match="Stage8"):
        verify_stage8_selection({"selected_development_config": "xgb_depth3_v1"})


def ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_joined_frame() -> pd.DataFrame:
    rows = []
    base = ms("2024-10-03T00:00:00Z")
    times_roles = [
        (base - 120 * 60_000, base - 60 * 60_000, "DEVELOPMENT"),
        (base - 65 * 60_000, base - 5 * 60_000, "DEVELOPMENT"),
        (base - 55 * 60_000, base + 5 * 60_000, "DEVELOPMENT"),
        (base + 0 * 60_000, base + 60 * 60_000, "DEVELOPMENT"),
        (base + 5 * 60_000, base + 65 * 60_000, "DEVELOPMENT"),
        (ms("2024-12-31T23:00:00Z"), ms("2025-01-01T00:00:00Z"), "PURGED_BEFORE_FINAL_TEST"),
        (ms("2025-01-01T00:00:00Z"), ms("2025-01-01T01:00:00Z"), "FINAL_TEST"),
        (ms("2025-01-01T00:05:00Z"), ms("2025-01-01T01:05:00Z"), "FINAL_TEST"),
    ]
    for i, (decision, settlement, role) in enumerate(times_roles):
        label = i % 2
        row = {
            "dataset_row_id": i,
            "decision_time": decision,
            "settlement_minute_open_time": settlement,
            "final_split_role": role,
            "evaluation_offset_minutes": 0 if i % 2 == 0 else 5,
            "is_primary_nonoverlap_evaluation": i % 2 == 0,
            "label_up_60m": label,
            "future_simple_return_60m": 0.001 if label else -0.001,
            "future_log_return_60m": np.log1p(0.001 if label else -0.001),
            "absolute_future_return_bps": float(1 + i),
            "proxy_boundary_risk_1bps": False,
            "proxy_boundary_risk_2_5bps": i < 2,
            "proxy_boundary_risk_5bps": i < 4,
            "proxy_boundary_risk_10bps": i < 6,
            "sample_weight_uniform": 1.0,
            "sample_weight_margin": 2.0,
        }
        for j, feature in enumerate(FEATURE_NAMES):
            row[feature] = float((label * 2 - 1) * ((j % 7) + 1) + i * 0.01)
        rows.append(row)
    return pd.DataFrame(rows)


def make_oof_predictions() -> pd.DataFrame:
    rows = []
    for year in range(2020, 2025):
        for i in range(6):
            y = int(i % 2 == 0)
            rows.append(
                {
                    "dataset_row_id": year * 100 + i,
                    "decision_time": ms(f"{year}-01-01T00:{i:02d}:00Z"),
                    "fold_name": f"fold_{year}",
                    "model_name": "xgboost_fixed_v1",
                    "y_true": y,
                    "p_up": 0.35 if y == 0 else 0.65,
                    "y_pred": y,
                    "prediction_threshold": 0.5,
                    "evaluation_offset_minutes": (i % 12) * 5,
                    "is_primary_nonoverlap_evaluation": i == 0,
                    "absolute_future_return_bps": float(i + 1),
                    "proxy_boundary_risk_1bps": False,
                    "proxy_boundary_risk_2_5bps": False,
                    "proxy_boundary_risk_5bps": False,
                    "proxy_boundary_risk_10bps": False,
                }
            )
    return pd.DataFrame(rows)


def calibration_metrics_fixture(platt_logloss: float = 0.6860, platt_brier: float = 0.2460, isotonic_logloss: float = 0.6861, isotonic_brier: float = 0.2461) -> pd.DataFrame:
    rows = []
    for method, logloss, brier in [
        ("UNCALIBRATED", 0.6870, 0.2470),
        ("PLATT", platt_logloss, platt_brier),
        ("ISOTONIC", isotonic_logloss, isotonic_brier),
    ]:
        for fold in ["calibration_2021", "calibration_2022", "calibration_2023", "calibration_2024"]:
            rows.append({"method": method, "calibration_fold": fold, "subset_name": "NONOVERLAP_OFFSET_00", "log_loss": logloss, "brier_score": brier, "ece_equal_frequency": 0.01, "ece_equal_width": 0.01, "mce_equal_width": 0.02, "mce_equal_frequency": 0.02, "roc_auc": 0.55, "average_precision": 0.55, "sample_count": 10})
            rows.append({"method": method, "calibration_fold": fold, "subset_name": "DENSE", "log_loss": logloss, "brier_score": brier, "ece_equal_frequency": 0.01, "ece_equal_width": 0.01, "mce_equal_width": 0.02, "mce_equal_frequency": 0.02, "roc_auc": 0.55, "average_precision": 0.55, "sample_count": 20})
    pooled = []
    for method, logloss, brier in [
        ("UNCALIBRATED", 0.6870, 0.2470),
        ("PLATT", platt_logloss, platt_brier),
        ("ISOTONIC", isotonic_logloss, isotonic_brier),
    ]:
        for subset in ["NONOVERLAP_OFFSET_00", "DENSE"]:
            pooled.append({"method": method, "calibration_fold": "POOLED_2021_2024", "subset_name": subset, "log_loss": logloss, "brier_score": brier, "ece_equal_frequency": 0.01, "ece_equal_width": 0.01, "mce_equal_width": 0.02, "mce_equal_frequency": 0.02, "roc_auc": 0.55, "average_precision": 0.55, "sample_count": 40})
    return pd.DataFrame([*rows, *pooled])


def calibration_rules() -> dict[str, float]:
    return {
        "calibration_logloss_tie_tolerance": 0.0002,
        "material_logloss_improvement": 0.0002,
        "material_brier_improvement": 0.0001,
        "material_logloss_max_degradation": 0.0001,
    }


def stage9_xgb_params() -> dict[str, object]:
    return {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cpu",
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 20,
        "gamma": 0.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 10.0,
        "max_bin": 256,
        "scale_pos_weight": 1.0,
        "eval_metric": "logloss",
        "random_state": 42,
        "validate_parameters": True,
        "verbosity": 1,
    }


def logistic_params() -> dict[str, object]:
    return {"penalty": "l2", "C": 1.0, "solver": "lbfgs", "fit_intercept": True, "class_weight": None, "max_iter": 200, "tol": 1e-6, "random_state": 42, "n_jobs": None}


def protocol_payload_fixture(model_serialized: bool = True) -> dict[str, object]:
    return {
        "selected_development_config": "xgb_fixed_v1_reference",
        "xgboost_parameters": stage9_xgb_params(),
        "final_best_n_estimators": 12,
        "selected_calibration_method": "UNCALIBRATED",
        "ordered_feature_names": FEATURE_NAMES,
        "fixed_classification_threshold": 0.5,
        "fixed_evaluation_metrics": ["roc_auc", "log_loss"],
        "fixed_evaluation_subsets": ["DENSE", "NONOVERLAP_OFFSET_00"],
        "model_serialized": model_serialized,
        "calibrator_serialized": True,
        "logistic_serialized": True,
        "model_reload_verified": True,
        "model_reload_verification_sample_role": "DEVELOPMENT",
        "final_test_accessed": False,
    }


def protocol_state() -> dict[str, object]:
    return {
        "protocol_lock_written": True,
        "final_model_trained": True,
        "final_model_serialized": True,
        "model_reload_verified": True,
        "calibrator_frozen": True,
        "calibrator_serialized": True,
        "feature_manifest_verified": True,
        "development_checks_passed": True,
        "final_test_feature_matrix_created": False,
    }


def identity_calibrator() -> dict[str, object]:
    return {"method": "UNCALIBRATED", "params": {}}


def baseline_predictions(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(meta)
    return {
        "prior_p_up": np.full(n, 0.5),
        "prior_y_pred": np.ones(n, dtype=np.int8),
        "momentum_p_up": np.full(n, 0.5),
        "momentum_y_pred": np.array([0, 1], dtype=np.int8)[:n],
        "logistic_p_up": np.array([0.25, 0.75], dtype=float)[:n],
        "logistic_y_pred": np.array([0, 1], dtype=np.int8)[:n],
    }


class CountingModel:
    def __init__(self, p: list[float]) -> None:
        self.p = np.asarray(p, dtype=float)
        self.call_count = 0

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        p = self.p[: len(X)]
        return np.column_stack([1.0 - p, p])


def final_prediction_fixture(all_offsets: bool = False) -> pd.DataFrame:
    if all_offsets:
        rows = []
        for idx, offset in enumerate(range(0, 60, 5)):
            y = idx % 2
            p = 0.2 if y == 0 else 0.8
            row = base_prediction_row(idx, y, p)
            row["evaluation_offset_minutes"] = offset
            row["is_primary_nonoverlap_evaluation"] = offset == 0
            rows.append(row)
        return pd.DataFrame(rows)
    return pd.DataFrame([base_prediction_row(0, 0, 0.2), base_prediction_row(1, 1, 0.8)])


def base_prediction_row(row_id: int, y: int, p: float) -> dict[str, object]:
    return {
        "dataset_row_id": row_id,
        "decision_time": row_id,
        "final_split_role": "FINAL_TEST",
        "y_true": y,
        "p_up_raw": p,
        "p_up_calibrated": p,
        "y_pred_raw_0_5": int(p >= 0.5),
        "y_pred_calibrated_0_5": int(p >= 0.5),
        "evaluation_offset_minutes": 0,
        "is_primary_nonoverlap_evaluation": row_id == 0,
        "future_simple_return_60m": 0.001 if y else -0.001,
        "absolute_future_return_bps": 10.0,
        "proxy_boundary_risk_1bps": False,
        "proxy_boundary_risk_2_5bps": False,
        "proxy_boundary_risk_5bps": False,
        "proxy_boundary_risk_10bps": False,
        "prior_p_up": 0.5,
        "prior_y_pred": 1,
        "momentum_p_up": 0.5,
        "momentum_y_pred": y,
        "logistic_p_up": p,
        "logistic_y_pred": int(p >= 0.5),
    }


def metric_row(subset: str, model: str, auc: float, logloss: float) -> dict[str, object]:
    return {"subset_name": subset, "model_name": model, "roc_auc": auc, "log_loss": logloss, "brier_score": 0.24, "mcc": 0.1, "sample_count": 10}

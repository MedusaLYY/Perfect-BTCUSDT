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
from scripts.build_stage5_dataset_and_splits import (  # noqa: E402
    Stage5ValidationError,
    build_dataset_and_splits,
    make_dataset_manifest,
    run_stage5,
)


CONFIG = {
    "dataset_version": "btcusdt_60m_direction_v1",
    "split_version": "expanding_yearly_v1",
    "join_key": "decision_time",
    "final_test_start": "2025-01-01T00:00:00Z",
    "cv_folds": [
        {"name": "fold_2020", "validation_start": "2020-01-01T00:00:00Z", "validation_end": "2021-01-01T00:00:00Z"},
        {"name": "fold_2021", "validation_start": "2021-01-01T00:00:00Z", "validation_end": "2022-01-01T00:00:00Z"},
        {"name": "fold_2022", "validation_start": "2022-01-01T00:00:00Z", "validation_end": "2023-01-01T00:00:00Z"},
        {"name": "fold_2023", "validation_start": "2023-01-01T00:00:00Z", "validation_end": "2024-01-01T00:00:00Z"},
        {"name": "fold_2024", "validation_start": "2024-01-01T00:00:00Z", "validation_end": "2025-01-01T00:00:00Z"},
    ],
    "horizon_minutes": 60,
    "prediction_interval_minutes": 5,
    "primary_nonoverlap_offset_minutes": 0,
    "required_feature_count": 63,
    "parquet_compression": "snappy",
    "hash_algorithm": "sha256",
    "symbol": "BTCUSDT",
}


def test_one_to_one_join_shuffle_stable_and_dataset_row_ids() -> None:
    times = [
        ms("2019-12-31T22:55:00Z"),
        ms("2020-01-01T00:00:00Z"),
        ms("2021-01-01T00:00:00Z"),
    ]
    features = make_features(times)
    labels = make_labels(times)
    result = build_dataset_and_splits(features.sample(frac=1, random_state=7), labels.sample(frac=1, random_state=3), manifest(), label_dictionary(), CONFIG)
    dataset = result.dataset

    assert dataset["decision_time"].tolist() == sorted(times)
    assert dataset["dataset_row_id"].tolist() == [0, 1, 2]
    assert dataset["dataset_row_id"].is_unique
    assert len(result.join_audit) == 3
    assert result.join_audit["final_intersection_candidate"].all()


def test_duplicate_feature_or_label_keys_fail() -> None:
    times = [ms("2020-01-01T00:00:00Z"), ms("2020-01-01T00:05:00Z")]
    features = make_features(times)
    labels = make_labels(times)

    with pytest.raises(Stage5ValidationError, match="Feature decision_time must be unique"):
        build_dataset_and_splits(pd.concat([features, features.iloc[[0]]], ignore_index=True), labels, manifest(), label_dictionary(), CONFIG)
    with pytest.raises(Stage5ValidationError, match="Label decision_time must be unique"):
        build_dataset_and_splits(features, pd.concat([labels, labels.iloc[[0]]], ignore_index=True), manifest(), label_dictionary(), CONFIG)


def test_time_relation_error_fails_and_segment_mismatch_is_audited() -> None:
    good = ms("2020-01-01T00:00:00Z")
    bad_time = ms("2020-01-01T00:05:00Z")
    bad_segment = ms("2020-01-01T00:10:00Z")
    features = make_features([good, bad_time, bad_segment])
    labels = make_labels([good, bad_time, bad_segment])
    labels.loc[labels["decision_time"] == bad_time, "entry_minute_open_time"] = bad_time + 60_000

    with pytest.raises(Stage5ValidationError, match="entry_minute_open_time"):
        build_dataset_and_splits(features, labels, manifest(), label_dictionary(), CONFIG)

    labels = make_labels([good, bad_segment])
    features = make_features([good, bad_segment])
    labels.loc[labels["decision_time"] == bad_segment, "continuity_segment_id"] = 99
    result = build_dataset_and_splits(features, labels, manifest(), label_dictionary(), CONFIG)
    audit = result.join_audit.set_index("decision_time")

    assert len(result.dataset) == 1
    assert "SEGMENT_MISMATCH" in audit.loc[bad_segment, "exclusion_reason"]


def test_final_intersection_excludes_incomplete_features_invalid_labels_and_missing_target() -> None:
    good = ms("2020-01-01T00:00:00Z")
    incomplete = ms("2020-01-01T00:05:00Z")
    invalid_label = ms("2020-01-01T00:10:00Z")
    missing_target = ms("2020-01-01T00:15:00Z")
    features = make_features([good, incomplete, invalid_label, missing_target])
    labels = make_labels([good, incomplete, invalid_label, missing_target])
    features.loc[features["decision_time"] == incomplete, "is_final_feature_candidate"] = False
    features.loc[features["decision_time"] == incomplete, "feature_missing_count"] = 2
    labels.loc[labels["decision_time"] == invalid_label, "is_final_model_label_candidate"] = False
    labels.loc[labels["decision_time"] == invalid_label, "is_valid_proxy_label"] = False
    labels.loc[labels["decision_time"] == missing_target, "label_up_60m"] = pd.NA

    result = build_dataset_and_splits(features, labels, manifest(), label_dictionary(), CONFIG)
    audit = result.join_audit.set_index("decision_time")

    assert result.dataset["decision_time"].tolist() == [good]
    assert "INCOMPLETE_FEATURES" in audit.loc[incomplete, "exclusion_reason"]
    assert "INVALID_LABEL" in audit.loc[invalid_label, "exclusion_reason"]
    invalid_reasons = audit.loc[invalid_label, "exclusion_reason"].split(";")
    assert invalid_reasons.count("INVALID_LABEL") == 1
    assert "MISSING_TARGET" in audit.loc[missing_target, "exclusion_reason"]


def test_manifest_roles_and_leakage_exclusions() -> None:
    dataset_manifest = make_dataset_manifest(manifest(), CONFIG, {}, {}, {}, {}, row_count=0, column_count=0)

    assert dataset_manifest["feature_columns"] == FEATURE_NAMES
    assert len(dataset_manifest["feature_columns"]) == 63
    assert "label_up_60m" not in dataset_manifest["feature_columns"]
    assert "sample_weight_margin" not in dataset_manifest["feature_columns"]
    assert "continuity_segment_id" in dataset_manifest["identifier_columns"]
    assert dataset_manifest["primary_target_columns"] == ["label_up_60m"]
    assert "sample_weight_margin" in dataset_manifest["sample_weight_columns"]
    assert "settlement_minute_open_time" in dataset_manifest["forbidden_model_input_columns"]


def test_final_test_and_development_purge_rules() -> None:
    dev = ms("2024-12-31T22:55:00Z")
    purged_equal = ms("2024-12-31T23:00:00Z")
    test = ms("2025-01-01T00:00:00Z")
    result = build_dataset_and_splits(make_features([dev, purged_equal, test]), make_labels([dev, purged_equal, test]), manifest(), label_dictionary(), CONFIG)
    split = result.split_assignments.set_index("decision_time")

    assert split.loc[dev, "final_split_role"] == "DEVELOPMENT"
    assert split.loc[purged_equal, "settlement_minute_open_time"] == ms("2025-01-01T00:00:00Z")
    assert split.loc[purged_equal, "final_split_role"] == "PURGED_BEFORE_FINAL_TEST"
    assert split.loc[test, "final_split_role"] == "FINAL_TEST"
    for fold in ["fold_2020_role", "fold_2021_role", "fold_2022_role", "fold_2023_role", "fold_2024_role"]:
        assert split.loc[test, fold] == "FINAL_TEST_EXCLUDED"


def test_fold_2020_train_validation_and_purge_boundaries() -> None:
    train = ms("2019-12-31T22:55:00Z")
    purge_before = ms("2019-12-31T23:00:00Z")
    validation = ms("2020-01-01T00:00:00Z")
    validation_end_purge = ms("2020-12-31T23:00:00Z")
    outside_next = ms("2021-01-01T00:00:00Z")
    times = [train, purge_before, validation, validation_end_purge, outside_next]
    result = build_dataset_and_splits(make_features(times), make_labels(times), manifest(), label_dictionary(), CONFIG)
    split = result.split_assignments.set_index("decision_time")

    assert split.loc[train, "fold_2020_role"] == "TRAIN"
    assert split.loc[purge_before, "fold_2020_role"] == "PURGED_BEFORE_VALIDATION"
    assert split.loc[validation, "fold_2020_role"] == "VALIDATION"
    assert split.loc[validation_end_purge, "fold_2020_role"] == "PURGED_AT_VALIDATION_END"
    assert split.loc[outside_next, "fold_2020_role"] == "OUTSIDE_FOLD"
    assert split.loc[outside_next, "fold_2021_role"] == "VALIDATION"
    fold = next(item for item in result.cv_fold_manifest["folds"] if item["name"] == "fold_2020")
    assert fold["train_max_settlement_time"] < fold["validation_min_decision_time"]


def test_240m_history_can_cross_validation_boundary_without_removal() -> None:
    row = ms("2020-01-01T00:00:00Z")
    features = make_features([row])
    labels = make_labels([row])
    features["has_history_240m"] = True
    result = build_dataset_and_splits(features, labels, manifest(), label_dictionary(), CONFIG)

    assert result.split_assignments.loc[0, "fold_2020_role"] == "VALIDATION"


def test_evaluation_offsets_and_primary_nonoverlap_flags() -> None:
    times = [ms("2020-01-01T00:00:00Z") + i * 5 * 60_000 for i in range(12)]
    result = build_dataset_and_splits(make_features(times), make_labels(times), manifest(), label_dictionary(), CONFIG)
    split = result.split_assignments

    assert split["evaluation_offset_minutes"].tolist() == list(range(0, 60, 5))
    assert split["evaluation_offset_minutes"].isin(list(range(0, 60, 5))).all()
    assert split["is_primary_nonoverlap_evaluation"].sum() == 1
    assert bool(split.loc[split["evaluation_offset_minutes"] == 0, "is_primary_nonoverlap_evaluation"].iloc[0]) is True


def test_output_dataset_and_split_one_to_one() -> None:
    times = [ms("2020-01-01T00:00:00Z"), ms("2020-01-01T00:05:00Z")]
    result = build_dataset_and_splits(make_features(times), make_labels(times), manifest(), label_dictionary(), CONFIG)

    assert result.dataset["dataset_row_id"].tolist() == result.split_assignments["dataset_row_id"].tolist()
    assert result.dataset["decision_time"].tolist() == result.split_assignments["decision_time"].tolist()


def test_run_stage5_preserves_stage3b_and_stage4_inputs(tmp_path: Path) -> None:
    times = [ms("2020-01-01T00:00:00Z"), ms("2020-01-01T00:05:00Z")]
    features = make_features(times)
    labels = make_labels(times)
    feature_path = tmp_path / "features.parquet"
    label_path = tmp_path / "labels.parquet"
    manifest_path = tmp_path / "manifest.json"
    label_dict_path = tmp_path / "label_dictionary.json"
    features.to_parquet(feature_path, index=False)
    labels.to_parquet(label_path, index=False)
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    label_dict_path.write_text(json.dumps(label_dictionary()), encoding="utf-8")
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in [feature_path, label_path]}
    config = {
        **CONFIG,
        "feature_input_path": "features.parquet",
        "label_input_path": "labels.parquet",
        "feature_manifest_path": "manifest.json",
        "label_field_dictionary_path": "label_dictionary.json",
        "output_dataset_path": "out/dataset.parquet",
        "output_split_path": "out/splits.parquet",
        "log_path": "reports/log.txt",
        "report_paths": {
            "main_report": "reports/report.md",
            "dataset_manifest": "reports/dataset_manifest.json",
            "cv_fold_manifest": "reports/cv_manifest.json",
            "split_summary": "reports/split_summary.csv",
            "split_distribution_by_year": "reports/by_year.csv",
            "split_distribution_by_month": "reports/by_month.csv",
            "excluded_samples": "reports/excluded.csv",
            "join_audit": "reports/join_audit.csv",
        },
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    run_stage5(config_path, tmp_path)

    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == before[p] for p in before)
    dataset = pd.read_parquet(tmp_path / "out/dataset.parquet")
    splits = pd.read_parquet(tmp_path / "out/splits.parquet")
    assert len(dataset) == len(splits) == 2
    role_columns = []
    for key in [
        "identifier_columns",
        "time_columns",
        "feature_columns",
        "primary_target_columns",
        "auxiliary_target_columns",
        "sample_weight_columns",
        "evaluation_metadata_columns",
    ]:
        role_columns.extend(json.loads((tmp_path / "reports/dataset_manifest.json").read_text(encoding="utf-8"))[key])
    assert set(dataset.columns) == set(role_columns)
    assert (tmp_path / "reports/dataset_manifest.json").exists()
    assert (tmp_path / "reports/cv_manifest.json").exists()


def ms(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def manifest() -> dict[str, object]:
    return {
        "feature_set_version": "kline_v1_63",
        "ordered_feature_names": FEATURE_NAMES,
        "feature_count": 63,
        "feature_definition_hash": "test_hash",
    }


def label_dictionary() -> dict[str, object]:
    return {
        "fields": [
            {"name": "label_up_60m", "allowed_as_model_input": False, "is_model_label": True},
            {"name": "sample_weight_margin", "allowed_as_model_input": False, "role": "sample_weight", "derived_from_target": True},
        ]
    }


def make_features(times: list[int]) -> pd.DataFrame:
    data: dict[str, object] = {
        "feature_open_time": np.array(times, dtype=np.int64) - 60_000,
        "decision_time": np.array(times, dtype=np.int64),
        "continuity_segment_id": np.ones(len(times), dtype=np.int64),
        "segment_row_number": np.arange(300, 300 + len(times), dtype=np.int64),
        "has_history_240m": np.ones(len(times), dtype=bool),
        "has_future_61m": np.ones(len(times), dtype=bool),
        "is_prediction_time_5m": np.ones(len(times), dtype=bool),
        "is_model_candidate": np.ones(len(times), dtype=bool),
        "feature_set_version": ["kline_v1_63"] * len(times),
        "feature_missing_count": np.zeros(len(times), dtype=np.int16),
        "has_nonfinite_feature": np.zeros(len(times), dtype=bool),
        "is_feature_complete": np.ones(len(times), dtype=bool),
        "is_final_feature_candidate": np.ones(len(times), dtype=bool),
    }
    for i, name in enumerate(FEATURE_NAMES):
        data[name] = np.full(len(times), i / 1000.0, dtype=np.float32)
    return pd.DataFrame(data)


def make_labels(times: list[int]) -> pd.DataFrame:
    simple_return = np.where(np.arange(len(times)) % 2 == 0, 0.01, -0.01)
    return pd.DataFrame(
        {
            "feature_open_time": np.array(times, dtype=np.int64) - 60_000,
            "decision_time": np.array(times, dtype=np.int64),
            "entry_minute_open_time": np.array(times, dtype=np.int64),
            "settlement_minute_open_time": np.array(times, dtype=np.int64) + 3_600_000,
            "continuity_segment_id": np.ones(len(times), dtype=np.int64),
            "is_prediction_time_5m": np.ones(len(times), dtype=bool),
            "is_valid_proxy_label": np.ones(len(times), dtype=bool),
            "is_final_model_label_candidate": np.ones(len(times), dtype=bool),
            "entry_price_proxy": np.full(len(times), 100.0),
            "settlement_price_proxy": 100.0 * (1.0 + simple_return),
            "future_simple_return_60m": simple_return,
            "future_log_return_60m": np.log1p(simple_return),
            "label_up_60m": (simple_return > 0).astype("int8"),
            "absolute_future_return_bps": np.abs(simple_return) * 10000,
            "proxy_margin_bucket": ["[50, +inf)"] * len(times),
            "proxy_boundary_risk_1bps": np.zeros(len(times), dtype=bool),
            "proxy_boundary_risk_2_5bps": np.zeros(len(times), dtype=bool),
            "proxy_boundary_risk_5bps": np.zeros(len(times), dtype=bool),
            "proxy_boundary_risk_10bps": np.zeros(len(times), dtype=bool),
            "sample_weight_uniform": np.ones(len(times), dtype=np.float32),
            "sample_weight_margin": np.ones(len(times), dtype=np.float32),
        }
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

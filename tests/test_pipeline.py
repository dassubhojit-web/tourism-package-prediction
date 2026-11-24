"""Check the data boundary, held-out evaluation, and fitted deployment artifact.

These tests use the saved training outputs and never fit another model.
"""

import json
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import fbeta_score

from tourism_project.model_building.schema import (
    DEPLOYMENT,
    EXCLUDED_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    PROCESSED,
    RAW_DATA,
    REPORTS,
    TARGET,
    clean_profiles,
    file_hash,
    profile_hashes,
    validate_raw,
)
from tourism_project.model_building.train import choose_threshold, load_splits


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(RAW_DATA)


@pytest.fixture(scope="module")
def splits():
    return load_splits(PROCESSED)


@pytest.fixture(scope="module")
def payload():
    return joblib.load(DEPLOYMENT / "model.joblib")


@pytest.fixture
def copied_splits(tmp_path):
    """Integrity failure tests operate exclusively on temporary copies."""
    for name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv", "split_manifest.json"]:
        shutil.copyfile(PROCESSED / name, tmp_path / name)
    return tmp_path


def replace_split_and_checksum(directory, name, frame):
    """Allow semantic integrity tests to get past the byte-integrity check."""
    path = directory / name
    frame.to_csv(path, index=False)
    manifest_path = directory / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["split_hashes"][name] = file_hash(path)
    manifest_path.write_text(json.dumps(manifest))


def test_registration_accepts_actual_data_and_missing_predictor_values(raw):
    validate_raw(raw)
    partial = raw.copy()
    partial.loc[partial.index[0], ["Age", "MonthlyIncome", "Occupation", "Gender"]] = np.nan
    validate_raw(partial)


@pytest.mark.parametrize("column", ["Age", "ProdTaken", "CustomerID", "DurationOfPitch"])
def test_registration_rejects_missing_contract_column(raw, column):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw(raw.drop(columns=column))


@pytest.mark.parametrize("invalid", [2, np.nan])
def test_registration_rejects_invalid_or_missing_target(raw, invalid):
    changed = raw.copy()
    changed.loc[changed.index[0], TARGET] = invalid
    with pytest.raises(ValueError, match="ProdTaken"):
        validate_raw(changed)


def test_registration_rejects_reused_customer_identifier(raw):
    changed = raw.copy()
    changed.loc[changed.index[1], "CustomerID"] = changed.iloc[0]["CustomerID"]
    with pytest.raises(ValueError, match="unique"):
        validate_raw(changed)


def test_saved_splits_match_source_and_are_profile_disjoint(raw, splits):
    Xtrain, Xtest, ytrain, ytest, manifest = splits
    assert manifest["raw_sha256"] == file_hash(RAW_DATA)
    assert list(Xtrain) == list(Xtest) == FEATURE_COLUMNS
    assert len(Xtrain) == len(ytrain) == manifest["train_rows"]
    assert len(Xtest) == len(ytest) == manifest["test_rows"]
    assert set(ytrain) == set(ytest) == {0, 1}
    assert not set(EXCLUDED_FEATURES + ["CustomerID", TARGET]) & set(Xtrain.columns)
    hashes = profile_hashes(pd.concat([Xtrain, Xtest], ignore_index=True))
    assert hashes.is_unique, "Replicated profiles must not reach CV folds or the holdout."
    assert manifest["train_test_profile_overlap"] == 0
    expected_rows = len(clean_profiles(raw).drop_duplicates(subset=[*FEATURE_COLUMNS, TARGET]))
    assert len(Xtrain) + len(Xtest) == expected_rows == manifest["unique_rows"]
    assert manifest["duplicates_removed"] == len(raw) - expected_rows
    assert abs(ytrain.mean() - ytest.mean()) < 0.01


def test_artifact_loader_rejects_modified_bytes(copied_splits):
    path = copied_splits / "Xtrain.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_splits(copied_splits)


def test_artifact_loader_rejects_changed_column_order(copied_splits):
    frame = pd.read_csv(copied_splits / "Xtrain.csv")
    replace_split_and_checksum(copied_splits, "Xtrain.csv", frame[frame.columns[::-1]])
    with pytest.raises(ValueError, match="feature order"):
        load_splits(copied_splits)


def test_artifact_loader_rejects_misaligned_targets(copied_splits):
    target = pd.read_csv(copied_splits / "ytrain.csv").iloc[:-1]
    replace_split_and_checksum(copied_splits, "ytrain.csv", target)
    with pytest.raises(ValueError, match="row counts differ"):
        load_splits(copied_splits)


def test_artifact_loader_rejects_profile_overlap(copied_splits):
    train = pd.read_csv(copied_splits / "Xtrain.csv")
    test = pd.read_csv(copied_splits / "Xtest.csv")
    test.iloc[0] = train.iloc[0]
    replace_split_and_checksum(copied_splits, "Xtest.csv", test)
    with pytest.raises(ValueError, match="Train/test profile overlap"):
        load_splits(copied_splits)


def test_artifact_loader_rejects_replicas_within_training(copied_splits):
    train = pd.read_csv(copied_splits / "Xtrain.csv")
    train.iloc[1] = train.iloc[0]
    replace_split_and_checksum(copied_splits, "Xtrain.csv", train)
    with pytest.raises(ValueError, match="Duplicate training profiles"):
        load_splits(copied_splits)


def test_threshold_maximizes_f2_for_supplied_training_predictions():
    labels = np.array([0, 0, 1, 1, 0, 1])
    training_oof_scores = np.array([0.05, 0.10, 0.20, 0.30, 0.40, 0.90])
    threshold, table = choose_threshold(labels, training_oof_scores)
    independent_scores = {
        candidate: fbeta_score(labels, training_oof_scores >= candidate, beta=2)
        for candidate in np.unique(training_oof_scores)
    }
    assert threshold == pytest.approx(0.20)
    assert independent_scores[threshold] == pytest.approx(max(independent_scores.values()))
    assert table["f2"].max() == pytest.approx(max(independent_scores.values()))


def test_threshold_tie_prefers_higher_contact_cutoff():
    # Contacting the highest-scoring buyer or all ten people produces identical F2.
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    scores = np.arange(1, 11) / 10
    threshold, _ = choose_threshold(labels, scores)
    assert fbeta_score(labels, scores >= 0.1, beta=2) == pytest.approx(
        fbeta_score(labels, scores >= 1.0, beta=2)
    )
    assert threshold == pytest.approx(1.0)


def test_serialized_pipeline_selects_features_by_name(payload, splits):
    frame = splits[1].iloc[:8].copy()
    pipeline = payload["pipeline"]
    expected = pipeline.predict_proba(frame)
    reordered = pipeline.predict_proba(frame[frame.columns[::-1]])
    np.testing.assert_allclose(reordered, expected, rtol=0, atol=0)


def test_serialized_pipeline_handles_unknown_categories_and_numeric_nulls(payload, splits):
    frame = splits[1].iloc[:3].copy()
    frame.loc[:, "Occupation"] = "Previously unseen occupation"
    frame.loc[:, "Designation"] = "Previously unseen designation"
    for column in NUMERIC_COLUMNS:
        frame[column] = np.nan
    probabilities = payload["pipeline"].predict_proba(frame)
    assert probabilities.shape == (3, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))


def test_deployment_payload_metadata_and_evaluation_are_consistent(payload, splits):
    metadata = json.loads((DEPLOYMENT / "metadata.json").read_text())
    report = json.loads((REPORTS / "metrics.json").read_text())
    assert metadata == report
    for key in ["threshold", "feature_columns", "model_name", "raw_sha256", "versions"]:
        assert payload[key] == metadata[key]
    assert payload["feature_columns"] == FEATURE_COLUMNS
    assert 0 <= payload["threshold"] <= 1
    assert "training" in metadata["threshold_selection"]
    scores = payload["pipeline"].predict_proba(splits[1])[:, 1]
    predictions = pd.read_csv(REPORTS / "test_predictions.csv")
    np.testing.assert_allclose(scores, predictions["score"], rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(predictions["actual"], splits[3])
    np.testing.assert_array_equal(predictions["prediction"], scores >= payload["threshold"])

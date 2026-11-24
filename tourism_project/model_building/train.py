"""Tune complete pipelines, track all candidates, select a training-only threshold, evaluate once."""
import argparse
import json
import math
import os
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from tourism_project.model_building.schema import (
    PROCESSED, REPORTS, DEPLOYMENT, PROJECT, SEED, TARGET,
    FEATURE_COLUMNS, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS,
    file_hash, profile_hashes, save_json,
)


def make_pipeline(model):
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_COLUMNS),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def evaluate(y, scores, threshold):
    predicted = (np.asarray(scores) >= threshold).astype(int)
    budget = max(1, math.ceil(0.20 * len(y)))
    top = np.argsort(-np.asarray(scores), kind="stable")[:budget]
    prevalence = float(np.asarray(y).mean())
    top_precision = float(np.asarray(y)[top].mean())
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "f2": float(fbeta_score(y, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "purchase_rate": prevalence,
        "precision_at_top_20_percent": top_precision,
        "lift_at_top_20_percent": top_precision / prevalence,
        "top_20_percent_customers": budget,
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
    }


def choose_threshold(y, scores):
    """Maximize F2 on out-of-fold training scores; ties prefer the higher threshold."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    best = np.flatnonzero(np.isclose(f2, f2.max()))[-1]
    return float(thresholds[best]), pd.DataFrame({
        "threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "f2": f2,
    })


def load_splits(split_dir):
    manifest = json.loads((split_dir / "split_manifest.json").read_text())
    for name, expected in manifest["split_hashes"].items():
        if file_hash(split_dir / name) != expected:
            raise ValueError(f"Split checksum mismatch: {name}")
    Xtrain = pd.read_csv(split_dir / "Xtrain.csv")
    Xtest = pd.read_csv(split_dir / "Xtest.csv")
    ytrain = pd.read_csv(split_dir / "ytrain.csv")[TARGET]
    ytest = pd.read_csv(split_dir / "ytest.csv")[TARGET]
    if list(Xtrain.columns) != FEATURE_COLUMNS or list(Xtest.columns) != FEATURE_COLUMNS:
        raise ValueError("Split feature order does not match the data contract.")
    if len(Xtrain) != len(ytrain) or len(Xtest) != len(ytest):
        raise ValueError("Feature and target row counts differ.")
    if set(profile_hashes(Xtrain)) & set(profile_hashes(Xtest)):
        raise ValueError("Train/test profile overlap detected.")
    if profile_hashes(Xtrain).duplicated().any():
        raise ValueError("Duplicate training profiles would contaminate cross-validation.")
    return Xtrain, Xtest, ytrain, ytest, manifest


def train(split_dir=PROCESSED, n_jobs=2):
    Xtrain, Xtest, ytrain, ytest, manifest = load_splits(Path(split_dir))
    REPORTS.mkdir(parents=True, exist_ok=True)
    DEPLOYMENT.mkdir(parents=True, exist_ok=True)
    figures = REPORTS / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 135, "axes.spines.top": False, "axes.spines.right": False})
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    candidates = {
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"), {
            "model__max_depth": [4, 8], "model__min_samples_leaf": [5, 15],
        }),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                   class_weight="balanced", n_jobs=1), {
            "model__max_depth": [None, 12], "model__min_samples_leaf": [1, 3],
            "model__max_features": ["sqrt", 0.8],
        }),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED), {
            "model__n_estimators": [120, 220], "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        }),
    }
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", (PROJECT / "mlruns").as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Tourism Package Prediction - Precontact")
    all_results, winners = [], []
    with mlflow.start_run(run_name="precontact-model-selection") as parent:
        mlflow.log_params({"random_state": SEED, "selection_metric": "average_precision",
                           "cv_folds": 3, "threshold_objective": "training_oof_f2",
                           "raw_sha256": manifest["raw_sha256"],
                           "feature_columns": json.dumps(FEATURE_COLUMNS),
                           "python_version": platform.python_version(),
                           "sklearn_version": sklearn.__version__,
                           "profile_deduplication": True})
        for name, (estimator, grid) in candidates.items():
            search = GridSearchCV(make_pipeline(estimator), grid, scoring={
                "average_precision": "average_precision", "roc_auc": "roc_auc",
                "recall": "recall", "f1": "f1",
            }, refit="average_precision", cv=cv, n_jobs=n_jobs,
                return_train_score=True, error_score="raise")
            search.fit(Xtrain, ytrain)
            result = pd.DataFrame(search.cv_results_)
            result.insert(0, "model_name", name)
            all_results.append(result)
            winners.append((search.best_score_, name, search.best_estimator_, search.best_params_))
            for i, row in result.iterrows():
                with mlflow.start_run(run_name=f"{name}-candidate-{i + 1:02}", nested=True):
                    mlflow.log_params({"model_name": name, **estimator.get_params(), **row["params"]})
                    for field in result.columns:
                        if field.startswith(("mean_", "std_", "split")) and isinstance(row[field], (int, float, np.number)):
                            mlflow.log_metric(field, float(row[field]))
            print(f"{name}: best 3-fold training average precision = {search.best_score_:.4f}", flush=True)

        best_score, model_name, pipeline, best_params = max(winners, key=lambda item: item[0])
        # The test set is still untouched. Threshold comes exclusively from training folds.
        oof_scores = cross_val_predict(clone(pipeline), Xtrain, ytrain, cv=cv,
                                      method="predict_proba", n_jobs=n_jobs)[:, 1]
        threshold, threshold_table = choose_threshold(ytrain, oof_scores)
        threshold_table.to_csv(REPORTS / "threshold_search.csv", index=False)
        cv_results = pd.concat(all_results, ignore_index=True)
        cv_results["params"] = cv_results["params"].map(lambda value: json.dumps(value, sort_keys=True))
        cv_results.to_csv(REPORTS / "cv_results.csv", index=False)
        # GridSearchCV already refitted the selected pipeline on every training row.
        train_scores = pipeline.predict_proba(Xtrain)[:, 1]
        test_scores = pipeline.predict_proba(Xtest)[:, 1]
        dummy = DummyClassifier(strategy="prior").fit(Xtrain, ytrain)
        baseline_scores = dummy.predict_proba(Xtest)[:, 1]
        test_metrics = evaluate(ytest, test_scores, threshold)
        report = {
            "model_name": model_name, "threshold": threshold,
            "threshold_selection": "maximum F2 on training out-of-fold predictions; no test tuning",
            "selection_metric": "3-fold training average precision", "best_cv_average_precision": float(best_score),
            "best_params": best_params, "feature_columns": FEATURE_COLUMNS,
            "train_rows": len(Xtrain), "test_rows": len(Xtest),
            "train_metrics": evaluate(ytrain, train_scores, threshold),
            "training_oof_metrics": evaluate(ytrain, oof_scores, threshold),
            "test_metrics": test_metrics,
            "test_metrics_at_0_5": evaluate(ytest, test_scores, 0.5),
            "baseline_test_metrics": evaluate(ytest, baseline_scores, 0.5),
            "raw_sha256": manifest["raw_sha256"], "mlflow_run_id": parent.info.run_id,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        }
        save_json(REPORTS / "metrics.json", report)
        save_json(DEPLOYMENT / "metadata.json", report)
        payload = {"pipeline": pipeline, "threshold": threshold, "feature_columns": FEATURE_COLUMNS,
                   "model_name": model_name, "raw_sha256": manifest["raw_sha256"], "versions": report["versions"]}
        joblib.dump(payload, DEPLOYMENT / "model.joblib", compress=3)
        predicted = (test_scores >= threshold).astype(int)
        (REPORTS / "classification_report.txt").write_text(classification_report(
            ytest, predicted, target_names=["No purchase", "Purchase"], zero_division=0), encoding="utf-8")
        pd.DataFrame({"actual": ytest, "score": test_scores, "prediction": predicted}).to_csv(
            REPORTS / "test_predictions.csv", index=False)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        ConfusionMatrixDisplay.from_predictions(ytest, predicted, ax=axes[0], colorbar=False, cmap="Blues")
        axes[0].set_title(f"Test decisions, threshold {threshold:.3f}")
        RocCurveDisplay.from_predictions(ytest, test_scores, ax=axes[1], name=model_name, plot_chance_level=True)
        PrecisionRecallDisplay.from_predictions(ytest, test_scores, ax=axes[2], name=model_name, plot_chance_level=True)
        fig.tight_layout()
        fig.savefig(figures / "test_evaluation.png", bbox_inches="tight")
        plt.close(fig)
        # Descriptive holdout importance only: never used to select features or tune this release.
        importance = permutation_importance(pipeline, Xtest, ytest, scoring="average_precision",
                                            n_repeats=5, random_state=SEED, n_jobs=n_jobs)
        importances = pd.DataFrame({"feature": FEATURE_COLUMNS,
                                    "mean_ap_decrease": importance.importances_mean,
                                    "std_ap_decrease": importance.importances_std}).sort_values("mean_ap_decrease", ascending=False)
        importances.to_csv(REPORTS / "feature_importance.csv", index=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        shown = importances.sort_values("mean_ap_decrease")
        ax.barh(shown["feature"], shown["mean_ap_decrease"], xerr=shown["std_ap_decrease"], color="#167d8d")
        ax.set_xlabel("Decrease in test average precision after shuffling (5 repeats)")
        ax.set_title("Pre-contact predictors: descriptive permutation importance")
        fig.tight_layout()
        fig.savefig(figures / "feature_importance.png", bbox_inches="tight")
        plt.close(fig)
        mlflow.log_params({"selected_model": model_name, "selected_threshold": threshold,
                           **{f"best_{key}": str(value) for key, value in best_params.items()}})
        mlflow.log_metrics({f"test_{key}": float(value) for key, value in test_metrics.items()
                            if isinstance(value, (int, float))})
        mlflow.log_metric("best_cv_average_precision", best_score)
        mlflow.log_artifact(str(DEPLOYMENT / "model.joblib"), artifact_path="model")
        mlflow.log_artifact(str(DEPLOYMENT / "metadata.json"), artifact_path="model")
        mlflow.log_artifact(str(Path(split_dir) / "split_manifest.json"), artifact_path="data")
        for artifact in REPORTS.glob("*"):
            if artifact.is_file():
                mlflow.log_artifact(str(artifact), artifact_path="reports")
        mlflow.log_artifacts(str(figures), artifact_path="figures")
    print(json.dumps({"model": model_name, "threshold": threshold, "test_metrics": test_metrics}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=PROCESSED)
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    train(args.split_dir, args.n_jobs)

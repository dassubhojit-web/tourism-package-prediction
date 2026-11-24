"""Build the original, self-contained learner submission from the repository's actual source."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


def source(path, explanation=None):
    if explanation:
        md(explanation)
    contents = (ROOT / path).read_text(encoding="utf-8")
    cells.append(nbf.v4.new_code_cell(f"%%writefile {path}\n{contents}"))


md("""
# Tourism Package Prediction
### Visit with Us · Advanced Machine Learning and MLOps

**Objective:** build a reproducible system to prioritize customers for a Wellness Tourism Package **before a sales conversation**.

This notebook follows the supplied learner template and the adjacent MLOps examples. It contains the complete project source, data validation and preparation, training with experiment tracking, evaluated predictions, a Streamlit frontend, and a GitHub Actions workflow. Run every cell from top to bottom; only the supplied `tourism.csv` and the Python dependencies are needed to reconstruct the local project.

**Delivery status:** local training, saved model, notebook execution, and application tests are real. GitHub publication, a remote Actions run, and public Streamlit deployment are pending. No public links or remote screenshots are claimed.
""")
md("""
## 1. Business context and success criteria

Manual targeting makes outreach inconsistent and expensive. The classifier ranks potential buyers so a limited marketing budget can reach interested customers. `ProdTaken=1` records historical package uptake; `0` records no purchase.

**Primary model-selection metric:** average precision (AP), which summarizes precision and recall across score thresholds. Most customers did not buy, so accuracy alone can reward a model that misses every buyer. Also report ROC AUC, precision, recall, F1, F2, balanced accuracy, and campaign lift.

The outreach threshold maximizes **F2 on training out-of-fold predictions**, giving recall more weight than precision. This is an explicit provisional business preference because acquisition cost and package margin were not supplied. The untouched test set evaluates that fixed decision rule. No threshold is selected from test results.

Historical package purchases do not prove future uptake of a new wellness offer or the incremental effect of contacting a customer. A future campaign pilot is needed to establish those outcomes.
""")
md("""
## 2. Data dictionary and prediction-time availability

| Field | Meaning | Treatment |
|---|---|---|
| `CustomerID`, exported CSV index | Record identifiers | Remove |
| `ProdTaken` | Historical purchase, 0/1 | Target |
| `Age` | Customer age | Numeric predictor |
| `CityTier` | City category, 1/2/3 | Categorical predictor |
| `Occupation`, `Gender` | Customer profile | Categorical predictors |
| `NumberOfPersonVisiting` | Total travel party | Numeric predictor |
| `PreferredPropertyStar` | Hotel rating preference, 3/4/5 | Categorical predictor |
| `MaritalStatus` | Marital status | Categorical predictor |
| `NumberOfTrips` | Annual trips | Numeric predictor |
| `Passport`, `OwnCar` | Binary profile flags | Categorical predictors |
| `NumberOfChildrenVisiting` | Children below five | Numeric predictor |
| `Designation` | Job designation | Categorical predictor; verify availability |
| `MonthlyIncome` | Gross monthly income in source units | Numeric predictor |
| `TypeofContact` | Actual contact method | Exclude before contact |
| `DurationOfPitch`, `NumberOfFollowups` | Realized sales activity | Exclude before contact |
| `ProductPitched`, `PitchSatisfactionScore` | Pitch content and response | Exclude before contact |

Travel preferences, income, and designation are assumed available from an existing customer profile before the next campaign. The app therefore requires **13 pre-contact fields**. Confirm this assumption with the business. `Designation` perfectly maps to `ProductPitched` in this historical file: it is acceptable only if the job designation was collected independently before the sales interaction. Otherwise remove it and retrain.
""")
md("""
## 3. Environment and project configuration

Training and serving use Python 3.12 with identical NumPy, pandas, scikit-learn, and joblib pins. MLflow writes to a local tracking directory; no token, ngrok tunnel, or tracking server is needed. The template and rubric specify Streamlit Community Cloud; that takes precedence over the older Hugging Face reference in the description.

Run from this notebook's directory. In Colab, upload this notebook and `tourism.csv` to the same runtime directory, select Python 3.12 where available, and install the pinned packages before running the rest. If an existing runtime has already imported different package versions, restart it after installation.
""")
code("""
from pathlib import Path
import importlib.metadata as metadata
import os
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path.cwd().resolve()
if (ROOT / "Tourism Package Prediction" / "tourism.csv").is_file():
    ROOT = ROOT / "Tourism Package Prediction"
os.chdir(ROOT)
print("Project root:", ROOT.name)
print("Executed:", datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds"))
print("Python:", sys.version.split()[0])
pins = {
    "numpy": "2.2.6", "pandas": "2.2.3", "scikit-learn": "1.6.1", "joblib": "1.5.1",
    "mlflow-skinny": "3.0.1", "streamlit": "1.45.1", "matplotlib": "3.10.3",
    "seaborn": "0.13.2", "nbformat": "5.10.4", "nbconvert": "7.16.6",
    "ipykernel": "6.29.5", "pytest": "8.3.5", "PyYAML": "6.0.2",
}
missing = []
for package, version in pins.items():
    try:
        actual = metadata.version(package)
    except metadata.PackageNotFoundError:
        actual = None
    if actual != version:
        missing.append(f"{package}=={version}")
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
else:
    print("All pinned project dependencies are installed.")
if sys.version_info[:2] != (3, 12):
    print("Use Python 3.12 when exporting a model for the matching deployment environment.")
for relative in ["tourism_project/data/processed", "tourism_project/model_building",
                 "tourism_project/deployment", "tourism_project/reports/figures",
                 ".github/workflows", ".streamlit", "tests"]:
    (ROOT / relative).mkdir(parents=True, exist_ok=True)
source_csv = ROOT / "tourism.csv"
registered_csv = ROOT / "tourism_project/data/tourism.csv"
if source_csv.is_file():
    shutil.copy2(source_csv, registered_csv)
if not registered_csv.is_file():
    raise FileNotFoundError("Upload tourism.csv beside this notebook, then rerun this cell.")
print("Dataset is present in tourism_project/data/tourism.csv.")
""")
source("tourism_project/__init__.py")
source("tourism_project/model_building/__init__.py")
source("tourism_project/model_building/schema.py", "### Shared schema and deterministic cleanup\nThe schema removes identifiers and interaction fields, corrects two spelling variants, and preserves the distinct source categories `Single` and `Unmarried`. Learned imputation stays inside the training pipeline.")
code("""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown, Image, Code
from tourism_project.model_building.schema import (
    PROJECT, REPORTS, PROCESSED, RAW_DATA, TARGET, FEATURE_COLUMNS,
    EXCLUDED_FEATURES, clean_profiles, profile_hashes,
)
sns.set_theme(style="whitegrid", palette="crest")
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 110)
plt.rcParams["figure.dpi"] = 125
""")
md("## 4. Data registration · rubric: 3 points\nValidate the repository input before any model is trained. Registration checks expected columns, unique IDs, valid labels, numeric values, and basic domains, then records a SHA-256 hash, target balance, missing values, and repeated profiles.")
source("tourism_project/model_building/data_register.py")
code("""
from tourism_project.model_building.data_register import register
registration = register()
raw = pd.read_csv(RAW_DATA)
display(raw.head())
display(raw.dtypes.rename("dtype").to_frame())
display(raw.describe().round(2))
""")
code("""
display(Markdown(
    f"**Observed data:** {registration['rows']:,} rows and {registration['columns']} columns; "
    f"{sum(registration['missing_values'].values())} missing cells; "
    f"{registration['target_counts']['1']:,} buyers ({registration['purchase_rate']:.2%}). "
    f"There are {registration['gender_spelling_corrections']} occurrences of `Fe Male`, "
    f"and {registration['duplicate_profiles_after_normalization']} repeated pre-contact profiles after normalization."
))
display(raw.select_dtypes(include="object").nunique().rename("distinct categories").to_frame())
designation_pairs = raw[["Designation", "ProductPitched"]].drop_duplicates().sort_values("Designation")
display(designation_pairs)
""")
md("""
## 5. Data preparation · rubric: 7 points

Drop duplicate normalized profiles **before** the 80/20 stratified split. These records have different IDs but the same 13 available predictors and outcome, so keeping them in both sets would overstate generalization. This deduplication estimates performance on unique observed profiles; it can remove distinct people with coincidentally identical recorded attributes, an explicit limitation of the source data. Identical profiles with conflicting labels cause a clear error for review rather than arbitrary deletion.

Save `Xtrain.csv`, `Xtest.csv`, `ytrain.csv`, and `ytest.csv` plus a checksum manifest. All transformations that learn statistics—imputation and one-hot encoding—are deferred until cross-validation. Keep valid income/trip extremes because no defensible correction rule was supplied. Tree models do not require scaling.
""")
source("tourism_project/model_building/prep.py")
code("""
from tourism_project.model_building.prep import prepare
split_manifest = prepare()
Xtrain = pd.read_csv(PROCESSED / "Xtrain.csv")
Xtest = pd.read_csv(PROCESSED / "Xtest.csv")
ytrain = pd.read_csv(PROCESSED / "ytrain.csv")[TARGET]
ytest = pd.read_csv(PROCESSED / "ytest.csv")[TARGET]
assert not (set(profile_hashes(Xtrain)) & set(profile_hashes(Xtest)))
assert not profile_hashes(Xtrain).duplicated().any()
assert not set(EXCLUDED_FEATURES) & set(Xtrain.columns)
display(pd.DataFrame({
    "split": ["Training", "Final test"], "rows": [len(Xtrain), len(Xtest)],
    "buyers": [int(ytrain.sum()), int(ytest.sum())],
    "purchase rate": [ytrain.mean(), ytest.mean()],
}))
print("Profile overlap: 0. No interaction-derived predictors remain.")
""")
md("### Exploratory analysis on the training set\nThe final test set stays out of exploratory feature analysis and model selection. Associations below describe this historical sample and are not causal effects.")
code("""
eda = Xtrain.assign(ProdTaken=ytrain)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
sns.countplot(data=eda, x=TARGET, ax=axes[0, 0], color="#167d8d")
axes[0, 0].set_title("Training target balance")
axes[0, 0].set_xticks([0, 1], ["No purchase", "Purchase"])
for axis, feature in zip(axes.flat[1:4], ["Age", "MonthlyIncome", "NumberOfTrips"]):
    sns.histplot(data=eda, x=feature, hue=TARGET, bins=25, element="step", ax=axis)
    axis.set_title(feature)
for axis, feature in zip(axes.flat[4:], ["Passport", "Designation"]):
    rates = eda.groupby(feature)[TARGET].agg(["mean", "size"]).reset_index()
    axis.bar(rates[feature].astype(str), rates["mean"], color="#167d8d")
    axis.set_title(f"Purchase rate by {feature}")
    axis.set_ylabel("Historical purchase fraction")
    axis.tick_params(axis="x", rotation=18)
fig.tight_layout()
fig.savefig(REPORTS / "figures/training_eda.png", bbox_inches="tight")
plt.show()
display(eda.groupby("Passport")[TARGET].agg(["size", "mean"]).rename(columns={"size": "customers", "mean": "purchase rate"}))
display(eda.groupby("Designation")[TARGET].agg(["size", "mean"]).rename(columns={"size": "customers", "mean": "purchase rate"}))
""")
code("""
passport_rates = eda.groupby("Passport")[TARGET].mean()
designation_rates = eda.groupby("Designation")[TARGET].mean().sort_values(ascending=False)
display(Markdown(
    f"**Training observations:** passport holders purchased at {passport_rates.get(1, float('nan')):.1%}, "
    f"compared with {passport_rates.get(0, float('nan')):.1%} for customers without one. "
    f"The highest observed designation purchase rate is {designation_rates.index[0]} "
    f"({designation_rates.iloc[0]:.1%}). Groups differ in size, so rates alone are not reliable targeting rules. "
    "The low purchase prevalence motivates precision–recall evaluation and a majority-class baseline. "
    "The income extremes are retained, and no imputation or clipping statistics are fitted on the full dataset."
))
""")
md("""
## 6. Model training and registration with experimentation tracking · rubric: 13 points

Compare **Decision Tree (4 settings), Random Forest (8), and Gradient Boosting (8)** across three stratified training folds: 20 configurations and 60 validation fits, followed by refitting. The full preprocessing pipeline is inside `GridSearchCV`, so each fold fits its own imputers and categorical encoder. Unknown categorical values can be scored without refitting the encoder.

Refit the best configuration by mean training-fold AP. Then generate out-of-fold training scores for that selected configuration and select the F2 threshold. These training OOF scores are used after hyperparameter selection and are **not an unbiased nested-CV estimate**. The final holdout supplies the independent performance assessment. A dummy model using training prevalence supplies a reference; it is not used as a candidate for deployment.

MLflow records a parent run and one nested run per configuration, including the complete estimator parameters, tuned values, fold scores, mean scores, variability, selected threshold, versions, dataset hash, metrics, report artifacts, and the saved model. The joblib payload carries the fitted pipeline, feature order, and threshold together.
""")
source("tourism_project/model_building/train.py")
code("""
from tourism_project.model_building.train import train
results = train(n_jobs=2)
""")
md("### Cross-validation comparison and final holdout results")
code("""
cv_results = pd.read_csv(REPORTS / "cv_results.csv")
best_by_model = cv_results.loc[cv_results.groupby("model_name")["mean_test_average_precision"].idxmax(),
    ["model_name", "mean_train_average_precision", "mean_test_average_precision",
     "std_test_average_precision", "mean_test_roc_auc", "params"]].sort_values("mean_test_average_precision", ascending=False)
display(best_by_model.reset_index(drop=True))
metrics_to_show = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc", "average_precision"]
comparison = pd.DataFrame({
    "Dummy prevalence baseline": results["baseline_test_metrics"],
    "Selected model at 0.5": results["test_metrics_at_0_5"],
    "Selected model at OOF threshold": results["test_metrics"],
}).loc[metrics_to_show]
display(comparison.round(4))
print((REPORTS / "classification_report.txt").read_text())
display(Image(filename=str(REPORTS / "figures/test_evaluation.png")))
""")
code("""
test = results["test_metrics"]
tn, fp, fn, tp = np.asarray(test["confusion_matrix"]).ravel()
display(Markdown(
    f"**Selected release:** {results['model_name']}; decision threshold **{results['threshold']:.3f}**. "
    f"On {len(ytest):,} holdout customers: AP **{test['average_precision']:.3f}**, ROC AUC **{test['roc_auc']:.3f}**, "
    f"precision **{test['precision']:.1%}**, recall **{test['recall']:.1%}**, and F2 **{test['f2']:.3f}**. "
    f"The campaign would contact {tp + fp} customers, including {tp} actual buyers and {fp} nonbuyers, "
    f"while missing {fn} buyers. The threshold therefore trades additional contacts for buyer coverage. "
    f"The mean training-CV AP ({results['best_cv_average_precision']:.3f}) is lower than this holdout AP; "
    "the difference is material and this single random split does not establish stable future performance. "
    "Validate on later customer cohorts before using the model for a live campaign."
))
""")
md("### Limited-budget targeting and descriptive feature importance\nRank by score when a campaign budget fixes how many people can be contacted. The top-20% fraction is a reporting scenario specified in advance, not a test-optimized threshold. Test-set permutation importance below explains the evaluated release; it is not fed back into model or feature selection.")
code("""
importance = pd.read_csv(REPORTS / "feature_importance.csv")
display(importance.round(4))
display(Image(filename=str(REPORTS / "figures/feature_importance.png")))
top_n = test["top_20_percent_customers"]
display(Markdown(
    f"Contacting the top **{top_n}** customers (approximately 20% of this holdout) yields "
    f"historical precision **{test['precision_at_top_20_percent']:.1%}**, compared with "
    f"the holdout purchase prevalence of **{test['purchase_rate']:.1%}**: "
    f"an observed lift of **{test['lift_at_top_20_percent']:.2f}×**. "
    "This is retrospective ranking lift, not proof that contacting those people causes purchases."
))
""")
md("### Inspect the actual MLflow experiment records")
code("""
import mlflow
client = mlflow.tracking.MlflowClient(tracking_uri=(PROJECT / "mlruns").as_uri())
experiment = client.get_experiment_by_name("Tourism Package Prediction - Precontact")
tracked = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
    filter_string=f"tags.mlflow.parentRunId = '{results['mlflow_run_id']}'")
assert len(tracked) == 20
print("Current parent run:", results["mlflow_run_id"])
print("Tracked candidate runs:", len(tracked))
display(tracked[["tags.mlflow.runName", "params.model_name", "metrics.mean_test_average_precision", "status"]]
        .sort_values("metrics.mean_test_average_precision", ascending=False).reset_index(drop=True))
""")
md("""
## 7. Streamlit deployment application · rubric: 11 points

The app loads `model.joblib` beside `app.py`, caches the fitted pipeline, gathers the 13 customer inputs into an ordered dataframe, scores them, and compares the score to the **same saved threshold** used for evaluation. The score is uncalibrated and is labeled accordingly. Missing artifacts and invalid party compositions produce useful messages.

The following cells recreate the actual app and dependency files. Locally launch it with `python -m streamlit run tourism_project/deployment/app.py`. The live cloud connection is a separate step after publishing the repository.
""")
source("tourism_project/deployment/app.py")
source("tourism_project/deployment/requirements.txt")
source(".streamlit/config.toml")
code("""
payload = joblib.load(PROJECT / "deployment/model.joblib")
example = Xtest.iloc[[0]].copy()
score = float(payload["pipeline"].predict_proba(example)[0, 1])
display(example)
print(f"Saved-model example score: {score:.1%}")
print(f"Stored threshold: {payload['threshold']:.1%}")
print("Decision:", "Prioritize for outreach" if score >= payload["threshold"] else "Standard outreach priority")
""")
md("""
## 8. MLOps pipeline with GitHub Actions · rubric: 16 points

The workflow is at repository-root `.github/workflows/pipeline.yml`. The first three jobs implement the template's registration → preparation → training chain. They transfer data through artifacts instead of relying on another runner's local files. A separate publication job gives only that job write permission.

1. **Register:** validate `tourism_project/data/tourism.csv`, write the summary, upload the registered-data artifact.
2. **Prepare:** download registered data, clean and split it, upload checksummed data-splits.
3. **Train and test:** download the splits, tune/log/evaluate, run the tests, upload the tested deployment and MLflow records.
4. **Publish:** only on `main` after success, verify that `main` is still the source revision, commit the fitted model and reports with `[skip ci]`, and push without force. Pull requests cannot publish.

The workflow automates repository model updates. It does not create a Streamlit account or connect a new app. After that one-time connection, the hosted app picks up the model commit.
""")
source("tourism_project/requirements.txt")
source("requirements-dev.txt")
source(".gitignore")
source(".github/workflows/pipeline.yml")
md("### Meaningful verification: contract, isolation, saved inference, and frontend behavior")
for path in sorted((ROOT / "tests").glob("test_*.py")):
    source(path.relative_to(ROOT).as_posix())
code("""
test_run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"], cwd=ROOT,
                          capture_output=True, text=True)
print(test_run.stdout)
if test_run.returncode:
    print(test_run.stderr)
test_run.check_returncode()
_ = (REPORTS / "verification.txt").write_text(test_run.stdout, encoding="utf-8")
""")
code("""
import yaml
workflow = yaml.safe_load((ROOT / ".github/workflows/pipeline.yml").read_text())
assert set(workflow["jobs"]) == {"register-dataset", "data-preparation", "model-training", "publish-model"}
assert workflow["permissions"]["contents"] == "read"
assert workflow["jobs"]["publish-model"]["permissions"]["contents"] == "write"
print("Workflow YAML parsed; four jobs and scoped publication permissions verified locally.")
print("A remote GitHub Actions execution is still required as submission evidence.")

include_roots = [ROOT / ".github", ROOT / ".streamlit", ROOT / "tourism_project", ROOT / "tests"]
for base in include_roots:
    for path in sorted(base.rglob("*")):
        if not path.is_file() or any(part in {"__pycache__", "mlruns", ".pytest_cache"} for part in path.parts):
            continue
        print(path.relative_to(ROOT).as_posix())
""")
md("""
## 9. Output evaluation and public deployment evidence · rubric: 4 points

**Local status:** the outputs above come from actual sequential execution. The notebook includes source code, real MLflow records, held-out results, a serialized serving model, and executed tests. A local app screenshot, if present below, is a preview only.

**Public evidence is pending:** the project has not been pushed to a GitHub repository and no Streamlit Community Cloud deployment or remote workflow run is claimed. The fields below deliberately remain empty. Before submitting the final HTML, add genuine links and screenshots of:

- The GitHub repository folder structure, including the hidden `.github/workflows/pipeline.yml`.
- A successful remote GitHub Actions run with registration, preparation, training/testing, and publication.
- The public Streamlit app showing an actual prediction.

Use **this notebook's directory as the repository root**. After publishing and a successful Actions run, create a Community Cloud app from the repository, branch `main`, entrypoint `tourism_project/deployment/app.py`, and Python 3.12. The adjacent serving `requirements.txt` is already provided. Re-export HTML after inserting the evidence. These external evidence items are required for the corresponding rubric marks; local execution cannot complete them.
""")
code("""
GITHUB_REPOSITORY_URL = ""
GITHUB_ACTIONS_RUN_URL = ""
STREAMLIT_APP_URL = ""
evidence = {
    "GitHub repository": GITHUB_REPOSITORY_URL,
    "Successful GitHub Actions run": GITHUB_ACTIONS_RUN_URL,
    "Public Streamlit app": STREAMLIT_APP_URL,
}
display(pd.DataFrame([{"Evidence": name, "URL": url or "Pending public publication/deployment"}
                      for name, url in evidence.items()]))
preview = REPORTS / "figures/local_app_preview.png"
if preview.is_file():
    display(Markdown("**Local app preview — this is not evidence of public hosting.**"))
    display(Image(filename=str(preview)))
else:
    print("The real frontend was checked with Streamlit AppTest; public screenshots remain pending.")
""")
md("""
## 10. Conclusions and actionable recommendations

The project replaces manual profile handling with a validated, repeatable workflow. The chosen release, training-only threshold, final holdout performance, and campaign tradeoffs are shown below. Retain this evaluated training-set model for serving; refitting it on the holdout would produce a different, unevaluated release.
""")
code("""
top_features = ", ".join(importance.head(3)["feature"])
display(Markdown(
    f"1. **Model:** {results['model_name']} is the training-CV winner among 20 configurations. "
    f"Its holdout AP is {test['average_precision']:.3f} and ROC AUC is {test['roc_auc']:.3f}.\\n\\n"
    f"2. **Recall-focused outreach:** at threshold {results['threshold']:.3f}, recall is {test['recall']:.1%} "
    f"and precision is {test['precision']:.1%}. Expect unnecessary contacts alongside improved buyer coverage; "
    "replace F2 with an economic objective when contact costs and package margins are known.\\n"
    f"3. **Budget scenario:** the top 20% scores give {test['lift_at_top_20_percent']:.2f}× retrospective lift. "
    "Pilot this ranking against randomized or current targeting to estimate incremental conversion.\\n"
    f"4. **Observed drivers:** the largest descriptive AP decreases came from {top_features}. "
    "These are predictive associations, not reasons to infer causality.\\n"
    "5. **Deployment:** publish this prepared repository, execute the Actions workflow, connect Streamlit Cloud, "
    "and add the authentic public evidence before submitting the HTML."
))
""")
md("""
### Limitations and operations after deployment

- **Scope and proxy check:** the data labels general historical package purchases rather than verified uptake of the new wellness offer. Confirm that every predictor is known before contact, especially designation and travel preferences. If designation was assigned from the pitched product, remove it and retrain.
- **Sampling:** source profiles contain repetitions; deduplication prevents exact leakage but does not establish independence of similar customers. There are no trustworthy timestamps or outcome windows for a temporal backtest. The material CV/holdout difference warrants validation on later cohorts.
- **Scores and decisions:** scores have not been calibrated as purchase probabilities. Evaluate calibration, actual conversion, precision, recall, contact cost, revenue, and feedback delay on a future campaign. Do not interpret retrospective ranking lift as causal uplift.
- **Fairness and drift:** gender, age, and income may produce unequal outreach patterns. Audit group coverage and outcome metrics with adequate sample sizes, review whether those features are appropriate, and track missingness, unseen categories, feature distributions, and realized performance. Use a new validation cohort for subsequent changes rather than repeatedly tuning on this test set.
- **Operations:** preserve the dataset hash, source revision, model version, and MLflow records; promote only a tested model, retain previous model commits for rollback, and retrain when new validated outcomes justify it. Automation alone does not establish model quality.

### Rubric coverage

| Criterion | Included evidence | Current status |
|---|---|---|
| Registration · 3 | Master/data folder, schema script, executed summary | Executed locally |
| Preparation · 7 | Cleaning, four CSV splits, checksum manifest, artifact steps | Executed locally; remote transfer pending |
| Modeling/tracking · 13 | 20 tracked candidates, complete tuned pipelines, metrics, saved winner | Executed locally |
| Deployment · 11 | `app.py`, input dataframe, fitted model, serving dependencies, app tests | Local app verified; public hosting pending |
| GitHub Actions · 16 | Complete root YAML, artifact jobs, guarded commit to main | Prepared; remote execution pending |
| Output evaluation · 4 | Genuine repository/workflow/public-app URLs and screenshots | Pending publication |
| Notebook quality · 6 | Sequential executed outputs, observations, source, no error outputs, HTML | Executed and exported |

### References

Project structure: supplied `Project-DESC.txt`, learner template, and the latest Streamlit/GitHub Actions MLOps notebooks in `Examples`. The implementation and analysis in this notebook are tailored to the supplied tourism CSV.

- [scikit-learn: preprocessing and leakage pitfalls](https://scikit-learn.org/1.6/common_pitfalls.html)
- [scikit-learn: GridSearchCV](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [MLflow 3.0.1 tracking API](https://mlflow.org/docs/3.0.1/tracking/tracking-api/)
- [GitHub Actions: storing and sharing workflow data](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
- [Streamlit Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.update({
    "kernelspec": {"display_name": "Python (Tourism MLOps)", "language": "python", "name": "tourism-mlops"},
    "language_info": {"name": "python", "version": "3.12"},
    "title": "Tourism Package Prediction — Visit with Us",
})
destination = ROOT / "Tourism_Package_Prediction_Completed.ipynb"
nbf.write(notebook, destination)
print(f"Created {destination.name}: {len(cells)} cells.")

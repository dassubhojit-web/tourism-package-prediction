"""Build the project notebook from the repository's source files."""
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

**Prepared by:** Subhojit Das

**Objective:** I aim to build a reproducible system to prioritize customers for a Wellness Tourism Package **before a sales conversation**.

In this notebook, I document my data validation and preparation, model training with experiment tracking, evaluated predictions, Streamlit frontend, and GitHub Actions workflow. I can reproduce the local project by running the cells from top to bottom with `tourism.csv` and the required Python dependencies.

**Delivery status:** I have completed local training, saved the model, executed the notebook, and checked the application. I will add the GitHub workflow and public Streamlit deployment evidence after publication.
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

I use Python 3.12 for training and serving, with matching NumPy, pandas, scikit-learn, and joblib versions. I store my MLflow experiment records in a local tracking directory and use Streamlit Community Cloud to host the application.

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
    f"**My data observations:** I found {registration['rows']:,} customer records and {registration['columns']} columns, "
    f"with {sum(registration['missing_values'].values())} missing cells. "
    f"I identified {registration['target_counts']['1']:,} buyers ({registration['purchase_rate']:.2%}). "
    f"I also found {registration['gender_spelling_corrections']} occurrences of the spelling `Fe Male` "
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
    f"**My training observations:** I observed a purchase rate of {passport_rates.get(1, float('nan')):.1%} "
    f"among passport holders and {passport_rates.get(0, float('nan')):.1%} among customers without one. "
    f"I found the highest designation purchase rate for {designation_rates.index[0]} "
    f"({designation_rates.iloc[0]:.1%}). Because group sizes differ, I do not use these rates alone as targeting rules. "
    "I use precision–recall evaluation and a majority-class baseline because purchases are relatively uncommon. "
    "I retained the income extremes and avoided fitting imputation or clipping statistics on the full dataset."
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
    f"**My selected model:** I selected {results['model_name']} with a decision threshold of **{results['threshold']:.3f}**. "
    f"On {len(ytest):,} holdout customers, I obtained AP **{test['average_precision']:.3f}**, ROC AUC **{test['roc_auc']:.3f}**, "
    f"precision **{test['precision']:.1%}**, recall **{test['recall']:.1%}**, and F2 **{test['f2']:.3f}**. "
    f"With this threshold, I would contact {tp + fp} customers, including {tp} actual buyers and {fp} nonbuyers, "
    f"while missing {fn} buyers. I accept additional contacts to improve buyer coverage. "
    f"I found that mean training-CV AP ({results['best_cv_average_precision']:.3f}) is lower than holdout AP. "
    "I consider this difference material, so I would validate performance on later customer cohorts "
    "before relying on the model for a live campaign."
))
""")
md("### Limited-budget targeting and descriptive feature importance\nRank by score when a campaign budget fixes how many people can be contacted. The top-20% fraction is a reporting scenario specified in advance, not a test-optimized threshold. Test-set permutation importance below explains the evaluated release; it is not fed back into model or feature selection.")
code("""
importance = pd.read_csv(REPORTS / "feature_importance.csv")
display(importance.round(4))
display(Image(filename=str(REPORTS / "figures/feature_importance.png")))
top_n = test["top_20_percent_customers"]
display(Markdown(
    f"If I contact the top **{top_n}** customers (approximately 20% of this holdout), I obtain "
    f"historical precision of **{test['precision_at_top_20_percent']:.1%}**, compared with "
    f"the holdout purchase prevalence of **{test['purchase_rate']:.1%}**. "
    f"I observed a lift of **{test['lift_at_top_20_percent']:.2f}×**. "
    "I interpret this as retrospective ranking performance and would use a campaign experiment to assess incremental purchases."
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

I defined the workflow at repository-root `.github/workflows/pipeline.yml`. I organized the first three jobs as registration → preparation → training and passed data between them through artifacts. I gave repository write permission only to the separate model-publication job.

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
    f"1. **My model selection:** I selected {results['model_name']} from 20 training-CV configurations. "
    f"I obtained holdout AP of {test['average_precision']:.3f} and ROC AUC of {test['roc_auc']:.3f}.\\n\\n"
    f"2. **My outreach decision:** I chose threshold {results['threshold']:.3f}, giving recall of {test['recall']:.1%} "
    f"and precision of {test['precision']:.1%}. I accept some unnecessary contacts to improve buyer coverage, "
    "and I would replace F2 with an economic objective when contact costs and package margins are available.\\n"
    f"3. **My budget scenario:** I observed {test['lift_at_top_20_percent']:.2f}× retrospective lift among the top 20% scores. "
    "I would pilot this ranking against randomized or current targeting to estimate incremental conversion.\\n"
    f"4. **My feature observations:** I found the largest descriptive AP decreases for {top_features}. "
    "I interpret these as predictive associations rather than causal effects.\\n"
    "5. **My deployment plan:** I will publish the repository, execute the Actions workflow, connect Streamlit Cloud, "
    "and add the public deployment evidence before submitting my HTML report."
))
""")
md("""
### Limitations and operations after deployment

- **Scope and proxy check:** the data labels general historical package purchases rather than verified uptake of the new wellness offer. Confirm that every predictor is known before contact, especially designation and travel preferences. If designation was assigned from the pitched product, remove it and retrain.
- **Sampling:** source profiles contain repetitions; deduplication prevents exact leakage but does not establish independence of similar customers. There are no trustworthy timestamps or outcome windows for a temporal backtest. The material CV/holdout difference warrants validation on later cohorts.
- **Scores and decisions:** scores have not been calibrated as purchase probabilities. Evaluate calibration, actual conversion, precision, recall, contact cost, revenue, and feedback delay on a future campaign. Do not interpret retrospective ranking lift as causal uplift.
- **Fairness and drift:** gender, age, and income may produce unequal outreach patterns. Audit group coverage and outcome metrics with adequate sample sizes, review whether those features are appropriate, and track missingness, unseen categories, feature distributions, and realized performance. Use a new validation cohort for subsequent changes rather than repeatedly tuning on this test set.
- **Operations:** preserve the dataset hash, source revision, model version, and MLflow records; promote only a tested model, retain previous model commits for rollback, and retrain when new validated outcomes justify it. Automation alone does not establish model quality.

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

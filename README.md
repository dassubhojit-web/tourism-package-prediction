# Tourism Package Prediction

**Prepared by:** Subhojit Das

In this project, I predict whether a customer of **Visit with Us** will purchase a Wellness Tourism Package. I bring together dataset registration, reproducible preparation, model tuning with MLflow tracking, a Streamlit prediction app, and a GitHub Actions pipeline.

I published the project on GitHub, where the four-job workflow completed successfully on **November 24, 2026 at 9:07 a.m. America/Chicago**. I checked the Streamlit Community Cloud app in an anonymous browser session and confirmed that it returned a prediction without requiring sign-in:

- [Public GitHub repository](https://github.com/dassubhojit-web/tourism-package-prediction)
- [Successful GitHub Actions run #1](https://github.com/dassubhojit-web/tourism-package-prediction/actions/runs/45421135704)
- [Streamlit prediction app](https://tourism-package-prediction-ge2zhlrmfksupcejr6skkp.streamlit.app/)

I retained my original local training results in the executed notebook and HTML report and identified the separately trained GitHub release used by the deployed app. I included publication evidence in the notebook's output-evaluation section. When refreshing the report, commit and push the updated files and screenshots after review.

## Project layout

Use this **Tourism Package Prediction directory as the GitHub repository root**. In particular, `.github/workflows/` must be at the repository root.

```text
.
├── .github/workflows/pipeline.yml
├── .streamlit/config.toml
├── requirements-dev.txt
├── README.md
├── Project-DESC.txt
├── Learner_Template_Notebook_AML_and_MLOps_Project.ipynb
├── Tourism_Package_Prediction_Completed.ipynb
├── Tourism_Package_Prediction_Completed.html
├── tourism.csv                         # Original supplied dataset
├── evidence/                           # Genuine repository, workflow, and public-app captures
├── tests/
└── tourism_project/
    ├── requirements.txt
    ├── data/
    │   ├── tourism.csv                 # Registered repository input
    │   └── processed/                  # Generated; passed as workflow artifact
    │       ├── Xtrain.csv
    │       ├── Xtest.csv
    │       ├── ytrain.csv
    │       ├── ytest.csv
    │       └── split_manifest.json
    ├── model_building/
    │   ├── data_register.py
    │   ├── prep.py
    │   └── train.py
    ├── deployment/
    │   ├── app.py
    │   ├── requirements.txt
    │   ├── model.joblib
    │   └── metadata.json
    ├── reports/                        # Validation and model evaluation outputs
    └── mlruns/                         # Local MLflow tracking; workflow artifact
```

I retained the project description and original CSV. I excluded generated train/test splits and local MLflow records from Git, while committing the fitted model, metadata, and reports for serving and review.

## Prediction scope and data handling

My objective is **prediction before contacting a customer**. I therefore use these 13 customer attributes:

`Age`, `CityTier`, `Occupation`, `Gender`, `NumberOfPersonVisiting`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `OwnCar`, `NumberOfChildrenVisiting`, `Designation`, and `MonthlyIncome`.

I removed `CustomerID` and exported row indices because they are identifiers. I also excluded `TypeofContact`, `DurationOfPitch`, `NumberOfFollowups`, `ProductPitched`, and `PitchSatisfactionScore` because they describe a contact or sales interaction. Using these after-contact fields would change my intended prediction use.

During registration, I validate the expected raw schema and binary `ProdTaken` target and record a data summary. In preparation, I apply deterministic text cleanup, including `Fe Male` → `Female` and `Free Lancer` → `Freelancer`. I keep `Single` and `Unmarried` as separate source categories and retain missing predictor values for imputers fitted inside the training pipeline.

I removed 194 duplicate rows with identical pre-contact predictors and target, reducing the 4,128 raw rows to 3,934 unique labeled profiles. I stop preparation for review if identical predictors have conflicting targets. My stratified 80/20 split with `random_state=42` produces 3,147 training rows and 787 test rows. I verified zero profile overlap and unique training profiles. After removing duplicates, I use three-fold `StratifiedKFold` and fit learned imputation and categorical encoding separately within each training fold.

I compared 20 hyperparameter configurations: four Decision Trees, eight Random Forests, and eight Gradient Boosting models. I selected the configuration with the highest training cross-validation average precision. I then selected its decision threshold by maximizing F2 on training out-of-fold predictions to give recall more weight than precision. Because I reused the three training folds after model selection, I treat these out-of-fold results as selection diagnostics rather than an unbiased nested-cross-validation estimate. I reserved the test holdout for final evaluation and excluded it from model and threshold selection.

I saved the complete fitted preprocessing/model pipeline, decision threshold, and expected feature columns in one model payload. In the app, I build a dataframe with this schema and apply the stored threshold. I interpret the purchase score as an estimate for prioritizing outreach; the historical data does not establish causal marketing impact or guarantee future conversion rates.

## Evaluation results and release provenance

I selected Random Forest in both my original local run and the GitHub Actions run, with a training-selected threshold of **0.185**. These runs produced distinct trained artifacts, so I retained and labeled my local outputs separately from the release serving the public app.

| Measure | Original local execution | GitHub-trained deployed release |
| --- | ---: | ---: |
| Training cross-validation average precision | 0.6470 | 0.6482 |
| Test average precision | 0.7958 | 0.8009 |
| Test ROC AUC | 0.9302 | 0.9302 |
| Test purchase recall | 0.9216 | 0.9281 |
| Test purchase precision | 0.5486 | 0.5680 |
| Test F1 | 0.6878 | 0.7047 |
| Test F2 | 0.8113 | 0.8237 |
| Test accuracy | 0.8374 | 0.8488 |
| Precision among the highest-scored 20% | 0.7215 | 0.7278 |
| Lift among the highest-scored 20% | 3.711× | 3.744× |

I recorded the deployed results in the repository's [metrics.json](https://github.com/dassubhojit-web/tourism-package-prediction/blob/main/tourism_project/reports/metrics.json), with MLflow parent run `9c74a6d5940f4b918cff3a871e5898a7`. My GitHub runner used Python 3.12.16 with scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.2.6, and joblib 1.5.1. At the saved threshold, I obtained the holdout confusion matrix `[[526, 108], [11, 142]]`: 142 buyers identified, 11 buyers missed, and 108 nonbuyers selected for outreach.

When I selected the highest-scored 20%, I targeted 158 of the 787 test customers. I interpret the observed lift as retrospective ranking performance. My threshold-based predictions prioritize recall and consequently include false positives. I saved the detailed metrics, confusion matrix, tuning results, and predictions in `tourism_project/reports/` for each trained release. Synchronize with the workflow's model-publication commit before inspecting the deployed release locally.

I recognize that the historical `ProdTaken` label records package uptake and does not confirm a future purchase of the new Wellness Tourism Package. Before production use, I would perform temporal validation, drift monitoring, fairness assessment, and probability-calibration checks. I also observed that `Designation` perfectly maps to `ProductPitched`. I would confirm that designation is independently known before outreach and was not assigned during sales segmentation before relying on it operationally.

## Run locally

Use **Python 3.12** for both training and deployment. The following commands run the full training workflow and create a new local model and reports. Use a separate working copy when preserving the exact GitHub-trained serving artifact. From that copy's project directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m tourism_project.model_building.data_register
python -m tourism_project.model_building.prep
python -m tourism_project.model_building.train
python -m pytest -q
python -m streamlit run tourism_project/deployment/app.py
```

I pinned matching NumPy, pandas, scikit-learn, and joblib versions for training and serving so the app can load the serialized fitted pipeline consistently. I excluded MLflow and notebook dependencies from the serving environment.

I write evaluation reports to `tourism_project/reports/` and MLflow records to `tourism_project/mlruns/`. I track each tuning candidate's parameters and validation score, then save the final parameters, evaluation metrics, and artifacts with the best run. My local file tracking does not require an MLflow server, ngrok account, or access token.

To view the experiments in the optional MLflow UI, install the full package in the development environment and point it at the same store:

```bash
python -m pip install mlflow==3.0.1
mlflow ui --backend-store-uri tourism_project/mlruns --port 5000
```

Open `http://127.0.0.1:5000`. Installing the full UI package is optional; the automated training pipeline uses `mlflow-skinny` and retains all experiment records as an artifact.

## Notebook and HTML submission

My worked analysis, source code, pipeline execution, results, and observations are in `Tourism_Package_Prediction_Completed.ipynb`. I exported the notebook as the self-contained `Tourism_Package_Prediction_Completed.html` for submission.

To refresh the final submission, run only its report/evidence cells in Jupyter, save the notebook with the original training outputs retained, and export it with embedded images:

```bash
jupyter nbconvert --to html --embed-images Tourism_Package_Prediction_Completed.ipynb
```

The report-only update preserves the local execution outputs and loads the deployed release metrics separately. Run `python tools/update_publication_evidence.py` with the `tourism-mlops` Jupyter kernel installed to refresh those evidence and conclusion cells without training. Executing the full notebook reruns training and overwrites its local model/report paths. `tools/build_notebook.py` creates the starting notebook from source; it does not merge manually added publication evidence. Use the saved completed notebook for the final evidence update and HTML export. Submit the updated HTML file, as required by the assignment.

From a fresh clone, first run `python -m tourism_project.model_building.prep` to recreate the ignored splits used by verification; this does not retrain or replace the serving model. Register the notebook kernel with `python -m ipykernel install --user --name tourism-mlops --display-name "Python (Tourism MLOps)"` if needed.

## GitHub Actions

I configured the workflow to run on pushes to `main`, pull requests targeting `main`, and manual dispatch. It has three machine-learning jobs followed by a separate publication job:

1. **Register and validate dataset:** validate the repository CSV and upload the registered CSV and data summary.
2. **Clean data and create holdout split:** download the registered input, prepare the data, and upload train/test splits and the split manifest.
3. **Tune, track, evaluate, and test:** download the prepared splits, train with MLflow tracking, evaluate, run the tests, and upload the tested model and reports. Tracking records are retained separately.
4. **Commit tested model to main:** runs only after successful training/tests on `main`, outside pull requests. It downloads the tested artifact and commits the model, metadata, and reports with `[skip ci]`.

I assigned read-only repository tokens to the first three jobs, including pull requests from the same repository. Only publication receives `contents: write`. I use concurrency controls to prevent overlapping runs for the same ref and check that `main` still matches the revision used for training before publication. If newer work exists, I retain the old model as a workflow artifact and skip publication. I do not force-push over newer commits.

For a new repository, upload this directory including its hidden `.github` directory. Choose `main` as the default branch, enable GitHub Actions, and ensure repository or organization policies permit the publication job to use its requested write permission. Protected-branch rules may require an approved alternative publication process. The workflow needs no personal access token or cloud secret.

I confirmed that the linked run completed all four jobs successfully. Its **Actions** page contains the tested model and reports in `trained-deployment` and tracking records in `mlflow-experiments`, subject to their retention periods. My workflow trains and commits the serving files; I connected the repository to Streamlit Community Cloud separately.

## Deploy on Streamlit Community Cloud

I deployed the prediction app on **Streamlit Community Cloud**, using this repository, branch `main`, and entrypoint `tourism_project/deployment/app.py`. To reproduce the deployment in another account:

1. Publish this directory as a GitHub repository and complete a successful workflow run.
2. Confirm `tourism_project/deployment/model.joblib` and `metadata.json` exist on `main`.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with access to that repository and create an app.
4. Select the repository, branch `main`, and main file `tourism_project/deployment/app.py`.
5. Select **Python 3.12** in the deployment settings and deploy. The app's adjacent `requirements.txt` contains its serving dependencies.
6. Open the public app, submit a sample customer, and record the actual public URL and screenshot in the completed notebook.

After the one-time connection, new serving-model commits are picked up by the connected deployment. GitHub Actions does not create the Streamlit account or the app connection.

## Publication evidence for the submission

Use the actual URLs above and the following screenshot files in the completed notebook's output-evaluation section:

| File | Evidence |
| --- | --- |
| `evidence/github_structure.png` | Public repository structure, including `.github/workflows/pipeline.yml` and `tourism_project/` |
| `evidence/workflow_success.png` | The successful remote run showing all four jobs |
| `evidence/streamlit_prediction.png` | The public Streamlit app displaying an actual prediction |

I retained `tourism_project/reports/figures/local_app_preview.png` as a local preview and included the public-app capture and canonical URL as deployment evidence. When updating the report, review and push the notebook, HTML, README, and evidence files with a `[skip ci]` commit message so a documentation update does not retrain or replace the evaluated serving model.

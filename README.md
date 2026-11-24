# Tourism Package Prediction

An end-to-end MLOps project for **Visit with Us** that predicts whether a customer will purchase a Wellness Tourism Package. It follows the supplied learner template and examples: dataset registration, reproducible preparation, model tuning with MLflow tracking, a Streamlit prediction app, and a GitHub Actions pipeline.

The project is published on GitHub, and its four-job workflow completed successfully on **November 24, 2026 at 9:07 a.m. America/Chicago**. The Streamlit Community Cloud app was verified in an anonymous browser session and returned a prediction without requiring sign-in:

- [Public GitHub repository](https://github.com/dassubhojit-web/tourism-package-prediction)
- [Successful GitHub Actions run #1](https://github.com/dassubhojit-web/tourism-package-prediction/actions/runs/45421135704)
- [Streamlit prediction app](https://tourism-package-prediction-ge2zhlrmfksupcejr6skkp.streamlit.app/)

The executed notebook and HTML report preserve the original local training results and identify the separately trained GitHub release used by the deployed app. Publication evidence belongs in the notebook's output-evaluation section. Updated local report files and screenshots must be committed and pushed when ready for publication.

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

The supplied description, original CSV, and learner template are retained. Generated train/test splits and local MLflow records are excluded from Git; the fitted model, metadata, and reports are intentionally committed for serving and review.

## Prediction scope and data handling

The business objective is **prediction before contacting a customer**. The model therefore uses these 13 customer attributes:

`Age`, `CityTier`, `Occupation`, `Gender`, `NumberOfPersonVisiting`, `PreferredPropertyStar`, `MaritalStatus`, `NumberOfTrips`, `Passport`, `OwnCar`, `NumberOfChildrenVisiting`, `Designation`, and `MonthlyIncome`.

`CustomerID` and exported row indices are identifiers, so they are removed. `TypeofContact`, `DurationOfPitch`, `NumberOfFollowups`, `ProductPitched`, and `PitchSatisfactionScore` describe a contact or sales interaction and are excluded from the predictive features. Using these after-contact fields would change the intended business use.

Registration validates the expected raw schema and binary `ProdTaken` target and records a data summary. Preparation applies deterministic text cleanup, including `Fe Male` → `Female` and `Free Lancer` → `Freelancer`. `Single` and `Unmarried` remain separate source categories. Missing predictor values are retained for imputers fitted inside the training pipeline.

Preparation removes 194 duplicate rows with identical pre-contact predictors and target, reducing the 4,128 raw rows to 3,934 unique labeled profiles. Identical predictors with conflicting targets cause preparation to stop for manual review. A stratified 80/20 split with `random_state=42` produces 3,147 training rows and 787 test rows; explicit checks verify zero profile overlap and unique training profiles. Because duplicates have already been removed, ordinary three-fold `StratifiedKFold` avoids duplicate-profile contamination. Learned imputation and categorical encoding are fitted separately within each training fold.

Model selection compares 20 hyperparameter configurations: four Decision Trees, eight Random Forests, and eight Gradient Boosting models. The winner maximizes training cross-validation average precision. Its decision threshold maximizes F2 on training out-of-fold predictions, which weights recall more heavily than precision. The threshold calculation reuses the three training folds after model selection; those out-of-fold results are selection diagnostics, not an unbiased nested-cross-validation performance estimate. The test holdout is used for the final evaluation and is excluded from model and threshold selection.

The saved model payload contains the complete fitted preprocessing/model pipeline, the decision threshold, and the expected feature columns. The app builds a dataframe with this same schema and uses the stored threshold. A purchase score is a model estimate for prioritizing outreach; the historical dataset does not establish causal marketing impact or guarantee future conversion rates.

## Evaluation results and release provenance

Both the original local execution and the GitHub Actions execution selected Random Forest, with the training-selected threshold **0.185**. They are distinct trained artifacts; the notebook's original local outputs are retained and labeled separately from the release serving the public app.

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

The deployed release results are recorded in the repository's [metrics.json](https://github.com/dassubhojit-web/tourism-package-prediction/blob/main/tourism_project/reports/metrics.json), with MLflow parent run `9c74a6d5940f4b918cff3a871e5898a7`. The GitHub runner used Python 3.12.16 with scikit-learn 1.6.1, pandas 2.2.3, NumPy 2.2.6, and joblib 1.5.1. At the saved threshold, its holdout confusion matrix is `[[526, 108], [11, 142]]`: 142 buyers identified, 11 buyers missed, and 108 nonbuyers selected for outreach.

The highest-scored 20% corresponds to 158 of the 787 test customers. These ranking results describe the historical holdout; the threshold-based predictions prioritize recall and consequently include false positives. Detailed metrics, the confusion matrix, tuning results, and predictions are saved in `tourism_project/reports/` for the corresponding trained release. Synchronize with the workflow's model-publication commit before inspecting the deployed release locally.

The historical `ProdTaken` label records package uptake; it does not confirm a future purchase of this newly introduced Wellness Tourism Package. Production use requires temporal validation, drift monitoring, fairness assessment, and probability-calibration checks. In this dataset, `Designation` perfectly maps to `ProductPitched`; confirm that designation is independently known before outreach and was not assigned during sales segmentation before relying on it operationally.

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

The dependencies for the training pipeline and app pin matching NumPy, pandas, scikit-learn, and joblib versions. This matters because the app loads a serialized fitted scikit-learn pipeline. The serving environment excludes MLflow and notebook dependencies.

Training writes evaluation reports to `tourism_project/reports/`, and MLflow records to `tourism_project/mlruns/`. Each tuning candidate has a tracked parameter set and validation score; the best run includes the final parameters, evaluation metrics, and saved artifacts. Local file tracking works without an MLflow server, ngrok account, or access token.

To view the experiments in the optional MLflow UI, install the full package in the development environment and point it at the same store:

```bash
python -m pip install mlflow==3.0.1
mlflow ui --backend-store-uri tourism_project/mlruns --port 5000
```

Open `http://127.0.0.1:5000`. Installing the full UI package is optional; the automated training pipeline uses `mlflow-skinny` and retains all experiment records as an artifact.

## Notebook and HTML submission

Open `Tourism_Package_Prediction_Completed.ipynb` for the worked analysis, source code, pipeline execution, results, and observations. `Tourism_Package_Prediction_Completed.html` is its self-contained exported submission. The original learner template is unchanged.

To refresh the final submission, run only its report/evidence cells in Jupyter, save the notebook with the original training outputs retained, and export it with embedded images:

```bash
jupyter nbconvert --to html --embed-images Tourism_Package_Prediction_Completed.ipynb
```

The report-only update preserves the local execution outputs and loads the deployed release metrics separately. Run `python tools/update_publication_evidence.py` with the `tourism-mlops` Jupyter kernel installed to refresh those evidence and conclusion cells without training. Executing the full notebook reruns training and overwrites its local model/report paths. `tools/build_notebook.py` creates the starting notebook from source; it does not merge manually added publication evidence. Use the saved completed notebook for the final evidence update and HTML export. Submit the updated HTML file, as required by the assignment.

From a fresh clone, first run `python -m tourism_project.model_building.prep` to recreate the ignored splits used by verification; this does not retrain or replace the serving model. Register the notebook kernel with `python -m ipykernel install --user --name tourism-mlops --display-name "Python (Tourism MLOps)"` if needed.

## GitHub Actions

The workflow runs on pushes to `main`, pull requests targeting `main`, and manual dispatch. It has three machine-learning jobs followed by a separate publication job:

1. **Register and validate dataset:** validate the repository CSV and upload the registered CSV and data summary.
2. **Clean data and create holdout split:** download the registered input, prepare the data, and upload train/test splits and the split manifest.
3. **Tune, track, evaluate, and test:** download the prepared splits, train with MLflow tracking, evaluate, run the tests, and upload the tested model and reports. Tracking records are retained separately.
4. **Commit tested model to main:** runs only after successful training/tests on `main`, outside pull requests. It downloads the tested artifact and commits the model, metadata, and reports with `[skip ci]`.

The first three jobs have read-only repository tokens, including on pull requests from the same repository. Only publication receives `contents: write`. Concurrency prevents overlapping runs for the same ref; publication checks that `main` still matches the source revision used for training. If newer work exists, the old model stays available as a workflow artifact and publication is skipped. The workflow never force-pushes over newer commits.

For a new repository, upload this directory including its hidden `.github` directory. Choose `main` as the default branch, enable GitHub Actions, and ensure repository or organization policies permit the publication job to use its requested write permission. Protected-branch rules may require an approved alternative publication process. The workflow needs no personal access token or cloud secret.

The linked run completed all four jobs successfully. In its **Actions** page, the `trained-deployment` artifact contains the tested model/reports and the `mlflow-experiments` artifact contains tracking records, subject to the configured retention periods. The workflow trains and commits the serving files; connecting the repository to Streamlit Community Cloud is a separate one-time deployment step.

## Deploy on Streamlit Community Cloud

The supplied template and deployment rubric specify **Streamlit Community Cloud**. The description also contains an older Hugging Face submission reference; this implementation follows the template and Streamlit rubric.

The linked public app uses this repository, branch `main`, and entrypoint `tourism_project/deployment/app.py`. To reproduce the deployment in another account:

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

The original `tourism_project/reports/figures/local_app_preview.png` remains a local preview. Use the public-app capture and its canonical URL as deployment evidence. Review the completed report, then commit and push its notebook, HTML, README, and evidence files with a `[skip ci]` commit message so a documentation update does not retrain or replace the evaluated serving model.

## Implementation references

- Supplied project description, learner template, and the MLOps/Streamlit examples in the adjacent `Examples` folder informed the project structure; the solution is implemented for the tourism dataset.
- [GitHub: passing data between workflow jobs](https://docs.github.com/en/actions/tutorials/store-and-share-data) explains the upload/download artifact mechanism used here.
- [GitHub: controlling workflow concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency) documents the run-serialization mechanism.
- [Streamlit: Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app) describes app preparation and deployment.

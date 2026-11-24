"""Apply Subhojit Das's report voice while preserving model results and figures."""
import ast
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Tourism_Package_Prediction_Completed.ipynb"


def personalize_notebook(notebook):
    paragraphs = {
        0: """
            # Tourism Package Prediction
            ### Visit with Us · Advanced Machine Learning and MLOps

            **Prepared by:** Subhojit Das

            My objective is to predict which customers are likely to purchase a Wellness Tourism Package **before a sales conversation**. I developed a complete workflow covering data validation, preparation, model tuning, experiment tracking, evaluation, deployment, and automated model updates.

            **My published project:** [GitHub repository](https://github.com/dassubhojit-web/tourism-package-prediction) · [Successful GitHub Actions workflow](https://github.com/dassubhojit-web/tourism-package-prediction/actions/runs/45421135704) · [Public Streamlit app](https://tourism-package-prediction-ge2zhlrmfksupcejr6skkp.streamlit.app/)

            **My deployed model:** Random Forest, with a decision threshold of **0.185**, test average precision of **0.8009**, ROC AUC of **0.9302**, buyer recall of **92.81%**, and precision of **56.80%**.

            I first trained and evaluated the model locally, then automated training through GitHub Actions. I retained my local execution outputs and report the GitHub-trained model separately in Section 9 because it is the model used by my public app. I refreshed my deployment evidence and final conclusions after publication without retraining the serving model.

            To reproduce my training, run the notebook in a separate working copy. To refresh the publication evidence while keeping the saved model, run `tools/update_publication_evidence.py` and export the notebook to HTML.
        """,
        1: """
            ## 1. Business context and success criteria

            I developed this classifier to help Visit with Us prioritize customers for its Wellness Tourism Package. My aim is to make outreach more consistent and help a limited marketing budget reach customers with stronger purchase interest. I use `ProdTaken=1` for historical buyers and `ProdTaken=0` for nonbuyers.

            I selected **average precision (AP)** as my primary model-selection metric because most customers did not purchase a package. Accuracy alone could favor a model that predicts nonpurchase for everyone. I also evaluated ROC AUC, precision, recall, F1, F2, balanced accuracy, and campaign lift.

            I chose the outreach threshold by maximizing **F2 on training out-of-fold predictions**, giving greater weight to recall. I used this provisional objective because contact costs and package margins were unavailable. I kept the final test set out of model and threshold selection.

            I interpret historical purchase patterns as predictive associations. I would use a future campaign pilot to assess uptake of the new wellness offer and the incremental effect of outreach.
        """,
        2: None,
        3: """
            ## 3. Environment and project configuration

            I used Python 3.12 for training and deployment, with matching NumPy, pandas, scikit-learn, and joblib versions. I recorded my MLflow experiments in a local tracking directory and deployed my prediction app on Streamlit Community Cloud.

            To reproduce my completed report in Colab, clone or upload the complete project, including `evidence/`, and work from the repository root. Use the pinned dependencies and Python 3.12. Restart the runtime after installation if it has already imported different package versions. My report-only update reads the saved release snapshots without rerunning training.
        """,
        7: """
            ### Shared schema and deterministic cleanup

            I defined a shared schema, removed identifiers and interaction-derived fields, and corrected two spelling variants. I retained `Single` and `Unmarried` as separate categories because their equivalence was not confirmed. I fitted learned imputation within the training pipeline.
        """,
        10: """
            ## 4. Data registration · rubric: 3 points

            I validated the repository CSV before training. I checked the expected columns, unique customer IDs, binary target, numeric values, and basic domain rules. I also recorded the dataset hash, purchase prevalence, missing values, and repeated profiles.
        """,
        14: """
            ## 5. Data preparation · rubric: 7 points

            I removed duplicate normalized profiles **before** creating my stratified 80/20 split. These records had different IDs but identical available predictors and outcomes, which could inflate performance if they appeared in both training and testing. I treat this as an evaluation of unique recorded profiles; different people could still share the same recorded attributes. I stop preparation for review if identical predictors have conflicting labels.

            I saved `Xtrain.csv`, `Xtest.csv`, `ytrain.csv`, and `ytest.csv`, together with a checksum manifest. I deferred imputation and categorical encoding until cross-validation. I retained the valid income and trip extremes because I had no defensible correction rule, and I did not scale the predictors because my candidate models were tree-based.
        """,
        17: """
            ### Exploratory analysis on the training set

            I explored the training set while keeping the final test set out of feature analysis and model selection. I interpret the associations below as historical patterns rather than causal effects.
        """,
        20: """
            ## 6. Model training and registration with experimentation tracking · rubric: 13 points

            I compared **Decision Tree (4 configurations), Random Forest (8), and Gradient Boosting (8)** across three stratified folds. This gave me 20 configurations and 60 validation fits before refitting. I placed preprocessing inside `GridSearchCV` so each fold fitted its own imputers and categorical encoder. I configured the encoder to handle unseen categories.

            I selected the configuration with the highest mean training-fold AP. I then generated out-of-fold training scores for that configuration and selected the F2 threshold. Because I reused the training folds after model selection, I treat those OOF results as selection diagnostics rather than an unbiased nested-CV estimate. I used the untouched holdout for my final assessment and a prevalence-based dummy classifier as my baseline.

            I logged a parent MLflow run and a nested run for each configuration, including estimator parameters, tuned values, fold metrics, variability, selected threshold, versions, dataset hash, and model artifacts. I saved the complete fitted pipeline with its feature order and decision threshold.

            **My local development run:** I retain its executed outputs below. I report the separately trained GitHub release used by my public app in Section 9.
        """,
        23: """
            ### My cross-validation comparison and holdout results
        """,
        26: """
            ### My limited-budget targeting analysis and feature importance

            I evaluated the highest-scored 20% as a predefined campaign-budget scenario. I did not select that fraction by optimizing test performance. I used test-set permutation importance to explain the evaluated model and did not feed those results back into feature selection or tuning.
        """,
        28: """
            ### My recorded MLflow experiments
        """,
        30: """
            ## 7. Streamlit deployment application · rubric: 11 points

            I created a Streamlit app that loads the cached `model.joblib` beside `app.py`, collects the 13 customer inputs into an ordered dataframe, and compares the model score with the saved decision threshold. I label the score as uncalibrated and handle missing model files and invalid party compositions with clear messages.

            I deployed the app on Streamlit Community Cloud. I include its source and dependencies below, and provide my public app link and prediction screenshot in Section 9. For local use, I launch it with `python -m streamlit run tourism_project/deployment/app.py`.
        """,
        35: """
            ## 8. MLOps pipeline with GitHub Actions · rubric: 16 points

            I placed my workflow at repository-root `.github/workflows/pipeline.yml`. I use registration, preparation, and training jobs to pass data through workflow artifacts. I give write permission only to a separate publication job.

            1. **Registration:** I validate the CSV, record its summary, and upload the registered dataset.
            2. **Preparation:** I download the registered input, clean and split it, and upload the checksummed splits.
            3. **Training and testing:** I download those splits, tune and track the model, evaluate it, run the checks, and upload the tested model and experiment records.
            4. **Publication:** I verify that `main` still matches the training revision before committing the model and reports with `[skip ci]`. I prevent pull requests from publishing and avoid force-pushing over newer work.

            I connected the repository to Streamlit Community Cloud so the app can load the model committed by my workflow.
        """,
        40: """
            ### My verification checks

            I checked the data contract, separation of training and testing profiles, saved-model inference, and frontend behavior.
        """,
        45: """
            ## 9. Output evaluation and public deployment evidence · rubric: 4 points

            I published my repository on GitHub and deployed my app on Streamlit Community Cloud. My first GitHub Actions run completed successfully on **November 24, 2026 at 9:07 a.m. America/Chicago**, with all four jobs passing. I verified the artifact handoffs, training results, tests, and automatic model commit to `main`.

            **My training source:** `62715eb35bf9746673cb0723318dd63591d5576b`.
            **My published model commit:** [`8a31082862dcd4c81f7ba3ad6baeede5bd1eb8fe`](https://github.com/dassubhojit-web/tourism-package-prediction/commit/8a31082862dcd4c81f7ba3ad6baeede5bd1eb8fe).
            **My MLflow parent run:** `9c74a6d5940f4b918cff3a871e5898a7`.

            My workflow saved `registered-data`, `data-splits`, `trained-deployment`, and `mlflow-experiments`. I recorded the artifact identifiers, digests, and job statuses in `evidence/github_verification.json`. The source, serving model, and reports are committed to my repository; workflow downloads remain subject to GitHub retention limits.

            I checked my public app without signing in and submitted the form's sample profile. I recorded the page URL and prediction in `evidence/browser_verification.json` and included the screenshots below.

            I show my local and GitHub-trained results separately because they are different fitted models. I use the deployed-release column when describing my live app and did not perform further tuning on the test results.
        """,
        47: """
            ## 10. My conclusions and recommendations

            I completed the GitHub workflow, automatically committed the tested model to `main`, and deployed it through my public Streamlit app. I base my final conclusions on that deployed release while retaining my local training outputs earlier in this report.
        """,
    }
    for index, value in paragraphs.items():
        if value is not None:
            notebook.cells[index].source = dedent(value).strip()
    dictionary = notebook.cells[2].source
    dictionary = dictionary.split("\n\nTravel preferences,")[0].split("\n\nI assume travel preferences")[0].rstrip()
    notebook.cells[2].source = dictionary + dedent("""


        I assume travel preferences, income, and designation are available in an existing customer profile before outreach. I therefore use **13 pre-contact fields**. I found that `Designation` maps perfectly to `ProductPitched` in this dataset, so I would confirm that designation was collected independently before relying on it in a future campaign. If that assumption fails, I would remove it and retrain.
    """)
    notebook.cells[13].source = dedent("""
        display(Markdown(
            f"**My data observations:** I reviewed {registration['rows']:,} rows and {registration['columns']} columns. "
            f"I found {sum(registration['missing_values'].values())} missing cells and "
            f"{registration['target_counts']['1']:,} buyers ({registration['purchase_rate']:.2%}). "
            f"I identified {registration['gender_spelling_corrections']} occurrences of `Fe Male` "
            f"and {registration['duplicate_profiles_after_normalization']} repeated pre-contact profiles after normalization."
        ))
        display(raw.select_dtypes(include="object").nunique().rename("distinct categories").to_frame())
        designation_pairs = raw[["Designation", "ProductPitched"]].drop_duplicates().sort_values("Designation")
        display(designation_pairs)
    """).strip()
    notebook.cells[19].source = dedent("""
        passport_rates = eda.groupby("Passport")[TARGET].mean()
        designation_rates = eda.groupby("Designation")[TARGET].mean().sort_values(ascending=False)
        display(Markdown(
            f"**My training observations:** I observed a purchase rate of {passport_rates.get(1, float('nan')):.1%} "
            f"among passport holders, compared with {passport_rates.get(0, float('nan')):.1%} among customers without one. "
            f"I found the highest designation purchase rate in {designation_rates.index[0]} "
            f"({designation_rates.iloc[0]:.1%}). I would not use these rates alone as targeting rules because the group sizes differ. "
            "I selected precision–recall evaluation and a majority-class baseline because the purchase rate is low. "
            "I retained the income extremes and avoided fitting imputation or clipping statistics on the full dataset."
        ))
    """).strip()
    notebook.cells[25].source = dedent("""
        test = results["test_metrics"]
        tn, fp, fn, tp = np.asarray(test["confusion_matrix"]).ravel()
        display(Markdown(
            f"**My local model assessment:** I selected {results['model_name']} with a threshold of "
            f"**{results['threshold']:.3f}**. On {len(ytest):,} holdout customers, I obtained AP "
            f"**{test['average_precision']:.3f}**, ROC AUC **{test['roc_auc']:.3f}**, precision "
            f"**{test['precision']:.1%}**, recall **{test['recall']:.1%}**, and F2 **{test['f2']:.3f}**. "
            f"At this threshold, I would contact {tp + fp} customers, including {tp} buyers and {fp} nonbuyers, "
            f"while missing {fn} buyers. I accept the additional contacts as a tradeoff for greater buyer coverage. "
            f"I also note that my training-CV AP ({results['best_cv_average_precision']:.3f}) is lower than my holdout AP. "
            "I regard this difference as a reason to validate on later customer cohorts before drawing conclusions about future performance."
        ))
    """).strip()
    original = notebook.cells[27].source
    before = original[:original.index("display(Markdown(")]
    notebook.cells[27].source = before + dedent("""
        display(Markdown(
            f"**My campaign-budget observation:** When I rank the holdout customers and select the top **{top_n}** "
            f"(approximately 20%), I obtain historical precision of **{test['precision_at_top_20_percent']:.1%}**, "
            f"compared with purchase prevalence of **{test['purchase_rate']:.1%}**. "
            f"I calculate retrospective lift of **{test['lift_at_top_20_percent']:.2f}×**. "
            "I interpret this as ranking performance and would use a campaign experiment to assess whether outreach causes additional purchases."
        ))
    """).strip()
    notebook.cells[48].source = dedent("""
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown

        report_root = Path.cwd().resolve()
        release = json.loads((report_root / "evidence/deployed_release_metrics.json").read_text())
        test = release["test_metrics"]
        importance = pd.read_csv(report_root / "tourism_project/reports/feature_importance.csv")
        top_features = ", ".join(importance.head(3)["feature"])
        tn, fp, fn, tp = np.asarray(test["confusion_matrix"]).ravel()
        lines = [
            f"1. **My selected model:** I selected {release['model_name']} from 20 training-CV configurations. "
            f"My deployed release achieved holdout AP **{test['average_precision']:.3f}**, ROC AUC "
            f"**{test['roc_auc']:.3f}**, and accuracy **{test['accuracy']:.1%}**.",
            f"2. **My outreach decision:** I used threshold **{release['threshold']:.3f}**, obtaining recall "
            f"**{test['recall']:.1%}** and precision **{test['precision']:.1%}**. "
            f"I identified {tp} buyers, missed {fn}, and selected {fp} nonbuyers for contact. "
            "I would replace F2 with an economic objective once contact costs and margins are available.",
            f"3. **My budget scenario:** Selecting the top {test['top_20_percent_customers']} customers gave me "
            f"retrospective precision of **{test['precision_at_top_20_percent']:.1%}** and lift of "
            f"**{test['lift_at_top_20_percent']:.2f}×**. I recommend a pilot against randomized or current targeting "
            "to estimate incremental conversion.",
            f"4. **My interpretation of predictors:** I observed the largest descriptive AP decreases for "
            f"{top_features}. I interpret these as predictive associations rather than causal explanations.",
            "5. **My deployment outcome:** I published the repository, completed the four-job workflow, "
            "automated the model commit, and deployed the Streamlit app. I included the public links and "
            "screenshots in Section 9 and retained both my local execution history and deployed-release results.",
        ]
        display(Markdown("\\n\\n".join(lines)))
    """).strip()
    notebook.cells[49].source = dedent("""
        ### My limitations and next steps

        - **Prediction scope:** I used historical package purchases rather than confirmed uptake of the new wellness offer. I would confirm that every predictor is available before contact, especially designation and travel preferences, and remove any field that fails this check.
        - **Sampling:** I removed repeated profiles to prevent exact overlap, but I cannot establish that similar customers are independent. I do not have reliable timestamps for a temporal backtest. I would evaluate the model on later cohorts because the CV and holdout results differ materially.
        - **Scores and decisions:** I have not calibrated the scores as purchase probabilities. I would monitor calibration, conversion, precision, recall, contact costs, revenue, and feedback delay. I would assess causal marketing impact through an experiment.
        - **Fairness and drift:** I would audit outreach coverage and outcomes across demographic groups, review whether sensitive predictors are appropriate, and monitor missingness, unseen categories, and feature distributions. I would use a new validation cohort for subsequent model changes.
        - **Operations:** I would retain dataset hashes, source revisions, model versions, and experiment records, promote only tested models, and keep previous model commits available for rollback. I would retrain when new validated outcomes justify an update.
    """).strip()
    notebook.metadata["authors"] = [{"name": "Subhojit Das"}]
    return notebook


def refresh_observations(notebook):
    """Execute narrative expressions and retain existing numeric tables and PNGs."""
    setup = dedent("""
        from pathlib import Path
        import json
        import numpy as np
        import pandas as pd
        from IPython.display import display, Markdown
        ROOT = Path.cwd().resolve()
        REPORTS = ROOT / "tourism_project/reports"
        TARGET = "ProdTaken"
        registration = json.loads((REPORTS / "data_summary.json").read_text())
        raw = pd.read_csv(ROOT / "tourism_project/data/tourism.csv")
        Xtrain = pd.read_csv(ROOT / "tourism_project/data/processed/Xtrain.csv")
        ytrain = pd.read_csv(ROOT / "tourism_project/data/processed/ytrain.csv")[TARGET]
        ytest = pd.read_csv(ROOT / "tourism_project/data/processed/ytest.csv")[TARGET]
        eda = Xtrain.assign(ProdTaken=ytrain)
        results = json.loads((ROOT / "evidence/local_development_metrics.json").read_text())
        test = results["test_metrics"]
        top_n = test["top_20_percent_customers"]
    """).strip()
    indexes = [13, 19, 25, 27, 48]
    expressions = []
    for index in indexes:
        source = notebook.cells[index].source
        if index in [13, 27]:
            tree = ast.parse(source)
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "display"
                     and node.args and isinstance(node.args[0], ast.Call)
                     and isinstance(node.args[0].func, ast.Name)
                     and node.args[0].func.id == "Markdown"]
            assert len(calls) == 1
            source = ast.unparse(calls[0])
        expressions.append(source)
    scratch = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(setup)] +
                                     [nbformat.v4.new_code_cell(source) for source in expressions])
    NotebookClient(scratch, kernel_name="tourism-mlops", timeout=120,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    for index, executed in zip(indexes, scratch.cells[1:]):
        generated = [out for out in executed.outputs if "text/markdown" in out.get("data", {})]
        original = [i for i, out in enumerate(notebook.cells[index].outputs)
                    if "text/markdown" in out.get("data", {})]
        assert len(generated) == len(original) == 1
        notebook.cells[index].outputs[original[0]] = generated[0]
    return notebook


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    personalize_notebook(notebook)
    refresh_observations(notebook)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print("Updated author, first-person observations, and report narrative; numeric tables and figures retained.")


if __name__ == "__main__":
    main()

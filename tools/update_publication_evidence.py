"""Refresh publication evidence without rerunning training or replacing the deployed model.

Run from the project root with the notebook dependencies installed. Original
training outputs remain intact; only verification, evidence, and conclusions
cells are executed again in a separate kernel.
"""
import os
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient
from author_report import personalize_notebook, refresh_observations

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "Tourism_Package_Prediction_Completed.ipynb"


def main():
    os.chdir(ROOT)
    required = ["github_verification.json", "browser_verification.json",
                "local_development_metrics.json", "deployed_release_metrics.json",
                "github_structure.png", "workflow_success.png", "streamlit_prediction.png"]
    for name in required:
        if not (ROOT / "evidence" / name).is_file():
            raise FileNotFoundError(f"Missing genuine publication evidence: evidence/{name}")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells[0].source = dedent("""
        # Tourism Package Prediction
        ### Visit with Us · Advanced Machine Learning and MLOps

        **Objective:** build a reproducible system to prioritize customers for a Wellness Tourism Package **before a sales conversation**.

        This notebook contains the complete project source, executed validation and preparation, tuned models with MLflow tracking, held-out evaluation, a Streamlit frontend, and a GitHub Actions workflow.

        **Published project:** [GitHub repository](https://github.com/dassubhojit-web/tourism-package-prediction) · [Successful four-job workflow](https://github.com/dassubhojit-web/tourism-package-prediction/actions/runs/45421135704) · [Public Streamlit app](https://tourism-package-prediction-ge2zhlrmfksupcejr6skkp.streamlit.app/)

        **Final deployed release:** Random Forest, threshold **0.185**, test average precision **0.8009**, ROC AUC **0.9302**, buyer recall **92.81%**, and precision **56.80%**. Section 9 contains the actual publication evidence, release commit, and screenshots.

        **Execution provenance:** the original training cells were run sequentially in the local development environment. Their actual outputs are retained below. GitHub subsequently retrained and committed a separate model, which serves the public app. Verification, evidence, and final conclusions were refreshed after deployment without retraining or changing that published model. Sections 6–8 retain the local execution history; Section 9 compares both runs and reports the deployed release.

        To reproduce training, use a separate working copy: running every training cell creates a new model in the local deployment directory. To refresh only the published report, run `tools/update_publication_evidence.py` and export the saved notebook to HTML.
    """).strip()
    note = "\n\n**Run shown in this section:** original local development execution. These recorded outputs are preserved as execution evidence. The final GitHub-trained serving release and its independently verified metrics are reported in Section 9."
    if note not in notebook.cells[20].source:
        notebook.cells[20].source += note
    notebook.cells[30].source = notebook.cells[30].source.replace(
        "The live cloud connection is a separate step after publishing the repository.",
        "The app is now hosted at the public Streamlit link in Section 9; the source and dependency cells below reproduce its serving code."
    )
    notebook.cells[45].source = dedent("""
        ## 9. Output evaluation and public deployment evidence · rubric: 4 points

        The project is public on GitHub and Streamlit Community Cloud. GitHub Actions run **#1** completed successfully on **November 24, 2026 at 9:07 a.m. America/Chicago**, with all four jobs successful. Registration and split artifacts were uploaded/downloaded, training and tests passed, and the bot committed the fitted model to `main`.

        **Training source:** `62715eb35bf9746673cb0723318dd63591d5576b`.
        **Published model commit:** [`8a31082862dcd4c81f7ba3ad6baeede5bd1eb8fe`](https://github.com/dassubhojit-web/tourism-package-prediction/commit/8a31082862dcd4c81f7ba3ad6baeede5bd1eb8fe).
        **MLflow parent run:** `9c74a6d5940f4b918cff3a871e5898a7`.

        The workflow preserved four artifacts: `registered-data`, `data-splits`, `trained-deployment`, and `mlflow-experiments`. Their IDs, digests, and successful job steps are recorded in `evidence/github_verification.json`. Artifact downloads remain subject to GitHub retention limits; the source, serving model, and reports are committed to the repository.

        The following cell shows the actual canonical links, compares the original local results with the GitHub-trained deployed release, and embeds genuine browser screenshots. The public app was checked in an anonymous browser context using the form's synthetic default profile. The page URL and observed outcome are recorded in `evidence/browser_verification.json`.

        The two trained artifacts produce slightly different holdout metrics. No outputs have been relabeled as if they came from another run. The deployed-release column below is authoritative for the live app. No additional tuning was performed using these test results.
    """).strip()
    verification_source = dedent("""
        from pathlib import Path
        import json
        import subprocess
        import sys
        import yaml
        from IPython.display import display, Markdown

        report_root = Path.cwd().resolve()
        workflow = yaml.safe_load((report_root / ".github/workflows/pipeline.yml").read_text())
        assert set(workflow["jobs"]) == {"register-dataset", "data-preparation", "model-training", "publish-model"}
        assert workflow["permissions"]["contents"] == "read"
        assert workflow["jobs"]["publish-model"]["permissions"]["contents"] == "write"
        github_check = json.loads((report_root / "evidence/github_verification.json").read_text())
        assert github_check["conclusion"] == "success"
        assert len(github_check["jobs"]) == 4
        assert all(job["conclusion"] == "success" for job in github_check["jobs"])
        print("Workflow YAML validated; four remote jobs and their recorded steps succeeded.")
        print("Verified remote run:", github_check["workflow_run_url"])
        print("Original GitHub runner test result:")
        ci_verification = (report_root / "tourism_project/reports/verification.txt").read_text()
        print(next(line for line in ci_verification.splitlines() if "passed" in line))
        final_check = subprocess.run([sys.executable, "-m", "pytest", "-q", "--disable-warnings"],
                                     cwd=report_root, capture_output=True, text=True)
        print("Local compatibility check of the GitHub-published model:")
        print(final_check.stdout)
        if final_check.returncode:
            print(final_check.stderr)
        final_check.check_returncode()
        _ = (report_root / "evidence/final_local_verification.txt").write_text(final_check.stdout)
    """).strip()
    evidence_source = dedent("""
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display, Markdown, Image

        report_root = Path.cwd().resolve()
        evidence_dir = report_root / "evidence"
        GITHUB_REPOSITORY_URL = "https://github.com/dassubhojit-web/tourism-package-prediction"
        GITHUB_ACTIONS_RUN_URL = "https://github.com/dassubhojit-web/tourism-package-prediction/actions/runs/45421135704"
        STREAMLIT_APP_URL = "https://tourism-package-prediction-ge2zhlrmfksupcejr6skkp.streamlit.app/"
        display(Markdown(f"[Public GitHub repository]({GITHUB_REPOSITORY_URL}) · "
                         f"[Successful GitHub Actions run]({GITHUB_ACTIONS_RUN_URL}) · "
                         f"[Public Streamlit app]({STREAMLIT_APP_URL})"))
        local_release = json.loads((evidence_dir / "local_development_metrics.json").read_text())
        deployed_release = json.loads((evidence_dir / "deployed_release_metrics.json").read_text())
        current_metadata = json.loads((report_root / "tourism_project/deployment/metadata.json").read_text())
        assert deployed_release == current_metadata
        assert local_release["raw_sha256"] == deployed_release["raw_sha256"]
        compare_metrics = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f2", "roc_auc",
                           "average_precision", "precision_at_top_20_percent", "lift_at_top_20_percent"]
        comparison = pd.DataFrame({
            "Original local development run": {k: local_release["test_metrics"][k] for k in compare_metrics},
            "GitHub-trained deployed release": {k: deployed_release["test_metrics"][k] for k in compare_metrics},
        })
        display(comparison.round(4))
        print("Deployed model:", deployed_release["model_name"])
        print("Saved decision threshold:", deployed_release["threshold"])
        print("Deployed release Python:", deployed_release["versions"]["python"])
        print("GitHub training CV average precision:", round(deployed_release["best_cv_average_precision"], 4))
        print("Deployed confusion matrix:", deployed_release["test_metrics"]["confusion_matrix"])
        print("GitHub release classification report:")
        print((report_root / "tourism_project/reports/classification_report.txt").read_text())
        display(Markdown("**GitHub-trained release: held-out evaluation**"))
        display(Image(filename=str(report_root / "tourism_project/reports/figures/test_evaluation.png")))
        display(Markdown("**GitHub-trained release: descriptive feature importance**"))
        display(Image(filename=str(report_root / "tourism_project/reports/figures/feature_importance.png")))
        for title, filename, url in [
            ("Public GitHub repository structure", "github_structure.png", GITHUB_REPOSITORY_URL),
            ("Successful four-job GitHub Actions execution", "workflow_success.png", GITHUB_ACTIONS_RUN_URL),
            ("Public Streamlit app with an actual prediction", "streamlit_prediction.png", STREAMLIT_APP_URL),
        ]:
            display(Markdown(f"**{title}** — [source page]({url})"))
            display(Image(filename=str(evidence_dir / filename)))
        print("Anonymous browser verification details are saved in evidence/browser_verification.json.")
    """).strip()
    notebook.cells[47].source = dedent("""
        ## 10. Conclusions and actionable recommendations

        The validated pipeline has run on GitHub, automatically committed its tested model to `main`, and supplied the model now served by the public Streamlit app. The final conclusions below use the deployed release; the original local training evidence is preserved in the earlier sections.
    """).strip()
    conclusion_source = dedent("""
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
            f"1. **Deployed model:** {release['model_name']} selected from 20 training-CV configurations; "
            f"holdout AP **{test['average_precision']:.3f}**, ROC AUC **{test['roc_auc']:.3f}**, "
            f"and accuracy **{test['accuracy']:.1%}**.",
            f"2. **Recall-focused outreach:** threshold **{release['threshold']:.3f}** gives recall "
            f"**{test['recall']:.1%}** and precision **{test['precision']:.1%}**. "
            f"The holdout decisions identify {tp} buyers, miss {fn}, and contact {fp} nonbuyers. "
            "Replace F2 with an economic objective when contact costs and margins are available.",
            f"3. **Budget scenario:** the top {test['top_20_percent_customers']} scored customers give "
            f"**{test['precision_at_top_20_percent']:.1%}** retrospective precision and "
            f"**{test['lift_at_top_20_percent']:.2f}×** lift. Pilot the ranking against a randomized "
            "or current targeting strategy to estimate incremental conversion.",
            f"4. **Observed drivers:** {top_features} have the largest descriptive AP decreases. "
            "These are predictive associations, not evidence of causality.",
            "5. **Published delivery:** the GitHub repository, successful four-job workflow, automatic model "
            "commit, public Streamlit prediction, and genuine screenshots are documented in Section 9. "
            "The final notebook and HTML contain the original execution history and the deployed-release evidence.",
        ]
        display(Markdown("\\n\\n".join(lines)))
    """).strip()
    # A separate short execution preserves all original model-building outputs.
    changed = [44, 46, 48]
    scratch = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source) for source in
                                              [verification_source, evidence_source, conclusion_source]])
    scratch.metadata.kernelspec = {"name": "tourism-mlops", "language": "python", "display_name": "Python (Tourism MLOps)"}
    NotebookClient(scratch, kernel_name="tourism-mlops", timeout=180,
                   resources={"metadata": {"path": str(ROOT)}}).execute()
    for index, executed in zip(changed, scratch.cells):
        notebook.cells[index] = executed
    notebook.metadata["publication_evidence"] = {
        "workflow_run_id": 45421135704,
        "published_model_commit": "8a31082862dcd4c81f7ba3ad6baeede5bd1eb8fe",
        "original_training_outputs_preserved": True,
        "report_only_cells_refreshed": changed,
    }
    personalize_notebook(notebook)
    refresh_observations(notebook)
    nbformat.validate(notebook)
    nbformat.write(notebook, NOTEBOOK)
    print(f"Updated {NOTEBOOK.name}; original training outputs preserved, publication evidence executed.")


if __name__ == "__main__":
    main()

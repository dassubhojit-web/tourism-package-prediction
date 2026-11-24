"""Exercise the deployed UI with the fitted model produced by training."""

from pathlib import Path
from unittest.mock import patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tourism_project" / "deployment" / "app.py"


@pytest.fixture(autouse=True)
def clear_model_cache():
    st.cache_resource.clear()
    st.cache_data.clear()
    yield
    st.cache_resource.clear()
    st.cache_data.clear()


def test_app_scores_customer_with_saved_model():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert not app.error
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    score = next(metric for metric in app.metric if metric.label == "Model interest score")
    assert 0 <= float(score.value.rstrip("%")) <= 100
    assert app.success or app.info


def test_app_rejects_children_exceeding_party_size():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    inputs = {widget.label: widget for widget in app.number_input}
    inputs["Number of people visiting"].set_value(1)
    inputs["Children under five visiting"].set_value(2)
    app.button[0].click().run()
    assert not app.exception
    assert any("cannot exceed" in error.value for error in app.error)
    assert not any(metric.label == "Model interest score" for metric in app.metric)


def test_app_handles_missing_model_without_traceback():
    with patch("joblib.load", side_effect=FileNotFoundError("model unavailable")):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert any("trained model is not available" in error.value for error in app.error)
    assert not app.button

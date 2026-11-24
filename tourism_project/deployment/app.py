"""Streamlit frontend for scoring customers before a sales conversation."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


DEPLOYMENT_DIR = Path(__file__).resolve().parent
MODEL_PATH = DEPLOYMENT_DIR / "model.joblib"
METADATA_PATH = DEPLOYMENT_DIR / "metadata.json"

st.set_page_config(
    page_title="Visit with Us | Wellness tourism",
    page_icon="🌿",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the prediction model…")
def load_model(model_path: str, modified_ns: int):
    """Load the fitted preprocessing pipeline and its training-selected threshold."""
    payload = joblib.load(model_path)
    required = {"pipeline", "threshold", "feature_columns"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("The model artifact does not match the app's expected format.")
    return payload


@st.cache_data
def load_metadata(metadata_path: str, modified_ns: int):
    """Metadata improves the explanation but is not required to serve predictions."""
    path = Path(metadata_path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


st.caption("VISIT WITH US  ·  CUSTOMER OUTREACH")
st.title("Find the next wellness traveler")
st.write(
    "Estimate a customer's interest in the Wellness Tourism Package using their "
    "profile and travel preferences, before a sales conversation."
)

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
except (OSError, ValueError, KeyError):
    st.error(
        "The trained model is not available. Run the training pipeline to create "
        "tourism_project/deployment/model.joblib, then restart the app."
    )
    st.stop()

metadata = load_metadata(
    str(METADATA_PATH), METADATA_PATH.stat().st_mtime_ns if METADATA_PATH.is_file() else 0
)
threshold = float(model["threshold"])

with st.sidebar:
    st.header("Plan your outreach")
    st.write(
        "Use this score to prioritize customers for a conversation about the "
        "Wellness Tourism Package."
    )
    st.divider()
    st.subheader("Model details")
    st.write(model.get("model_name", metadata.get("model_name", "Tourism classifier")))
    st.caption(f"Outreach decision threshold: {threshold:.1%}")
    metrics = metadata.get("test_metrics", metadata.get("metrics", {}))
    if "average_precision" in metrics:
        st.metric("Held-out average precision", f"{metrics['average_precision']:.3f}")
    if "recall" in metrics:
        st.metric("Held-out buyer recall", f"{metrics['recall']:.1%}")
    st.caption(
        "Scores come from historical purchase patterns and have not been calibrated "
        "as purchase probabilities. Validate performance on future campaigns."
    )

with st.form("customer_profile"):
    profile, travel, household = st.columns(3, gap="large")

    with profile:
        st.subheader("Customer profile")
        age = st.number_input("Age", min_value=18, max_value=61, value=36, step=1)
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox(
            "Marital status", ["Married", "Single", "Unmarried", "Divorced"]
        )
        city_tier = st.selectbox("City tier", [1, 2, 3])
        passport = st.selectbox("Valid passport", ["No", "Yes"])

    with travel:
        st.subheader("Travel preferences")
        party_size = st.number_input(
            "Number of people visiting", min_value=1, max_value=5, value=3, step=1
        )
        children = st.number_input(
            "Children under five visiting", min_value=0, max_value=3, value=0, step=1
        )
        trips = st.number_input("Annual trips", min_value=1, max_value=22, value=2, step=1)
        property_star = st.selectbox("Preferred hotel star rating", [3, 4, 5])

    with household:
        st.subheader("Work and household")
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"]
        )
        designation = st.selectbox(
            "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
        )
        monthly_income = st.number_input(
            "Gross monthly income",
            min_value=1000,
            max_value=98678,
            value=23000,
            step=500,
            help="Use the same currency and monthly units as the training dataset.",
        )
        own_car = st.selectbox("Owns a car", ["No", "Yes"])

    st.caption("Input ranges reflect the historical dataset used to develop this model.")
    submitted = st.form_submit_button("Estimate customer interest", type="primary")

if submitted:
    if children >= party_size:
        st.error(
            "Children cannot exceed the total number of people visiting, "
            "and the party must include at least one adult."
        )
    else:
        customer = pd.DataFrame(
            [
                {
                    "Age": age,
                    "CityTier": city_tier,
                    "Occupation": occupation,
                    "Gender": gender,
                    "NumberOfPersonVisiting": party_size,
                    "PreferredPropertyStar": property_star,
                    "MaritalStatus": marital_status,
                    "NumberOfTrips": trips,
                    "Passport": int(passport == "Yes"),
                    "OwnCar": int(own_car == "Yes"),
                    "NumberOfChildrenVisiting": children,
                    "Designation": designation,
                    "MonthlyIncome": monthly_income,
                }
            ],
            columns=model["feature_columns"],
        )
        score = float(model["pipeline"].predict_proba(customer)[0, 1])
        st.divider()
        result, explanation = st.columns([1, 2], gap="large")
        with result:
            st.metric("Model interest score", f"{score:.1%}")
            st.caption("An uncalibrated model score, not a guaranteed purchase probability.")
        with explanation:
            if score >= threshold:
                st.success("Prioritize for outreach")
                st.write(
                    "This customer meets the model's outreach threshold. A conversation "
                    "can help confirm their interest and the suitability of the package."
                )
            else:
                st.info("Standard outreach priority")
                st.write(
                    "This customer falls below the current outreach threshold. "
                    "The score can support campaign planning alongside business judgment."
                )
            st.caption(f"Decision uses the saved threshold of {threshold:.1%}.")

st.divider()
st.caption(
    "This model predicts purchase propensity from historical customer records. "
    "A higher score does not guarantee a purchase."
)

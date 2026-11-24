"""Shared data contract. All filesystem defaults are independent of the working directory."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT.parent
RAW_DATA = PROJECT / "data" / "tourism.csv"
PROCESSED = PROJECT / "data" / "processed"
REPORTS = PROJECT / "reports"
DEPLOYMENT = PROJECT / "deployment"
TARGET = "ProdTaken"
SEED = 42
EXCLUDED_FEATURES = ["TypeofContact", "DurationOfPitch", "NumberOfFollowups",
                     "ProductPitched", "PitchSatisfactionScore"]
FEATURE_COLUMNS = ["Age", "CityTier", "Occupation", "Gender",
                   "NumberOfPersonVisiting", "PreferredPropertyStar", "MaritalStatus",
                   "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
                   "Designation", "MonthlyIncome"]
CATEGORICAL_COLUMNS = ["CityTier", "Occupation", "Gender", "PreferredPropertyStar",
                       "MaritalStatus", "Passport", "OwnCar", "Designation"]
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]
EXPECTED_COLUMNS = ["CustomerID", TARGET, *FEATURE_COLUMNS, *EXCLUDED_FEATURES]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_raw(frame):
    """Fail early on incompatible schema or invalid labels; predictor nulls are imputed later."""
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    extras = [c for c in frame if c not in EXPECTED_COLUMNS and not c.startswith("Unnamed:")]
    if extras:
        raise ValueError(f"Unexpected columns: {extras}. Review the data contract first.")
    if frame.empty:
        raise ValueError("Dataset is empty.")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("ProdTaken must contain both classes 0 and 1 and no missing labels.")
    if frame["CustomerID"].isna().any() or frame["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be present and unique.")
    numeric = [c for c in EXPECTED_COLUMNS if c not in
               ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]]
    for column in numeric:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any() or np.isinf(converted.dropna().astype(float)).any():
            raise ValueError(f"{column} contains nonnumeric or infinite values.")
    for column, allowed in {"CityTier": {1, 2, 3}, "PreferredPropertyStar": {3, 4, 5},
                            "Passport": {0, 1}, "OwnCar": {0, 1}}.items():
        if not set(frame[column].dropna()).issubset(allowed):
            raise ValueError(f"{column} must use {sorted(allowed)}.")
    for column in ["Age", "MonthlyIncome", "NumberOfPersonVisiting", "NumberOfTrips"]:
        if (frame[column].dropna() <= 0).any():
            raise ValueError(f"{column} must be positive.")
    if (frame["NumberOfChildrenVisiting"].dropna() < 0).any():
        raise ValueError("NumberOfChildrenVisiting cannot be negative.")
    if (frame["NumberOfChildrenVisiting"] >= frame["NumberOfPersonVisiting"]).any():
        raise ValueError("Each party must include at least one adult.")


def clean_profiles(frame):
    """Deterministic text fixes and feature selection, without learning any population statistics."""
    result = frame[[*FEATURE_COLUMNS, TARGET]].copy()
    for column in result.select_dtypes(include="object"):
        result[column] = result[column].str.strip().replace("", np.nan)
    result["Gender"] = result["Gender"].replace({"Fe Male": "Female"})
    result["Occupation"] = result["Occupation"].replace({"Free Lancer": "Freelancer"})
    # Single and Unmarried remain separate; their equivalence has not been confirmed.
    return result


def profile_hashes(frame):
    return pd.util.hash_pandas_object(frame[FEATURE_COLUMNS], index=False)

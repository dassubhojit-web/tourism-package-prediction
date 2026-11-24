"""Prepare a reproducible, profile-disjoint 80/20 holdout and workflow artifacts."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tourism_project.model_building.schema import (
    RAW_DATA, PROCESSED, REPORTS, TARGET, SEED, FEATURE_COLUMNS,
    EXCLUDED_FEATURES, clean_profiles, file_hash, profile_hashes, save_json, validate_raw,
)


def prepare(data_path=RAW_DATA, output_dir=PROCESSED):
    data_path, output_dir = Path(data_path), Path(output_dir)
    raw = pd.read_csv(data_path)
    validate_raw(raw)
    cleaned = clean_profiles(raw)
    # Identical predictors can otherwise appear in train and test under distinct CustomerIDs.
    cleaned = cleaned.drop_duplicates(subset=[*FEATURE_COLUMNS, TARGET]).reset_index(drop=True)
    if profile_hashes(cleaned).duplicated().any():
        raise ValueError("Identical profiles have conflicting labels; review or use a grouped split.")
    X, y = cleaned[FEATURE_COLUMNS], cleaned[TARGET].astype(int)
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED,
    )
    overlap = set(profile_hashes(Xtrain)) & set(profile_hashes(Xtest))
    if overlap:
        raise ValueError("Duplicate profiles crossed the train/test boundary.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, split in {"Xtrain": Xtrain, "Xtest": Xtest, "ytrain": ytrain, "ytest": ytest}.items():
        split.to_csv(output_dir / f"{name}.csv", index=False)
    manifest = {
        "raw_sha256": file_hash(data_path), "random_state": SEED, "test_size": 0.20,
        "raw_rows": len(raw), "unique_rows": len(cleaned),
        "duplicates_removed": len(raw) - len(cleaned),
        "train_rows": len(Xtrain), "test_rows": len(Xtest),
        "train_purchase_rate": float(ytrain.mean()), "test_purchase_rate": float(ytest.mean()),
        "train_test_profile_overlap": len(overlap), "feature_columns": FEATURE_COLUMNS,
        "excluded_before_contact": EXCLUDED_FEATURES,
        "split_hashes": {f"{name}.csv": file_hash(output_dir / f"{name}.csv")
                         for name in ["Xtrain", "Xtest", "ytrain", "ytest"]},
    }
    save_json(output_dir / "split_manifest.json", manifest)
    save_json(REPORTS / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED)
    arguments = parser.parse_args()
    prepare(arguments.data, arguments.output_dir)

"""Register and validate the repository CSV; no external storage or credentials required."""
import argparse
import json
from pathlib import Path

import pandas as pd

from tourism_project.model_building.schema import (
    RAW_DATA, REPORTS, TARGET, FEATURE_COLUMNS, clean_profiles, file_hash,
    profile_hashes, save_json, validate_raw,
)


def register(data_path=RAW_DATA):
    data_path = Path(data_path)
    frame = pd.read_csv(data_path)
    validate_raw(frame)
    cleaned = clean_profiles(frame)
    summary = {
        "dataset": data_path.name, "sha256": file_hash(data_path),
        "rows": len(frame), "columns": len(frame.columns),
        "expected_predictors": FEATURE_COLUMNS,
        "missing_values": {c: int(v) for c, v in frame.isna().sum().items()},
        "target_counts": {str(k): int(v) for k, v in frame[TARGET].value_counts().sort_index().items()},
        "purchase_rate": float(frame[TARGET].mean()),
        "duplicate_profiles_after_normalization": int(profile_hashes(cleaned).duplicated().sum()),
        "gender_spelling_corrections": int(frame["Gender"].eq("Fe Male").sum()),
    }
    save_json(REPORTS / "data_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=RAW_DATA)
    register(parser.parse_args().data)

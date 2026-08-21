"""
Select Review 1 features and target from the crop dataset.
"""

from __future__ import annotations

import pandas as pd

from load_dataset import REQUIRED_FEATURES, TARGET_COLUMN
from paths import FEATURE_COLUMNS, TARGET


def select_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the 12 soil features and Crop_Type target.

    The original CSV is not modified; this returns a new DataFrame.
    """
    missing = [col for col in REQUIRED_FEATURES + [TARGET_COLUMN] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for Review 1 model: {missing}")

    selected = df[REQUIRED_FEATURES + [TARGET_COLUMN]].copy()
    print(f"Feature selection complete: {len(REQUIRED_FEATURES)} features + 1 target")
    print(f"Selected columns: {REQUIRED_FEATURES + [TARGET_COLUMN]}")
    print(f"Selected dataset shape: {selected.shape[0]} rows x {selected.shape[1]} columns")
    return selected


def verify_data_types(df: pd.DataFrame) -> dict[str, str]:
    """Verify numeric features and categorical target types."""
    report: dict[str, str] = {}

    for feature in FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[feature]):
            report[feature] = f"INVALID (expected numeric, got {df[feature].dtype})"
        else:
            report[feature] = f"OK ({df[feature].dtype})"

    if not pd.api.types.is_string_dtype(df[TARGET]) and not pd.api.types.is_object_dtype(df[TARGET]):
        report[TARGET] = f"INVALID (expected string/object, got {df[TARGET].dtype})"
    else:
        report[TARGET] = f"OK ({df[TARGET].dtype})"

    print("\n=== Data Type Verification ===")
    for column, status in report.items():
        print(f"{column}: {status}")

    invalid = [column for column, status in report.items() if status.startswith("INVALID")]
    if invalid:
        raise TypeError(f"Invalid data types found: {invalid}")

    return report


if __name__ == "__main__":
    from load_dataset import load_crop_dataset

    crop_df = load_crop_dataset()
    model_df = select_model_columns(crop_df)
    verify_data_types(model_df)

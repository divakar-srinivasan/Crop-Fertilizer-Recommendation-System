"""
Validate crop and fertilizer datasets for Review 1.
"""

from pathlib import Path

import pandas as pd

from load_dataset import (
    CROP_DATASET_PATH,
    FERTILIZER_DATASET_PATH,
    REQUIRED_FEATURES,
    TARGET_COLUMN,
    load_crop_dataset,
    load_fertilizer_dataset,
)


def validate_crop_dataset(df: pd.DataFrame) -> bool:
    """Validate crop dataset structure and Review 1 columns."""
    print("\n=== Crop Dataset Validation ===")
    errors = []

    if df.empty:
        errors.append("Crop dataset is empty.")

    missing_features = [col for col in REQUIRED_FEATURES if col not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        errors.append(f"Missing target column: {TARGET_COLUMN}")

    duplicate_rows = df.duplicated().sum()
    missing_values = df[REQUIRED_FEATURES + [TARGET_COLUMN]].isnull().sum().sum() if not missing_features and TARGET_COLUMN in df.columns else 0

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Duplicate rows: {duplicate_rows}")
    print(f"Missing values in Review 1 columns: {missing_values}")

    if TARGET_COLUMN in df.columns:
        crop_counts = df[TARGET_COLUMN].value_counts()
        print(f"Unique crops: {df[TARGET_COLUMN].nunique()}")
        print("Crop distribution:")
        print(crop_counts.to_string())

    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nCrop dataset validation PASSED.")
    return True


def validate_fertilizer_dataset(df: pd.DataFrame) -> bool:
    """Validate fertilizer dataset structure."""
    print("\n=== Fertilizer Dataset Validation ===")
    errors = []

    if df.empty:
        errors.append("Fertilizer dataset is empty.")

    expected_columns = {"Crop", "Nitrogen", "Phosphorus", "Potassium"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        errors.append(f"Missing expected columns: {sorted(missing_columns)}")

    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")

    if errors:
        print("\nValidation FAILED:")
        for error in errors:
            print(f"  - {error}")
        return False

    print("\nFertilizer dataset validation PASSED.")
    return True


def validate_file_paths() -> bool:
    """Check that required dataset files exist."""
    print("=== File Path Validation ===")
    paths_ok = True

    for path in [CROP_DATASET_PATH, FERTILIZER_DATASET_PATH]:
        if path.exists():
            print(f"[OK] Found: {path}")
        else:
            print(f"[MISSING] {path}")
            paths_ok = False

    return paths_ok


if __name__ == "__main__":
    print("Starting dataset validation...\n")

    if not validate_file_paths():
        raise SystemExit("Required dataset files are missing.")

    crop_df = load_crop_dataset()
    fertilizer_df = load_fertilizer_dataset()

    crop_ok = validate_crop_dataset(crop_df)
    fertilizer_ok = validate_fertilizer_dataset(fertilizer_df)

    if crop_ok and fertilizer_ok:
        print("\nAll dataset validations completed successfully.")
    else:
        raise SystemExit("Dataset validation failed.")

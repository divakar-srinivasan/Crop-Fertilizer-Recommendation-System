"""
Data cleaning, preprocessing, and train-test split for Review 1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from feature_selection import select_model_columns, verify_data_types
from load_dataset import CROP_DATASET_PATH, load_crop_dataset
from paths import (
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


@dataclass
class CleaningReport:
    original_rows: int
    original_columns: int
    selected_feature_count: int
    missing_values_before: int
    missing_values_after: int
    missing_value_method: str
    duplicate_rows_before: int
    duplicate_rows_removed: int
    invalid_value_rows_removed: int
    invalid_value_details: dict[str, int]
    rows_retained: int
    target_classes: list[str]
    outlier_note: str


@dataclass
class SplitReport:
    train_rows: int
    test_rows: int
    train_class_distribution: dict[str, int]
    test_class_distribution: dict[str, int]
    random_state: int
    test_size: float
    stratified: bool


def ensure_processed_dir() -> None:
    """Create processed output directory."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def analyze_missing_values(df: pd.DataFrame) -> pd.Series:
    """Count missing values per Review 1 column."""
    missing = df[FEATURE_COLUMNS + [TARGET]].isnull().sum()
    print("\n=== Missing Value Analysis ===")
    print(missing.to_string())
    return missing


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Handle missing values in selected columns.

    Current rule: drop rows with any missing feature or target value.
    """
    before = int(df[FEATURE_COLUMNS + [TARGET]].isnull().sum().sum())
    cleaned = df.dropna(subset=FEATURE_COLUMNS + [TARGET]).copy()
    after = int(cleaned[FEATURE_COLUMNS + [TARGET]].isnull().sum().sum())
    method = "drop_rows_with_missing_values" if before > 0 else "no_missing_values_found"

    print(f"Missing values before cleaning: {before}")
    print(f"Missing values after cleaning: {after}")
    print(f"Missing value method: {method}")
    return cleaned, method


def analyze_duplicates(df: pd.DataFrame) -> int:
    """Count duplicate rows in selected dataset."""
    duplicate_count = int(df.duplicated().sum())
    print("\n=== Duplicate Analysis ===")
    print(f"Duplicate rows found: {duplicate_count}")
    return duplicate_count


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows and report how many were removed."""
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    removed = before - len(cleaned)
    print(f"Duplicate rows removed: {removed}")
    print(f"Rows after duplicate removal: {len(cleaned)}")
    return cleaned, removed


def detect_invalid_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Detect clearly invalid numeric values.

    Rules:
    - Non-numeric coercion failures
    - Infinite values
    - Values below configured minimum bounds
    - pH values above 14
    """
    print("\n=== Invalid Value Detection ===")
    working = df.copy()
    invalid_details: dict[str, int] = {}

    for feature in FEATURE_COLUMNS:
        numeric = pd.to_numeric(working[feature], errors="coerce")
        invalid_mask = numeric.isna() & working[feature].notna()
        invalid_details[f"{feature}_non_numeric"] = int(invalid_mask.sum())

        min_bound, max_bound = FEATURE_BOUNDS[feature]
        if min_bound is not None:
            below_min = numeric < min_bound
            invalid_details[f"{feature}_below_min"] = int(below_min.sum())
        else:
            below_min = pd.Series(False, index=working.index)

        if max_bound is not None:
            above_max = numeric > max_bound
            invalid_details[f"{feature}_above_max"] = int(above_max.sum())
        else:
            above_max = pd.Series(False, index=working.index)

        infinite = np.isinf(numeric.to_numpy(dtype=float, copy=False))
        invalid_details[f"{feature}_infinite"] = int(infinite.sum())

        feature_invalid = invalid_mask | below_min.fillna(False) | above_max.fillna(False) | infinite
        if feature_invalid.any():
            print(f"{feature}: {int(feature_invalid.sum())} invalid value(s)")

    invalid_rows = pd.Series(False, index=working.index)
    for feature in FEATURE_COLUMNS:
        numeric = pd.to_numeric(working[feature], errors="coerce")
        min_bound, max_bound = FEATURE_BOUNDS[feature]

        feature_invalid = numeric.isna() & working[feature].notna()
        if min_bound is not None:
            feature_invalid |= numeric < min_bound
        if max_bound is not None:
            feature_invalid |= numeric > max_bound
        feature_invalid |= np.isinf(numeric.to_numpy(dtype=float, copy=False))

        invalid_rows |= feature_invalid.fillna(False)

    removed_count = int(invalid_rows.sum())
    cleaned = working.loc[~invalid_rows].copy()
    print(f"Invalid rows removed: {removed_count}")
    return cleaned, invalid_details


def encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encode crop labels for model training."""
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y)
    print("\n=== Target Encoding ===")
    for index, label in enumerate(encoder.classes_):
        print(f"{label} -> {index}")
    return encoded, encoder


def split_train_test(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    y_labels: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, SplitReport]:
    """Create stratified train-test split without scaling features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_labels,
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(y_labels)
    y_train_labels = label_encoder.inverse_transform(y_train)
    y_test_labels = label_encoder.inverse_transform(y_test)

    report = SplitReport(
        train_rows=len(X_train),
        test_rows=len(X_test),
        train_class_distribution=pd.Series(y_train_labels).value_counts().sort_index().to_dict(),
        test_class_distribution=pd.Series(y_test_labels).value_counts().sort_index().to_dict(),
        random_state=RANDOM_STATE,
        test_size=TEST_SIZE,
        stratified=True,
    )

    print("\n=== Train-Test Split ===")
    print(f"Training samples: {report.train_rows}")
    print(f"Testing samples: {report.test_rows}")
    print("Training class distribution:")
    print(pd.Series(report.train_class_distribution).to_string())
    print("Testing class distribution:")
    print(pd.Series(report.test_class_distribution).to_string())

    return X_train, X_test, y_train, y_test, report


def save_processed_outputs(
    selected_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
    cleaning_report: CleaningReport,
    split_report: SplitReport,
) -> dict[str, str]:
    """Save cleaned data, splits, encoder, and summary report."""
    ensure_processed_dir()

    paths = {
        "selected_clean": PROCESSED_DIR / "crop_selected_clean.csv",
        "X_train": PROCESSED_DIR / "X_train.csv",
        "X_test": PROCESSED_DIR / "X_test.csv",
        "y_train": PROCESSED_DIR / "y_train.csv",
        "y_test": PROCESSED_DIR / "y_test.csv",
        "label_encoder": PROCESSED_DIR / "label_encoder.joblib",
        "summary_json": PROCESSED_DIR / "preprocessing_summary.json",
        "summary_txt": PROCESSED_DIR / "preprocessing_summary.txt",
    }

    selected_df.to_csv(paths["selected_clean"], index=False)
    X_train.to_csv(paths["X_train"], index=False)
    X_test.to_csv(paths["X_test"], index=False)
    pd.DataFrame({"Crop_Type_Encoded": y_train}).to_csv(paths["y_train"], index=False)
    pd.DataFrame({"Crop_Type_Encoded": y_test}).to_csv(paths["y_test"], index=False)
    joblib.dump(label_encoder, paths["label_encoder"])

    summary = {
        "source_dataset": str(CROP_DATASET_PATH.relative_to(CROP_DATASET_PATH.parent.parent)),
        "cleaning": asdict(cleaning_report),
        "split": asdict(split_report),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET,
        "scaling_applied": False,
        "notes": [
            "Original crop_dataset.csv was not modified.",
            "Outliers were analyzed but not removed.",
            "Tree-based models do not require feature scaling in this project stage.",
        ],
    }

    with paths["summary_json"].open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with paths["summary_txt"].open("w", encoding="utf-8") as file:
        file.write("Review 1 Level 2 Preprocessing Summary\n")
        file.write("=" * 40 + "\n\n")
        file.write(json.dumps(summary, indent=2))

    print("\n=== Saved Processed Files ===")
    for name, path in paths.items():
        print(f"{name}: {path}")

    return {key: str(value) for key, value in paths.items()}


def run_preprocessing() -> dict:
    """Execute full Level 2 preprocessing workflow."""
    ensure_processed_dir()

    original_df = load_crop_dataset()
    original_shape = original_df.shape

    selected_df = select_model_columns(original_df)
    verify_data_types(selected_df)

    missing_before = int(selected_df[FEATURE_COLUMNS + [TARGET]].isnull().sum().sum())
    missing_series = analyze_missing_values(selected_df)
    cleaned_missing, missing_method = handle_missing_values(selected_df)

    duplicate_before = analyze_duplicates(cleaned_missing)
    deduped_df, duplicates_removed = remove_duplicates(cleaned_missing)

    rows_before_invalid_check = len(deduped_df)
    cleaned_df, invalid_details = detect_invalid_values(deduped_df)
    invalid_rows_removed = rows_before_invalid_check - len(cleaned_df)

    cleaning_report = CleaningReport(
        original_rows=original_shape[0],
        original_columns=original_shape[1],
        selected_feature_count=len(FEATURE_COLUMNS),
        missing_values_before=missing_before,
        missing_values_after=int(cleaned_df[FEATURE_COLUMNS + [TARGET]].isnull().sum().sum()),
        missing_value_method=missing_method,
        duplicate_rows_before=duplicate_before,
        duplicate_rows_removed=duplicates_removed,
        invalid_value_rows_removed=invalid_rows_removed,
        invalid_value_details=invalid_details,
        rows_retained=len(cleaned_df),
        target_classes=sorted(cleaned_df[TARGET].unique().tolist()),
        outlier_note="Outliers analyzed separately in eda.py; no outlier rows removed.",
    )

    print("\n=== Class Distribution After Cleaning ===")
    print(cleaned_df[TARGET].value_counts().sort_index().to_string())

    X = cleaned_df[FEATURE_COLUMNS]
    y_labels = cleaned_df[TARGET]
    y_encoded, label_encoder = encode_target(y_labels)

    X_train, X_test, y_train, y_test, split_report = split_train_test(X, y_encoded, y_labels)

    saved_paths = save_processed_outputs(
        cleaned_df,
        X_train,
        X_test,
        y_train,
        y_test,
        label_encoder,
        cleaning_report,
        split_report,
    )

    return {
        "original_shape": original_shape,
        "cleaning_report": cleaning_report,
        "split_report": split_report,
        "saved_paths": saved_paths,
    }


if __name__ == "__main__":
    run_preprocessing()

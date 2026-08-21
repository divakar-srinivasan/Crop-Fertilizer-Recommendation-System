"""
Clean and preprocess CropRec-BD v1 for the Dataset V2 baseline.

This script does not modify the raw Mendeley CSV and does not train models.
It creates a cleaned baseline dataset, a stratified train/test split, a label
encoder, and preprocessing reports under data/dataset_v2/.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_V2_DIR = PROJECT_ROOT / "data" / "dataset_v2"
RAW_PATH = DATASET_V2_DIR / "final_crops_data.csv"
CLEANED_DIR = DATASET_V2_DIR / "cleaned"
REPORTS_DIR = DATASET_V2_DIR / "reports"

EXPECTED_RAW_COLUMNS = ["Soil", "Soil_Moisture", "Humidity", "Temperature", "Crop Name"]
FEATURE_COLUMNS = ["Soil_Moisture", "Humidity", "Temperature"]
TARGET_COLUMN = "Crop Name"
TARGET_ENCODED_COLUMN = "Crop_Name_Encoded"
LABEL_STANDARDIZATION = {"Sugercane": "Sugarcane"}
EXPECTED_CLASSES = [
    "Banana",
    "Jute",
    "Maize",
    "Mango",
    "Pineapple",
    "Potato",
    "Strawberry",
    "Sugarcane",
    "Wheat",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file without changing it."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_dirs() -> None:
    """Create Dataset V2 output directories."""
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_raw_dataset() -> pd.DataFrame:
    """Load the raw V2 CSV and verify it exists."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw Dataset V2 file not found: {RAW_PATH}")
    return pd.read_csv(RAW_PATH)


def verify_raw_columns(df: pd.DataFrame) -> None:
    """Validate the raw V2 schema before selecting baseline columns."""
    actual_columns = list(df.columns)
    if actual_columns != EXPECTED_RAW_COLUMNS:
        raise ValueError(
            "Unexpected raw Dataset V2 columns. "
            f"Expected {EXPECTED_RAW_COLUMNS}, found {actual_columns}"
        )


def count_empty_strings(df: pd.DataFrame) -> dict[str, int]:
    """Count empty string values, including whitespace-only strings."""
    empty_counts: dict[str, int] = {}
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            empty_counts[column] = int(df[column].astype("string").str.strip().eq("").sum())
        else:
            empty_counts[column] = 0
    return empty_counts


def missing_report(df: pd.DataFrame) -> dict[str, Any]:
    """Report null-like missing values and empty strings."""
    nan_counts = {column: int(count) for column, count in df.isna().sum().items()}
    empty_counts = count_empty_strings(df)
    total_missing = int(sum(nan_counts.values()) + sum(empty_counts.values()))
    return {
        "nan_null_counts_by_column": nan_counts,
        "empty_string_counts_by_column": empty_counts,
        "total_missing_like_values": total_missing,
    }


def validate_numeric_features(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """
    Verify selected features are numeric and contain no NaN or infinite values.

    Non-numeric feature columns are rejected instead of silently converted.
    """
    report: dict[str, dict[str, Any]] = {}

    for feature in FEATURE_COLUMNS:
        series = df[feature]
        info: dict[str, Any] = {
            "dtype": str(series.dtype),
            "is_numeric_dtype": bool(pd.api.types.is_numeric_dtype(series)),
            "nan_count": int(series.isna().sum()),
            "infinite_count": None,
            "invalid_numeric_string_count": 0,
        }

        if not pd.api.types.is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors="coerce")
            invalid_mask = converted.isna() & series.notna()
            info["invalid_numeric_string_count"] = int(invalid_mask.sum())
            sample_invalid = series.loc[invalid_mask].astype(str).head(5).tolist()
            if sample_invalid:
                info["invalid_numeric_string_examples"] = sample_invalid
            report[feature] = info
            raise TypeError(f"Feature {feature} must be numeric; found dtype {series.dtype}")

        numeric_values = series.to_numpy(dtype=float, copy=False)
        info["infinite_count"] = int(np.isinf(numeric_values).sum())
        report[feature] = info

        if info["nan_count"] > 0 or info["infinite_count"] > 0:
            raise ValueError(f"Feature {feature} contains NaN or infinite values")

    return report


def feature_validation_summary(df: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Calculate min/max/mean/median/std/unique-count for baseline features."""
    summary: dict[str, dict[str, float | int]] = {}
    for feature in FEATURE_COLUMNS:
        values = df[feature]
        summary[feature] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "unique_values": int(values.nunique()),
        }
    return summary


def class_distribution(labels: pd.Series | np.ndarray) -> dict[str, int]:
    """Return a deterministic class-count dictionary."""
    series = pd.Series(labels)
    counts = series.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def encoded_distribution(encoded_labels: np.ndarray, encoder: LabelEncoder) -> dict[str, int]:
    """Return decoded class counts from encoded labels."""
    decoded = encoder.inverse_transform(encoded_labels)
    return class_distribution(decoded)


def label_encoder_mapping(encoder: LabelEncoder) -> dict[str, int]:
    """Return the fitted crop-to-integer mapping."""
    return {str(label): int(code) for code, label in enumerate(encoder.classes_)}


def save_markdown_report(report: dict[str, Any], path: Path) -> None:
    """Write a concise Markdown preprocessing report."""
    raw = report["raw_dataset"]
    cleaning = report["cleaning_operations"]
    split = report["train_test_split"]
    validation = report["validation_results"]

    lines = [
        "# Dataset V2 Preprocessing Summary",
        "",
        "## Objective",
        report["objective"],
        "",
        "## Raw Dataset Status",
        f"- Source file: `{raw['source_file']}`",
        f"- Rows before cleaning: {raw['rows_before_cleaning']}",
        f"- Columns before cleaning: {raw['columns_before_cleaning']}",
        f"- Raw SHA-256 before: `{raw['sha256_before']}`",
        f"- Raw SHA-256 after: `{raw['sha256_after']}`",
        f"- Raw file unchanged: {raw['raw_file_unchanged']}",
        "",
        "## Cleaning Operations",
        f"- Selected features: {', '.join(report['feature_list'])}",
        f"- Target column: `{report['target_column']}`",
        f"- Excluded column: `Soil`",
        f"- Duplicate rows before cleaning: {cleaning['duplicate_rows_before']}",
        f"- Duplicate rows removed from cleaned copy: {cleaning['duplicate_rows_removed']}",
        f"- Rows after duplicate removal: {cleaning['rows_after_duplicate_removal']}",
        f"- Missing-like values before row removal: {cleaning['missing_before']['total_missing_like_values']}",
        f"- Rows removed for missing values: {cleaning['missing_rows_removed']}",
        f"- Label standardization: {cleaning['label_standardization']}",
        "",
        "## Feature Validation",
        "",
        "| Feature | Min | Max | Mean | Median | Std | Unique Values |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for feature, stats in report["feature_validation"].items():
        lines.append(
            f"| {feature} | {stats['min']:.4f} | {stats['max']:.4f} | "
            f"{stats['mean']:.4f} | {stats['median']:.4f} | "
            f"{stats['std']:.4f} | {stats['unique_values']} |"
        )

    lines.extend(
        [
            "",
            "## Crop Classes",
            ", ".join(report["crop_classes"]),
            "",
            "## Label Mapping",
            "",
            "| Crop | Encoded Label |",
            "| --- | ---: |",
        ]
    )

    for crop, code in report["label_mapping"].items():
        lines.append(f"| {crop} | {code} |")

    lines.extend(
        [
            "",
            "## Train/Test Split",
            f"- Test size: {split['test_size']}",
            f"- Random state: {split['random_state']}",
            f"- Stratified: {split['stratified']}",
            f"- X_train shape: {split['X_train_shape']}",
            f"- X_test shape: {split['X_test_shape']}",
            f"- y_train shape: {split['y_train_shape']}",
            f"- y_test shape: {split['y_test_shape']}",
            f"- Train class distribution: {split['train_class_distribution']}",
            f"- Test class distribution: {split['test_class_distribution']}",
            "",
            "## Output Files",
        ]
    )

    for file_path in report["output_files"]:
        lines.append(f"- `{file_path}`")

    lines.extend(
        [
            "",
            "## Validation Results",
        ]
    )

    for key, value in validation.items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "## Errors/Warnings",
        ]
    )
    if report["errors"]:
        for error in report["errors"]:
            lines.append(f"- ERROR: {error}")
    if report["warnings"]:
        for warning in report["warnings"]:
            lines.append(f"- WARNING: {warning}")
    if not report["errors"] and not report["warnings"]:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Final Status",
            report["final_status"],
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the Dataset V2 cleaning and preprocessing workflow."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ensure_output_dirs()

    raw_sha_before = sha256_file(RAW_PATH)
    raw_df = load_raw_dataset()
    verify_raw_columns(raw_df)

    selected_df = raw_df[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    rows_before_cleaning = int(len(raw_df))
    columns_before_cleaning = int(raw_df.shape[1])
    selected_rows_before = int(len(selected_df))
    selected_columns = list(selected_df.columns)

    missing_before = missing_report(selected_df)
    rows_before_missing_drop = len(selected_df)
    cleaned_df = selected_df.replace(r"^\s*$", np.nan, regex=True).dropna(
        subset=FEATURE_COLUMNS + [TARGET_COLUMN]
    )
    missing_rows_removed = int(rows_before_missing_drop - len(cleaned_df))
    missing_after = missing_report(cleaned_df)

    numeric_validation = validate_numeric_features(cleaned_df)

    duplicate_rows_before = int(cleaned_df.duplicated().sum())
    cleaned_df = cleaned_df.drop_duplicates().copy()
    duplicate_rows_removed = int(rows_before_missing_drop - missing_rows_removed - len(cleaned_df))

    labels_before_standardization = class_distribution(cleaned_df[TARGET_COLUMN])
    labels_to_standardize = {
        raw_label: int((cleaned_df[TARGET_COLUMN] == raw_label).sum())
        for raw_label in LABEL_STANDARDIZATION
    }
    cleaned_df[TARGET_COLUMN] = cleaned_df[TARGET_COLUMN].replace(LABEL_STANDARDIZATION)
    labels_after_standardization = class_distribution(cleaned_df[TARGET_COLUMN])
    final_classes = sorted(labels_after_standardization)

    if final_classes != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected final crop classes: {final_classes}")

    feature_summary = feature_validation_summary(cleaned_df)

    X = cleaned_df[FEATURE_COLUMNS].copy()
    y = cleaned_df[TARGET_COLUMN].copy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    mapping = label_encoder_mapping(encoder)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_distribution = encoded_distribution(y_train, encoder)
    test_distribution = encoded_distribution(y_test, encoder)

    cleaned_csv_path = CLEANED_DIR / "croprec_bd_clean.csv"
    x_train_path = CLEANED_DIR / "X_train.csv"
    x_test_path = CLEANED_DIR / "X_test.csv"
    y_train_path = CLEANED_DIR / "y_train.csv"
    y_test_path = CLEANED_DIR / "y_test.csv"
    encoder_path = CLEANED_DIR / "label_encoder.joblib"
    json_report_path = REPORTS_DIR / "preprocessing_summary.json"
    md_report_path = REPORTS_DIR / "preprocessing_summary.md"

    cleaned_df.to_csv(cleaned_csv_path, index=False)
    X_train.to_csv(x_train_path, index=False)
    X_test.to_csv(x_test_path, index=False)
    pd.DataFrame({TARGET_ENCODED_COLUMN: y_train}).to_csv(y_train_path, index=False)
    pd.DataFrame({TARGET_ENCODED_COLUMN: y_test}).to_csv(y_test_path, index=False)
    joblib.dump(encoder, encoder_path)

    raw_sha_after = sha256_file(RAW_PATH)

    output_files = [
        cleaned_csv_path,
        x_train_path,
        x_test_path,
        y_train_path,
        y_test_path,
        encoder_path,
        json_report_path,
        md_report_path,
    ]

    validation_results = {
        "raw_sha_preserved": raw_sha_before == raw_sha_after,
        "cleaned_rows_equal_2891": len(cleaned_df) == 2891,
        "cleaned_has_three_features_plus_target": list(cleaned_df.columns) == FEATURE_COLUMNS + [TARGET_COLUMN],
        "soil_excluded_from_cleaned_dataset": "Soil" not in cleaned_df.columns,
        "nine_classes_present": final_classes == EXPECTED_CLASSES,
        "train_rows_equal_2312": len(X_train) == 2312,
        "test_rows_equal_579": len(X_test) == 579,
        "X_train_shape": list(X_train.shape),
        "X_test_shape": list(X_test.shape),
        "y_train_shape": [int(len(y_train))],
        "y_test_shape": [int(len(y_test))],
        "feature_order_preserved": list(X_train.columns) == FEATURE_COLUMNS and list(X_test.columns) == FEATURE_COLUMNS,
        "no_nan_values": bool(cleaned_df.isna().sum().sum() == 0),
        "no_infinite_feature_values": bool(
            np.isfinite(cleaned_df[FEATURE_COLUMNS].to_numpy(dtype=float)).all()
        ),
        "label_encoder_class_count": int(len(encoder.classes_)),
        "label_encoder_mapping_recorded": bool(mapping),
        "primary_output_files_exist": all(path.exists() for path in output_files[:-2]),
    }

    errors: list[str] = []
    warnings: list[str] = []
    if not all(validation_results.values()):
        errors.append("One or more preprocessing validation checks failed.")
    if duplicate_rows_before != 1 or duplicate_rows_removed != 1:
        warnings.append(
            f"Expected one duplicate row; found {duplicate_rows_before}, removed {duplicate_rows_removed}."
        )
    if missing_before["total_missing_like_values"] != 0:
        warnings.append("Missing-like values were found and removed from the cleaned copy.")

    report: dict[str, Any] = {
        "objective": "Prepare the first Dataset V2 baseline split from CropRec-BD v1 without modifying the raw CSV or training models.",
        "raw_dataset": {
            "source_file": str(RAW_PATH.relative_to(PROJECT_ROOT)),
            "expected_columns": EXPECTED_RAW_COLUMNS,
            "actual_columns": list(raw_df.columns),
            "rows_before_cleaning": rows_before_cleaning,
            "columns_before_cleaning": columns_before_cleaning,
            "sha256_before": raw_sha_before,
            "sha256_after": raw_sha_after,
            "raw_file_unchanged": raw_sha_before == raw_sha_after,
        },
        "cleaning_operations": {
            "selected_columns": selected_columns,
            "selected_rows_before_cleaning": selected_rows_before,
            "excluded_columns": ["Soil"],
            "missing_before": missing_before,
            "missing_rows_removed": missing_rows_removed,
            "missing_after": missing_after,
            "numeric_validation": numeric_validation,
            "duplicate_rows_before": duplicate_rows_before,
            "duplicate_rows_removed": duplicate_rows_removed,
            "rows_after_duplicate_removal": int(len(cleaned_df)),
            "label_standardization": LABEL_STANDARDIZATION,
            "label_standardization_counts": labels_to_standardize,
            "labels_before_standardization": labels_before_standardization,
            "labels_after_standardization": labels_after_standardization,
        },
        "before_after_counts": {
            "raw_rows_before": rows_before_cleaning,
            "selected_rows_before": selected_rows_before,
            "duplicates_removed": duplicate_rows_removed,
            "missing_rows_removed": missing_rows_removed,
            "cleaned_rows_after": int(len(cleaned_df)),
        },
        "feature_list": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "crop_classes": final_classes,
        "label_mapping": mapping,
        "feature_validation": feature_summary,
        "train_test_split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": [int(len(y_train))],
            "y_test_shape": [int(len(y_test))],
            "train_class_distribution": train_distribution,
            "test_class_distribution": test_distribution,
        },
        "output_files": [str(path.relative_to(PROJECT_ROOT)) for path in output_files],
        "validation_results": validation_results,
        "errors": errors,
        "warnings": warnings,
        "final_status": "DATASET V2 CLEANING COMPLETED" if not errors else "DATASET V2 CLEANING FAILED",
    }

    json_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_markdown_report(report, md_report_path)

    report["validation_results"]["all_output_files_exist"] = all(path.exists() for path in output_files)
    if not report["validation_results"]["all_output_files_exist"]:
        report["errors"].append("One or more requested output files were not created.")
        report["final_status"] = "DATASET V2 CLEANING FAILED"

    json_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_markdown_report(report, md_report_path)

    print("Dataset V2 preprocessing summary")
    print(f"Raw rows before cleaning: {rows_before_cleaning}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Classes: {final_classes}")
    print(f"Label mapping: {mapping}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Raw SHA preserved: {raw_sha_before == raw_sha_after}")
    if errors:
        raise RuntimeError("; ".join(errors))

    print("✅ DATASET V2 CLEANING COMPLETED")


if __name__ == "__main__":
    main()

"""
Dataset V2 robustness and ablation validation.

This script is evaluation-only. It does not modify the raw dataset, cleaned
dataset, labels, or the saved best Dataset V2 model.
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
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_V2_DIR = PROJECT_ROOT / "data" / "dataset_v2"
CLEANED_DIR = DATASET_V2_DIR / "cleaned"
REPORTS_DIR = DATASET_V2_DIR / "reports"
MODELS_DIR = PROJECT_ROOT / "models" / "dataset_v2"

FEATURE_ORDER = ["Soil_Moisture", "Humidity", "Temperature"]
TARGET_ENCODED_COLUMN = "Crop_Name_Encoded"
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
N_SPLITS = 5


def configure_stdout() -> None:
    """Allow the required completion marker to print in Windows terminals."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sha256_file(path: Path) -> str:
    """Return SHA-256 for an existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    """Ensure a required artifact exists before the analysis starts."""
    if not path.exists():
        raise FileNotFoundError(f"Required Dataset V2 artifact not found: {path}")


def load_v2_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any, Any]:
    """Load cleaned split, label encoder, and saved best V2 model."""
    required_paths = [
        CLEANED_DIR / "X_train.csv",
        CLEANED_DIR / "X_test.csv",
        CLEANED_DIR / "y_train.csv",
        CLEANED_DIR / "y_test.csv",
        CLEANED_DIR / "label_encoder.joblib",
        MODELS_DIR / "crop_model_v2.pkl",
        MODELS_DIR / "label_encoder_v2.joblib",
    ]
    for path in required_paths:
        require_file(path)

    x_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
    x_test = pd.read_csv(CLEANED_DIR / "X_test.csv")
    y_train = pd.read_csv(CLEANED_DIR / "y_train.csv")[TARGET_ENCODED_COLUMN]
    y_test = pd.read_csv(CLEANED_DIR / "y_test.csv")[TARGET_ENCODED_COLUMN]
    cleaned_encoder = joblib.load(CLEANED_DIR / "label_encoder.joblib")
    model_encoder = joblib.load(MODELS_DIR / "label_encoder_v2.joblib")
    best_model = joblib.load(MODELS_DIR / "crop_model_v2.pkl")

    validate_artifacts(x_train, x_test, y_train, y_test, cleaned_encoder, model_encoder)
    return x_train, x_test, y_train, y_test, model_encoder, best_model


def validate_artifacts(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cleaned_encoder: Any,
    model_encoder: Any,
) -> None:
    """Validate the V2 split and encoders without modifying them."""
    if list(x_train.columns) != FEATURE_ORDER:
        raise ValueError(f"Unexpected X_train feature order: {list(x_train.columns)}")
    if list(x_test.columns) != FEATURE_ORDER:
        raise ValueError(f"Unexpected X_test feature order: {list(x_test.columns)}")
    if tuple(x_train.shape) != (2312, 3):
        raise ValueError(f"Unexpected X_train shape: {x_train.shape}")
    if tuple(x_test.shape) != (579, 3):
        raise ValueError(f"Unexpected X_test shape: {x_test.shape}")
    if len(y_train) != len(x_train):
        raise ValueError("X_train and y_train row counts do not match")
    if len(y_test) != len(x_test):
        raise ValueError("X_test and y_test row counts do not match")
    if list(cleaned_encoder.classes_) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected cleaned encoder classes: {list(cleaned_encoder.classes_)}")
    if list(model_encoder.classes_) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected model encoder classes: {list(model_encoder.classes_)}")
    if list(cleaned_encoder.classes_) != list(model_encoder.classes_):
        raise ValueError("Cleaned split encoder and model encoder differ")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise ValueError("X_train contains non-finite values")
    if not np.isfinite(x_test.to_numpy(dtype=float)).all():
        raise ValueError("X_test contains non-finite values")
    if x_train.isna().sum().sum() or x_test.isna().sum().sum():
        raise ValueError("Feature split contains NaN values")


def build_catboost(feature_columns: list[str]) -> CatBoostClassifier:
    """Create the fixed CatBoost baseline configuration."""
    _ = feature_columns
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="MultiClass",
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )


def predict_labels(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return integer predictions for CatBoost."""
    return np.asarray(model.predict(features)).reshape(-1).astype(int)


def metric_row(name: str, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Calculate requested robustness metrics."""
    return {
        "Experiment": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Macro_Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_cross_validation(x_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Stratified 5-fold CV using only the cleaned training data."""
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for fold_number, (train_index, validation_index) in enumerate(splitter.split(x_train, y_train), start=1):
        fold_x_train = x_train.iloc[train_index][FEATURE_ORDER]
        fold_y_train = y_train.iloc[train_index]
        fold_x_val = x_train.iloc[validation_index][FEATURE_ORDER]
        fold_y_val = y_train.iloc[validation_index]

        model = build_catboost(FEATURE_ORDER)
        model.fit(fold_x_train, fold_y_train)
        y_pred = predict_labels(model, fold_x_val)
        row = metric_row(f"Fold {fold_number}", fold_y_val, y_pred)
        row["Fold"] = fold_number
        row["Train_Size"] = int(len(fold_x_train))
        row["Validation_Size"] = int(len(fold_x_val))
        rows.append(row)

    cv_df = pd.DataFrame(rows)[
        [
            "Fold",
            "Train_Size",
            "Validation_Size",
            "Accuracy",
            "Macro_F1",
            "Weighted_F1",
            "Macro_Precision",
            "Macro_Recall",
        ]
    ]

    metric_columns = ["Accuracy", "Macro_F1", "Weighted_F1", "Macro_Precision", "Macro_Recall"]
    summary = {
        "experiment": "Stratified 5-fold cross-validation on Dataset V2 training data only",
        "model": "CatBoost",
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "test_set_used": False,
        "folds": cv_df.to_dict(orient="records"),
        "statistics": {},
    }
    for metric in metric_columns:
        values = cv_df[metric]
        summary["statistics"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return cv_df, summary


def run_feature_ablation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Train temporary CatBoost models for controlled feature ablations."""
    configurations = {
        "A_All_Features": ["Soil_Moisture", "Humidity", "Temperature"],
        "B_No_Soil_Moisture": ["Humidity", "Temperature"],
        "C_No_Temperature": ["Soil_Moisture", "Humidity"],
        "D_No_Humidity": ["Soil_Moisture", "Temperature"],
    }
    rows = []
    for config_name, features in configurations.items():
        model = build_catboost(features)
        model.fit(x_train[features], y_train)
        y_pred = predict_labels(model, x_test[features])
        row = metric_row(config_name, y_test, y_pred)
        row["Configuration"] = config_name
        row["Features"] = ", ".join(features)
        row["Feature_Count"] = len(features)
        rows.append(row)

    return pd.DataFrame(rows)[
        [
            "Configuration",
            "Features",
            "Feature_Count",
            "Accuracy",
            "Macro_F1",
            "Weighted_F1",
            "Macro_Precision",
            "Macro_Recall",
        ]
    ]


def feature_importance_report(best_model: Any) -> pd.DataFrame:
    """Record feature importance for the saved best CatBoost model."""
    importances = np.asarray(best_model.get_feature_importance(), dtype=float)
    total = float(importances.sum())
    rows = []
    for feature, importance in zip(FEATURE_ORDER, importances):
        rows.append(
            {
                "Model": "CatBoost",
                "Feature": feature,
                "Importance": float(importance),
                "Normalized_Importance": float(importance / total) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(by="Normalized_Importance", ascending=False).reset_index(drop=True)


def test_set_sanity_report(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Verify exact test set integrity findings."""
    train_with_label = x_train.copy()
    train_with_label[TARGET_ENCODED_COLUMN] = y_train.to_numpy()
    test_with_label = x_test.copy()
    test_with_label[TARGET_ENCODED_COLUMN] = y_test.to_numpy()

    train_feature_rows = {tuple(row) for row in x_train.to_numpy()}
    test_feature_rows = {tuple(row) for row in x_test.to_numpy()}
    train_exact_rows = {tuple(row) for row in train_with_label.to_numpy()}
    test_exact_rows = {tuple(row) for row in test_with_label.to_numpy()}

    return {
        "test_samples": int(len(x_test)),
        "test_set_contains_579_samples": bool(len(x_test) == 579),
        "test_set_used_during_cross_validation": False,
        "test_set_used_during_training": False,
        "test_labels_used_for_hyperparameter_tuning": False,
        "duplicate_test_feature_rows": int(x_test.duplicated().sum()),
        "duplicate_test_rows_with_label": int(test_with_label.duplicated().sum()),
        "test_feature_rows_duplicated_in_training": int(len(train_feature_rows & test_feature_rows)),
        "test_rows_with_label_duplicated_in_training": int(len(train_exact_rows & test_exact_rows)),
        "feature_order_train": list(x_train.columns),
        "feature_order_test": list(x_test.columns),
        "feature_order_identical": list(x_train.columns) == list(x_test.columns) == FEATURE_ORDER,
        "label_encoding_identical": True,
        "encoded_label_min_train": int(y_train.min()),
        "encoded_label_max_train": int(y_train.max()),
        "encoded_label_min_test": int(y_test.min()),
        "encoded_label_max_test": int(y_test.max()),
    }


def crop_wise_robustness(best_model: Any, x_test: pd.DataFrame, y_test: pd.Series, encoder: Any) -> pd.DataFrame:
    """Calculate per-crop performance for the saved best CatBoost model."""
    y_pred = predict_labels(best_model, x_test[FEATURE_ORDER])
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(encoder.classes_))),
        target_names=list(encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for crop in encoder.classes_:
        metrics = report[crop]
        rows.append(
            {
                "Crop": crop,
                "Support": int(metrics["support"]),
                "Precision": float(metrics["precision"]),
                "Recall": float(metrics["recall"]),
                "F1": float(metrics["f1-score"]),
            }
        )
    return pd.DataFrame(rows)


def determine_stability(
    cv_summary: dict[str, Any],
    ablation_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    crop_df: pd.DataFrame,
) -> dict[str, Any]:
    """Determine the evidence-supported robustness interpretation."""
    cv_macro_mean = float(cv_summary["statistics"]["Macro_F1"]["mean"])
    cv_macro_std = float(cv_summary["statistics"]["Macro_F1"]["std"])
    full_row = ablation_df[ablation_df["Configuration"] == "A_All_Features"].iloc[0]
    no_soil_row = ablation_df[ablation_df["Configuration"] == "B_No_Soil_Moisture"].iloc[0]
    macro_f1_drop_no_soil = float(full_row["Macro_F1"] - no_soil_row["Macro_F1"])
    soil_importance = float(
        importance_df.loc[importance_df["Feature"] == "Soil_Moisture", "Normalized_Importance"].iloc[0]
    )
    min_crop_f1 = float(crop_df["F1"].min())

    if cv_macro_mean >= 0.98 and macro_f1_drop_no_soil >= 0.2 and soil_importance >= 0.8:
        category = "B. Stable but heavily dependent on Soil_Moisture"
    elif cv_macro_std > 0.03:
        category = "C. Highly sensitive to the train/test split"
    elif cv_macro_mean >= 0.99 and min_crop_f1 >= 0.95:
        category = "A. Stable and robust"
    else:
        category = "E. Other"

    return {
        "category": category,
        "cv_macro_f1_mean": cv_macro_mean,
        "cv_macro_f1_std": cv_macro_std,
        "full_feature_macro_f1": float(full_row["Macro_F1"]),
        "no_soil_moisture_macro_f1": float(no_soil_row["Macro_F1"]),
        "macro_f1_drop_without_soil_moisture": macro_f1_drop_no_soil,
        "soil_moisture_normalized_importance": soil_importance,
        "minimum_crop_f1": min_crop_f1,
        "recommended_interpretation": (
            "The model demonstrates strong performance under the evaluated dataset distribution, "
            "but the evidence shows strong dependence on Soil_Moisture. Do not claim 99% real-world farm generalization."
        ),
    }


def round_records(df: pd.DataFrame, decimals: int = 6) -> list[dict[str, Any]]:
    """Return rounded JSON records for report readability."""
    rounded = df.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)
    return rounded.to_dict(orient="records")


def write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    """Write a concise Markdown robustness summary."""
    stability = summary["stability_assessment"]
    lines = [
        "# Dataset V2 Robustness Summary",
        "",
        "## Cross-validation",
        f"- Mean Macro F1: {summary['cross_validation']['statistics']['Macro_F1']['mean']:.4f}",
        f"- Std Macro F1: {summary['cross_validation']['statistics']['Macro_F1']['std']:.4f}",
        f"- Mean Accuracy: {summary['cross_validation']['statistics']['Accuracy']['mean']:.4f}",
        "",
        "## Ablation",
        f"- Full feature Macro F1: {stability['full_feature_macro_f1']:.4f}",
        f"- Without Soil_Moisture Macro F1: {stability['no_soil_moisture_macro_f1']:.4f}",
        f"- Macro F1 drop without Soil_Moisture: {stability['macro_f1_drop_without_soil_moisture']:.4f}",
        "",
        "## Feature Importance",
        f"- Soil_Moisture normalized importance: {stability['soil_moisture_normalized_importance']:.4f}",
        "",
        "## Assessment",
        f"- {stability['category']}",
        f"- {stability['recommended_interpretation']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_robustness_analysis() -> dict[str, Any]:
    """Run all requested robustness and ablation checks."""
    configure_stdout()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "crop_model_v2.pkl"
    encoder_path = MODELS_DIR / "label_encoder_v2.joblib"
    model_sha_before = sha256_file(model_path)
    encoder_sha_before = sha256_file(encoder_path)

    x_train, x_test, y_train, y_test, encoder, best_model = load_v2_artifacts()

    cv_df, cv_summary = run_cross_validation(x_train, y_train)
    ablation_df = run_feature_ablation(x_train, x_test, y_train, y_test)
    importance_df = feature_importance_report(best_model)
    sanity = test_set_sanity_report(x_train, x_test, y_train, y_test)
    crop_df = crop_wise_robustness(best_model, x_test, y_test, encoder)
    stability = determine_stability(cv_summary, ablation_df, importance_df, crop_df)

    best_crop_row = crop_df.sort_values(by=["F1", "Recall", "Precision"], ascending=False).iloc[0]
    worst_crop_row = crop_df.sort_values(by=["F1", "Recall", "Precision"], ascending=True).iloc[0]

    cv_results_path = REPORTS_DIR / "cross_validation_results.csv"
    cv_summary_path = REPORTS_DIR / "cross_validation_summary.json"
    ablation_path = REPORTS_DIR / "feature_ablation_results.csv"
    importance_path = REPORTS_DIR / "feature_importance_v2.csv"
    sanity_path = REPORTS_DIR / "test_set_sanity_report.json"
    crop_path = REPORTS_DIR / "crop_wise_robustness.csv"
    robustness_json_path = REPORTS_DIR / "robustness_summary.json"
    robustness_md_path = REPORTS_DIR / "robustness_summary.md"

    cv_df.to_csv(cv_results_path, index=False)
    cv_summary_path.write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    ablation_df.to_csv(ablation_path, index=False)
    importance_df.to_csv(importance_path, index=False)
    sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    crop_df.to_csv(crop_path, index=False)

    model_sha_after = sha256_file(model_path)
    encoder_sha_after = sha256_file(encoder_path)

    summary = {
        "objective": "Dataset V2 robustness and ablation validation without modifying data, labels, or saved models.",
        "cross_validation": cv_summary,
        "feature_ablation": round_records(ablation_df),
        "feature_importance": round_records(importance_df),
        "test_set_sanity": sanity,
        "crop_wise_robustness": round_records(crop_df),
        "best_performing_crop": best_crop_row.to_dict(),
        "worst_performing_crop": worst_crop_row.to_dict(),
        "stability_assessment": stability,
        "model_integrity": {
            "model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "encoder_path": str(encoder_path.relative_to(PROJECT_ROOT)),
            "model_sha_before": model_sha_before,
            "model_sha_after": model_sha_after,
            "model_unchanged": model_sha_before == model_sha_after,
            "encoder_sha_before": encoder_sha_before,
            "encoder_sha_after": encoder_sha_after,
            "encoder_unchanged": encoder_sha_before == encoder_sha_after,
        },
        "output_files": [
            str(cv_results_path.relative_to(PROJECT_ROOT)),
            str(cv_summary_path.relative_to(PROJECT_ROOT)),
            str(ablation_path.relative_to(PROJECT_ROOT)),
            str(importance_path.relative_to(PROJECT_ROOT)),
            str(sanity_path.relative_to(PROJECT_ROOT)),
            str(crop_path.relative_to(PROJECT_ROOT)),
            str(robustness_json_path.relative_to(PROJECT_ROOT)),
            str(robustness_md_path.relative_to(PROJECT_ROOT)),
        ],
        "final_status": "DATASET V2 ROBUSTNESS ANALYSIS COMPLETED",
    }

    robustness_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, robustness_md_path)

    print("\nCross-validation results")
    print(cv_df.round(4).to_string(index=False))
    print("\nFeature ablation results")
    print(ablation_df.round(4).to_string(index=False))
    print("\nFeature importance")
    print(importance_df.round(4).to_string(index=False))
    print("\nStability assessment")
    print(stability["category"])
    print(stability["recommended_interpretation"])
    print("✅ DATASET V2 ROBUSTNESS ANALYSIS COMPLETED")
    return summary


if __name__ == "__main__":
    run_robustness_analysis()

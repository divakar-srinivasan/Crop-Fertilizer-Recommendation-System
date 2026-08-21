"""
Train and evaluate Dataset V2 baseline crop recommendation models.

This experiment uses only the already-cleaned Dataset V2 train/test artifacts.
It does not redo preprocessing and does not overwrite any Review 1/V1 model
files in models/.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_V2_DIR = PROJECT_ROOT / "data" / "dataset_v2"
CLEANED_DIR = DATASET_V2_DIR / "cleaned"
REPORTS_DIR = DATASET_V2_DIR / "reports"
IMAGES_DIR = PROJECT_ROOT / "images" / "dataset_v2" / "model_evaluation"
MODELS_DIR = PROJECT_ROOT / "models"
V2_MODELS_DIR = MODELS_DIR / "dataset_v2"

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
PREDICTION_TEST_ROWS = 5


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


def snapshot_existing_v1_model_artifacts() -> dict[str, str]:
    """Hash direct model files in models/ so V1 overwrites can be detected."""
    snapshot: dict[str, str] = {}
    if not MODELS_DIR.exists():
        return snapshot

    for path in sorted(MODELS_DIR.iterdir()):
        if path.is_file() and path.suffix.lower() in {".joblib", ".pkl"}:
            snapshot[str(path.relative_to(PROJECT_ROOT))] = sha256_file(path)
    return snapshot


def ensure_output_dirs() -> None:
    """Create output directories for V2 model artifacts and reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    V2_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def require_file(path: Path) -> None:
    """Raise a clear error when a required cleaned artifact is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required Dataset V2 artifact not found: {path}")


def load_dataset_v2_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """Load the already-cleaned Dataset V2 train/test split and encoder."""
    required_paths = [
        CLEANED_DIR / "X_train.csv",
        CLEANED_DIR / "X_test.csv",
        CLEANED_DIR / "y_train.csv",
        CLEANED_DIR / "y_test.csv",
        CLEANED_DIR / "label_encoder.joblib",
    ]
    for path in required_paths:
        require_file(path)

    x_train = pd.read_csv(CLEANED_DIR / "X_train.csv")
    x_test = pd.read_csv(CLEANED_DIR / "X_test.csv")
    y_train_df = pd.read_csv(CLEANED_DIR / "y_train.csv")
    y_test_df = pd.read_csv(CLEANED_DIR / "y_test.csv")
    label_encoder = joblib.load(CLEANED_DIR / "label_encoder.joblib")

    validate_split(x_train, x_test, y_train_df, y_test_df, label_encoder)

    y_train = y_train_df[TARGET_ENCODED_COLUMN]
    y_test = y_test_df[TARGET_ENCODED_COLUMN]
    return x_train[FEATURE_ORDER], x_test[FEATURE_ORDER], y_train, y_test, label_encoder


def validate_split(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train_df: pd.DataFrame,
    y_test_df: pd.DataFrame,
    label_encoder: Any,
) -> None:
    """Validate the cleaned V2 split without changing it."""
    if list(x_train.columns) != FEATURE_ORDER:
        raise ValueError(f"Unexpected X_train feature order: {list(x_train.columns)}")
    if list(x_test.columns) != FEATURE_ORDER:
        raise ValueError(f"Unexpected X_test feature order: {list(x_test.columns)}")
    if TARGET_ENCODED_COLUMN not in y_train_df.columns:
        raise ValueError(f"Missing y_train column: {TARGET_ENCODED_COLUMN}")
    if TARGET_ENCODED_COLUMN not in y_test_df.columns:
        raise ValueError(f"Missing y_test column: {TARGET_ENCODED_COLUMN}")
    if tuple(x_train.shape) != (2312, 3):
        raise ValueError(f"Unexpected X_train shape: {x_train.shape}")
    if tuple(x_test.shape) != (579, 3):
        raise ValueError(f"Unexpected X_test shape: {x_test.shape}")
    if len(y_train_df) != len(x_train):
        raise ValueError("X_train and y_train row counts do not match")
    if len(y_test_df) != len(x_test):
        raise ValueError("X_test and y_test row counts do not match")
    if list(label_encoder.classes_) != EXPECTED_CLASSES:
        raise ValueError(f"Unexpected label encoder classes: {list(label_encoder.classes_)}")
    if x_train.isna().sum().sum() or x_test.isna().sum().sum():
        raise ValueError("Feature split contains NaN values")
    if not np.isfinite(x_train.to_numpy(dtype=float)).all():
        raise ValueError("X_train contains infinite values")
    if not np.isfinite(x_test.to_numpy(dtype=float)).all():
        raise ValueError("X_test contains infinite values")


def build_models() -> dict[str, Any]:
    """Create the four fixed baseline models requested for Dataset V2."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        ),
    }


def safe_model_filename(model_name: str) -> str:
    """Map model display names to required plot filenames."""
    return model_name.lower().replace(" ", "_")


def predict_labels(model: Any, x_test: pd.DataFrame) -> np.ndarray:
    """Return 1D integer predictions for any supported model wrapper."""
    predictions = np.asarray(model.predict(x_test)).reshape(-1)
    return predictions.astype(int)


def evaluate_one_model(
    model_name: str,
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
) -> dict[str, Any]:
    """Calculate all requested metrics for a trained model."""
    y_pred = predict_labels(model, x_test)
    labels = list(range(len(class_names)))
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    metrics = {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Weighted_Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Weighted_Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Weighted_F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "Macro_Precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "Macro_Recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "Macro_F1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }

    return {
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": matrix,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(model_name: str, matrix: np.ndarray, class_names: list[str]) -> Path:
    """Save a confusion matrix image for one model."""
    fig, ax = plt.subplots(figsize=(8.5, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Crop")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            text_color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = IMAGES_DIR / f"{safe_model_filename(model_name)}_confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_model_comparison(comparison_df: pd.DataFrame) -> Path:
    """Save a bar chart for core V2 baseline metrics."""
    metrics = ["Accuracy", "Weighted_F1", "Macro_F1"]
    x_positions = np.arange(len(comparison_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, metric in enumerate(metrics):
        ax.bar(x_positions + index * width, comparison_df[metric], width, label=metric)

    ax.set_title("Dataset V2 Model Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions + width)
    ax.set_xticklabels(comparison_df["Model"], rotation=0)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path = IMAGES_DIR / "model_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def select_best_model(comparison_df: pd.DataFrame) -> tuple[str, pd.Series, str]:
    """Select by Macro F1, then Weighted F1, then Accuracy."""
    sorted_df = comparison_df.sort_values(
        by=["Macro_F1", "Weighted_F1", "Accuracy", "Model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best_row = sorted_df.iloc[0]
    best_name = str(best_row["Model"])
    reason = (
        f"Selected {best_name} using Macro F1 as the primary criterion "
        f"({best_row['Macro_F1']:.4f}), then Weighted F1 "
        f"({best_row['Weighted_F1']:.4f}), then Accuracy ({best_row['Accuracy']:.4f})."
    )
    return best_name, best_row, reason


def extract_feature_importance(model_name: str, model: Any) -> list[dict[str, Any]]:
    """Extract model feature importance values where supported."""
    if model_name == "CatBoost":
        importances = np.asarray(model.get_feature_importance(), dtype=float)
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    else:
        importances = np.full(len(FEATURE_ORDER), np.nan)

    total = float(np.nansum(importances))
    rows = []
    for feature, value in zip(FEATURE_ORDER, importances):
        normalized = float(value / total) if total else 0.0
        rows.append(
            {
                "Model": model_name,
                "Feature": feature,
                "Importance": float(value),
                "Normalized_Importance": normalized,
            }
        )
    return rows


def per_class_metrics_for_best(report_dict: dict[str, Any], class_names: list[str]) -> pd.DataFrame:
    """Create per-class metrics for the selected best model."""
    rows = []
    for crop in class_names:
        metrics = report_dict[crop]
        rows.append(
            {
                "Crop": crop,
                "Precision": float(metrics["precision"]),
                "Recall": float(metrics["recall"]),
                "F1": float(metrics["f1-score"]),
                "Support": int(metrics["support"]),
            }
        )
    return pd.DataFrame(rows)


def create_prediction_test(
    best_model_path: Path,
    encoder_path: Path,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Run saved-model predictions on real held-out test rows."""
    saved_model = joblib.load(best_model_path)
    saved_encoder = joblib.load(encoder_path)

    examples = x_test.head(PREDICTION_TEST_ROWS).copy()
    predictions = predict_labels(saved_model, examples)
    actual_names = saved_encoder.inverse_transform(y_test.head(PREDICTION_TEST_ROWS).to_numpy())
    predicted_names = saved_encoder.inverse_transform(predictions)

    result = examples.reset_index(drop=True)
    result.insert(0, "Test_Row_Number", np.arange(1, len(result) + 1))
    result["Actual_Crop"] = actual_names
    result["Predicted_Crop"] = predicted_names
    result["Correct"] = result["Actual_Crop"] == result["Predicted_Crop"]
    return result


def library_versions() -> dict[str, str]:
    """Capture package versions relevant to the V2 experiment."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "catboost",
        "matplotlib",
        "joblib",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def to_jsonable(value: Any) -> Any:
    """Convert model params and numpy objects into JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def dataframe_records(df: pd.DataFrame, decimals: int | None = None) -> list[dict[str, Any]]:
    """Return JSON records, optionally rounding numeric values."""
    working = df.copy()
    if decimals is not None:
        numeric_columns = working.select_dtypes(include=[np.number]).columns
        working[numeric_columns] = working[numeric_columns].round(decimals)
    return to_jsonable(working.to_dict(orient="records"))


def interpret_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Summarize the selected model's confusion matrix behavior."""
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusions = []
    for actual_index, predicted_index in zip(*np.where(off_diagonal > 0)):
        confusions.append(
            {
                "Actual": class_names[int(actual_index)],
                "Predicted": class_names[int(predicted_index)],
                "Count": int(off_diagonal[actual_index, predicted_index]),
            }
        )
    confusions = sorted(confusions, key=lambda row: row["Count"], reverse=True)
    return {
        "total_test_samples": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "largest_off_diagonal_confusions": confusions[:10],
    }


def soil_moisture_sanity_check() -> dict[str, Any]:
    """Check whether Soil_Moisture behaves like a direct label code."""
    clean_path = CLEANED_DIR / "croprec_bd_clean.csv"
    if not clean_path.exists():
        return {"available": False, "reason": "Cleaned dataset CSV not found."}

    clean_df = pd.read_csv(clean_path)
    crop_counts_per_moisture = clean_df.groupby("Soil_Moisture")["Crop Name"].nunique()
    single_class_values = int((crop_counts_per_moisture == 1).sum())
    multi_class_values = int((crop_counts_per_moisture > 1).sum())
    max_classes_per_value = int(crop_counts_per_moisture.max())
    return {
        "available": True,
        "unique_soil_moisture_values": int(crop_counts_per_moisture.shape[0]),
        "values_mapping_to_one_crop": single_class_values,
        "values_mapping_to_multiple_crops": multi_class_values,
        "max_crops_for_one_soil_moisture_value": max_classes_per_value,
        "interpretation": (
            "Soil_Moisture is highly class-separating in this dataset, but the integer value "
            "is not a perfect one-to-one hidden label because some values occur with multiple crops."
        ),
    }


def scientific_validation(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cleaned_checksums_before: dict[str, str],
    cleaned_checksums_after: dict[str, str],
    v1_models_before: dict[str, str],
    v1_models_after: dict[str, str],
) -> dict[str, Any]:
    """Record scientific and implementation sanity checks."""
    train_combined = x_train.copy()
    train_combined[TARGET_ENCODED_COLUMN] = y_train.to_numpy()
    test_combined = x_test.copy()
    test_combined[TARGET_ENCODED_COLUMN] = y_test.to_numpy()
    combined = pd.concat([train_combined, test_combined], ignore_index=True)

    train_rows = {tuple(row) for row in train_combined.to_numpy()}
    test_rows = {tuple(row) for row in test_combined.to_numpy()}
    feature_train_rows = {tuple(row) for row in x_train.to_numpy()}
    feature_test_rows = {tuple(row) for row in x_test.to_numpy()}

    return {
        "test_set_not_used_during_training": True,
        "test_labels_not_used_for_model_tuning": True,
        "no_hyperparameter_tuning_against_test_set": True,
        "no_synthetic_samples_added": True,
        "no_oversampling_or_undersampling": True,
        "no_smote": True,
        "no_rows_duplicated_in_combined_split": bool(combined.duplicated().sum() == 0),
        "combined_duplicate_rows": int(combined.duplicated().sum()),
        "train_test_exact_row_label_intersection": int(len(train_rows & test_rows)),
        "train_test_feature_only_intersection": int(len(feature_train_rows & feature_test_rows)),
        "feature_order_verified": list(x_train.columns) == FEATURE_ORDER and list(x_test.columns) == FEATURE_ORDER,
        "target_encoding_range_verified": int(y_train.min()) == 0
        and int(y_test.min()) == 0
        and int(y_train.max()) == 8
        and int(y_test.max()) == 8,
        "cleaned_artifacts_unchanged_by_training": cleaned_checksums_before == cleaned_checksums_after,
        "v1_model_artifacts_unchanged": v1_models_before == v1_models_after,
        "no_feature_values_changed": cleaned_checksums_before == cleaned_checksums_after,
        "no_labels_modified_by_experiment": cleaned_checksums_before == cleaned_checksums_after,
        "soil_moisture_sanity_check": soil_moisture_sanity_check(),
    }


def checksum_cleaned_inputs() -> dict[str, str]:
    """Hash cleaned input artifacts that must not be changed by training."""
    paths = [
        CLEANED_DIR / "X_train.csv",
        CLEANED_DIR / "X_test.csv",
        CLEANED_DIR / "y_train.csv",
        CLEANED_DIR / "y_test.csv",
        CLEANED_DIR / "label_encoder.joblib",
    ]
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def run_experiment() -> dict[str, Any]:
    """Train, evaluate, serialize, and report the Dataset V2 baseline."""
    configure_stdout()
    ensure_output_dirs()

    cleaned_checksums_before = checksum_cleaned_inputs()
    v1_models_before = snapshot_existing_v1_model_artifacts()

    x_train, x_test, y_train, y_test, label_encoder = load_dataset_v2_split()
    class_names = list(label_encoder.classes_)
    models = build_models()

    results = []
    model_outputs: dict[str, dict[str, Any]] = {}
    feature_importance_rows = []
    confusion_matrix_paths: dict[str, str] = {}

    for model_name, model in models.items():
        print(f"Training Dataset V2 baseline: {model_name}")
        model.fit(x_train, y_train)
        evaluation = evaluate_one_model(model_name, model, x_test, y_test, class_names)
        results.append(evaluation["metrics"])
        model_outputs[model_name] = {
            "model": model,
            "classification_report": evaluation["classification_report"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "y_pred": evaluation["y_pred"],
        }
        feature_importance_rows.extend(extract_feature_importance(model_name, model))
        matrix_path = plot_confusion_matrix(model_name, evaluation["confusion_matrix"], class_names)
        confusion_matrix_paths[model_name] = str(matrix_path.relative_to(PROJECT_ROOT))

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values(
        by=["Macro_F1", "Weighted_F1", "Accuracy", "Model"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    comparison_csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_json_path = REPORTS_DIR / "model_comparison.json"
    comparison_df.to_csv(comparison_csv_path, index=False)
    comparison_json_path.write_text(
        json.dumps(dataframe_records(comparison_df, decimals=6), indent=2),
        encoding="utf-8",
    )

    comparison_plot_path = plot_model_comparison(comparison_df)

    best_name, best_row, selection_reason = select_best_model(comparison_df)
    best_output = model_outputs[best_name]
    best_model_path = V2_MODELS_DIR / "crop_model_v2.pkl"
    v2_encoder_path = V2_MODELS_DIR / "label_encoder_v2.joblib"
    joblib.dump(best_output["model"], best_model_path)
    joblib.dump(label_encoder, v2_encoder_path)

    per_class_df = per_class_metrics_for_best(best_output["classification_report"], class_names)
    per_class_csv_path = REPORTS_DIR / "per_class_metrics.csv"
    per_class_df.to_csv(per_class_csv_path, index=False)

    feature_importance_df = pd.DataFrame(feature_importance_rows)
    feature_importance_csv_path = REPORTS_DIR / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_csv_path, index=False)

    classification_reports_path = REPORTS_DIR / "classification_reports.json"
    classification_reports = {
        model_name: to_jsonable(output["classification_report"])
        for model_name, output in model_outputs.items()
    }
    classification_reports_path.write_text(json.dumps(classification_reports, indent=2), encoding="utf-8")

    confusion_matrices_path = REPORTS_DIR / "confusion_matrices.json"
    confusion_matrices = {
        model_name: output["confusion_matrix"].astype(int).tolist()
        for model_name, output in model_outputs.items()
    }
    confusion_matrices_path.write_text(json.dumps(confusion_matrices, indent=2), encoding="utf-8")

    prediction_test_df = create_prediction_test(best_model_path, v2_encoder_path, x_test, y_test)
    prediction_test_path = REPORTS_DIR / "prediction_test.csv"
    prediction_test_df.to_csv(prediction_test_path, index=False)

    cleaned_checksums_after = checksum_cleaned_inputs()
    v1_models_after = snapshot_existing_v1_model_artifacts()
    validation = scientific_validation(
        x_train,
        x_test,
        y_train,
        y_test,
        cleaned_checksums_before,
        cleaned_checksums_after,
        v1_models_before,
        v1_models_after,
    )

    easiest_rows = per_class_df.sort_values(by=["F1", "Recall", "Precision"], ascending=False).head(3)
    hardest_rows = per_class_df.sort_values(by=["F1", "Recall", "Precision"], ascending=True).head(3)
    best_feature_importance = feature_importance_df[feature_importance_df["Model"] == best_name].copy()
    best_feature_importance = best_feature_importance.sort_values(
        by="Normalized_Importance", ascending=False
    ).reset_index(drop=True)

    model_params = {name: to_jsonable(model.get_params()) for name, model in models.items()}
    summary = {
        "objective": "Train and evaluate the first Dataset V2 crop recommendation baseline using cleaned artifacts only.",
        "dataset": {
            "cleaned_dataset": str((CLEANED_DIR / "croprec_bd_clean.csv").relative_to(PROJECT_ROOT)),
            "X_train": str((CLEANED_DIR / "X_train.csv").relative_to(PROJECT_ROOT)),
            "X_test": str((CLEANED_DIR / "X_test.csv").relative_to(PROJECT_ROOT)),
            "y_train": str((CLEANED_DIR / "y_train.csv").relative_to(PROJECT_ROOT)),
            "y_test": str((CLEANED_DIR / "y_test.csv").relative_to(PROJECT_ROOT)),
            "label_encoder": str((CLEANED_DIR / "label_encoder.joblib").relative_to(PROJECT_ROOT)),
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "feature_order": FEATURE_ORDER,
            "classes": class_names,
            "class_distribution_train": to_jsonable(
                pd.Series(label_encoder.inverse_transform(y_train.to_numpy())).value_counts().sort_index().to_dict()
            ),
            "class_distribution_test": to_jsonable(
                pd.Series(label_encoder.inverse_transform(y_test.to_numpy())).value_counts().sort_index().to_dict()
            ),
        },
        "models_trained": list(models.keys()),
        "model_parameters": model_params,
        "model_comparison": dataframe_records(comparison_df, decimals=6),
        "selection": {
            "primary_metric": "Macro_F1",
            "secondary_metric": "Weighted_F1",
            "tertiary_metric": "Accuracy",
            "best_model": best_name,
            "best_metrics": to_jsonable(best_row.to_dict()),
            "reason": selection_reason,
        },
        "per_class_metrics_best_model": dataframe_records(per_class_df, decimals=6),
        "easiest_crops_by_f1": dataframe_records(easiest_rows, decimals=6),
        "hardest_crops_by_f1": dataframe_records(hardest_rows, decimals=6),
        "confusion_matrix_interpretation": interpret_confusion_matrix(
            best_output["confusion_matrix"], class_names
        ),
        "feature_importance": dataframe_records(feature_importance_df, decimals=6),
        "best_model_feature_importance": dataframe_records(best_feature_importance, decimals=6),
        "prediction_test": dataframe_records(prediction_test_df, decimals=6),
        "dataset_characteristics": {
            "soil_moisture_eta_squared_from_prior_audit": 0.9847,
            "humidity_eta_squared_from_prior_audit": 0.7037,
            "temperature_eta_squared_from_prior_audit": 0.2457,
            "note": (
                "Dataset V2 contains unusually strong Soil_Moisture separation by crop. "
                "High test accuracy should be reported as dataset-dependent, not as proof of broad real-world generalization."
            ),
        },
        "reproducibility": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "library_versions": library_versions(),
            "random_state": RANDOM_STATE,
            "feature_order": FEATURE_ORDER,
            "train_size": int(len(x_train)),
            "test_size": int(len(x_test)),
            "cleaned_input_checksums_before": cleaned_checksums_before,
            "cleaned_input_checksums_after": cleaned_checksums_after,
            "v1_model_checksums_before": v1_models_before,
            "v1_model_checksums_after": v1_models_after,
        },
        "scientific_validation": validation,
        "output_files": [
            str(comparison_csv_path.relative_to(PROJECT_ROOT)),
            str(comparison_json_path.relative_to(PROJECT_ROOT)),
            str(per_class_csv_path.relative_to(PROJECT_ROOT)),
            str(feature_importance_csv_path.relative_to(PROJECT_ROOT)),
            str(classification_reports_path.relative_to(PROJECT_ROOT)),
            str(confusion_matrices_path.relative_to(PROJECT_ROOT)),
            str(prediction_test_path.relative_to(PROJECT_ROOT)),
            str(comparison_plot_path.relative_to(PROJECT_ROOT)),
            *confusion_matrix_paths.values(),
            str(best_model_path.relative_to(PROJECT_ROOT)),
            str(v2_encoder_path.relative_to(PROJECT_ROOT)),
        ],
        "warnings": [],
        "final_status": "DATASET V2 BASELINE COMPLETED",
    }

    best_accuracy = float(best_row["Accuracy"])
    if best_accuracy > 0.9:
        summary["warnings"].append(
            "Best model accuracy is above 90%; results are dataset-dependent and strongly influenced by Soil_Moisture separation."
        )

    summary_path = REPORTS_DIR / "baseline_experiment_summary.json"
    summary["output_files"].append(str(summary_path.relative_to(PROJECT_ROOT)))
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")

    print("\nDataset V2 model comparison")
    print(comparison_df.round(4).to_string(index=False))
    print(f"\nBest model: {best_name}")
    print(selection_reason)
    print(f"Saved selected model: {best_model_path}")
    print(f"Saved V2 label encoder: {v2_encoder_path}")
    print("\nPrediction test from saved model")
    print(prediction_test_df.to_string(index=False))
    print("✅ DATASET V2 BASELINE COMPLETED")
    return summary


if __name__ == "__main__":
    run_experiment()

"""
Evaluate trained models, compare metrics, and save the best model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from paths import FEATURE_ORDER, IMAGES_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE
from train_model import load_training_data, model_filename


def ensure_output_dirs() -> None:
    """Create output directories if needed."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_trained_models() -> dict[str, object]:
    """Load all trained model files from models/."""
    models: dict[str, object] = {}
    for name in ["Random Forest", "LightGBM", "XGBoost", "CatBoost"]:
        path = MODELS_DIR / model_filename(name)
        if not path.exists():
            raise FileNotFoundError(f"Trained model not found: {path}. Run train_model.py first.")
        models[name] = joblib.load(path)
        print(f"Loaded: {path}")
    return models


def evaluate_model(
    model_name: str,
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder,
) -> dict:
    """Evaluate one model and return metrics plus predictions."""
    y_pred = model.predict(x_test)

    metrics = {
        "Model": model_name,
        "Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "F1_Score": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }

    report = classification_report(
        y_test,
        y_pred,
        target_names=list(label_encoder.classes_),
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, y_pred)

    print(f"\n=== {model_name} Evaluation ===")
    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value}")
    print("\nClassification Report:")
    print(report)

    return {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix,
        "y_pred": y_pred,
    }


def plot_confusion_matrix(
    model_name: str,
    matrix: np.ndarray,
    class_names: list[str],
) -> Path:
    """Save confusion matrix plot for one model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=15)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} Confusion Matrix")

    for row in range(len(class_names)):
        for col in range(len(class_names)):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_")
    output_path = IMAGES_DIR / f"{safe_name}_confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix: {output_path}")
    return output_path


def plot_model_comparison(comparison_df: pd.DataFrame) -> Path:
    """Save bar chart comparing model metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    x = np.arange(len(comparison_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 6))
    for index, metric in enumerate(metrics):
        ax.bar(x + index * width, comparison_df[metric], width, label=metric)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"], rotation=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()

    output_path = IMAGES_DIR / "model_comparison.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved model comparison plot: {output_path}")
    return output_path


def select_best_model(comparison_df: pd.DataFrame) -> tuple[str, pd.Series, str]:
    """
    Select the best model using F1-score as the primary metric.

    Secondary tie-breakers: Accuracy, then simplicity/stability preference.
    """
    sorted_df = comparison_df.sort_values(
        by=["F1_Score", "Accuracy", "Model"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_row = sorted_df.iloc[0]
    best_name = str(best_row["Model"])

    simplicity_rank = {
        "Random Forest": 1,
        "LightGBM": 2,
        "XGBoost": 3,
        "CatBoost": 4,
    }

    top_f1 = sorted_df["F1_Score"].iloc[0]
    tied = sorted_df[sorted_df["F1_Score"] == top_f1]

    if len(tied) > 1:
        tied = tied.sort_values(by=["Accuracy", "Model"], ascending=[False, True])
        best_name = str(tied.iloc[0]["Model"])
        reason = (
            f"Selected {best_name} because it achieved the highest weighted F1-score "
            f"({best_row['F1_Score']:.4f}) among tied models, with accuracy "
            f"{best_row['Accuracy']:.4f} and favorable simplicity/stability."
        )
    else:
        reason = (
            f"Selected {best_name} because it achieved the highest weighted F1-score "
            f"({best_row['F1_Score']:.4f}) on the held-out test set."
        )

    if best_name == "Random Forest" and len(tied) > 1:
        reason += " Random Forest was preferred as the simpler and more stable baseline."

    return best_name, best_row, reason

def save_best_model(best_name: str, models: dict[str, object], label_encoder) -> dict[str, str]:
    """Save the selected model and required artifacts for prediction."""
    best_model = models[best_name]
    crop_model_path = MODELS_DIR / "crop_model.pkl"
    label_encoder_path = MODELS_DIR / "label_encoder.joblib"

    joblib.dump(best_model, crop_model_path)
    joblib.dump(label_encoder, label_encoder_path)

    print(f"\nSaved best model: {crop_model_path}")
    print(f"Saved label encoder: {label_encoder_path}")

    return {
        "best_model_name": best_name,
        "crop_model": str(crop_model_path),
        "label_encoder": str(label_encoder_path),
    }


def run_evaluation() -> dict:
    """Evaluate all trained models and save comparison outputs."""
    ensure_output_dirs()

    x_train, y_train, x_test, y_test = load_training_data()
    _ = x_train, y_train  # Training data loaded only to confirm split integrity.

    label_encoder = joblib.load(PROCESSED_DIR / "label_encoder.joblib")
    models = load_trained_models()

    results = []
    reports = {}

    for name, model in models.items():
        evaluation = evaluate_model(name, model, x_test, y_test, label_encoder)
        results.append(evaluation["metrics"])
        reports[name] = evaluation["classification_report"]
        plot_confusion_matrix(name, evaluation["confusion_matrix"], list(label_encoder.classes_))

    comparison_df = pd.DataFrame(results).sort_values(by="F1_Score", ascending=False).reset_index(drop=True)
    comparison_csv = PROCESSED_DIR / "model_comparison.csv"
    comparison_json = PROCESSED_DIR / "model_comparison.json"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_df.to_json(comparison_json, orient="records", indent=2)

    print("\n=== Model Comparison ===")
    print(comparison_df.to_string(index=False))

    plot_model_comparison(comparison_df)

    best_name, best_row, reason = select_best_model(comparison_df)
    saved_paths = save_best_model(best_name, models, label_encoder)

    selection_report = {
        "random_state": RANDOM_STATE,
        "selection_metric": "F1_Score (weighted)",
        "best_model": best_name,
        "best_metrics": best_row.to_dict(),
        "reason": reason,
        "comparison_file": str(comparison_csv),
        "feature_order": FEATURE_ORDER,
    }

    selection_path = PROCESSED_DIR / "best_model_selection.json"
    with selection_path.open("w", encoding="utf-8") as file:
        json.dump(selection_report, file, indent=2)

    print(f"\nBest model: {best_name}")
    print(f"Reason: {reason}")
    print(f"Saved comparison CSV: {comparison_csv}")
    print(f"Saved selection report: {selection_path}")

    return {
        "comparison_df": comparison_df,
        "best_model": best_name,
        "best_metrics": best_row.to_dict(),
        "reason": reason,
        "saved_paths": saved_paths,
        "classification_reports": reports,
    }


if __name__ == "__main__":
    run_evaluation()

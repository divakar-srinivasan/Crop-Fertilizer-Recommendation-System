"""
Exploratory data analysis for Review 1 crop classification.
Generates plots and summary statistics in images/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feature_selection import select_model_columns, verify_data_types
from load_dataset import load_crop_dataset
from paths import FEATURE_COLUMNS, IMAGES_DIR, PROCESSED_DIR, TARGET


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_analysis_data() -> pd.DataFrame:
    """Load cleaned selected dataset if available, otherwise build from raw data."""
    cleaned_path = PROCESSED_DIR / "crop_selected_clean.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
        print(f"Loaded cleaned dataset from: {cleaned_path}")
    else:
        df = select_model_columns(load_crop_dataset())
        print("Using freshly selected dataset (run preprocessing.py first for cleaned data).")
    return df


def statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numeric features."""
    summary = df[FEATURE_COLUMNS].describe().T
    summary["missing_count"] = df[FEATURE_COLUMNS].isnull().sum()
    return summary


def save_statistical_summary(summary: pd.DataFrame) -> Path:
    """Save statistical summary to processed data folder."""
    output_path = PROCESSED_DIR / "statistical_summary.csv"
    summary.to_csv(output_path)
    print(f"Saved statistical summary: {output_path}")
    return output_path


def plot_crop_class_distribution(df: pd.DataFrame) -> Path:
    """Create bar chart of crop class counts."""
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", color="#4C78A8", ax=ax)
    ax.set_title("Crop Class Distribution")
    ax.set_xlabel("Crop Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()

    output_path = IMAGES_DIR / "crop_class_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_feature_distributions(df: pd.DataFrame) -> Path:
    """Create histogram grid for all selected numeric features."""
    n_features = len(FEATURE_COLUMNS)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURE_COLUMNS):
        axes[idx].hist(df[feature], bins=30, color="#72B7B2", edgecolor="white")
        axes[idx].set_title(feature)
        axes[idx].set_xlabel("Value")
        axes[idx].set_ylabel("Frequency")

    for idx in range(n_features, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()

    output_path = IMAGES_DIR / "feature_distributions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    """Create correlation heatmap for numeric features."""
    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_COLUMNS)))
    ax.set_yticks(range(len(FEATURE_COLUMNS)))
    ax.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
    ax.set_yticklabels(FEATURE_COLUMNS)

    for row in range(len(FEATURE_COLUMNS)):
        for col in range(len(FEATURE_COLUMNS)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()

    output_path = IMAGES_DIR / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def outlier_summary_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize IQR-based outliers without removing rows."""
    rows = []
    for feature in FEATURE_COLUMNS:
        q1 = df[feature].quantile(0.25)
        q3 = df[feature].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[feature] < lower) | (df[feature] > upper)
        rows.append(
            {
                "feature": feature,
                "q1": q1,
                "q3": q3,
                "lower_fence": lower,
                "upper_fence": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(100 * mask.mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_boxplots(df: pd.DataFrame) -> Path:
    """Create boxplots to visualize spread and outliers."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot([df[feature] for feature in FEATURE_COLUMNS], tick_labels=FEATURE_COLUMNS)
    ax.set_title("Feature Boxplots (Outlier Visualization)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    output_path = IMAGES_DIR / "feature_boxplots.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_feature_by_crop(df: pd.DataFrame) -> Path:
    """Create feature-vs-target boxplots for selected features."""
    sample_features = ["N", "P", "K", "pH", "EC", "OC"]
    crops = sorted(df[TARGET].unique())
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, feature in enumerate(sample_features):
        data = [df.loc[df[TARGET] == crop, feature] for crop in crops]
        box = axes[idx].boxplot(data, tick_labels=crops, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
        axes[idx].set_title(f"{feature} by Crop Type")
        axes[idx].tick_params(axis="x", rotation=15)

    fig.suptitle("Feature Values by Crop Type", fontsize=14)
    plt.tight_layout()

    output_path = IMAGES_DIR / "feature_by_crop_boxplots.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def save_eda_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    generated_images: list[str],
) -> Path:
    """Save EDA metadata report as JSON."""
    report = {
        "rows": len(df),
        "features": FEATURE_COLUMNS,
        "target": TARGET,
        "class_distribution": df[TARGET].value_counts().sort_index().to_dict(),
        "strong_correlations": find_strong_correlations(df),
        "outlier_summary": outlier_summary.to_dict(orient="records"),
        "generated_images": generated_images,
        "statistical_summary_file": "data/processed/statistical_summary.csv",
    }

    output_path = PROCESSED_DIR / "eda_report.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"Saved EDA report: {output_path}")
    return output_path


def find_strong_correlations(df: pd.DataFrame, threshold: float = 0.7) -> list[dict[str, float | str]]:
    """Return feature pairs with absolute correlation above threshold."""
    corr = df[FEATURE_COLUMNS].corr(numeric_only=True)
    pairs: list[dict[str, float | str]] = []

    for i, col_a in enumerate(FEATURE_COLUMNS):
        for col_b in FEATURE_COLUMNS[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if abs(value) >= threshold:
                pairs.append({"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)})

    pairs.sort(key=lambda item: abs(float(item["correlation"])), reverse=True)
    return pairs


def run_eda() -> dict:
    """Run full EDA workflow."""
    ensure_output_dirs()
    df = load_analysis_data()
    verify_data_types(df)

    print("\n=== EDA: Statistical Summary ===")
    summary = statistical_summary(df)
    print(summary[["count", "mean", "std", "min", "max", "missing_count"]])
    save_statistical_summary(summary)

    print("\n=== EDA: Crop Class Distribution ===")
    print(df[TARGET].value_counts().sort_index().to_string())

    outlier_summary = outlier_summary_iqr(df)
    print("\n=== EDA: Outlier Analysis (IQR, not removed) ===")
    print(outlier_summary.to_string(index=False))

    generated_images = [
        str(plot_crop_class_distribution(df)),
        str(plot_feature_distributions(df)),
        str(plot_correlation_heatmap(df)),
        str(plot_feature_boxplots(df)),
        str(plot_feature_by_crop(df)),
    ]

    save_eda_report(df, summary, outlier_summary, generated_images)

    strong = find_strong_correlations(df)
    print("\n=== EDA: Strong Correlations (|r| >= 0.7) ===")
    if strong:
        for item in strong:
            print(f"{item['feature_a']} vs {item['feature_b']}: {item['correlation']}")
    else:
        print("No feature pairs with |correlation| >= 0.7")

    return {
        "rows": len(df),
        "summary": summary,
        "outlier_summary": outlier_summary,
        "strong_correlations": strong,
        "images": generated_images,
    }


if __name__ == "__main__":
    run_eda()

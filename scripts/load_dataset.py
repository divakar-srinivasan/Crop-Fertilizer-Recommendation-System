"""
Load crop and fertilizer datasets for Review 1.
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CROP_DATASET_PATH = DATA_DIR / "crop_dataset.csv"
FERTILIZER_DATASET_PATH = DATA_DIR / "fertilizer_dataset.csv"

# Review 1 feature columns (soil parameters only)
REQUIRED_FEATURES = ["N", "P", "K", "pH", "EC", "OC", "S", "Zn", "Fe", "Cu", "Mn", "B"]
TARGET_COLUMN = "Crop_Type"


def load_crop_dataset() -> pd.DataFrame:
    """Load the crop recommendation dataset from CSV."""
    if not CROP_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Crop dataset not found at: {CROP_DATASET_PATH}\n"
            "Place crop_dataset.csv inside the data/ folder."
        )

    df = pd.read_csv(CROP_DATASET_PATH)
    print(f"Crop dataset loaded successfully from: {CROP_DATASET_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def load_fertilizer_dataset() -> pd.DataFrame:
    """Load the fertilizer mapping dataset from CSV."""
    if not FERTILIZER_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Fertilizer dataset not found at: {FERTILIZER_DATASET_PATH}\n"
            "Place fertilizer_dataset.csv inside the data/ folder."
        )

    df = pd.read_csv(FERTILIZER_DATASET_PATH)
    print(f"Fertilizer dataset loaded successfully from: {FERTILIZER_DATASET_PATH}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def show_dataset_preview(df: pd.DataFrame, name: str, rows: int = 5) -> None:
    """Print basic preview information for a dataset."""
    print(f"\n--- {name} Preview (first {rows} rows) ---")
    print(df.head(rows))
    print(f"\nColumn names ({len(df.columns)}):")
    print(list(df.columns))


if __name__ == "__main__":
    crop_df = load_crop_dataset()
    show_dataset_preview(crop_df, "Crop Dataset")

    fertilizer_df = load_fertilizer_dataset()
    show_dataset_preview(fertilizer_df, "Fertilizer Dataset")

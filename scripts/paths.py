"""Shared project paths and constants for Review 1 scripts."""

from pathlib import Path

from load_dataset import REQUIRED_FEATURES, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = PROJECT_ROOT / "images"
MODELS_DIR = PROJECT_ROOT / "models"

# Explicit feature order for training and prediction (do not rely on dict ordering).
FEATURE_ORDER = [
    "N",
    "P",
    "K",
    "pH",
    "EC",
    "OC",
    "S",
    "Zn",
    "Fe",
    "Cu",
    "Mn",
    "B",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLUMNS = REQUIRED_FEATURES
TARGET = TARGET_COLUMN

# Reasonable numeric bounds for soil-quality checks (reporting only unless clearly invalid).
FEATURE_BOUNDS = {
    "N": (0.0, None),
    "P": (0.0, None),
    "K": (0.0, None),
    "pH": (0.0, 14.0),
    "EC": (0.0, None),
    "OC": (0.0, None),
    "S": (0.0, None),
    "Zn": (0.0, None),
    "Fe": (0.0, None),
    "Cu": (0.0, None),
    "Mn": (0.0, None),
    "B": (0.0, None),
}

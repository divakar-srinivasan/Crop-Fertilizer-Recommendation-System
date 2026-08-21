"""
Train Review 1 baseline models using Level 2 processed data.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from paths import FEATURE_ORDER, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE


def ensure_models_dir() -> None:
    """Create models directory if needed."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed train/test splits from Level 2."""
    x_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    x_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")["Crop_Type_Encoded"]
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")["Crop_Type_Encoded"]

    x_train = x_train[FEATURE_ORDER]
    x_test = x_test[FEATURE_ORDER]

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")
    print(f"Feature count: {len(FEATURE_ORDER)}")
    return x_train, y_train, x_test, y_test


def build_models() -> dict[str, object]:
    """Create baseline model instances with beginner-friendly settings."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
            verbosity=0,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=100,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }


def model_filename(model_name: str) -> str:
    """Convert model display name to a safe filename."""
    mapping = {
        "Random Forest": "random_forest.joblib",
        "LightGBM": "lightgbm.joblib",
        "XGBoost": "xgboost.joblib",
        "CatBoost": "catboost.joblib",
    }
    return mapping[model_name]


def train_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """Train all baseline models on the training split only."""
    ensure_models_dir()
    trained_models: dict[str, object] = {}

    for name, model in build_models().items():
        print(f"\nTraining {name}...")
        model.fit(x_train, y_train)
        output_path = MODELS_DIR / model_filename(name)
        joblib.dump(model, output_path)
        print(f"Saved: {output_path}")
        trained_models[name] = model

    return trained_models


def run_training() -> dict[str, object]:
    """Execute model training workflow."""
    x_train, y_train, _, _ = load_training_data()
    return train_models(x_train, y_train)


if __name__ == "__main__":
    run_training()

"""
Prediction utilities for the Review 1 crop recommendation prototype.

This module loads the saved Level 3 model artifacts and predicts one crop
from the 12 soil parameters selected during Review 1 Level 2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from paths import FEATURE_BOUNDS, FEATURE_ORDER, MODELS_DIR


MODEL_PATH = MODELS_DIR / "crop_model.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"


class PredictionInputError(ValueError):
    """Raised when prediction input fails validation."""


def load_prediction_artifacts(
    model_path: Path = MODEL_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
) -> tuple[object, object]:
    """Load the saved model and label encoder required for prediction."""
    if not model_path.exists():
        raise FileNotFoundError(f"Saved crop model not found: {model_path}. Run evaluate_model.py first.")

    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Saved label encoder not found: {label_encoder_path}. Run evaluate_model.py first.")

    return joblib.load(model_path), joblib.load(label_encoder_path)


def validate_soil_inputs(values: Mapping[str, Any]) -> dict[str, float]:
    """Validate and convert the 12 required soil inputs."""
    missing = [feature for feature in FEATURE_ORDER if feature not in values]
    if missing:
        raise PredictionInputError(f"Missing required input(s): {', '.join(missing)}")

    cleaned: dict[str, float] = {}
    errors: list[str] = []

    for feature in FEATURE_ORDER:
        raw_value = values[feature]
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{feature} must be numeric.")
            continue

        if math.isnan(numeric_value):
            errors.append(f"{feature} cannot be NaN.")
            continue

        if math.isinf(numeric_value):
            errors.append(f"{feature} cannot be infinite.")
            continue

        min_bound, max_bound = FEATURE_BOUNDS.get(feature, (None, None))
        if min_bound is not None and numeric_value < min_bound:
            errors.append(f"{feature} must be greater than or equal to {min_bound:g}.")

        if max_bound is not None and numeric_value > max_bound:
            errors.append(f"{feature} must be less than or equal to {max_bound:g}.")

        cleaned[feature] = numeric_value

    if errors:
        raise PredictionInputError("Invalid prediction input:\n- " + "\n- ".join(errors))

    return cleaned


def create_feature_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """Create a one-row DataFrame in the exact model feature order."""
    cleaned = validate_soil_inputs(values)
    return pd.DataFrame([[cleaned[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)


def predict_crop(
    values: Mapping[str, Any],
    model: object | None = None,
    label_encoder: object | None = None,
) -> str:
    """Predict and return the crop name for one set of soil inputs."""
    if model is None or label_encoder is None:
        model, label_encoder = load_prediction_artifacts()

    features = create_feature_frame(values)
    encoded_prediction = np.asarray(model.predict(features)).reshape(-1)
    crop_name = label_encoder.inverse_transform(encoded_prediction.astype(int))[0]
    return str(crop_name)


def parse_args() -> argparse.Namespace:
    """Parse command-line soil parameter inputs."""
    parser = argparse.ArgumentParser(description="Predict crop type from 12 soil parameters.")
    for feature in FEATURE_ORDER:
        parser.add_argument(f"--{feature}", required=True, help=f"Value for {feature}")
    return parser.parse_args()


def main() -> None:
    """Run a single command-line prediction."""
    args = parse_args()
    values = {feature: vars(args)[feature] for feature in FEATURE_ORDER}

    try:
        prediction = predict_crop(values)
    except (FileNotFoundError, PredictionInputError) as exc:
        raise SystemExit(str(exc)) from exc

    print("Input values:")
    for feature in FEATURE_ORDER:
        print(f"{feature}: {float(values[feature]):.6g}")
    print(f"\nPredicted Crop: {prediction}")


if __name__ == "__main__":
    main()

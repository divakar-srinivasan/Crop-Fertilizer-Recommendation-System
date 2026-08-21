"""
Confidence-aware prediction utilities for Dataset V2.

This module loads the locked Dataset V2 CatBoost model, validates the three
required inputs, predicts the crop, and returns class probabilities from
model.predict_proba().
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_MODELS_DIR = PROJECT_ROOT / "models" / "dataset_v2"
MODEL_PATH = V2_MODELS_DIR / "crop_model_v2.pkl"
LABEL_ENCODER_PATH = V2_MODELS_DIR / "label_encoder_v2.joblib"
FEATURE_ORDER = ["Soil_Moisture", "Humidity", "Temperature"]


class ConfidencePredictionInputError(ValueError):
    """Raised when Dataset V2 confidence prediction input is invalid."""


@dataclass(frozen=True)
class ConfidencePredictionResult:
    """Structured result for one Dataset V2 confidence-aware prediction."""

    predicted_crop: str
    confidence: float
    probabilities: pd.DataFrame


def load_v2_prediction_artifacts(
    model_path: Path = MODEL_PATH,
    label_encoder_path: Path = LABEL_ENCODER_PATH,
) -> tuple[object, object]:
    """Load the locked Dataset V2 model and label encoder."""
    if not model_path.exists():
        raise FileNotFoundError(f"Saved Dataset V2 model not found: {model_path}")

    if not label_encoder_path.exists():
        raise FileNotFoundError(f"Saved Dataset V2 label encoder not found: {label_encoder_path}")

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    model_name = model.__class__.__name__
    if model_name != "CatBoostClassifier":
        raise TypeError(f"Expected CatBoostClassifier for Dataset V2, found {model_name}")

    model_feature_names = list(getattr(model, "feature_names_", []) or [])
    if model_feature_names and model_feature_names != FEATURE_ORDER:
        raise ValueError(
            f"Dataset V2 model feature order mismatch. Expected {FEATURE_ORDER}, "
            f"found {model_feature_names}"
        )

    return model, label_encoder


def validate_v2_inputs(values: Mapping[str, Any]) -> dict[str, float]:
    """Validate and convert the three Dataset V2 prediction inputs."""
    missing = [feature for feature in FEATURE_ORDER if feature not in values]
    if missing:
        raise ConfidencePredictionInputError(f"Missing required input(s): {', '.join(missing)}")

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

        cleaned[feature] = numeric_value

    if errors:
        raise ConfidencePredictionInputError("Invalid Dataset V2 input:\n- " + "\n- ".join(errors))

    return cleaned


def create_v2_feature_frame(values: Mapping[str, Any]) -> pd.DataFrame:
    """Create a one-row DataFrame in the locked Dataset V2 feature order."""
    cleaned = validate_v2_inputs(values)
    return pd.DataFrame([[cleaned[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)


def probability_table(probabilities: np.ndarray, label_encoder: object) -> pd.DataFrame:
    """Convert a probability vector into a crop/probability table."""
    classes = list(label_encoder.classes_)
    if len(probabilities) != len(classes):
        raise ValueError(
            f"Probability vector length mismatch. Expected {len(classes)}, found {len(probabilities)}"
        )

    table = pd.DataFrame(
        {
            "Crop": classes,
            "Probability": probabilities.astype(float),
        }
    )
    table["Probability_Percent"] = table["Probability"] * 100.0
    return table.sort_values(by="Probability", ascending=False).reset_index(drop=True)


def predict_crop_with_confidence(
    values: Mapping[str, Any],
    model: object | None = None,
    label_encoder: object | None = None,
) -> ConfidencePredictionResult:
    """Predict crop, confidence, and all class probabilities for Dataset V2."""
    if model is None or label_encoder is None:
        model, label_encoder = load_v2_prediction_artifacts()

    features = create_v2_feature_frame(values)
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != 1:
        raise ValueError(f"Unexpected predict_proba output shape: {probabilities.shape}")

    class_probabilities = probabilities[0]
    predicted_index = int(np.argmax(class_probabilities))
    predicted_crop = str(label_encoder.inverse_transform([predicted_index])[0])
    confidence = float(class_probabilities[predicted_index])

    return ConfidencePredictionResult(
        predicted_crop=predicted_crop,
        confidence=confidence,
        probabilities=probability_table(class_probabilities, label_encoder),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line Dataset V2 prediction inputs."""
    parser = argparse.ArgumentParser(description="Predict Dataset V2 crop with confidence.")
    for feature in FEATURE_ORDER:
        parser.add_argument(f"--{feature}", required=True, help=f"Value for {feature}")
    return parser.parse_args()


def main() -> None:
    """Run a single command-line confidence-aware prediction."""
    args = parse_args()
    values = {feature: vars(args)[feature] for feature in FEATURE_ORDER}
    result = predict_crop_with_confidence(values)

    print(f"Predicted Crop: {result.predicted_crop}")
    print(f"Confidence: {result.confidence:.2%}")
    print("\nCrop Probability Distribution:")
    display = result.probabilities.copy()
    display["Probability_Percent"] = display["Probability_Percent"].map(lambda value: f"{value:.2f}%")
    print(display[["Crop", "Probability_Percent"]].to_string(index=False))


if __name__ == "__main__":
    main()

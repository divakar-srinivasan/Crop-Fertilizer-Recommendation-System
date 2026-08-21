"""
Local SHAP explanations for Dataset V2 crop predictions.

The explanation uses the locked Dataset V2 CatBoost model and SHAP TreeExplainer.
It does not train or modify any model artifact.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
import shap

try:
    from confidence_prediction import (
        FEATURE_ORDER,
        create_v2_feature_frame,
        load_v2_prediction_artifacts,
        predict_crop_with_confidence,
    )
except ModuleNotFoundError:
    from scripts.confidence_prediction import (
        FEATURE_ORDER,
        create_v2_feature_frame,
        load_v2_prediction_artifacts,
        predict_crop_with_confidence,
    )


class ShapExplanationError(ValueError):
    """Raised when SHAP explanation output is not usable."""


@dataclass(frozen=True)
class ShapExplanationResult:
    """Structured local SHAP explanation for one Dataset V2 prediction."""

    predicted_crop: str
    contributions: pd.DataFrame


def create_v2_shap_explainer(model: object) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the locked CatBoost model."""
    return shap.TreeExplainer(model)


def predicted_class_index(predicted_crop: str, label_encoder: object) -> int:
    """Return the encoded class index for a predicted crop name."""
    matches = np.where(label_encoder.classes_ == predicted_crop)[0]
    if len(matches) != 1:
        raise ShapExplanationError(f"Predicted crop not found in label encoder: {predicted_crop}")
    return int(matches[0])


def extract_predicted_class_contributions(
    shap_values: object,
    predicted_index: int,
) -> np.ndarray:
    """Extract one contribution per feature for the predicted class."""
    if isinstance(shap_values, list):
        if predicted_index >= len(shap_values):
            raise ShapExplanationError("Predicted class index is outside SHAP value list.")
        contributions = np.asarray(shap_values[predicted_index], dtype=float)
        if contributions.ndim != 2 or contributions.shape[0] != 1:
            raise ShapExplanationError(f"Unexpected SHAP value shape: {contributions.shape}")
        return contributions[0]

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3 and values.shape[0] == 1 and values.shape[1] == len(FEATURE_ORDER):
        if predicted_index >= values.shape[2]:
            raise ShapExplanationError("Predicted class index is outside SHAP value array.")
        return values[0, :, predicted_index]

    if values.ndim == 2 and values.shape[0] == 1 and values.shape[1] == len(FEATURE_ORDER):
        return values[0]

    raise ShapExplanationError(f"Unexpected SHAP value shape: {values.shape}")


def build_contribution_table(contributions: np.ndarray) -> pd.DataFrame:
    """Create a simple feature/contribution table."""
    if len(contributions) != len(FEATURE_ORDER):
        raise ShapExplanationError(
            f"Expected {len(FEATURE_ORDER)} SHAP contributions, found {len(contributions)}"
        )

    if not np.isfinite(contributions).all():
        raise ShapExplanationError("SHAP produced NaN or infinite contribution values.")

    table = pd.DataFrame(
        {
            "Feature": FEATURE_ORDER,
            "Contribution": contributions.astype(float),
        }
    )
    table["Abs_Contribution"] = table["Contribution"].abs()
    return table.sort_values(by="Abs_Contribution", ascending=False).reset_index(drop=True)


def explain_v2_prediction(
    values: Mapping[str, Any],
    model: object,
    label_encoder: object,
    predicted_crop: str,
    explainer: shap.TreeExplainer | None = None,
) -> ShapExplanationResult:
    """Generate a local SHAP explanation for the predicted crop."""
    if explainer is None:
        explainer = create_v2_shap_explainer(model)

    features = create_v2_feature_frame(values)
    shap_values = explainer.shap_values(features)
    predicted_index = predicted_class_index(predicted_crop, label_encoder)
    contributions = extract_predicted_class_contributions(shap_values, predicted_index)

    return ShapExplanationResult(
        predicted_crop=predicted_crop,
        contributions=build_contribution_table(contributions),
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line Dataset V2 explanation inputs."""
    parser = argparse.ArgumentParser(description="Explain Dataset V2 crop prediction using SHAP.")
    for feature in FEATURE_ORDER:
        parser.add_argument(f"--{feature}", required=True, help=f"Value for {feature}")
    return parser.parse_args()


def main() -> None:
    """Run one local SHAP explanation from the command line."""
    args = parse_args()
    values = {feature: vars(args)[feature] for feature in FEATURE_ORDER}
    model, label_encoder = load_v2_prediction_artifacts()
    prediction = predict_crop_with_confidence(values, model=model, label_encoder=label_encoder)
    explanation = explain_v2_prediction(
        values,
        model=model,
        label_encoder=label_encoder,
        predicted_crop=prediction.predicted_crop,
    )

    print(f"Predicted Crop: {prediction.predicted_crop}")
    print(f"Confidence: {prediction.confidence:.2%}")
    print("\nPrediction Explanation")
    display = explanation.contributions[["Feature", "Contribution"]].copy()
    display["Contribution"] = display["Contribution"].map(lambda value: f"{value:+.6f}")
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()

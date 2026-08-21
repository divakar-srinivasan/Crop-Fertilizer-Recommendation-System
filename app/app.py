"""Streamlit app for crop recommendation prototypes."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from confidence_prediction import (  # noqa: E402
    FEATURE_ORDER as V2_FEATURE_ORDER,
    ConfidencePredictionInputError,
    load_v2_prediction_artifacts,
    predict_crop_with_confidence,
)
from explain_prediction import (  # noqa: E402
    ShapExplanationError,
    create_v2_shap_explainer,
    explain_v2_prediction,
)
from paths import FEATURE_ORDER as REVIEW1_FEATURE_ORDER  # noqa: E402
from predict import PredictionInputError, load_prediction_artifacts, predict_crop  # noqa: E402


REVIEW1_FEATURE_LABELS = {
    "N": "Nitrogen (N)",
    "P": "Phosphorus (P)",
    "K": "Potassium (K)",
    "pH": "pH",
    "EC": "Electrical Conductivity (EC)",
    "OC": "Organic Carbon (OC)",
    "S": "Sulfur (S)",
    "Zn": "Zinc (Zn)",
    "Fe": "Iron (Fe)",
    "Cu": "Copper (Cu)",
    "Mn": "Manganese (Mn)",
    "B": "Boron (B)",
}

REVIEW1_DEFAULT_VALUES = {
    "N": 98.7453,
    "P": 5.2732,
    "K": 28.1108,
    "pH": 6.9376,
    "EC": 0.8918,
    "OC": 0.2680,
    "S": 36.7357,
    "Zn": 1.2065,
    "Fe": 39.5567,
    "Cu": 0.2825,
    "Mn": 23.0873,
    "B": 2.2658,
}

V2_FEATURE_LABELS = {
    "Soil_Moisture": "Soil Moisture",
    "Humidity": "Humidity",
    "Temperature": "Temperature",
}

V2_DEFAULT_VALUES = {
    "Soil_Moisture": 44.0,
    "Humidity": 55.56,
    "Temperature": 29.58,
}


@st.cache_resource
def cached_review1_prediction_artifacts() -> tuple[object, object]:
    """Load Review 1 model artifacts once per Streamlit session."""
    return load_prediction_artifacts()


@st.cache_resource
def cached_v2_prediction_artifacts() -> tuple[object, object]:
    """Load Dataset V2 model artifacts once per Streamlit session."""
    return load_v2_prediction_artifacts()


@st.cache_resource
def cached_v2_shap_explainer() -> object:
    """Create the Dataset V2 SHAP explainer once per Streamlit session."""
    model, _ = load_v2_prediction_artifacts()
    return create_v2_shap_explainer(model)


def render_review1_inputs() -> dict[str, float]:
    """Render the 12 soil inputs and return their values."""
    values: dict[str, float] = {}
    columns = st.columns(3)

    for index, feature in enumerate(REVIEW1_FEATURE_ORDER):
        with columns[index % 3]:
            max_value = 14.0 if feature == "pH" else None
            values[feature] = st.number_input(
                REVIEW1_FEATURE_LABELS[feature],
                min_value=0.0,
                max_value=max_value,
                value=float(REVIEW1_DEFAULT_VALUES[feature]),
                step=0.1,
                format="%.4f",
                key=f"review1_{feature}",
            )

    return values


def render_v2_inputs() -> dict[str, float]:
    """Render the three Dataset V2 inputs and return their values."""
    values: dict[str, float] = {}
    columns = st.columns(3)

    for index, feature in enumerate(V2_FEATURE_ORDER):
        with columns[index]:
            values[feature] = st.number_input(
                V2_FEATURE_LABELS[feature],
                value=float(V2_DEFAULT_VALUES[feature]),
                step=0.1,
                format="%.4f",
                key=f"v2_{feature}",
            )

    return values


def render_v2_prediction() -> None:
    """Render Dataset V2 confidence-aware prediction."""
    values = render_v2_inputs()

    if st.button("Predict Crop", key="predict_v2"):
        try:
            model, label_encoder = cached_v2_prediction_artifacts()
            result = predict_crop_with_confidence(values, model=model, label_encoder=label_encoder)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except ConfidencePredictionInputError as exc:
            st.warning(str(exc))
        except (TypeError, ValueError) as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            st.markdown("### Crop Prediction")
            st.subheader("Recommended Crop:")
            st.success(result.predicted_crop)
            st.subheader("Confidence:")
            st.info(f"{result.confidence:.2%}")

            st.markdown("### Crop Probability Distribution")
            display_table = result.probabilities[["Crop", "Probability_Percent"]].copy()
            display_table["Probability"] = display_table["Probability_Percent"].map(lambda value: f"{value:.2f}%")
            st.dataframe(display_table[["Crop", "Probability"]], hide_index=True, width="stretch")
            chart_data = result.probabilities.set_index("Crop")["Probability_Percent"]
            st.bar_chart(chart_data)

            try:
                explanation = explain_v2_prediction(
                    values,
                    model=model,
                    label_encoder=label_encoder,
                    predicted_crop=result.predicted_crop,
                    explainer=cached_v2_shap_explainer(),
                )
            except ShapExplanationError as exc:
                st.warning(str(exc))
            else:
                st.markdown("### Why was this crop predicted?")
                st.caption(
                    "SHAP estimates the contribution of each input feature to the model's prediction."
                )
                explanation_table = explanation.contributions[["Feature", "Contribution"]].copy()
                explanation_table["Contribution"] = explanation_table["Contribution"].map(
                    lambda value: f"{value:+.4f}"
                )
                st.dataframe(explanation_table, hide_index=True, width="stretch")
                shap_chart_data = explanation.contributions.set_index("Feature")["Contribution"]
                st.bar_chart(shap_chart_data)


def render_review1_prediction() -> None:
    """Render the original Review 1 prediction flow."""
    values = render_review1_inputs()

    if st.button("Predict Crop", key="predict_review1"):
        try:
            model, label_encoder = cached_review1_prediction_artifacts()
            prediction = predict_crop(values, model=model, label_encoder=label_encoder)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except PredictionInputError as exc:
            st.warning(str(exc))
        else:
            st.subheader("Recommended Crop:")
            st.success(prediction)


def main() -> None:
    """Render the Streamlit crop recommendation UI."""
    st.set_page_config(page_title="Crop Recommendation System")
    st.title("Crop Recommendation System")

    v2_tab, review1_tab = st.tabs(["Dataset V2", "Review 1"])
    with v2_tab:
        render_v2_prediction()
    with review1_tab:
        render_review1_prediction()


if __name__ == "__main__":
    main()

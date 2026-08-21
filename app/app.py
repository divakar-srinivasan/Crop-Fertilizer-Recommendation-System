"""Basic Streamlit app for the Review 1 crop recommendation prototype."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from paths import FEATURE_ORDER  # noqa: E402
from predict import PredictionInputError, load_prediction_artifacts, predict_crop  # noqa: E402


FEATURE_LABELS = {
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

DEFAULT_VALUES = {
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


@st.cache_resource
def cached_prediction_artifacts() -> tuple[object, object]:
    """Load model artifacts once per Streamlit session."""
    return load_prediction_artifacts()


def render_inputs() -> dict[str, float]:
    """Render the 12 soil inputs and return their values."""
    values: dict[str, float] = {}
    columns = st.columns(3)

    for index, feature in enumerate(FEATURE_ORDER):
        with columns[index % 3]:
            max_value = 14.0 if feature == "pH" else None
            values[feature] = st.number_input(
                FEATURE_LABELS[feature],
                min_value=0.0,
                max_value=max_value,
                value=float(DEFAULT_VALUES[feature]),
                step=0.1,
                format="%.4f",
            )

    return values


def main() -> None:
    """Render the Streamlit crop recommendation UI."""
    st.set_page_config(page_title="Crop Recommendation System")
    st.title("Crop Recommendation System")

    values = render_inputs()

    if st.button("Predict Crop"):
        try:
            model, label_encoder = cached_prediction_artifacts()
            prediction = predict_crop(values, model=model, label_encoder=label_encoder)
        except FileNotFoundError as exc:
            st.error(str(exc))
        except PredictionInputError as exc:
            st.warning(str(exc))
        else:
            st.subheader("Recommended Crop:")
            st.success(prediction)


if __name__ == "__main__":
    main()

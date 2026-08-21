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

V2_FEATURE_UNITS = {
    "Soil_Moisture": "%",
    "Humidity": "%",
    "Temperature": "°C",
}

V2_DEFAULT_VALUES = {
    "Soil_Moisture": 44.0,
    "Humidity": 55.56,
    "Temperature": 29.58,
}


def apply_simple_premium_styles() -> None:
    """Apply a simple polished presentation theme."""
    st.markdown(
        """
        <style>
            :root {
                --green: #0f6b3a;
                --green-dark: #0b4e2c;
                --emerald: #25a35a;
                --gold: #d6a520;
                --bg: #f7fbf8;
                --card: #ffffff;
                --text: #17231a;
                --muted: #647269;
                --border: #deeadf;
            }

            .stApp {
                background: var(--bg);
                color: var(--text);
            }

            .block-container {
                max-width: 980px;
                padding-top: 2rem;
                padding-bottom: 1.5rem;
            }

            #MainMenu,
            footer,
            header {
                visibility: hidden;
            }

            h1, h2, h3, p {
                letter-spacing: 0;
            }

            div[data-testid="stAppViewContainer"] h1 {
                color: var(--green-dark);
                font-size: 2.35rem;
                font-weight: 850;
                line-height: 1.1;
                margin: 0 0 0.45rem 0;
                text-align: center;
            }

            div[data-testid="stCaptionContainer"] {
                color: var(--muted);
                text-align: center;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 8px;
                box-shadow: 0 12px 28px rgba(15, 80, 45, 0.08);
            }

            div[data-testid="stMarkdownContainer"] h2,
            div[data-testid="stMarkdownContainer"] h3,
            div[data-testid="stMarkdownContainer"] h4,
            div[data-testid="stMarkdownContainer"] h5 {
                color: var(--green-dark);
            }

            div[data-testid="stNumberInput"] label {
                color: var(--text);
                font-weight: 650;
            }

            div[data-testid="stNumberInput"] input {
                background-color: #173723 !important;
                border-radius: 8px;
                border: 1px solid #7fc79b !important;
                caret-color: #ffffff !important;
                color: #ffffff !important;
                font-size: 1.05rem;
                font-weight: 700;
                text-align: center;
                -webkit-text-fill-color: #ffffff !important;
            }

            div[data-testid="stNumberInput"] input:focus {
                border-color: var(--gold) !important;
                box-shadow: 0 0 0 2px rgba(214, 165, 32, 0.28) !important;
                outline: none !important;
            }

            div[data-testid="stNumberInput"] button {
                background-color: #eef7f0 !important;
                border-color: #bedac6 !important;
                color: var(--green-dark) !important;
            }

            div[data-testid="stNumberInput"] button svg {
                color: var(--green-dark) !important;
                fill: currentColor !important;
            }

            div.stButton > button {
                background: var(--green);
                border: 1px solid var(--green);
                border-radius: 8px;
                color: #ffffff;
                font-weight: 800;
                min-height: 3rem;
                box-shadow: 0 10px 22px rgba(15, 107, 58, 0.22);
                transition: all 160ms ease;
            }

            div.stButton > button:hover {
                background: var(--green-dark);
                border-color: var(--green-dark);
                color: #ffffff;
                transform: translateY(-1px);
            }

            @keyframes fadeIn {
                from {
                    opacity: 0;
                    transform: translateY(6px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }

            @media (max-width: 760px) {
                div[data-testid="stAppViewContainer"] h1 {
                    font-size: 1.9rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the simple centered app header."""
    st.title("🌱 Crop Recommendation")
    st.caption("Find the most suitable crop based on soil and environmental conditions.")


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
    """Render the three user-facing inputs and return their values."""
    values: dict[str, float] = {}
    with st.container(border=True):
        st.markdown("### Your Conditions")
        columns = st.columns(3)

        for index, feature in enumerate(V2_FEATURE_ORDER):
            with columns[index]:
                label = f"{V2_FEATURE_LABELS[feature]} ({V2_FEATURE_UNITS[feature]})"
                values[feature] = st.number_input(
                    label,
                    value=float(V2_DEFAULT_VALUES[feature]),
                    step=0.1,
                    format="%.4f",
                    key=f"v2_{feature}",
                )

    return values


def render_predict_button() -> bool:
    """Render one centered prediction button."""
    left, middle, right = st.columns([1, 2, 1])
    with middle:
        return st.button("🌱 Predict Crop", key="predict_v2", type="primary", use_container_width=True)


def render_empty_state() -> None:
    """Render a minimal pre-prediction state."""
    st.info("Enter your conditions above and select Predict Crop.")


def render_result_card(predicted_crop: str, confidence: float) -> None:
    """Render the crop and confidence as the main visual result."""
    confidence_percent = confidence * 100.0
    with st.container(border=True):
        st.markdown("##### Recommended Crop")
        st.markdown(f"## {predicted_crop.upper()}")
        st.markdown(f"**{confidence_percent:.2f}% Confidence**")
        st.progress(min(max(confidence, 0.0), 1.0))


def render_probability_bars(probabilities: object, predicted_crop: str) -> None:
    """Render sorted crop probabilities as horizontal bars."""
    with st.container(border=True):
        st.markdown("### Crop Probabilities")
        sorted_probabilities = probabilities.sort_values(by="Probability", ascending=False)
        for item in sorted_probabilities.itertuples(index=False):
            crop = str(item.Crop)
            probability = min(max(float(item.Probability), 0.0), 1.0)
            percent = float(item.Probability_Percent)
            label = f"**{crop}**" if crop == predicted_crop else crop
            name_col, bar_col, value_col = st.columns([2, 5, 1])
            name_col.markdown(label)
            bar_col.progress(probability)
            value_col.markdown(f"**{percent:.2f}%**")


def render_why_this_crop(explanation: object) -> None:
    """Render existing SHAP contributions in reviewer-friendly language."""
    records = explanation.contributions[["Feature", "Contribution"]].to_dict("records")
    max_abs = max((abs(float(record["Contribution"])) for record in records), default=1.0) or 1.0

    with st.container(border=True):
        st.markdown("### Why this crop?")
        st.caption("Feature contributions show how each input influenced this individual prediction.")
        for record in records:
            feature = str(record["Feature"])
            contribution = float(record["Contribution"])
            label = V2_FEATURE_LABELS.get(feature, feature.replace("_", " "))
            normalized = min(max(abs(contribution) / max_abs, 0.0), 1.0)
            name_col, bar_col, value_col = st.columns([2, 5, 1])
            name_col.markdown(label)
            bar_col.progress(normalized)
            value_col.markdown(f"**{contribution:+.2f}**")


def render_v2_prediction() -> None:
    """Render the final user-facing crop prediction workflow."""
    values = render_v2_inputs()
    predict_clicked = render_predict_button()

    if not predict_clicked:
        render_empty_state()
        return

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
        render_result_card(result.predicted_crop, result.confidence)
        render_probability_bars(result.probabilities, result.predicted_crop)

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
            render_why_this_crop(explanation)


def render_review1_prediction() -> None:
    """Render the original Review 1 prediction flow in a collapsed section."""
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
            st.success(f"Recommended Crop: {prediction}")


def render_footer() -> None:
    """Render a small, non-technical footer."""
    st.caption("Enter values → Predict → View recommendation")
    st.caption("9 crop categories supported")


def main() -> None:
    """Render the Streamlit crop recommendation UI."""
    st.set_page_config(page_title="Crop Recommendation", page_icon="🌱", layout="centered")
    apply_simple_premium_styles()
    render_header()
    render_v2_prediction()
    render_footer()

    with st.expander("Additional soil-parameter check", expanded=False):
        render_review1_prediction()


if __name__ == "__main__":
    main()

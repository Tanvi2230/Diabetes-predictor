from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_from_csv, predict_single
from src.train_model import main as train_model_main
from src.utils import FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, get_feature_metadata


st.set_page_config(
    page_title="Smart Diabetes Predictor",
    page_icon="🩺",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(159, 196, 255, 0.22), transparent 32%),
            radial-gradient(circle at top right, rgba(116, 198, 157, 0.18), transparent 28%),
            linear-gradient(135deg, #f7fbff 0%, #eef7f2 55%, #f9f3e7 100%);
        color: #16202a;
    }
    .hero-card, .metric-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(22, 32, 42, 0.08);
        border-radius: 22px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 18px 40px rgba(37, 68, 86, 0.08);
        backdrop-filter: blur(10px);
    }
    .prediction-good {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(45, 106, 79, 0.12);
        border: 1px solid rgba(45, 106, 79, 0.3);
    }
    .prediction-alert {
        padding: 1rem 1.2rem;
        border-radius: 18px;
        background: rgba(188, 71, 73, 0.1);
        border: 1px solid rgba(188, 71, 73, 0.28);
    }
    .stButton button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(90deg, #2d6a4f, #1d3557);
        color: white;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
    }
    @media (max-width: 640px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .hero-card, .metric-card {
            padding: 1rem;
            border-radius: 14px;
        }
        .hero-card h1 {
            font-size: 1.4rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if not METRICS_PATH.exists() or not MODEL_PATH.exists():
        train_model_main()
    import json

    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def render_sidebar(metrics: dict) -> None:
    st.sidebar.title("Model Control Center")
    st.sidebar.caption("CPU-friendly clinical risk screening assistant")
    st.sidebar.success(f"Best model: {metrics['best_model_name']}")

    with st.sidebar.expander("Model leaderboard", expanded=True):
        ranking = pd.DataFrame(metrics["all_model_metrics"]).T.reset_index()
        ranking = ranking.rename(columns={"index": "Model"})
        st.dataframe(
            ranking[["Model", "accuracy", "precision", "recall", "f1_score"]]
            .sort_values("f1_score", ascending=False)
            .style.format({col: "{:.3f}" for col in ["accuracy", "precision", "recall", "f1_score"]}),
            use_container_width=True,
        )

    if st.sidebar.button("Retrain Model"):
        with st.spinner("Training models and refreshing visuals..."):
            train_model_main()
        st.sidebar.success("Training completed.")
        st.rerun()


def render_prediction_form() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Smart Diabetes Predictor</h1>
            <p>
                A lightweight machine learning assistant that estimates diabetes risk from core
                clinical indicators, explains the result, and keeps a local prediction history.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    feature_meta = get_feature_metadata()
    with st.form("prediction_form"):
        st.subheader("Patient Inputs")
        columns = st.columns(2)
        values = {}
        for index, feature in enumerate(FEATURE_COLUMNS):
            meta = feature_meta[feature]
            with columns[index % 2]:
                values[feature] = st.number_input(
                    feature,
                    min_value=meta["min"],
                    max_value=meta["max"],
                    value=meta["default"],
                    step=meta["step"],
                )
        submitted = st.form_submit_button("Predict Diabetes Risk")

    if submitted:
        result = predict_single(values)
        css_class = "prediction-alert" if result["prediction"] == "Diabetic" else "prediction-good"
        st.markdown(
            f"""
            <div class="{css_class}">
                <h3>Prediction: {result['prediction']}</h3>
                <p>Probability score: <strong>{result['probability']:.2%}</strong></p>
                <p>Risk level: <strong>{result['risk_level']}</strong></p>
                <p>Primary explanation: {result['explanation_text']}</p>
                <p>Chosen model: {result['best_model']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explanation_frame = pd.DataFrame(result["explanations"])
        if not explanation_frame.empty:
            st.subheader("Top Influencing Factors")
            st.bar_chart(explanation_frame.set_index("feature")["impact"])
            st.dataframe(explanation_frame, use_container_width=True)


def render_visuals(metrics: dict) -> None:
    st.subheader("Model Visualizations")
    chart_columns = st.columns(2)
    plot_labels = {
        "correlation_heatmap": "Correlation heatmap",
        "feature_importance": "Feature importance",
        "feature_distributions": "Distribution plots",
        "confusion_matrix": "Confusion matrix",
    }
    plots_dir = PROJECT_ROOT / "models" / "plots"
    for index, key in enumerate(plot_labels):
        plot_path = plots_dir / f"{key}.png"
        if plot_path.exists():
            with chart_columns[index % 2]:
                st.markdown(f"**{plot_labels[key]}**")
                st.image(str(plot_path), use_container_width=True)


def render_bulk_prediction() -> None:
    st.subheader("Bulk Prediction")
    st.caption("Upload a CSV with the same eight medical input columns.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        dataframe = pd.read_csv(uploaded)
        st.write("Preview")
        st.dataframe(dataframe.head(), use_container_width=True)
        if st.button("Run Bulk Prediction"):
            scored = predict_from_csv(uploaded)
            st.success("Bulk scoring finished.")
            st.dataframe(scored, use_container_width=True)
            csv_bytes = scored.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Results",
                data=csv_bytes,
                file_name="bulk_predictions.csv",
                mime="text/csv",
            )


def main() -> None:
    metrics = load_metrics()
    render_sidebar(metrics)

    prediction_tab, visual_tab, bulk_tab = st.tabs(
        ["Prediction Assistant", "Analytics Dashboard", "Bulk Prediction"]
    )
    with prediction_tab:
        render_prediction_form()
    with visual_tab:
        render_visuals(metrics)
    with bulk_tab:
        render_bulk_prediction()


if __name__ == "__main__":
    main()

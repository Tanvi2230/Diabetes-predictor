from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.utils import (
    FEATURE_COLUMNS,
    HISTORY_PATH,
    MODEL_PATH,
    append_prediction_history,
    probability_to_risk,
    summarize_drivers,
)


def load_trained_bundle(model_path: Path | str = MODEL_PATH) -> dict:
    """Load the serialized training bundle."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run src/train_model.py first."
        )
    return joblib.load(model_path)


def validate_input(input_data: dict[str, float | int]) -> dict[str, float]:
    """Validate and normalize input values."""
    missing = [column for column in FEATURE_COLUMNS if column not in input_data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    normalized = {}
    for field, value in input_data.items():
        try:
            normalized[field] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric value for {field}: {value}") from exc

        if normalized[field] < 0:
            raise ValueError(f"{field} cannot be negative.")

    return normalized


def explain_prediction(bundle: dict, input_frame: pd.DataFrame) -> list[dict]:
    """Estimate per-feature influence for a single prediction."""
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    selector = preprocessor.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_features = [
        feature for feature, is_selected in zip(FEATURE_COLUMNS, selected_mask) if is_selected
    ]
    transformed = preprocessor.transform(input_frame)

    if hasattr(model, "coef_"):
        weights = model.coef_[0]
        values = transformed[0]
    elif hasattr(model, "feature_importances_"):
        weights = model.feature_importances_
        values = transformed[0]
    else:
        baseline_frame = bundle["feature_importance"].copy()
        explanations = []
        for _, row in baseline_frame.head(5).iterrows():
            explanations.append(
                {
                    "feature": row["feature"],
                    "impact": float(abs(row["importance"])),
                    "direction": "higher risk" if row["importance"] >= 0 else "lower risk",
                }
            )
        return explanations

    explanations = []
    for feature, value, weight in zip(selected_features, values, weights):
        influence = float(value * weight)
        explanations.append(
            {
                "feature": feature,
                "impact": abs(influence),
                "direction": "higher risk" if influence >= 0 else "lower risk",
            }
        )

    return sorted(explanations, key=lambda item: item["impact"], reverse=True)


def predict_single(input_data: dict[str, float | int], save_history: bool = True) -> dict:
    """Run a single patient prediction and return enriched output."""
    bundle = load_trained_bundle()
    validated = validate_input(input_data)
    input_frame = pd.DataFrame([validated], columns=FEATURE_COLUMNS)
    pipeline = bundle["pipeline"]

    prediction = int(pipeline.predict(input_frame)[0])
    probability = float(pipeline.predict_proba(input_frame)[0][1])
    risk_level = probability_to_risk(probability)
    label = "Diabetic" if prediction == 1 else "Non-Diabetic"
    explanations = explain_prediction(bundle, input_frame)
    explanation_text = summarize_drivers(explanations)

    result = {
        "prediction": label,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "explanations": explanations,
        "explanation_text": explanation_text,
        "best_model": bundle["best_model_name"],
        "history_path": str(HISTORY_PATH),
    }

    if save_history:
        append_prediction_history(validated, probability, label, risk_level)

    return result


def predict_from_csv(csv_path: Path | str | object) -> pd.DataFrame:
    """Run bulk predictions from a CSV path or uploaded file object."""
    if hasattr(csv_path, "read"):
        dataframe = pd.read_csv(csv_path)
    else:
        dataframe = pd.read_csv(Path(csv_path))
    missing = [column for column in FEATURE_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

    bundle = load_trained_bundle()
    pipeline = bundle["pipeline"]
    scored = dataframe.copy()
    scored["diabetes_probability"] = pipeline.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scored["prediction"] = pipeline.predict(scored[FEATURE_COLUMNS])
    scored["prediction"] = scored["prediction"].map({1: "Diabetic", 0: "Non-Diabetic"})
    scored["risk_level"] = scored["diabetes_probability"].apply(probability_to_risk)
    return scored

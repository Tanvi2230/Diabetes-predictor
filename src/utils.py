from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"

ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
PLOTS_DIR = MODELS_DIR / "plots"
HISTORY_PATH = DATA_DIR / "prediction_history.csv"
DATA_PATH = DATA_DIR / "diabetes.csv"

RISK_BANDS = {
    "Low": (0.0, 0.35),
    "Medium": (0.35, 0.65),
    "High": (0.65, 1.01),
}


def ensure_directories() -> None:
    """Create standard project directories if they do not exist."""
    for directory in (DATA_DIR, MODELS_DIR, PLOTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_feature_metadata() -> dict[str, dict[str, float | int]]:
    """UI-friendly ranges for validating medical inputs."""
    return {
        "Pregnancies": {"min": 0, "max": 20, "default": 2, "step": 1},
        "Glucose": {"min": 0, "max": 250, "default": 120, "step": 1},
        "BloodPressure": {"min": 0, "max": 140, "default": 72, "step": 1},
        "SkinThickness": {"min": 0, "max": 100, "default": 20, "step": 1},
        "Insulin": {"min": 0, "max": 900, "default": 79, "step": 1},
        "BMI": {"min": 0.0, "max": 70.0, "default": 32.0, "step": 0.1},
        "DiabetesPedigreeFunction": {
            "min": 0.05,
            "max": 3.0,
            "default": 0.47,
            "step": 0.01,
        },
        "Age": {"min": 10, "max": 100, "default": 33, "step": 1},
    }


def probability_to_risk(probability: float) -> str:
    """Map a prediction probability to a user-friendly risk bucket."""
    for risk_level, (lower, upper) in RISK_BANDS.items():
        if lower <= probability < upper:
            return risk_level
    return "High"


def summarize_drivers(explanations: Iterable[dict[str, Any]], top_n: int = 3) -> str:
    """Build a short explanation sentence from ranked feature contributions."""
    ranked = list(explanations)[:top_n]
    if not ranked:
        return "The model did not surface any dominant drivers for this prediction."

    phrases = []
    for item in ranked:
        name = item["feature"]
        impact = item["impact"]
        direction = item["direction"]
        phrases.append(f"{name} ({direction}, impact {impact:.3f})")

    return "Top influencing factors: " + ", ".join(phrases) + "."


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_prediction_history(
    input_data: dict[str, Any],
    probability: float,
    label: str,
    risk_level: str,
    path: Path = HISTORY_PATH,
) -> None:
    """Persist prediction history for later review."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    fieldnames = ["timestamp", *FEATURE_COLUMNS, "probability", "prediction", "risk_level"]

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **{column: input_data[column] for column in FEATURE_COLUMNS},
        "probability": round(probability, 4),
        "prediction": label,
        "risk_level": risk_level,
    }

    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

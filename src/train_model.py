from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data_preprocessing import (
    clean_data,
    create_preprocessor,
    load_dataset,
    prepare_train_test_data,
)
from src.evaluate_model import (
    create_visualizations,
    evaluate_predictions,
    generate_classification_report,
    get_feature_importance_frame,
)
from src.utils import METRICS_PATH, MODEL_PATH, PLOTS_DIR, ensure_directories, save_json


def build_model_registry() -> dict[str, object]:
    """Define the supervised models to compare."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=4,
            random_state=42,
        ),
        "Support Vector Machine": SVC(
            kernel="rbf",
            probability=True,
            C=1.2,
            gamma="scale",
            random_state=42,
        ),
    }


def train_and_select_best_model(dataframe: pd.DataFrame) -> dict:
    """Train every candidate model, compare metrics, and keep the strongest one."""
    cleaned = clean_data(dataframe)
    x_train, x_test, y_train, y_test = prepare_train_test_data(cleaned)

    model_results = {}
    best_name = None
    best_score = -1.0
    best_pipeline = None

    for model_name, estimator in build_model_registry().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(k_features="all")),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, predictions)
        report = generate_classification_report(y_test, predictions)

        model_results[model_name] = {
            **metrics,
            "classification_report": report,
        }

        ranking_score = metrics["f1_score"] + (metrics["recall"] * 0.05)
        if ranking_score > best_score:
            best_score = ranking_score
            best_name = model_name
            best_pipeline = pipeline

    assert best_pipeline is not None
    assert best_name is not None

    feature_importance = get_feature_importance_frame(best_pipeline, x_test, y_test)
    plots = create_visualizations(cleaned, best_pipeline, x_test, y_test, PLOTS_DIR)

    bundle = {
        "pipeline": best_pipeline,
        "best_model_name": best_name,
        "all_model_metrics": model_results,
        "feature_importance": feature_importance,
        "plots": plots,
        "test_shape": {"rows": int(len(x_test)), "columns": int(x_test.shape[1])},
    }
    return bundle


def save_training_artifacts(bundle: dict) -> None:
    """Persist the trained model and metadata."""
    ensure_directories()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    serializable_metrics = {
        "best_model_name": bundle["best_model_name"],
        "all_model_metrics": bundle["all_model_metrics"],
        "feature_importance": bundle["feature_importance"].to_dict(orient="records"),
        "plots": bundle["plots"],
        "test_shape": bundle["test_shape"],
    }
    save_json(serializable_metrics, METRICS_PATH)


def main() -> int:
    """Entry point for model training."""
    ensure_directories()
    dataset = load_dataset()
    bundle = train_and_select_best_model(dataset)
    save_training_artifacts(bundle)

    print(f"Training complete. Best model: {bundle['best_model_name']}")
    print(f"Saved bundle to: {MODEL_PATH}")
    print("Model comparison:")
    for model_name, metrics in bundle["all_model_metrics"].items():
        print(
            f"  - {model_name}: "
            f"accuracy={metrics['accuracy']:.3f}, "
            f"precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, "
            f"f1={metrics['f1_score']:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

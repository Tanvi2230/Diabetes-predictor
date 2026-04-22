from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import FEATURE_COLUMNS


def evaluate_predictions(y_true, y_pred) -> dict[str, float]:
    """Return core classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


def generate_classification_report(y_true, y_pred) -> dict[str, Any]:
    """Return the full sklearn classification report as a dict."""
    return classification_report(y_true, y_pred, output_dict=True)


def create_visualizations(
    dataframe: pd.DataFrame,
    trained_pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
) -> dict[str, str]:
    """Generate project charts for the UI and README."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    generated_paths = {}

    correlation_path = output_dir / "correlation_heatmap.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Diabetes Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(correlation_path, dpi=200)
    plt.close()
    generated_paths["correlation_heatmap"] = str(correlation_path)

    distribution_path = output_dir / "feature_distributions.png"
    figure, axes = plt.subplots(2, 4, figsize=(18, 9))
    for axis, feature in zip(axes.flatten(), FEATURE_COLUMNS):
        sns.histplot(dataframe[feature], kde=True, ax=axis, color="#2d6a4f")
        axis.set_title(feature)
    figure.suptitle("Input Feature Distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(distribution_path, dpi=200)
    plt.close(figure)
    generated_paths["feature_distributions"] = str(distribution_path)

    importance_path = output_dir / "feature_importance.png"
    importance_frame = get_feature_importance_frame(trained_pipeline, x_test, y_test)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_frame, x="importance", y="feature", palette="crest")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(importance_path, dpi=200)
    plt.close()
    generated_paths["feature_importance"] = str(importance_path)

    confusion_path = output_dir / "confusion_matrix.png"
    predictions = trained_pipeline.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(confusion_path, dpi=200)
    plt.close()
    generated_paths["confusion_matrix"] = str(confusion_path)

    return generated_paths


def get_feature_importance_frame(trained_pipeline, x_test, y_test) -> pd.DataFrame:
    """Compute feature importance from the best model or via permutation importance."""
    preprocessor = trained_pipeline.named_steps["preprocessor"]
    selector = preprocessor.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_features = [
        feature for feature, is_selected in zip(FEATURE_COLUMNS, selected_mask) if is_selected
    ]

    model = trained_pipeline.named_steps["model"]
    transformed_x = preprocessor.transform(x_test)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        result = permutation_importance(
            trained_pipeline,
            x_test,
            y_test,
            n_repeats=10,
            random_state=42,
            scoring="f1",
        )
        return (
            pd.DataFrame(
                {"feature": FEATURE_COLUMNS, "importance": result.importances_mean}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    importance_frame = pd.DataFrame(
        {"feature": selected_features, "importance": importances[: len(selected_features)]}
    )
    return importance_frame.sort_values("importance", ascending=False).reset_index(drop=True)

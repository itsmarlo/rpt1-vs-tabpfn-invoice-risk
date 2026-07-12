"""Shared evaluation and reporting helpers."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Any | None = None,
    inference_time_seconds: float | None = None,
) -> dict[str, float | None]:
    """Calculate standard binary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if y_proba is not None else None,
        "inference_time_seconds": inference_time_seconds,
    }


def save_metrics_comparison(
    metrics_list: list[dict[str, Any]], output_path: str | Path
) -> pd.DataFrame:
    """Save model metrics with a stable, human-friendly column order."""
    columns = [
        "model_name",
        "status",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "inference_time_seconds",
        "notes",
    ]
    output = pd.DataFrame(metrics_list).reindex(columns=columns)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return output


def plot_confusion_matrix(
    y_true: Any, y_pred: Any, output_path: str | Path, title: str
) -> None:
    """Save a confusion matrix as a PNG."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["On time", "Late"], cmap="Blues", colorbar=False
    )
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(destination, dpi=150)
    plt.close(display.figure_)


def save_skipped_plot(output_path: str | Path, message: str) -> None:
    """Create a small explanatory artifact when a model cannot run."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.axis("off")
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)

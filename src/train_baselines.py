"""Train and evaluate classic machine-learning baselines."""

from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import calculate_classification_metrics
from src.preprocessing import PreparedData


def _save_predictions(
    output_path: Path,
    invoice_ids: pd.Series,
    y_true: pd.Series,
    y_pred: Any,
    y_proba: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "invoice_id": invoice_ids.to_numpy(),
            "actual_target_paid_late": y_true.to_numpy(),
            "predicted_target_paid_late": y_pred,
            "probability_paid_late": y_proba,
        }
    ).to_csv(output_path, index=False)


def train_classic_baselines(
    prepared: PreparedData,
    invoice_ids: pd.Series,
    results_dir: str | Path,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Fit logistic regression and random forest on the shared split."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1_000, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    filenames = {
        "Logistic Regression": "predictions_logistic_regression.csv",
        "Random Forest": "predictions_random_forest.csv",
    }
    metrics_rows: list[dict[str, Any]] = []
    destination = Path(results_dir)

    for model_name, model in models.items():
        model.fit(prepared.X_train, prepared.y_train)
        start = perf_counter()
        predictions = model.predict(prepared.X_test)
        probabilities = model.predict_proba(prepared.X_test)[:, 1]
        inference_time = perf_counter() - start

        metrics = calculate_classification_metrics(
            prepared.y_test, predictions, probabilities, inference_time
        )
        metrics_rows.append(
            {"model_name": model_name, "status": "completed", **metrics, "notes": ""}
        )
        _save_predictions(
            destination / filenames[model_name],
            invoice_ids,
            prepared.y_test,
            predictions,
            probabilities,
        )

    return metrics_rows


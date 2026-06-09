"""Optional local SAP-RPT-1 OSS evaluation using Hugging Face checkpoints."""

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import calculate_classification_metrics


def _write_empty_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "invoice_id",
            "actual_target_paid_late",
            "predicted_target_paid_late",
            "probability_paid_late",
        ]
    ).to_csv(path, index=False)


def _predict_proba_in_batches(
    model: Any, X_test: pd.DataFrame, batch_size: int
) -> np.ndarray:
    batches = [
        model.predict_proba(X_test.iloc[start : start + batch_size])
        for start in range(0, len(X_test), batch_size)
    ]
    return np.vstack(batches)


def train_rpt1_oss(
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    invoice_ids: pd.Series,
    results_dir: str | Path,
    max_training_samples: int = 256,
    inference_batch_size: int = 10,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run SAP-RPT-1 OSS when installed and authenticated with Hugging Face."""
    output_path = Path(results_dir) / "predictions_rpt1_oss.csv"
    try:
        from sap_rpt_oss import SAP_RPT_OSS_Classifier
    except ImportError:
        message = (
            "SAP-RPT-1 OSS skipped: optional package not installed. "
            "Install requirements-rpt1-oss.txt after accepting the Hugging Face terms."
        )
        print(message)
        _write_empty_predictions(output_path)
        return {
            "model_name": "SAP-RPT-1 OSS (HF)",
            "status": "skipped_not_installed",
            "notes": message,
        }, None

    training_size = min(max_training_samples, len(y_train))
    rng = np.random.default_rng(random_state)
    sample_positions = rng.choice(len(y_train), size=training_size, replace=False)
    X_train = X_train_raw.iloc[sample_positions].copy()
    y_train_sample = y_train.iloc[sample_positions].copy()

    try:
        model = SAP_RPT_OSS_Classifier(
            max_context_size=max_training_samples,
            bagging=1,
        )
        model.fit(X_train, y_train_sample)
        start = perf_counter()
        probability_matrix = _predict_proba_in_batches(
            model, X_test_raw, inference_batch_size
        )
        inference_time = perf_counter() - start
        classes = np.asarray(model.classes_)
        predictions = classes[np.argmax(probability_matrix, axis=1)]
        positive_class_position = np.flatnonzero(classes == 1)
        if len(positive_class_position) != 1:
            raise RuntimeError("SAP-RPT-1 OSS did not expose binary classes {0, 1}")
        probabilities = probability_matrix[:, positive_class_position[0]]
    except Exception as exc:
        message = (
            "SAP-RPT-1 OSS could not run locally. Confirm gated model access with "
            f"`hf auth login` and check available memory. Details: {exc}"
        )
        print(message)
        _write_empty_predictions(output_path)
        return {
            "model_name": "SAP-RPT-1 OSS (HF)",
            "status": "skipped_runtime_error",
            "notes": message,
        }, None

    pd.DataFrame(
        {
            "invoice_id": invoice_ids.to_numpy(),
            "actual_target_paid_late": y_test.to_numpy(),
            "predicted_target_paid_late": predictions,
            "probability_paid_late": probabilities,
        }
    ).to_csv(output_path, index=False)
    metrics = calculate_classification_metrics(
        y_test, predictions, probabilities, inference_time
    )
    return {
        "model_name": "SAP-RPT-1 OSS (HF)",
        "status": "completed",
        **metrics,
        "notes": (
            f"Training capped at {training_size} rows; context {max_training_samples}; "
            f"bagging 1; inference batch size {inference_batch_size}."
        ),
    }, np.asarray(predictions)


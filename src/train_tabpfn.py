"""Optional local TabPFN training and evaluation."""

import gc
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from src.evaluate import calculate_classification_metrics
from src.preprocessing import PreparedData


def _predict_proba_in_batches(
    model: Any, X_test: np.ndarray, batch_size: int
) -> np.ndarray:
    """Run inference in small batches to limit GPU/MPS peak memory."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    batches = [
        model.predict_proba(X_test[start : start + batch_size])
        for start in range(0, len(X_test), batch_size)
    ]
    return np.vstack(batches)


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


def write_empty_tabpfn_predictions(path: str | Path) -> None:
    """Write the stable empty TabPFN prediction schema."""
    _write_empty_predictions(Path(path))


def _release_accelerator_memory() -> None:
    """Best-effort cleanup between retries without requiring PyTorch."""
    gc.collect()
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def train_tabpfn(
    prepared: PreparedData,
    invoice_ids: pd.Series,
    results_dir: str | Path,
    max_training_samples: int = 750,
    inference_batch_size: int = 10,
    random_state: int = 42,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Run TabPFN when available; otherwise return a non-fatal skipped result."""
    output_path = Path(results_dir) / "predictions_tabpfn.csv"
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        message = "TabPFN is not installed. Install it with: pip install tabpfn"
        print(message)
        _write_empty_predictions(output_path)
        return {
            "model_name": "TabPFN",
            "status": "skipped_not_installed",
            "notes": message,
        }, None

    retry_profiles = [
        (max_training_samples, inference_batch_size, "auto"),
        (500, 5, "auto"),
        (300, 1, "auto"),
        (200, 1, "cpu"),
    ]
    last_error: Exception | None = None
    successful_profile: tuple[int, int, str] | None = None
    inference_time = 0.0
    probability_matrix: np.ndarray | None = None
    model: Any = None

    for training_limit, batch_size, device in retry_profiles:
        training_limit = min(training_limit, len(prepared.y_train))
        rng = np.random.default_rng(random_state)
        sample_positions = rng.choice(
            len(prepared.y_train), size=training_limit, replace=False
        )
        X_train = prepared.X_train[sample_positions]
        y_train = prepared.y_train.iloc[sample_positions]
        print(
            "TabPFN attempt: "
            f"{training_limit} training rows, batch size {batch_size}, device {device}"
        )

        try:
            model = TabPFNClassifier(
                device=device,
                n_estimators=4,
                fit_mode="low_memory",
                memory_saving_mode=True,
                random_state=random_state,
            )
            model.fit(X_train, y_train)
            start = perf_counter()
            probability_matrix = _predict_proba_in_batches(
                model, prepared.X_test, batch_size
            )
            inference_time = perf_counter() - start
            successful_profile = (training_limit, batch_size, device)
            break
        except Exception as exc:
            last_error = exc
            del model
            model = None
            _release_accelerator_memory()

    if probability_matrix is None or successful_profile is None or model is None:
        message = (
            "TabPFN could not run after low-memory retries. Its package may need model "
            f"weights, more memory, or Hugging Face access. Details: {last_error}"
        )
        print(message)
        _write_empty_predictions(output_path)
        return {
            "model_name": "TabPFN",
            "status": "skipped_runtime_error",
            "notes": message,
        }, None

    class_positions = np.argmax(probability_matrix, axis=1)
    predictions = np.asarray(model.classes_)[class_positions]
    positive_class_position = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive_class_position) != 1:
        raise RuntimeError("TabPFN did not expose the expected binary classes {0, 1}")
    probabilities = probability_matrix[:, positive_class_position[0]]
    training_limit, batch_size, device = successful_profile

    pd.DataFrame(
        {
            "invoice_id": invoice_ids.to_numpy(),
            "actual_target_paid_late": prepared.y_test.to_numpy(),
            "predicted_target_paid_late": predictions,
            "probability_paid_late": probabilities,
        }
    ).to_csv(output_path, index=False)
    metrics = calculate_classification_metrics(
        prepared.y_test, predictions, probabilities, inference_time
    )
    return {
        "model_name": "TabPFN",
        "status": "completed",
        **metrics,
        "notes": (
            f"Training capped at {training_limit:,} rows; "
            f"inference batch size {batch_size}; device {device}."
        ),
    }, np.asarray(predictions)

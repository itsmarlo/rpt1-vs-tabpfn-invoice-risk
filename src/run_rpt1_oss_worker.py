"""Run optional SAP-RPT-1 OSS evaluation in an isolated Python process."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from src.preprocessing import load_dataset, prepare_train_test_data
from src.train_rpt1_oss import train_rpt1_oss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_invoices.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "rpt1_oss_metrics.json"


def _positive_int_from_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_worker() -> None:
    """Evaluate SAP-RPT-1 OSS and save metrics for the parent process."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA_PATH)
    prepared = prepare_train_test_data(df)
    test_invoice_ids = df.loc[prepared.y_test.index, "invoice_id"]
    max_training_samples = _positive_int_from_env("RPT1_OSS_MAX_TRAINING_SAMPLES", 256)
    inference_batch_size = _positive_int_from_env("RPT1_OSS_INFERENCE_BATCH_SIZE", 10)
    metrics, predictions = train_rpt1_oss(
        prepared.X_train_raw,
        prepared.X_test_raw,
        prepared.y_train,
        prepared.y_test,
        test_invoice_ids,
        RESULTS_DIR,
        max_training_samples=max_training_samples,
        inference_batch_size=inference_batch_size,
    )

    payload = {
        "metrics": metrics,
        "has_predictions": predictions is not None,
    }
    METRICS_PATH.write_text(
        json.dumps(payload, indent=2, default=_json_default),
        encoding="utf-8",
    )


if __name__ == "__main__":
    run_worker()

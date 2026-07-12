"""Run the complete local invoice payment risk comparison."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluate import (
    plot_confusion_matrix,
    save_metrics_comparison,
    save_skipped_plot,
)
from src.generate_data import generate_synthetic_invoices
from src.preprocessing import load_dataset, prepare_train_test_data
from src.rpt1_client import RPT1Client
from src.train_baselines import train_classic_baselines
from src.train_rpt1_oss import write_empty_rpt1_oss_predictions
from src.train_tabpfn import write_empty_tabpfn_predictions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_invoices.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
TABPFN_METRICS_PATH = RESULTS_DIR / "tabpfn_metrics.json"
TABPFN_PREDICTIONS_PATH = RESULTS_DIR / "predictions_tabpfn.csv"
RPT1_OSS_METRICS_PATH = RESULTS_DIR / "rpt1_oss_metrics.json"
RPT1_OSS_PREDICTIONS_PATH = RESULTS_DIR / "predictions_rpt1_oss.csv"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


def _native_optional_models_enabled() -> bool:
    """Return whether optional native model workers should run in this shell."""
    override = os.environ.get("RUN_NATIVE_OPTIONAL_MODELS", "").strip().lower()
    if override:
        return override in TRUTHY_ENV_VALUES
    return os.environ.get("CODEX_SANDBOX", "").strip() == ""


def _rpt1_metrics_row() -> dict[str, Any]:
    client = RPT1Client(PROJECT_ROOT / ".env")
    if not client.is_configured():
        message = (
            "SAP-RPT-1 skipped: credentials are not configured. "
            "See .env.example for the required variables."
        )
        print(message)
        return {
            "model_name": "SAP-RPT-1 AI Core",
            "status": "skipped_if_not_configured",
            "notes": message,
        }
    message = (
        "SAP-RPT-1 credentials detected, but inference is not executed because the "
        "environment-specific SAP AI Core API contract remains a TODO."
    )
    print(message)
    return {
        "model_name": "SAP-RPT-1 AI Core",
        "status": "configured_integration_pending",
        "notes": message,
    }


def _tabpfn_skipped_row(
    message: str, status: str = "skipped_subprocess_error"
) -> dict[str, Any]:
    print(message)
    write_empty_tabpfn_predictions(TABPFN_PREDICTIONS_PATH)
    return {
        "model_name": "TabPFN",
        "status": status,
        "notes": message,
    }


def _run_tabpfn_safely() -> tuple[dict[str, Any], Any | None]:
    """Run optional TabPFN in a subprocess so native aborts do not kill the run."""
    TABPFN_METRICS_PATH.unlink(missing_ok=True)
    if not _native_optional_models_enabled():
        message = (
            "TabPFN skipped: optional native model workers are disabled in this "
            "sandboxed/headless environment. Set RUN_NATIVE_OPTIONAL_MODELS=1 to "
            "attempt them on a machine where the native runtime is stable."
        )
        return _tabpfn_skipped_row(message, "skipped_native_disabled"), None

    env = os.environ.copy()
    env.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    result = subprocess.run(
        [sys.executable, "-m", "src.run_tabpfn_worker"],
        cwd=PROJECT_ROOT,
        env=env,
    )
    if result.returncode != 0:
        message = (
            "TabPFN skipped: isolated subprocess exited with code "
            f"{result.returncode}. The optional native runtime failed, but the "
            "classic baselines and other comparison rows were rebuilt."
        )
        return _tabpfn_skipped_row(message), None

    try:
        payload = json.loads(TABPFN_METRICS_PATH.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        message = f"TabPFN skipped: metrics handoff was unavailable. Details: {exc}"
        return _tabpfn_skipped_row(message), None
    finally:
        TABPFN_METRICS_PATH.unlink(missing_ok=True)

    if payload.get("has_predictions") and metrics.get("status") == "completed":
        predictions_frame = pd.read_csv(TABPFN_PREDICTIONS_PATH)
        return metrics, predictions_frame["predicted_target_paid_late"].to_numpy()
    return metrics, None


def _rpt1_oss_skipped_row(
    message: str, status: str = "skipped_subprocess_error"
) -> dict[str, Any]:
    print(message)
    write_empty_rpt1_oss_predictions(RPT1_OSS_PREDICTIONS_PATH)
    return {
        "model_name": "SAP-RPT-1 OSS (HF)",
        "status": status,
        "notes": message,
    }


def _run_rpt1_oss_safely() -> dict[str, Any]:
    """Run optional SAP-RPT-1 OSS in a subprocess to contain native failures."""
    RPT1_OSS_METRICS_PATH.unlink(missing_ok=True)
    if not _native_optional_models_enabled():
        message = (
            "SAP-RPT-1 OSS skipped: optional native model workers are disabled "
            "in this sandboxed/headless environment. Set "
            "RUN_NATIVE_OPTIONAL_MODELS=1 to attempt them on a machine where the "
            "native runtime is stable."
        )
        return _rpt1_oss_skipped_row(message, "skipped_native_disabled")

    result = subprocess.run(
        [sys.executable, "-m", "src.run_rpt1_oss_worker"],
        cwd=PROJECT_ROOT,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        message = (
            "SAP-RPT-1 OSS skipped: isolated subprocess exited with code "
            f"{result.returncode}. The optional native runtime failed, but the "
            "rest of the comparison was rebuilt."
        )
        return _rpt1_oss_skipped_row(message)

    try:
        payload = json.loads(RPT1_OSS_METRICS_PATH.read_text(encoding="utf-8"))
        return payload["metrics"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        message = (
            "SAP-RPT-1 OSS skipped: metrics handoff was unavailable. "
            f"Details: {exc}"
        )
        return _rpt1_oss_skipped_row(message)
    finally:
        RPT1_OSS_METRICS_PATH.unlink(missing_ok=True)


def run_experiment() -> None:
    """Generate data, train available models, and save comparison artifacts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print(f"Dataset missing; generating synthetic data at {DATA_PATH}")
        generate_synthetic_invoices(output_path=DATA_PATH)

    df = load_dataset(DATA_PATH)
    prepared = prepare_train_test_data(df)
    test_invoice_ids = df.loc[prepared.y_test.index, "invoice_id"]

    metrics_rows = train_classic_baselines(prepared, test_invoice_ids, RESULTS_DIR)
    tabpfn_metrics, tabpfn_predictions = _run_tabpfn_safely()
    metrics_rows.append(tabpfn_metrics)
    metrics_rows.append(_run_rpt1_oss_safely())
    metrics_rows.append(_rpt1_metrics_row())

    confusion_matrix_path = RESULTS_DIR / "confusion_matrix_tabpfn.png"
    if tabpfn_predictions is not None:
        plot_confusion_matrix(
            prepared.y_test,
            tabpfn_predictions,
            confusion_matrix_path,
            "TabPFN Confusion Matrix",
        )
    else:
        save_skipped_plot(
            confusion_matrix_path,
            "TabPFN confusion matrix unavailable\nbecause TabPFN inference was skipped.",
        )

    comparison = save_metrics_comparison(
        metrics_rows, RESULTS_DIR / "metrics_comparison.csv"
    )
    display_columns = ["model_name", "status", "accuracy", "f1", "roc_auc"]
    print("\nModel comparison")
    print(comparison[display_columns].to_string(index=False, na_rep="-"))
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    run_experiment()

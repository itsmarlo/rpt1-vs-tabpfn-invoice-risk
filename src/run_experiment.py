"""Run the complete local invoice payment risk comparison."""

from pathlib import Path
from typing import Any

from src.evaluate import (
    plot_confusion_matrix,
    save_metrics_comparison,
    save_skipped_plot,
)
from src.generate_data import generate_synthetic_invoices
from src.preprocessing import load_dataset, prepare_train_test_data
from src.rpt1_client import RPT1Client
from src.train_baselines import train_classic_baselines
from src.train_rpt1_oss import train_rpt1_oss
from src.train_tabpfn import train_tabpfn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_invoices.csv"
RESULTS_DIR = PROJECT_ROOT / "results"


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
    tabpfn_metrics, tabpfn_predictions = train_tabpfn(
        prepared, test_invoice_ids, RESULTS_DIR
    )
    metrics_rows.append(tabpfn_metrics)
    rpt1_oss_metrics, _ = train_rpt1_oss(
        prepared.X_train_raw,
        prepared.X_test_raw,
        prepared.y_train,
        prepared.y_test,
        test_invoice_ids,
        RESULTS_DIR,
    )
    metrics_rows.append(rpt1_oss_metrics)
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

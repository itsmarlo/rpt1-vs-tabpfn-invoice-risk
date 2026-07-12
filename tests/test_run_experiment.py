from types import SimpleNamespace

import pandas as pd

import src.run_experiment as run_experiment


def test_tabpfn_subprocess_failure_writes_empty_predictions(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("CODEX_SANDBOX", raising=False)
    metrics_path = tmp_path / "tabpfn_metrics.json"
    predictions_path = tmp_path / "predictions_tabpfn.csv"
    monkeypatch.setattr(run_experiment, "TABPFN_METRICS_PATH", metrics_path)
    monkeypatch.setattr(run_experiment, "TABPFN_PREDICTIONS_PATH", predictions_path)
    monkeypatch.setattr(
        run_experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=134),
    )

    metrics, predictions = run_experiment._run_tabpfn_safely()

    assert predictions is None
    assert metrics["model_name"] == "TabPFN"
    assert metrics["status"] == "skipped_subprocess_error"
    output = pd.read_csv(predictions_path)
    assert output.empty
    assert output.columns.tolist() == [
        "invoice_id",
        "actual_target_paid_late",
        "predicted_target_paid_late",
        "probability_paid_late",
    ]


def test_rpt1_oss_subprocess_failure_writes_empty_predictions(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("CODEX_SANDBOX", raising=False)
    metrics_path = tmp_path / "rpt1_oss_metrics.json"
    predictions_path = tmp_path / "predictions_rpt1_oss.csv"
    monkeypatch.setattr(run_experiment, "RPT1_OSS_METRICS_PATH", metrics_path)
    monkeypatch.setattr(run_experiment, "RPT1_OSS_PREDICTIONS_PATH", predictions_path)
    monkeypatch.setattr(
        run_experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=134),
    )

    metrics = run_experiment._run_rpt1_oss_safely()

    assert metrics["model_name"] == "SAP-RPT-1 OSS (HF)"
    assert metrics["status"] == "skipped_subprocess_error"
    output = pd.read_csv(predictions_path)
    assert output.empty
    assert output.columns.tolist() == [
        "invoice_id",
        "actual_target_paid_late",
        "predicted_target_paid_late",
        "probability_paid_late",
    ]


def test_native_workers_are_skipped_in_codex_sandbox(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    monkeypatch.delenv("RUN_NATIVE_OPTIONAL_MODELS", raising=False)
    monkeypatch.setattr(
        run_experiment,
        "TABPFN_PREDICTIONS_PATH",
        tmp_path / "predictions_tabpfn.csv",
    )
    monkeypatch.setattr(
        run_experiment,
        "RPT1_OSS_PREDICTIONS_PATH",
        tmp_path / "predictions_rpt1_oss.csv",
    )
    monkeypatch.setattr(
        run_experiment.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected run")),
    )

    tabpfn_metrics, tabpfn_predictions = run_experiment._run_tabpfn_safely()
    rpt1_metrics = run_experiment._run_rpt1_oss_safely()

    assert tabpfn_predictions is None
    assert tabpfn_metrics["status"] == "skipped_native_disabled"
    assert rpt1_metrics["status"] == "skipped_native_disabled"

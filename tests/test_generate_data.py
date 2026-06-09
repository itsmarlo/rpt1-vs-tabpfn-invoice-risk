from src.generate_data import EXPECTED_COLUMNS, generate_synthetic_invoices


def test_generated_dataset_has_expected_columns() -> None:
    df = generate_synthetic_invoices(n_rows=100)
    assert df.columns.tolist() == EXPECTED_COLUMNS
    assert len(df) == 100


def test_target_is_binary() -> None:
    df = generate_synthetic_invoices(n_rows=500)
    assert set(df["target_paid_late"].unique()).issubset({0, 1})
    assert df["target_paid_late"].nunique() == 2


import numpy as np

from src.generate_data import generate_synthetic_invoices
from src.preprocessing import prepare_train_test_data


def test_train_test_split_and_preprocessing_are_numeric() -> None:
    df = generate_synthetic_invoices(n_rows=500)
    prepared = prepare_train_test_data(df)

    assert len(prepared.y_train) == 400
    assert len(prepared.y_test) == 100
    assert prepared.X_train.shape[0] == 400
    assert prepared.X_test.shape[0] == 100
    assert np.issubdtype(prepared.X_train.dtype, np.number)
    assert np.issubdtype(prepared.X_test.dtype, np.number)
    assert prepared.X_train.shape[1] == len(prepared.feature_names)
    assert not any("customer_id" in name for name in prepared.feature_names)

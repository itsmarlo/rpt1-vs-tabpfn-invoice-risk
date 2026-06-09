import numpy as np
import pandas as pd

from src.train_rpt1_oss import _predict_proba_in_batches


class RecordingModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.batch_sizes.append(len(X))
        positive = np.full(len(X), 0.55)
        return np.column_stack([1 - positive, positive])


def test_rpt1_oss_probability_inference_is_batched() -> None:
    model = RecordingModel()
    X_test = pd.DataFrame({"amount": range(12)})

    probabilities = _predict_proba_in_batches(model, X_test, batch_size=5)

    assert model.batch_sizes == [5, 5, 2]
    assert probabilities.shape == (12, 2)


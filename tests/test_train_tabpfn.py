import numpy as np

from src.train_tabpfn import _predict_proba_in_batches


class RecordingModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.batch_sizes.append(len(X))
        positive = np.full(len(X), 0.6)
        return np.column_stack([1 - positive, positive])


def test_tabpfn_probability_inference_is_batched() -> None:
    model = RecordingModel()
    X_test = np.zeros((12, 4))

    probabilities = _predict_proba_in_batches(model, X_test, batch_size=5)

    assert model.batch_sizes == [5, 5, 2]
    assert probabilities.shape == (12, 2)

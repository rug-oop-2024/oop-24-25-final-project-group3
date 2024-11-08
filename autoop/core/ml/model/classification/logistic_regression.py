import numpy as np
from autoop.core.ml.model import Model
import pydoc  # noqa: F401

from sklearn.linear_model import LogisticRegression as SkLogisticRegression


class LogisticRegression(Model):
    def __init__(self, **kwargs):
        super().__init__(model_type="classification")
        self.model = SkLogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # If y is one-hot encoded, convert it to a 1D array
        # If y is one-hot encoded, convert it to binary labels (0 or 1)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.trained:
            raise ValueError("Model must be trained before making predictions."
                             )
        return self.model.predict(X)

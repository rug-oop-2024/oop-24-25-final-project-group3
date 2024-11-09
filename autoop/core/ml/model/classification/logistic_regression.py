import numpy as np
from autoop.core.ml.model import Model

from sklearn.linear_model import LogisticRegression as SkLogisticRegression


class LogisticRegression(Model):
    """Wrapper for logistic regression using scikit-learn's implementation."""
    def __init__(self, **kwargs) -> None:
        """Initialising the LogisticRegression class"""
        super().__init__(model_type="classification")
        self.model = SkLogisticRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the logistic regression model.

            Args:
                X (np.ndarray): Feature matrix for training.
                y (np.ndarray): Target vector for training.

            Returns:
                None

            Raises:
                ValueError: If the model cannot be trained.
        """
        # If y is one-hot encoded, convert it to a 1D array
        # If y is one-hot encoded, convert it to binary labels (0 or 1)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

            Args:
                X (np.ndarray): Feature matrix for predictions.

            Returns:
                np.ndarray: Predicted class labels.

            Raises:
                ValueError: If the model is not trained before prediction.
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions."
                             )
        return self.model.predict(X)

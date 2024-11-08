import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import Ridge
import pydoc  # noqa: F401


class RidgeRegression(Model):
    """Wrapper for ridge regression using scikit-learn's implementation."""
    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialize the Ridge Regression model with a specified alpha
        (regularization strength).

        Parameters:
        - alpha (float): Regularization strength. Must be a positive float.
          Default is 1.0.
        """
        super().__init__(model_type="regression")
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Ridge Regression model.

        Parameters:
        - X (np.ndarray): Feature matrix with shape (n_samples, n_features).
        - y (np.ndarray): Target vector with shape (n_samples,).

        Sets:
        - self.trained (bool): Marks the model as trained.
        """
        self.model.fit(X, y)
        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Ridge Regression model.

        Parameters:
        - X (np.ndarray): Feature matrix with shape (n_samples, n_features).

        Returns:
        - np.ndarray: Array of predicted values with shape (n_samples,).

        Raises:
        - ValueError: If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model must be trained before making "
                             "predictions.")
        return self.model.predict(X)

#  pydoc.writedoc('ridge_regression')

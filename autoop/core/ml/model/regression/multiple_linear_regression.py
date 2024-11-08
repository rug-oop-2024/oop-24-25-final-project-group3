import numpy as np
from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression
import pydoc


class MultipleLinearRegression(Model):
    def __init__(self):
        """
        Initialize the Multiple Linear Regression model.
        """
        super().__init__(model_type="regression")
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Multiple Linear Regression model.

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
        Make predictions using the trained model.

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

#  pydoc.writedoc('multiple_linear_regression')

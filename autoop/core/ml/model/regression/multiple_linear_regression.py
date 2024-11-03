import numpy as np
from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    def __init__(self):
        """
        Initialize the MultipleLinearRegression model with no parameters.
        """
        self.parameters = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the model using the normal equation method.

        Args:
            X (np.ndarray): The input feature matrix (n_samples, n_features).
            y (np.ndarray): The target values (n_samples,).
        """
        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Calculate parameters (weights) using the Normal Equation
        self.parameters = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X (np.ndarray): The input feature matrix for prediction
            (n_samples, n_features).

        Returns:
            np.ndarray: The predicted values (n_samples,).

        Raises:
            ValueError: If the model is not trained (parameters are None).
        """
        if self.parameters is None:
            raise ValueError("Model must be trained before making "
                             "predictions.")

        # Add a column of ones to X for the intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X @ self.parameters

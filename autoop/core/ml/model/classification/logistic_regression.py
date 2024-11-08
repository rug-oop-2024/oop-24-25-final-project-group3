import numpy as np
from autoop.core.ml.model import Model
import pydoc


class LogisticRegression(Model):
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000) -> None:
        super().__init__(model_type="classification")
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0  # Bias is a scalar, not part of weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if y.ndim > 1 and y.shape[1] == 2:
            y = y[:, 1]  # Use the second column for binary classification

        # Initialize weights and bias
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)  # Match exactly the number of features

        # Gradient descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        self.trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions."
                             )

        # Calculate the linear model with the trained weights and bias
        linear_model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(linear_model)

        # Return binary predictions (0 or 1)
        return np.where(predictions >= 0.5, 1, 0)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

# pydoc.write('logistic_regression')

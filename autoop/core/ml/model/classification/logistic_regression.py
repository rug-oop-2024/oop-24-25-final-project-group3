import numpy as np
from autoop.core.ml.model import Model


class LogisticRegression(Model):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """Initialize the logistic regression model."""
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            model = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        model = np.dot(X, self.weights) + self.bias
        predictions = self._sigmoid(model)
        return np.where(predictions >= 0.5, 1, 0)

    def _sigmoid(self, x):
        """Apply sigmoid function."""
        return 1 / (1 + np.exp(-x))

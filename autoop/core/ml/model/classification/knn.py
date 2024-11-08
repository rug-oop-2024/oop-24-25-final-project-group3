import numpy as np
from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier
import pydoc  # noqa: F401


class KNearestNeighbors(Model):
    def __init__(self, n_neighbors: int = 3) -> None:
        """
        Initialize the K-Nearest Neighbors model with the specified number
        of neighbors.

        Parameters:
        - n_neighbors (int): The number of neighbors to use. Default is 3.
        """
        super().__init__(model_type="classification")
        self.n_neighbors: int = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.trained: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the K-Nearest Neighbors model.

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
        - np.ndarray: Array of predictions with shape (n_samples,).

        Raises:
        - ValueError: If the model has not been trained yet.
        """
        if not self.trained:
            raise ValueError("Model must be trained before making predictions."
                             )
        return self.model.predict(X)

#  pydoc.write('knn')

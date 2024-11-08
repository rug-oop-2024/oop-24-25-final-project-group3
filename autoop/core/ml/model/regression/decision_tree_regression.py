import numpy as np
from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeRegressor
import pydoc  # noqa: F401


class DecisionTreeRegression(Model):
    """
    Wrapper for decision tree regression
    using scikit-learn's implementation.
    """
    def __init__(self, max_depth: int = None) -> None:
        """
        Initialize the Decision Tree Regression model with a specified maximum
        depth.

        Parameters:
        - max_depth (int): The maximum depth of the tree. If None, nodes are
          expanded until all leaves are pure.
        """
        super().__init__(model_type="regression")
        self.model = DecisionTreeRegressor(max_depth=max_depth)
        self.trained: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Decision Tree Regression model.

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
            raise ValueError("Model must be trained before making predictions."
                             )
        return self.model.predict(X)

# pydoc.writedoc('decision_tree_regression')

import numpy as np
from autoop.core.ml.model import Model
from sklearn.tree import DecisionTreeClassifier
import pydoc


class DecisionTreeClassification(Model):
    """
    A Decision Tree classifier model that recursively splits the dataset
    based on features to maximize the purity of each resulting node, creating
    a hierarchical structure for classification tasks.

    Attributes:
    - model (DecisionTreeClassifier): The underlying scikit-learn Decision
      Tree classifier.
    - trained (bool): Boolean flag indicating if the model has been trained.

    Methods:
    - fit(X, y): Trains the Decision Tree model on the provided dataset.
    - predict(X): Makes predictions on new data, given that the model is
      already trained.
    """
    def __init__(self, max_depth: int = None) -> None:
        """
        Initialize the Decision Tree Classification model with a specified
        maximum depth.

        Parameters:
        - max_depth (int): The maximum depth of the tree. If None, nodes are
          expanded until all leaves are pure.
        """
        super().__init__(model_type="classification")
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.trained: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the Decision Tree Classification model.

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


if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)

"""
This package offers machine learning models.

Modules:
- REGRESSION_MODELS: Implements regression models.
- CLASSIFICATION_MODELS: Implements classification models .

These modules can be imported for building, training and using.
"""

from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import (MultipleLinearRegression,
                                             RidgeRegression,
                                             DecisionTreeRegression)
from autoop.core.ml.model.classification import (LogisticRegression,
                                                 KNearestNeighbors,
                                                 DecisionTreeClassification)
import pydoc

REGRESSION_MODELS = ["MultipleLinearRegression", "RidgeRegression",
                     "DecisionTreeRegression"]
CLASSIFICATION_MODELS = ["LogisticRegression", "KNearestNeighbors",
                         "DecisionTreeClassification"]


def get_model(model_name: str) -> Model:
    """Factory function to get a model by name.

    Args:
        model_name (str): Name of the model to retrieve.

    Returns:
        Model: An instance of the requested model.

    Raises:
        ValueError: If the model_name is not found in either
        REGRESSION_MODELS or CLASSIFICATION_MODELS.
    """
    # Dictionary mapping model names to their classes
    model_mapping = {
        "MultipleLinearRegression": MultipleLinearRegression,
        "RidgeRegression": RidgeRegression,
        "DecisionTreeRegression": DecisionTreeRegression,
        "LogisticRegression": LogisticRegression,
        "KNearestNeighbors": KNearestNeighbors,
        "DecisionTreeClassification": DecisionTreeClassification,
    }

    # Check if the model_name exists in the mapping
    if model_name not in model_mapping:
        raise ValueError(
            f"Model '{model_name}' not implemented. Available models: "
            f"{', '.join(REGRESSION_MODELS + CLASSIFICATION_MODELS)}"
        )

    # Return an instance of the requested model
    return model_mapping[model_name]()


if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)

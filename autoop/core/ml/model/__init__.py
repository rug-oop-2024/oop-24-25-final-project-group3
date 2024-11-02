
from autoop.core.ml.model.model import Model
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import RidgeRegression
from autoop.core.ml.model.regression import DecisionTreeRegression
from autoop.core.ml.model.classification import LogisticRegression
from autoop.core.ml.model.classification import KNearestNeighbors
from autoop.core.ml.model.classification import DecisionTreeClassification

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
        "DecisionTreeRegressorModel": DecisionTreeRegression,
        "LogisticRegression": LogisticRegression,
        "KNearestNeighbors": KNearestNeighbors,
        "DecisionTreeClassifierModel": DecisionTreeClassification,
    }

    # Check if the model_name exists in the mapping
    if model_name not in model_mapping:
        raise ValueError(f"Model '{model_name}' is not implemented. Available"
                         f" models are: {', '.join(REGRESSION_MODELS +
                                         CLASSIFICATION_MODELS)}")

    # Return an instance of the requested model
    return model_mapping[model_name]()

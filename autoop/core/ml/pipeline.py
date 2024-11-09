from typing import Dict, List, Union
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    Represents a machine learning pipeline for training and evaluating models
    with defined input features, target feature, metrics, and data split.
    """

    def __init__(self, metrics: List[Metric], dataset: Dataset, model: Model,
                 input_features: List[Feature], target_feature: Feature,
                 split: float = 0.8) -> None:
        """
        Initialize a Pipeline instance.

            Args:
                metrics (List[Metric]): List of evaluation metrics.
                dataset (Dataset): The dataset object.
                model (Model): The model to be trained and evaluated.
                input_features (List[Feature]): List of input features.
                target_feature (Feature): The target feature.
                split (float): Train-test split ratio (default is 0.8).
        """

        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and (
           model.type != "classification"):
            raise ValueError("Model type must be classification for "
                             "categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError("Model type must be regression for continuous "
                             "target feature")

    def __str__(self) -> str:
        """
        Return a string representation of the pipeline.

            Returns:
                str: String representation with model and features.
        """

        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Return the trained model.

            Returns:
                Model: The model used in the pipeline.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Collect and return artifacts generated during the pipeline.

            Returns:
                List[Artifact]: List of generated artifacts.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                         data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
                         name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Artifact) -> None:
        """
        Register an artifact with the given name.

            Args:
                name (str): The name of the artifact.
                artifact: The artifact object.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocess input and target features, generating artifacts and
        data vectors.
        """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(self._input_features,
                                            self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact)
                               in input_results]

    def _split_data(self) -> None:
        """
        Split data into training and testing sets based on the split ratio.
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in
                         self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in
                        self._input_vectors]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(
            self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Concatenate multiple vectors into a single 2D array.

            Args:
                vectors (List[np.array]): List of feature vectors.

            Returns:
                np.array: Concatenated 2D array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Train the model using the prepared training data.
        """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """
        Evaluate the model on the test data and store results.
        """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)  # Use __call__ directly
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> Dict[str, Union[np.ndarray, List, Model]]:
        """
        Execute the pipeline, including training and evaluation.

            Returns:
                Dict[str, Any]: Dictionary with metrics, predictions, and
                trained model.
        """
        self._preprocess_features()
        self._split_data()
        self._train()

        # Evaluate on training set
        train_X = self._compact_vectors(self._train_X)

        train_Y = self._train_y
        train_metrics_results = []
        train_predictions = self._model.predict(train_X)
        for metric in self._metrics:
            result = metric(train_predictions, train_Y)
            train_metrics_results.append((metric, result))

        # Evaluate on test set (evaluation set)
        test_X = self._compact_vectors(self._test_X)
        test_Y = self._test_y
        test_metrics_results = []
        test_predictions = self._model.predict(test_X)
        for metric in self._metrics:
            result = metric(test_predictions, test_Y)
            test_metrics_results.append((metric, result))

        return {
            # Test metrics (for compatibility with TestPipeline)
            "metrics": test_metrics_results,
            "train_metrics": train_metrics_results,    # Training metrics
            "test_predictions": test_predictions,      # Test predictions
            "train_predictions": train_predictions,    # Training predictions
            "trained_model": self._model,  # Return the trained model
            "train_X": train_X,
            "train_Y": train_Y,
        }

from abc import ABC, abstractmethod
import numpy as np
from typing import Union
import pydoc  # noqa: F401

REGRESSION_METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "r_squared",
]

CLASSIFICATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
]

LOG_CLASSIFICATION_METRICS = [
    "logistic_accuracy",
    "logistic_precision",
    "logistic_recall"
]


def get_metric(name: str) -> Union['MeanSquaredError', 'Accuracy', 'Precision',
                                   'Recall', 'LogisticAccuracy',
                                   'LogisticPrecision', 'LogisticRecall',
                                   'MeanAbsoluteError', 'RSquared']:
    """
    Factory function to get a metric instance by its name.

    Args:
        name (str): The name of the metric to retrieve.

    Returns:
        Union[MeanSquaredError, Accuracy, Precision, Recall, LogisticAccuracy,
        LogisticPrecision, LogisticRecall, MeanAbsoluteError, RSquared]:
        The metric instance.

    Raises:
        ValueError: If the metric name is not implemented.
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "logistic_accuracy":
        return LogisticAccuracy()
    elif name == "logistic_precision":
        return LogisticPrecision()
    elif name == "logistic_recall":
        return LogisticRecall()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the metric given ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The calculated metric.
        """
        pass


# Regression Metrics
class MeanSquaredError(Metric):
    """Mean Squared Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean squared error."""
        return float(np.mean((y_true - y_pred) ** 2))

    def __str__(self) -> str:
        """String representation"""
        return "MeanSquaredError"


class MeanAbsoluteError(Metric):
    """Mean Absolute Error metric for regression."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate mean absolute error."""
        return float(np.mean(np.abs(y_true - y_pred)))

    def __str__(self) -> str:
        """String representation"""
        return "MeanAbsoluteError"


class RSquared(Metric):
    """R-squared (R²) metric for regression, measuring goodness of fit."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')

    def __str__(self) -> str:
        """String representation"""
        return "RSquared"


class LogisticAccuracy(Metric):
    """Accuracy metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy for classification."""
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        return float(np.mean(y_true == y_pred))

    def __str__(self) -> str:
        """String representation"""
        return "LogisticAccuracy"


class LogisticPrecision(Metric):
    """Precision metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision for classification."""
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (true_positives / predicted_positives
                if predicted_positives != 0 else 0.0)

    def __str__(self) -> str:
        """String representation"""
        return "LogisticPrecision"


class LogisticRecall(Metric):
    """Recall metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Recall metric for logistic classification."""
        # Convert one-hot predictions to class labels if necessary
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=1)
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (true_positives / actual_positives
                if actual_positives != 0 else 0.0)

    def __str__(self) -> str:
        """String representation"""
        return "LogisticRecall"


class Accuracy(Metric):
    """Accuracy metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy."""
        return float(np.mean(y_true == y_pred))

    def __str__(self) -> str:
        """String representation"""
        return "LogisticAccuracy"


class Precision(Metric):
    """Precision metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision."""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (true_positives / predicted_positives if
                predicted_positives != 0 else 0.0)

    def __str__(self) -> str:
        """String representation"""
        return "Precision"


class Recall(Metric):
    """Recall metric for classification."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall."""
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (true_positives / actual_positives
                if actual_positives != 0 else 0.0)

    def __str__(self) -> str:
        """String representation"""
        return "Recall"

# pydoc.writedoc('metric')

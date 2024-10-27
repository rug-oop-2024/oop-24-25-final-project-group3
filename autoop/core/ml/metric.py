from abc import ABC, abstractmethod
import numpy as np

# List of supported metrics
METRICS = [
    "mean_squared_error",
    "accuracy",
    "precision",
    "recall",
    "f1_score"
]

class Metric(ABC):
    """Abstract base class for a metric."""
    
    @abstractmethod
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        pass

class Accuracy(Metric):
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean(predictions == ground_truth)

class MeanSquaredError(Metric):
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        return np.mean((predictions - ground_truth) ** 2)

class Precision(Metric):
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        predicted_positives = np.sum(predictions == 1)
        return true_positives / predicted_positives if predicted_positives > 0 else 0.0

class Recall(Metric):
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        actual_positives = np.sum(ground_truth == 1)
        return true_positives / actual_positives if actual_positives > 0 else 0.0

class F1Score(Metric):
    def __call__(self, predictions: np.ndarray, ground_truth: np.ndarray) -> float:
        precision = Precision()(predictions, ground_truth)
        recall = Recall()(predictions, ground_truth)
        return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def get_metric(name: str) -> Metric:
    """Factory function to retrieve the correct metric instance based on name."""
    if name == "accuracy":
        return Accuracy()
    elif name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    elif name == "f1_score":
        return F1Score()
    else:
        raise ValueError(f"Unknown metric: {name}")

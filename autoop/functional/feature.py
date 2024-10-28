
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd
from pandas.api.types import CategoricalDtype

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects feature types (categorical or numerical) in a dataset.

    Args:
        dataset (Dataset): The dataset to analyze.

    Returns:
        List[Feature]: A list of features with detected types.
    """
    raise NotImplementedError("This should be implemented by you.")
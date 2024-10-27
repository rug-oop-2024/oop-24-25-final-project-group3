from typing import List
import pandas as pd
import numpy as np
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect the types of features in a dataset.

    Args:
        dataset (Dataset): The dataset to analyze.

    Returns:
        List[Feature]: A list of Feature objects with initialized metadata.
    """
    features = []
    for column in dataset.columns:
        data = dataset[column].values
        if np.issubdtype(data.dtype, np.number):
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, type=feature_type)
        feature.initialize_metadata(data)
        features.append(feature)

    return features

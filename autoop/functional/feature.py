
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
    # Extract the DataFrame from the dataset using `read`
    df = dataset.read()

    # Initialize an empty list to store the feature objects
    features = []

    # Iterate through the columns of the DataFrame
    for column in df.columns:
        # Determine the type of feature
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_type = "numerical"
        elif isinstance(df[column].dtype, CategoricalDtype) or pd.api.types.is_object_dtype(df[column]):
            feature_type = "categorical"
        else:
            raise ValueError(f"Unknown data type for column {column}")

        # Create a Feature object with values and append it to the list
        features.append(Feature(name=column, type=feature_type, values=df[column].values))

    return features

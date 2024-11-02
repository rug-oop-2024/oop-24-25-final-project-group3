from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas as pd

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detects feature types (categorical or numerical) in a dataset.
    
    Args:
        dataset (Dataset): The dataset to analyze.
        
    Raises:
        ValueError: If the dataset contains any NaN values.
        
    Returns:
        List[Feature]: A list of features with detected types.
    """
    # Extract the DataFrame from the dataset
    df = dataset.to_dataframe()

    # Check for any NaN values in the dataset
    if df.isnull().any().any():
        raise ValueError("Dataset contains NaN values, which is not allowed.")

    # Initialize an empty list to store the feature objects
    features = []

    # Iterate through the columns of the DataFrame
    for column in df.columns:
        # Determine the type of feature
        if pd.api.types.is_numeric_dtype(df[column]):
            feature_type = "numerical"
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            feature_type = "categorical"
        else:
            raise ValueError(f"Unknown data type for column {column}")

        # Create a Feature object and append it to the list
        features.append(Feature(name=column, type=feature_type))

    return features

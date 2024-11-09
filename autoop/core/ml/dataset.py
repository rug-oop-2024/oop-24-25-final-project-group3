from autoop.core.ml.artifact import Artifact
from typing import Dict, List
import pandas as pd
import io
import pydoc


class Dataset(Artifact):
    """
    Dataset class for managing and representing dataset artifacts.

    This class extends the Artifact class, providing methods for handling
    datasets, including creating from dataframes, reading, and saving data in
    CSV format. It integrates with Pydantic for validation and serialization.

    Attributes:
        Inherits all attributes from the Artifact class, such as:
        - name (str): Name of the dataset.
        - asset_path (str): Path for the dataset asset.
        - version (str): Version of the dataset.
        - type (str): Fixed as "dataset".
        - data (Optional[bytes]): Encoded CSV data of the dataset.
        - metadata (Dict[str, str]): Metadata for the dataset.
        - tags (List[str]): Tags for categorizing the dataset.
        - id (Optional[str]): Unique identifier generated from asset_path and
                              version.

    Methods:
        __init__(*args, metadata, tags, **kwargs) -> None:
            Initializes the Dataset instance with defaults
            for metadata and tags.

        from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
            Creates a Dataset instance from a DataFrame.

        read() -> pd.DataFrame:
            Reads and decodes CSV data to a DataFrame.

        save(data: pd.DataFrame) -> bytes:
            Saves a DataFrame as encoded CSV data and persists it.
    """

    def __init__(self, *args, metadata: Dict[str, str] | None = None,
                 tags: List[str] | None = None, **kwargs) -> None:
        """initialising the Dataset class"""
        # Set default values for missing fields
        metadata = metadata or {}
        tags = tags or []
        super().__init__(type="dataset", metadata=metadata, tags=tags,
                         *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str, asset_path: str,
                       version: str = "1.0.0") -> 'Dataset':
        """returns the dataset"""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            metadata={},   # Default empty metadata
            tags=[],       # Default empty tags list
        )

    def read(self) -> pd.DataFrame:
        """ Read data from a given path """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """ Save data to a given path """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)


if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)

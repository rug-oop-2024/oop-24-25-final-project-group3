from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import base64


class Artifact(BaseModel):
    """
    Artifact class for representing and managing data assets.

    This class defines a data model for artifacts such as datasets or models,
    including methods for generating unique IDs, reading, and saving data. It
    is built on Pydantic for data validation and parsing, supporting storage
    interaction simulation.

    Attributes:
        name (str): The name of the artifact.
        asset_path (str): Path where the artifact asset is stored.
        version (str): Version of the artifact.
        type (str): Type of the artifact (e.g., dataset, model).
        data (Optional[bytes]): The raw data of the artifact.
        metadata (Dict[str, str]): Metadata associated with the artifact.
        tags (List[str]): Tags to categorize the artifact.
        id (Optional[str]): Unique identifier generated from asset_path and
        version.

    Methods:
        __init__(**kwargs) -> None:
            Initializes the Artifact instance and generates an ID if not set.

        generate_id() -> str:
            Generates a unique ID based on the asset path and version.

        read() -> bytes:
            Reads and returns the data of the artifact.

        save(data: bytes) -> None:
            Saves data to the artifact, simulating storage interaction.

        Config:
            Configuration class for Pydantic model behavior.
    """

    name: str
    asset_path: str
    version: str
    type: str = Field(..., description="Type of the artifact (e.g., dataset, "
                      "model)")
    data: Optional[bytes] = None
    metadata: Dict[str, str] = Field(default_factory=dict,
                                     description="Metadata for the artifact")
    tags: List[str] = Field(default_factory=list, description="Tags "
                            "associated with the artifact")
    id: Optional[str] = None  # This is set based on `asset_path` and `version`

    def __init__(self, **kwargs) -> None:
        """initialising Artifact class"""
        super().__init__(**kwargs)
        if self.id is None:  # Generate an id if not provided
            self.id = self.generate_id()

    def generate_id(self) -> str:
        """Generates a unique ID based on asset_path and version."""
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()
                                                ).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        """
        Simulates reading data for the artifact. In practice, this would
        interact with storage.
        """
        if self.data is None:
            raise ValueError("Data not available for this artifact")
        return self.data

    def save(self, data: bytes) -> None:
        """
        Simulates saving data for the artifact. In practice, this would
        save to storage.
        """
        self.data = data

    class Config:
        """
        Adhering the code to Pydantic v2
        """
        arbitrary_types_allowed = True
        populate_by_name = True

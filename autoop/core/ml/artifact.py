import base64
from typing import Dict, List, Union, Optional
from pydantic import BaseModel, Field, validator
from autoop.core.storage import Storage  # Assuming this is the storage interface

# A global storage instance could be set here, or through a method to set it up later
_global_storage: Optional[Storage] = None

def set_global_storage(storage: Storage):
    global _global_storage
    _global_storage = storage


class Artifact(BaseModel):
    name: str = Field(..., description="Name of the artifact.")
    asset_path: str = Field(..., description="The path where the asset is stored.")
    version: str = Field(..., description="The version of the artifact.")
    _data: Optional[bytes] = None  # Internal variable for storing loaded data
    metadata: Dict[str, Union[str, int]] = Field(..., description="Metadata related to the artifact.")
    type: str = Field(..., description="Type of the artifact (e.g., model:torch).")
    tags: List[str] = Field(..., description="Tags describing the artifact.")
    id: str = Field(init=False, description="Unique ID for the artifact, derived from asset_path and version.")

    @validator("id", pre=True, always=True)
    def generate_id(cls, v, values) -> str:
        asset_path = values.get("asset_path")
        version = values.get("version")
        if asset_path is None or version is None:
            raise ValueError("Both asset_path and version must be provided to generate an id.")
        encoded_path = base64.urlsafe_b64encode(asset_path.encode()).decode()
        return f"{encoded_path}:{version}"

    @property
    def data(self) -> bytes:
        if self._data is None:
            if _global_storage is not None:
                # Lazy-load the data from the global storage if available
                self._data = _global_storage.load(self.asset_path)
            else:
                raise ValueError("Global storage is not set, unable to load data.")
        return self._data

    def __str__(self) -> str:
        return f"Artifact(id={self.id}, type={self.type}, tags={self.tags})"

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        validate_assignment = True

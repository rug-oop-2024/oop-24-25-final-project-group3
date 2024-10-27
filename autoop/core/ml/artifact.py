from abc import ABC
from pydantic import BaseModel, Field
from typing import Dict, List
import base64

class Artifact(BaseModel, ABC):
    name: str = Field(..., description="Name of the artifact")
    asset_path: str = Field(..., description="Path to the asset file")
    version: str = Field(..., description="Version of the artifact")
    data: bytes = Field(..., description="Binary data of the artifact")
    metadata: Dict[str, str] = Field(default_factory=dict, description="Metadata associated with the artifact")
    type: str = Field(..., description="Type of the artifact, e.g., model or dataset")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the artifact")
    
    @property
    def id(self) -> str:
        encoded_path = base64.urlsafe_b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

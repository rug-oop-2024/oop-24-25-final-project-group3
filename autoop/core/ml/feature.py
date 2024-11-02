from pydantic import BaseModel, Field, validator
from typing import Literal, Union
import numpy as np

class Feature(BaseModel):
    name: str = Field(..., description="The name of the feature.")
    type: Literal["numerical", "categorical"] = Field(..., description="The type of the feature, either numerical or categorical.")
    values: Union[np.ndarray, list] = Field(..., description="The values for the feature.")

    @validator("values", pre=True)
    def convert_to_array(cls, v):
        """Ensures values are stored as a NumPy array for consistency."""
        return np.array(v) if not isinstance(v, np.ndarray) else v

    def __str__(self):
        """String representation for Feature."""
        return f"{self.name} (Type: {self.type})"

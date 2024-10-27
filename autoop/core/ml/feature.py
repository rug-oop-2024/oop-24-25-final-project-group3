from pydantic import BaseModel, Field
from typing import Literal, Union, List, Any
import numpy as np

class Feature(BaseModel):
    name: str = Field(..., description="Name of the feature")
    type: Literal["categorical", "numerical"] = Field(..., description="Type of the feature, either 'categorical' or 'numerical'")
    unique_values: Union[List[Any], None] = Field(default=None, description="Unique values if categorical")
    statistics: Union[dict, None] = Field(default=None, description="Statistics if numerical")

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type}, unique_values={self.unique_values}, statistics={self.statistics})"

    def initialize_metadata(self, data: np.ndarray):
        if self.type == "categorical":
            self.unique_values = list(np.unique(data))
        elif self.type == "numerical":
            self.statistics = {
                "mean": float(np.mean(data)),
                "std_dev": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data))
            }

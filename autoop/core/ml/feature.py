from pydantic import BaseModel, Field, field_validator
from typing import Any, List, Literal, Tuple, Optional, Union
import numpy as np


class Feature(BaseModel):
    """
    Feature class for representing and managing dataset features.

    This class defines a feature within a dataset, including its name, type,
    and associated values. The feature can be either numerical or categorical.
    It ensures that the values are stored as a NumPy array for consistency.

    Attributes:
        name (str): The name of the feature.
        type (Literal["numerical", "categorical"]): The type of the feature,
            specifying whether it is numerical or categorical.
        values (Optional[np.ndarray]): The values associated with the feature,
            stored as a NumPy array. Defaults to an empty array if not
            provided.

    Methods:
        convert_to_array(v: Any) -> np.ndarray:
            Class method to validate and convert the 'values' attribute to a
            NumPy array, ensuring consistency in data handling.

        __str__() -> str:
            Returns a string representation of the feature, displaying its name
            and type.

    Model Config:
        - populate_by_name: Allows the population of fields by name.
        - arbitrary_types_allowed: Permits the use of arbitrary types for
        attribute definitions.
    """

    name: str = Field(..., description="The name of the feature.")
    type: Literal["numerical", "categorical"] = Field(..., description="The "
                                                      "type of the feature, "
                                                      "either numerical or "
                                                      "categorical.")
    values: Optional[np.ndarray] = Field(default_factory=lambda: np.array([]),
                                         description="The values for the "
                                                     "feature.")

    @field_validator("values", mode="before")
    def convert_to_array(cls, v: Union[List[Any], Tuple[Any, ...], np.ndarray,
                                       None]) -> np.ndarray:
        """Ensures values are stored as a NumPy array for consistency."""
        return np.array(v) if v is not None else np.array([])

    def __str__(self) -> str:
        """String representation for Feature."""
        return f"{self.name} (Type: {self.type})"

    # New model configuration for Pydantic v2``
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }

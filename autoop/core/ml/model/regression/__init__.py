from .multiple_linear_regression import MultipleLinearRegression
from .ridge_regression import RidgeRegression
from .decision_tree_regression import DecisionTreeRegression
import pydoc  # noqa: F401

__all__ = ["MultipleLinearRegression", "RidgeRegression",
           "DecisionTreeRegression"]

#  pydoc.writedoc('__init__')

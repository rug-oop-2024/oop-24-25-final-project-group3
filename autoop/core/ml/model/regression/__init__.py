from .multiple_linear_regression import MultipleLinearRegression
from .ridge_regression import RidgeRegression
from .decision_tree_regression import DecisionTreeRegression
import pydoc

__all__ = ["MultipleLinearRegression", "RidgeRegression",
           "DecisionTreeRegression"]

#  pydoc.writedoc('__init__')

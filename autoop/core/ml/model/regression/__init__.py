from .multiple_linear_regression import MultipleLinearRegression
from .ridge_regression import RidgeRegression
from .decision_tree_regression import DecisionTreeRegression
import pydoc  # noqa: F401

"""
This package offers machine learning models for regression tasks.

Modules:
- MultipleLinearRegression: Implements multiple linear regression.
- RidgeRegression: A ridge regression model.
- DecisionTreeRegression: A decision tree-based regression model.

These modules can be imported for building, training, and using regression
models.
"""
__all__ = ["MultipleLinearRegression", "RidgeRegression",
           "DecisionTreeRegression"]

#  pydoc.writedoc('__init__')

"""
This package offers machine learning models for classification tasks.

Modules:
- LogisticRegression: Implements logistic regression for binary classification.
- KNearestNeighbors: A KNN classifier for multi-class classification.
- DecisionTreeClassification: A decision tree-based classifier.

These modules can be imported for building, training, and using classifiers.
"""

import pydoc  # noqa: F401
from .logistic_regression import LogisticRegression
from .knn import KNearestNeighbors
from .decision_tree_classification import DecisionTreeClassification

__all__ = ["LogisticRegression", "KNearestNeighbors",
           "DecisionTreeClassification"]

#  pydoc.writedoc('__init__')

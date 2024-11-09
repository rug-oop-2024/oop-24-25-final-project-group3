"""
This package offers machine learning models for classification tasks.

Modules:
- LogisticRegression: Implements logistic regression for binary classification.
- KNearestNeighbors: A KNN classifier for multi-class classification.
- DecisionTreeClassification: A decision tree-based classifier.

These modules can be imported for building, training, and using classifiers.
"""

import pydoc
from .logistic_regression import LogisticRegression
from .knn import KNearestNeighbors
from .decision_tree_classification import DecisionTreeClassification

__all__ = ["LogisticRegression", "KNearestNeighbors",
           "DecisionTreeClassification"]

if __name__ == "__main__":
    # Generate documentation for this module and save it as an HTML file
    pydoc.writedoc(__name__)

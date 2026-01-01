"""
Models module.

==================================================================

The module aims to provide methods and classes for outcome modeling.
"""

# Author: Tim Ortkamp

from ._machine_learning_model import MachineLearningModel

from .forest import RandomForest
from .logistic import LogisticRegression
from .naive_bayes import NaiveBayes
from .neighbors import KNearestNeighbors
from .neural_network import FeedForwardNet
from .svm import SupportVectorMachine
from .tree import DecisionTree

__all__ = [
    'MachineLearningModel',
    'RandomForest',
    'LogisticRegression',
    'NaiveBayes',
    'KNearestNeighbors',
    'FeedForwardNet',
    'SupportVectorMachine',
    'DecisionTree']

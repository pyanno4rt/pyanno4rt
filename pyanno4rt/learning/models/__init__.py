"""
Models module.

==================================================================

The module aims to provide methods and classes for outcome modeling.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.models._machine_learning_model import MachineLearningModel

from pyanno4rt.learning.models.forest import RandomForest
from pyanno4rt.learning.models.logistic import LogisticRegression
from pyanno4rt.learning.models.naive_bayes import NaiveBayes
from pyanno4rt.learning.models.neighbors import KNearestNeighbors
from pyanno4rt.learning.models.neural_network import FeedForwardNet
from pyanno4rt.learning.models.svm import SupportVectorMachine
from pyanno4rt.learning.models.tree import DecisionTree, SoftDecisionTree

__all__ = [
    'MachineLearningModel',
    'RandomForest',
    'LogisticRegression',
    'NaiveBayes',
    'KNearestNeighbors',
    'FeedForwardNet',
    'SupportVectorMachine',
    'DecisionTree',
    'SoftDecisionTree']

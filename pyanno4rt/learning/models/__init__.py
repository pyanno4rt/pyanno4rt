"""
Models module.

==================================================================

The module aims to provide methods and classes for model fitting.
"""

# Author: Tim Ortkamp

from ._machine_learning_model import MachineLearningModel
from ._model_parameters import ModelParameters

from . import (
    forest, logistic, naive_bayes, neighbors, neural_network, svm, tree)

__all__ = [
    'MachineLearningModel',
    'ModelParameters',
    'forest',
    'logistic',
    'naive_bayes',
    'neighbors',
    'neural_network',
    'svm',
    'tree']

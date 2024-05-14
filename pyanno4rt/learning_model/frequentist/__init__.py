"""
Frequentist learning models module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
frequentist learning models, e.g. logistic regression, neural networks and \
support vector machines, including individual preprocessing and evaluation \
pipelines and Bayesian hyperparameter optimization with k-fold cross-\
validation.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._decision_tree import DecisionTreeModel
from ._k_nearest_neighbors import KNeighborsModel
from ._logistic_regression import LogisticRegressionModel
from ._naive_bayes import NaiveBayesModel
from ._neural_network import NeuralNetworkModel
from ._random_forest import RandomForestModel
from ._support_vector_machine import SupportVectorMachineModel

from . import addons

__all__ = ['DecisionTreeModel',
           'KNeighborsModel',
           'LogisticRegressionModel',
           'NaiveBayesModel',
           'NeuralNetworkModel',
           'RandomForestModel',
           'SupportVectorMachineModel',
           'addons']

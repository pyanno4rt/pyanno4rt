"""
Objectives module.

==================================================================

The module aims to provide methods and classes to handle dose-related and \
outcome model-based objective functions for the optimization problem.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._conventional_objective_class import ConventionalObjectiveClass
from ._machine_learning_objective_class import MachineLearningObjectiveClass
from ._radiobiology_objective_class import RadiobiologyObjectiveClass

from ._decision_tree_ntcp import DecisionTreeNTCP
from ._dose_uniformity import DoseUniformity
from ._equivalentUniformDose import EquivalentUniformDose
from ._extreme_gradient_boosting_ntcp import ExtremeGradientBoostingNTCP
from ._k_nearest_neighbors_ntcp import KNeighborsNTCP
from ._logistic_regression_ntcp import LogisticRegressionNTCP
from ._logistic_regression_tcp import LogisticRegressionTCP
from ._lq_poisson_tcp import LQPoissonTCP
from ._lyman_kutcher_burman_ntcp import LymanKutcherBurmanNTCP
from ._maximum_dvh import MaximumDVH
from ._mean_dose import MeanDose
from ._minimum_dvh import MinimumDVH
from ._moments import Moments
from ._naive_bayes_ntcp import NaiveBayesNTCP
from ._neural_network_ntcp import NeuralNetworkNTCP
from ._neural_network_tcp import NeuralNetworkTCP
from ._random_forest_ntcp import RandomForestNTCP
from ._squared_deviation import SquaredDeviation
from ._squared_overdosing import SquaredOverdosing
from ._squared_underdosing import SquaredUnderdosing
from ._support_vector_machine_ntcp import SupportVectorMachineNTCP
from ._support_vector_machine_tcp import SupportVectorMachineTCP

from ._objective_map import objective_map

__all__ = ['ConventionalObjectiveClass',
           'DecisionTreeNTCP',
           'DoseUniformity',
           'EquivalentUniformDose',
           'ExtremeGradientBoostingNTCP',
           'KNeighborsNTCP',
           'LogisticRegressionNTCP',
           'LogisticRegressionTCP',
           'LQPoissonTCP',
           'LymanKutcherBurmanNTCP',
           'MachineLearningObjectiveClass',
           'MaximumDVH',
           'MeanDose',
           'MinimumDVH',
           'Moments',
           'NaiveBayesNTCP',
           'NeuralNetworkNTCP',
           'NeuralNetworkTCP',
           'RadiobiologyObjectiveClass',
           'RandomForestNTCP',
           'SquaredDeviation',
           'SquaredOverdosing',
           'SquaredUnderdosing',
           'SupportVectorMachineNTCP',
           'SupportVectorMachineTCP',
           'objective_map']

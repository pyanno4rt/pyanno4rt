"""
Components module.

==================================================================

The module aims to provide methods and classes to handle dose-related and \
outcome model-based component functions for the optimization problem.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._conventional_component_class import ConventionalComponentClass
from ._machine_learning_component_class import MachineLearningComponentClass
from ._radiobiology_component_class import RadiobiologyComponentClass

from ._decision_tree_ntcp import DecisionTreeNTCP
from ._decision_tree_tcp import DecisionTreeTCP
from ._dose_uniformity import DoseUniformity
from ._equivalent_uniform_dose import EquivalentUniformDose
from ._k_nearest_neighbors_ntcp import KNeighborsNTCP
from ._k_nearest_neighbors_tcp import KNeighborsTCP
from ._logistic_regression_ntcp import LogisticRegressionNTCP
from ._logistic_regression_tcp import LogisticRegressionTCP
from ._lq_poisson_tcp import LQPoissonTCP
from ._lyman_kutcher_burman_ntcp import LymanKutcherBurmanNTCP
from ._maximum_dvh import MaximumDVH
from ._mean_dose import MeanDose
from ._minimum_dvh import MinimumDVH
from ._naive_bayes_ntcp import NaiveBayesNTCP
from ._naive_bayes_tcp import NaiveBayesTCP
from ._neural_network_ntcp import NeuralNetworkNTCP
from ._neural_network_tcp import NeuralNetworkTCP
from ._random_forest_ntcp import RandomForestNTCP
from ._random_forest_tcp import RandomForestTCP
from ._squared_deviation import SquaredDeviation
from ._squared_overdosing import SquaredOverdosing
from ._squared_underdosing import SquaredUnderdosing
from ._support_vector_machine_ntcp import SupportVectorMachineNTCP
from ._support_vector_machine_tcp import SupportVectorMachineTCP

from ._component_map import component_map

__all__ = ['ConventionalComponentClass',
           'DecisionTreeNTCP',
           'DecisionTreeTCP',
           'DoseUniformity',
           'EquivalentUniformDose',
           'KNeighborsNTCP',
           'KNeighborsTCP',
           'LogisticRegressionNTCP',
           'LogisticRegressionTCP',
           'LQPoissonTCP',
           'LymanKutcherBurmanNTCP',
           'MachineLearningComponentClass',
           'MaximumDVH',
           'MeanDose',
           'MinimumDVH',
           'NaiveBayesNTCP',
           'NaiveBayesTCP',
           'NeuralNetworkNTCP',
           'NeuralNetworkTCP',
           'RadiobiologyComponentClass',
           'RandomForestNTCP',
           'RandomForestTCP',
           'SquaredDeviation',
           'SquaredOverdosing',
           'SquaredUnderdosing',
           'SupportVectorMachineNTCP',
           'SupportVectorMachineTCP',
           'component_map']

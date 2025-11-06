"""
Components module.

==================================================================

The module aims to provide methods and classes to handle dose-related and \
outcome model-based component functions for the optimization problem.
"""

# Author: Tim Ortkamp

# Import the base classes
from ._conventional_component import ConventionalComponent
from ._machine_learning_component import MachineLearningComponent
from ._radiobiological_component import RadiobiologicalComponent

# Import the component classes
from ._decision_tree_outcome import DecisionTreeOutcome
from ._dose_uniformity import DoseUniformity
from ._equivalent_uniform_dose import EquivalentUniformDose
from ._k_nearest_neighbors_outcome import KNeighborsOutcome
from ._logistic_regression_outcome import LogisticRegressionOutcome
from ._lq_poisson_tcp import LQPoissonTCP
from ._lyman_kutcher_burman_ntcp import LymanKutcherBurmanNTCP
from ._maximum_dvh import MaximumDVH
from ._mean_dose import MeanDose
from ._minimum_dvh import MinimumDVH
from ._naive_bayes_outcome import NaiveBayesOutcome
from ._neural_network_outcome import NeuralNetworkOutcome
from ._random_forest_outcome import RandomForestOutcome
from ._squared_deviation import SquaredDeviation
from ._squared_overdosing import SquaredOverdosing
from ._squared_underdosing import SquaredUnderdosing
from ._support_vector_machine_outcome import SupportVectorMachineOutcome

__all__ = [
    'ConventionalComponent',
    'DecisionTreeOutcome',
    'DoseUniformity',
    'EquivalentUniformDose',
    'KNeighborsOutcome',
    'LogisticRegressionOutcome',
    'LQPoissonTCP',
    'LymanKutcherBurmanNTCP',
    'MachineLearningComponent',
    'MaximumDVH',
    'MeanDose',
    'MinimumDVH',
    'NaiveBayesOutcome',
    'NeuralNetworkOutcome',
    'RadiobiologicalComponent',
    'RandomForestOutcome',
    'SquaredDeviation',
    'SquaredOverdosing',
    'SquaredUnderdosing',
    'SupportVectorMachineOutcome']

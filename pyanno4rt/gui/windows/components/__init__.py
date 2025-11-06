"""
GUI component windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from ._decision_tree_ntcp_window import DecisionTreeNTCPWindow
from ._dose_uniformity_window import DoseUniformityWindow
from ._equivalent_uniform_dose_window import EquivalentUniformDoseWindow
from ._k_nearest_neighbors_ntcp_window import KNeighborsNTCPWindow
from ._logistic_regression_ntcp_window import LogisticRegressionNTCPWindow
from ._lq_poisson_tcp_window import LQPoissonTCPWindow
from ._lkb_ntcp_window import LKBNTCPWindow
from ._maximum_dvh_window import MaximumDVHWindow
from ._mean_dose_window import MeanDoseWindow
from ._minimum_dvh_window import MinimumDVHWindow
from ._naive_bayes_ntcp_window import NaiveBayesNTCPWindow
from ._neural_network_ntcp_window import NeuralNetworkNTCPWindow
from ._random_forest_ntcp_window import RandomForestNTCPWindow
from ._squared_deviation_window import SquaredDeviationWindow
from ._squared_overdosing_window import SquaredOverdosingWindow
from ._squared_underdosing_window import SquaredUnderdosingWindow
from ._support_vector_machine_ntcp_window import SupportVectorMachineNTCPWindow

from ._component_window_map import component_window_map

__all__ = [
    'DecisionTreeNTCPWindow',
    'DoseUniformityWindow',
    'EquivalentUniformDoseWindow',
    'KNeighborsNTCPWindow',
    'LogisticRegressionNTCPWindow',
    'LQPoissonTCPWindow',
    'LKBNTCPWindow',
    'MaximumDVHWindow',
    'MeanDoseWindow',
    'MinimumDVHWindow',
    'NaiveBayesNTCPWindow',
    'NeuralNetworkNTCPWindow',
    'RandomForestNTCPWindow',
    'SquaredDeviationWindow',
    'SquaredOverdosingWindow',
    'SquaredUnderdosingWindow',
    'SupportVectorMachineNTCPWindow',
    'component_window_map']

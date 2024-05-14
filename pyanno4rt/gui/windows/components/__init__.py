"""
GUI component windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._decision_tree_ntcp_window import DecisionTreeNTCPWindow
from ._decision_tree_tcp_window import DecisionTreeTCPWindow
from ._dose_uniformity_window import DoseUniformityWindow
from ._equivalent_uniform_dose_window import EquivalentUniformDoseWindow
from ._k_nearest_neighbors_ntcp_window import KNeighborsNTCPWindow
from ._k_nearest_neighbors_tcp_window import KNeighborsTCPWindow
from ._logistic_regression_ntcp_window import LogisticRegressionNTCPWindow
from ._logistic_regression_tcp_window import LogisticRegressionTCPWindow
from ._lq_poisson_tcp_window import LQPoissonTCPWindow
from ._lkb_ntcp_window import LKBNTCPWindow
from ._maximum_dvh_window import MaximumDVHWindow
from ._mean_dose_window import MeanDoseWindow
from ._minimum_dvh_window import MinimumDVHWindow
from ._naive_bayes_ntcp_window import NaiveBayesNTCPWindow
from ._naive_bayes_tcp_window import NaiveBayesTCPWindow
from ._neural_network_ntcp_window import NeuralNetworkNTCPWindow
from ._neural_network_tcp_window import NeuralNetworkTCPWindow
from ._random_forest_ntcp_window import RandomForestNTCPWindow
from ._random_forest_tcp_window import RandomForestTCPWindow
from ._squared_deviation_window import SquaredDeviationWindow
from ._squared_overdosing_window import SquaredOverdosingWindow
from ._squared_underdosing_window import SquaredUnderdosingWindow
from ._support_vector_machine_ntcp_window import SupportVectorMachineNTCPWindow
from ._support_vector_machine_tcp_window import SupportVectorMachineTCPWindow

from ._component_window_map import component_window_map

__all__ = ['DecisionTreeNTCPWindow',
           'DecisionTreeTCPWindow',
           'DoseUniformityWindow',
           'EquivalentUniformDoseWindow',
           'KNeighborsNTCPWindow',
           'KNeighborsTCPWindow',
           'LogisticRegressionNTCPWindow',
           'LogisticRegressionTCPWindow',
           'LQPoissonTCPWindow',
           'LKBNTCPWindow',
           'MaximumDVHWindow',
           'MeanDoseWindow',
           'MinimumDVHWindow',
           'NaiveBayesNTCPWindow',
           'NaiveBayesTCPWindow',
           'NeuralNetworkNTCPWindow',
           'NeuralNetworkTCPWindow',
           'RandomForestNTCPWindow',
           'RandomForestTCPWindow',
           'SquaredDeviationWindow',
           'SquaredOverdosingWindow',
           'SquaredUnderdosingWindow',
           'SupportVectorMachineNTCPWindow',
           'SupportVectorMachineTCPWindow',
           'component_window_map']

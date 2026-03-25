"""
GUI component windows module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from pyanno4rt.gui.windows.components._decision_tree_ntcp_window import DecisionTreeNTCPWindow
from pyanno4rt.gui.windows.components._dose_uniformity_window import DoseUniformityWindow
from pyanno4rt.gui.windows.components._equivalent_uniform_dose_window import EquivalentUniformDoseWindow
from pyanno4rt.gui.windows.components._k_nearest_neighbors_ntcp_window import KNeighborsNTCPWindow
from pyanno4rt.gui.windows.components._logistic_regression_ntcp_window import LogisticRegressionNTCPWindow
from pyanno4rt.gui.windows.components._lq_poisson_tcp_window import LQPoissonTCPWindow
from pyanno4rt.gui.windows.components._lkb_ntcp_window import LKBNTCPWindow
from pyanno4rt.gui.windows.components._maximum_dvh_window import MaximumDVHWindow
from pyanno4rt.gui.windows.components._mean_dose_window import MeanDoseWindow
from pyanno4rt.gui.windows.components._minimum_dvh_window import MinimumDVHWindow
from pyanno4rt.gui.windows.components._naive_bayes_ntcp_window import NaiveBayesNTCPWindow
from pyanno4rt.gui.windows.components._neural_network_ntcp_window import NeuralNetworkNTCPWindow
from pyanno4rt.gui.windows.components._random_forest_ntcp_window import RandomForestNTCPWindow
from pyanno4rt.gui.windows.components._squared_deviation_window import SquaredDeviationWindow
from pyanno4rt.gui.windows.components._squared_overdosing_window import SquaredOverdosingWindow
from pyanno4rt.gui.windows.components._squared_underdosing_window import SquaredUnderdosingWindow
from pyanno4rt.gui.windows.components._support_vector_machine_ntcp_window import SupportVectorMachineNTCPWindow

from pyanno4rt.gui.windows.components._component_window_map import component_window_map

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

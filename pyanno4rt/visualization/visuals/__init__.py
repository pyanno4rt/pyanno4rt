"""
Visual elements module.

==================================================================

The module aims to provide methods and classes to be embedded via clickable \
buttons in the visualization interface.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from pyanno4rt.visualization.visuals._ct_dose_slicing_window_pyqt import (
    CtDoseSlicingWindowPyQt)
from pyanno4rt.visualization.visuals._dosimetrics_table_plotter_mpl import (
    DosimetricsTablePlotterMPL)
from pyanno4rt.visualization.visuals._dvh_graph_plotter_mpl import (
    DVHGraphPlotterMPL)
from pyanno4rt.visualization.visuals._feature_select_window_pyqt import (
    FeatureSelectWindowPyQt)
from pyanno4rt.visualization.visuals._iter_graph_plotter_mpl import (
    IterGraphPlotterMPL)
from pyanno4rt.visualization.visuals._metrics_graphs_plotter_mpl import (
    MetricsGraphsPlotterMPL)
from pyanno4rt.visualization.visuals._metrics_tables_plotter_mpl import (
    MetricsTablesPlotterMPL)
from pyanno4rt.visualization.visuals._ntcp_graph_plotter_mpl import (
    NTCPGraphPlotterMPL)
from pyanno4rt.visualization.visuals._permutation_importance_plotter_mpl import (
    PermutationImportancePlotterMPL)

__all__ = ['CtDoseSlicingWindowPyQt',
           'DosimetricsTablePlotterMPL',
           'DVHGraphPlotterMPL',
           'FeatureSelectWindowPyQt',
           'IterGraphPlotterMPL',
           'MetricsGraphsPlotterMPL',
           'MetricsTablesPlotterMPL',
           'NTCPGraphPlotterMPL',
           'PermutationImportancePlotterMPL']

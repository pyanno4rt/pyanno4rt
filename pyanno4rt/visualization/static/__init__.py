"""
Visual elements module.

==================================================================

The module aims to provide methods and classes to be embedded via clickable \
buttons in the visualization interface.
"""

# Author: Tim Ortkamp

from pyanno4rt.visualization.static._dosimetrics_table import DosimetricsTable
from pyanno4rt.visualization.static._dvh_graph import DVHGraph
from pyanno4rt.visualization.static._iter_graph import IterGraph
from pyanno4rt.visualization.static._metrics_graph import MetricsGraph
from pyanno4rt.visualization.static._metrics_table import MetricsTable
from pyanno4rt.visualization.static._ntcp_graph import NTCPGraph
from pyanno4rt.visualization.static._permutation_importance_boxplot import (
    PermutationImportanceBoxplot)

__all__ = [
    'DosimetricsTable',
    'DVHGraph',
    'IterGraph',
    'MetricsGraph',
    'MetricsTable',
    'NTCPGraph',
    'PermutationImportanceBoxplot']

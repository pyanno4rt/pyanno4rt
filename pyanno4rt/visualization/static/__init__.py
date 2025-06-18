"""
Visual elements module.

==================================================================

The module aims to provide methods and classes for embedding static plots in \
the visualization window.
"""

# Author: Tim Ortkamp

from pyanno4rt.visualization.static._component_graph import ComponentGraph
from pyanno4rt.visualization.static._dosimetrics_table import DosimetricsTable
from pyanno4rt.visualization.static._dvh_graph import DVHGraph
from pyanno4rt.visualization.static._metrics_graph import MetricsGraph
from pyanno4rt.visualization.static._metrics_table import MetricsTable
from pyanno4rt.visualization.static._outcome_graph import OutcomeGraph
from pyanno4rt.visualization.static._permutation_importance_boxplot import (
    PermutationImportanceBoxplot)

__all__ = [
    'ComponentGraph',
    'DosimetricsTable',
    'DVHGraph',
    'MetricsGraph',
    'MetricsTable',
    'OutcomeGraph',
    'PermutationImportanceBoxplot']

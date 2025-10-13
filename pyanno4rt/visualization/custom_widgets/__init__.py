"""
Custom widgets module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from ._checkable_combo_box import CheckableComboBox
from ._component_graph_widget import ComponentGraphWidget
from ._dvh_graph_widget import DVHGraphWidget
from ._feature_graph_widget import FeatureGraphWidget
from ._outcome_graph_widget import OutcomeGraphWidget
from ._permutation_importance_widget import PermutationImportanceWidget
from ._slice_widget import SliceWidget

__all__ = [
    'CheckableComboBox',
    'ComponentGraphWidget',
    'DVHGraphWidget',
    'FeatureGraphWidget',
    'OutcomeGraphWidget',
    'PermutationImportanceWidget',
    'SliceWidget']

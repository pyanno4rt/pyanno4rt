"""
Custom widgets module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from pyanno4rt.visualization.custom_widgets._checkable_combo_box import CheckableComboBox
from pyanno4rt.visualization.custom_widgets._component_graph_widget import ComponentGraphWidget
from pyanno4rt.visualization.custom_widgets._dvh_graph_widget import DVHGraphWidget
from pyanno4rt.visualization.custom_widgets._feature_graph_widget import FeatureGraphWidget
from pyanno4rt.visualization.custom_widgets._outcome_graph_widget import OutcomeGraphWidget
from pyanno4rt.visualization.custom_widgets._permutation_importance_widget import PermutationImportanceWidget
from pyanno4rt.visualization.custom_widgets._slice_widget import SliceWidget

__all__ = [
    'CheckableComboBox',
    'ComponentGraphWidget',
    'DVHGraphWidget',
    'FeatureGraphWidget',
    'OutcomeGraphWidget',
    'PermutationImportanceWidget',
    'SliceWidget']

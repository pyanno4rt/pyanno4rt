"""
Custom widgets module.

==================================================================

The module aims to provide methods and classes to ...
"""

# Author: Tim Ortkamp

from ._checkable_combo_box import CheckableComboBox
from ._dvh_graph_compare_widget import DVHGraphCompareWidget
from ._dvh_graph_widget import DVHGraphWidget
from ._slice_compare_widget import SliceCompareWidget
from ._slice_widget import SliceWidget

__all__ = [
    'CheckableComboBox',
    'DVHGraphCompareWidget',
    'DVHGraphWidget',
    'SliceCompareWidget',
    'SliceWidget']

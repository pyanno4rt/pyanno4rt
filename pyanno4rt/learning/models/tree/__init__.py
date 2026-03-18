"""
Decision tree module.

==================================================================

The module aims to provide methods and classes for decision tree outcome \
modeling.
"""

# Author: Tim Ortkamp

from ._projection_tree import ProjectionTree

from ._decision_tree import DecisionTree
from ._soft_decision_tree import SoftDecisionTree

__all__ = [
    'ProjectionTree',
    'DecisionTree',
    'SoftDecisionTree']

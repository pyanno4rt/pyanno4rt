"""
Decision tree module.

==================================================================

The module aims to provide methods and classes for decision tree outcome \
modeling.
"""

# Author: Tim Ortkamp

from ._optimizable_decision_tree import OptimizableDecisionTree
from ._soft_decision_tree import SoftDecisionTree

from ._decision_tree import DecisionTree

__all__ = [
    'OptimizableDecisionTree',
    'SoftDecisionTree',
    'DecisionTree']

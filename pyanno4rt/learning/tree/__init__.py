"""
Decision tree model module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
decision tree models.
"""

# Author: Tim Ortkamp

from ._decision_tree import DecisionTreeModel
from ._optimizable_decision_tree import OptimizableDecisionTree
from ._soft_decision_tree import SoftDecisionTree

__all__ = [
    'DecisionTreeModel',
    'OptimizableDecisionTree',
    'SoftDecisionTree']

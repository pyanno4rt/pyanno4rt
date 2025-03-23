"""
Decision tree model module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
decision tree models.
"""

# Author: Tim Ortkamp

from ._optimizable_decision_tree import OptimizableDecisionTree
from ._decision_tree import DecisionTreeModel

__all__ = [
    'OptimizableDecisionTree',
    'DecisionTreeModel']

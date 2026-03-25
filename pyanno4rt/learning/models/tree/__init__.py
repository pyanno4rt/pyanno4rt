"""
Decision tree module.

==================================================================

The module aims to provide methods and classes for decision tree outcome \
modeling.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.models.tree._projection_tree import ProjectionTree
from pyanno4rt.learning.models.tree._decision_tree import DecisionTree

from pyanno4rt.learning.models.tree._soft_tree import SoftTree
from pyanno4rt.learning.models.tree._soft_decision_tree import SoftDecisionTree

__all__ = [
    'ProjectionTree',
    'DecisionTree',
    'SoftTree',
    'SoftDecisionTree']

"""
Random forest model module.

==================================================================

The module aims to provide methods and classes for random forest modeling.
"""

# Author: Tim Ortkamp

from ._optimizable_random_forest import OptimizableRandomForest
from ._random_forest import RandomForestModel

__all__ = [
    'OptimizableRandomForest',
    'RandomForestModel']

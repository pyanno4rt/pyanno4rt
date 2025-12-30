"""
Random forest module.

==================================================================

The module aims to provide methods and classes for random forest outcome \
modeling.
"""

# Author: Tim Ortkamp

from ._optimizable_random_forest import OptimizableRandomForest
from ._random_forest import RandomForest

__all__ = [
    'OptimizableRandomForest',
    'RandomForest']

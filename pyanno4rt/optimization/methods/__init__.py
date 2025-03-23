"""
Optimization methods module.

==================================================================

This module aims to provide different types of optimization methods.
"""

# Author: Tim Ortkamp

from ._lexicographic_optimization import LexicographicOptimization
from ._pareto_optimization import ParetoOptimization
from ._weighted_sum_optimization import WeightedSumOptimization

__all__ = [
    'LexicographicOptimization',
    'ParetoOptimization',
    'WeightedSumOptimization']

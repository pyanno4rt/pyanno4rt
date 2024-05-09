"""
Optimization methods module.

==================================================================

This module aims to provide different types of optimization methods.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._lexicographic_optimization import LexicographicOptimization
from ._pareto_optimization import ParetoOptimization
from ._weighted_sum_optimization import WeightedSumOptimization

from ._method_map import method_map

__all__ = ['LexicographicOptimization',
           'ParetoOptimization',
           'WeightedSumOptimization',
           'method_map']

"""
Optimization problems module.

==================================================================

The module aims to provide different optimization problem formulations, e.g.
weighted-sum approach or Pareto analysis.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._lexicographic_optimization import LexicographicOptimization
from ._pareto_optimization import ParetoOptimization
from ._weighted_sum_optimization import WeightedSumOptimization

from ._methods_map import methods_map

__all__ = ['LexicographicOptimization',
           'ParetoOptimization',
           'WeightedSumOptimization',
           'methods_map']

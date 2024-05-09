"""Optimization methods map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.methods import (
    LexicographicOptimization, ParetoOptimization, WeightedSumOptimization)

# %% Map definition


method_map = {'lexicographic': LexicographicOptimization,
              'pareto': ParetoOptimization,
              'weighted-sum': WeightedSumOptimization}

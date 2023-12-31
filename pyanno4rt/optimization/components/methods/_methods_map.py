"""Optimization methods map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.components.methods import (
    LexicographicOptimization, ParetoOptimization, WeightedSumOptimization)

# %% Map definition


methods_map = {'lexicographic': LexicographicOptimization,
               'pareto': ParetoOptimization,
               'weighted-sum': WeightedSumOptimization}

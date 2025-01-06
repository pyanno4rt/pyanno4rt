"""
Extensions module.

==================================================================

The module aims to provide extension files for the frequentist learning models.
"""

# Author: Tim Ortkamp

from ._neural_network_architectures import (
    build_vanilla_iocnn, build_vanilla_nn)
from ._neural_network_maps import loss_map, optimizer_map
from ._optimizable_decision_tree import OptimizableDecisionTree
from ._optimizable_random_forest import OptimizableRandomForest
from ._support_vector_machine_decision_functions import (
    linear_decision_function, linear_decision_gradient, rbf_decision_function,
    rbf_decision_gradient, poly_decision_function, poly_decision_gradient,
    sigmoid_decision_function, sigmoid_decision_gradient)

__all__ = ['build_vanilla_iocnn',
           'build_vanilla_nn',
           'loss_map',
           'optimizer_map',
           'OptimizableDecisionTree',
           'OptimizableRandomForest',
           'linear_decision_function',
           'linear_decision_gradient',
           'rbf_decision_function',
           'rbf_decision_gradient',
           'poly_decision_function',
           'poly_decision_gradient',
           'sigmoid_decision_function',
           'sigmoid_decision_gradient']

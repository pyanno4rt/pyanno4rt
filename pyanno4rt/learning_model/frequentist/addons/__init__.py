"""
Additional model files module.

==================================================================

The module aims to provide functions as a supplement for the frequentist \
learning models.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._neural_network_architectures import (
    build_iocnn,
    build_standard_nn)

from ._neural_network_maps import loss_map, optimizer_map

from ._support_vector_machine_decision_functions import (
    linear_decision_function,
    rbf_decision_function,
    poly_decision_function,
    sigmoid_decision_function,
    linear_decision_gradient,
    rbf_decision_gradient,
    poly_decision_gradient,
    sigmoid_decision_gradient)

__all__ = ['build_iocnn',
           'build_standard_nn',
           'loss_map',
           'optimizer_map',
           'linear_decision_function',
           'rbf_decision_function',
           'poly_decision_function',
           'sigmoid_decision_function',
           'linear_decision_gradient',
           'rbf_decision_gradient',
           'poly_decision_gradient',
           'sigmoid_decision_gradient']

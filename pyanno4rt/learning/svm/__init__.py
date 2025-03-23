"""
Support vector machine model module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
Support vector machine models.
"""

# Author: Tim Ortkamp

from ._support_vector_machine_decision_functions import (
    linear_decision_function, linear_decision_gradient,
    poly_decision_function, poly_decision_gradient,
    rbf_decision_function, rbf_decision_gradient,
    sigmoid_decision_function, sigmoid_decision_gradient)
from ._support_vector_machine import SupportVectorMachineModel

__all__ = [
    'linear_decision_function',
    'linear_decision_gradient',
    'poly_decision_function',
    'poly_decision_gradient',
    'rbf_decision_function',
    'rbf_decision_gradient',
    'sigmoid_decision_function',
    'sigmoid_decision_gradient',
    'SupportVectorMachineModel']

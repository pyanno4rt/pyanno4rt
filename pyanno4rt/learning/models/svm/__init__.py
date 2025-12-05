"""
Support vector machine model module.

==================================================================

The module aims to provide methods and classes for modeling NTCP and TCP with \
Support vector machine models.
"""

# Author: Tim Ortkamp

from ._support_vector_machine_gradients import (
    linear_gradient, poly_gradient, rbf_gradient, sigmoid_gradient)
from ._support_vector_machine import SupportVectorMachineModel

__all__ = [
    'linear_gradient',
    'poly_gradient',
    'rbf_gradient',
    'sigmoid_gradient',
    'SupportVectorMachineModel']

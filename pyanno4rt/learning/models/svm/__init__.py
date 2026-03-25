"""
Support vector machine module.

==================================================================

The module aims to provide methods and classes for support vector machine \
outcome modeling.
"""

# Author: Tim Ortkamp

from pyanno4rt.learning.models.svm._support_vector_machine_gradients import (
    linear_gradient, poly_gradient, rbf_gradient, sigmoid_gradient)
from pyanno4rt.learning.models.svm._support_vector_machine import SupportVectorMachine

__all__ = [
    'linear_gradient',
    'poly_gradient',
    'rbf_gradient',
    'sigmoid_gradient',
    'SupportVectorMachine']

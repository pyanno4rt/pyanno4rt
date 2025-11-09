"""
Optimization module.

==================================================================

This module aims to provide methods and classes for setting up and solving \
the inverse planning problem.
"""

# Author: Tim Ortkamp

# Import the submodules
from . import components, initializers, problems, projections, solvers

# Import the fluence optimizer
from ._fluence_optimizer import FluenceOptimizer

__all__ = [
    'components',
    'initializers',
    'problems',
    'projections',
    'solvers',
    'FluenceOptimizer']

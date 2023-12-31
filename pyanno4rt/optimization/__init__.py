"""
Optimization module.

==================================================================

The module aims to provide methods and classes for setting up and solving the \
inverse planning problem, including objectives and constraints for single- \
and multi-criteria optimization, multiple solvers, and dose-fluence \
projections to handle both photon and proton treatments.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._fluence_initializer import FluenceInitializer
from ._fluence_optimizer import FluenceOptimizer

from . import components
from . import projections
from . import solvers

__all__ = ['FluenceInitializer',
           'FluenceOptimizer',
           'components',
           'projections',
           'solvers']

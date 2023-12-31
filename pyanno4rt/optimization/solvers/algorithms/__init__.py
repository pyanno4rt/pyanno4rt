"""
Solver algorithms module.

==================================================================

The module aims to provide functions to get the callable algorithms together \
with their configurations. It is designed to hold one extensible file per \
solver, each of which specifies the available algorithms.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._proxmin_algorithms import get_proxmin_configuration
from ._pypop7_algorithms import get_pypop7_configuration
from ._scipy_algorithms import get_scipy_configuration

__all__ = ['get_proxmin_configuration',
           'get_pypop7_configuration',
           'get_scipy_configuration']

"""
Solution algorithms module.

==================================================================

The module aims to provide functions to configure the solution algorithms for \
the optimization packages.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._configure_proxmin import configure_proxmin
from ._configure_scipy import configure_scipy

__all__ = ['configure_proxmin',
           'configure_scipy']

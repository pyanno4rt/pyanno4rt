"""
Optimization components module.

==================================================================

The module aims to provide methods and classes to represent the optimization \
components and different optimization methods.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from . import constraints
from . import methods
from . import objectives

__all__ = ['constraints',
           'methods',
           'objectives']

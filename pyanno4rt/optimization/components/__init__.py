"""
Optimization components module.

==================================================================

The module aims to provide methods and classes to represent the optimization \
components, i.e., objective and constraint functions, and different \
optimization methods, e.g. weighted-sum approach or Pareto analysis.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from . import constraints
from . import methods
from . import objectives

__all__ = ['constraints',
           'methods',
           'objectives']

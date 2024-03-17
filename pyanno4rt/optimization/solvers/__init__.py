"""
Solvers module.

==================================================================

The module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the integrated optimization packages.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._proxmin_solver import ProxminSolver
from ._pymoo_solver import PymooSolver
from ._scipy_solver import SciPySolver

from ._solver_map import solver_map

__all__ = ['ProxminSolver',
           'PymooSolver',
           'SciPySolver',
           'solver_map']

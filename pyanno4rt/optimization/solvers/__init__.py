"""
Solvers module.

==================================================================

This module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the integrated optimization packages.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

from ._proxmin_solver import ProxminSolver
from ._pyanno4rt_solver import Pyanno4rtSolver
from ._pymoo_solver import PymooSolver
from ._pypop7_solver import PyPop7Solver
from ._scipy_solver import SciPySolver

from ._solver_map import solver_map

from . import configurations
from . import internals

__all__ = ['ProxminSolver',
           'Pyanno4rtSolver',
           'PymooSolver',
           'PyPop7Solver',
           'SciPySolver',
           'solver_map',
           'configurations',
           'internals']

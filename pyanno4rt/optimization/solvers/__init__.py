"""
Solvers module.

==================================================================

This module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the external optimization packages as well as \
setting up the internal custom implementations.
"""

# Author: Tim Ortkamp

from . import custom

from ._ipyopt_solver import IpyoptSolver
from ._pyanno4rt_solver import Pyanno4rtSolver
from ._pymoo_solver import PymooSolver
from ._pypop7_solver import PyPop7Solver
from ._scipy_solver import SciPySolver

__all__ = [
    'custom',
    'IpyoptSolver',
    'Pyanno4rtSolver',
    'PymooSolver',
    'PyPop7Solver',
    'SciPySolver']

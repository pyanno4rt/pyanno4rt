"""
Solvers module.

==================================================================

This module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the external optimization packages.
"""

# Author: Tim Ortkamp

from ._ipyopt_solver import IpyoptSolver
from ._pymoo_solver import PymooSolver
from ._pypop7_solver import PyPop7Solver
from ._scipy_solver import SciPySolver

__all__ = [
    'IpyoptSolver',
    'PymooSolver',
    'PyPop7Solver',
    'SciPySolver']

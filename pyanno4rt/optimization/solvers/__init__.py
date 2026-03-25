"""
Solvers module.

==================================================================

This module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the external optimization packages as well as \
setting up the internal pyanno4rt implementations.
"""

# Author: Tim Ortkamp

from pyanno4rt.optimization.solvers import cmaes

from pyanno4rt.optimization.solvers._ipyopt_solver import IpyoptSolver
from pyanno4rt.optimization.solvers._pyanno4rt_solver import Pyanno4rtSolver
from pyanno4rt.optimization.solvers._pymoo_solver import PymooSolver
from pyanno4rt.optimization.solvers._pypop7_solver import PyPop7Solver
from pyanno4rt.optimization.solvers._scipy_solver import SciPySolver

__all__ = [
    'cmaes',
    'IpyoptSolver',
    'Pyanno4rtSolver',
    'PymooSolver',
    'PyPop7Solver',
    'SciPySolver']

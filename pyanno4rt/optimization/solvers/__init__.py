"""
Solvers module.

==================================================================

This module aims to provide methods and classes for wrapping the local and \
global solution algorithms from the external optimization packages as well as \
setting up the internal pyanno4rt implementations.
"""

# Author: Tim Ortkamp

from pyanno4rt.optimization.solvers._ipyopt_solver import IpyoptSolver
from pyanno4rt.optimization.solvers._pymoo_solver import PymooSolver
from pyanno4rt.optimization.solvers._pypop7_solver import PyPop7Solver
from pyanno4rt.optimization.solvers._scipy_solver import SciPySolver
from pyanno4rt.optimization.solvers._seamaze_solver import SeaMazeSolver

__all__ = [
    'IpyoptSolver',
    'PymooSolver',
    'PyPop7Solver',
    'SciPySolver',
    'SeaMazeSolver']

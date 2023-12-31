"""
Solvers module.

==================================================================

The module aims to provide methods and classes for wrapping and running the \
local and global solution algorithms from the integrated solver packages.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

try:
    from ._ipopt_solver import IpoptSolver
except ImportError:
    IpoptSolver = None
from ._proxmin_solver import ProxminSolver
from ._pymoo_solver import PymooSolver
from ._pypop7_solver import Pypop7Solver
from ._scipy_solver import SciPySolver

from ._solvers_map import solvers_map

__all__ = ['IpoptSolver',
           'ProxminSolver',
           'PymooSolver',
           'Pypop7Solver',
           'SciPySolver',
           'solvers_map']

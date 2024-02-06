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
from ._scipy_solver import SciPySolver

from ._solver_map import solver_map

__all__ = ['IpoptSolver',
           'ProxminSolver',
           'PymooSolver',
           'SciPySolver',
           'solver_map']

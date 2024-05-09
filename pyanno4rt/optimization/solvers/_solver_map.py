"""Solvers map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.solvers import (
    ProxminSolver, Pyanno4rtSolver, PymooSolver, PyPop7Solver, SciPySolver)

# %% Map definition


solver_map = {'proxmin': ProxminSolver,
              'pyanno4rt': Pyanno4rtSolver,
              'pymoo': PymooSolver,
              'pypop7': PyPop7Solver,
              'scipy': SciPySolver}

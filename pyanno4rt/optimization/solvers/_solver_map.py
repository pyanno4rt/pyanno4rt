"""Solvers map."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.optimization.solvers import (
    IpyoptSolver, ProxminSolver, PymooSolver, PyPop7Solver,
    SciPySolver)

# %% Map definition


solver_map = {'ipyopt': IpyoptSolver,
              'proxmin': ProxminSolver,
              # 'pyanno4rt': Pyanno4rtSolver,
              'pymoo': PymooSolver,
              'pypop7': PyPop7Solver,
              'scipy': SciPySolver}

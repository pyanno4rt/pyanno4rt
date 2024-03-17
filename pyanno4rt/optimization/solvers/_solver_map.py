"""Solvers map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.solvers import (
    ProxminSolver, PymooSolver, SciPySolver)

# %% Map definition


solver_map = {'proxmin': ProxminSolver,
              'pymoo': PymooSolver,
              'scipy': SciPySolver}

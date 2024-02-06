"""Solvers map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.solvers import (
    IpoptSolver, ProxminSolver, PymooSolver, SciPySolver)

# %% Map definition


solver_map = {'ipopt': IpoptSolver,
              'proxmin': ProxminSolver,
              'pymoo': PymooSolver,
              'scipy': SciPySolver}

"""Solvers map."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.optimization.solvers import (
    IpoptSolver, ProxminSolver, PymooSolver, Pypop7Solver, SciPySolver)

# %% Map definition


solvers_map = {'ipopt': IpoptSolver,
               'proxmin': ProxminSolver,
               'pymoo': PymooSolver,
               'pypop7': Pypop7Solver,
               'scipy': SciPySolver}

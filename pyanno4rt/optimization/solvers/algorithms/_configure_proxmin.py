"""Proxmin algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, clip
from proxmin import admm, pgm, sdmm
from scipy.optimize import line_search

# %% Function definition


def configure_proxmin(problem_instance, lower_variable_bounds,
                      upper_variable_bounds, algorithm, max_iter):
    """
    Get the optimizer function and arguments for the Proxmin solver.

    Supported algorithms: ADMM, PGM, SDMM
    """

    def get_objective_gradient(X):
        """Get the objective gradient for the current solution."""
        return problem_instance.gradient(X)[:, None]

    def estimate_lipschitz(X, it=0):
        """Estimate the Lipschitz constant of the gradient function."""
        X = X.reshape(-1)
        L = line_search(partial(problem_instance.objective, track=False),
                        problem_instance.gradient,
                        X, -problem_instance.gradient(X))
        if not L[0]:
            return 1e-100

        return L[0]/2

    def project_on_bounds(X, step):
        """Project the current solution on the bounded set."""
        return clip(X, a_min=lower_variable_bounds,
                    a_max=upper_variable_bounds)

    def perform_proximal_grad_step(X, step):
        """Perform a proximal gradient step."""
        return X - step*get_objective_gradient(X)

    # Convert the bound lists to 2D arrays
    lower_variable_bounds = array(lower_variable_bounds)[:, None]
    upper_variable_bounds = array(upper_variable_bounds)[:, None]

    # Check if the algorithm is 'admm'
    if algorithm == 'admm':

        # Set the solution function
        fun = admm

        # Set the function arguments
        arguments = {'prox_f': perform_proximal_grad_step,
                     'step_f': estimate_lipschitz,
                     'prox_g': project_on_bounds,
                     'step_g': None,
                     'L': None,
                     'e_rel': 5e-3,
                     'e_abs': 5e-3,
                     'max_iter': max_iter}

    # Else, check if the algorithm is 'pgm'
    elif algorithm == 'pgm':

        # Set the solution function
        fun = pgm

        # Set the function arguments
        arguments = {'grad': get_objective_gradient,
                     'step': estimate_lipschitz,
                     'prox': project_on_bounds,
                     'accelerated': True,
                     'backtracking': False,
                     'f': None,
                     'e_rel': 5e-3,
                     'max_iter': max_iter}

    # Else, check if the algorithm is 'sdmm'
    elif algorithm == 'sdmm':

        # Set the solution function
        fun = sdmm

        # Set the function arguments
        arguments = {'prox_f': perform_proximal_grad_step,
                     'step_f': estimate_lipschitz,
                     'proxs_g': [project_on_bounds],
                     'steps_g': None,
                     'Ls': None,
                     'e_rel': 5e-3,
                     'e_abs': 5e-3,
                     'max_iter': max_iter}

    return fun, arguments

"""Proxmin algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, clip
from proxmin import admm, pgm, sdmm
from scipy.optimize import line_search

# %% Function definition


def configure_proxmin(problem_instance, lower_variable_bounds,
                      upper_variable_bounds, lower_constraint_bounds,
                      upper_constraint_bounds, algorithm, max_iter, tolerance,
                      callback):
    """
    Configure the Proxmin solver.

    Supported algorithms: ADMM, PGM, SDMM.

    Parameters
    ----------
    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.methods._lexicographic_optimization.LexicographicOptimization`\
        :class:`~pyanno4rt.optimization.methods._weighted_sum_optimization.WeightedSumOptimization`
        The object representing the optimization problem.

    lower_variable_bounds : list
        Lower bounds on the decision variables.

    upper_variable_bounds : list
        Upper bounds on the decision variables.

    lower_constraint_bounds : list
        Lower bounds on the constraints.

    upper_constraint_bounds : list
        Upper bounds on the constraints.

    algorithm : str
        Label for the solution algorithm.

    max_iter : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    callback : callable
        Callback function from the class \
        :class:`~pyanno4rt.optimization.solvers._proxmin_solver.ProxminSolver`.

    Returns
    -------
    fun : callable
        Minimization function from the Proxmin library.

    arguments : dict
        Dictionary with the function arguments.
    """

    def get_objective_gradient(X):
        """Get the objective gradient for the current solution."""

        return problem_instance.gradient(X)[:, None]

    def estimate_lipschitz(X, it=0):
        """Estimate the Lipschitz constant of the gradient function."""

        # Reshape the array
        X = X.reshape(-1)

        # Determine the step size from a line search
        L = line_search(
            partial(problem_instance.objective, track=False),
            problem_instance.gradient, X, -problem_instance.gradient(X))

        # Check if no step size value has been found
        if not L[0]:

            # Return the default value
            return 1e-20

        return L[0]/2

    def project_on_bounds(X, step):
        """Project the current solution on the bounded set."""

        return clip(
            X, a_min=lower_variable_bounds, a_max=upper_variable_bounds)

    def perform_proximal_grad_step(X, step):
        """Perform a proximal gradient step."""

        return X - step*get_objective_gradient(X)

    # Convert the lower and upper variable bounds into 2D arrays
    lower_variable_bounds = array(lower_variable_bounds)[:, None]
    upper_variable_bounds = array(upper_variable_bounds)[:, None]

    # Initialize the arguments dictionary
    arguments = {'e_rel': 1e-2,
                 'max_iter': max_iter,
                 'callback': partial(
                     callback, objective=problem_instance.objective)}

    # Check if the algorithm is 'admm'
    if algorithm == 'admm':

        # Set the optimization function
        fun = admm

        # Update by the arguments of the 'admm' algorithm
        arguments |= {'prox_f': perform_proximal_grad_step,
                      'step_f': estimate_lipschitz,
                      'prox_g': project_on_bounds,
                      'step_g': None,
                      'L': None,
                      'e_abs': tolerance}

    # Else, check if the algorithm is 'pgm'
    elif algorithm == 'pgm':

        # Set the optimization function
        fun = pgm

        # Update by the arguments of the 'pgm' algorithm
        arguments |= {'grad': get_objective_gradient,
                      'step': estimate_lipschitz,
                      'prox': project_on_bounds,
                      'accelerated': True,
                      'backtracking': False,
                      'f': None}

    # Else, check if the algorithm is 'sdmm'
    elif algorithm == 'sdmm':

        # Set the optimization function
        fun = sdmm

        # Update by the arguments of the 'sdmm' algorithm
        arguments |= {'prox_f': perform_proximal_grad_step,
                      'step_f': estimate_lipschitz,
                      'proxs_g': [project_on_bounds],
                      'steps_g': None,
                      'Ls': None,
                      'e_abs': tolerance}

    return fun, arguments

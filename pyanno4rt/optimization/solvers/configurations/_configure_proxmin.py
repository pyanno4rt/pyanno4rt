"""Proxmin algorithm configuration."""

# Author: Tim Ortkamp
# Reference: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, clip
from proxmin import admm, pgm, sdmm
from scipy.optimize import line_search

# %% Function definition


def configure_proxmin(
        problem_instance, lower_variable_bounds, upper_variable_bounds,
        lower_constraint_bounds, upper_constraint_bounds, algorithm,
        maximum_iterations, tolerance, callback):
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

    maximum_iterations : int
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
        Dictionary with the solver arguments.
    """

    def get_objective_gradient(X):
        """Get the objective gradient."""

        return problem_instance.gradient(X)

    def estimate_lipschitz(X, it=0):
        """Estimate the Lipschitz constant of the gradient function."""

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

    # Convert the lower and upper variable bounds into arrays
    lower_variable_bounds = array(lower_variable_bounds)
    upper_variable_bounds = array(upper_variable_bounds)

    # Initialize the arguments dictionary
    arguments = {
        'e_rel': 1e-6,
        'max_iter': maximum_iterations,
        'callback': partial(callback, objective=problem_instance.objective)}

    # Check if the algorithm is 'admm'
    if algorithm == 'admm':

        # Set the optimization function
        fun = admm

        # Update the arguments dictionary
        arguments |= {
            'prox_f': perform_proximal_grad_step,
            'step_f': estimate_lipschitz,
            'prox_g': project_on_bounds,
            'step_g': None,
            'L': None,
            'e_abs': tolerance}

    # Else, check if the algorithm is 'pgm'
    elif algorithm == 'pgm':

        # Set the optimization function
        fun = pgm

        # Update the arguments dictionary
        arguments |= {
            'grad': get_objective_gradient,
            'step': estimate_lipschitz,
            'prox': project_on_bounds,
            'accelerated': True,
            'backtracking': False,
            'f': None}

    # Else, check if the algorithm is 'sdmm'
    elif algorithm == 'sdmm':

        # Set the optimization function
        fun = sdmm

        # Update the arguments dictionary
        arguments |= {
            'prox_f': perform_proximal_grad_step,
            'step_f': estimate_lipschitz,
            'proxs_g': [project_on_bounds],
            'steps_g': None,
            'Ls': None,
            'e_abs': tolerance}

    return fun, arguments

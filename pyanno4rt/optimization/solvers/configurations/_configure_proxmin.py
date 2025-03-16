"""Proxmin algorithm configuration."""

# Author: Tim Ortkamp, Viviana Deisner
# Reference: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, clip, dot, sqrt
from numpy.linalg import norm
from proxmin import admm, pgm, sdmm

# %% Function definition


def configure_proxmin(
        problem_instance, lower_variable_bounds, upper_variable_bounds,
        lower_constraint_bounds, upper_constraint_bounds, algorithm,
        initial_fluence, maximum_iterations, tolerance, callback):
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

    initial_fluence : ndarray
        Initial fluence vector.

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

    def get_upper_bound(X):
        """Get an upper bound on the step size interval."""

        # Get the objective function
        objective = partial(problem_instance.objective, track=False)

        # Get the gradient function
        gradient = get_objective_gradient(X)

        # Initialize the step size
        step = 1.0

        # Loop while the Armijo condition is fulfilled
        while (objective(X - step*gradient)
               > objective(X) + step*1e-4*dot(-gradient, gradient)):

            # Damp the step size
            step *= 0.3

        # # 
        # for i in range(maximum_iterations):

        #     # 
        #     if (objective(X - 2**(i-2)*step*gradient)
        #             < objective(X - 2**(i-1)*step*gradient)):

        #         # 
        #         step *= 2**(i-1)

        #         # 
        #         break

        return step

    def golden(X, upper, it=0):
        """Estimate the step size using the golden-section search method."""

        # Get the objective function
        objective = partial(problem_instance.objective, track=False)

        # Get the gradient function
        gradient = get_objective_gradient(X)

        # Initialize the lower bound
        lower = 0

        # 
        rho = (sqrt(5)-1)/2

        # 
        while upper - lower >= 1e-4:

            # Calculate the internal lower value
            int_lower = lower + (1-rho)*(upper-lower)

            # Calculate the internal upper value
            int_upper = lower + rho*(upper-lower)

            # Check if the lower objective is smaller than the upper
            if (objective(X - int_lower*gradient)
                    < objective(X - int_upper*gradient)):

                # Adjust the upper value
                upper = int_upper

                # # 
                # int_upper = int_lower

                # # 
                # int_upper = lower + (1-rho)*(upper-lower)

            else:

                # Adjust the lower value
                lower = int_lower

                # # 
                # int_lower = int_upper

                # # 
                # int_upper = lower + rho*(upper-lower)

        return (lower + upper) / 2

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
        'e_rel': 1e-3*clip(
            norm(initial_fluence)/norm(lower_variable_bounds), a_min=1e-3,
            a_max=10),
        'max_iter': maximum_iterations,
        'callback': partial(callback, objective=problem_instance.objective)}

    # Check if the algorithm is 'admm'
    if algorithm == 'admm':

        # Set the optimization function
        fun = admm

        # Update by the arguments dictionary
        arguments |= {
            'prox_f': perform_proximal_grad_step,
            'step_f': partial(golden, upper=get_upper_bound(initial_fluence)),
            'prox_g': project_on_bounds,
            'step_g': 0.3,
            'L': None,
            'e_abs': 3e-3}

    # Else, check if the algorithm is 'pgm'
    elif algorithm == 'pgm':

        # Set the optimization function
        fun = pgm

        # Update by the arguments dictionary
        arguments |= {
            'grad': get_objective_gradient,
            'step': partial(golden, upper=get_upper_bound(initial_fluence)),
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
            'step_f': partial(golden, upper=get_upper_bound(initial_fluence)),
            'proxs_g': [project_on_bounds],
            'steps_g': None,
            'Ls': None,
            'e_abs': tolerance}

    return fun, arguments

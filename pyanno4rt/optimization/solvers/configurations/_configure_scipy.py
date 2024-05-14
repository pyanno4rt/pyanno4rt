"""SciPy algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from functools import partial
from scipy.optimize import minimize, NonlinearConstraint, SR1

# %% Function definition


def configure_scipy(problem_instance, lower_variable_bounds,
                    upper_variable_bounds, lower_constraint_bounds,
                    upper_constraint_bounds, algorithm, max_iter, tolerance,
                    callback):
    """
    Configure the SciPy solver.

    Supported algorithms: L-BFGS-B, TNC, trust-constr.

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
        :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`.

    Returns
    -------
    fun : callable
        Minimization function from the SciPy library.

    arguments : dict
        Dictionary with the function arguments.
    """

    # Set the optimization function
    fun = minimize

    # Check if the algorithm is 'L-BFGS-B'
    if algorithm == 'L-BFGS-B':

        # Initialize the arguments of the 'L-BFGS-B' algorithm
        arguments = {'lexicographic': False,
                     'fun': problem_instance.objective,
                     'jac': problem_instance.gradient,
                     'method': 'L-BFGS-B',
                     'bounds': tuple(
                         zip(lower_variable_bounds, upper_variable_bounds)),
                     'tol': tolerance,
                     'options': {'disp': False,
                                 'ftol': tolerance,
                                 'maxiter': max_iter,
                                 'maxls': 20},
                     'callback': callback}

    # Else, check if the algorithm is 'TNC'
    elif algorithm == 'TNC':

        # Initialize the arguments of the 'TNC' algorithm
        arguments = {'lexicographic': False,
                     'fun': problem_instance.objective,
                     'jac': problem_instance.gradient,
                     'method': 'TNC',
                     'bounds': tuple(
                         zip(lower_variable_bounds, upper_variable_bounds)),
                     'tol': tolerance,
                     'options': {'disp': True,
                                 'maxCGit': 0,
                                 'eta': -1,
                                 'stepmx': 0,
                                 'ftol': tolerance,
                                 'maxfun': max_iter}
                     }

    # Else, check if the algorithm is 'trust-constr'
    elif algorithm == 'trust-constr':

        # Check if the method is 'lexicographic'
        if type(problem_instance).__name__ == 'LexicographicOptimization':

            # Initialize the arguments by the multi-rank items
            arguments = {
                rank: {
                    'fun': partial(problem_instance.objective, layer=rank),
                    'jac': partial(problem_instance.gradient, layer=rank),
                    'method': 'trust-constr',
                    'bounds': tuple(
                        zip(lower_variable_bounds, upper_variable_bounds)),
                    'tol': tolerance,
                    'options': {'disp': False,
                                'verbose': 0,
                                'initial_tr_radius': 1,
                                'sparse_jacobian': True,
                                'factorization_method': 'AugmentedSystem',
                                'maxiter': max_iter},
                    'callback': callback}
                for rank in problem_instance.objectives}

            # Loop over the argument ranks
            for rank in arguments:

                # Check if any constraints have been passed at the rank
                if (lower_constraint_bounds[rank],
                        upper_constraint_bounds[rank]) != ([], []):

                    # Update the arguments by the constraint items
                    arguments[rank] |= {
                        'constraints': NonlinearConstraint(
                            partial(problem_instance.constraint, layer=rank),
                            lower_constraint_bounds[rank],
                            upper_constraint_bounds[rank],
                            jac=partial(problem_instance.jacobian, layer=rank),
                            hess=SR1()),
                        'cfun': partial(
                            problem_instance.constraint, layer=rank)}

            # Add the indicator for the 'lexicographic' method
            arguments |= {'lexicographic': True}

        else:

            # Initialize the arguments of the 'trust-constr' algorithm
            arguments = {
                'lexicographic': False,
                'fun': problem_instance.objective,
                'jac': problem_instance.gradient,
                'method': 'trust-constr',
                'bounds': tuple(
                    zip(lower_variable_bounds, upper_variable_bounds)),
                'tol': tolerance,
                'options': {'disp': False,
                            'verbose': 0,
                            'initial_tr_radius': 1,
                            'sparse_jacobian': True,
                            'factorization_method': 'AugmentedSystem',
                            'maxiter': max_iter},
                'callback': callback}

            # Check if any constraints have been passed
            if (lower_constraint_bounds, upper_constraint_bounds) != ([], []):

                # Update the arguments by the constraint items
                arguments |= {
                    'constraints': NonlinearConstraint(
                        problem_instance.constraint,
                        lower_constraint_bounds,
                        upper_constraint_bounds,
                        jac=problem_instance.jacobian,
                        hess=SR1()),
                    'cfun': problem_instance.constraint}

    return fun, arguments

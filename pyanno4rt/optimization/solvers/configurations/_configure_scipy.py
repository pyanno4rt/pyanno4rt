"""SciPy algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from scipy.optimize import BFGS, minimize, NonlinearConstraint

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

    # Initialize the arguments dictionary
    arguments = {'fun': problem_instance.objective,
                 'jac': problem_instance.gradient,
                 'bounds': tuple(
                     zip(lower_variable_bounds, upper_variable_bounds)),
                 'tol': tolerance,
                 'callback': None}

    # Check if the algorithm is 'L-BFGS-B'
    if algorithm == 'L-BFGS-B':

        # Update by the arguments of the 'L-BFGS-B' algorithm
        arguments |= {'method': 'L-BFGS-B',
                      'options': {'disp': False,
                                  'ftol': tolerance,
                                  'maxiter': max_iter,
                                  'maxls': 20},
                      'callback': callback}

    # Else, check if the algorithm is 'TNC'
    elif algorithm == 'TNC':

        # Update by the arguments of the 'TNC' algorithm
        arguments |= {'method': 'TNC',
                      'options': {'disp': True,
                                  'maxCGit': 0,
                                  'eta': -1,
                                  'stepmx': 0,
                                  'ftol': tolerance,
                                  'maxfun': max_iter}
                      }

    # Else, check if the algorithm is 'trust-constr'
    elif algorithm == 'trust-constr':

        # Update by the arguments of the 'trust-constr' algorithm
        arguments |= {'method': 'trust-constr',
                      'options': {'disp': False,
                                  'verbose': 0,
                                  'initial_tr_radius': 1,
                                  'maxiter': max_iter},
                      'callback': callback}

        # Check if any constraints have been passed
        if len(problem_instance.constraints) > 0:

            # Update the arguments by the constraints object
            arguments |= {'constraints': NonlinearConstraint(
                        problem_instance.constraint, lower_constraint_bounds,
                        upper_constraint_bounds, jac=problem_instance.jacobian,
                        hess=BFGS()),
                    'cfun': problem_instance.constraint}

    return fun, arguments

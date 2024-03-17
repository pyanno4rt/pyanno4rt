"""SciPy algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from scipy.optimize import minimize

# %% Function definition


def configure_pymoo(problem_instance, lower_variable_bounds,
                    upper_variable_bounds, lower_constraint_bounds,
                    upper_constraint_bounds, algorithm, max_iter):
    """
    Configure the SciPy solver.

    Supported algorithms: L-BFGS-B, TNC, trust-constr.

    Parameters
    ----------
    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.components.methods.\
            _lexicographic_optimization.LexicographicOptimization`,\
        :class:`~pyanno4rt.optimization.components.methods.\
            _pareto_optimization.ParetoOptimization` or \
        :class:`~pyanno4rt.optimization.components.methods.\
            _weighted_sum_optimization.WeightedSumOptimization`
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

    Returns
    -------
    fun : function from :mod:`~scipy.optimize`
        Minimization function from the SciPy library.

    arguments : dict
        Dictionary with the function arguments.
    """

    # Set the optimization function
    fun = minimize

    # Set the common function arguments
    arguments = {'fun': problem_instance.objective,
                 'jac': problem_instance.gradient,
                 'bounds': tuple(
                     zip(lower_variable_bounds, upper_variable_bounds)),
                 'tol': 1e-3,
                 'callback': None}

    # Check if the algorithm is 'L-BFGS-B'
    if algorithm == 'L-BFGS-B':

        # Update by the arguments of the 'L-BFGS-B' algorithm
        arguments.update({'method': 'L-BFGS-B',
                          'options': {'disp': False,
                                      'ftol': 1e-3,
                                      'maxiter': max_iter,
                                      'maxls': 20}
                          })

    # Else, check if the algorithm is 'TNC'
    elif algorithm == 'TNC':

        # Update by the arguments of the 'TNC' algorithm
        arguments.update({'method': 'TNC',
                          'options': {'disp': True,
                                      'maxCGit': 0,
                                      'eta': -1,
                                      'stepmx': 0,
                                      'ftol': 1e-3,
                                      'maxfun': max_iter}
                          })

    # Else, check if the algorithm is 'trust-constr'
    elif algorithm == 'trust-constr':

        # Update by the arguments of the 'trust-constr' algorithm
        arguments.update({'method': 'trust-constr',
                          'options': {'disp': False,
                                      'verbose': 0,
                                      'maxiter': max_iter}
                          })

    return fun, arguments

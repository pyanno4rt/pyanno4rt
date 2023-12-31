"""SciPy algorithms."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypi.org/project/pypop7/#description

# %% External package import

from scipy.optimize import minimize

# %% Function definition


def get_scipy_configuration(problem_instance, lower_variable_bounds,
                            upper_variable_bounds, algorithm, max_iter):
    """
    Get the optimizer function and arguments for the SciPy solver.

    Supported algorithms: L-BFGS-B, TNC, trust-constr
    """
    # Check if the algorithm is 'L-BFGS-B'
    if algorithm == 'L-BFGS-B':

        # Set the solution function
        fun = minimize

        # Set the function arguments
        arguments = {'fun': problem_instance.objective,
                     'method': 'L-BFGS-B',
                     'jac': problem_instance.gradient,
                     'bounds': tuple(zip(lower_variable_bounds,
                                         upper_variable_bounds)),
                     'tol': 1e-3,
                     'options': {'disp': False,
                                 'ftol': 1e-3,
                                 'maxiter': max_iter,
                                 'maxls': 20}}

    # Else, check if the algorithm is 'TNC'
    elif algorithm == 'TNC':

        # Set the solution function
        fun = minimize

        # Set the function arguments
        arguments = {'fun': problem_instance.objective,
                     'method': 'TNC',
                     'jac': problem_instance.gradient,
                     'bounds': tuple(zip(lower_variable_bounds,
                                         upper_variable_bounds)),
                     'tol': 1e-3,
                     'options': {'disp': True,
                                 'maxCGit': 0,
                                 'eta': -1,
                                 'stepmx': 0,
                                 'ftol': 1e-3,
                                 'maxfun': max_iter}}

    # Check if the algorithm is 'trust-constr'
    if algorithm == 'trust-constr':

        # Set the solution function
        fun = minimize

        # Set the function arguments
        arguments = {'fun': problem_instance.objective,
                     'method': 'trust-constr',
                     'jac': problem_instance.gradient,
                     'bounds': tuple(zip(lower_variable_bounds,
                                         upper_variable_bounds)),
                     'tol': 1e-3,
                     'options': {'disp': False,
                                 'verbose': 2,
                                 'maxiter': max_iter}}

    return fun, arguments

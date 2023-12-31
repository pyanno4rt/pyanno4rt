"""Proxmin algorithms."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, clip
from numpy.linalg import norm
from proxmin import admm, pgm, sdmm
from scipy.optimize import line_search

# %% Function definition


def get_proxmin_configuration(number_of_variables, number_of_constraints,
                              problem_instance, lower_variable_bounds,
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

    def customize_output(X, it=0):
        """Customize the console output of the solver."""
        # Check if the current iteration is the first
        if it == 0:

            # Print the initial statements
            print("\n***************************************************")
            print("Running the {} algorithm from the Proxmin package"
                  .format(algorithm.upper()))
            print("***************************************************\n")
            print("{0:<25}{1:>8}"
                  .format('Number of variables:', number_of_variables))
            print("{0:<25}{1:>8}"
                  .format('Number of constraints:', number_of_constraints))
            print("\n{0:^9}\t\t{1:^9}\t\t{2:^8}"
                  .format("iteration", "objective", "||grad||"))
            print("{0:^9}\t\t{1:^9}\t\t{2:^8}"
                  .format("---------", "---------", "--------"))

        # Print the objective value and gradient norm per iteration
        print("{0:^9}\t\t{1:^9}\t\t{2:^8}"
              .format(
                  it,
                  round(problem_instance.objective(X.reshape(-1)), 4),
                  round(norm(get_objective_gradient(X)), 4)))

        # Check if the algorithm has converged
        if it == -1:

            # Print the convergence message and the final statements
            print("\nConvergence of solution reached.")
            print("{0:<25}{1:>8}"
                  .format('Number of iterations', it+1))
            print("{0:<25}{1:>8}"
                  .format(
                      'Optimal value',
                      round(problem_instance.objective(X.reshape(-1)), 4)))

        # Else, check if the maximum number of iterations is reached
        elif it == max_iter-1:

            # Print the termination message and the final statements
            print("\nMaximum number of iterations reached.")
            print("{0:<25}{1:>8}"
                  .format('Number of iterations:', it+1))
            print("{0:<25}{1:>8}"
                  .format(
                      'Optimal value:',
                      round(problem_instance.objective(X.reshape(-1)), 4)))

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
                     'max_iter': max_iter,
                     'callback': customize_output}

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
                     'max_iter': max_iter,
                     'callback': customize_output}

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
                     'max_iter': max_iter,
                     'callback': customize_output}

    return fun, arguments

"""Ipyopt algorithm configuration."""

# Author: Tim Ortkamp
# Reference: https://gitlab.com/ipyopt-devs/ipyopt

# %% External package import

from ipyopt import Problem
from numpy import array, indices

# %% Function definition


def configure_ipyopt(number_of_variables, number_of_constraints,
                     problem_instance, lower_variable_bounds,
                     upper_variable_bounds, lower_constraint_bounds,
                     upper_constraint_bounds, algorithm, max_iter, tolerance,
                     callback):
    """
    Configure the Ipyopt solver.

    Supported algorithms: MUMPS

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

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
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`.

    Returns
    -------
    nlp : class :class:`~ipyopt.Problem`
        The class used to represent the nonlinear optimization problem.

    arguments : dict
        Dictionary with the problem arguments.
    """

    def objective(fluence):
        """Get the objective."""

        return problem_instance.objective(fluence)

    def gradient(fluence, out):
        """Get the gradient of the objective."""

        out[()] = problem_instance.gradient(fluence)
        return out

    def constraint(fluence, out):
        """Get the constraints."""

        out[()] = problem_instance.constraint(fluence)
        return out

    def jacobian(fluence, out):
        """Get the jacobian of the constraints."""

        out[()] = problem_instance.jacobian(fluence).flatten()
        return out

    # Set the optimization problem class
    nlp = Problem

    # Check if the method is 'lexicographic'
    if type(problem_instance).__name__ == 'LexicographicOptimization':

        # Initialize the arguments by the multi-rank items (not implemented)
        raise ValueError("Lexicographic optimization with ipyopt is not "
                         "yet implemented ...")

    else:

        # Initialize the arguments of the 'mumps' algorithm
        arguments = {
            'n': number_of_variables,
            'x_l': array(lower_variable_bounds),
            'x_u': array(upper_variable_bounds),
            'm': number_of_constraints,
            'g_l': array(lower_constraint_bounds),
            'g_u': array(upper_constraint_bounds),
            'sparsity_indices_jac_g': tuple(
                arr for arr in indices(
                    (number_of_constraints, number_of_variables)
                    ).reshape(2, -1)),
            'sparsity_indices_h': None,
            'eval_f': objective,
            'eval_grad_f': gradient,
            'eval_g': constraint,
            'eval_jac_g': jacobian,
            'eval_h': None,
            'intermediate_callback': callback,
            'ipopt_options': {
                'sb': 'yes',
                'print_level': 0,
                'tol': 1e-10,
                'dual_inf_tol': 1e-4,
                'constr_viol_tol': 1e-4,
                'compl_inf_tol': 1e-4,
                'acceptable_iter': 3,
                'acceptable_tol': 1e10,
                'acceptable_constr_viol_tol': 1e-2,
                'acceptable_dual_inf_tol': 1e10,
                'acceptable_compl_inf_tol': 1e10,
                'acceptable_obj_change_tol': tolerance,
                'max_iter': max_iter,
                'mu_strategy': 'adaptive',
                'hessian_approximation': 'limited-memory',
                'limited_memory_max_history': 6,
                'limited_memory_initialization': 'scalar2',
                'linear_solver': algorithm
                }}

    return nlp, arguments

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
                     upper_constraint_bounds, algorithm, max_iter, tolerance):
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

    Returns
    -------
    nlp : ...
        ...
    """

    def objective(x):
        """."""

        return problem_instance.objective(x)

    def gradient(x, out):
        """."""

        out[()] = problem_instance.gradient(x)

        return out

    def constraint(x, out):
        """."""

        out[()] = problem_instance.constraint(x)
        return out

    def jacobian(x, out):
        """."""

        out[()] = problem_instance.jacobian(x)
        return out

    # Set the optimization function
    nlp = Problem(
        n=number_of_variables,
        x_l=array(lower_variable_bounds),
        x_u=array(upper_variable_bounds),
        m=number_of_constraints,
        g_l=array(lower_constraint_bounds),
        g_u=array(upper_constraint_bounds),
        sparsity_indices_jac_g=tuple(
            arr for arr in indices(
                (number_of_constraints, number_of_variables)).reshape(2, -1)),
        sparsity_indices_h=tuple(
            arr for arr in indices(
                (number_of_variables, number_of_variables)).reshape(2, -1)),
        eval_f=objective,
        eval_grad_f=gradient,
        eval_g=constraint,
        eval_jac_g=jacobian,
        # intermediate_callback=callback,
        ipopt_options={
           'tol': 1e-10,
           'dual_inf_tol': 1e-4,
           'constr_viol_tol': 1e-4,
           'compl_inf_tol': 1e-4,
           'acceptable_iter': 5,
           'acceptable_tol': 1e10,
           'acceptable_constr_viol_tol': 1e-2,
           'acceptable_dual_inf_tol': 1e10,
           'acceptable_compl_inf_tol': 1e10,
           'acceptable_obj_change_tol': tolerance,
           'max_iter': max_iter,
           'mu_strategy': 'adaptive',
           'hessian_approximation': 'limited-memory',
           'limited_memory_max_history': 50,
           'limited_memory_initialization': 'scalar2',
           'linear_solver': 'mumps'
           })

    return nlp

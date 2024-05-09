"""PyPop7 algorithm configuration."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pypop.readthedocs.io/en/latest/
# Paper: https://doi.org/10.48550/arXiv.2212.05652

# %% External package import

from numpy import array, log
from pypop7.optimizers.es.lmcma import LMCMA
from pypop7.optimizers.es.lmmaes import LMMAES

# %% Function definition


def configure_pypop7(number_of_variables, problem_instance,
                     lower_variable_bounds, upper_variable_bounds,
                     lower_constraint_bounds, upper_constraint_bounds,
                     algorithm, max_iter, tolerance):
    """
    Configure the PyPop7 solver.

    Supported algorithms: LMCMA, LMMAES.

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

    max_iter : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Returns
    -------
    fun : object
        The object representing the optimization algorithm.

    arguments : dict
        Dictionary with the function arguments.
    """

    # Compute the number of individuals
    number_of_individuals = 4 + int(3*log(number_of_variables))

    # Check if the algorithm is 'LMCMA'
    if algorithm == 'LMCMA':

        # Set the optimization function
        fun = LMCMA

        # Initialize the arguments dictionary
        arguments = {
            'problem': {
                'fitness_function': problem_instance.objective,
                'ndim_problem': number_of_variables,
                'lower_boundary': array(lower_variable_bounds),
                'upper_boundary': array(upper_variable_bounds)},
            'options': {
                'max_function_evaluations': number_of_individuals*max_iter,
                'fitness_threshold': tolerance,
                'seed_rng': 0,
                'sigma': 0.3,
                'm': number_of_individuals,
                'base_m': 4,
                'period': int(max(1, number_of_variables)),
                'n_steps': number_of_variables,
                'c_c': 0.5/(number_of_variables**(1/2)),
                'c_1': 1.0/(10.0*log(number_of_variables)+1.0),
                'c_s': 0.3,
                'd_s': 1.0,
                'z_star': 0.3,
                'n_individuals': number_of_individuals,
                'n_parents': int(number_of_individuals/2),
                'verbose': 1}
            }

    # Else, check if the algorithm is 'LMMAES'
    elif algorithm == 'LMMAES':

        # Set the optimization function
        fun = LMMAES

        # Initialize the arguments dictionary
        arguments = {
            'problem': {
                'fitness_function': problem_instance.objective,
                'ndim_problem': number_of_variables,
                'lower_boundary': array(lower_variable_bounds),
                'upper_boundary': array(upper_variable_bounds)},
            'options': {
                'max_function_evaluations': number_of_individuals*max_iter,
                'fitness_threshold': tolerance,
                'seed_rng': 0,
                'sigma': 0.3,
                'is_restart': False,
                'n_evolution_paths': number_of_individuals,
                'n_individuals': number_of_individuals,
                'n_parents': int(number_of_individuals/2),
                'c_s': 2.0*number_of_individuals/number_of_variables,
                'verbose': 1}
            }

    return fun, arguments

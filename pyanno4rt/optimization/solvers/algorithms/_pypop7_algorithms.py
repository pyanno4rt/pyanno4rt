"""PyPOP7 algorithms."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypi.org/project/pypop7/#description

# %% External package import

from numpy import array, log
from pypop7.optimizers.es.mmes import MMES
from pypop7.optimizers.es.lmcma import LMCMA
from pypop7.optimizers.es.rmes import RMES
from pypop7.optimizers.rs.bes import BES
from pypop7.optimizers.rs.gs import GS

# %% Function definition


def get_pypop7_configuration(number_of_variables, problem_instance,
                             lower_variable_bounds, upper_variable_bounds,
                             algorithm, max_iter, max_cpu_time):
    """
    Get the optimizer function and arguments for the PyPOP7 solver.

    Supported algorithms: MMES, LMCMA, RMES, BES, GS
    """
    # Check if the algorithm is 'MMES'
    if algorithm == 'MMES':

        # Set the solution function
        fun = MMES

        # Set the function arguments
        arguments = {
            'problem': {'fitness_function': problem_instance.objective,
                        'ndim_problem': number_of_variables,
                        'lower_boundary': array(
                            lower_variable_bounds),
                        'upper_boundary': array(
                            upper_variable_bounds)},
            'options': {'max_runtime': max_cpu_time,
                        'seed_rng': 0,
                        'sigma': 0.3,
                        'n_individuals': 4 + int(3*log(number_of_variables)),
                        'verbose': 1}}

        # Determine the maximum number of function evaluations from max_iter
        arguments['options']['max_function_evaluations'] = arguments[
            'options']['n_individuals']*max_iter

    # Check if the algorithm is 'LMCMA'
    if algorithm == 'LMCMA':

        # Set the solution function
        fun = LMCMA

        # Set the function arguments
        arguments = {
            'problem': {'fitness_function': problem_instance.objective,
                        'ndim_problem': number_of_variables,
                        'lower_boundary': array(
                            lower_variable_bounds),
                        'upper_boundary': array(
                            upper_variable_bounds)},
            'options': {'max_runtime': max_cpu_time,
                        'seed_rng': 0,
                        'sigma': 0.3,
                        'n_individuals': 4 + int(3*log(number_of_variables)),
                        'verbose': 1}}

        # Determine the maximum number of function evaluations from max_iter
        arguments['options']['max_function_evaluations'] = arguments[
            'options']['n_individuals']*max_iter

    # Check if the algorithm is 'RMES'
    if algorithm == 'RMES':

        # Set the solution function
        fun = RMES

        # Set the function arguments
        arguments = {
            'problem': {'fitness_function': problem_instance.objective,
                        'ndim_problem': number_of_variables,
                        'lower_boundary': array(
                            lower_variable_bounds),
                        'upper_boundary': array(
                            upper_variable_bounds)},
            'options': {'max_runtime': max_cpu_time,
                        'seed_rng': 0,
                        'sigma': 0.3,
                        'n_individuals': 4 + int(3*log(number_of_variables)),
                        'verbose': 1}}

        # Determine the maximum number of function evaluations from max_iter
        arguments['options']['max_function_evaluations'] = arguments[
            'options']['n_individuals']*max_iter

    # Check if the algorithm is 'BES'
    if algorithm == 'BES':

        # Set the solution function
        fun = BES

        # Set the function arguments
        arguments = {
            'problem': {'fitness_function': problem_instance.objective,
                        'ndim_problem': number_of_variables,
                        'lower_boundary': array(
                            lower_variable_bounds),
                        'upper_boundary': array(
                            upper_variable_bounds)},
            'options': {'max_runtime': max_cpu_time,
                        'seed_rng': 0,
                        'n_individuals': 4 + int(3*log(number_of_variables)),
                        'verbose': 1}}

        # Determine the maximum number of function evaluations from max_iter
        arguments['options']['max_function_evaluations'] = arguments[
            'options']['n_individuals']*max_iter

    # Check if the algorithm is 'GS'
    if algorithm == 'GS':

        # Set the solution function
        fun = GS

        # Set the function arguments
        arguments = {
            'problem': {'fitness_function': problem_instance.objective,
                        'ndim_problem': number_of_variables,
                        'lower_boundary': array(
                            lower_variable_bounds),
                        'upper_boundary': array(
                            upper_variable_bounds)},
            'options': {'max_runtime': max_cpu_time,
                        'seed_rng': 0,
                        'n_individuals': 4 + int(3*log(number_of_variables)),
                        'verbose': 1}}

        # Determine the maximum number of function evaluations from max_iter
        arguments['options']['max_function_evaluations'] = arguments[
            'options']['n_individuals']*max_iter

    return fun, arguments

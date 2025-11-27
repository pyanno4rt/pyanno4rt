"""PyPop7 wrapper."""

# Author: Tim Ortkamp
# Reference: https://pypop.readthedocs.io/en/latest/
# Paper: https://doi.org/10.48550/arXiv.2212.05652

# %% External package import

from numpy import array, clip, log
from pypop7.optimizers.es.lmcma import LMCMA
from pypop7.optimizers.es.lmmaes import LMMAES

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class PyPop7Solver():
    """
    PyPop7 wrapper class.

    This class serves as a wrapper for the population-based optimization \
    algorithms from the PyPop7 solver. It takes the problem structure, \
    configures the selected algorithm, and defines the method to run the \
    solver.

    Parameters
    ----------
    algorithm : str
        Label for the solution algorithm.

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    algorithm : str
        See 'Parameters'.

    maximum_iterations : int
        See 'Parameters'.

    tolerance : float
        See 'Parameters'.

    instance : None or object
        The object used to represent the optimization algorithm.

    arguments : None or dict
        Dictionary with the solver arguments.
    """

    def __init__(
            self,
            algorithm,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info(
            "Initializing PyPop7 solver with %s algorithm ...", algorithm)

        # Get the input arguments
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the instance and the arguments
        self.instance, self.arguments = None, None

    def configure(
            self,
            problem):
        """
        Configure the PyPop7 solver.

        Supported algorithms: LMCMA, LMMAES.

        Parameters
        ----------
        problem : object of class \
            :class:`~pyanno4rt.optimization.problems._weighted_sum_problem.WeightedSumProblem`
            The object used to represent the optimization problem.
        """

        # Get the number of variables
        number_of_variables = len(problem.initial_fluence)

        # Compute the number of individuals
        number_of_individuals = 4 + int(3*log(number_of_variables))

        # Check if the algorithm is 'LMCMA'
        if self.algorithm == 'LMCMA':

            # Set the optimization function
            self.instance = LMCMA

            # Initialize the arguments dictionary
            self.arguments = {
                'problem': {
                    'fitness_function': problem.objective,
                    'ndim_problem': number_of_variables,
                    'lower_boundary': array(problem.variable_bounds[0]),
                    'upper_boundary': array(problem.variable_bounds[1])},
                'options': {
                    'max_function_evaluations': (
                        number_of_individuals*self.maximum_iterations),
                    'early_stopping_threshold': self.tolerance,
                    'early_stopping_evaluations': number_of_individuals*50,
                    'seed_rng': 42,
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
                    'verbose': 1}}

        else:

            # Set the optimization function
            self.instance = LMMAES

            # Initialize the arguments dictionary
            self.arguments = {
                'problem': {
                    'fitness_function': problem.objective,
                    'ndim_problem': number_of_variables,
                    'lower_boundary': array(problem.variable_bounds[0]),
                    'upper_boundary': array(problem.variable_bounds[1])},
                'options': {
                    'max_function_evaluations': (
                        number_of_individuals*self.maximum_iterations),
                    'early_stopping_threshold': self.tolerance,
                    'early_stopping_evaluations': (
                        0.05*self.maximum_iterations*number_of_individuals),
                    'seed_rng': 42,
                    'sigma': 0.3,
                    'is_restart': False,
                    'n_evolution_paths': number_of_individuals,
                    'n_individuals': number_of_individuals,
                    'n_parents': int(number_of_individuals/2),
                    'c_s': 2.0*number_of_individuals/number_of_variables,
                    'verbose': 1}}

    def run(
            self,
            initial_fluence):
        """
        Run the PyPop7 solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        str
            Description for the cause of termination.
        """

        # Enter the initial fluence into the arguments dictionary
        self.arguments['options']['mean'] = initial_fluence
        self.arguments['options']['x'] = initial_fluence

        # Initialize the instance
        self.instance = self.instance(**self.arguments)

        # Solve the optimization problem
        result = self.instance.optimize()

        # Clip the optimal fluence to account for negative values
        result['best_so_far_x'] = clip(
            result['best_so_far_x'], a_min=0, a_max=None)

        return result['best_so_far_x'], result['termination_signal']

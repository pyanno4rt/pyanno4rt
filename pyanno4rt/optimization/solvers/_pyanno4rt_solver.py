"""Pyanno4rt custom solver wrapper."""

# Author: Tim Ortkamp

# %% External package import

from numpy import around, array

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.optimization.solvers.cmaes import CMAES, LRCMAES

# %% Class definition


class Pyanno4rtSolver():
    """
    Pyanno4rt custom solver wrapper class.

    This class serves as a wrapper for the internal custom optimization \
    algorithms from pyanno4rt. It takes the problem structure, configures the \
    selected algorithm, and defines the method to run the solver.

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

    counter : None or int
        Iteration counter.
    """

    def __init__(
            self,
            algorithm,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info(
            "Initializing pyanno4rt solver with %s algorithm ...", algorithm)

        # Get the input attributes
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the algorithm instance, arguments, and iteration counter
        self.instance, self.arguments, self.counter = None, None, None

    def callback(
            self,
            intermediate_result):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        intermediate_result : dict
            Dictionary with the intermediate results of the current iteration.
        """

        # Log a message about the intermediate result
        get_logger().info(
            "At iterate %s: f=%s",
            self.counter, around(intermediate_result['optimal_value'], 4))

        # Increment the iteration counter
        self.counter += 1

    def configure(
            self,
            problem):
        """
        Configure the pyanno4rt solver.

        Supported algorithms: CMAES.

        Parameters
        ----------
        problem : object of class \
            :class:`~pyanno4rt.optimization.problems.weighted._weighted_sum_problem.WeightedSumProblem`
            The object used to represent the optimization problem.
        """

        # Check if the algorithm is 'CMAES'
        if self.algorithm == 'CMAES':

            # Set the optimization function
            self.instance = CMAES

            # Initialize the arguments dictionary
            self.arguments = {
                'number_of_variables': len(problem.initial_fluence),
                'objective': problem.objective,
                'gradient': problem.gradient,
                'lower_variable_bounds': array(problem.variable_bounds[0]),
                'upper_variable_bounds': array(problem.variable_bounds[1]),
                'number_of_individuals': None,
                'initial_sigma': 0.2*max(problem.initial_fluence),
                'maximum_iterations': self.maximum_iterations,
                'maximum_wall_time': 7200,
                'fitness_threshold': -float('inf'),
                'fitness_window_size': 20,
                'tolerance': self.tolerance,
                'sigma_threshold': 1e-3,
                'store_singular_values': True,
                'update_interval': 1,
                'rank': None,
                'callback': self.callback}

        # Check if the algorithm is 'LRCMAES'
        elif self.algorithm == 'LRCMAES':

            # Set the optimization function
            self.instance = LRCMAES

            # Initialize the arguments dictionary
            self.arguments = {
                'number_of_variables': len(problem.initial_fluence),
                'objective': problem.objective,
                'gradient': problem.gradient,
                'lower_variable_bounds': array(problem.variable_bounds[0]),
                'upper_variable_bounds': array(problem.variable_bounds[1]),
                'number_of_individuals': None,
                'initial_sigma': 0.2*max(problem.initial_fluence),
                'low_rank_integrator': 'symmetricaugBUG',
                'low_rank_dimension': len(problem.initial_fluence),
                'low_rank_tolerance_rel': 1e-2,
                'low_rank_tolerance_abs': 1e-8,
                'maximum_iterations': self.maximum_iterations,
                'maximum_wall_time': 7200,
                'fitness_threshold': -float('inf'),
                'fitness_window_size': 20,
                'tolerance': self.tolerance,
                'sigma_threshold': 1e-3,
                'update_interval': 1,
                'callback': self.callback}

    def run(
            self,
            initial_fluence):
        """
        Run the Pyanno4rt solver.

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

        # Reset the counter
        self.counter = 1

        # Get the initial objective value
        objective_value = self.arguments['objective'](initial_fluence, False)

        # Log a message about the initial function value
        get_logger().info("At iterate 0: f=%s", around(objective_value, 4))

        # Initialize the algorithm
        self.instance = self.instance(**self.arguments)

        # Solve the optimization problem
        result = self.instance.optimize(initial_fluence)

        return result['optimal_point'], result['solver_info']

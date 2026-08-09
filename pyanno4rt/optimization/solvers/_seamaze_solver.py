"""SeaMaze wrapper."""

# Author: Tim Ortkamp

# %% External package import

from numpy import around, array, median
from seamaze.optimizers import CMAES
from seamaze.optimizers import DLRCMAES
from seamaze.optimizers import LMMAES

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class SeaMazeSolver():
    """
    SeaMaze wrapper class.

    This class serves as a wrapper for the (low-rank) evolutionary algorithms \
    from SeaMaze. It takes the problem structure, configures the selected \
    algorithm, and defines the method to run the solver.

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
            "Initializing SeaMaze solver with %s algorithm ...", algorithm)

        # Get the input attributes
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the algorithm instance, arguments, and iteration counter
        self.instance, self.arguments, self.counter = None, None, None

        	# Initialize the rank memory
        self.ranks = []

    def callback(
            self,
            solver):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        intermediate_result : dict
            Dictionary with the intermediate results of the current iteration.
        """

        # Check if the DLR-CMA-ES algorithm has been selected
        if self.algorithm == 'DLRCMAES':

            # Append the current rank
            self.ranks.append(solver.rank)

            # Log a message about the intermediate result and the rank
            get_logger().info(
                "At iterate %s: f=%s (r=%s)",
                self.counter, around(solver._result['optimal_value'], 4),
                solver.rank)

        else:

            # Log a message about the intermediate result
            get_logger().info(
                "At iterate %s: f=%s",
                self.counter, around(solver._result['optimal_value'], 4))

        # Increment the iteration counter
        self.counter += 1

    def configure(
            self,
            problem):
        """
        Configure the SeaMaze solver.

        Supported algorithms: CMAES, DLRCMAES, LMMAES.

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
                # 'gradient': problem.gradient,
                'lower_variable_bounds': array(problem.variable_bounds[0]),
                'upper_variable_bounds': array(problem.variable_bounds[1]),
                'number_of_individuals': None,
                'initial_sigma': 0.3*median(problem.initial_fluence),
                'maximum_iterations': self.maximum_iterations,
                'maximum_wall_time': 43200,
                'fitness_threshold': None,
                'fitness_window_size': 20,
                'tolerance': self.tolerance,
                'sigma_threshold': 1e-6,
                'update_interval': None,
                'min_log_level': 'critical',
                'callback': self.callback,
                'random_state': 42}

        # Check if the algorithm is 'DLRCMAES'
        elif self.algorithm == 'DLRCMAES':

            # Set the optimization function
            self.instance = DLRCMAES

            # Initialize the arguments dictionary
            self.arguments = {
                'number_of_variables': len(problem.initial_fluence),
                'objective': problem.objective,
                # 'gradient': problem.gradient,
                'lower_variable_bounds': array(problem.variable_bounds[0]),
                'upper_variable_bounds': array(problem.variable_bounds[1]),
                'number_of_individuals': None,
                'initial_sigma': 0.3*median(problem.initial_fluence),
                'low_rank_init_dimension': None,
                'low_rank_max_dimension': 50,
                'low_rank_is_adaptive': True,
                'low_rank_energy_tolerance': 1e-4,
                'maximum_iterations': self.maximum_iterations,
                'maximum_wall_time': 43200,
                'fitness_threshold': None,
                'fitness_window_size': 20,
                'tolerance': self.tolerance,
                'sigma_threshold': 1e-6,
                'update_interval': None,
                'min_log_level': 'critical',
                'callback': self.callback,
                'random_state': 42}

      	# Check if the algorithm is 'LMMAES'
        elif self.algorithm == 'LMMAES':

            # Set the optimization function
            self.instance = LMMAES

            # Initialize the arguments dictionary
            self.arguments = {
                'number_of_variables': len(problem.initial_fluence),
                'objective': problem.objective,
                # 'gradient': problem.gradient,
                # 'lower_variable_bounds': array(problem.variable_bounds[0]),
                # 'upper_variable_bounds': array(problem.variable_bounds[1]),
                'number_of_individuals': None,
                'initial_sigma': 0.3*median(problem.initial_fluence),
                'memory_size': None,
                'maximum_iterations': self.maximum_iterations,
                'maximum_wall_time': 43200,
                'fitness_threshold': None,
                'fitness_window_size': 20,
                'tolerance': self.tolerance,
                'sigma_threshold': 1e-6,
                'min_log_level': 'critical',
                'callback': self.callback,
                'random_state': 42}

    def run(
            self,
            initial_fluence):
        """
        Run the SeaMaze solver.

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

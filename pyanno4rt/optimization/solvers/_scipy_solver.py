"""SciPy wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.algorithms import configure_scipy

# %% Class definition


class SciPySolver():
    """
    SciPy wrapper class.

    This class serves as a wrapper for the local optimization algorithms from \
    the SciPy solver. It takes the problem structure, configures the selected \
    algorithm, and defines the method to run the solver.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.components.methods._lexicographic_optimization.LexicographicOptimization`\
        :class:`~pyanno4rt.optimization.components.methods._weighted_sum_optimization.WeightedSumOptimization`
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

    Attributes
    ----------
    fun : callable
        Minimization function from the SciPy library.

    arguments : dict
        Dictionary with the function arguments.

    counter : int
        Counter for the iterations.
    """

    def __init__(
            self,
            number_of_variables,
            number_of_constraints,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            algorithm,
            initial_fluence,
            max_iter,
            tolerance):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing SciPy solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = configure_scipy(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            max_iter, tolerance, self.callback)

        # Initialize the iteration counter
        self.counter = 1

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

        # Log a message about the intermediate objective function value
        Datahub().logger.display_info(
            f"At iterate {self.counter}: "
            f"f={round(intermediate_result['fun'], 4)}")

        # Increment the iteration counter
        self.counter += 1

    def run(
            self,
            initial_fluence):
        """
        Run the SciPy solver.

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

        # Check if the algorithm is different from 'TNC'
        if self.arguments['method'] != 'TNC':

            # Log a message about the initial objective function value
            Datahub().logger.display_info(
                "At iterate 0: "
                f"f={round(self.arguments['fun'](initial_fluence), 4)}")

        # Solve the optimization problem
        result = self.fun(x0=initial_fluence, **self.arguments)

        return result.x, result.message

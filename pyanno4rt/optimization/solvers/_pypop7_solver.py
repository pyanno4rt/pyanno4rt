"""PyPop7 wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pypop.readthedocs.io/en/latest/
# Paper: https://doi.org/10.48550/arXiv.2212.05652

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.configurations import configure_pypop7

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

    initial_fluence : ndarray
        Initial fluence vector.

    max_iter : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    fun : object
        The object representing the optimization algorithm.

    arguments : dict
        Dictionary with the function arguments.
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
            f"Initializing PyPop7 solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = configure_pypop7(
            number_of_variables, problem_instance, lower_variable_bounds,
            upper_variable_bounds, lower_constraint_bounds,
            upper_constraint_bounds, algorithm, max_iter, tolerance)

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

        # Solve the optimization problem
        result = self.fun(**self.arguments).optimize()

        return result['best_so_far_x'], result['termination_signal']

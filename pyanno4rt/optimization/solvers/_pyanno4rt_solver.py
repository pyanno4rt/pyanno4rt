"""Internal wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.configurations import configure_pyanno4rt

# %% Class definition


class Pyanno4rtSolver():
    """
    Internal wrapper class.

    This class serves as a wrapper for the internal optimization algorithms. \
    It takes the problem structure, configures the selected algorithm, and \
    defines the method to run the solver.

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
    fun : callable
        Minimization function from the pyanno4rt library.

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
            f"Initializing pyanno4rt solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = configure_pyanno4rt(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            max_iter, tolerance)

    def run(
            self,
            initial_fluence):
        """
        Run the pyanno4rt solver.

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

        # Solve the optimization problem
        result = self.fun(x0=initial_fluence, **self.arguments)

        return result.x, result.message

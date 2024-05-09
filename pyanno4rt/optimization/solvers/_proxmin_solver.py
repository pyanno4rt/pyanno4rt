"""Proxmin wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pypi.org/project/proxmin/

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.configurations import configure_proxmin

# %% Class definition


class ProxminSolver():
    """
    Proxmin wrapper class.

    This class serves as a wrapper for the proximal optimization algorithms \
    from the Proxmin solver. It takes the problem structure, configures the \
    selected algorithm, and defines the method to run the solver.

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
        Minimization function from the Proxmin library.

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
            f"Initializing Proxmin solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = configure_proxmin(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            max_iter, tolerance, self.callback)

    def callback(
            self,
            X,
            it,
            objective):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        X : ndarray
            Optimal point of the current iteration.

        it : int
            Iteration counter.

        fun : callable
            Objective value function.
        """

        # Log a message about the intermediate objective function value
        Datahub().logger.display_info(
            f"At iterate {it}: f={round(objective(X.reshape(-1)), 4)}")

    def run(
            self,
            initial_fluence):
        """
        Run the Proxmin solver.

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

        # Make a deep copy of the initial fluence vector
        decision_vector = initial_fluence.copy()[:, None]

        # Solve the optimization problem
        result = self.fun(X=decision_vector, **self.arguments)

        # Check if the algorithm has converged
        if result[0]:

            # Assign the convergence message
            message = "Convergence of solution reached."

        else:

            # Assign the maximum number of iterations message
            message = "Maximum number of iterations reached."

        return decision_vector.reshape(-1), message

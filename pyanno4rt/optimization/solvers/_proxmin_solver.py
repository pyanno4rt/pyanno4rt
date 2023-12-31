"""Proxmin wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypi.org/project/proxmin/

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.algorithms import get_proxmin_configuration

# %% Class definition


class ProxminSolver():
    """
    Proxmin wrapper class.

    This class provides methods for wrapping the Proxmin solver, including \
    the definition of support functions for calculations and printouts, the \
    initialization of the algorithms from the arguments set in the treatment \
    plan, the composition of a Proxmin-compatible optimization problem, and a \
    method to start the algorithm.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class `LexicographicOptimization`, \
    `ParetoOptimization`, or `WeightedSumOptimization`
        Instance of the optimization problem.

    lower_variable_bounds : list
        Lower bounds on the decision variables.

    upper_variable_bounds : list
        Upper bounds on the decision variables.

    lower_constraint_bounds : list
        Lower bounds on the constraints.

    upper_constraint_bounds : list
        Upper bounds on the constraints.

    algorithm : string
        Label for the solution algorithm.

    max_iter : int
        Maximum number of iterations taken for the solver to converge.

    max_cpu_time : float
        Maximum CPU time taken for the solver to converge.

    Attributes
    ----------
    fun : object of class `function`
        Function from the Proxmin library to be called with ``arguments``.

    arguments : dict
        Dictionary with the arguments used to configure ``fun``.
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
            max_iter,
            max_cpu_time):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing Proxmin solver with {} ..."
                                      .format(algorithm))

        # Get the optimizer function and the arguments
        self.fun, self.arguments = get_proxmin_configuration(
            number_of_variables, number_of_constraints, problem_instance,
            lower_variable_bounds, upper_variable_bounds, algorithm,
            max_iter)

    def start(
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

        string
            Description of the cause of termination.
        """
        # Create a deep copy of the initial fluence vector
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

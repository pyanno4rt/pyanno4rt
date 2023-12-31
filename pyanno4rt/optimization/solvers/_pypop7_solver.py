"""PyPOP7 wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://pypop.readthedocs.io/en/latest/

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.algorithms import get_pypop7_configuration

# %% Class definition


class Pypop7Solver():
    """
    PyPOP7 wrapper class.

    This class provides methods for wrapping the PyPOP7 solver, including the \
    initialization of the algorithms from the arguments set in the treatment \
    plan, the composition of a PyPOP7-compatible optimization problem, and a \
    method to start the algorithms.

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
        Function from the PyPOP7 library to be called with ``arguments``.

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
        Datahub().logger.display_info("Initializing PyPOP7 solver with {} ..."
                                      .format(algorithm))

        # Get the algorithm from the argument
        self.algorithm = algorithm

        # Get the optimizer function and the arguments
        self.fun, self.arguments = get_pypop7_configuration(
            number_of_variables, problem_instance, lower_variable_bounds,
            upper_variable_bounds, algorithm, max_iter, max_cpu_time)

    def start(
            self,
            initial_fluence):
        """
        Run the PyPOP7 solver.

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
        # Enter the initial fluence into the problem
        self.arguments['options']['mean'] = initial_fluence
        self.arguments['options']['x'] = initial_fluence

        # Print the initial statements
        print("\n***************************************************")
        print("Running the {} algorithm from the PyPOP7 package"
              .format(self.algorithm.upper()))
        print("***************************************************\n")

        # Solve the optimization problem
        optimizer = self.fun(**self.arguments)
        result = optimizer.optimize()

        return result['best_so_far_x'], result['termination_signal']

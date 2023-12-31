"""SciPy wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.algorithms import get_scipy_configuration

# %% Class definition


class SciPySolver():
    """
    SciPy wrapper class.

    This class provides methods for wrapping the SciPy solver, including the \
    initialization of the algorithms from the arguments set in the treatment \
    plan, the composition of a SciPy-compatible optimization problem, and a \
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
        Function from the SciPy library to be called with ``arguments``.

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
        Datahub().logger.display_info("Initializing SciPy solver with {} ..."
                                      .format(algorithm))

        # Get the optimizer function and the arguments
        self.fun, self.arguments = get_scipy_configuration(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            algorithm, max_iter)

        # Initialize the iteration number
        self.iter = 1

    def callback(
            self,
            intermediate_result):
        """."""

        # Log a message about the objective value in the current iteration
        Datahub().logger.display_info(''.join((
            f'At iterate {self.iter}: f=',
            str(round(intermediate_result['fun'], 4)))))

        # Increment the iteration counter
        self.iter += 1

    def start(
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

        string
            Description of the cause of termination.
        """

        # Log a message about the initial objective value
        Datahub().logger.display_info(''.join((
            'At iterate 0: f=',
            str(round(self.arguments['fun'](initial_fluence), 4)))))

        # Check if the algorithm is different from 'TNC'
        if self.arguments['method'] != 'TNC':

            # Assign the callback function
            callback = self.callback

        else:

            # Assign the default callback
            callback = None

        # Solve the optimization problem
        result = self.fun(x0=initial_fluence, **self.arguments,
                          callback=callback)

        return result.x, result.message

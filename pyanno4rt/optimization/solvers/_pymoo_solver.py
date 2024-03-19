"""Pymoo wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://pymoo.org/

# %% External package import

from numpy import mean
from pymoo.core.callback import Callback

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.algorithms import configure_pymoo
from pyanno4rt.tools import get_all_objectives, get_objective_segments

# %% Class definition


class PymooSolver():
    """
    Pymoo wrapper class.

    This class serves as a wrapper for the multi-objective (Pareto) \
    optimization algorithms from the Pymoo solver. It takes the problem \
    structure, configures the selected algorithm, and defines the method to \
    run the solver.

    Parameters
    ----------
    number_of_variables : int
        Number of decision variables.

    number_of_constraints : int
        Number of constraints.

    problem_instance : object of class \
        :class:`~pyanno4rt.optimization.components.methods._pareto_optimization.ParetoOptimization`\
        The object representing the (Pareto) optimization problem.

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

    max_cpu_time : float
        Maximum CPU time.

    Attributes
    ----------
    fun : callable
        Minimization function from the Pymoo library.

    algorithm_object : object of class from :mod:`pymoo.algorithms`
        The object representing the solution algorithm.

    problem : object of class from :mod:`pymoo.core.problem`
        The object representing the Pymoo-compatible structure of the \
        multi-objective (Pareto) optimization problem.

    termination : object of class from :mod:`pymoo.termination`
        The object representing the termination criterion.
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
            max_cpu_time):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info(
            f"Initializing Pymoo solver with {algorithm} algorithm ...")

        # Get the callable optimization function and the solver objects
        self.fun, self.algorithm_object, self.problem, self.termination = (
            configure_pymoo(
                number_of_variables, len(get_all_objectives(hub.segmentation)),
                problem_instance, lower_variable_bounds, upper_variable_bounds,
                lower_constraint_bounds, upper_constraint_bounds, algorithm,
                initial_fluence, max_iter))

    def run(
            self,
            initial_fluence):
        """
        Run the Pymoo solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        ndarray
            Optimized (Pareto) set of fluence vectors.

        str
            Description for the cause of termination.
        """

        # Solve the optimization problem
        result = self.fun(self.problem, self.algorithm_object,
                          self.termination, seed=1, save_history=False,
                          verbose=False, callback=CustomCallback())

        return result.X, result.message


class CustomCallback(Callback):
    """
    Custom callback object for the Pymoo solver.

    Attributes
    ----------
    objectives : tuple
        Tuple with the names and segments of the objective functions.
    """

    def __init__(self):

        # Call the superclass constructor
        super().__init__()

        # Get the objective names and segments
        self.objectives = tuple(
            f"mod. {objective.name}-{segment}"
            for objective, segment in zip(
                    get_all_objectives(Datahub().segmentation),
                    get_objective_segments(Datahub().segmentation)))

    def notify(
            self,
            algorithm):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        algorithm : object of class from :mod:`pymoo.algorithms`
            The object representing the solution algorithm.
        """

        # Get the objectives string from the mean objective values
        objectives_string = ', '.join(
            (f"{round(value, 4)} ({objective_name})"
             for value, objective_name in zip(
                     mean(algorithm.pop.get("F"), axis=0), self.objectives)))

        # Log a message about the intermediate mean objective function values
        Datahub().logger.display_info(
            f"At generation {algorithm.n_gen}: {objectives_string}")

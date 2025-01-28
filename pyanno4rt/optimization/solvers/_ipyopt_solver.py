"""Ipyopt wrapper."""

# Author: Tim Ortkamp
# Reference: https://gitlab.com/ipyopt-devs/ipyopt

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.configurations import configure_ipyopt

# %% Class definition


class IpyoptSolver():
    """
    Ipyopt wrapper class.

    This class serves as a wrapper for the interior-point optimization \
    algorithms from the Ipyopt solver. It takes the problem structure and \
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
    nlp : object of class :class:`~ipyopt.Problem`
        The object used to represent the nonlinear optimization problem.

    arguments : dict
        Dictionary with the problem arguments.
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
            f"Initializing Ipyopt solver with {algorithm} algorithm ...")

        # Get the callable nonlinear problem object
        self.nlp, self.arguments = configure_ipyopt(
            number_of_variables, number_of_constraints, problem_instance,
            lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            max_iter, tolerance, self.callback)

        # Initialize the layer indicator (for 'lexicographic' method)
        self.layer = None

    def callback(
            self,
            *args):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        *args : tuple
            Tuple with the callback parameters of the Ipyopt solver.

        Returns
        -------
        bool
            Indicator for the success of the callback iteration.
        """

        # Set the base output string
        output_string = f"At iterate {args[1]}: f={round(args[2], 4)}"

        # Log a message about the intermediate function value(s)
        Datahub().logger.display_info(output_string)

        return True

    def run(
            self,
            initial_fluence):
        """
        Run the Ipyopt solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        int
            Indicator for the cause of termination.
        """

        # Solve the optimization problem
        optimized_fluence, _, status = self.nlp(**self.arguments).solve(
            x0=initial_fluence)

        return optimized_fluence, status

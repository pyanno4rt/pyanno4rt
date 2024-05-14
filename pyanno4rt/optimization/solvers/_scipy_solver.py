"""SciPy wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from numpy import around

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.solvers.configurations import configure_scipy

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

        # Initialize the layer indicator (for 'lexicographic' method)
        self.layer = None

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

        # Set the base output string
        output_string = (f"At iterate {self.counter}: "
                         f"f={round(intermediate_result['fun'], 4)}")

        # Check if any constraints have been passed to the algorithm
        if 'constraints' in self.arguments.get(self.layer, self.arguments):

            # Add the constraint values to the output string
            output_string = (
                f"{output_string}, "
                f"g={around(intermediate_result['constr'][0], 4)}")

        # Log a message about the intermediate function value(s)
        Datahub().logger.display_info(output_string)

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

        # Initialize the datahub
        hub = Datahub()

        # Check if the optimization problem is lexicographic
        if self.arguments.pop('lexicographic'):

            # Get all ranks from the arguments dictionary
            ranks = tuple(self.arguments)

            # Get tracker, objectives and constraints from the datahub
            tracker = hub.optimization['problem'].tracker
            objectives = hub.optimization['problem'].objectives
            constraints = hub.optimization['problem'].constraints

            # Loop over the argument subdictionaries
            for rank, arguments in self.arguments.items():

                # Log a message about the lexicographic rank
                hub.logger.display_info(
                    f"Considering lexicography at rank {rank} ...")

                # Set the current layer for the callback
                self.layer = rank

                # Get the initial objective value
                objective_value = arguments['fun'](initial_fluence, False)

                # Set the base output string
                output_string = (
                    f"At iterate {self.counter-1}: "
                    f"f={round(objective_value, 4)}")

                # Check if the constraint function is included
                if 'cfun' in arguments:

                    # Get the initial constraint value
                    constraint_value = arguments.pop('cfun')(
                        initial_fluence, False)

                    # Add the initial constraint value to the output string
                    output_string = (
                        f"{output_string}, g={around(constraint_value, 4)}")

                # Log a message about the initial function value(s)
                hub.logger.display_info(output_string)

                # Solve the optimization problem of the current rank
                result = self.fun(x0=initial_fluence, **arguments)

                # Update the initial fluence for the next layer
                initial_fluence = result.x

                # Check if the current rank is not from the final layer
                if rank != ranks[-1]:

                    # Get the value of the next rank
                    next_rank = ranks[ranks.index(rank)+1]

                    # Get the previous ranks
                    prev_ranks = ranks[:ranks.index(next_rank)]

                    # Get the constraint labels with the index positions
                    constraint_index = {
                        label: tuple(constraints[next_rank]).index(label)
                        for label in (label for rank in prev_ranks
                                      for label in objectives[rank])}

                    # Loop over the constraint-index pairs
                    for label, index in constraint_index.items():

                        # Adapt the upper bound by the current best value
                        self.arguments[next_rank]['constraints'].ub[index] = (
                            tracker[label][-1])

        else:

            # Check if the algorithm is different from 'TNC'
            if self.arguments['method'] != 'TNC':

                # Get the initial objective value
                objective_value = self.arguments['fun'](initial_fluence, False)

                # Set the base output string
                output_string = (
                    f"At iterate 0: f={round(objective_value, 4)}")

                # Check if the constraint function is included
                if 'cfun' in self.arguments:

                    # Get the initial constraint value
                    constraint_value = self.arguments.pop('cfun')(
                        initial_fluence, False)

                    # Add the initial constraint value to the output string
                    output_string = (
                        f"{output_string}, g={around(constraint_value, 4)}")

                # Log a message about the initial function value(s)
                Datahub().logger.display_info(output_string)

            # Solve the optimization problem
            result = self.fun(x0=initial_fluence, **self.arguments)

        return result.x, result.message

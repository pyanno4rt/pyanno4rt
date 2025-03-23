"""SciPy wrapper."""

# Author: Tim Ortkamp
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from functools import partial
from numpy import around
from scipy.optimize import minimize, NonlinearConstraint, SR1

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import filter_dict

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

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    fun : callable
        Minimization function from the SciPy library.

    arguments : dict
        Dictionary with the solver arguments.

    rank : None or int
        Current rank of the lexicography.

    counter : None or int
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
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing SciPy solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = self.configure(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            maximum_iterations, tolerance)

        # Initialize the rank indicator
        self.rank = None

        # Initialize the iteration counter
        self.counter = None

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
        output_string = (
            f"At iterate {self.counter}: "
            f"f={round(intermediate_result['fun'], 4)}")

        # Check if any constraints have been passed
        if 'constraints' in self.arguments.get(self.rank, self.arguments):

            # Extend the output string
            output_string = (
                f"{output_string}, "
                f"g={around(intermediate_result['constr'][0], 4)}")

        # Log a message about the intermediate function value(s)
        Datahub().logger.display_info(output_string)

        # Increment the iteration counter
        self.counter += 1

    def configure(
            self,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            algorithm,
            maximum_iterations,
            tolerance):
        """
        Configure the SciPy solver.

        Supported algorithms: L-BFGS-B, TNC, trust-constr.

        Parameters
        ----------
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

        maximum_iterations : int
            Maximum number of iterations.

        tolerance : float
            Precision goal for the objective function value.

        Returns
        -------
        fun : callable
            Minimization function from the SciPy library.

        arguments : dict
            Dictionary with the solver arguments.
        """

        # Set the optimization function
        fun = minimize

        # Check if the algorithm is 'L-BFGS-B'
        if algorithm == 'L-BFGS-B':

            # Initialize the argument dictionary
            arguments = {
                'lexicographic': False,
                'fun': problem_instance.objective,
                'jac': problem_instance.gradient,
                'method': 'L-BFGS-B',
                'bounds': tuple(zip(
                    lower_variable_bounds, upper_variable_bounds)),
                'tol': tolerance,
                'options': {
                    'disp': False,
                    'ftol': tolerance,
                    'maxiter': maximum_iterations,
                    'maxls': 20},
                'callback': self.callback}

        # Else, check if the algorithm is 'TNC'
        elif algorithm == 'TNC':

            # Initialize the argument dictionary
            arguments = {
                'lexicographic': False,
                'fun': problem_instance.objective,
                'jac': problem_instance.gradient,
                'method': 'TNC',
                'bounds': tuple(zip(
                    lower_variable_bounds, upper_variable_bounds)),
                'tol': tolerance,
                'options': {
                    'disp': True,
                    'maxCGit': 0,
                    'eta': -1,
                    'stepmx': 0,
                    'ftol': tolerance,
                    'maxfun': maximum_iterations}}

        # Else, check if the algorithm is 'trust-constr'
        elif algorithm == 'trust-constr':

            # Check if the method is 'lexicographic'
            if type(problem_instance).__name__ == 'LexicographicOptimization':

                # Initialize the rank-wise argument dictionaries
                arguments = {
                    rank: {
                        'fun': partial(problem_instance.objective, rank=rank),
                        'jac': partial(problem_instance.gradient, rank=rank),
                        'method': 'trust-constr',
                        'bounds': tuple(
                            zip(lower_variable_bounds, upper_variable_bounds)),
                        'tol': tolerance,
                        'options': {
                            'disp': False,
                            'verbose': 0,
                            'initial_tr_radius': 1,
                            'sparse_jacobian': True,
                            'factorization_method': 'AugmentedSystem',
                            'maxiter': maximum_iterations},
                        'callback': self.callback}
                    for rank in problem_instance.objectives}

                # Loop over the ranks
                for rank in arguments:

                    # Check if any constraints have been passed at the rank
                    if (lower_constraint_bounds[rank],
                            upper_constraint_bounds[rank]) != ([], []):

                        # Update the argument dictionary
                        arguments[rank] |= {
                            'constraints': NonlinearConstraint(
                                partial(
                                    problem_instance.constraint, rank=rank),
                                lower_constraint_bounds[rank],
                                upper_constraint_bounds[rank],
                                jac=partial(
                                    problem_instance.jacobian, rank=rank),
                                hess=SR1()),
                            'constraint_function': partial(
                                problem_instance.constraint, rank=rank)}

                # Add the indicator for the 'lexicographic' method
                arguments |= {'lexicographic': True}

            else:

                # Initialize the arguments dictionary
                arguments = {
                    'lexicographic': False,
                    'fun': problem_instance.objective,
                    'jac': problem_instance.gradient,
                    'method': 'trust-constr',
                    'bounds': tuple(
                        zip(lower_variable_bounds, upper_variable_bounds)),
                    'tol': tolerance,
                    'options': {
                        'disp': False,
                        'verbose': 0,
                        'initial_tr_radius': 100,
                        'sparse_jacobian': None,
                        'factorization_method': None,
                        'maxiter': maximum_iterations},
                    'callback': self.callback}

                # Check if any constraints have been passed
                if ((lower_constraint_bounds, upper_constraint_bounds)
                        != ([], [])):

                    # Update the argument dictionary
                    arguments |= {
                        'constraints': NonlinearConstraint(
                            problem_instance.constraint,
                            lower_constraint_bounds,
                            upper_constraint_bounds,
                            jac=problem_instance.jacobian,
                            hess=SR1()),
                        'constraint_function': problem_instance.constraint}

        return fun, arguments

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

        # Reset the iteration counter
        self.counter = 1

        # Check if the optimization problem is lexicographic
        if self.arguments['lexicographic']:

            # Get all ranks from the arguments dictionary
            ranks = tuple(self.arguments)

            # Get tracker, objectives and constraints from the datahub
            tracker = hub.optimization['problem'].tracker
            objectives = hub.optimization['problem'].objectives
            constraints = hub.optimization['problem'].constraints

            # Loop over the rank arguments
            for rank, arguments in self.arguments.items():

                # Log a message about the lexicographic rank
                hub.logger.display_info(
                    f"Considering lexicography at rank {rank} ...")

                # Set the current rank
                self.rank = rank

                # Get the initial objective value
                objective_value = arguments['fun'](initial_fluence, False)

                # Set the base output string
                output_string = (
                    f"At iterate {self.counter-1}: "
                    f"f={round(objective_value, 4)}")

                # Check if the constraint function is included
                if 'constraint_function' in arguments:

                    # Get the initial constraint value
                    constraint_value = arguments.pop('constraint_function')(
                        initial_fluence, False)

                    # Extend the output string
                    output_string = (
                        f"{output_string}, g={around(constraint_value, 4)}")

                # Log a message about the initial function values
                hub.logger.display_info(output_string)

                # Solve the optimization problem at the current rank
                result = self.fun(x0=initial_fluence, **arguments)

                # Update the initial fluence for the next rank
                initial_fluence = result.x

                # Check if the current rank does not equal the final rank
                if rank != ranks[-1]:

                    # Get the value of the next rank
                    next_rank = ranks[ranks.index(rank)+1]

                    # Get the previous ranks
                    prev_ranks = ranks[:ranks.index(next_rank)]

                    # Get the constraint labels with the index positions
                    constraint_index = {
                        label: tuple(constraints[next_rank]).index(label)
                        for label in (
                                label for rank in prev_ranks
                                for label in objectives[rank])}

                    # Loop over the constraint-index pairs
                    for label, index in constraint_index.items():

                        # Adjust the upper bound by the current best value
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
                if 'constraint_function' in self.arguments:

                    # Get the initial constraint value
                    constraint_value = self.arguments.pop(
                        'constraint_function')(initial_fluence, False)

                    # Extend the output string
                    output_string = (
                        f"{output_string}, g={around(constraint_value, 4)}")

                # Log a message about the initial function values
                Datahub().logger.display_info(output_string)

            # Solve the optimization problem
            result = self.fun(
                x0=initial_fluence,
                **filter_dict(self.arguments, remove_keys=('lexicographic',)))

        return result.x, result.message

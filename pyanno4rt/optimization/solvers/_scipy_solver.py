"""SciPy wrapper."""

# Author: Tim Ortkamp
# Reference: https://docs.scipy.org/doc/scipy/reference/optimize.html

# %% External package import

from numpy import around
from scipy.optimize import minimize, NonlinearConstraint, SR1

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class SciPySolver():
    """
    SciPy wrapper class.

    This class serves as a wrapper for the local optimization algorithms from \
    the SciPy solver. It takes the problem structure, configures the selected \
    algorithm, and defines the method to run the solver.

    Parameters
    ----------
    algorithm : str
        Label for the solution algorithm.

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    algorithm : str
        See 'Parameters'.

    maximum_iterations : int
        See 'Parameters'.

    tolerance : float
        See 'Parameters'.

    fun : callable
        Minimization function from the SciPy library.

    arguments : dict
        Dictionary with the solver arguments.

    counter : None or int
        Iteration counter.
    """

    def __init__(
            self,
            algorithm,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info(
            "Initializing SciPy solver with %s algorithm ...", algorithm)

        # Get the input arguments
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the function, arguments, and iteration counter
        self.fun, self.arguments, self.counter = None, None, None

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
            f"f={around(intermediate_result['fun'], 4)}")

        # Check if any constraints have been passed
        if 'constraints' in self.arguments:

            # Extend the output string
            output_string = (
                f"{output_string}, "
                f"g={around(intermediate_result['constr'][0], 4)}")

        # Log a message about the intermediate function value(s)
        get_logger().info(output_string)

        # Increment the iteration counter
        self.counter += 1

    def configure(
            self,
            problem):
        """
        Configure the SciPy solver.

        Supported algorithms: L-BFGS-B, TNC, trust-constr.

        Parameters
        ----------
        problem : object of class \
            :class:`~pyanno4rt.optimization.problems._lexicographic_problem.LexicographicProblem`\
            :class:`~pyanno4rt.optimization.problems._weighted_sum_problem.WeightedSumProblem`
            The object used to represent the optimization problem.
        """

        # Set the optimization function
        self.fun = minimize

        # Check if the algorithm is 'L-BFGS-B'
        if self.algorithm == 'L-BFGS-B':

            # Initialize the argument dictionary
            self.arguments = {
                'fun': problem.objective,
                'jac': problem.gradient,
                'method': 'L-BFGS-B',
                'bounds': zip(*problem.variable_bounds),
                'tol': self.tolerance,
                'options': {
                    'disp': False,
                    'ftol': self.tolerance,
                    'maxiter': self.maximum_iterations,
                    'maxls': 20},
                'callback': self.callback}

        # Else, check if the algorithm is 'TNC'
        elif self.algorithm == 'TNC':

            # Initialize the argument dictionary
            self.arguments = {
                'fun': problem.objective,
                'jac': problem.gradient,
                'method': 'TNC',
                'bounds': zip(*problem.variable_bounds),
                'tol': self.tolerance,
                'options': {
                    'disp': True,
                    'maxCGit': 0,
                    'eta': -1,
                    'stepmx': 0,
                    'ftol': self.tolerance,
                    'maxfun': self.maximum_iterations}}

        # Else, check if the algorithm is 'trust-constr'
        elif self.algorithm == 'trust-constr':

            # Initialize the arguments dictionary
            self.arguments = {
                'fun': problem.objective,
                'jac': problem.gradient,
                'method': 'trust-constr',
                'bounds': zip(*problem.variable_bounds),
                'tol': self.tolerance,
                'options': {
                    'disp': False,
                    'verbose': 0,
                    'initial_tr_radius': 10,
                    'sparse_jacobian': None,
                    'factorization_method': None,
                    'maxiter': self.maximum_iterations},
                'callback': self.callback}

            # Check if any constraints have been passed
            if problem.constraint_bounds != ([], []):

                # Update the argument dictionary
                self.arguments |= {
                    'constraints': NonlinearConstraint(
                        problem.constraint,
                        problem.constraint_bounds[0],
                        problem.constraint_bounds[1],
                        jac=problem.jacobian,
                        hess=SR1()),
                    'cfun': problem.constraint}

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

        # Reset the counter
        self.counter = 1

        # Check if the algorithm is different from 'TNC'
        if self.arguments['method'] != 'TNC':

            # Get the initial objective value
            objective_value = self.arguments['fun'](initial_fluence, False)

            # Set the base output string
            output_string = (
                f"At iterate 0: f={around(objective_value, 4)}")

            # Check if the constraint function is included
            if 'cfun' in self.arguments:

                # Get the initial constraint value
                constraint_value = self.arguments.pop('cfun')(
                    initial_fluence, False)

                # Extend the output string
                output_string = (
                    f"{output_string}, g={around(constraint_value, 4)}")

            # Log a message about the initial function values
            get_logger().info(output_string)

        # Solve the optimization problem
        result = self.fun(x0=initial_fluence, **self.arguments)

        return result.x, result.message

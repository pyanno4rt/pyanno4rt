"""Proxmin wrapper."""

# Author: Tim Ortkamp, Viviana Deisner
# Reference: https://pypi.org/project/proxmin/

# %% External package import

from functools import partial
from numpy import array, array_equal, clip, sqrt, zeros
from proxmin import admm, pgm, sdmm

# %% Internal package import

from pyanno4rt.datahub import Datahub

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

    maximum_iterations : int
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
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            f"Initializing Proxmin solver with {algorithm} algorithm ...")

        # Get the callable optimization function and its arguments
        self.fun, self.arguments = self.configure(
            problem_instance, lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            initial_fluence, maximum_iterations, tolerance)

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

    def configure(
            self,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            algorithm,
            initial_fluence,
            maximum_iterations,
            tolerance):
        """
        Configure the Proxmin solver.

        Supported algorithms: ADMM, PGM, SDMM.

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

        initial_fluence : ndarray
            Initial fluence vector.

        maximum_iterations : int
            Maximum number of iterations.

        tolerance : float
            Precision goal for the objective function value.

        Returns
        -------
        fun : callable
            Minimization function from the Proxmin library.

        arguments : dict
            Dictionary with the solver arguments.
        """

        def golden(X, it=0):
            """Estimate the step size with the golden-section search method."""

            # Get the objective function
            objective = partial(problem_instance.objective, track=False)

            # Get the gradient function
            gradient = problem_instance.gradient(X)

            # Initialize the distance alpha for the interval search
            alpha = 1e-2

            # Initialize the factor for the interval search
            factor = 0

            # Initialize the point set for the interval search
            points = [0]

            # Loop while the objective function decreases along the search line
            while (objective(X - 2**factor*alpha*gradient)
                   >= objective(X - 2**(factor+1)*alpha*gradient)
                   and not array_equal(gradient, zeros(gradient.shape))):

                # Append the new point
                points.append(2**(factor+1)*alpha)

                # Increment the factor
                factor += 1

            # Check if the length of the point set is at least 3
            if len(points) >= 3:

                # Set the calculated interval bounds
                lower, upper = points[-3], points[-1]

            else:

                # Set the default interval bounds
                lower, upper = 0, alpha

            # Initialize the threshold for the interval reduction
            threshold = 1e-6

            # Set the golden section value
            rho = (sqrt(5)-1)/2

            # Calculate the internal lower value
            int_lower = lower + (1-rho)*(upper-lower)

            # Calculate the internal upper value
            int_upper = lower + rho*(upper-lower)

            # Loop while the reduced interval size is too large
            while upper - lower >= threshold:

                # Check if the objective value at the lower point is smaller
                if (objective(X - int_lower*gradient)
                        < objective(X - int_upper*gradient)):

                    # Adjust the upper value
                    upper = int_upper

                    # Adjust the internal upper value
                    int_upper = int_lower

                    # Recalculate the internal lower value
                    int_lower = lower + (1-rho)*(upper-lower)

                else:

                    # Adjust the lower value
                    lower = int_lower

                    # Adjust the internal lower value
                    int_lower = int_upper

                    # Recalculate the internal upper value
                    int_upper = lower + rho*(upper-lower)

            return (lower + upper) / 2

        def project_on_bounds(X, step):
            """Project the current solution on the bounded set."""

            return clip(
                X, a_min=lower_variable_bounds, a_max=upper_variable_bounds)

        def perform_proximal_grad_step(X, step):
            """Perform a proximal gradient step."""

            return X - step*problem_instance.gradient(X)

        # Convert the lower and upper variable bounds into arrays
        lower_variable_bounds = array(lower_variable_bounds)
        upper_variable_bounds = array(upper_variable_bounds)

        # Initialize the arguments dictionary
        arguments = {
            'e_rel': 5e-3,
            'max_iter': maximum_iterations,
            'callback': partial(
                self.callback, objective=problem_instance.objective)}

        # Check if the algorithm is 'admm'
        if algorithm == 'admm':

            # Set the optimization function
            fun = admm

            # Update by the arguments dictionary
            arguments |= {
                'prox_f': perform_proximal_grad_step,
                'step_f': golden,
                'prox_g': project_on_bounds,
                'step_g': None,
                'L': None,
                'e_abs': 5e-3}

        # Else, check if the algorithm is 'pgm'
        elif algorithm == 'pgm':

            # Set the optimization function
            fun = pgm

            # Update by the arguments dictionary
            arguments |= {
                'grad': problem_instance.gradient,
                'step': golden,
                'prox': project_on_bounds,
                'accelerated': True}

        else:

            # Set the optimization function
            fun = sdmm

            # Update the arguments dictionary
            arguments |= {
                'prox_f': perform_proximal_grad_step,
                'step_f': golden,
                'proxs_g': [project_on_bounds],
                'steps_g': None,
                'Ls': None,
                'e_abs': tolerance}

        return fun, arguments

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
        decision_vector = initial_fluence.copy()

        # Solve the optimization problem
        result = self.fun(X=decision_vector, **self.arguments)

        # Check if the algorithm has converged
        if result[0]:

            # Assign the convergence message
            message = "Convergence of solution reached."

        else:

            # Assign the maximum number of iterations message
            message = "Maximum number of iterations reached."

        return decision_vector, message

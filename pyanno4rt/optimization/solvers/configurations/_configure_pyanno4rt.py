"""Internal algorithm configuration."""

# Author: Tim Ortkamp

# %% Internal package import

# from pyanno4rt.optimization.solvers.internal import ...

# %% Function definition


def configure_pyanno4rt(
        problem_instance, lower_variable_bounds, upper_variable_bounds,
        lower_constraint_bounds, upper_constraint_bounds, algorithm,
        maximum_iterations, tolerance, callback):
    """
    Configure the pyanno4rt solver.

    Supported algorithms: ...

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

    callback : callable
        Callback function from the class \
        :class:`~pyanno4rt.optimization.solvers._pyanno4rt_solver.Pyanno4rtSolver`.

    Returns
    -------
    fun : callable
        Minimization function from the pyanno4rt library.

    arguments : dict
        Dictionary with the solver arguments.
    """

    # Set the optimization function
    fun = ...

    # Initialize the arguments dictionary
    arguments = {
        'fun': problem_instance.objective,
        'grad': problem_instance.gradient,
        'bounds': tuple(
            zip(lower_variable_bounds, upper_variable_bounds)),
        'max_iter': maximum_iterations,
        'tol': tolerance,
        'disp': False,
        'callback': callback}

    # Check if the algorithm is ''
    if algorithm == '':

        # Update by the argument dictionary
        arguments |= {}

        # Check if any constraints have been passed
        if len(problem_instance.constraints) > 0:

            # Update the arguments by the constraints object
            arguments |= {
                'constraints': (
                        problem_instance.constraint, lower_constraint_bounds,
                        upper_constraint_bounds, problem_instance.jacobian),
                'constraint_function': problem_instance.constraint}

    return fun, arguments

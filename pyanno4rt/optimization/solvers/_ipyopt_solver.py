"""Ipyopt wrapper."""

# Author: Tim Ortkamp
# Reference: https://gitlab.com/ipyopt-devs/ipyopt

# %% External package import

from ipyopt import Problem
from numpy import around, array, indices

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class IpyoptSolver():
    """
    Ipyopt wrapper class.

    This class serves as a wrapper for the interior-point optimization \
    algorithms from Ipyopt. It takes the problem structure, configures the \
    selected algorithm, and defines the method to run the solver.

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

    nlp : object of class :class:`~ipyopt.Problem`
        The object used to represent the nonlinear optimization problem.

    arguments : dict
        Dictionary with the solver arguments.
    """

    def __init__(
            self,
            algorithm,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info(
            "Initializing Ipyopt solver with %s algorithm ...", algorithm)

        # Get the input attributes
        self.algorithm = algorithm
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

        # Initialize the NLP and the arguments
        self.nlp, self.arguments = None, None

    def callback(
            self,
            *args):
        """
        Log the intermediate results after each iteration.

        Parameters
        ----------
        *args : tuple
            Callback parameters of the Ipyopt solver.

        Returns
        -------
        bool
            Indicator for the success of the callback iteration.
        """

        # Set the base output string
        output_string = f"At iterate {args[1]}: f={around(args[2], 4)}"

        # Check if any constraints have been passed
        if self.arguments['m'] > 0:

            # Extend the output string
            output_string = f"{output_string}, viol_g={around(args[3], 4)}"

        # Log a message about the intermediate result
        get_logger().info(output_string)

        return True

    def configure(
            self,
            problem):
        """
        Configure the Ipyopt solver.

        Supported algorithms: MUMPS.

        Parameters
        ----------
        problem : object of class \
            :class:`~pyanno4rt.optimization.problems.lexicographic._lexicographic_problem.LexicographicProblem`\
            :class:`~pyanno4rt.optimization.problems.weighted._weighted_sum_problem.WeightedSumProblem`
            The object used to represent the optimization problem.
        """

        def objective(fluence):
            """Get the objective."""

            out = problem.objective(fluence)
            return out

        def gradient(fluence, out):
            """Get the objective gradient."""

            out[()] = problem.gradient(fluence)
            return out

        def constraint(fluence, out):
            """Get the constraints."""

            out[()] = problem.constraint(fluence)
            return out

        def jacobian(fluence, out):
            """Get the constraint Jacobian."""

            out[()] = problem.jacobian(fluence).flatten()
            return out

        # Set the optimization problem class
        self.nlp = Problem

        # Initialize the arguments dictionary
        self.arguments = {
            'n': len(problem.initial_fluence),
            'x_l': array(problem.variable_bounds[0]),
            'x_u': array(problem.variable_bounds[1]),
            'm': len(problem.constraints),
            'g_l': array(problem.constraint_bounds[0]),
            'g_u': array(problem.constraint_bounds[1]),
            'sparsity_indices_jac_g': tuple(
                arr for arr in indices((
                    len(problem.constraints), len(problem.initial_fluence))
                    ).reshape(2, -1)),
            'sparsity_indices_h': None,
            'eval_f': objective,
            'eval_grad_f': gradient,
            'eval_g': constraint,
            'eval_jac_g': jacobian,
            'eval_h': None,
            'intermediate_callback': self.callback,
            'ipopt_options': {
                'sb': 'yes',
                'print_level': 0,
                'tol': 1e-10,
                'dual_inf_tol': 1e-4,
                'constr_viol_tol': 1e-4,
                'compl_inf_tol': 1e-4,
                'acceptable_iter': 5,
                'acceptable_tol': 1e10,
                'acceptable_constr_viol_tol': 1e-2,
                'acceptable_dual_inf_tol': 1e10,
                'acceptable_compl_inf_tol': 1e10,
                'acceptable_obj_change_tol': self.tolerance,
                'max_iter': self.maximum_iterations,
                'mu_strategy': 'adaptive',
                'hessian_approximation': 'limited-memory',
                'limited_memory_max_history': 50,
                'limited_memory_initialization': 'scalar2',
                'linear_solver': self.algorithm}}

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

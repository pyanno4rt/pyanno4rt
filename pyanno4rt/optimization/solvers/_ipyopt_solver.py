"""Ipyopt wrapper."""

# Author: Tim Ortkamp
# Reference: https://gitlab.com/ipyopt-devs/ipyopt

# %% External package import

from ipyopt import Problem
from numpy import around, array, indices

# %% Internal package import

from pyanno4rt.datahub import Datahub

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

    maximum_iterations : int
        Maximum number of iterations.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    nlp : object of class :class:`~ipyopt.Problem`
        The object used to represent the nonlinear optimization problem.

    arguments : dict
        Dictionary with the solver arguments.
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
            f"Initializing Ipyopt solver with {algorithm} algorithm ...")

        # Get the callable nonlinear problem object and its arguments
        self.nlp, self.arguments = self.configure(
            number_of_variables, number_of_constraints, problem_instance,
            lower_variable_bounds, upper_variable_bounds,
            lower_constraint_bounds, upper_constraint_bounds, algorithm,
            maximum_iterations, tolerance)

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
        output_string = f"At iterate {args[1]}: f={'%.4f' % args[2]}"

        # Check if any constraints have been passed
        if self.arguments['m'] > 0:

            # Extend the output string
            output_string = (
                f"{output_string}, viol_g={around(args[3], 4)}")

        # Log a message about the intermediate function values
        Datahub().logger.display_info(output_string)

        return True

    def configure(
            self,
            number_of_variables,
            number_of_constraints,
            problem_instance,
            lower_variable_bounds,
            upper_variable_bounds,
            lower_constraint_bounds,
            upper_constraint_bounds,
            algorithm,
            maximum_iterations,
            tolerance):
        """
        Configure the Ipyopt solver.

        Supported algorithms: MUMPS

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

        maximum_iterations : int
            Maximum number of iterations.

        tolerance : float
            Precision goal for the objective function value.

        Returns
        -------
        nlp : class :class:`~ipyopt.Problem`
            The class used to represent the nonlinear optimization problem.

        arguments : dict
            Dictionary with the solver arguments.
        """

        def objective(fluence):
            """Get the objective."""

            out = problem_instance.objective(fluence)
            return out

        def gradient(fluence, out):
            """Get the gradient of the objective."""

            out[()] = problem_instance.gradient(fluence)
            return out

        def constraint(fluence, out):
            """Get the constraints."""

            out[()] = problem_instance.constraint(fluence)
            return out

        def jacobian(fluence, out):
            """Get the jacobian of the constraints."""

            out[()] = problem_instance.jacobian(fluence).flatten()
            return out

        # Set the optimization problem class
        nlp = Problem

        # Check if the method is 'lexicographic'
        if type(problem_instance).__name__ == 'LexicographicOptimization':

            # Raise an error to indicate the missing implementation
            raise ValueError(
                "Lexicographic optimization is not yet implemented for Ipyopt "
                "...")

        # Initialize the arguments dictionary
        arguments = {
            'n': number_of_variables,
            'x_l': array(lower_variable_bounds),
            'x_u': array(upper_variable_bounds),
            'm': number_of_constraints,
            'g_l': array(lower_constraint_bounds),
            'g_u': array(upper_constraint_bounds),
            'sparsity_indices_jac_g': tuple(
                arr for arr in indices((
                    number_of_constraints,
                    number_of_variables)).reshape(2, -1)),
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
                'acceptable_obj_change_tol': tolerance,
                'max_iter': maximum_iterations,
                'mu_strategy': 'adaptive',
                'hessian_approximation': 'limited-memory',
                'limited_memory_max_history': 50,
                'limited_memory_initialization': 'scalar2',
                'linear_solver': algorithm}}

        return nlp, arguments

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

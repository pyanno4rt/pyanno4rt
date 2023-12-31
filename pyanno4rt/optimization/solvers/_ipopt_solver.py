"""IPOPT wrapper."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>
# Package: https://coin-or.github.io/Ipopt/
# See INSTALL_IPOPT_ON_LINUX.md for Linux installation instructions

# %% External package import

import cyipopt as ipopt

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class IpoptSolver():
    """
    IPOPT wrapper class.

    This class provides methods for wrapping the IPOPT solver, including the \
    initialization of the algorithm from the arguments set in the treatment \
    plan, the composition of an IPOPT-compatible optimization problem, and a \
    method to start the algorithm.

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
    nlp : object of class `cyipopt.problem`
        IPOPT-compatible instance of the optimization problem.

    options : dict
        Dictionary with the solver options set by the user.
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
        Datahub().logger.display_info("Initializing IPOPT solver with {} ..."
                                      .format(algorithm))

        # Initialize the IPOPT problem instance
        self.nlp = ipopt.problem(n=number_of_variables,
                                 m=number_of_constraints,
                                 problem_obj=problem_instance,
                                 lb=lower_variable_bounds,
                                 ub=upper_variable_bounds,
                                 cl=lower_constraint_bounds,
                                 cu=upper_constraint_bounds)

        # Set the solver options
        self.set_solver_options(algorithm, max_iter, max_cpu_time)

    def set_solver_options(
            self,
            algorithm,
            max_iter,
            max_cpu_time):
        """
        Set the optional parameters for the IPOPT solver.

        Parameters
        ----------
        algorithm : string
            Label for the solution algorithm.

        max_iter : int
            Maximum number of iterations taken for the solver to converge.

        max_cpu_time : float
            Maximum CPU time taken for the solver to converge.

        Returns
        -------
        dict
            Dictionary with the solver options set by the user.
        """
        # Create a dictionary with the solver options
        options = {
               'tol': 1e-8,
               'dual_inf_tol': 1.0,
               'constr_viol_tol': 1e-4,
               'compl_inf_tol': 1e-4,
               'acceptable_iter': 3,
               'acceptable_tol': 1e10,
               'acceptable_constr_viol_tol': 1e10,
               'acceptable_dual_inf_tol': 1e10,
               'acceptable_compl_inf_tol': 1e10,
               'acceptable_obj_change_tol': 1e-3,
               'max_iter': max_iter,
               'max_cpu_time': max_cpu_time,
               'mu_strategy': 'adaptive',
               'hessian_approximation': 'limited-memory',
               'limited_memory_max_history': 6,
               'limited_memory_initialization': 'scalar2',
               'linear_solver': algorithm
               }

        # Add the options to the IPOPT problem instance
        for key, val in options.items():
            self.nlp.addOption(key, val)

    def start(
            self,
            initial_fluence):
        """
        Run the IPOPT solver.

        Parameters
        ----------
        initial_fluence : ndarray
            Initial fluence vector.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        dict
            Dictionary with information on the algorithm state.
        """
        # Solve the optimization problem
        optimized_fluence, solver_info = self.nlp.solve(initial_fluence)

        return optimized_fluence, solver_info

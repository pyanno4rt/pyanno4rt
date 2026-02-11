"""Fluence optimization."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from functools import reduce
from numpy import load, save, union1d
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.logging import get_logger
import pyanno4rt.optimization._maps as maps
from pyanno4rt.tools import (
   get_constraints, get_machine_learning_components, get_objectives,
   get_radiobiological_components, get_all_segments)

# %% Class definition


class FluenceOptimizer():
    """
    Fluence optimization class.

    This class provides methods to optimize the fluence vector by solving the \
    inverse planning problem. It takes the configuration inputs, sets up the \
    optimization problem and the solver, and allows to compute both optimized \
    fluence vector and optimized dose cube.

    Parameters
    ----------
    handlers : dict
        Dictionary with the handlers (patient, plan, dose, data models).

    Attributes
    ----------
    handlers : dict
        See 'Parameters'.

    initializer : None or object of class \
        :class:`~pyanno4rt.optimization.initializers._data_medoid_initializer.DataMedoidInitializer`\
        :class:`~pyanno4rt.optimization.initializers._target_coverage_initializer.TargetCoverageInitializer`\
        :class:`~pyanno4rt.optimization.initializers._warm_start_initializer.WarmStartInitializer`
        The object used to represent the fluence vector initializer.

    problem : None or object of class \
        :class:`~pyanno4rt.optimization.problems.lexicographic._lexicographic_problem.LexicographicProblem`\
        :class:`~pyanno4rt.optimization.problems.pareto._pareto_problem.ParetoProblem`\
        :class:`~pyanno4rt.optimization.problems.weighted._weighted_sum_problem.WeightedSumProblem`
        The object used to represent the optimization problem.

    solver : None or object of class \
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`\
        :class:`~pyanno4rt.optimization.solvers._pyanno4rt_solver.Pyanno4rtSolver`\
        :class:`~pyanno4rt.optimization.solvers._pymoo_solver.PymooSolver`\
        :class:`~pyanno4rt.optimization.solvers._pypop7_solver.PyPop7Solver`\
        :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`
        The object used to represent the solver.

    solver_time : float
        Runtime for solving the optimization problem.

    optimized_fluence : ndarray
        Optimized fluence vector.

    solver_info : str
        Description for the cause of termination.

    optimized_dose : ndarray
        Optimized dose cube (CT resolution).
    """

    def __init__(
            self,
            handlers):

        # Log a message about the initialization of the class
        get_logger().info("Initializing fluence optimizer ...")

        # Get the data handlers
        self.handlers = handlers

        # Initialize the optimization model
        self.initializer = None
        self.problem = None
        self.solver = None

        # Initialize the solver runtime
        self.solver_time = None

        # Initialize the optimization results
        self.optimized_fluence, self.solver_info, self.optimized_dose = (
            None, None, None)

    def initialize_fluence(
            self,
            initial_strategy='target-coverage',
            initial_fluence=None):
        """
        Initialize the fluence vector.

        Parameters
        ----------
        initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}, \
            default='target-coverage'
            Initialization strategy for the fluence vector.

        initial_fluence : None or list, default=None
            User-defined initial fluence vector for the optimization problem.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        # Set the fluence initializer
        self.initializer = maps.INITIALIZERS[initial_strategy](initial_fluence)

        return self.initializer.run(self.handlers)

    def initialize_problem(
            self,
            method='weighted-sum',
            initial_strategy='target-coverage',
            initial_fluence=None,
            lower_variable_bounds=None,
            upper_variable_bounds=None):
        """
        Initialize the optimization problem.

        Parameters
        ----------
        method : {'lexicographic', 'pareto', 'weighted-sum'}, \
            default='weighted-sum'
            Single- or multi-criteria optimization method.

        initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}, \
            default='target-coverage'
            Initialization strategy for the fluence vector.

        initial_fluence : None or list, default=None
            User-defined initial fluence vector for the optimization problem.

        lower_variable_bounds : None, int, float, or list, default=None
            Lower bound(s) on the decision variables.

        upper_variable_bounds : None, int, float, or list, default=None
            Upper bound(s) on the decision variables.
        """

        # Get the data handlers
        plan_handler = self.handlers['plan_handler']
        dose_handler = self.handlers['dose_handler']

        # Get the backprojection
        backprojection = maps.PROJECTIONS[plan_handler.modality](
            dose_handler.dose_influence_matrix, plan_handler.RBE)

        # Get the objective and constraint functions
        objectives = get_objectives(plan_handler.components)
        constraints = get_constraints(plan_handler.components)

        # Get the initial fluence vector
        initial_fluence = self.initialize_fluence(
            initial_strategy, initial_fluence)

        # Construct the optimization problem
        self.problem = maps.PROBLEMS[method](
            backprojection, objectives, constraints, lower_variable_bounds,
            upper_variable_bounds, initial_fluence)

    def initialize_solver(
            self,
            solver='scipy',
            algorithm='L-BFGS-B',
            maximum_iterations=500,
            tolerance=1e-3):
        """
        Initialize the solver.

        Parameters
        ----------
        solver : {'ipyopt', 'pyanno4rt', 'pymoo', 'pypop7', 'scipy'}, \
            default='scipy'
            Python package to be used for solving the optimization problem.

        algorithm : str, default='L-BFGS-B'
            Solution algorithm from the chosen solver.

        maximum_iterations : int, default=500
            Maximum number of iterations taken for the solver to converge. If \
            set to zero, the initial fluence is used as solution.

        tolerance : float, default=1e-3
            Precision goal for the objective function value.
        """

        # Check if the solver ignores any constraints
        if len(self.problem.constraints) > 0 and algorithm not in (
                'mumps', 'NSGA3', 'trust-constr'):

            # Log a message about ignoring the constraints
            get_logger().warning(
                "The '%s' algorithm only allows for unconstrained "
                "optimization problems - constraints set will be ignored ...",
                algorithm)

            # Reset the constraints
            self.problem.constraints = ()

        # Initialize the solver instance
        self.solver = maps.SOLVERS[solver](
            algorithm=algorithm, maximum_iterations=maximum_iterations,
            tolerance=tolerance)

    def solve(self):
        """Solve the optimization problem."""

        # Log a message about the problem solving
        get_logger().info("Solving optimization problem ...")

        # Reset the optimization outputs
        self.reset()

        # Start the solver runtime recording
        start_time = time()

        # Check if the maximum number of iterations is set to zero
        if self.solver.maximum_iterations > 0:

            # Map the problem to the solution methods
            methods = {
                'lexicographic': self._solve_lexicography,
                'pareto': self._solve_pareto,
                'weighted-sum': self._solve_weighted}

            # Solve the problem
            self.optimized_fluence, self.solver_info, self.optimized_dose = (
                methods[self.problem.name]())

        else:

            # Log a message about falling back to the initial solution
            get_logger().info(
                "Maximum number of iterations is set to zero - retrieving "
                "optimized fluence from the initialization ...")

            # Get the optimized fluence from the initialization
            self.optimized_fluence = self.problem.initial_fluence

            # Compute the optimized dose
            self.optimized_dose = self.compute_dose_3d(self.optimized_fluence)

        # Reevaluate the optimized fluence
        self.problem.objective(self.optimized_fluence)
        self.problem.constraint(self.optimized_fluence)

        # Stop the solver runtime recording
        self.solver_time = round(time()-start_time, 2)

        # Loop over the machine learning outcome model-based components
        for component in get_machine_learning_components(
                self.problem.constraints + self.problem.objectives):

            # Convert the feature histories
            component.model.feature_calculator.history_to_dict()

        # Log the outcome result
        self.log_outcome()

        # Log a message about the solver runtime
        get_logger().info(
            "Fluence optimizer took %s seconds for problem solving ...",
            self.solver_time)

    def _solve_lexicography(self):
        """
        Solve the lexicographic optimization problem.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        str
            Description for the cause of termination.
        """

        # Get all ranks
        ranks = tuple(self.problem.subproblems)

        # Get the initial fluence vector
        fluence = self.problem.subproblems[ranks[0]].initial_fluence

        # Loop over the lexicographic ranks and problems
        for rank, problem in self.problem.subproblems.items():

            # Log a message about the current rank
            get_logger().info("Considering lexicography at rank %s ...", rank)

            # Configure the solver
            self.solver.configure(problem)

            # Solve the optimization problem
            fluence, solver_info = self.solver.run(fluence)

            # Check if the current rank is not the final one
            if rank != ranks[-1]:

                # Get the next subproblem
                next_problem = self.problem.subproblems[
                    ranks[ranks.index(rank)+1]]

                # Update the initial fluence
                next_problem.initial_fluence = fluence

                # Loop over the dynamic components
                for objective in problem.objectives:

                    # Update the upper bound
                    objective.bounds[1] = (
                        problem.tracker[objective.track_id][-1])

                # Update the constraint bounds
                next_problem.constraint_bounds = (
                    next_problem.get_constraint_bounds())

        # Restore the lexicographic tracker
        self.problem.restore_tracker()

        # Check if a solution has been found
        if fluence is not None:

            # Compute the optimized dose
            optimized_dose = self.compute_dose_3d(fluence)

        else:

            # Log a message about the infeasibility
            get_logger().info(
                "Fluence optimizer did not find a feasible solution for the "
                "lexicographic optimization problem ...")

            # Set the optimized dose to None
            optimized_dose = None

        return fluence, solver_info, optimized_dose

    def _solve_pareto(self):
        """
        Solve the Pareto optimization problem.

        Returns
        -------
        ndarray
            Optimized fluence vector(s).

        str
            Description for the cause of termination.
        """

        # Configure the solver
        self.solver.configure(self.problem)

        # Solve the optimization problem
        optimized_fluence, solver_info = self.solver.run(
            self.problem.initial_fluence)

        # Check if a solution has been found
        if optimized_fluence is not None:

            # Log a message about the number of Pareto-optimal solutions
            get_logger().info(
                "Pareto analysis resulted in %s non-dominated solutions ...",
                optimized_fluence.shape[0])

            # Log a message about the sorting criterion
            get_logger().info(
                "Selecting best solution with respect to the maximum mean "
                "dose difference between targets and organs at risk ...")

            # Get the segmentation data
            segmentation = self.handlers['patient_handler'].segmentation

            # Get the dose-influence matrix
            dose_matrix = self.handlers['dose_handler'].dose_influence_matrix

            # Get the indices of relevant targets and OARs
            target_indices, oar_indices = (reduce(
                union1d,
                (segmentation[segment]['resized_indices']
                 for segment in get_all_segments(
                     self.problem.constraints + self.problem.objectives)
                 if segmentation[segment]['type'] == string), -1)
                for string in ('TARGET', 'OAR'))

            # Initialize the results list
            results = []

            # Loop over the non-dominated solutions
            for fluence in optimized_fluence:

                # Calculate the dose vector
                dose = dose_matrix @ fluence

                # Calculate the mean dose difference
                delta = dose[target_indices].mean() - dose[oar_indices].mean()

                # Add the fluence/score
                results.append((fluence, delta))

            # Sort the results by the score
            results = sorted(results, key=lambda x: x[1])

            # Compute the optimized dose from the "best" result
            optimized_dose = self.compute_dose_3d(results[0][0])

        else:

            # Log a message about the infeasibility
            get_logger().info(
                "Fluence optimizer did not find a feasible solution for the "
                "Pareto optimization problem ...")

            # Set the optimized dose to None
            optimized_dose = None

        return optimized_fluence, solver_info, optimized_dose

    def _solve_weighted(self):
        """
        Solve the weighted-sum optimization problem.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        str
            Description for the cause of termination.
        """

        # Configure the solver
        self.solver.configure(self.problem)

        # Solve the optimization problem
        optimized_fluence, solver_info = self.solver.run(
            self.problem.initial_fluence)

        # Check if a solution has been found
        if optimized_fluence is not None:

            # Compute the optimized dose
            optimized_dose = self.compute_dose_3d(optimized_fluence)

        else:

            # Log a message about the infeasibility
            get_logger().info(
                "Fluence optimizer did not find a feasible solution for the "
                "weighted-sum optimization problem ...")

            # Set the optimized dose to None
            optimized_dose = None

        return optimized_fluence, solver_info, optimized_dose

    def compute_dose_3d(
            self,
            optimized_fluence):
        """
        Compute the dose cube from the optimized fluence vector.

        Parameters
        ----------
        optimized_fluence : ndarray
            Optimized fluence vector.

        Returns
        -------
        ndarray
            Optimized dose cube (CT resolution).
        """

        # Get the data handlers
        patient_handler = self.handlers['patient_handler']
        plan_handler = self.handlers['plan_handler']
        dose_handler = self.handlers['dose_handler']

        # Log a message about the 3D dose computation
        get_logger().info(
            "Computing dose cube from optimized fluence vector ...")

        # Get the CT and dose grid dimensions
        ct_dim, dose_dim = (
            patient_handler.computed_tomography['cube_dimensions'],
            dose_handler.cube_dimensions)

        # Compute the optimized dose vector from the optimized fluence
        optimized_dose = dose_handler.dose_influence_matrix @ optimized_fluence

        # Reshape the optimized dose vector to the dose cube
        optimized_dose = optimized_dose.reshape(dose_dim, order='F')

        # Get the zoom factors for all cube dimensions
        zooms = (pair[0]/pair[1] for pair in zip(ct_dim, dose_dim))

        # Interpolate the dose cube to the CT grid and apply rescaling
        optimized_dose = (
            zoom(optimized_dose, zooms, order=1)
            * plan_handler.RBE
            * dose_handler.number_of_fractions)

        return optimized_dose

    def reset(self):
        """Reset the optimization and evaluation outputs."""

        # Reset the problem tracker
        self.problem.tracker = {key: [] for key in self.problem.tracker}

        # Check if the lexicographic method has been selected
        if self.problem.name == 'lexicographic':

            # Loop over the subproblems
            for subproblem in self.problem.subproblems.values():

                # Reset the subproblem tracker
                subproblem.tracker = {key: [] for key in subproblem.tracker}

        # Loop over the machine learning outcome model-based components
        for component in get_machine_learning_components(
                self.problem.constraints + self.problem.objectives):

            # Get the feature calculator
            feature_calculator = component.model.feature_calculator

            # Reset the feature history
            feature_calculator.feature_history = []

    def load_fluence(
            self,
            path):
        """
        Load the fluence array from a path.

        Parameters
        ----------
        path : str
            Path for loading the fluence array.
        """

        self.optimized_fluence = load(path)

    def save_fluence(
            self,
            path):
        """
        Save the fluence to a binary file.

        Parameters
        ----------
        path : str
            Path for storing the optimized fluence array.
        """

        save(path, self.optimized_fluence)

    def load_dose(
            self,
            path):
        """
        Load the dose array from a path.

        Parameters
        ----------
        path : str
            Path for loading the dose array.
        """

        self.optimized_dose = load(path)

    def save_dose(
            self,
            path):
        """
        Save the dose array to a binary file.

        Parameters
        ----------
        path : str
            Path for storing the optimized dose array.
        """

        save(path, self.optimized_dose)

    def log_outcome(self):
        """Log the outcome model-based component results."""

        # Loop over the radiobiological outcome model-based components
        for component in get_radiobiological_components(
                self.problem.constraints + self.problem.objectives):

            # Get the final outcome prediction value
            value = (
                (-1)**('NTCP' not in component.name)
                * self.problem.tracker[component.track_id][-1])

            # Log a message about the outcome value
            get_logger().info(
                "%s for the optimized plan: %s %% ...", component.name,
                round(100*value, 2))

        # Loop over the machine learning outcome model-based components
        for component in get_machine_learning_components(
                self.problem.constraints + self.problem.objectives):

            # Get the final outcome prediction
            value = component.translate(
                self.problem.tracker[component.track_id][-1])

            # Store the outcome value
            self.handlers['data_model_handler'].outcomes[
                component.model.label] = value

            # Log a message about the outcome value
            get_logger().info(
                "%s (%s) for the optimized plan: %s %% ...",
                component.name, component.outcome_type, round(100*value, 2))

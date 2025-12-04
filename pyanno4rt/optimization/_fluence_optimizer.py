"""Fluence optimization."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from functools import reduce
from numpy import empty, union1d
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

    method : {'lexicographic', 'pareto', 'weighted-sum'}
        Single- or multi-criteria optimization method.

    solver : {'ipyopt', 'pyanno4rt', 'pymoo', 'pypop7', 'scipy'}
        Python package to be used for solving the optimization problem.

    algorithm : str
        Solution algorithm from the chosen solver.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        Initialization strategy for the fluence vector.

    initial_fluence : None or list
        User-defined initial fluence vector for the optimization problem.

    lower_variable_bounds : None, int, float, or list
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list
        Upper bound(s) on the decision variables.

    maximum_iterations : int
        Maximum number of iterations taken for the solver to converge.

    tolerance : float
        Precision goal for the objective function value.

    Attributes
    ----------
    handlers : dict
        See 'Parameters'.

    problem : object of class \
        :class:`~pyanno4rt.optimization.problems.lexicographic._lexicographic_problem.LexicographicProblem`\
        :class:`~pyanno4rt.optimization.problems.pareto._pareto_problem.ParetoProblem`\
        :class:`~pyanno4rt.optimization.problems.weighted._weighted_sum_problem.WeightedSumProblem`
        The object used to represent the optimization problem.

    solver : object of class \
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`\
        :class:`~pyanno4rt.optimization.solvers._pyanno4rt_solver.Pyanno4rtSolver`\
        :class:`~pyanno4rt.optimization.solvers._pymoo_solver.PymooSolver`\
        :class:`~pyanno4rt.optimization.solvers._pypop7_solver.PyPop7Solver`\
        :class:`~pyanno4rt.optimization.solvers._scipy_solver.SciPySolver`
        The object used to represent the solver.

    initial_time : float
        Runtime for initializing the optimizer.

    solver_time : float
        Runtime for solving the optimization problem.

    optimizer_time : float
        Total runtime for the optimizer.

    optimized_fluence : ndarray
        Optimized fluence vector.

    solver_info : str
        Description for the cause of termination.

    optimized_dose : ndarray
        Optimized dose cube (CT resolution).

    from_copycat : bool
        Indicator for loading the optimized fluence from a copycat.
    """

    def __init__(
            self,
            handlers,
            method,
            solver,
            algorithm,
            initial_strategy,
            initial_fluence,
            lower_variable_bounds,
            upper_variable_bounds,
            maximum_iterations,
            tolerance):

        # Log a message about the initialization of the class
        get_logger().info("Initializing fluence optimizer ...")

        # Start the constructor runtime recording
        start_time = time()

        # Get the data handlers
        self.handlers = handlers

        # Get the objective and constraint functions
        objectives = get_objectives(handlers['plan_handler'].components)
        constraints = get_constraints(handlers['plan_handler'].components)

        # Check if the solver ignores any constraints
        if len(constraints) > 0 and algorithm not in (
                'mumps', 'NSGA3', 'trust-constr'):

            # Log a message about ignoring the constraints
            get_logger().warning(
                "The '%s' algorithm only allows for unconstrained "
                "optimization problems - constraints set will be ignored ...",
                algorithm)

            # Reset the constraints
            constraints = ()

        # Initialize the backprojection
        backprojection = maps.PROJECTIONS[handlers['plan_handler'].modality](
            handlers['dose_handler'].dose_influence_matrix,
            handlers['plan_handler'].RBE)

        # Calculate the initial fluence
        initializer = maps.INITIALIZERS[initial_strategy](initial_fluence)
        initial_fluence = initializer.run(handlers)

        # Construct the optimization problem
        self.problem = maps.PROBLEMS[method](
            backprojection, objectives, constraints, lower_variable_bounds,
            upper_variable_bounds, initial_fluence)

        # Initialize the solver instance
        self.solver = maps.SOLVERS[solver](
            algorithm=algorithm, maximum_iterations=maximum_iterations,
            tolerance=tolerance)

        # Get the initialization runtime
        self.initial_time = time()-start_time

        # Initialize the solver and optimizer runtimes
        self.solver_time, self.optimizer_time = None, None

        # Initialize the optimization results
        self.optimized_fluence, self.solver_info, self.optimized_dose = (
            None, None, None)

        # Initialize the copycat indicator
        self.from_copycat = False

    def solve(self):
        """Solve the optimization problem."""

        # Log a message about the problem solving
        get_logger().info("Solving optimization problem ...")

        # Reset the optimization outputs
        self.reset()

        # Start the solver runtime recording
        start_time = time()

        # Check if the fluence can not be loaded from a copycat
        if not self.from_copycat:

            # Map the problem to the solution methods
            methods = {
                'lexicographic': self._solve_lexicography,
                'pareto': self._solve_pareto,
                'weighted-sum': self._solve_weighted}

            # Run the solution algorithm
            self.optimized_fluence, self.solver_info, self.optimized_dose = (
                methods[self.problem.name]())

        else:

            # Log a message about the imported fluence
            get_logger().info(
                "Retrieving solution from the loaded treatment plan ...")

            # Compute the optimized dose
            self.optimized_dose = self.compute_dose_3d(self.optimized_fluence)

            # Reset the copycat indicator
            self.from_copycat = False

        # Get the runtime for problem solving
        self.solver_time = round(time()-start_time, 2)

        # Log the outcome result
        self.log_outcome()

        # Get the runtime for the fluence optimizer
        self.optimizer_time = round(time()-start_time+self.initial_time, 2)

        # Log a message about the optimization runtimes
        get_logger().info(
            "Fluence optimizer took %s seconds (%s seconds for problem "
            "solving) ...", self.optimizer_time, self.solver_time)

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

        # Loop over the lexicographic ranks
        for rank, problem in self.problem.subproblems.items():

            # Log a message about the lexicographic rank
            get_logger().info("Considering lexicography at rank %s ...", rank)

            # Configure the solver
            self.solver.configure(problem)

            # Solve the optimization problem at the current rank
            fluence, solver_info = self.solver.run(fluence)

            # Check if the current rank does not equal the final rank
            if rank != ranks[-1]:

                # Get the next subproblem
                next_problem = self.problem.subproblems[
                    ranks[ranks.index(rank)+1]]

                # Overwrite the initial fluence
                next_problem.initial_fluence = fluence

                # Loop over the dynamic components
                for objective in problem.objectives:

                    # Adapt the upper bound
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

            # Log a message about the unsolved problem
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
            Optimized fluence vector.

        str
            Description for the cause of termination.
        """

        # Get the segmentation data
        segmentation = self.handlers['patient_handler'].segmentation

        # Get the dose-influence matrix
        dose_matrix = self.handlers['dose_handler'].dose_influence_matrix

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

            # Log a message about the selection procedure
            get_logger().info(
                "Selecting best solution with respect to the maximum mean "
                "dose difference between targets and organs at risk ...")

            # Get the indices of relevant targets and OARs
            target_indices, oar_indices = (reduce(
                union1d,
                (segmentation[segment]['resized_indices']
                 for segment in get_all_segments(
                     self.problem.constraints + self.problem.objectives)
                 if segmentation[segment]['type'] == string), -1)
                for string in ('TARGET', 'OAR'))

            # Initialize the score list
            scores = []

            # Loop over the non-dominated solutions
            for fluence in optimized_fluence:

                # Calculate the dose vector
                dose = dose_matrix @ fluence

                # Add the solution-score pair
                scores.append(
                    (fluence,
                     dose[target_indices].mean() - dose[oar_indices].mean()))

            # Sort the results
            scores = sorted(scores, key=lambda x: x[1])

            # Compute the optimized dose from the "best" fluence
            optimized_dose = self.compute_dose_3d(scores[0][0])

        else:

            # Log a message about the unsolved problem
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

            # Log a message about the unsolved problem
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
            Optimized fluence vector(s).

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

        # Interpolate the dose cube to the CT grid and multiply by the RBE
        optimized_dose = (
            zoom(optimized_dose, zooms, order=1)
            * plan_handler.RBE
            * dose_handler.number_of_fractions)

        return optimized_dose

    def reset(self):
        """Reset the optimization and evaluation outputs."""

        # Reset the problem tracker
        self.problem.tracker = {key: [] for key in self.problem.tracker}

        # Check if lexicographic optimization has been selected
        if self.problem.name == 'lexicographic':

            # Loop over the subproblems
            for subproblem in self.problem.subproblems.values():

                # Reset the subproblem tracker
                subproblem.tracker = {key: [] for key in subproblem.tracker}

        # Loop over the machine learning model-based components
        for component in get_machine_learning_components(
                self.problem.constraints + self.problem.objectives):

            # Get the feature calculator of the component
            feature_calculator = (
                component.data_model_handler.feature_calculator)

            # Reset the feature history
            feature_calculator.feature_history = empty(
                shape=(1, len(feature_calculator.feature_map)))

    def log_outcome(self):
        """Log the outcome model-based component results."""

        # Check if the optimization problem has a tracker dictionary
        if hasattr(self.problem, 'tracker') and all(value != [] for value
           in self.problem.tracker.values()):

            # Loop over the radiobiological outcome model-based components
            for component in get_radiobiological_components(
                    self.problem.constraints + self.problem.objectives):

                # Get the final (N)TCP prediction value
                value = (
                    (-1)**('NTCP' not in component.name)
                    * self.problem.tracker[component.track_id][-1])

                # Log a message about the prediction value
                get_logger().info(
                    "%s for the optimized plan: %s %% ...",
                    component.name, round(100*value, 2))

            # Loop over the machine learning outcome model-based components
            for component in get_machine_learning_components(
                    self.problem.constraints + self.problem.objectives):

                # Process the feature history
                component.data_model_handler.process_feature_history()

                # Get the final (N)TCP prediction
                value = component.translate(
                    self.problem.tracker[component.track_id][-1])

                # Store the prediction value
                component.data_model_handler.model_outcomes[
                    component.data_model_handler.model_label] = value

                # Log a message about the prediction value
                get_logger().info(
                    "%s for the optimized plan: %s %% ...",
                    component.name, round(100*value, 2))

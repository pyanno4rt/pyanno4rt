"""Fluence optimization."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from functools import reduce
from numpy import (
    empty, ravel_multi_index, setdiff1d, union1d, unravel_index, where, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.logging import get_logger
import pyanno4rt.optimization._maps as maps
from pyanno4rt.tools import (
   apply, flatten, get_constraint_segments, get_machine_learning_constraints,
   get_machine_learning_objectives, get_radiobiological_constraints,
   get_radiobiological_objectives, get_objective_segments)

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
    method : {'lexicographic', 'pareto', 'weighted-sum'}
        Single- or multi-criteria optimization method.

    solver : {'ipyopt', 'pymoo', 'pypop7', 'scipy'}
        Python package to be used for solving the optimization problem.

    algorithm : str
        Solution algorithm from the chosen solver.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        Initialization strategy for the fluence vector.

    initial_fluence : None or list
        User-defined initial fluence vector for the optimization problem \
        (only used if initial_strategy='warm-start').

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
    problem : object of class \
        :class:`~pyanno4rt.optimization.problems._lexicographic_problem.LexicographicProblem`\
        :class:`~pyanno4rt.optimization.problems._pareto_problem.ParetoProblem`\
        :class:`~pyanno4rt.optimization.problems._weighted_sum_problem.WeightedSumProblem`
        The object used to represent the optimization problem.
        ...

    solver : object of class \
        :class:`~pyanno4rt.optimization.solvers._ipyopt_solver.IpyoptSolver`\
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
    """

    def __init__(
            self,
            method,
            solver,
            algorithm,
            initial_strategy,
            initial_fluence,
            lower_variable_bounds,
            upper_variable_bounds,
            maximum_iterations,
            tolerance):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        get_logger().info("Initializing fluence optimizer ...")

        # Start the constructor runtime recording
        start_time = time()

        # Get the objective and constraint functions
        objectives, constraints = (
            hub.optimization['objectives'], hub.optimization['constraints'])

        # Remove overlaps between segments according to their priority
        objectives, constraints = FluenceOptimizer.remove_overlap(
            objectives, constraints)

        # Resize the segments to the dose grid
        FluenceOptimizer.resize_segments_to_dose()

        # Adjust the dose-volume-related parameters for fractionation
        FluenceOptimizer.adjust_parameters_for_fractionation(
            objectives | constraints)

        # Initialize the backprojection
        backprojection = maps.PROJECTIONS[hub.plan_configuration['modality']]()

        # Check if the solver ignores any constraints
        if len(constraints) > 0 and algorithm not in (
                'mumps', 'NSGA3', 'trust-constr'):

            # Log a message about the ignored constraints
            get_logger().warning(
                "The '%s' algorithm only allows for unconstrained "
                "optimization problems - constraints set will be ignored ...",
                algorithm)

            # Reset the internal constraints
            hub.optimization['constraints'], constraints = {}, {}

            # Loop over the segments
            for segment in hub.segmentation:

                # Reset the segment constraints
                hub.segmentation[segment]['constraint'] = None

        # Calculate the initial fluence
        initial_fluence = maps.INITIALIZERS[initial_strategy](
            initial_fluence).run()

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

        # Initialize the optimization results
        self.optimized_fluence, self.solver_info, self.optimized_dose = (
            None, None, None)

        # Initialize the solver and optimizer runtimes
        self.solver_time, self.optimizer_time = None, None

        # Enter the optimization dictionary into the datahub
        hub.optimization |= {
            'problem': self.problem, 'solver_instance': self.solver}

    @staticmethod
    def remove_overlap(objectives, constraints):
        """
        Remove overlaps between segments.

        Parameters
        ----------
        objectives : dict
            Dictionary with the plan objectives.

        constraints : dict
            Dictionary with the plan constraints.

        Returns
        -------
        dict
            Cleaned dictionary with the plan objectives.

        dict
            Cleaned dictionary with the plan constraints.
        """

        def remove_segment_overlap(reference):
            """Remove the overlap from a reference segment."""

            # Get the indices from all higher prioritized VOIs
            superior_indices = (
                segmentation[segment]['raw_indices'] for segment in segments
                if (segmentation[segment]['parameters']['priority']
                    < segmentation[reference]['parameters']['priority']))

            # Get the overlap-free (prioritized) indices
            segmentation[reference]['prioritized_indices'] = setdiff1d(
                segmentation[reference]['raw_indices'],
                reduce(union1d, superior_indices, -1))

            # Check if the prioritized index set is empty and relevant
            if (len(segmentation[reference]['prioritized_indices']) == 0 and (
                    segmentation[reference]['objective'] is not None or
                    segmentation[reference]['constraint'] is not None)):

                # Loop over the component types and dictionaries
                for label, dictionary in {
                        'objective': objectives,
                        'constraint': constraints}.items():

                    # Loop over the component keys
                    for key in (
                        key for key in dictionary
                            if reference in dictionary[key]['segments']):

                        # Remove the reference segment
                        dictionary[key]['segments'].remove(reference)

                        # Check if the segment list is empty
                        if len(dictionary[key]['segments']) == 0:

                            # Log a message about the component removal
                            get_logger().info(
                                "Removing %s '%s' from fully enclosed segment "
                                "'%s' ...",
                                label, dictionary[key]['instance'].name,
                                reference)

                            # Delete the component from the dictionaries
                            del dictionary[key]
                            segmentation[reference][label] = None

        # Log a message about the overlap removal
        get_logger().info("Removing segment overlaps ...")

        # Get the segmentation data
        segmentation = Datahub().segmentation

        # Get all segments from the components
        segments = set(flatten(
            component['segments']
            for component in (objectives | constraints).values()))

        # Remove the overlaps from all segments
        apply(remove_segment_overlap, (*segmentation,))

        return objectives, constraints

    @staticmethod
    def resize_segments_to_dose():
        """Resize the segments from CT to dose grid."""

        def resize_segment(segment):
            """Resize a segment to the dose grid."""

            # Initialize the segment mask
            mask = zeros(ct_dim)

            # Fill the mask at the indices of the segment
            mask[unravel_index(
                segmentation[segment]['prioritized_indices'], ct_dim,
                order='F')] = 1

            # Get the zoom factors for all cube dimensions
            zooms = (pair[0]/pair[1] for pair in zip(dose_dim, ct_dim))

            # Enter the dose grid level (resized) indices into the datahub
            segmentation[segment]['resized_indices'] = ravel_multi_index(
                where(zoom(mask, zooms, order=0)), dose_dim, order='F')

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the segment resizing
        get_logger().info("Resizing segments from CT to dose grid ...")

        # Get the segmentation data
        segmentation = hub.segmentation

        # Get the CT and dose cube dimensions
        ct_dim, dose_dim = (
            hub.computed_tomography['cube_dimensions'],
            hub.dose_information['cube_dimensions'])

        # Resize all segments
        apply(resize_segment, (*segmentation,))

    @staticmethod
    def adjust_parameters_for_fractionation(components):
        """
        Adjust the dose parameters according to the number of fractions.

        Parameters
        ----------
        components : dict
            Dictionary with the internally configured objectives/constraints.
        """

        def adjust_component(component):
            """Adjust the dose parameters for a component."""

            # Get the component parameters
            parameters = component.get_parameter_value()

            # Loop over the indices of the dose-related parameter values
            for index in (index for index, category in enumerate(
                    component.parameter_category) if category == 'dose'):

                # Adjust the indexed parameters by the number of fractions
                parameters[index] /= number_of_fractions

            # Set the adjusted objective parameters
            component.set_parameter_value(parameters)

            # Activate the adjustment indicator of the component
            component.adjusted_parameters = True

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the parameter adjustment
        get_logger().info("Adjusting dose parameters for fractionation ...")

        # Get the number of fractions
        number_of_fractions = hub.dose_information['number_of_fractions']

        # Adjust all non-adjusted components with dose-related parameters
        apply(adjust_component, (
            component['instance'] for component in components.values()
            if not component['instance'].adjusted_parameters
            and 'dose' in component['instance'].parameter_category))

    def solve(self):
        """Solve the optimization problem."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the problem solving
        get_logger().info("Solving optimization problem ...")

        # Start the solver runtime recording
        start_time = time()

        # Reset the optimization outputs
        self.reset()

        # Check if the fluence can not be loaded from a copycat
        if 'from_copycat' not in hub.optimization:

            # Collect the solution methods
            methods = {
                'lexicographic': self.solve_lexicography,
                'pareto': self.solve_pareto,
                'weighted-sum': self.solve_weighted}

            # Run the solution algorithm
            self.optimized_fluence, self.solver_info, self.optimized_dose = (
                methods[self.problem.name]())

            #
            (hub.optimization['optimized_fluence'],
             hub.optimization['solver_info'],
             hub.optimization['optimized_dose']) = (
                 self.optimized_fluence, self.solver_info, self.optimized_dose)

        else:

            # Log a message about the loaded fluence
            get_logger().info(
                "Retrieving solution from the loaded treatment plan ...")

            # Delete the copycat indicator
            del hub.optimization['from_copycat']

        # Get the runtime for problem solving
        self.solver_time = round(time()-start_time, 2)

        # Postprocess the outcome model-based component results
        self.postprocess()

        # Get the runtime for the fluence optimizer
        self.optimizer_time = round(time()-start_time+self.initial_time, 2)

        # Log a message about the optimization runtimes
        get_logger().info(
            "Fluence optimizer took %s seconds (%s seconds for problem "
            "solving) ...", self.optimizer_time, self.solver_time)

    def reset(self):
        """Reset the optimization and evaluation outputs."""

        # Get the segmentation from the datahub
        segmentation = Datahub().segmentation

        # Check if a weighted-sum or Pareto problem is solved
        if self.problem.name in ('pareto', 'weighted-sum'):

            # Reset the problem tracker
            self.problem.tracker = {key: [] for key in self.problem.tracker}

        # Else, check if lexicographic optimization has been selected
        elif self.problem.name == 'lexicographic':

            # Loop over the subproblems
            for subproblem in self.problem.subproblems.values():

                # Reset the subproblem tracker
                subproblem.tracker = {key: [] for key in subproblem.tracker}

        # Loop over the machine learning model-based components
        for component in (
                get_machine_learning_constraints(segmentation)
                + get_machine_learning_objectives(segmentation)):

            # Get the feature calculator of the component
            feature_calculator = (
                component.data_model_handler.feature_calculator)

            # Reset the feature history
            feature_calculator.feature_history = empty(
                shape=(1, len(feature_calculator.feature_map)))

    def solve_lexicography(self):
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
        for rank, subproblem in self.problem.subproblems.items():

            # Log a message about the lexicographic rank
            get_logger().info("Considering lexicography at rank %s ...", rank)

            # Configure the solver
            self.solver.configure(subproblem)

            # Solve the optimization problem at the current rank
            fluence, solver_info = self.solver.run(fluence)

            # Check if the current rank does not equal the final rank
            if rank != ranks[-1]:

                # Get the next subproblem
                next_problem = self.problem.subproblems[
                    ranks[ranks.index(rank)+1]]

                # Overwrite the initial fluence
                next_problem.initial_fluence = fluence

                # Loop over the dynamic constraints
                for label in subproblem.objectives:

                    # Get the dynamic constraint threshold
                    threshold = subproblem.tracker[label][-1]

                    # Update the upper constraint bound
                    next_problem.constraints[label]['instance'].bounds[1] = (
                        threshold)

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

    def solve_pareto(self):
        """
        Solve the Pareto optimization problem.

        Returns
        -------
        ndarray
            Optimized fluence vector.

        str
            Description for the cause of termination.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the segmentation data and dose information
        segmentation, dose_information = hub.segmentation, hub.dose_information

        # Get the dose-influence matrix
        dose_matrix = dose_information['dose_influence_matrix']

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
                 for segment in (get_constraint_segments(segmentation)
                                 + get_objective_segments(segmentation))
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

    def solve_weighted(self):
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

        # Initialize the datahub
        hub = Datahub()

        # Get the dose information
        dose_information = hub.dose_information

        # Log a message about the 3D dose computation
        get_logger().info(
            "Computing dose cube from optimized fluence vector ...")

        # Get the CT and dose grid dimensions
        ct_dim, dose_dim = (
            hub.computed_tomography['cube_dimensions'],
            dose_information['cube_dimensions'])

        # Compute the optimized dose vector from the optimized fluence
        optimized_dose = (
            dose_information['dose_influence_matrix'] @ optimized_fluence)

        # Reshape the optimized dose vector to the dose cube
        optimized_dose = optimized_dose.reshape(dose_dim, order='F')

        # Get the zoom factors for all cube dimensions
        zooms = (pair[0]/pair[1] for pair in zip(ct_dim, dose_dim))

        # Interpolate the dose cube to the CT grid and multiply by the RBE
        optimized_dose = (
            zoom(optimized_dose, zooms, order=1)
            * hub.plan_configuration['RBE']
            * dose_information['number_of_fractions'])

        return optimized_dose

    def postprocess(self):
        """Postprocess the outcome model-based component results."""

        # Get the segmentation data
        segmentation = Datahub().segmentation

        # Check if the optimization problem has a tracker dictionary
        if hasattr(self.problem, 'tracker') and all(value != [] for value
           in self.problem.tracker.values()):

            # Loop over the radiobiological outcome model-based components
            for component in (
                    get_radiobiological_constraints(segmentation)
                    + get_radiobiological_objectives(segmentation)):

                # Get the final (N)TCP prediction value
                value = (
                    (-1)**('NTCP' not in component.name)
                    * self.problem.tracker[component.track_id][-1])

                # Log a message about the prediction value
                get_logger().info(
                    "%s for the optimized plan: %s %% ...",
                    component.name, round(100*value, 2))

            # Loop over the machine learning outcome model-based components
            for component in (
                    get_machine_learning_constraints(segmentation)
                    + get_machine_learning_objectives(segmentation)):

                # Process the feature history
                component.data_model_handler.process_feature_history()

                # Get the final (N)TCP prediction
                value = component.translate(
                    self.problem.tracker[component.track_id][-1])

                # Store the prediction value
                Datahub().model_outcomes[
                    component.data_model_handler.model_label] = value

                # Log a message about the prediction value
                get_logger().info(
                    "%s for the optimized plan: %s %% ...",
                    component.name, round(100*value, 2))

"""Fluence optimization."""

# Author: Tim Ortkamp

# %% External package import

from time import time

from functools import reduce
from math import inf
from numpy import (
    ravel_multi_index, setdiff1d, union1d, unravel_index, where, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.initializers import FluenceInitializer
import pyanno4rt.optimization._maps as maps
from pyanno4rt.tools import (
   apply, flatten, get_constraint_segments, get_machine_learning_constraints,
   get_machine_learning_objectives, get_radiobiological_constraints,
   get_radiobiological_objectives, get_objective_segments, reset_outputs)

# %% Class definition


class FluenceOptimizer():
    """
    Fluence optimization class.

    This class provides methods to optimize the fluence vector by solving the \
    inverse planning problem. It takes the configuration inputs, sets up the \
    optimization problem and the solver, and allows to compute both optimized \
    fluence vector and optimized dose cube (CT resolution).

    Parameters
    ----------
    method : {'lexicographic', 'pareto', 'weighted-sum'}
        Single- or multi-criteria optimization method.

    solver : {'ipyopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}
        Python package to be used for solving the optimization problem.

    algorithm : str
        Solution algorithm from the chosen solver.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        Initialization strategy for the fluence vector.

    initial_fluence_vector : None or list
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
    """

    def __init__(
            self,
            method,
            solver,
            algorithm,
            initial_strategy,
            initial_fluence_vector,
            lower_variable_bounds,
            upper_variable_bounds,
            maximum_iterations,
            tolerance):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing fluence optimizer ...")

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

        # Initialize the backprojection by the selected modality
        backprojection = maps.PROJECTIONS[hub.plan_configuration['modality']]()

        # Check if the solver ignores any constraints
        if len(constraints) > 0 and algorithm not in ('trust-constr', 'mumps'):

            # Log a message about the ignored constraints
            hub.logger.display_warning(
                f"The '{algorithm}' algorithm only allows for unconstrained "
                "optimization problems - constraints set will be ignored ...")

            # Reset the constraints to the default
            constraints = {}

        # Initialize the optimization problem by the selected method
        problem = maps.METHODS[method](backprojection, objectives, constraints)

        # Initialize the fluence initializer
        initializer = FluenceInitializer(
            initial_strategy, initial_fluence_vector)

        # Get the initial fluence vector
        initial_fluence = initializer.initialize_fluence()

        # Get the decision variable bounds
        variable_bounds = FluenceOptimizer.get_variable_bounds(
            lower_variable_bounds, upper_variable_bounds, len(initial_fluence))

        # Get the box constraint bounds
        constraint_bounds = FluenceOptimizer.get_constraint_bounds(
            method, problem.constraints)

        # Initialize the solver object by the selected solver
        solver_object = maps.SOLVERS[solver](
            number_of_variables=len(initial_fluence),
            number_of_constraints=len(constraints),
            problem_instance=problem,
            lower_variable_bounds=variable_bounds[0],
            upper_variable_bounds=variable_bounds[1],
            lower_constraint_bounds=constraint_bounds[0],
            upper_constraint_bounds=constraint_bounds[1],
            algorithm=algorithm,
            initial_fluence=initial_fluence,
            maximum_iterations=maximum_iterations,
            tolerance=tolerance)

        # Enter the optimization dictionary into the datahub
        hub.optimization |= {
            'problem': problem,
            'initializer': initializer,
            'initial_fluence': initial_fluence,
            'initial_strategy': initial_strategy,
            'solver_object': solver_object,
            'initial_time': time()-start_time}

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

            # Enter the overlap-free (prioritized) indices into the datahub
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
                            hub.logger.display_info(
                                f"Removing {label} "
                                f"'{dictionary[key]['instance'].name}' from "
                                f"fully enclosed segment '{reference}' ...")

                            # Delete the component from the dictionaries
                            del dictionary[key]
                            segmentation[reference][label] = None

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the overlap removal
        hub.logger.display_info("Removing segment overlaps ...")

        # Get the segmentation data
        segmentation = hub.segmentation

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
        hub.logger.display_info("Resizing segments from CT to dose grid ...")

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
        hub.logger.display_info(
            "Adjusting dose parameters for fractionation ...")

        # Get the number of fractions
        number_of_fractions = hub.dose_information['number_of_fractions']

        # Adjust all non-adjusted components with dose-related parameters
        apply(adjust_component, (
            component['instance'] for component in components.values()
            if not component['instance'].adjusted_parameters
            and 'dose' in component['instance'].parameter_category))

    @staticmethod
    def get_variable_bounds(lower, upper, length):
        """
        Get the lower and upper variable bounds.

        Parameters
        ----------
        lower : int, float, list or None
            Lower bound(s) on the decision variables.

        upper : int, float, list or None
            Upper bound(s) on the decision variables.

        length : int
            Length of the initial fluence vector.

        Returns
        -------
        list
            Lower bounds on the decision variables.

        list
            Upper bounds on the decision variables.
        """

        def get_bounds(value, limit):
            """Get the lower or upper bounds by the input value and limit."""

            # Check if the value is scalar
            if isinstance(value, (int, float)):

                # Generate a uniform list from the value
                return [value]*length

            # Check if the value is None
            if value is None:

                # Generate a uniform list from the limit
                return [limit]*length

            # Generate a cleansed list by replacing None with the limit
            return [limit if bound is None else bound for bound in value]

        return get_bounds(lower, -inf), get_bounds(upper, inf)

    @staticmethod
    def get_constraint_bounds(method, constraints):
        """
        Get the lower and upper constraint bounds.

        Parameters
        ----------
        method : {'lexicographic', 'pareto', 'weighted-sum'}
            Single- or multi-criteria optimization method.

        constraints : dict
            Dictionary with the internally configured problem constraints.

        Returns
        -------
        tuple
            Lower and upper bounds on the constraints.
        """

        # Check if no constraints have been passed
        if len(constraints) == 0:

            # Return the default empty bounds
            return [], []

        # Check if the method is 'lexicographic'
        if method == 'lexicographic':

            # Return the rank-ordered, transformed bounds
            return tuple({
                rank: [
                    constraint['instance'].bounds[index]
                    for constraint in rank_constraints.values()]
                for rank, rank_constraints in constraints.items()}
                for index in range(2))

        # Else, return the unranked, transformed bounds
        return tuple(zip(*(
            constraint['instance'].bounds
            for constraint in constraints.values())))

    def solve(self):
        """Solve the optimization problem."""

        # Initialize the datahub
        hub = Datahub()

        # Get the logger, segmentation data and optimization problem
        logger, segmentation, problem = (
            hub.logger, hub.segmentation, hub.optimization['problem'])

        # Reset the tracker and feature history if applicable
        reset_outputs()

        # Log a message about the problem solving
        logger.display_info("Solving optimization problem ...")

        # Start the solver runtime recording
        start_time = time()

        # Check if the fluence can not be loaded from a copycat
        if 'from_copycat' not in hub.optimization:

            # Solve the optimization problem
            (hub.optimization['optimized_fluence'],
             hub.optimization['solver_info']) = hub.optimization[
                 'solver_object'].run(hub.optimization['initial_fluence'])

        else:

            # Delete the copycat indicator
            del hub.optimization['from_copycat']

        # Get the runtime for problem solving
        solver_runtime = round(time()-start_time, 2)

        # Check if a solution has been found
        if hub.optimization['optimized_fluence'] is not None:

            # Compute the optimized dose from the fluence
            hub.optimization['optimized_dose'] = self.compute_dose_3d(
                hub.optimization['optimized_fluence'])

        else:

            # Log a message about the unsolved problem
            logger.display_info(
                "Fluence optimizer has not found a feasible solution for the "
                "treatment plan ...")

            # Set the optimized dose to None
            hub.optimization['optimized_dose'] = None

        # Check if the optimization problem has a tracker dictionary
        if hasattr(problem, 'tracker') and all(value != [] for value
           in problem.tracker.values()):

            # Loop over the radiobiological outcome model-based components
            for component in (
                    get_radiobiological_constraints(segmentation)
                    + get_radiobiological_objectives(segmentation)):

                # Get the final (N)TCP prediction value
                value = component.reverse(
                    problem.tracker[component.track_id][-1]/component.weight)

                # Log a message about the prediction value
                logger.display_info(
                    f"{component.name} for the optimized plan: "
                    f"{round(100*value, 2)} % ...")

            # Loop over the machine learning outcome model-based components
            for component in (
                    get_machine_learning_constraints(segmentation)
                    + get_machine_learning_objectives(segmentation)):

                # Process the feature history
                component.data_model_handler.process_feature_history()

                # Get the final (N)TCP prediction value
                value = component.reverse(
                    problem.tracker[component.track_id][-1]/component.weight)

                # Add the prediction value to the datahub
                hub.model_outcomes[
                    component.data_model_handler.model_label] = value

                # Log a message about the prediction value
                logger.display_info(
                    f"{component.name} for the optimized plan: "
                    f"{round(100*value, 2)} % ...")

        # Get the runtime for the fluence optimizer
        optimizer_runtime = round(
            time()-start_time+hub.optimization['initial_time'], 2)

        # Log a message about the optimization runtimes
        logger.display_info(
            f"Fluence optimizer took {optimizer_runtime} seconds "
            f"({solver_runtime} seconds for problem solving) ...")

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

        # Get the logger and the segmentation data
        logger, segmentation = hub.logger, hub.segmentation

        # Get the dose-influence matrix
        dose_matrix = hub.dose_information['dose_influence_matrix']

        # Get the CT and dose grid dimensions
        ct_dim, dose_dim = (
            hub.computed_tomography['cube_dimensions'],
            hub.dose_information['cube_dimensions'])

        # Check if a single fluence vector is passed
        if optimized_fluence.ndim == 1:

            # Compute the optimized dose vector from the optimized fluence
            optimized_dose = dose_matrix @ optimized_fluence

        else:

            # Log a message about the number of Pareto-optimal solutions
            logger.display_info(
                f"Pareto analysis resulted in {optimized_fluence.shape[0]} "
                "trade-off solutions ...")

            # Log a message about the mean dose difference selection criterion
            logger.display_info(
                "Selecting best solution with respect to the maximum mean "
                "dose difference between targets and organs at risk ...")

            # Initialize the current best score
            best_score = -inf

            # Initialize the current best fluence
            best_fluence = zeros(optimized_fluence[0].shape)

            # Get the indices of targets and OARs of interest
            target_indices, oar_indices = (reduce(
                union1d,
                (segmentation[segment]['resized_indices']
                 for segment in (
                    get_constraint_segments(segmentation)
                    + get_objective_segments(segmentation))
                    if segmentation[segment]['type'] == string),
                -1)
                for string in ('TARGET', 'OAR'))

            # Loop over the number of trade-off solutions
            for fluence in optimized_fluence:

                # Compute the dose from the solution
                dose = dose_matrix @ fluence

                # Calculate the mean dose difference between targets and OARs
                score = dose[target_indices].mean() - dose[oar_indices].mean()

                # Check if the score is better than the current best score
                if score > best_score:

                    # Update the best score and the best solution
                    best_score, best_fluence = score, fluence

            # Compute the optimized dose vector from the best solution
            optimized_dose = dose_matrix @ best_fluence

        # Log a message about the 3D dose computation
        logger.display_info(
            "Computing dose cube from optimized fluence vector ...")

        # Reshape the optimized dose vector to the dose cube
        optimized_dose = optimized_dose.reshape(dose_dim, order='F')

        # Get the zoom factors for all cube dimensions
        zooms = (pair[0]/pair[1] for pair in zip(ct_dim, dose_dim))

        # Interpolate the dose cube to the CT grid and multiply by the RBE
        optimized_dose = (
            zoom(optimized_dose, zooms, order=1)
            * hub.plan_configuration['RBE']
            * hub.dose_information['number_of_fractions'])

        return optimized_dose

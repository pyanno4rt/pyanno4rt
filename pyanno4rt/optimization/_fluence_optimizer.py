"""Fluence optimization."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

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
from pyanno4rt.optimization.projections import projection_map
from pyanno4rt.optimization.methods import method_map
from pyanno4rt.optimization.components import component_map
from pyanno4rt.optimization.solvers import solver_map
from pyanno4rt.tools import (
   apply, flatten, get_constraint_segments, get_machine_learning_constraints,
   get_machine_learning_objectives, get_radiobiology_constraints,
   get_radiobiology_objectives, get_objective_segments, sigmoid)

# %% Class definition


class FluenceOptimizer():
    """
    Fluence optimization class.

    This class provides methods to optimize the fluence vector by solving the \
    inverse planning problem. It preprocesses the configuration inputs, sets \
    up the optimization problem and the solver, and allows to compute both \
    optimized fluence vector and optimized 3D dose cube (CT resolution).

    Parameters
    ----------
    components : dict
        Optimization components for each segment of interest, i.e., \
        objective functions and constraints.

    method : {'lexicographic', 'pareto', 'weighted-sum'}
        Single- or multi-criteria optimization method.

    solver : {'proxmin', 'pymoo', 'scipy'}
        Python package to be used for solving the optimization problem.

    algorithm : str
        Solution algorithm from the chosen solver.

    initial_strategy : {'data-medoid', 'target-coverage', 'warm-start'}
        Initialization strategy for the fluence vector.

    initial_fluence_vector : None or list
        User-defined initial fluence vector for the optimization problem, \
        only used if initial_strategy='warm-start'.

    lower_variable_bounds : None, int, float, or list
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list
        Upper bound(s) on the decision variables.

    max_iter : int
        Maximum number of iterations taken for the solver to converge.

    tolerance : float
        Precision goal for the objective function value.
    """

    def __init__(
            self,
            components,
            method,
            solver,
            algorithm,
            initial_strategy,
            initial_fluence_vector,
            lower_variable_bounds,
            upper_variable_bounds,
            max_iter,
            tolerance):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing fluence optimizer ...")

        # Start the constructor runtime recording
        start_time = time()

        # Set the objective and constraint functions
        objectives, constraints = FluenceOptimizer.set_optimization_components(
            components)

        # Remove overlaps between segments according to their priority
        FluenceOptimizer.remove_overlap(objectives | constraints)

        # Resize the segments to the dose grid
        FluenceOptimizer.resize_segments_to_dose()

        # Adjust the dose-volume-related parameters for fractionation
        FluenceOptimizer.adjust_parameters_for_fractionation(
            objectives | constraints)

        # Initialize the backprojection by the selected modality
        backprojection = projection_map[hub.plan_configuration['modality']]()

        # Initialize the optimization problem by the selected method
        problem = method_map[method](backprojection, objectives, constraints)

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
        solver_object = solver_map[solver](
            number_of_variables=len(initial_fluence),
            number_of_constraints=len(constraints),
            problem_instance=problem,
            lower_variable_bounds=variable_bounds[0],
            upper_variable_bounds=variable_bounds[1],
            lower_constraint_bounds=constraint_bounds[0],
            upper_constraint_bounds=constraint_bounds[1],
            algorithm=algorithm,
            initial_fluence=initial_fluence,
            max_iter=max_iter,
            tolerance=tolerance)

        # Enter the optimization dictionary into the datahub
        hub.optimization = {
            'problem': problem,
            'initializer': initializer,
            'initial_fluence': initial_fluence,
            'initial_strategy': initial_strategy,
            'solver_object': solver_object,
            'initial_time': time()-start_time}

    @staticmethod
    def set_optimization_components(components):
        """
        Set the components of the optimization problem.

        Parameters
        ----------
        components : dict
            Optimization components for each segment of interest, i.e., \
            objectives and constraints, in the raw user format.

        Returns
        -------
        dict
            Dictionary with the internally configured objectives.

        dict
            Dictionary with the internally configured constraints.
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the logger and the segmentation data
        logger, segmentation = hub.logger, hub.segmentation

        # Loop over the segments
        for segment in segmentation:

            # Reset the segment objective and constraint key
            segmentation[segment]['objective'] = None
            segmentation[segment]['constraint'] = None

        # Log a message about the components setting
        logger.display_info("Setting the optimization components ...")

        # Initialize the objective and constraint dictionaries
        objectives, constraints = {}, {}

        # Set the base dictionaries for the component types
        bases = {'objective': objectives, 'constraint': constraints}

        def set_component(component, segment, category, base_dict):
            """Set the component by its segment and type assignment."""

            # Get the instance from the component map
            instance = component_map[component['class']](
                **component['parameters'])

            # Log a message about setting the instance
            logger.display_info(
                f"Setting {category} '{instance.name}' for "
                f"{[segment]+instance.link} ...")

            # Get the instance key for the base dictionary
            instance_key = '-'.join(filter(
                None, (f"{[segment]+instance.link}", instance.name,
                       instance.identifier)))

            # Check if the instance is already included in the base dictionary
            if instance_key not in base_dict:

                # Add the instance to the base dictionary
                base_dict[instance_key] = {
                    'segments': [segment]+instance.link,
                    'instance': instance}

                # Check if no instance has been set yet
                if not segmentation[segment][category]:

                    # Add the instance to the segment
                    segmentation[segment][category] = instance

                else:

                    # Check if the component is a list
                    if isinstance(segmentation[segment][category], list):

                        # Append the instance
                        segmentation[segment][category].append(instance)

                    else:

                        # Make a list and add the instance
                        segmentation[segment][category] = [
                            segmentation[segment][category], instance]

        # Loop over the segments in the components dictionary
        for segment in components:

            # Check if the segment holds a list of components
            if isinstance(components[segment], list):

                # Loop over the component list
                for element in components[segment]:

                    # Get the category and component
                    category, component = element.values()

                    # Get the base dictionary
                    base_dict = bases[category]

                    # Set the component
                    set_component(component, segment, category, base_dict)

            else:

                # Get the category and component
                category, component = components[segment].values()

                # Get the base dictionary
                base_dict = bases[category]

                # Set the component
                set_component(component, segment, category, base_dict)

        # Add the outcome models for all machine learning components
        apply(lambda component: component.add_model(),
              get_machine_learning_constraints(segmentation)
              + get_machine_learning_objectives(segmentation))

        return objectives, constraints

    @staticmethod
    def remove_overlap(components):
        """
        Remove overlaps between segments.

        Parameters
        ----------
        voi : tuple
            Tuple with the labels for the volumes of interest.
        """

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the overlap removal
        hub.logger.display_info("Removing segment overlaps ...")

        # Get the segmentation data
        segmentation = hub.segmentation

        def remove_segment_overlap(reference):
            """Remove the overlap from a reference segment."""

            # Get the superior indices from all VOIs
            superior_indices = [
                segmentation[segment]['raw_indices']
                for segment in set(flatten(
                        [component['segments']
                         for component in components.values()]))
                if (segmentation[segment]['parameters']['priority']
                    < segmentation[reference]['parameters']['priority'])]

            # Enter the overlap-free (prioritized) indices into the datahub
            segmentation[reference]['prioritized_indices'] = setdiff1d(
                segmentation[reference]['raw_indices'],
                reduce(union1d, superior_indices, -1))

        # Remove the overlaps from all segments
        apply(remove_segment_overlap, (*segmentation,))

    @staticmethod
    def resize_segments_to_dose():
        """Resize the segments from CT to dose grid."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the segment resizing
        hub.logger.display_info("Resizing segments from CT to dose grid ...")

        # Get the segmentation data
        segmentation = hub.segmentation

        # Get the CT and dose cube dimensions
        ct_dim, dose_dim = (hub.computed_tomography['cube_dimensions'],
                            hub.dose_information['cube_dimensions'])

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

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the parameter adjustment
        hub.logger.display_info(
            "Adjusting dose parameters for fractionation ...")

        # Get the number of fractions
        number_of_fractions = hub.dose_information['number_of_fractions']

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

        # Adjust all non-adjusted components with dose-related parameters
        apply(adjust_component, (
            component['instance'] for component in components.values()
            if not component['instance'].adjusted_parameters
            and 'dose' in component['instance'].parameter_category))

    @staticmethod
    def get_variable_bounds(lower, upper, length):
        """
        Get the lower and upper variable bounds in a compatible form.

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
            Transformed lower bounds on the decision variables.

        list
            Transformed upper bounds on the decision variables.
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
        Get the lower and upper constraint bounds in a compatible form.

        Parameters
        ----------
        method : {'lexicographic', 'pareto', 'weighted-sum'}
            Single- or multi-criteria optimization method.

        constraints : dict
            Dictionary with the internally configured problem constraints.

        Returns
        -------
        tuple
            Transformed lower and upper bounds on the constraints.
        """

        def transform(bounds, limit):
            """Get the lower or upper bounds by the input value and limit."""

            # Generate a cleansed list by replacing None with the limit
            return [limit if bound is None else bound for bound in bounds]

        # Check if no constraints have been passed
        if len(constraints) == 0:

            # Return the default empty bounds
            return [], []

        # Check if the method is 'lexicographic'
        if method == 'lexicographic':

            # Return the rank-ordered, transformed bounds
            return tuple(
                {rank: transform(
                    [constraint['instance'].bounds[index]
                     for constraint in rank_constraints.values()], limit)
                    for rank, rank_constraints in constraints.items()}
                for index, limit in enumerate((-inf, inf)))

        else:

            # Return the unranked, transformed bounds
            return tuple(
                transform([constraint['instance'].bounds[index]
                           for constraint in constraints.values()], limit)
                for index, limit in enumerate((-inf, inf)))

    def solve(self):
        """Solve the optimization problem."""

        # Initialize the datahub
        hub = Datahub()

        # Get the logger, segmentation data and optimization problem
        logger, segmentation, problem = (
            hub.logger, hub.segmentation, hub.optimization['problem'])

        # Log a message about the problem solving
        logger.display_info("Solving optimization problem ...")

        # Start the solver runtime recording
        start_time = time()

        # Solve the optimization problem
        (hub.optimization['optimized_fluence'],
         hub.optimization['solver_info']) = hub.optimization[
             'solver_object'].run(hub.optimization['initial_fluence'])

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
        if hasattr(problem, 'tracker'):

            # Loop over the radiobiological outcome model-based components
            for name, value in (
                (component.name, track[-1]/component.weight)
                for label, track in problem.tracker.items()
                for component in (
                        get_radiobiology_objectives(segmentation)
                        + get_radiobiology_constraints(segmentation))
                    if component.name in label):

                # Log a message about the (N)TCP prediction
                logger.display_info(
                    f"{name} for the optimized plan: "
                    f"{(-1)**('NTCP' not in name)*round(100*value, 2)} % ...")

            # Loop over the machine learning outcome model-based components
            for name, value, component in (
                (component.name, track[-1]/component.weight, component)
                for label, track in problem.tracker.items()
                for component in (
                        get_machine_learning_constraints(segmentation)
                        + get_machine_learning_objectives(segmentation))
                    if component.name in label):

                # Process the feature history
                component.data_model_handler.process_feature_history()

                # Get the boolean value for sigmoidal models
                is_sigmoidal = any(string in name for string in (
                    'Logistic Regression', 'Neural Network',
                    'Support Vector Machine'))

                # Get the default sigmoid coefficients
                multiplier, summand = 1, 0

                # Check if a support vector machine is present
                if 'Support Vector Machine' in name:

                    # Get the support vector machine prediction model
                    svm = component.model.prediction_model

                    # Get the Platt scaling coefficients
                    multiplier, summand = -svm.probA_[0], svm.probB_[0]

                # Get the value sign
                sign = (-1)**('NTCP' not in name)

                # Get the (N)TCP prediction value
                value = round(100*(sign*value)**(1-is_sigmoidal)
                              * sigmoid(sign*value, multiplier, summand)
                              ** is_sigmoidal, 2)

                # Log a message about the (N)TCP prediction
                logger.display_info(
                    f"{name} for the optimized plan: {value} % ...")

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
        Compute the 3D dose cube from the optimized fluence vector.

        Parameters
        ----------
        optimized_fluence : ndarray
            Optimized fluence vector(s).

        Returns
        -------
        ndarray
            Optimized 3D dose cube (CT resolution).
        """

        # Initialize the datahub
        hub = Datahub()

        # Get the logger and the segmentation data
        logger, segmentation = hub.logger, hub.segmentation

        # Get the dose-influence matrix
        dose_matrix = hub.dose_information['dose_influence_matrix']

        # Get the CT and dose grid dimensions
        ct_dim, dose_dim = (hub.computed_tomography['cube_dimensions'],
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

            # Get the indices of targets and OARs of interest
            target_indices, oar_indices = (reduce(
                union1d, [segmentation[segment]['resized_indices']
                          for segment in (
                                  get_constraint_segments(segmentation)
                                  + get_objective_segments(segmentation))
                          if segmentation[segment]['type'] == string], -1)
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
            "Computing 3D dose cube from optimized fluence vector ...")

        # Reshape the optimized dose vector to the 3D dose cube
        optimized_dose = optimized_dose.reshape(dose_dim, order='F')

        # Get the zoom factors for all cube dimensions
        zooms = (pair[0]/pair[1] for pair in zip(ct_dim, dose_dim))

        # Interpolate the dose cube to the CT grid and multiply by the RBE
        optimized_dose = (zoom(optimized_dose, zooms, order=1)
                          * hub.plan_configuration['RBE'])

        return optimized_dose

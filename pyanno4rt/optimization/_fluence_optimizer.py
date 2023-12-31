"""Fluence optimization."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from functools import reduce
from math import inf
from time import time
from numpy import (ravel_multi_index, setdiff1d, union1d, unravel_index,
                   where, zeros)
from scipy.ndimage import zoom

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization import FluenceInitializer
from pyanno4rt.optimization.projections import projections_map
from pyanno4rt.optimization.components.methods import methods_map
from pyanno4rt.optimization.components.objectives import objectives_map
from pyanno4rt.optimization.solvers import solvers_map
from pyanno4rt.tools import get_model_objectives, sigmoid

# %% Class definition


class FluenceOptimizer():
    """
    Fluence optimization class.

    This class implements methods for optimizing the fluence vector, based on \
    CT, segmentation, plan configuration, and dose information. It handles \
    both photon and proton treatment plan optimization with different \
    backprojection types, includes dose-volume and outcome model-based \
    objective functions and constraints, allows for single-criteria \
    (scalarization, lexicographization) and multi-criteria optimization \
    (Pareto analysis) with different fluence initialization strategies, \
    and integrates multiple local and global solvers from IPOPT, Proxmin, \
    Pymoo, PyPOP7 and SciPy.

    Parameters
    ----------
    components : dict
        Optimization components (objectives and constraints), passed as a \
        dictionary which maps segmented structures to tuples of length 2 \
        (holding the component type and instance, or a tuple of instances, if \
        multiple components should be assigned to a single structure).

    projection : {'constantRBE', 'dose'}
        Type of projection between fluence and dose:

        - 'constantRBE' : linear dose projection with a constant RBE factor \
            of 1.1;
        - 'dose' : linear dose projection with a neutral RBE factor of 1.0.

    method : {'lexicographic', 'pareto', 'weighted-sum'}
        Single- or multi-criteria optimization method:

        - 'lexicographic' : optimize all objectives sequentially based on a \
            preference order (single-criteria);
        - 'pareto' : optimize all objectives with trade-offs to obtain a \
            set of pareto-optimal points (multi-criteria);
        - 'weighted-sum' : optimize all objectives based on a weighted sum \
            (single-criteria).

    solver : {'ipopt', 'proxmin', 'pymoo', 'pypop7', 'scipy'}
        Python package for solving the optimization problem:

        - 'ipopt' : interior-point algorithms provided by COIN-OR;
        - 'proxmin' : proximal algorithms provided by Proxmin;
        - 'pymoo' : multi-objective algorithms provided by Pymoo;
        - 'pypop7' : population-based algorithms provided by PyPOP7;
        - 'scipy' : local algorithms provided by SciPy.

    algorithm : string
        Solution algorithm from the chosen solver:

        - ``solver`` = 'ipopt' : {'ma27', 'ma57', 'ma77', 'ma86'};
        - ``solver`` = 'proxmin' : {'admm', 'pgm', 'sdmm'};
        - ``solver`` = 'pymoo' : {'NSGA3'};
        - ``solver`` = 'pypop7' : {'MMES', 'LMCMA', 'RMES', 'BES', 'GS'};
        - ``solver`` = 'scipy' : {'L-BFGS-B', 'TNC', 'trust-constr'}.

    initial_strategy : {'target-coverage', 'warm-start'}
        Initialization strategy for the fluence vector:

        - ``target-coverage`` : initialize the fluence vector with respect to \
            tumor coverage;
        - ``warm-start`` : initialize the fluence vector with respect to a \
            reference optimal point.

    initial_fluence_vector : ndarray
        Initial fluence vector for warm-starting the plan (either by passing \
        an array of values or the treatment plan instance holding the values).

    lower_variable_bounds : int, float or tuple
        Lower bounds on the decision variables.

    upper_variable_bounds : int, float or tuple
        Upper bounds on the decision variables.

    max_iter : int
        Maximum number of iterations taken for the solver to converge.

    max_cpu_time : float
        Maximum CPU time taken for the solver to converge.

    Attributes
    ----------
    optimization_problem : object of class `LexicographicOptimization`, \
        `ParetoOptimization` or `WeightedSumOptimization`
        Instance of the optimization problem depending on the parameters \
        ``method`` and ``solver``. This includes methods to compute \
        objective, gradient and constraint functions, and initializes the \
        tracker dictionary to store computed values for each objective \
        function.

    fluence_initializer : object of class `FluenceInitializer`
        Instance of the fluence initializer, which includes different \
        strategies to find the initial fluence vector.

    initial_fluence : ndarray
        Initialization of the fluence vector.

    solver_object : object of class `IpoptSolver`, `ProxminSolver`, \
        `PymooSolver`, `Pypop7Solver` or `SciPySolver`
        Instance of the solver with its configuration.

    optimized_fluence : ndarray
        Optimized fluence vector as returned by the optimizer.

    optimized_dose : ndarray
        Optimized dose cube as returned by the optimizer.

    optimized_info : dict
        Dictionary with information on the state of the optimizer.

    initial_time : float
        Runtime for the initialization of the fluence optimizer (in seconds).
    """

    # Class constructor
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
            max_cpu_time):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info("Initializing fluence optimizer ...")

        # Start the constructor runtime recording
        start_time = time()

        # Preprocess the segments
        FluenceOptimizer.set_overlap_priorities(segments=(*components,))
        FluenceOptimizer.resize_segments_to_dose()

        # Set the optimization components based on the 'components' argument
        objectives, constraints = (
            FluenceOptimizer.set_optimization_components(
                components=components))

        # Preprocess the objectives
        FluenceOptimizer.adjust_parameters_for_fractionation(objectives)

        # Initialize the backprojection based on the 'projection' argument
        backprojection = projections_map[hub.plan_configuration['modality']]()

        # Initialize the optimization problem based on the 'method' argument
        hub.optimization['problem'] = methods_map[method](
            backprojection, objectives, constraints)

        # Initialize the fluence initializer
        hub.optimization['initializer'] = FluenceInitializer(
            initial_strategy, initial_fluence_vector)

        # Get the initial fluence vector
        hub.optimization['initial_fluence'] = hub.optimization[
            'initializer'].run_strategy()

        # Check if the lower variable bounds are represented by a single value
        if isinstance(lower_variable_bounds, (int, float, type(None))):

            # Check if the bounds are set to None
            if lower_variable_bounds is None:

                # Generate a negative infinity vector for the lower bounds
                lower_variable_bounds = [-inf]*len(
                    hub.optimization['initial_fluence'])

            else:

                # Generate a uniform vector for the lower bounds
                lower_variable_bounds = [lower_variable_bounds]*len(
                    hub.optimization['initial_fluence'])

        else:

            # Replace all values of None in the lower variable bounds list
            lower_variable_bounds = [-inf if bound is None else bound
                                     for bound in lower_variable_bounds]

        # Check if the upper variable bounds are represented by a single value
        if isinstance(upper_variable_bounds, (int, float, type(None))):

            # Check if the bounds are set to None
            if upper_variable_bounds is None:

                # Generate an infinity vector for the upper bounds
                upper_variable_bounds = [inf]*len(
                    hub.optimization['initial_fluence'])

            else:

                # Generate a uniform vector for the upper bounds
                upper_variable_bounds = [upper_variable_bounds]*len(
                    hub.optimization['initial_fluence'])

        else:

            # Replace all values of None in the upper variable bounds list
            upper_variable_bounds = [inf if bound is None else bound
                                     for bound in upper_variable_bounds]

        # Check if the IPOPT solver is selected but not available
        if solver == 'ipopt' and solvers_map['ipopt'] is None:

            # Overwrite the missing IPOPT solver with the default solver
            solvers_map['ipopt'] = solvers_map['scipy']

            # Overwrite the algorithm to the default SciPy algorithm
            algorithm = 'L-BFGS-B'

            # Log a message about the fallback to SciPy
            hub.logger.display_warning("IPOPT solver is not available, "
                                       "falling back to SciPy solver with "
                                       "L-BFGS-B ...")

        # Initialize the solver object based on the 'solver' argument
        hub.optimization['solver_object'] = solvers_map[solver](
            number_of_variables=len(hub.optimization['initial_fluence']),
            number_of_constraints=len(constraints),
            problem_instance=hub.optimization['problem'],
            lower_variable_bounds=lower_variable_bounds,
            upper_variable_bounds=upper_variable_bounds,
            lower_constraint_bounds=[],
            upper_constraint_bounds=[],
            algorithm=algorithm,
            max_iter=max_iter,
            max_cpu_time=max_cpu_time)

        # End the constructor runtime recording
        end_time = time()

        # Compute the initialization time for the class
        hub.optimization['initial_time'] = end_time - start_time

    @staticmethod
    def set_optimization_components(components):
        """
        Set the components of the optimization problem.

        Parameters
        ----------
        components : dict
            Optimization components (objectives and constraints), passed as a \
            dictionary which maps segmented structures to iterables of length \
            2 (holding the component type and instance, or a list of \
            instances, if multiple components should be assigned to a single \
            structure).

        Returns
        -------
        tuple
            Tuple with pairs of segmented structures and their associated \
            objectives or constraints.
        """
        # Initialize the datahub
        hub = Datahub()

        # Get the logger and the segmentation data from the datahub
        logger = hub.logger
        segmentation = hub.segmentation

        # Log a message about the components setting
        logger.display_info("Setting the optimization components ...")

        # Initialize the iterables for the objectives and constraints
        objectives = []
        constraints = []

        def set_objective(segment, objective):
            """Set the objective for a segment."""
            # Enter the objective into the datahub
            segmentation[segment]['objective'] = objective

            # Check if the objective is not an iterable
            if not isinstance(objective, tuple):

                # Add the segment/link and the instance to the objectives
                objectives.append(([segment]+objective.link, objective))

            else:

                # Iterate over all segment objectives
                for subobjective in objective:

                    # Add the segment/link and the instance to the objectives
                    objectives.append(([segment]+subobjective.link,
                                       subobjective))

        def set_constraint(segment, constraint):
            """Set the constraint for a segment."""
            # Enter the constraint into the datahub
            segmentation[segment]['constraint'] = constraint

            # Check if the constraint is not an iterable
            if not isinstance(constraint, tuple):

                # Add the segment/link and the instance to the constraints
                constraints.append(([segment]+constraint.link, constraint))

            else:

                # Iterate over all segment constraints
                for subconstraint in constraint:

                    # Add the segment/link and the instance to the constraints
                    constraints.append(([segment]+subconstraint.link,
                                        subconstraint))

        # Map the component categories to the set functions
        categories = {'objective': set_objective,
                      'constraint': set_constraint}

        # Iterate over all items in the 'components' dictionary
        for segment in components:

            # Get the component category and element
            category, element = components[segment]

            # Check if the element is a dictionary
            if isinstance(element, dict):

                # Convert the element to a single instance
                instance = objectives_map[element['class']](
                    **element['parameters'])

            else:

                # Convert the element to a tuple of instances
                instance = tuple(objectives_map[sub['class']](
                    **sub['parameters']) for sub in element)

            # Run the corresponding set function
            categories[category](segment, instance)

        # Iterate over all objectives
        for objective in objectives:

            # Log a message about the objective set
            logger.display_info("Setting {} objective '{}' for {} ..."
                                .format(objective[1].embedding,
                                        objective[1].name,
                                        objective[0][0]
                                        if len(objective[0]) >= 1
                                        else objective[0]))

            # Check if the objective depends on data
            if objective[1].DEPENDS_ON_MODEL:

                # Add the outcome model to the objective
                objective[1].add_model()

        # Iterate over all constraints
        for constraint in constraints:

            # Log a message about the constraint set
            logger.display_info("Setting constraint '{}' for {} ..."
                                .format(constraint[1].name, constraint[0][0]
                                        if len(constraint[0]) >= 1
                                        else constraint[0]))

        return tuple(objectives), tuple(constraints)

    @staticmethod
    def adjust_parameters_for_fractionation(objectives):
        """
        Adjust the dose parameters according to the number of fractions.

        Parameters
        ----------
        objectives : tuple
            Tuple with pairs of segmented structures and their associated \
            objectives and constraints.
        """
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the parameter adjustment
        hub.logger.display_info("Adjusting dose parameters for fractionation "
                                "...")

        def adjust_single_objective(objective):
            """Adjust the dose parameters for a single objective."""
            # Get the indices of the dose-related parameters
            indices = tuple(item[0]
                            for item in enumerate(objective.parameter_category)
                            if item[1] == 'dose')

            # Get the parameter value of the objective
            parameters = objective.get_parameter_value()

            # Iterate over the indices
            for index in indices:

                # Adjust the indexed parameters with the number of fractions
                parameters[index] /= hub.dose_information[
                    'number_of_fractions']

            # Set the objective parameters to the adjusted values
            objective.set_parameter_value(parameters)

            # Set the 'adjusted_parameters' attribute to True
            objective.adjusted_parameters = True

        # Loop over all non-adjusted, dose-related parameters
        for objective in (objective[1] for objective in objectives
                          if not objective[1].adjusted_parameters
                          and 'dose' in objective[1].parameter_category):

            # Adjust the objective parameters
            adjust_single_objective(objective)

    @staticmethod
    def set_overlap_priorities(segments):
        """Set the overlap priorities."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the overlap prioritization
        hub.logger.display_info("Setting overlap priorities ...")

        # Get the segmentation data from the datahub
        segmentation = hub.segmentation

        def remove_single_segment_overlap(segment_A):
            """Remove the overlap for a single segment."""
            # Get the overlapping indices
            overlapping_indices = [
                segmentation[segment_B]['raw_indices']
                for segment_B in (*segmentation,)
                if (segmentation[segment_B]['parameters']['priority']
                    < segmentation[segment_A]['parameters']['priority'])
                and segment_B in segments]

            # Compute the union over the index list
            removable_indices = reduce(union1d, overlapping_indices, -1)

            # Remove the common indices to get the indices after prioritization
            segmentation[segment_A]['prioritized_indices'] = setdiff1d(
                segmentation[segment_A]['raw_indices'], removable_indices)

        # Loop over all segments
        for segment in (*segmentation,):

            # Remove the overlaps by priority
            remove_single_segment_overlap(segment)

    @staticmethod
    def resize_segments_to_dose():
        """Resize the segments from CT to dose grid."""
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the resizing
        hub.logger.display_info("Resizing segments from CT to dose grid ...")

        # Get the CT, segmentation and dose information data from the datahub
        computed_tomography = hub.computed_tomography
        segmentation = hub.segmentation
        dose_information = hub.dose_information

        def resize_single_segment(segment):
            """Resize a single segment to the dose grid."""
            # Initialize the segment mask
            mask = zeros(computed_tomography['cube_dimensions'])

            # Insert ones at the indices of the segment
            mask[unravel_index(
                segmentation[segment]['prioritized_indices'],
                computed_tomography['cube_dimensions'], order='F')] = 1

            # Get the resized indices of the segment
            resized_indices = where(
                zoom(
                    mask,
                    (dose_information['cube_dimensions'][j]
                     / computed_tomography['cube_dimensions'][j]
                     for j, _ in enumerate(
                             dose_information['cube_dimensions'])),
                    order=0))

            # Enter the resized indices into the datahub
            segmentation[segment]['resized_indices'] = ravel_multi_index(
                resized_indices,
                dose_information['cube_dimensions'], order='F')

        # Loop over all segments
        for segment in (*segmentation,):

            # Resize the segment
            resize_single_segment(segment)

    def solve(self):
        """Solve the optimization problem."""

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the problem solving
        hub.logger.display_info("Solving optimization problem ...")

        # Start the solver runtime recording
        start_time = time()

        # Solve the optimization problem and retrieve the optimized fluence
        (hub.optimization['optimized_fluence'],
         hub.optimization['solver_info']) = hub.optimization[
             'solver_object'].start(hub.optimization['initial_fluence'])

        # Check the time of solver running
        check_time = time()

        # Log a message about the successful optimization
        hub.logger.display_info("Optimization reached acceptable level and "
                                "returned optimal results ...")

        # Compute the optimized dose from the fluence
        hub.optimization['optimized_dose'] = self.compute_dose_3d()

        # Get the final (N)TCP values if outcome prediction models are present
        model_objectives = get_model_objectives(hub.segmentation)

        # Check if the number of data-dependent objectives is nonzero
        if (len(model_objectives) > 0
                and hasattr(hub.optimization['problem'], 'tracker')):

            # Loop over the data-dependent objectives
            for objective in model_objectives:

                # Process the feature and gradient histories
                objective.data_model_handler.process_histories(
                    objective.model.model_label)

            # Get the data-dependent model results from the tracker
            data_model_results = tuple(
                (objective.name, hub.optimization[
                    'problem'].tracker[track][-1]/objective.weight)
                for _, track in enumerate(
                        (*hub.optimization['problem'].tracker,))
                for objective in model_objectives if objective.name in track)

            # Iterate over the data model results
            for i, _ in enumerate(data_model_results):

                # Get the component name
                component_name = data_model_results[i][0]

                # Check if a non-sigmoidal model is used
                if any(model_name in component_name
                       for model_name in ('Decision Tree',
                                          'Extreme Gradient Boosting',
                                          'K-Nearest Neighbor',
                                          'Naive Bayes',
                                          'Random Forest')):

                    # Get the rounded final outcome prediction
                    rounded_result = round(100*(data_model_results[i][1]), 2)

                    # Log a message about their final (N)TCP values
                    hub.logger.display_info(f"{component_name} for the "
                                            f"optimized plan: {rounded_result}"
                                            " % ...")

                # Check if logistic regression or neural networks are used
                if any(model_name in component_name
                       for model_name in ('Logistic Regression',
                                          'Neural Network')):

                    # Check if the model predicts NTCP
                    if 'NTCP' in model_objectives[i].name:

                        # Get the rounded final outcome prediction
                        rounded_result = round(100*sigmoid(
                            data_model_results[i][1], 1, 0), 2)

                    else:

                        # Get the rounded final outcome prediction
                        rounded_result = round(100*(1-sigmoid(
                            data_model_results[i][1], 1, 0)), 2)

                    # Log a message about their final (N)TCP values
                    hub.logger.display_info(f"{component_name} for the "
                                            f"optimized plan: {rounded_result}"
                                            " % ...")

                # Check if support vector machines are used
                elif 'Support Vector Machine' in component_name:

                    # Get the SVM prediction model
                    svm = model_objectives[i].model.prediction_model

                    # Get the rounded final outcome prediction
                    rounded_result = round(100*sigmoid(
                        data_model_results[i][1], -svm.probA_[0],
                        svm.probB_[0]), 2)

                    # Log a message about its final (N)TCP value
                    hub.logger.display_info(f"{component_name} for the "
                                            f"optimized plan: {rounded_result}"
                                            " % ...")

        # End the solver runtime recording
        end_time = time()

        # Get the total fluence optimization runtime
        total_runtime = round((end_time-start_time)
                              + hub.optimization['initial_time'], 2)

        # Get the problem solving runtime
        solver_runtime = round((check_time-start_time), 2)

        # Log a message about the overall optimization runtime
        hub.logger.display_info(f"Fluence optimization took {total_runtime} "
                                f"seconds ({solver_runtime} seconds for "
                                "problem solving) ...")

    def compute_dose_3d(self):
        """
        Compute the 3D dose distribution from the optimized fluence vector.

        Returns
        -------
        ndarray
            Optimized dose cube on the CT grid.
        """
        # Initialize the datahub
        hub = Datahub()

        # Log a message about the 3D dose computation
        hub.logger.display_info("Computing 3D dose distribution from "
                                "optimized fluence vector ...")

        # Get the CT and dose information data form the datahub
        computed_tomography = hub.computed_tomography
        segmentation = hub.segmentation
        dose_information = hub.dose_information

        try:

            # Compute the optimized dose vector from the optimized fluence
            optimized_dose = (dose_information['dose_influence_matrix']
                              @ hub.optimization['optimized_fluence'])

        except ValueError:

            # Initialize the selectivity values
            selectivity_values = []

            # Get the index lists for targets and OARs
            target_indices = [segmentation[segment]['resized_indices']
                              for segment in segmentation
                              if segmentation[segment]['type'] == 'TARGET']
            oar_indices = [segmentation[segment]['resized_indices']
                           for segment in segmentation
                           if segmentation[segment]['type'] == 'OAR']

            # Get the union sets over the index lists
            union_target_indices = reduce(union1d, target_indices, -1)
            union_oar_indices = reduce(union1d, oar_indices, -1)

            # Loop over the number of candidate optimized fluence vectors
            for j in range(hub.optimization['optimized_fluence'].shape[0]):

                # Calculate the dose from the candidate vector
                dose = (dose_information['dose_influence_matrix']
                        @ hub.optimization['optimized_fluence'][j])

                # Calculate the mean doses to the targets and OARs
                mean_target_dose = dose[union_target_indices].mean()
                mean_oar_dose = dose[union_oar_indices].mean()

                # Add the mean dose difference to the selectivity values
                selectivity_values.append(mean_target_dose - mean_oar_dose)

            # Determine the selected index from the maximum difference
            index = selectivity_values.index(max(selectivity_values))

            # Calculate the optimized dose vector from the selected fluence
            optimized_dose = (dose_information['dose_influence_matrix']
                              @ hub.optimization['optimized_fluence'][index])

        # Reshape the optimized dose vector to the 3D dose cube
        optimized_dose = optimized_dose.reshape(
            dose_information['cube_dimensions'], order='F')

        # Get the zoom factors for all cube dimensions
        zooms = (computed_tomography['cube_dimensions'][j]
                 / dose_information['cube_dimensions'][j]
                 for j, _ in enumerate(computed_tomography['cube_dimensions']))

        # Interpolate the dose cube to fit the CT grid and weight with the RBE
        optimized_dose = (zoom(optimized_dose, zooms, order=1)
                          * hub.plan_configuration['RBE'])

        return optimized_dose

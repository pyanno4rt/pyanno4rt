"""Lexicographic problem."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array, concatenate, vstack, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    apply, flatten, get_machine_learning_constraints,
    get_machine_learning_objectives)

# %% Class definition


class LexicographicOptimization():
    """
    Lexicographic optimization problem class.

    This class provides methods to perform lexicographic optimization. It \
    features a component tracker and implements the respective objective, \
    gradient, constraint and constraint Jacobian functions.

    Parameters
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        The object representing the type of backprojection.

    objectives : dict
        Dictionary with the internally configured objectives.

    constraints : dict
        Dictionary with the internally configured constraints.

    Attributes
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        See 'Parameters'.

    objectives : dict
        Dictionary with the rank-ordered objectives.

    constraints : dict
        Dictionary with the rank-ordered constraints.

    tracker : dict
        Dictionary with the iteration-wise plan component values.
    """

    def __init__(
            self,
            backprojection,
            objectives,
            constraints):

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info(
            "Initializing lexicographic optimization method ...")

        # Get the backprojection from the arguments
        self.backprojection = backprojection

        # Get the rank-ordered objectives
        self.objectives = {
            rank: {
                label: objective for label, objective in objectives.items()
                if objective['instance'].rank == rank}
            for rank in sorted(set(
                objective['instance'].rank
                for objective in objectives.values()))}

        # Initialize the rank-ordered constraints by the "static" constraints
        self.constraints = {
            rank: {
                label: constraint for label, constraint in constraints.items()
                if constraint['instance'].rank == rank}
            for rank in self.objectives}

        # Loop over the lexicographic layers
        for rank in self.constraints:

            # Update the constraints with the "dynamic" constraints
            self.constraints[rank] |= {
                label: constraint for dictionary in (
                    self.objectives[label] for label in tuple(
                        self.constraints)[:list(self.constraints).index(rank)])
                for label, constraint in dictionary.items()}

        # Get the number of dose voxels
        self.number_of_voxels = hub.dose_information['number_of_voxels']

        # Initialize the tracker dictionary
        self.tracker = {
            label: [] for label in tuple(objectives) + tuple(constraints)}

    def objective(
            self,
            fluence,
            track=True,
            rank=1):
        """
        Compute the objective function value at a rank of the lexicography.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        rank : int
            Current rank of the lexicographic order.

        track : bool, default=True
            Indicator for tracking the objective function values.

        Returns
        -------
        float
            Objective function value.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Loop over the machine learning objectives
        for objective in get_machine_learning_objectives(segmentation):

            # Increment the feature calculator iteration
            (objective.data_model_handler.
             feature_calculator.__iteration__[1]) += 1

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_objective(label, objective):
            """Compute the value of a single objective function."""

            # Get the associated segments and the instance
            segments = objective['segments']
            instance = objective['instance']

            # Get the segment indices
            indices = (
                segmentation[segment]['resized_indices']
                for segment in segments)

            # Compute the objective function value
            objective_value = instance.weight * instance.compute_value(
                tuple(dose[index] for index in indices), segments)

            # Check if the objective value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (objective_value,)

            # Return the objective function value depending on the embedding
            return objective_value * (instance.embedding == 'active')

        # Check if the objective value should be tracked
        if track:

            # Get the labels from all upper ranks
            upper_labels = flatten(
                tuple(self.objectives[key].keys()) for key in {
                    key: value for key, value in self.objectives.items()
                    if key > rank})

            # Loop over the labels
            for label in upper_labels:

                # Enter the value of None into the tracking dictionary
                self.tracker[label] += (None,)

        return sum(
            compute_single_objective(label, objective)
            for label, objective in self.objectives[rank].items())

    def gradient(
            self,
            fluence,
            rank=1):
        """
        Compute the fluence gradient vector at a rank of the lexicography.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        rank : int
            Current rank of the lexicographic order.

        Returns
        -------
        ndarray
            Fluence gradient vector.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        # Initialize the dose gradient vector
        dose_gradient = zeros((self.number_of_voxels,))

        def compute_single_gradient(objective):
            """Compute the value of a single gradient function."""

            # Get the associated segments and the instance
            segments = objective['segments']
            instance = objective['instance']

            # Check if the instance is set to active
            if instance.embedding == 'active':

                # Get the segment indices
                indices = tuple(
                    segmentation[segment]['resized_indices']
                    for segment in segments)

                # Add the single gradient to the dose gradient vector
                dose_gradient[concatenate(indices)] += (
                    instance.weight * instance.compute_gradient(
                        tuple(dose[index] for index in indices), segments))

        # Compute the gradient function for each objective
        apply(compute_single_gradient, self.objectives[rank].values())

        return self.backprojection.compute_fluence_gradient(dose_gradient)

    def constraint(
            self,
            fluence,
            track=True,
            rank=1):
        """
        Compute the constraint function values at a rank of the lexicography.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        rank : int
            Current rank of the lexicographic order.

        track : bool, default=True
            Indicator for tracking the constraint function values.

        Returns
        -------
        float
            Constraint function values.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Loop over the machine learning constraints
        for constraint in get_machine_learning_constraints(segmentation):

            # Increment the feature calculator iteration
            (constraint.data_model_handler.
             feature_calculator.__iteration__[1]) += 1

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_constraint(label, constraint):
            """Compute the value of a single constraint function."""

            # Get the associated segments and the instance
            segments = constraint['segments']
            instance = constraint['instance']

            # Get the segment indices
            indices = (
                segmentation[segment]['resized_indices']
                for segment in segments)

            # Compute the constraint function value
            constraint_value = instance.compute_value(
                tuple(dose[index] for index in indices), segments)

            # Check if the constraint value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (constraint_value,)

            # Return the value of the constraint function
            return constraint_value

        # Check if the constraint value should be tracked
        if track:

            # Get the labels from all upper ranks
            upper_labels = flatten(
                tuple(self.constraints[key].keys()) for key in {
                    key: value for key, value in self.constraints.items()
                    if key > rank})

            # Loop over the labels
            for label in upper_labels:

                # Enter the value of None into the tracking dictionary
                self.tracker[label] += (None,)

        return array([
            compute_single_constraint(label, constraint)
            for label, constraint in self.constraints[rank].items()])

    def jacobian(
            self,
            fluence,
            rank=1):
        """
        Compute the Jacobian matrix at a rank of the lexicography.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        rank : int
            Current rank of the lexicographic order.

        Returns
        -------
        ndarray
            Fluence Jacobian matrix.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        # Initialize the dose Jacobian matrix
        dose_jacobian = zeros((len(self.constraints), self.number_of_voxels))

        def compute_single_jacobian(enumerator):
            """Compute the value of a single constraint jacobian function."""

            # Get the row and constraint
            row, constraint = enumerator

            # Get the associated segments and the instance
            segments = constraint['segments']
            instance = constraint['instance']

            # Get the segment indices
            indices = tuple(
                segmentation[segment]['resized_indices']
                for segment in segments)

            # Insert the single Jacobian into the dose Jacobian matrix
            dose_jacobian[row][concatenate(indices)] = (
                instance.compute_gradient(
                    tuple(dose[index] for index in indices), segments))

        # Compute the Jacobian for each constraint
        apply(
            compute_single_jacobian,
            enumerate(self.constraints[rank].values()))

        return vstack([
            self.backprojection.compute_fluence_gradient(row)
            for row in dose_jacobian])

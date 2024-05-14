"""Weighted-sum optimization problem."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array, vstack

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class WeightedSumOptimization():
    """
    Weighted-sum optimization problem class.

    This class provides methods to perform weighted-sum optimization. It \
    features a component tracker and implements the respective objective, \
    gradient, constraint and constraint jacobian functions.

    Parameters
    ----------
    backprojection : object of class
    :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
    :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        The object representing the type of backprojection.

    objectives : dict
        Dictionary with the internally configured objectives.

    constraints : dict
        Dictionary with the internally configured constraints.

    Attributes
    ----------
    backprojection : object of class
    :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
    :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        See 'Parameters'.

    objectives : dict
        See 'Parameters'.

    constraints : dict
        See 'Parameters'.

    tracker : dict
        Dictionary with the iteration-wise component values.
    """

    def __init__(
            self,
            backprojection,
            objectives,
            constraints):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            "Initializing weighted-sum optimization method ...")

        # Get the instance attributes from the arguments
        self.backprojection = backprojection
        self.objectives = objectives
        self.constraints = constraints

        # Initialize the tracker dictionary
        self.tracker = {label: []
                        for label in tuple(objectives) + tuple(constraints)}

    def objective(
            self,
            fluence,
            track=True):
        """
        Compute the objective function value.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        track : bool, default=True
            Indicator for tracking the single objective function values.

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

            # Compute the objective function value
            objective_value = instance.compute_value(
                tuple(dose[segmentation[segment]['resized_indices']]
                      for segment in segments), segments) * instance.weight

            # Check if the objective value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (objective_value,)

            # Check if the instance is set to active
            if instance.embedding == 'active':

                # Return the value of the objective function
                return objective_value

            # Otherwise, return zero
            return 0.0

        return sum(compute_single_objective(label, objective)
                   for label, objective in self.objectives.items())

    def gradient(
            self,
            fluence):
        """
        Compute the gradient function value.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Gradient function value.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_gradient(objective):
            """Compute the value of a single gradient function."""

            # Get the associated segments and the instance
            segments = objective['segments']
            instance = objective['instance']

            # Check if the instance is set to active
            if instance.embedding == 'active':

                # Return the value of the gradient function
                return instance.compute_gradient(
                    tuple(dose[segmentation[segment]['resized_indices']]
                          for segment in segments), segments) * instance.weight

            # Otherwise, return zero
            return 0.0

        # Compute the total gradient function value
        total_gradient = sum(compute_single_gradient(objective)
                             for objective in self.objectives.values())

        return self.backprojection.compute_fluence_gradient(total_gradient)

    def constraint(
            self,
            fluence,
            track=True):
        """
        Compute the constraint function value(s).

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        track : bool, default=True
            Indicator for tracking the constraint function value(s).

        Returns
        -------
        float
            Constraint function value(s).
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

            # Compute the constraint function value
            constraint_value = instance.compute_value(
                tuple(dose[segmentation[segment]['resized_indices']]
                      for segment in segments), segments) * instance.weight

            # Check if the objective value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (constraint_value,)

            # Check if the instance is set to active
            if instance.embedding == 'active':

                # Return the value of the constraint function
                return constraint_value

            # Otherwise, return the lower constraint bound
            return instance.bounds[0]

        return array([compute_single_constraint(label, constraint)
                      for label, constraint in self.constraints.items()])

    def jacobian(
            self,
            fluence):
        """
        Compute the constraint jacobian function value.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Constraint jacobian function value.
        """

        # Get the segmentation data from the datahub
        segmentation = Datahub().segmentation

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_jacobian(constraint):
            """Compute the value of a single constraint jacobian function."""

            # Get the associated segments and the instance
            segments = constraint['segments']
            instance = constraint['instance']

            # Check if the instance is set to active
            if instance.embedding == 'active':

                # Return the value of the constraint jacobian function
                return instance.compute_gradient(
                    tuple(dose[segmentation[segment]['resized_indices']]
                          for segment in segments), segments) * instance.weight

            # Otherwise, return zero array
            return array([0.0]*Datahub().dose_information['number_of_voxels'])

        # Compute all constraint jacobian function values
        jacobians = (compute_single_jacobian(constraint)
                     for constraint in self.constraints.values())

        return vstack([self.backprojection.compute_fluence_gradient(
            jacobian) for jacobian in jacobians])

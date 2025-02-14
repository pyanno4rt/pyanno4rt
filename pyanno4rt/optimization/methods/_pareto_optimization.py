"""Pareto optimization problem."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class ParetoOptimization():
    """
    Pareto problem class.

    This class provides methods to perform pareto optimization. It implements \
    the respective objective and constraint functions.

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
    """

    def __init__(
            self,
            backprojection,
            objectives,
            constraints):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            "Initializing Pareto optimization problem ...")

        # Get the instance attributes from the arguments
        self.backprojection = backprojection
        self.objectives = objectives
        self.constraints = constraints

    def objective(
            self,
            fluence):
        """
        Compute the objective function values.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        list
            Objective function values.
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

        def compute_single_objective(objective):
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

            # Return the objective function value depending on the embedding
            return objective_value * (instance.embedding == 'active')

        return [
            compute_single_objective(objective)
            for objective in self.objectives.values()]

    def constraint(
            self,
            fluence):
        """
        Compute the constraint function values.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        list
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

        def compute_single_constraint(constraint):
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

            # Return the value of the constraint function
            return constraint_value

        return [
            compute_single_constraint(constraint)
            for constraint in self.constraints.values()]

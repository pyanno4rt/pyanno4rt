"""Pareto optimization problem."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class ParetoProblem():
    """
    Pareto optimization problem class.

    This class provides methods to build a Pareto optimization problem, \
    including the definition of objectives, constraints, bounds, initial \
    fluence, and methods to calculate important quantities.

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

    lower_variable_bounds : None, int, float, or list
        Lower bound(s) on the decision variables.

    upper_variable_bounds : None, int, float, or list
        Upper bound(s) on the decision variables.

    initial_fluence : ndarray
        Initial fluence vector.

    Attributes
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        See 'Parameters'.

    objectives : dict
        See 'Parameters'.

    constraints : dict
        See 'Parameters'.

    initial_fluence : ndarray
        See 'Parameters'.

    variable_bounds : tuple
        Lower and upper bounds on the decision variables.

    constraint_bounds : tuple
        Lower and upper bounds on the constraints.
    """

    # Set the problem name
    name = 'pareto'

    def __init__(
            self,
            backprojection,
            objectives,
            constraints,
            lower_variable_bounds,
            upper_variable_bounds,
            initial_fluence):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            "Building Pareto optimization problem ...")

        # Get the instance attributes from the arguments
        self.backprojection = backprojection
        self.objectives = objectives
        self.constraints = constraints

        # Get the initial fluence
        self.initial_fluence = initial_fluence

        # Get the variable bounds
        self.variable_bounds = self.get_variable_bounds(
            lower_variable_bounds, upper_variable_bounds)

        # Get the constraint bounds
        self.constraint_bounds = self.get_constraint_bounds()

        # Initialize the tracker dictionary
        self.tracker = {
            label: [] for label in tuple(objectives) + tuple(constraints)}

    def get_variable_bounds(
            self,
            lower,
            upper):
        """
        Get the lower and upper variable bounds.

        Parameters
        ----------
        lower : int, float, list or None
            Lower bound(s) on the decision variables.

        upper : int, float, list or None
            Upper bound(s) on the decision variables.

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
                return [value]*len(self.initial_fluence)

            # Check if the value is None
            if value is None:

                # Generate a uniform list from the limit
                return [limit]*len(self.initial_fluence)

            # Generate a cleansed list by replacing None with the limit
            return [limit if bound is None else bound for bound in value]

        return get_bounds(lower, -inf), get_bounds(upper, inf)

    def get_constraint_bounds(self):
        """
        Get the lower and upper constraint bounds.

        Returns
        -------
        tuple
            Lower and upper bounds on the constraints.
        """

        # Check if no constraints have been passed
        if len(self.constraints) == 0:

            # Return the default empty bounds
            return [], []

        # Else, return the unranked, transformed bounds
        return tuple(zip(*(
            constraint['instance'].bounds
            for constraint in self.constraints.values())))

    def objective(
            self,
            fluence,
            track=True):
        """
        Compute the objective function values.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        track : bool, default=True
            Indicator for tracking the objective function values.

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

        def compute_single_objective(label, objective):
            """Compute the value of a single objective function."""

            # Get the associated segments and the instance
            segments = objective['segments']
            instance = objective['instance']

            # Get the segment indices
            indices = (
                segmentation[segment]['resized_indices']
                for segment in segments)

            # Compute the weighted objective function value
            value = instance.weight * instance.compute_value(
                tuple(dose[index] for index in indices), segments)

            # Check if the objective value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (value/instance.weight,)

            # Return the objective function value depending on the embedding
            return value * (instance.embedding == 'active')

        return [
            compute_single_objective(label, objective)
            for label, objective in self.objectives.items()]

    def constraint(
            self,
            fluence,
            track=True):
        """
        Compute the constraint function values.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        track : bool, default=True
            Indicator for tracking the constraint function values.

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
            value = instance.compute_value(
                tuple(dose[index] for index in indices), segments)

            # Check if the constraint value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[label] += (value,)

            # Return the value of the constraint function
            return value

        return [
            compute_single_constraint(label, constraint)
            for label, constraint in self.constraints.items()]

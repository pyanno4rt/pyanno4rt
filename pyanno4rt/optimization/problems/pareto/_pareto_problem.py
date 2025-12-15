"""Pareto optimization problem."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import (
    get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class ParetoProblem():
    """
    Pareto optimization problem class.

    This class provides methods to build a Pareto optimization problem, \
    including the definition of objectives, constraints, bounds, initial \
    fluence, and methods to calculate important quantities. It also features \
    a tracking dictionary with the component-wise evaluations.

    Parameters
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        The object representing the projection between dose and fluence.

    objectives : list
        Plan objectives.

    constraints : list
        Plan constraints.

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

    objectives : list
        See 'Parameters'.

    constraints : list
        See 'Parameters'.

    initial_fluence : ndarray
        See 'Parameters'.

    variable_bounds : tuple
        Lower and upper bounds on the decision variables.

    constraint_bounds : tuple
        Lower and upper bounds on the constraints.

    tracker : dict
        Dictionary with the iteration-wise plan component values.
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
        get_logger().info("Building Pareto optimization problem ...")

        # Get the backprojection
        self.backprojection = backprojection

        # Get the components
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
            label: [] for label in
            tuple(objective.track_id for objective in objectives)
            + tuple(constraint.track_id for constraint in constraints)}

        # Loop over the objectives
        for objective in objectives:

            # Log a message about the objective
            get_logger().info(
                "Using objective '%s' for %s ...",
                objective.name, objective.segment)

        # Loop over the constraints
        for constraint in constraints:

            # Log a message about the constraint
            get_logger().info(
                "Using constraint '%s' for %s ...",
                constraint.name, constraint.segment)

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
            constraint.bounds for constraint in self.constraints)))

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

        # Loop over the machine learning objectives
        for objective in get_machine_learning_objectives(self.objectives):

            # Increment the feature calculator iteration
            objective.feature_calculator._iteration[1] += 1

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_objective(objective):
            """Compute the value of a single objective function."""

            # Compute the weighted objective function value
            value = objective.weight * objective.compute_value(
                tuple(dose[indices] for indices in objective.indices))

            # Check if the objective value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[objective.track_id] += (value/objective.weight,)

            # Return the objective function value depending on the embedding
            return value * (objective.embedding == 'active')

        return [
            compute_single_objective(objective)
            for objective in self.objectives]

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

        # Loop over the machine learning constraints
        for constraint in get_machine_learning_constraints(self.constraints):

            # Increment the feature calculator iteration
            constraint.feature_calculator._iteration[1] += 1

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        def compute_single_constraint(constraint):
            """Compute the value of a single constraint function."""

            # Compute the constraint function value
            value = constraint.compute_value(
                tuple(dose[indices] for indices in constraint.indices))

            # Check if the constraint value should be tracked
            if track:

                # Enter the value into the tracking dictionary
                self.tracker[constraint.track_id] += (value,)

            # Return the value of the constraint function
            return value

        return [
            compute_single_constraint(constraint)
            for constraint in self.constraints]

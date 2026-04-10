"""Weighted-sum optimization problem."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

from json import dump, load
from numpy import array, concatenate, vstack, zeros

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.tools import (
    apply, get_machine_learning_constraints, get_machine_learning_objectives)

# %% Class definition


class WeightedSumProblem():
    """
    Weighted-sum optimization problem class.

    This class provides methods to build a weighted-sum optimization problem, \
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
    name = 'weighted-sum'

    def __init__(
            self,
            backprojection,
            objectives,
            constraints,
            lower_variable_bounds,
            upper_variable_bounds,
            initial_fluence):

        # Log a message about the initialization of the class
        get_logger().info("Building weighted-sum optimization problem ...")

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
                objective.name, list(objective.segment))

        # Loop over the constraints
        for constraint in constraints:

            # Log a message about the constraint
            get_logger().info(
                "Using constraint '%s' for %s ...",
                constraint.name, list(constraint.segment))

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
        Compute the objective function value.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        track : bool, default=True
            Indicator for tracking the objective function values.

        Returns
        -------
        float
            Objective function value.
        """

        # Loop over the machine learning objectives
        for objective in get_machine_learning_objectives(self.objectives):

            # Increment the feature calculator iteration
            objective.model.feature_calculator._iteration[1] += 1

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

        return sum(
            compute_single_objective(objective)
            for objective in self.objectives)

    def gradient(
            self,
            fluence):
        """
        Compute the fluence gradient vector.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Fluence gradient vector.
        """

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        # Initialize the dose gradient vector
        dose_gradient = zeros((len(dose),))

        def compute_single_gradient(objective):
            """Compute the value of a single gradient function."""

            # Check if the instance is set to active
            if objective.embedding == 'active':

                # Add the single gradient to the dose gradient vector
                dose_gradient[concatenate(objective.indices)] += (
                    objective.weight * objective.compute_gradient(
                        tuple(dose[indices] for indices in objective.indices)))

        # Compute the gradient function for each objective
        apply(compute_single_gradient, self.objectives)

        return self.backprojection.compute_fluence_gradient(dose_gradient)

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
        ndarray
            Constraint function values.
        """

        # Loop over the machine learning constraints
        for constraint in get_machine_learning_constraints(self.constraints):

            # Increment the feature calculator iteration
            constraint.model.feature_calculator._iteration[1] += 1

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

        return array([
            compute_single_constraint(constraint)
            for constraint in self.constraints])

    def jacobian(
            self,
            fluence):
        """
        Compute the Jacobian matrix.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Fluence Jacobian matrix.
        """

        # Compute the dose from the fluence
        dose = self.backprojection.compute_dose(fluence)

        # Initialize the dose Jacobian matrix
        dose_jacobian = zeros((len(self.constraints), len(dose)))

        def compute_single_jacobian(enumerator):
            """Compute the value of a single constraint Jacobian function."""

            # Get the row and constraint
            row, constraint = enumerator

            # Insert the single Jacobian into the dose Jacobian matrix
            dose_jacobian[row][concatenate(constraint.indices)] = (
                constraint.compute_gradient(
                    tuple(dose[indices] for indices in constraint.indices)))

        # Compute the Jacobian for each constraint
        apply(compute_single_jacobian, enumerate(self.constraints))

        return vstack([
            self.backprojection.compute_fluence_gradient(row)
            for row in dose_jacobian])

    def load_tracker(
            self,
            path):
        """
        Load the component tracker from a path.

        Parameters
        ----------
        path : str
            Path for loading the component tracker.
        """

        # Open a file stream
        with open(path, 'r', encoding='utf-8') as file:

            # Load the tracker
            self.tracker = load(file)

    def save_tracker(
            self,
            path):
        """
        Save the component tracker to a json file.

        Parameters
        ----------
        path : str
            Path for storing the component tracker.
        """

        # Open a file stream
        with open(path, 'w', encoding='utf-8') as file:

            # Save the tracker
            dump(self.tracker, file, indent=4)

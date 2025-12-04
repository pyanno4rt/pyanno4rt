"""Lexicographic problem."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.optimization.problems.weighted import WeightedSumProblem

# %% Class definition


class LexicographicProblem():
    """
    Lexicographic optimization problem class.

    This class provides methods to build a lexicographic optimization \
    problem, including the definition of objectives, constraints, bounds, \
    initial fluence, subproblems, and methods to calculate important \
    quantities. It also features a tracking dictionary with the \
    component-wise evaluations.

    Parameters
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        The object representing the type of backprojection.

    objectives : list
        Internally configured plan objectives.

    constraints : list
        Internally configured plan constraints.

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

    subproblems : dict
        Dictionary with the rank-ordered subproblems.

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
    name = 'lexicographic'

    def __init__(
            self,
            backprojection,
            objectives,
            constraints,
            lower_variable_bounds,
            upper_variable_bounds,
            initial_fluence):

        # Log a message about the initialization of the class
        get_logger().info("Building lexicographic optimization problem ...")

        # Get the backprojection
        self.backprojection = backprojection

        # Get the components
        self.objectives = objectives
        self.constraints = constraints

        # Get the rank-wise objectives
        rank_objectives = {
            rank: [
                objective for objective in objectives
                if objective.rank == rank]
            for rank in sorted(set(
                    objective.rank for objective in objectives))}

        # Initialize the rank-wise constraints by the "static" constraints
        rank_constraints = {
            rank: [
                constraint for constraint in constraints
                if constraint.rank == rank]
            for rank in rank_objectives}

        # Loop over the lexicographic layers
        for rank in rank_constraints:

            # Update the constraints with the "dynamic" constraints
            rank_constraints[rank] += [
                constraint for constraint_list in (
                    rank_objectives[rank] for rank in tuple(
                        rank_constraints)[:list(rank_constraints).index(rank)])
                for constraint in constraint_list]

        # Initialize the rank-wise optimization problems
        self.subproblems = {
            rank: WeightedSumProblem(
                backprojection, rank_objectives[rank], rank_constraints[rank],
                lower_variable_bounds, upper_variable_bounds, initial_fluence)
            for rank in rank_objectives}

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

        # Return the rank-wise bounds
        return {
            rank: self.subproblems[rank].constraint_bounds
            for rank in self.subproblems}

    def objective(
            self,
            fluence,
            rank=1,
            track=True):
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

        return self.subproblems[rank].objective(fluence, track)

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

        return self.subproblems[rank].gradient(fluence)

    def constraint(
            self,
            fluence,
            rank=1,
            track=True):
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

        return self.subproblems[rank].constraint(fluence, track)

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

        return self.subproblems[rank].jacobian(fluence)

    def restore_tracker(self):
        """
        Restore the full lexicographic tracker from the subproblems.

        Returns
        -------
        dict
            Dictionary with the component-wise values.
        """

        # Get the subproblem trackers
        trackers = tuple(
            problem.tracker for problem in self.subproblems.values())

        # Initialize the lexicographic tracker keys
        self.tracker = {key: [] for key in set().union(*trackers)}

        # Loop over the subproblem trackers
        for tracker in trackers:

            # Get the equal and different keys
            equal_keys = set(self.tracker).intersection(tracker)
            diff_keys = set(self.tracker).difference(tracker)

            # Get the number of rank iterations
            length = len(tracker[next(iter(tracker))])

            # Loop over the equal keys
            for key in equal_keys:

                # Extend the respective tracker values
                self.tracker[key].extend(tracker[key])

            # Loop over the different keys
            for key in diff_keys:

                # Extend the respective tracker values by None
                self.tracker[key].extend([None]*length)

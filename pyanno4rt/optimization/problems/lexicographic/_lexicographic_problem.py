"""Lexicographic problem."""

# Author: Tim Ortkamp

# %% External package import

from math import inf

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.problems.weighted import WeightedSumProblem

# %% Class definition


class LexicographicProblem():
    """
    Lexicographic optimization problem class.

    This class provides methods to build a lexicographic optimization \
    problem, including the definition of the rank-ordered subproblems, and \
    methods to calculate important quantities.

    Parameters
    ----------
    backprojection : object of class \
        :class:`~pyanno4rt.optimization.projections._dose_projection.DoseProjection`\
        :class:`~pyanno4rt.optimization.projections._constant_rbe_projection.ConstantRBEProjection`
        The object used to represent the dose-fluence backprojection.

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

    subproblems : dict
        Dictionary with the rank-ordered subproblems.

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

        # Initialize the datahub
        hub = Datahub()

        # Log a message about the initialization of the class
        hub.logger.display_info(
            "Building lexicographic optimization problem ...")

        # Get the rank-ordered objectives
        objectives = {
            rank: {
                label: objective for label, objective in objectives.items()
                if objective['instance'].rank == rank}
            for rank in sorted(set(
                objective['instance'].rank
                for objective in objectives.values()))}

        # Initialize the rank-ordered constraints by the "static" constraints
        constraints = {
            rank: {
                label: constraint for label, constraint in constraints.items()
                if constraint['instance'].rank == rank}
            for rank in objectives}

        # Loop over the lexicographic layers
        for rank in constraints:

            # Update the constraints with the "dynamic" constraints
            constraints[rank] |= {
                label: constraint for dictionary in (
                    objectives[label] for label in tuple(
                        constraints)[:list(constraints).index(rank)])
                for label, constraint in dictionary.items()}

        # Initialize the rank-wise optimization problems
        self.subproblems = {
            rank: WeightedSumProblem(
                backprojection, objectives[rank], constraints[rank],
                lower_variable_bounds, upper_variable_bounds, initial_fluence)
            for rank in objectives}

        # Initialize the tracker
        self.tracker = {}

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

        # Return the rank-ordered, transformed bounds
        return tuple({
            rank: [
                constraint['instance'].bounds[index]
                for constraint in rank_constraints.values()]
            for rank, rank_constraints in self.constraints.items()}
            for index in range(2))

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

"""Lexicographic problem."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.methods.weighted import WeightedSumOptimization

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

        # Initialize the rank-wise optimization problems
        self.subproblem = {
            rank: WeightedSumOptimization(
                backprojection, self.objectives[rank], self.constraints[rank])
            for rank in self.objectives}

        # Initialize the tracker
        self.tracker = {}

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

        return self.subproblem[rank].objective(fluence, track)

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

        return self.subproblem[rank].gradient(fluence)

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

        return self.subproblem[rank].constraint(fluence, track)

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

        return self.subproblem[rank].jacobian(fluence)

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
            problem.tracker for problem in self.subproblem.values())

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

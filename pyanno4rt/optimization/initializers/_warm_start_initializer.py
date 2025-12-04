"""Warm-start initialization."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array

# %% Internal package import

from pyanno4rt.logging import get_logger

# %% Class definition


class WarmStartInitializer():
    """
    Warm-start initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to a reference point.

    Parameters
    ----------
    initial_fluence_vector: None or list
        User-defined initial fluence vector for the optimization problem.

    Attributes
    ----------
    initial_fluence_vector : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence_vector):

        # Log a message about the initialization of the class
        get_logger().info("Initializing warm-start strategy ...")

        # Get the initial fluence
        self.initial_fluence_vector = initial_fluence_vector

    def run(
            self,
            _):
        """
        Initialize the fluence vector with respect to target coverage.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        # Log a message about the initialization
        get_logger().info(
            "Initializing fluence vector with respect to a reference point "
            "...")

        return array(self.initial_fluence_vector)

"""Warm-start initialization."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array

# %% Internal package import

from pyanno4rt.datahub import Datahub

# %% Class definition


class WarmStartInitializer():
    """
    Warm-start initialization class.

    This class provides methods to initialize the fluence vector with respect \
    to a reference point.

    Parameters
    ----------
    initial_fluence_vector: None or list
        User-defined initial fluence vector.

    Attributes
    ----------
    initial_fluence_vector : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            initial_fluence_vector):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            "Initializing warm-start initializer ...")

        # Get the initial fluence from the argument
        self.initial_fluence_vector = initial_fluence_vector

    def run(self):
        """
        Initialize the fluence vector with respect to a reference point.

        Returns
        -------
        ndarray
            Initial fluence vector.
        """

        # Log a message about the initialization
        Datahub().logger.display_info(
            "Initializing fluence vector with respect to a reference point "
            "...")

        return array(self.initial_fluence_vector)

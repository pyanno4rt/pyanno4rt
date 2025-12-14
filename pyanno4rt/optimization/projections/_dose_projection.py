"""Dose projection."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.optimization.projections import Backprojection

# %% Class definition


class DoseProjection(Backprojection):
    """
    Dose projection class.

    This class provides an implementation of the abstract forward and \
    backward projection methods in \
    :class:`~pyanno4rt.optimization.projections._backprojection.Backprojection`\
    by a linear transformation assuming a neutral RBE value of 1.0.

    Parameters
    ----------
    dose_influence_matrix : csr_matrix
        Dose-influence matrix.
    """

    def __init__(
            self,
            dose_influence_matrix,
            _):

        # Log a message about the initialization of the class
        get_logger().info("Initializing dose projection ...")

        # Call the superclass constructor
        super().__init__()

        # Get the input attributes
        self.dose_influence_matrix = dose_influence_matrix

    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose vector from the fluence vector.

        Parameters
        ----------
        fluence : ndarray
            Fluence vector.

        Returns
        -------
        ndarray
            Dose vector.
        """

        return self.dose_influence_matrix @ fluence

    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Dose gradient.

        Returns
        -------
        ndarray
            Fluence gradient.
        """

        return self.dose_influence_matrix.T @ dose_gradient

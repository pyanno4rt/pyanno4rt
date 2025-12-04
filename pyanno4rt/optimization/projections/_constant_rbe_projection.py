"""Constant RBE projection."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.logging import get_logger
from pyanno4rt.optimization.projections import Backprojection

# %% Class definition


class ConstantRBEProjection(Backprojection):
    """
    Constant RBE projection class.

    This class provides an implementation of the abstract forward and \
    backward projection methods in \
    :class:`~pyanno4rt.optimization.projections._backprojection.Backprojection`\
    by a linear transformation assuming a constant RBE value of 1.1.

    Parameters
    ----------
    dose_influence_matrix : csr_matrix
        Dose-influence matrix.

    RBE : int or float, default=1.1
        Relative biological effectiveness.
    """

    def __init__(
            self,
            dose_influence_matrix,
            RBE=1.1):

        # Log a message about the initialization of the class
        get_logger().info("Initializing constant RBE projection ...")

        # Call the superclass constructor
        super().__init__()

        # Get the input attributes
        self.dose_influence_matrix = dose_influence_matrix
        self.RBE = RBE

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

        return self.dose_influence_matrix @ (self.RBE * fluence)

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

        return self.dose_influence_matrix.T @ (self.RBE * dose_gradient)

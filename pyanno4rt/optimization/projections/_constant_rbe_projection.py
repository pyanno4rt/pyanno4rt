"""Constant RBE projection."""

# Author: Tim Ortkamp

# %% Internal package import

from pyanno4rt.datahub import Datahub
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
    """

    def __init__(self):

        # Log a message about the initialization of the class
        get_logger().info("Initializing constant RBE projection ...")

        # Call the superclass constructor
        super().__init__()

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

        return (
            Datahub().dose_information['dose_influence_matrix'] @ (
                Datahub().plan_configuration['RBE'] * fluence))

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

        return (
            Datahub().dose_information['dose_influence_matrix'].T @ (
                Datahub().plan_configuration['RBE'] * dose_gradient))

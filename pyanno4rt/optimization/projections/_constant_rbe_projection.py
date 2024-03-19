"""Constant RBE projection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.projections import BackProjection

# %% Class definition


class ConstantRBEProjection(BackProjection):
    """
    Constant RBE projection class.

    This class provides an implementation of the abstract forward and \
    backward projection methods in \
    :class:`~pyanno4rt.optimization.projections._backprojection.Backprojection`\
    by a linear function with a constant RBE value of 1.1.
    """

    def __init__(self):

        # Log a message about the initialization of the class
        Datahub().logger.display_info(
            "Initializing constant RBE projection ...")

        # Call the superclass constructor
        super().__init__()

    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose projection from the fluence vector.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence vector.

        Returns
        -------
        ndarray
            Values of the dose vector.
        """

        return (Datahub().dose_information['dose_influence_matrix'] @ (
                Datahub().plan_configuration['RBE'] * fluence))

    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient projection from the dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Values of the dose gradient.

        Returns
        -------
        ndarray
            Values of the fluence gradient.
        """

        return (Datahub().dose_information['dose_influence_matrix'].T @ (
                Datahub().plan_configuration['RBE'] * dose_gradient))

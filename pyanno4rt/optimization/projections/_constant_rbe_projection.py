"""Constant RBE projection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.projections import BackProjection

# %% Class definition


class ConstantRBEProjection(BackProjection):
    """
    Constant RBE projection class. Inherits from the class `Backprojection`.

    This class initializes the calculation methods with a linear dose \
    projection and a constant RBE value of 1.1, which is suitable for proton \
    treatment plan optimization.
    """

    def __init__(self):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing constant RBE projection "
                                      "...")

        # Call the superclass constructor
        super().__init__()

    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose from the RBE-weighted fluence.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the dose.
        """
        # Initialize the datahub
        hub = Datahub()

        return (hub.dose_information['dose_influence_matrix'] @ (
                hub.plan_configuration['RBE'] * fluence))

    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the RBE-weighted dose gradient.

        Parameters
        ----------
        dose_gradient : ndarray
            Values of the dose derivatives.

        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the fluence derivatives.
        """
        # Initialize the datahub
        hub = Datahub()

        return (hub.dose_information['dose_influence_matrix'].T @ (
                hub.plan_configuration['RBE'] * dose_gradient))

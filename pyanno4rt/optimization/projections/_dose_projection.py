"""Dose projection."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.projections import BackProjection

# %% Class definition


class DoseProjection(BackProjection):
    """
    Dose projection class. Inherits from the class `Backprojection`.

    This class initializes the calculation methods with a linear dose \
    projection and a neutral RBE value of 1.0, which is suitable for photon \
    treatment plan optimization.
    """

    def __init__(self):

        # Log a message about the initialization of the class
        Datahub().logger.display_info("Initializing dose projection ...")

        # Call the superclass constructor
        super().__init__()

    def compute_dose_result(
            self,
            fluence):
        """
        Compute the dose from the fluence.

        Parameters
        ----------
        fluence : ndarray
            Values of the fluence.

        Returns
        -------
        ndarray
            Values of the dose.
        """
        return Datahub().dose_information['dose_influence_matrix'] @ fluence

    def compute_fluence_gradient_result(
            self,
            dose_gradient):
        """
        Compute the fluence gradient from the dose gradient.

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
        return (Datahub().dose_information['dose_influence_matrix'].T
                @ dose_gradient)

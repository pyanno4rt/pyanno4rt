"""Dose energy feature."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import array, exp, linspace

# %% Internal package import

from pyanno4rt.learning.features import DosiomicFeature

# %% Class definition


@njit
def sigmoid(value):
    """
    Calculate the sigmoid function value.

    Parameters
    ----------
    value : int or float
        Value at which to calculate the sigmoid function.

    Returns
    -------
    float
        Value of the sigmoid function.
    """

    return 1/(1 + exp(-value))


class DoseEnergy(DosiomicFeature):
    """Dose energy feature class."""

    @staticmethod
    @njit
    def value(dose):
        """
        Compute the dose energy.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose energy value.
        """

        # Set the number of histogram bins
        number_of_bins = 256

        # Determine the boundary values for the bins
        bounds = linspace(dose.min(), dose.max(), number_of_bins+1)

        # Derive the bin sequence
        bins = [(bounds[i], bounds[i+1]) for i in range(number_of_bins)]

        # Set the approximation parameter
        prox = 1e6

        # Get the length of the dose vector
        length = len(dose)

        # Set the offset parameter
        eps = 1e-4

        # Compute the bin probabilities with the double sigmoid approximation
        prob = array([sum([(
            sigmoid(-prox*(dos-bns[1] if dos != bns[1] else dos-bns[1]-eps))
            - sigmoid(-prox*(dos-bns[0] if dos != bns[0] else dos-bns[0]+eps)))
            / length for dos in dose]) for bns in bins])

        # Extract nonzero probabilities for numerical stability
        prob = prob[prob > 0]

        return (prob**2).sum()

    @staticmethod
    @njit
    def gradient(dose):
        """
        Compute the dose energy gradient.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose energy gradient.
        """

        def compute_prob(dos, bns):
            """Compute the bin probability."""

            return 1/length * (
                sigmoid(-prox*(
                    dos-bns[1] if dos != bns[1] else dos-bns[1]-eps))
                - sigmoid(-prox*(
                    dos-bns[0] if dos != bns[0] else dos-bns[0]+eps)))

        def compute_prob_gradient(dos, bns):
            """Compute the gradient of the bin probability."""

            return -prox/length * (
                sigmoid(-prox*(
                    dos-bns[1] if dos != bns[1] else dos-bns[1]-eps))
                * (1-sigmoid(-prox*(
                    dos-bns[1] if dos != bns[1] else dos-bns[1]-eps)))
                - sigmoid(-prox*(
                    dos-bns[0] if dos != bns[0] else dos-bns[0]+eps))
                * (1-sigmoid(-prox*(
                    dos-bns[0] if dos != bns[0] else dos-bns[0]+eps))))

        # Set the number of histogram bins
        number_of_bins = 256

        # Determine the boundary values for the bins
        bounds = linspace(dose.min(), dose.max(), number_of_bins+1)

        # Derive the bin sequence
        bins = [(bounds[i], bounds[i+1]) for i in range(number_of_bins)]

        # Set the approximation parameter
        prox = 1e6

        # Get the length of the dose vector
        length = len(dose)

        # Set the offset parameter
        eps = 1e-4

        # Compute the total gradient over all dose values
        gradient = array([sum([
            2*compute_prob_gradient(dos, bns) * compute_prob(dos, bns)
            for bns in bins]) for dos in dose])

        return gradient

    @staticmethod
    def compute(
            dose,
            *args):
        """
        Call the value function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose energy value.
        """

        return DoseEnergy.value(dose)

    @staticmethod
    def differentiate(
            dose,
            *args):
        """
        Call the gradient function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose energy gradient.
        """

        return DoseEnergy.gradient(dose)

"""Dose entropy feature."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import array, exp, linspace, log, log2

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


class DoseEntropy(DosiomicFeature):
    """Dose entropy feature class."""

    @staticmethod
    @njit
    def function(dose):
        """
        Compute the dose entropy.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose entropy value.
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

        return -(prob*log2(prob)).sum()

    @staticmethod
    @njit
    def gradient(dose):
        """
        Compute the dose entropy gradient.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose entropy gradient.
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

        # Compute the total gradient over all dose values
        gradient = array([sum([
            compute_prob_gradient(dos, bns) * log2(compute_prob(dos, bns))
            + compute_prob_gradient(dos, bns)/log(2)
            for bns in bins if compute_prob(dos, bns) != 0]) for dos in dose])

        return gradient

    @staticmethod
    def compute(dose, *args):
        """
        Call the value function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose entropy value.
        """

        return DoseEntropy.function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """
        Call the gradient function.

        Parameters
        ----------
        dose : ndarray
            Dose array.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        object of class :class:`~jaxlib.xla_extension.ArrayImpl`
            Dose entropy gradient.
        """

        return DoseEntropy.gradient(dose)

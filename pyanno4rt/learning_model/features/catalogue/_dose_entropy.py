"""Dose entropy feature."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numba import njit
from numpy import array, exp, linspace, log, log2
from scipy.sparse import lil_matrix

# %% Internal package import

from pyanno4rt.learning_model.features.catalogue import DosiomicFeature

# %% Class definition


@njit
def sigmoid(
        value,
        coeff_1,
        coeff_2):
    """
    Compute the sigmoid function value.

    Parameters
    ----------
    value : int or float, or tuple of int or float
        Value(s) at which to compute the sigmoid function.

    coeff_1 : int or float
        Multiplicate coefficient for the value in the linear predictor term.

    coeff_2 : int or float
        Additive coefficient for the linear predictor term.

    Returns
    -------
    float or tuple of floats
        Value(s) of the sigmoid function.
    """
    # Check if the passed value is tuple or a list
    if isinstance(value, list):

        return [1/(1 + exp(-coeff_1*val + coeff_2)) for val in value]

    return 1/(1 + exp(-coeff_1*value + coeff_2))


class DoseEntropy(DosiomicFeature):
    """Dose entropy feature class."""

    @staticmethod
    @njit
    def function(dose):
        """Compute the entropy."""
        # Set the number of histogram bins
        number_of_bins = 256

        # Determine the boundary values for the bins
        bounds = linspace(dose.min(), dose.max(), number_of_bins+1)

        # Derive the bin sequence
        bins = [(bounds[i], bounds[i+1]) for i in range(number_of_bins)]

        # Set the approximation parameter
        parameter = 1e6

        # Get the length of the dose vector
        length = len(dose)

        # Set the offset parameter
        epsilon = 1e-4

        # Compute the bin probabilities with the double sigmoid approximation
        prob = array([sum([(sigmoid(-parameter*(dos-bns[1] if dos != bns[1]
                                                else dos-bns[1]-epsilon),
                                    1, 0)
                            - sigmoid(-parameter*(dos-bns[0] if dos != bns[0]
                                                  else dos-bns[0]+epsilon),
                                      1, 0))
                           / length for dos in dose]) for bns in bins])

        # Extract nonzero probabilities for numerical stability
        prob = prob[prob > 0]

        return -(prob*log2(prob)).sum()

    @staticmethod
    @njit
    def gradient(dose):
        """Compute the entropy gradient."""
        # Set the number of histogram bins
        number_of_bins = 256

        # Determine the boundary values for the bins
        bounds = linspace(dose.min(), dose.max(), number_of_bins+1)

        # Derive the bin sequence
        bins = [(bounds[i], bounds[i+1]) for i in range(number_of_bins)]

        # Set the approximation parameter
        parameter = 1e6

        # Get the length of the dose vector
        length = len(dose)

        # Set the offset parameter
        epsilon = 1e-4

        def compute_probability(dos, bns):
            """Compute the bin probability."""
            return 1/length * (sigmoid(-parameter*(dos-bns[1]
                                                   if dos != bns[1]
                                                   else dos-bns[1]-epsilon),
                                       1, 0)
                               - sigmoid(-parameter*(dos-bns[0]
                                                     if dos != bns[0]
                                                     else dos-bns[0]+epsilon),
                                         1, 0))

        def compute_probability_gradient(dos, bns):
            """Compute the gradient of the bin probability."""
            return -parameter/length * (
                sigmoid(
                    -parameter*(dos-bns[1] if dos != bns[1]
                                else dos-bns[1]-epsilon), 1, 0)
                * (1-sigmoid(
                    -parameter*(dos-bns[1] if dos != bns[1]
                                else dos-bns[1]-epsilon), 1, 0))
                - sigmoid(
                    -parameter*(dos-bns[0] if dos != bns[0]
                                else dos-bns[0]+epsilon), 1, 0)
                * (1-sigmoid(
                    -parameter*(dos-bns[0] if dos != bns[0]
                                else dos-bns[0]+epsilon), 1, 0)))

        # Compute the total gradient over all dose values
        gradient = array([sum([compute_probability_gradient(dos, bns)
                               * log2(compute_probability(dos, bns))
                               + compute_probability_gradient(dos, bns)/log(2)
                               for bns in bins
                               if compute_probability(dos, bns) != 0])
                          for dos in dose])

        return gradient

    @staticmethod
    def compute(dose, *args):
        """Call the computation function."""
        return DoseEntropy.function(dose)

    @staticmethod
    def differentiate(dose, *args):
        """Call the differentiation function."""
        # Initialize the gradient vector
        gradient = lil_matrix((1, args[0]))

        # Insert the gradient values at the indices of the segment
        gradient[:, args[1]] = DoseEntropy.gradient(dose)

        return gradient

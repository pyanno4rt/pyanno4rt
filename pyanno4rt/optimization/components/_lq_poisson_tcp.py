"""Linear-quadratic Poisson TCP component."""

# Author: Tim Ortkamp

# %% External package import

from math import log
from numba import njit
from numpy import concatenate, exp

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import RadiobiologicalComponent

# %% Class definition


class LQPoissonTCP(RadiobiologicalComponent):
    """
    Linear-quadratic Poisson TCP component class.

    This class provides methods to compute the value and the gradient of the \
    linear-quadratic Poisson TCP component.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    alpha : int or float
        Alpha coefficient for the tumor volume (in the LQ model).

    beta : int or float
        Beta coefficient for the tumor volume (in the LQ model).

    volume_parameter : int or float
        Dose-volume effect parameter.

    embedding : {'active', 'passive'}, default='active'
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float, default=1.0
        Weight of the component function.

    rank : int, default=1
        Rank of the component in the lexicographic order.

    bounds : None or list, default=None
        Constraint bounds for the component.

    link : None or list, default=None
        Other segments used for joint evaluation.

    identifier : None or str, default=None
        Additional string for naming the component.

    display : bool, default=True
        Indicator for the display of the component.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.
    """

    def __init__(
            self,
            segment,
            alpha=None,
            beta=None,
            volume_parameter=None,
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='LQ Poisson TCP',
            segment=segment,
            parameter_name=('alpha', 'beta', 'volume_parameter'),
            parameter_category=('coefficient', 'coefficient', 'coefficient'),
            parameter_value=(alpha, beta, volume_parameter),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

        # Get the number of fractions
        self.number_of_fractions = Datahub().dose_information[
            'number_of_fractions']

    def compute_value(
            self,
            dose,
            *args):
        """
        Return the function value from the jitted 'compute' function.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose arrays.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        float
            Function value.
        """

        return compute(dose, *self.parameter_value, self.number_of_fractions)

    def compute_gradient(
            self,
            dose,
            *args):
        """
        Return the gradient vector from the jitted 'differentiate' function.

        Parameters
        ----------
        dose : tuple
            Tuple with the dose arrays.

        *args : tuple
            Tuple with optional (non-keyworded) parameters.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        return differentiate(
            dose, *self.parameter_value, self.number_of_fractions)


@njit
def compute(dose, alpha, beta, volume_parameter, number_of_fractions):
    """
    Compute the function value.

    Adapted from Schinkel et al. (2007): \
    https://doi.org/10.2478/v10019-007-0016-7

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    alpha : float
        Alpha coefficient for the tumor volume (in the LQ model).

    beta : float
        Beta coefficient for the tumor volume (in the LQ model).

    volume_parameter : float
        Dose-volume effect parameter.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    # Estimate the tolerance dose at 50% tumor control
    tolerance_dose_50 = (
        log(len(dose)/log(2))/(alpha+beta*eud)) / number_of_fractions

    # Estimate the normalized slope at 50% tumor control
    normalized_slope = (log(2)*log(len(dose)/log(2)))/2

    return -0.5**exp((2*normalized_slope/log(2)) * (1-eud/tolerance_dose_50))


@njit
def differentiate(dose, alpha, beta, volume_parameter, number_of_fractions):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    alpha : float
        Alpha coefficient for the tumor volume (in the LQ model).

    beta : float
        Beta coefficient for the tumor volume (in the LQ model).

    volume_parameter : float
        Dose-volume effect parameter.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    # Estimate the tolerance dose at 50% tumor control
    tolerance_dose_50 = (
        log(len(dose)/log(2))/(alpha+beta*eud)) / number_of_fractions

    # Estimate the normalized slope at 50% tumor control
    normalized_slope = (log(2)*log(len(dose)/log(2)))/2

    # Compute the dose gradient of the EUD
    dose_gradient = (
        (dose**(1/volume_parameter)).sum()**(volume_parameter-1)
        * dose**(1/volume_parameter-1) / (len(dose)**volume_parameter))

    # Compute the xi parameter
    xi = 2*normalized_slope*(tolerance_dose_50-eud)/(tolerance_dose_50*log(2))

    # Compute the EUD gradient of the function
    eud_gradient = (
        -(normalized_slope/tolerance_dose_50)*(0.5**(exp(xi)-1) * exp(xi)))

    return eud_gradient * dose_gradient

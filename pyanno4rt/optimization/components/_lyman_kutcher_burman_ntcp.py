"""Lyman-Kutcher-Burman (LKB) NTCP component."""

# Author: Tim Ortkamp

# %% External package import

from math import erf, pi, sqrt
from numba import njit
from numpy import concatenate, exp

# %% Internal package import

from pyanno4rt.optimization.components import RadiobiologicalComponent

# %% Class definition


class LymanKutcherBurmanNTCP(RadiobiologicalComponent):
    """
    Lyman-Kutcher-Burman (LKB) NTCP component class.

    This class provides methods to compute the value and the gradient of the \
    LKB NTCP component.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    tolerance_dose_50 : int or float
        Tolerance value for the dose at 50% tumor control.

    slope_parameter : int or float
        Slope parameter.

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
    """

    def __init__(
            self,
            segment,
            tolerance_dose_50=None,
            slope_parameter=None,
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
            name='Lyman-Kutcher-Burman NTCP',
            segment=segment,
            parameter_name=(
                'tolerance_dose_50', 'slope_parameter', 'volume_parameter'),
            parameter_category=('dose', 'coefficient', 'coefficient'),
            parameter_value=(
                tolerance_dose_50, slope_parameter, volume_parameter),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

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

        return compute(dose, *self.parameter_value)

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

        return differentiate(dose, *self.parameter_value)


@njit
def compute(dose, tolerance_dose_50, slope_parameter, volume_parameter):
    """
    Compute the function value.

    Adapted from Samant et al. (2023): \
    https://doi.org/10.1016/j.ctro.2023.100595

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    tolerance_dose_50 : float
        Tolerance value for the dose at 50% tumor control.

    slope_parameter : float
        Slope parameter.

    volume_parameter : float
        Dose-volume effect parameter.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    return 0.5*(1+erf(
        (eud-tolerance_dose_50)/(sqrt(2)*tolerance_dose_50*slope_parameter)))


@njit
def differentiate(dose, tolerance_dose_50, slope_parameter, volume_parameter):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    tolerance_dose_50 : float
        Tolerance value for the dose at 50% tumor control.

    slope_parameter : float
        Slope parameter.

    volume_parameter : float
        Dose-volume effect parameter.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    # Compute the dose gradient of the EUD
    dose_gradient = (
        (dose**(1/volume_parameter)).sum()**(volume_parameter-1)
        * dose**(1/volume_parameter-1) / (len(dose)**volume_parameter))

    # Compute the EUD gradient of the function
    eud_gradient = (exp(-(
        (eud-tolerance_dose_50)/(sqrt(2)*tolerance_dose_50*slope_parameter))
        ** 2) / sqrt(pi))

    return eud_gradient * dose_gradient

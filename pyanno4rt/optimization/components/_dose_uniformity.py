"""Dose uniformity component."""

# Author: Tim Ortkamp

# %% External package import

from math import sqrt
from numba import njit
from numpy import concatenate

# %% Internal package import

from pyanno4rt.optimization.components import ConventionalComponent

# %% Class definition


class DoseUniformity(ConventionalComponent):
    """
    Dose uniformity component class.

    This class provides methods to compute the value and the gradient of the \
    dose uniformity component.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

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

    Attributes
    ----------
    parameter_value : list
        Value of the component parameters.
    """

    def __init__(
            self,
            segment,
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Dose Uniformity',
            segment=segment,
            parameter_name=(),
            parameter_category=(),
            parameter_value=(),
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

        return compute(dose)

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

        return differentiate(dose)


@njit
def compute(dose):
    """
    Compute the function value.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    return sqrt(len(dose) / (len(dose)-1))*dose.std()


@njit
def differentiate(dose):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    return (dose - dose.mean()) / (sqrt((len(dose)-1)*len(dose))*dose.std())

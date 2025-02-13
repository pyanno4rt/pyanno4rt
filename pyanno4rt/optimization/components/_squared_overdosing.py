"""Squared overdosing component."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import clip, concatenate

# %% Internal package import

from pyanno4rt.optimization.components import ConventionalComponent

# %% Class definition


class SquaredOverdosing(ConventionalComponent):
    """
    Squared overdosing component class.

    This class provides methods to compute the value and the gradient of the \
    squared overdosing component.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    maximum_dose : int or float
        Maximum value for the dose.

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
            maximum_dose=None,
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Squared Overdosing',
            segment=segment,
            parameter_name=('maximum_dose',),
            parameter_category=('dose',),
            parameter_value=(maximum_dose,),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

        # Set the individual parameter value
        self.parameter_value = [float(maximum_dose)]

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
def compute(dose, maximum_dose):
    """
    Compute the function value.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    maximum_dose : float
        Maximum value for the dose.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the deviation from the maximum dose and clip values below zero
    overdose = clip(dose - maximum_dose, a_min=0, a_max=None)

    return (overdose @ overdose)/len(dose)


@njit
def differentiate(dose, maximum_dose):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    maximum_dose : float
        Maximum value for the dose.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the deviation from the maximum dose and clip values below zero
    overdose = clip(dose - maximum_dose, a_min=0, a_max=None)

    return 2*overdose/len(overdose)

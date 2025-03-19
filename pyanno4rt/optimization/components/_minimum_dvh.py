"""Minimum dose-volume histogram (Minimum DVH) component."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import concatenate, logical_or, quantile, sort

# %% Internal package import

from pyanno4rt.optimization.components import ConventionalComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class MinimumDVH(ConventionalComponent):
    """
    Minimum dose-volume histogram (Minimum DVH) component class.

    This class provides methods to compute the value and the gradient of the \
    minimum DVH component.

    Parameters
    ----------
    segment : str
        Name of the segment associated with the component.

    target_dose : int or float
        Target value for the dose.

    quantile_volume : int or float
        Volume level at which to evaluate the dose quantile.

    component_type : {'constraint', 'objective'}, default='objective'
        Type of the component.

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
    arguments : dict
        Dictionary with the component input arguments (for serialization).
    """

    def __init__(
            self,
            segment,
            target_dose,
            quantile_volume,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='Minimum DVH',
            segment=segment,
            component_type=component_type,
            parameter_name=('target_dose', 'quantile_volume'),
            parameter_category=('dose', 'volume'),
            parameter_value=(target_dose, quantile_volume),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            identifier=identifier,
            display=display)

        # Convert the quantile volume to a relative number
        self.parameter_value[1] /= 100

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

    def to_dict(self):
        """Return the component input dictionary."""

        return {'Minimum DVH': self.arguments}

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
def compute(dose, target_dose, quantile_volume):
    """
    Compute the function value.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    target_dose : float
        Target value for the dose.

    quantile_volume : float
        Relative volume level at which to evaluate the dose quantile.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the deviation from the target dose
    deviation = dose - target_dose

    # Compute the dose quantile
    dose_quantile = quantile(sort(dose)[::-1], quantile_volume)

    # Generate a boolean mask to indicate dose values outside the ROI
    mask = logical_or(dose > target_dose, dose < dose_quantile)

    # Set all deviations outside the ROI to zero
    deviation[mask] = 0

    return (deviation @ deviation)/len(dose)


@njit
def differentiate(
        dose, target_dose, quantile_volume):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Tuple with the dose arrays.

    target_dose : float
        Target value for the dose.

    quantile_volume : float
        Relative volume level at which to evaluate the dose quantile.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose arrays
    dose = concatenate(dose)

    # Compute the deviation from the target dose
    deviation = dose - target_dose

    # Compute the dose quantile
    dose_quantile = quantile(sort(dose)[::-1], quantile_volume)

    # Generate a boolean mask to indicate dose values outside the ROI
    mask = logical_or(dose > target_dose, dose < dose_quantile)

    # Set all deviations outside the ROI to zero
    deviation[mask] = 0

    return 2*deviation/len(dose)

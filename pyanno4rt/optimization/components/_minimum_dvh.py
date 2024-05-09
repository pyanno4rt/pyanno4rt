"""Minimum dose-volume histogram (Minimum DVH) component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numba import njit
from numpy import concatenate, logical_or, quantile, sort, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import ConventionalComponentClass

# %% Class definition


class MinimumDVH(ConventionalComponentClass):
    """
    Minimum dose-volume histogram (Minimum DVH) component class.

    This class provides methods to compute the value and the gradient of the \
    minimum DVH component.

    Parameters
    ----------
    target_dose : int or float, default=None
        Target value for the dose.

    quantile_volume : int or float, default=None
        Volume level at which to evaluate the dose quantile.

    embedding : {'active', 'passive'}, default='active'
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float, default=1.0
        Weight of the component function.

    bounds : None or list, default=None
        Constraint bounds for the component.

    link : None or list, default=None
        Other segments used for joint evaluation.

    identifier : str, default=None
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
            target_dose=None,
            quantile_volume=None,
            embedding='active',
            weight=1.0,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Minimum DVH',
                         parameter_name=('target_dose', 'quantile_volume'),
                         parameter_category=('dose', 'volume'),
                         parameter_value=(target_dose, quantile_volume),
                         embedding=embedding,
                         weight=weight,
                         bounds=bounds,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(target_dose), float(quantile_volume)/100]

    def compute_value(
            self,
            *args):
        """
        Return the component value from the jitted 'compute' function.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters, where args[0] must be the dose vector(s) to \
            evaluate.

        Returns
        -------
        float
            Value of the component function.
        """

        return compute(args[0], self.parameter_value)

    def compute_gradient(
            self,
            *args):
        """
        Return the component gradient from the jitted 'differentiate' function.

        Parameters
        ----------
        *args : tuple
            Keyworded parameters, where args[0] must be the dose vector(s) to \
            evaluate and args[1] the corresponding segment(s).

        Returns
        -------
        ndarray
            Value of the component gradient.
        """

        # Initialize the datahub
        hub = Datahub()

        return differentiate(args[0], self.parameter_value,
                             hub.dose_information['number_of_voxels'],
                             tuple(hub.segmentation[segment]['resized_indices']
                                   for segment in args[1]))


@njit
def compute(dose, parameter_value):
    """
    Compute the component value.

    Parameters
    ----------
    dose : tuple
        Values of the dose in the segment(s).

    parameter_value : list
        Value of the component parameters.

    Returns
    -------
    float
        Value of the component function.
    """

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    # Compute the deviation from the target dose
    deviation = full_dose - parameter_value[0]

    # Compute the dose quantile
    dose_quantile = quantile(sort(full_dose)[::-1], parameter_value[1])

    # Generate a boolean mask to indicate dose values outside the ROI
    mask = logical_or(
        full_dose > parameter_value[0], full_dose < dose_quantile)

    # Set all deviations outside the ROI to zero
    deviation[mask] = 0

    return (deviation @ deviation) / len(full_dose)


@njit
def differentiate(dose, parameter_value, number_of_voxels, segment_indices):
    """
    Compute the component gradient.

    Parameters
    ----------
    dose : tuple
        Values of the dose in the segment(s).

    parameter_value : list
        Value of the component parameters.

    number_of_voxels : int
        Total number of dose voxels.

    segment_indices : tuple
        Indices of the segment(s).

    Returns
    -------
    ndarray
        Value of the component gradient.
    """

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    # Concatenate the segment index arrays
    full_indices = concatenate(segment_indices)

    # Compute the deviation from the target dose
    deviation = full_dose - parameter_value[0]

    # Compute the dose quantile
    dose_quantile = quantile(sort(full_dose)[::-1], parameter_value[1])

    # Generate a boolean mask to indicate dose values outside the ROI
    mask = logical_or(
        full_dose > parameter_value[0], full_dose < dose_quantile)

    # Set all deviations outside the ROI to zero
    deviation[mask] = 0

    # Initialize the component gradient
    component_gradient = zeros((number_of_voxels,))

    # Compute the component gradient
    component_gradient[full_indices] = 2*deviation/len(full_dose)

    return component_gradient

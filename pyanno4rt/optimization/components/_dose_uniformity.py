"""Dose uniformity component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import sqrt
from numba import njit
from numpy import concatenate, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import ConventionalComponentClass

# %% Class definition


class DoseUniformity(ConventionalComponentClass):
    """
    Dose uniformity component class.

    This class provides methods to compute the value and the gradient of the \
    dose uniformity component.

    Parameters
    ----------
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
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Dose Uniformity',
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

        # Set the individual parameter value
        self.parameter_value = []

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

        return compute(args[0])

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

        return differentiate(args[0], hub.dose_information['number_of_voxels'],
                             tuple(hub.segmentation[segment]['resized_indices']
                                   for segment in args[1]))


@njit
def compute(dose):
    """
    Compute the component value.

    Parameters
    ----------
    dose : tuple
        Values of the dose in the segment(s).

    Returns
    -------
    float
        Value of the component function.
    """

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    return sqrt(len(full_dose)/(len(full_dose)-1)) * full_dose.std()


@njit
def differentiate(dose, number_of_voxels, segment_indices):
    """
    Compute the component gradient.

    Parameters
    ----------
    dose : tuple
        Values of the dose in the segment(s).

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

    # Initialize the component gradient
    component_gradient = zeros((number_of_voxels,))

    # Compute the component gradient
    component_gradient[full_indices] = (
        (full_dose-full_dose.mean())
        / (sqrt((len(full_dose)-1)*len(full_dose))*full_dose.std()))

    return component_gradient

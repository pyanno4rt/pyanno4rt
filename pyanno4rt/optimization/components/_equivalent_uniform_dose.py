"""Equivalent uniform dose (EUD) component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numba import njit
from numpy import concatenate, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import ConventionalComponentClass

# %% Class definition


class EquivalentUniformDose(ConventionalComponentClass):
    """
    Equivalent uniform dose (EUD) component class.

    This class provides methods to compute the value and the gradient of the \
    EUD component.

    Parameters
    ----------
    target_eud : int or float, default=None
        Target value for the EUD.

    volume_parameter : int or float, default=None
        Dose-volume effect parameter.

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
            target_eud=None,
            volume_parameter=None,
            embedding='active',
            weight=1.0,
            bounds=None,
            link=None,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(name='Equivalent Uniform Dose',
                         parameter_name=('target_eud', 'volume_parameter'),
                         parameter_category=('dose', 'parameter'),
                         parameter_value=(target_eud, volume_parameter),
                         embedding=embedding,
                         weight=weight,
                         bounds=bounds,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [float(target_eud), float(volume_parameter)]

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

    # Compute the EUD
    eud = (sum(full_dose**(1/parameter_value[1]))/len(full_dose)
           )**parameter_value[1]

    return (eud - parameter_value[0])**2


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

    # Compute the EUD
    eud = (sum(full_dose**(1/parameter_value[1]))/len(full_dose)
           )**parameter_value[1]

    # Compute the dose gradient of the EUD
    eud_gradient = (
        sum(full_dose**(1/parameter_value[1]))**(parameter_value[1]-1)
        * full_dose**(1/parameter_value[1]-1)
        / (len(full_dose)**parameter_value[1]))

    # Initialize the component gradient
    component_gradient = zeros((number_of_voxels,))

    # Compute the gradient
    component_gradient[full_indices] = 2*(eud-parameter_value[0])*eud_gradient

    return component_gradient
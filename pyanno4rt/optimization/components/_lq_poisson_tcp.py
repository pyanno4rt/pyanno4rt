"""Linear-quadratic Poisson TCP component."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import log
from numba import njit
from numpy import concatenate, exp, zeros

# %% Internal package import

from pyanno4rt.datahub import Datahub
from pyanno4rt.optimization.components import RadiobiologyComponentClass

# %% Class definition


class LQPoissonTCP(RadiobiologyComponentClass):
    """
    Linear-quadratic Poisson TCP component class.

    This class provides methods to compute the value and the gradient of the \
    linear-quadratic Poisson TCP component.

    Parameters
    ----------
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

    Attributes
    ----------
    parameter_value : list
        Value of the component parameters.
    """

    def __init__(
            self,
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
        super().__init__(name='LQ Poisson TCP',
                         parameter_name=('alpha', 'beta', 'volume_parameter'),
                         parameter_category=(
                             'coefficient', 'coefficient', 'coefficient'),
                         parameter_value=(alpha, beta, volume_parameter),
                         embedding=embedding,
                         weight=weight,
                         rank=rank,
                         bounds=bounds,
                         link=link,
                         identifier=identifier,
                         display=display)

        # Set the individual parameter value
        self.parameter_value = [
            float(alpha), float(beta), float(volume_parameter)]

        # Transform the component bounds
        self.bounds = sorted(-self.weight*bound for bound in self.bounds)

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

        return compute(args[0], self.parameter_value,
                       Datahub().dose_information['number_of_fractions'])

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
                                   for segment in args[1]),
                             Datahub().dose_information['number_of_fractions'])


@njit
def compute(dose, parameter_value, number_of_fractions):
    """
    Compute the component value.

    Adapted from Schinkel et al. (2007): \
    https://doi.org/10.2478/v10019-007-0016-7

    Parameters
    ----------
    dose : tuple
        Values of the dose in the segment(s).

    parameter_value : list
        Value of the component parameters.

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

    Returns
    -------
    float
        Value of the component function.
    """

    # Concatenate the dose arrays
    full_dose = concatenate(dose)

    # Compute the EUD
    eud = ((full_dose**(1/parameter_value[2])).sum()/len(full_dose)
           )**parameter_value[2]

    # Estimate the tolerance dose at 50% tumor control
    tolerance_dose_50 = (
        log(len(full_dose)/log(2))/(parameter_value[0]+parameter_value[1]*eud)
        ) / number_of_fractions

    # Estimate the normalized slope at 50% tumor control
    normalized_slope = (log(2)*log(len(full_dose)/log(2)))/2

    return -0.5**exp((2*normalized_slope/log(2)) * (1-eud/tolerance_dose_50))


@njit
def differentiate(dose, parameter_value, number_of_voxels, segment_indices,
                  number_of_fractions):
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

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

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
    eud = (1/len(full_dose)*(full_dose**(1/parameter_value[2])).sum()
           )**parameter_value[2]

    # Estimate the tolerance dose at 50% tumor control
    tolerance_dose_50 = (
        log(len(full_dose)/log(2))/(parameter_value[0]+parameter_value[1]*eud)
        ) / number_of_fractions

    # Estimate the normalized slope at 50% tumor control
    normalized_slope = (log(2)*log(len(full_dose)/log(2)))/2

    # Compute the dose gradient of the EUD
    eud_gradient = (
        (full_dose**(1/parameter_value[2])).sum()**(parameter_value[2]-1)
        * full_dose**(1/parameter_value[2]-1)
        / (len(full_dose)**parameter_value[2]))

    # Compute the EUD gradient of the component
    tcp_gradient = -(normalized_slope/tolerance_dose_50)*(
        0.5**(exp(2*normalized_slope*(tolerance_dose_50-eud)
                  / (tolerance_dose_50*log(2)))-1)
        * exp(2*normalized_slope*(tolerance_dose_50-eud)
              / (tolerance_dose_50*log(2))))

    # Initialize the component gradient
    component_gradient = zeros((number_of_voxels,))

    # Compute the gradient
    component_gradient[full_indices] = tcp_gradient * eud_gradient

    return component_gradient

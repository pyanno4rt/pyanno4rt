"""Linear-quadratic Poisson TCP component."""

# Author: Tim Ortkamp

# %% External package import

from math import log
from numba import njit
from numpy import concatenate, exp

# %% Internal package import

from pyanno4rt.optimization.components import RadiobiologicalComponent
from pyanno4rt.tools import filter_dict

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

    number_of_fractions : int
        Number of fractions according to the treatment scheme.

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

    transform : bool, default=False
        Indicator for the transformation of the outcome function.

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
            alpha,
            beta,
            volume_parameter,
            number_of_fractions,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            link=None,
            transform=False,
            identifier=None,
            display=True):

        # Call the superclass constructor to initialize and check attributes
        super().__init__(
            name='LQ Poisson TCP',
            segment=segment,
            component_type=component_type,
            parameter_name=(
                'alpha', 'beta', 'volume_parameter', 'number_of_fractions'),
            parameter_category=(
                'coefficient', 'coefficient', 'coefficient', 'coefficient'),
            parameter_value=(
                alpha, beta, volume_parameter, number_of_fractions),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            link=link,
            transform=transform,
            identifier=identifier,
            display=display)

        # Set the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

        # Convert the bounds
        self.bounds = [
            -self.weight*self.reverse(bound) for bound in self.bounds]

    def to_dict(self):
        """Serialize the component into a dictionary."""

        return {self.name: self.arguments}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the component from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the component parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.optimization.components._lq_poisson_tcp.LQPoissonTCP`
            The object used to handle the component parameters.
        """

        return cls(**dictionary)

    def translate(
            self,
            value):
        """
        Translate function values to outcome values.

        Parameters
        ----------
        value : int, float, tuple or list
            Function value to translate.

        Returns
        -------
        int, float, tuple or list
            Outcome value.
        """

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of outcome values
            return [-val for val in value]

        # Return a single outcome value
        return -value

    def reverse(
            self,
            value):
        """
        Reverse outcome values to function values.

        Parameters
        ----------
        value : int, float, tuple or list
            Outcome value to reverse.

        Returns
        -------
        int, float, tuple or list
            Function value.
        """

        # Check if the value is an iterable
        if isinstance(value, (tuple, list)):

            # Return a list of function values
            return [-val for val in value]

        # Return a single function value
        return -value

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

        return self.reverse(compute(dose, *self.parameter_value))

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

    return 0.5**exp((2*normalized_slope/log(2)) * (1-eud/tolerance_dose_50))


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

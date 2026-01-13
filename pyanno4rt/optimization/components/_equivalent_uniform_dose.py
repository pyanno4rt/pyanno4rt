"""Equivalent uniform dose (EUD) component."""

# Author: Tim Ortkamp

# %% External package import

from numba import njit
from numpy import concatenate

# %% Internal package import

from pyanno4rt.optimization.components import ConventionalComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class EquivalentUniformDose(ConventionalComponent):
    """
    Equivalent uniform dose (EUD) component class.

    This class provides methods to compute the value and the gradient of the \
    equivalent uniform dose (EUD) component.

    Parameters
    ----------
    segment : str or list
        Segment(s) associated with the component.

    target_eud : int or float
        Target value for the EUD.

    volume_parameter : int or float
        Dose-volume effect parameter.

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

    identifier : None or str, default=None
        Additional string for naming the component.

    Attributes
    ----------
    arguments : dict
        Dictionary with the input arguments (for serialization).

    Notes
    -----
    See :class:`~pyanno4rt.optimization.components._conventional_component.ConventionalComponent`\
    for details on the inherited attributes.
    """

    def __init__(
            self,
            segment,
            target_eud,
            volume_parameter,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            identifier=None):

        # Call the superclass constructor
        super().__init__(
            name='Equivalent Uniform Dose',
            segment=segment,
            component_type=component_type,
            parameter_name=('target_eud', 'volume_parameter'),
            parameter_category=('dose', 'parameter'),
            parameter_value=(target_eud, volume_parameter),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            identifier=identifier)

        # Get the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

    def to_dict(self):
        """
        Serialize the component into a dictionary.

        Returns
        -------
        dict
            Dictionary with the component's arguments.
        """

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
            Dictionary with the component's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.optimization.components._equivalent_uniform_dose.EquivalentUniformDose`
            The object used to represent the equivalent uniform dose component.
        """

        return cls(**dictionary)

    def compute_value(
            self,
            dose):
        """
        Return the function value from the jitted 'compute' function.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        Returns
        -------
        float
            Function value.
        """

        return compute(dose, *self.parameter_value)

    def compute_gradient(
            self,
            dose):
        """
        Return the gradient vector from the jitted 'differentiate' function.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        Returns
        -------
        ndarray
            Gradient vector.
        """

        return differentiate(dose, *self.parameter_value)


@njit
def compute(dose, target_eud, volume_parameter):
    """
    Compute the function value.

    Parameters
    ----------
    dose : tuple
        Dose vectors.

    target_eud : float
        Target value for the EUD.

    volume_parameter : float
        Dose-volume effect parameter.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose vectors
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    return (eud - target_eud)**2


@njit
def differentiate(
        dose, target_eud, volume_parameter):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Dose vectors.

    target_eud : float
        Target value for the EUD.

    volume_parameter : float
        Dose-volume effect parameter.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose vectors
    dose = concatenate(dose)

    # Compute the EUD
    eud = ((dose**(1/volume_parameter)).sum()/len(dose))**volume_parameter

    # Compute the dose gradient of the EUD
    eud_gradient = (
        (dose**(1/volume_parameter)).sum()**(volume_parameter-1)
        * dose**(1/volume_parameter-1) / (len(dose)**volume_parameter))

    return 2*(eud - target_eud)*eud_gradient

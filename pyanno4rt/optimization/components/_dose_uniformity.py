"""Dose uniformity component."""

# Author: Tim Ortkamp

# %% External package import

from math import sqrt
from numba import njit
from numpy import concatenate

# %% Internal package import

from pyanno4rt.optimization.components import ConventionalComponent
from pyanno4rt.tools import filter_dict

# %% Class definition


class DoseUniformity(ConventionalComponent):
    """
    Dose uniformity component class.

    This class provides methods to compute the value and the gradient of the \
    dose uniformity component.

    Parameters
    ----------
    segment : str or list
        Segment(s) associated with the component.

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
        Dictionary with the component input arguments (for serialization).

    Notes
    -----
    See :class:`~pyanno4rt.optimization.components._conventional_component.ConventionalComponent`\
    for details on the inherited attributes.
    """

    def __init__(
            self,
            segment,
            component_type='objective',
            embedding='active',
            weight=1.0,
            rank=1,
            bounds=None,
            identifier=None):

        # Call the superclass constructor
        super().__init__(
            name='Dose Uniformity',
            segment=segment,
            component_type=component_type,
            parameter_name=(),
            parameter_category=(),
            parameter_value=(),
            embedding=embedding,
            weight=weight,
            rank=rank,
            bounds=bounds,
            identifier=identifier)

        # Get the input arguments
        self.arguments = filter_dict(
            locals(), remove_keys=('self', '__class__'))

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
            :class:`~pyanno4rt.optimization.components._dose_uniformity.DoseUniformity`
            The object used to represent the dose uniformity component.
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

        return compute(dose)

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

        return differentiate(dose)


@njit
def compute(dose):
    """
    Compute the function value.

    Parameters
    ----------
    dose : tuple
        Dose vectors.

    Returns
    -------
    float
        Function value.
    """

    # Concatenate the dose vectors
    dose = concatenate(dose)

    return sqrt(len(dose) / (len(dose)-1))*dose.std()


@njit
def differentiate(dose):
    """
    Compute the gradient vector.

    Parameters
    ----------
    dose : tuple
        Dose vectors.

    Returns
    -------
    ndarray
        Gradient vector.
    """

    # Concatenate the dose vectors
    dose = concatenate(dose)

    return (dose - dose.mean()) / (sqrt((len(dose)-1)*len(dose))*dose.std())

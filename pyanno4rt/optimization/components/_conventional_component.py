"""Conventional component template."""

# Author: Tim Ortkamp

# %% External package import

from abc import ABCMeta, abstractmethod
from functools import partial
from math import inf

# %% Internal package import

from pyanno4rt.tools import filter_dict, wrap
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

# %% Class definition


class ConventionalComponent(metaclass=ABCMeta):
    """
    Conventional component template class.

    Parameters
    ----------
    name : str
        Name of the component class.

    segment : str or list
        Segment(s) associated with the component.

    component_type : {'constraint', 'objective'}
        Type of the component.

    parameter_name : tuple
        Name of the component parameters.

    parameter_category : tuple
        Category of the component parameters.

    parameter_value : tuple
        Value of the component parameters.

    embedding : {'active', 'passive'}
        Mode of embedding for the component. In 'passive' mode, the component \
        value is computed and tracked, but not considered in the optimization \
        problem, unlike in 'active' mode.

    weight : int or float
        Weight of the component function.

    rank : int
        Rank of the component in the lexicographic order.

    bounds : None or list
        Constraint bounds for the component.

    identifier : None or str
        Additional string for naming the component.

    Attributes
    ----------
    name : str
        See 'Parameters'.

    segment : tuple
        See 'Parameters'.

    component_type : {'constraint', 'objective'}
        See 'Parameters'.

    parameter_name : tuple
        See 'Parameters'.

    parameter_category : tuple
        See 'Parameters'.

    parameter_value : list
        See 'Parameters'.

    embedding : {'active', 'passive'}
        See 'Parameters'.

    weight : float
        See 'Parameters'.

    rank : int
        See 'Parameters'.

    bounds : list
        See 'Parameters'.

    identifier : None or str
        See 'Parameters'.

    adjusted_parameters : bool
        Indicator for the adjustment of the parameters due to fractionation.

    track_id : str
        Component identifier in the optimization problem tracker.

    indices : list
        Indices of the segment(s).
    """

    def __init__(
            self,
            name,
            segment,
            component_type,
            parameter_name,
            parameter_category,
            parameter_value,
            embedding,
            weight,
            rank,
            bounds,
            identifier):

        # Validate the input arguments
        self.validate(
            filter_dict(locals(), remove_keys=('self', 'parameter_value'))
            | dict(zip(parameter_name, parameter_value)))

        # Get the instance attributes
        self.name = name
        self.segment = wrap(segment)
        self.component_type = component_type
        self.parameter_name = parameter_name
        self.parameter_category = parameter_category
        self.parameter_value = list(map(float, parameter_value))
        self.embedding = embedding
        self.weight = float(weight)
        self.rank = rank
        self.bounds = self.convert_bounds(bounds, embedding)
        self.identifier = identifier

        # Initialize the adjustment indicator
        self.adjusted_parameters = False

        # Initialize the tracker identifier
        self.track_id = '-'.join(filter(
            None, (str(self.segment), self.name, self.identifier)))

        # Initialize the segment indices
        self.indices = None

    def __eq__(
            self,
            other):
        """
        Compare an instance with another object.

        Parameters
        ----------
        other : object
            The object to compare the instance with.

        Returns
        -------
        bool
            Indicator for the equality of the objects.
        """

        return all(self.__dict__[key] == other.__dict__[key] for key in (
            'name', 'segment', 'component_type', 'identifier'))

    def __hash__(self):
        """
        Return the hash value.

        Returns
        -------
        int
            Hash value.
        """

        return hash(
            (self.name, self.segment, self.component_type, self.identifier))

    def get_class(self):
        """
        Get the name of the component class.

        Returns
        -------
        str
            Name of the component class.
        """

        return 'ConventionalComponent'

    def convert_bounds(
            self,
            bounds,
            embedding):
        """
        Convert the bounds to function bounds.

        Parameters
        ----------
        bounds : None or list
            Constraint bounds for the component.

        embedding : {'active', 'passive'}
            Mode of embedding for the component.

        Returns
        -------
        list
            Lower and upper function bounds.
        """

        # Check if the bounds are None
        if bounds is None or embedding == 'passive':

            # Return the default function bounds
            return [-inf, inf]

        # Return the transformed function bounds
        return sorted(
            (-inf if bounds[0] is None else float(bounds[0]),
             inf if bounds[1] is None else float(bounds[1])))

    @abstractmethod
    def to_dict(self):
        """Serialize the component into a dictionary."""

    @classmethod
    @abstractmethod
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
            :class:`~pyanno4rt.optimization.components._conventional_component.ConventionalComponent`
            The object used to represent the conventional component.
        """

    @abstractmethod
    def compute_value(
            self,
            dose):
        """
        Compute the component value.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        Returns
        -------
        float
            Function value.
        """

    @abstractmethod
    def compute_gradient(
            self,
            dose):
        """
        Compute the component gradient.

        Parameters
        ----------
        dose : tuple
            Dose vectors.

        Returns
        -------
        ndarray
            Gradient vector.
        """

    def validate(
            self,
            inputs):
        """
        Validate the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the validation map
        validation_map = {
            'name': (
                partial(validate_type, options=str),
                ),
            'segment': (
                partial(validate_type, options=(str, list)),
                ),
            'component_type': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'objective', 'constraint'))
                ),
            'parameter_name': (
                partial(validate_type, options=tuple),
                partial(validate_subtype, options=str)
                ),
            'parameter_category': (
                partial(validate_type, options=tuple),
                partial(validate_subtype, options=str)
                ),
            'embedding': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('active', 'passive'))
                ),
            'weight': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'rank': (
                partial(validate_type, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'bounds': (
                partial(validate_type, options=(type(None), list)),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(type(None), int, float))
                ),
            'identifier': (
                partial(validate_type, options=(type(None), str)),
                ),
            'target_eud': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'volume_parameter': (
                partial(validate_type, options=(int, float)),
                ),
            'target_dose': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'quantile_volume': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=100, sign='<=')
                ),
            'maximum_dose': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'minimum_dose': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

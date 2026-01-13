"""Input columns."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length,
    validate_string_number, validate_subtype, validate_type)

# %% Class definitions


class DynamicFeature():
    """
    Dynamic input feature class.

    This class provides methods to set, validate and serialize a dynamic \
    input feature for the learning models.

    Parameters
    ----------
    column : str
        Name of the feature.

    segment : str
        Segment associated with the feature.

    function : str
        Name of the (re)calculation function.

    argument : None, int, float or str, default=None
        Additional function argument.

    scale : {'metric', 'nominal', 'ordinal'}, default='metric'
        Feature scale.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    segment : str
        See 'Parameters'.

    function : str
        See 'Parameters'.

    argument : None, int, float or str
        See 'Parameters'.

    scale : {'metric', 'nominal', 'ordinal'}
        See 'Parameters'.
    """

    # Set the column category
    category = 'feature'

    def __init__(
            self,
            column,
            segment,
            function,
            argument=None,
            scale='metric'):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Validate the input arguments
        self.validate(inputs)

        # Get the input attributes
        self.column = column
        self.segment = segment
        self.function = function
        self.argument = argument
        self.scale = scale

    def to_dict(self):
        """
        Serialize the feature into a dictionary.

        Returns
        -------
        dict
            Dictionary with the feature's arguments.
        """

        return {'Dynamic Feature': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the feature from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the feature's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.features._columns.DynamicFeature`
            The object used to handle the feature parameters.
        """

        return cls(**dictionary)

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

        # Get the validation functions for the 'function' argument
        argument_map = {
            'Dose Gradient': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=('x', 'y', 'z'))
                ),
            'Dose Moment': (
                partial(validate_type, options=str),
                partial(validate_length, reference=3, sign='=='),
                validate_string_number
                ),
            'Dose Subvolume': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3', 'y1of2',
                    'y2of2', 'y1of3', 'y2of3', 'y3of3', 'z1of2', 'z2of2',
                    'z1of3', 'z2of3', 'z3of3'))
                ),
            'Dx': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>'),
                partial(validate_item, reference=100, sign='<')
                ),
            'Vx': (
                partial(validate_type, options=(int, float)),
                partial(validate_item, reference=0, sign='>'),
                partial(validate_item, reference=100, sign='<')
                ),
            'none': (
                partial(validate_type, options=type(None)),
                )
            }

        # Get the validation map
        validation_map = {
            'column': (
                partial(validate_type, options=str),
                ),
            'segment': (
                partial(validate_type, options=str),
                ),
            'function': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(*maps.FEATURES,))
                ),
            'argument': argument_map[
                inputs['function'] if inputs['function'] in (
                    'Dx', 'Vx', 'Dose Gradient', 'Dose Moment',
                    'Dose Subvolume')
                else 'none'
                ],
            'scale': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'metric', 'nominal', 'ordinal'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)


class StaticFeature():
    """
    Static input feature class.

    This class provides methods to set, validate and serialize a static input \
    feature for the learning models.

    Parameters
    ----------
    column : str
        Name of the feature.

    value : int, float or str
        Static feature value.

    scale : {'metric', 'nominal', 'ordinal'}, default='metric'
        Feature scale.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    value : int, float or str
        See 'Parameters'.

    scale : {'metric', 'nominal', 'ordinal'}
        See 'Parameters'.
    """

    # Set the column category
    category = 'feature'

    def __init__(
            self,
            column,
            value,
            scale='metric'):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Validate the input arguments
        self.validate(inputs)

        # Get the input attributes
        self.column = column
        self.value = value
        self.scale = scale

    def to_dict(self):
        """
        Serialize the feature into a dictionary.

        Returns
        -------
        dict
            Dictionary with the feature's arguments.
        """

        return {'Static Feature': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the feature from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the feature's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.features._columns.StaticFeature`
            The object used to handle the feature parameters.
        """

        return cls(**dictionary)

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
            'column': (
                partial(validate_type, options=str),
                ),
            'value': (
                partial(validate_type, options=(int, float, str)),
                ),
            'scale': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'metric', 'nominal', 'ordinal'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)


class Label():
    """
    Input label class.

    This class provides methods to set, validate and serialize an input label \
    for the learning models.

    Parameters
    ----------
    column : str
        Name of the label.

    viewpoint : {'early', 'late', 'longitudinal', 'long-term'}, \
        default='longitudinal'
        Label viewpoint for time-dependent modeling.

    time_variable : None or str, default=None
        Name of the data column for time-dependent modeling.

    bounds : tuple or list, default=(1, 1)
        Bounds for binarization of the label values.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    viewpoint : {'early', 'late', 'longitudinal', 'long-term'}
        See 'Parameters'.

    time_variable : None or str
        See 'Parameters'.

    bounds : list
        See 'Parameters'.
    """

    # Set the column category
    category = 'label'

    def __init__(
            self,
            column,
            viewpoint='longitudinal',
            time_variable=None,
            bounds=(1, 1)):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Validate the input arguments
        self.validate(inputs)

        # Get the input attributes
        self.column = column
        self.viewpoint = viewpoint
        self.time_variable = time_variable
        self.bounds = list(bounds)

    def to_dict(self):
        """
        Serialize the label into a dictionary.

        Returns
        -------
        dict
            Dictionary with the label's arguments.
        """

        return {'Label': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the label from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the label's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.features._columns.Label`
            The object used to handle the label parameters.
        """

        return cls(**dictionary)

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
            'column': (
                partial(validate_type, options=str),
                ),
            'viewpoint': (
                partial(validate_type, options=str),
                partial(validate_item_in_set, options=(
                    'early', 'late', 'long-term', 'longitudinal'))
                ),
            'time_variable': (
                partial(validate_type, options=(type(None), str)),
                ),
            'bounds': (
                partial(validate_type, options=(tuple, list)),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(type(None), int, float))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

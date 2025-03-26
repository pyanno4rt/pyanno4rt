"""Input columns."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_string_is_number, check_subtype, check_type,
    check_value, check_value_in_set)
import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict

# %% Class definitions


class DynamicFeature():
    """
    Dynamic input feature class.

    This class provides methods to set, validate and serialize a dynamic \
    input feature for the learning models.

    Parameters
    ----------
    column : str
        Name of the data column.

    segment : None or str
        Segment associated with the feature.

    function : None or str
        Name of the (re)calculation function.

    argument : None, int, float or str, default=None
        Additional function argument.

    scale : {'metric', 'nominal', 'ordinal'}, default='metric'
        Feature scale.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    segment : None or str
        See 'Parameters'.

    function : None or str
        See 'Parameters'.

    argument : None, int, float or str
        See 'Parameters'.

    scale : {'metric', 'nominal', 'ordinal'}
        See 'Parameters'.
    """

    def __init__(
            self,
            column,
            segment,
            function,
            argument=None,
            scale='metric'):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Return the object dictionary."""

        return {'Feature': vars(self)}

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the check functions for the function argument
        check_argument = {
            'Dose Gradient': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=('x', 'y', 'z'))),
            'Dose Moment': (
                partial(check_type, types=str),
                partial(check_length, reference=3, sign='=='),
                check_string_is_number),
            'Dose Subvolume': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'x1of2', 'x2of2', 'x1of3', 'x2of3', 'x3of3', 'y1of2',
                    'y2of2', 'y1of3', 'y2of3', 'y3of3', 'z1of2', 'z2of2',
                    'z1of3', 'z2of3', 'z3of3'))),
            'Dx': (
                partial(check_type, types=(int, float)),
                partial(check_value, reference=0, sign='>'),
                partial(check_value, reference=100, sign='<')),
            'Vx': (
                partial(check_type, types=(int, float)),
                partial(check_value, reference=0, sign='>'),
                partial(check_value, reference=100, sign='<')),
            'other': (
                partial(check_type, types=type(None)),)}

        # Get the check map
        check_map = {
            'column': (
                partial(check_type, types=str),),
            'segment': (
                partial(check_type, types=str),),
            'function': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=tuple(maps.FEATURES))),
            'argument': check_argument[
                inputs['function'] if inputs['function'] in (
                    'Dx', 'Vx', 'Dose Gradient', 'Dose Moment',
                    'Dose Subvolume') else 'other'],
            'scale': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'metric', 'nominal', 'ordinal')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)


class StaticFeature():
    """
    Static input feature class.

    This class provides methods to set, validate and serialize a static input \
    feature for the learning models.

    Parameters
    ----------
    column : str
        Name of the data column.

    value : None, int, float or str
        Static feature value.

    scale : {'metric', 'nominal', 'ordinal'}, default='metric'
        Feature scale.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    value : None, int, float or str
        See 'Parameters'.

    scale : {'metric', 'nominal', 'ordinal'}
        See 'Parameters'.
    """

    def __init__(
            self,
            column,
            value,
            scale='metric'):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Return the object dictionary."""

        return {'Feature': vars(self)}

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the check map
        check_map = {
            'column': (
                partial(check_type, types=str),),
            'value': (
                partial(check_type, types=(int, float, str)),),
            'scale': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'metric', 'nominal', 'ordinal')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)


class Label():
    """
    Input label class.

    This class provides methods to set, validate and serialize an input label \
    for the learning models.

    Parameters
    ----------
    column : str
        Name of the data column.

    viewpoint : {'early', 'late', 'longitudinal', 'long-term', 'profile'}, \
        default='longitudinal'
        Label viewpoint for time-dependent modeling.

    time_variable : None or str, default=None
        Name of the data column for time-dependent modeling.

    bounds : list, default=[1, 1]
        Bounds for binarization of the label values.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    viewpoint : {'early', 'late', 'longitudinal', 'long-term', 'profile'}
        See 'Parameters'.

    time_variable : None or str
        See 'Parameters'.

    bounds : list
        See 'Parameters'.
    """

    def __init__(
            self,
            column,
            viewpoint='longitudinal',
            time_variable=None,
            bounds=[1, 1]):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Return the attribute dictionary."""

        return {'Label': vars(self)}

    def check(
            self,
            inputs):
        """
        Check the input arguments.

        Parameters
        ----------
        inputs : dict
            Dictionary with the mappings between argument names and values.
        """

        # Get the check map
        check_map = {
            'column': (
                partial(check_type, types=str),),
            'viewpoint': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'early', 'late', 'long-term', 'longitudinal', 'profile'))),
            'time_variable': (
                partial(check_type, types=(type(None), str)),),
            'bounds': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(type(None), int, float)))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

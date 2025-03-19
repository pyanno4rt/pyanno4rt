"""Input feature."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check import (
    check_length, check_string_is_number, check_type, check_value,
    check_value_in_set)
from pyanno4rt.learning_model.features import feature_map
from pyanno4rt.tools import filter_dict

# %% Class definition


class Feature():
    """
    Input feature class.

    This class provides methods to set, validate and serialize an input \
    feature for the learning models.

    Parameters
    ----------
    column : str
        Name of the data column.

    scale : {'metric', 'nominal', 'ordinal'}
        Feature scale.

    segment : None or str
        Segment associated with the feature.

    function : None or str
        Name of the (re)calculation function.

    argument : None, int, float or str
        Additional function argument.

    value : None, int, float or str
        Static feature value.

    Attributes
    ----------
    column : str
        See 'Parameters'.

    scale : {'metric', 'nominal', 'ordinal'}
        See 'Parameters'.

    segment : None or str
        See 'Parameters'.

    function : None or str
        See 'Parameters'.

    argument : None, int, float or str
        See 'Parameters'.

    value : None, int, float or str
        See 'Parameters'.
    """

    def __init__(
            self,
            column,
            scale,
            segment,
            function,
            argument,
            value):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Check the input arguments
        self.check(inputs)

        # Loop over the input arguments
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

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
            'Dx': (
                partial(check_type, types=(int, float)),
                partial(check_value, reference=0, sign='>'),
                partial(check_value, reference=100, sign='<')),
            'Vx': (
                partial(check_type, types=(int, float)),
                partial(check_value, reference=0, sign='>'),
                partial(check_value, reference=100, sign='<')),
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
            'other': (
                partial(check_type, types=type(None)),)}

        # Get the check map
        check_map = {
            'column': (
                partial(check_type, types=str),),
            'scale': (
                partial(check_type, types=str),
                partial(check_value_in_set, options=(
                    'metric', 'nominal', 'ordinal'))),
            'segment': (
                partial(
                    check_type, types={True: type(None), False: str},
                    type_condition=(inputs['function'] is None)),),
            'function': (
                partial(check_type, types=(type(None), str)),
                partial(check_value_in_set, options=tuple(feature_map))),
            'argument': check_argument[
                inputs['function'] if inputs['function'] in (
                    'Dx', 'Vx', 'Dose Gradient', 'Dose Moment',
                    'Dose Subvolume') else 'other'],
            'value': (
                partial(check_type, types=(type(None), int, float, str)),)}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

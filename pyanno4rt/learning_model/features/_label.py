"""Input label."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.input_check import (
    check_length, check_subtype, check_type, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


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
        for key, value in inputs.items():

            # Set the attribute
            setattr(self, key, value)

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

"""Soft decision tree tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from hyperopt.hp import choice, uniform

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

# %% Class definition


class TuneSpaceSoftDT():
    """
    Soft decision tree tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for a soft decision tree model.

    Parameters
    ----------
    criterion : None or list, default=None
        Options ('focal_loss', 'log_loss') for the weight optimization \
        criterion.

    max_depth : None or list, default=None
        Options for the maximum tree depth.

    temperature : None or list, default=None
        Range for the logit scaling parameter.

    tolerance : None or list, default=None
        Options for the criterion precision goal.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    criterion : list
        See 'Parameters'.

    max_depth : list
        See 'Parameters'.

    temperature : list
        See 'Parameters'.

    tolerance : list
        See 'Parameters'.
    """

    def __init__(
            self,
            criterion=None,
            max_depth=None,
            temperature=None,
            tolerance=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'criterion': ['log_loss'],
            'max_depth': [5],
            'temperature': [0.01, 2.0],
            'tolerance': [1e-3]
            }

        # Update the input arguments
        arguments = {
            key: value if value is not None else defaults[key]
            for key, value in arguments.items()}

        # Validate the input arguments
        self.validate(arguments)

        # Loop over the input arguments
        for item in arguments.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """
        Serialize the tune space into a dictionary.

        Returns
        -------
        dict
            Dictionary with the tune space's arguments.
        """

        return {'Soft Decision Tree': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune space from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune space's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning.spaces._tune_space_soft_dt.TuneSpaceSoftDT`
            The object used to handle the tune space parameters.
        """

        return cls(**dictionary)

    def to_space(self):
        """
        Get the search space.

        Returns
        -------
        dict
            Dictionary with the search intervals.
        """

        return {
            'criterion': choice('criterion', self.criterion),
            'max_depth': choice('max_depth', self.max_depth),
            'temperature': uniform(
                'temperature', self.temperature[0], self.temperature[1]),
            'tolerance': choice('tolerance', self.tolerance)
            }

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
            'criterion': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'focal_loss', 'log_loss'))
                ),
            'max_depth': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'temperature': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'tolerance': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

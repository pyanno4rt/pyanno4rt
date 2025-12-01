"""Naive Bayes tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_length, validate_subtype, validate_type)

# %% Class definition


class TuneSpaceNB():
    """
    Naive Bayes tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the naive Bayes model.

    Parameters
    ----------
    priors : None or list, default=None
        Options for the prior probabilities of the classes.

    var_smoothing : None or list, default=None
        Range for the portion of the largest variance of all features added \
        to variances for calculation stability.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    priors : None or list
        See 'Parameters'.

    var_smoothing : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            priors=None,
            var_smoothing=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'priors': [None] + [[i/10, 1-i/10] for i in range(1, 10)],
            'var_smoothing': [1e-12, 1]}

        # Update the input arguments with the defaults, if applicable
        inputs = {
            key: value if value is not None else defaults[key]
            for key, value in inputs.items()}

        # Validate the input arguments
        self.validate(inputs)

        # Loop over the input arguments
        for item in inputs.items():

            # Set the attribute
            setattr(self, *item)

    def to_dict(self):
        """Serialize the tune space into a dictionary."""

        return vars(self)

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune space from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune space parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_nb.TuneSpaceNB`
            The object used to handle the tune space parameters.
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
            'priors': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(list, type(None)))),
            'var_smoothing': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>'))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

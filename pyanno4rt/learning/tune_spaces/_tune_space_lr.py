"""Logistic regression tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_length, validate_subtype, validate_type, validate_value,
    validate_value_in_set)

# %% Class definition


class TuneSpaceLR():
    """
    Logistic regression tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the logistic regression model.

    Parameters
    ----------
    C : None or list, default=None
        Range for the inverse of the regularization strength.

    penalty : None or list, default=None
        Options ('l1', 'l2', 'elasticnet') for the norm of the penalty \
        function.

    tol : None or list, default=None
        Options for the stopping criteria tolerance.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    C : None or list
        See 'Parameters'.

    penalty : None or list
        See 'Parameters'.

    tol : None or list
        See 'Parameters'.

    class_weight : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            C=None,
            penalty=None,
            tol=None,
            class_weight=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'C': [2**-5, 2**10],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'tol': [1e-4, 1e-5, 1e-6],
            'class_weight': [None, 'balanced']}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_lr.TuneSpaceLR`
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
            'C': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_value, reference=0, sign='>')),
            'penalty': (
                partial(validate_type, options=list),
                partial(validate_value_in_set, options=(
                    'l1', 'l2', 'elasticnet'))),
            'tol': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_value, reference=0, sign='>')),
            'class_weight': (
                partial(validate_type, options=list),
                partial(validate_value_in_set, options=(None, 'balanced')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

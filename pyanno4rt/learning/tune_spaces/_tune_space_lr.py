"""Logistic regression tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

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
        Options for the norm of the penalty function.

    tol : None or list, default=None
        Options for the stopping criteria tolerance.

    class_weight : None or list, default=None
        Options for the weights associated with the classes.

    .. note:: If any argument is None, default values will be applied.

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

        # Loop over the inputs
        for key, value in inputs.items():

            # Check if the value is None
            if value is None:

                # Overwrite the value with the default
                inputs[key] = defaults[key]

        # Check the input arguments
        self.check(inputs)

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
            :class:`~pyanno4rt.learning.logistic._tune_space_lr.TuneSpaceLR`
            The object used to handle the tune space parameters.
        """

        return cls(**dictionary)

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
            'C': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'penalty': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'l1', 'l2', 'elasticnet'))),
            'tol': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'class_weight': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(None, 'balanced')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

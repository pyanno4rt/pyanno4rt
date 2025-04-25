"""Support vector machine tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class TuneSpaceSVM():
    """
    Support vector machine tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the support vector machine model.

    Parameters
    ----------
    C : None or list, default=None
        Range for the inverse proportional of the regularization strength.

    kernel : None or list, default=None
        Options for the kernel type.

    degree : None or list, default=None
        Options ('linear', 'poly', 'rbf', 'sigmoid') for the degree of the \
        polynomial kernel function.

    gamma : None or list, default=None
        Range for the kernel coefficient in 'rbf', 'poly' and 'sigmoid'.

    tol : None or list, default=None
        Options for the stopping criteria tolerance.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    C : None or list
        See 'Parameters'.

    kernel : None or list
        See 'Parameters'.

    degree : None or list
        See 'Parameters'.

    gamma : None or list
        See 'Parameters'.

    tol : None or list
        See 'Parameters'.

    class_weight : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            C=None,
            kernel=None,
            degree=None,
            gamma=None,
            tol=None,
            class_weight=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'C': [2**-5, 2**10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [3, 4, 5, 6],
            'gamma': [0.01, 100],
            'tol': [1e-4, 1e-5, 1e-6],
            'class_weight': [None, 'balanced']}

        # Update the input arguments with the defaults, if applicable
        inputs = {
            key: value if value is not None else defaults[key]
            for key, value in inputs.items()}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_svm.TuneSpaceSVM`
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
                partial(check_type, options=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, options=(int, float)),
                partial(check_value, reference=0, sign='>')),
            'kernel': (
                partial(check_type, options=list),
                partial(check_value_in_set, options=(
                    'linear', 'rbf', 'poly', 'sigmoid'))),
            'degree': (
                partial(check_type, options=list),
                partial(check_subtype, options=int),
                partial(check_value, reference=0, sign='>')),
            'gamma': (
                partial(check_type, options=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, options=(int, float)),
                partial(check_value, reference=0, sign='>')),
            'tol': (
                partial(check_type, options=list),
                partial(check_subtype, options=(int, float)),
                partial(check_value, reference=0, sign='>')),
            'class_weight': (
                partial(check_type, options=list),
                partial(check_value_in_set, options=(None, 'balanced')))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

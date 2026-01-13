"""Support vector machine tune space."""

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


class TuneSpaceSVM():
    """
    Support vector machine tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for a support vector machine model.

    Parameters
    ----------
    C : None or list, default=None
        Range for the inverse proportional of the regularization strength.

    kernel : None or list, default=None
        Options ('linear', 'poly', 'rbf', 'sigmoid') for the kernel type.

    degree : None or list, default=None
        Options for the degree of the polynomial kernel function.

    gamma : None or list, default=None
        Range for the kernel coefficient in 'poly', 'rbf' and 'sigmoid'.

    tol : None or list, default=None
        Options for the stopping criteria tolerance.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    C : list
        See 'Parameters'.

    kernel : list
        See 'Parameters'.

    degree : list
        See 'Parameters'.

    gamma : list
        See 'Parameters'.

    tol : list
        See 'Parameters'.

    class_weight : list
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
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'C': [0.1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': [0.001, 1],
            'tol': [1e-3, 1e-4, 1e-5],
            'class_weight': [None, 'balanced']
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

        return {'Support Vector Machine': vars(self)}

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
            :class:`~pyanno4rt.learning.tuning.spaces._tune_space_svm.TuneSpaceSVM`
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
            'C': uniform('C', self.C[0], self.C[1]),
            'kernel': choice('kernel', self.kernel),
            'degree': choice('degree', self.degree),
            'gamma': uniform('gamma', self.gamma[0], self.gamma[1]),
            'tol': choice('tol', self.tol),
            'class_weight': choice('class_weight', self.class_weight)
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
            'C': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'kernel': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'linear', 'rbf', 'poly', 'sigmoid'))
                ),
            'degree': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'gamma': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'tol': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'class_weight': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(None, 'balanced'))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

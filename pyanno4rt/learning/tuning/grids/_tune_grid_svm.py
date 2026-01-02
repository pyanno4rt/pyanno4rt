"""Support vector machine tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import chain, product

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneGridSVM():
    """
    Support vector machine tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a support vector machine model.

    Parameters
    ----------
    C : None or list, default=None
        Options for the inverse proportional of the regularization strength.

    kernel : None or list, default=None
        Options ('linear', 'poly', 'rbf', 'sigmoid') for the kernel type.

    degree : None or list, default=None
        Options for the degree of the polynomial kernel function.

    gamma : None or list, default=None
        Options for the kernel coefficient in 'poly', 'rbf' and 'sigmoid'.

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
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3, 4],
            'gamma': [0.001, 0.01, 0.1, 1],
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
        """Serialize the tune grid into a dictionary."""

        return vars(self)|{'name': 'Support Vector Machine'}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune grid from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune grid parameters.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_svm.TuneGridSVM`
            The object used to handle the tune grid parameters.
        """

        return cls(**dictionary)

    def to_grid(self):
        """
        Get the grid search proposals.

        Returns
        -------
        list
            Grid search proposals.
        """

        # Set the parameter keys
        keys = ('C', 'kernel', 'degree', 'gamma', 'tol', 'class_weight')

        # Set the parameter values
        values = list(chain(
            product(self.C, ['linear'], [3], [1], self.tol, self.class_weight),
            product(
                self.C, ['poly'], self.degree, self.gamma, self.tol,
                self.class_weight),
            product(
                self.C, ['rbf', 'sigmoid'], [3], self.gamma, self.tol,
                self.class_weight)))

        return [dict(zip(keys, value)) for value in values]

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

"""K-nearest neighbors tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import product

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneGridKNN():
    """
    K-nearest neighbors tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a k-nearest neighbors model.

    Parameters
    ----------
    n_neighbors : None or list, default=None
        Options for the number of neighbors.

    weights : None or list, default=None
        Options ('uniform', 'distance') for the weight function.

    leaf_size : None or list, default=None
        Options for the BallTree or KDTree leaf size.

    p : None or list, default=None
        Options for the power parameter in the Minkowski metric.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    n_neighbors : list
        See 'Parameters'.

    weights : list
        See 'Parameters'.

    leaf_size : list
        See 'Parameters'.

    p : list
        See 'Parameters'.
    """

    def __init__(
            self,
            n_neighbors=None,
            weights=None,
            leaf_size=None,
            p=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'n_neighbors': [1, 3, 5, 11, 21],
            'weights': ['uniform', 'distance'],
            'leaf_size': [10, 20, 50],
            'p': [1, 2, 3]
            }

        # Update the input arguments with the defaults, if applicable
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

        return vars(self)|{'name': 'K-Nearest Neighbors'}

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
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_knn.TuneGridKNN`
            The object used to handle the tune grid parameters.
        """

        return cls(**dictionary)

    def to_list(self):
        """
        Get the grid search proposals.

        Returns
        -------
        list
            Grid search proposals.
        """

        # Set the parameter keys
        keys = ('n_neighbors', 'weights', 'leaf_size', 'p')

        # Set the parameter values
        values = list(product(
            self.n_neighbors, self.weights, self.leaf_size, self.p))

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
            'n_neighbors': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>=')
                ),
            'weights': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'distance', 'uniform'))
                ),
            'leaf_size': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'p': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

"""K-nearest neighbors tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from hyperopt.hp import choice

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneSpaceKNN():
    """
    K-nearest neighbors tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for a k-nearest neighbors model.

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

        # Set the defaults
        defaults = {
            'n_neighbors': list(range(1, 21)),
            'weights': ['uniform', 'distance'],
            'leaf_size': list(range(10, 51)),
            'p': [1, 2, 3]
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

        return {'K-Nearest Neighbors': vars(self)}

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
            :class:`~pyanno4rt.learning.tuning.spaces._tune_space_knn.TuneSpaceKNN`
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
            'n_neighbors': choice('n_neighbors', self.n_neighbors),
            'weights': choice('weights', self.weights),
            'leaf_size': choice('leaf_size', self.leaf_size),
            'p': choice('p', self.p)
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

"""K-nearest neighbors tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class TuneSpaceKNN():
    """
    K-nearest neighbors tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the k-nearest neighbors model.

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
    n_neighbors : None or list
        See 'Parameters'.

    weights : None or list
        See 'Parameters'.

    leaf_size : None or list
        See 'Parameters'.

    p : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            n_neighbors=None,
            weights=None,
            leaf_size=None,
            p=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'n_neighbors': list(range(1, 11)),
            'weights': ['uniform', 'distance'],
            'leaf_size': list(range(1, 501)),
            'p': [1, 2, 3]}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_knn.TuneSpaceKNN`
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
            'n_neighbors': (
                partial(check_type, options=list),
                partial(check_subtype, options=int),
                partial(check_value, reference=0, sign='>=')),
            'weights': (
                partial(check_type, options=list),
                partial(check_value_in_set, options=('distance', 'uniform'))),
            'leaf_size': (
                partial(check_type, options=list),
                partial(check_subtype, options=int),
                partial(check_value, reference=0, sign='>')),
            'p': (
                partial(check_type, options=list),
                partial(check_subtype, options=int),
                partial(check_value, reference=0, sign='>'))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

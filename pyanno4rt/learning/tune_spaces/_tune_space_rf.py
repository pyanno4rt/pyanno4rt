"""Random forest tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.tools import filter_dict

# %% Class definition


class TuneSpaceRF():
    """
    Random forest tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the random forest.

    Parameters
    ----------
    n_estimators : None or list, default=None
        Options for the number of trees in the forest.

    criterion : None or list, default=None
        Options ('gini', 'entropy') for the split quality measure.

    max_depth : None or list, default=None
        Options for the maximum tree depth.

    min_samples_split : None or list, default=None
        Range for the minimum relative number of node splitting samples.

    min_samples_leaf : None or list, default=None
        Range for the minimum relative number of leaf samples.

    min_weight_fraction_leaf : None or list, default=None
        Range for the minimum weighted fraction of the sum of weights \
        required at each node.

    max_features : None or list, default=None
        Options for the maximum number of features considered at each split.

    bootstrap : None or list, default=None
        Options (False, True) for the bootstrap sampling indicator.

    warm_start : None or list, default=None
        Options (False, True) for the warm-start indicator.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    ccp_alpha : None or list, default=None
        Range for the complexity parameter for minimal cost-complexity pruning.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    n_estimators : None or list
        See 'Parameters'.

    criterion : None or list
        See 'Parameters'.

    max_depth : None or list
        See 'Parameters'.

    min_samples_split : None or list
        See 'Parameters'.

    min_samples_leaf : None or list
        See 'Parameters'.

    min_weight_fraction_leaf : None or list
        See 'Parameters'.

    max_features : None or list
        See 'Parameters'.

    bootstrap : None or list
        See 'Parameters'.

    warm_start : None or list
        See 'Parameters'.

    class_weight : None or list
        See 'Parameters'.

    ccp_alpha : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            n_estimators=None,
            criterion=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            min_weight_fraction_leaf=None,
            max_features=None,
            bootstrap=None,
            warm_start=None,
            class_weight=None,
            ccp_alpha=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'n_estimators': [100],
            'criterion': ['gini', 'entropy'],
            'max_depth': [5],
            'min_samples_split': [0.0, 1.0],
            'min_samples_leaf': [0.0, 0.5],
            'min_weight_fraction_leaf': [0.0, 0.5],
            'max_features': [0],
            'bootstrap': [False, True],
            'warm_start': [False],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 1.0]}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_rf.TuneSpaceRF`
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
            'n_estimators': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'criterion': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=('entropy', 'gini'))),
            'max_depth': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'min_samples_split': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'min_samples_leaf': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'min_weight_fraction_leaf': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True)),
            'max_features': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'bootstrap': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(False, True))),
            'warm_start': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(False, True))),
            'class_weight': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(None, 'balanced'))),
            'ccp_alpha': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=float),
                partial(check_value, reference=0, sign='>=', is_vector=True),
                partial(check_value, reference=1, sign='<=', is_vector=True))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

"""Random forest tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

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
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 1.0]}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_rf.TuneSpaceRF`
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
            'n_estimators': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'criterion': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=('entropy', 'gini'))
                ),
            'max_depth': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'min_samples_split': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'min_samples_leaf': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'min_weight_fraction_leaf': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'max_features': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>=')
                ),
            'bootstrap': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(False, True))
                ),
            'class_weight': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(None, 'balanced'))
                ),
            'ccp_alpha': (
                partial(validate_type, options=list),
                partial(validate_length, reference=2, sign='=='),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

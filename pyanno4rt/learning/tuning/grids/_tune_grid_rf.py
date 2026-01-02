"""Random forest tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import product

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneGridRF():
    """
    Random forest tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a random forest model.

    Parameters
    ----------
    n_estimators : None or list, default=None
        Options for the number of trees in the forest.

    criterion : None or list, default=None
        Options ('entropy', 'gini', 'log_loss') for the split quality measure.

    max_depth : None or list, default=None
        Options for the maximum tree depth.

    min_samples_split : None or list, default=None
        Options for the minimum relative number of node splitting samples.

    min_samples_leaf : None or list, default=None
        Options for the minimum relative number of leaf samples.

    min_weight_fraction_leaf : None or list, default=None
        Options for the minimum weighted fraction of the sum of weights \
        required at each node.

    max_features : None or list, default=None
        Options for the maximum number of features considered at each split.

    bootstrap : None or list, default=None
        Options (False, True) for the bootstrap sampling indicator.

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    ccp_alpha : None or list, default=None
        Options for the complexity parameter for minimal cost-complexity \
        pruning.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    n_estimators : list
        See 'Parameters'.

    criterion : list
        See 'Parameters'.

    max_depth : list
        See 'Parameters'.

    min_samples_split : list
        See 'Parameters'.

    min_samples_leaf : list
        See 'Parameters'.

    min_weight_fraction_leaf : list
        See 'Parameters'.

    max_features : list
        See 'Parameters'.

    bootstrap : list
        See 'Parameters'.

    class_weight : list
        See 'Parameters'.

    ccp_alpha : list
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
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'n_estimators': [100, 200, 500],
            'criterion': ['gini'],
            'max_depth': [5],
            'min_samples_split': [0.01, 0.01, 0.1],
            'min_samples_leaf': [0.01, 0.01, 0.1],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.1],
            'max_features': ['sqrt'],
            'bootstrap': [False, True],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 0.01, 0.1]
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

        return vars(self)|{'name': 'Random Forest'}

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
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_rf.TuneGridRF`
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
        keys = (
            'n_estimators', 'criterion', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
            'bootstrap', 'class_weight', 'ccp_alpha')

        # Set the parameter values
        values = list(product(
            self.n_estimators, self.criterion, self.max_depth,
            self.min_samples_split, self.min_samples_leaf,
            self.min_weight_fraction_leaf, self.max_features,
            self.bootstrap, self.class_weight, self.ccp_alpha))

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
            'n_estimators': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'criterion': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'entropy', 'gini', 'log_loss'))
                ),
            'max_depth': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'min_samples_split': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'min_samples_leaf': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'min_weight_fraction_leaf': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=float),
                partial(validate_item, reference=0, sign='>='),
                partial(validate_item, reference=1, sign='<=')
                ),
            'max_features': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, str)),
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

"""Decision tree tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import product

# %% Internal package import

from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneGridDT():
    """
    Decision tree tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a decision tree model.

    Parameters
    ----------
    criterion : None or list, default=None
        Options ('entropy', 'gini', 'log_loss') for the split quality measure.

    splitter : None or list, default=None
        Options ('best', 'random') for the splitting strategy at each node.

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

    class_weight : None or list, default=None
        Options (None, 'balanced') for the weights associated with the classes.

    ccp_alpha : None or list, default=None
        Options for the complexity parameter for minimal cost-complexity \
        pruning.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    criterion : list
        See 'Parameters'.

    splitter : list
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

    class_weight : list
        See 'Parameters'.

    ccp_alpha : list
        See 'Parameters'.
    """

    def __init__(
            self,
            criterion=None,
            splitter=None,
            max_depth=None,
            min_samples_split=None,
            min_samples_leaf=None,
            min_weight_fraction_leaf=None,
            max_features=None,
            class_weight=None,
            ccp_alpha=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'criterion': ['gini'],
            'splitter': ['best'],
            'max_depth': [5],
            'min_samples_split': [0.01, 0.01, 0.1],
            'min_samples_leaf': [0.01, 0.01, 0.1],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.1],
            'max_features': ['sqrt'],
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
        """
        Serialize the tune grid into a dictionary.

        Returns
        -------
        dict
            Dictionary with the tune grid's arguments.
        """

        return {'Decision Tree': vars(self)}

    @classmethod
    def from_dict(
            cls,
            dictionary):
        """
        Deserialize the tune grid from a dictionary.

        Parameters
        ----------
        dictionary : dict
            Dictionary with the tune grid's arguments.

        Returns
        -------
        object of class \
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_dt.TuneGridDT`
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
            'criterion', 'splitter', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features',
            'class_weight', 'ccp_alpha')

        # Set the parameter values
        values = list(product(
            self.criterion, self.splitter, self.max_depth,
            self.min_samples_split, self.min_samples_leaf,
            self.min_weight_fraction_leaf, self.max_features,
            self.class_weight, self.ccp_alpha))

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
            'criterion': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'entropy', 'gini', 'log_loss'))
                ),
            'splitter': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=('best', 'random'))
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

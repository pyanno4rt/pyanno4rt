"""Neural network tune grid."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from itertools import chain, product

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_subtype, validate_type)

# %% Class definition


class TuneGridNN():
    """
    Neural network tune grid class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune grid for a neural network model.

    Parameters
    ----------
    max_hidden_layers : int, default=2
        Maximum number of hidden layers.

    hidden_neuron_number : None or list, default=None
        Options for the number of hidden layer neurons.

    hidden_activation : None or list, default=None
        Options ('elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax', \
        'softplus', 'swish') for the hidden layer activation function.

    hidden_dropout_rate : None or list, default=None
        Options for the hidden layer dropout rate.

    learning_rate : None or list, default=None
        Options for the learning rate.

    optimizer : None or list, default=None
        Options ('adam', 'ftrl', 'sgd') for the network optimizer.

    loss : None or list, default=None
        Options ('binary_crossentropy', 'binary_focal_crossentropy', \
        'kl_divergence') for the network loss function.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    max_hidden_layers : int
        See 'Parameters'.

    hidden_neuron_number : list
        See 'Parameters'.

    hidden_activation : list
        See 'Parameters'.

    hidden_dropout_rate : list
        See 'Parameters'.

    learning_rate : list
        See 'Parameters'.

    optimizer : list
        See 'Parameters'.

    loss : list
        See 'Parameters'.
    """

    def __init__(
            self,
            max_hidden_layers=2,
            hidden_neuron_number=None,
            hidden_activation=None,
            hidden_dropout_rate=None,
            learning_rate=None,
            optimizer=None,
            loss=None):

        # Get the input arguments
        arguments = filter_dict(vars(), remove_keys=('self',))

        # Set the defaults
        defaults = {
            'hidden_neuron_number': [4, 8, 16, 32],
            'hidden_activation': ['gelu', 'leaky_relu', 'relu'],
            'hidden_dropout_rate': [0.0, 0.2, 0.5],
            'learning_rate': [1e-3, 1e-2],
            'optimizer': ['adam'],
            'loss': ['binary_crossentropy']
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

        return {'Neural Network': vars(self)}

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
            :class:`~pyanno4rt.learning.tuning.grids._tune_grid_nn.TuneGridNN`
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
            'hidden_layer_number', 'hidden_neuron_number', 'hidden_activation',
            'hidden_dropout', 'learning_rate', 'optimizer', 'loss')

        # Set the parameter values
        values = list(chain(*[product(
            [m+1],
            [list(item) for item in product(
                self.hidden_neuron_number, repeat=m+1)],
            [list(item) for item in product(
                self.hidden_activation, repeat=m+1)],
            [list(item) for item in product(
                self.hidden_dropout_rate, repeat=m+1)],
            self.learning_rate, self.optimizer, self.loss)
            for m in range(self.max_hidden_layers)]))

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
            'max_hidden_layers': (
                partial(validate_type, options=int),
                partial(validate_item, reference=1, sign='>=')
                ),
            'hidden_neuron_number': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
                ),
            'hidden_activation': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                    'softplus', 'swish'))
                ),
            'hidden_dropout_rate': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>=')
                ),
            'learning_rate': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=(int, float)),
                partial(validate_item, reference=0, sign='>')
                ),
            'optimizer': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(
                    *maps.NETWORK_OPTIMIZERS,))
                ),
            'loss': (
                partial(validate_type, options=list),
                partial(validate_item_in_set, options=(*maps.NETWORK_LOSSES,))
                )
            }

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the validation functions
            for function in validation_map[key]:

                # Run the validation function
                function(key, value)

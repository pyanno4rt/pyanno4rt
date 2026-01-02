"""Neural network tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial
from hyperopt.hp import choice, uniform

# %% Internal package import

import pyanno4rt.learning._maps as maps
from pyanno4rt.tools import filter_dict
from pyanno4rt.validation import (
    validate_item, validate_item_in_set, validate_length, validate_subtype,
    validate_type)

# %% Class definition


class TuneSpaceNN():
    """
    Neural network tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for a neural network model.

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
        Range for the learning rate.

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
        """Serialize the tune space into a dictionary."""

        return vars(self)|{'name': 'Neural Network'}

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
            :class:`~pyanno4rt.learning.tuning.spaces._tune_space_nn.TuneSpaceNN`
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
            'hidden_layers': choice(
                'hidden_layers', [
                    {'hidden_layer_number': n+1,
                     'hidden_neuron_number': [
                         choice(
                             f'{n+1}L{m+1}_neuron_number',
                             self.hidden_neuron_number)
                         for m in range(n+1)],
                     'hidden_activation': [
                         choice(
                             f'{n+1}L{m+1}_activation', self.hidden_activation)
                         for m in range(n+1)],
                     'hidden_dropout_rate': [
                         choice(
                             f'{n+1}L{m+1}_hidden_dropout',
                             self.hidden_dropout_rate)
                         for m in range(n+1)]}
                    for n in range(self.max_hidden_layers)]),
            'learning_rate': uniform(
                'learning_rate', self.learning_rate[0], self.learning_rate[1]),
            'optimizer': choice('optimizer', self.optimizer),
            'loss': choice('loss', self.loss)
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
                partial(validate_length, reference=2, sign='=='),
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

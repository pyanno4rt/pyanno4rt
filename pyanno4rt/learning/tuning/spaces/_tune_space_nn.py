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
    hyperparameter tune space for the neural network model.

    Parameters
    ----------
    max_hidden_layers : int, default=2
        Maximum number of hidden layers for a neural network model.

    hidden_neuron_number : None or list, default=None
        Options for the number of hidden layer neurons.

    hidden_activation : None or list, default=None
        Options ('elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
        'softplus', 'swish') for the hidden layer activation functions.

    hidden_dropout_rate : None or list, default=None
        Options for the hidden layer dropout rates.

    batch_size : None or list, default=None
        Options for the batch size.

    learning_rate : None or list, default=None
        Range for the learning rate.

    optimizer : None or list, default=None
        Options ('Adam', 'Ftrl', 'SGD') for the network optimization algorithm.

    loss : None or list, default=None
        Options ('BCE', 'FocalBCE', 'KLD') for the network optimization loss \
        function.

    .. note:: If arguments are passed as None, default values will be applied.

    Attributes
    ----------
    hidden_neuron_number : None or list
        See 'Parameters'.

    hidden_activation : None or list
        See 'Parameters'.

    hidden_dropout_rate : None or list
        See 'Parameters'.

    batch_size : None or list
        See 'Parameters'.

    learning_rate : None or list
        See 'Parameters'.

    optimizer : None or list
        See 'Parameters'.

    loss : None or list
        See 'Parameters'.
    """

    def __init__(
            self,
            max_hidden_layers=2,
            hidden_neuron_number=None,
            hidden_activation=None,
            hidden_dropout_rate=None,
            batch_size=None,
            learning_rate=None,
            optimizer=None,
            loss=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'hidden_neuron_number': [2**x for x in range(1, 12)],
            'hidden_activation': [
                'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                'softplus', 'swish'],
            'hidden_dropout_rate': [0.0, 0.1, 0.25, 0.5, 0.75],
            'batch_size': [4, 8, 16, 32],
            'learning_rate': [1e-5, 1e-2],
            'optimizer': list(maps.NETWORK_OPTIMIZERS),
            'loss': list(maps.NETWORK_LOSSES)}

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

        return {'Neural Network': vars(self)}

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_nn.TuneSpaceNN`
            The object used to handle the tune space parameters.
        """

        return cls(**dictionary)

    def to_hyperopt(self):
        """
        Get the hyperopt search space.

        Returns
        -------
        dict
            Dictionary with the hyperopt search intervals.
        """

        # Get the hyperopt search space
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
            'batch_size': choice('batch_size', self.batch_size),
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
            'batch_size': (
                partial(validate_type, options=list),
                partial(validate_subtype, options=int),
                partial(validate_item, reference=0, sign='>')
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

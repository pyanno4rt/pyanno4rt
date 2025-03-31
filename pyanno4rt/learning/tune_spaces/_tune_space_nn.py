"""Neural network tune space."""

# Author: Tim Ortkamp

# %% External package import

from functools import partial

# %% Internal package import

from pyanno4rt.checking import (
    check_length, check_subtype, check_type, check_value, check_value_in_set)
from pyanno4rt.learning._maps import NN_LOSSES, NN_OPTS
from pyanno4rt.tools import filter_dict

# %% Class definition


class TuneSpaceNN():
    """
    Neural network tune space class.

    This class provides methods to set, validate and serialize a \
    hyperparameter tune space for the neural network model.

    Parameters
    ----------
    input_neuron_number : None or list, default=None
        Options for the number of input layer neurons.

    input_activation : None or list, default=None
        Options for the input layer activation function.

    hidden_neuron_number : None or list, default=None
        Options for the number of hidden layer neurons.

    hidden_activation : None or list, default=None
        Options for the hidden layer activation functions.

    input_dropout_rate : None or list, default=None
        Range for the input layer dropout rate.

    hidden_dropout_rate : None or list, default=None
        Range for the hidden layer dropout rates.

    batch_size : None or list, default=None
        Options for the batch size.

    learning_rate : None or list, default=None
        Range for the learning rate.

    optimizer : None or list, default=None
        Options for the network optimization algorithm.

    loss : None or list, default=None
        Options for the network optimization loss function.

    .. note:: If any argument is None, default values will be applied.

    Attributes
    ----------
    input_neuron_number : None or list
        See 'Parameters'.

    input_activation : None or list
        See 'Parameters'.

    hidden_neuron_number : None or list
        See 'Parameters'.

    hidden_activation : None or list
        See 'Parameters'.

    input_dropout_rate : None or list
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
            input_neuron_number=None,
            input_activation=None,
            hidden_neuron_number=None,
            hidden_activation=None,
            input_dropout_rate=None,
            hidden_dropout_rate=None,
            batch_size=None,
            learning_rate=None,
            optimizer=None,
            loss=None):

        # Get the input arguments
        inputs = filter_dict(vars(), remove_keys=('self',))

        # Set the default argument values
        defaults = {
            'input_neuron_number': [2**x for x in range(1, 12)],
            'input_activation': [
                'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                'softplus', 'swish'],
            'hidden_neuron_number': [2**x for x in range(1, 12)],
            'hidden_activation': [
                'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                'softplus', 'swish'],
            'input_dropout_rate': [0.0, 0.1, 0.25, 0.5, 0.75],
            'hidden_dropout_rate': [0.0, 0.1, 0.25, 0.5, 0.75],
            'batch_size': [4, 8, 16, 32],
            'learning_rate': [1e-5, 1e-2],
            'optimizer': list(NN_OPTS),
            'loss': list(NN_LOSSES)}

        # Loop over the inputs
        for key, value in inputs.items():

            # Check if the value is None
            if value is None:

                # Overwrite the value with the default
                inputs[key] = defaults[key]

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
            :class:`~pyanno4rt.learning.tune_spaces._tune_space_nn.TuneSpaceNN`
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
            'input_neuron_number': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'input_activation': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                    'softplus', 'swish'))),
            'hidden_neuron_number': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'hidden_activation': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=(
                    'elu', 'gelu', 'leaky_relu', 'linear', 'relu', 'softmax',
                    'softplus', 'swish'))),
            'input_dropout_rate': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'hidden_dropout_rate': (
                partial(check_type, types=list),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>=', is_vector=True)),
            'batch_size': (
                partial(check_type, types=list),
                partial(check_subtype, types=int),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'learning_rate': (
                partial(check_type, types=list),
                partial(check_length, reference=2, sign='=='),
                partial(check_subtype, types=(int, float)),
                partial(check_value, reference=0, sign='>', is_vector=True)),
            'optimizer': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=tuple(NN_OPTS))),
            'loss': (
                partial(check_type, types=list),
                partial(check_value_in_set, options=tuple(NN_LOSSES)))}

        # Loop over the dictionary items
        for key, value in inputs.items():

            # Loop over the check functions
            for function in check_map[key]:

                # Run the check function
                function(key, value)

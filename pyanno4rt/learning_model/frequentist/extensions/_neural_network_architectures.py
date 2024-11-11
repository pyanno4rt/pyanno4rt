"""Neural network architectures."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import log
from tensorflow.keras import Input, Model
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout

# %% Build functions


def build_vanilla_iocnn(
        input_shape,
        output_shape,
        bias,
        hyperparameters,
        squash_output):
    """
    Build the vanilla input-output convex neural network architecture.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    squash_output : bool
        Indicator for the squashing of the network output.

    Returns
    -------
    object of class 'Functional'
        The object used to represent the prediction model.
    """

    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the input layer
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['input_dropout_rate'])(hidden)
    hidden = Dense(
        units=hyperparameters['input_neuron_number'],
        activation=hyperparameters['input_activation'])(hidden)

    # Loop over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layer
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer],
            kernel_constraint=non_neg())(hidden)

    # Check if the network output should be squashed
    if squash_output:

        # Apply the custom output activation
        activation = hyperparameters['output_activation']

    else:

        # Apply no output activation
        activation = None

    # Define the output layer
    outputs = Dense(
        units=output_shape, activation=activation, kernel_constraint=non_neg(),
        bias_initializer=Constant(log(bias)))(hidden)

    return Model(inputs, outputs)


def build_vanilla_nn(
        input_shape,
        output_shape,
        bias,
        hyperparameters,
        squash_output):
    """
    Build the vanilla neural network architecture.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    squash_output : bool
        Indicator for the squashing of the network output.

    Returns
    -------
    object of class 'Functional'
        The object used to represent the prediction model.
    """

    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the input layer
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['input_dropout_rate'])(hidden)
    hidden = Dense(
        units=hyperparameters['input_neuron_number'],
        activation=hyperparameters['input_activation'])(hidden)

    # Loop over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layer
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer])(hidden)

    # Check if the network output should be squashed
    if squash_output:

        # Apply the custom output activation
        activation = hyperparameters['output_activation']

    else:

        # Apply no output activation
        activation = None

    # Define the output layer
    outputs = Dense(
        units=output_shape, activation=activation,
        bias_initializer=Constant(log(bias)))(hidden)

    return Model(inputs, outputs)

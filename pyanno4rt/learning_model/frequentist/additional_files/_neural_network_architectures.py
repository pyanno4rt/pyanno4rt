"""Neural network architectures."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import log
from tensorflow.keras import Input, Model
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout)
from tensorflow.keras.constraints import non_neg

# %% Build functions


def build_iocnn(
        input_shape,
        output_shape,
        labels,
        hyperparameters,
        squash_output):
    """
    Build the input-output convex neural network architecture with the \
    functional API.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the hyperparameter names and values for the \
        neural network outcome prediction model.

    squash_output : bool
        Indicator for the use of a sigmoid activation function in the \
        output layer.

    Returns
    -------
    object of class 'Functional'
        Instance of the class `Functional`, which provides a functional \
        input-output convex neural network architecture.
    """
    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the input layer by dropout and dense layers
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['input_dropout_rate'])(hidden)
    hidden = Dense(
        units=hyperparameters['input_neuron_number'],
        activation=hyperparameters['input_activation'])(hidden)

    # Iterate over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layers by dropout and dense layers
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(
            hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer],
            kernel_constraint=non_neg())(hidden)

    # Check if the network output should be squashed
    if squash_output:
        activation = hyperparameters['output_activation']
    else:
        activation = None

    # Define the output layer by a dense layer
    outputs = Dense(
        units=output_shape,
        activation=activation,
        kernel_constraint=non_neg(),
        bias_initializer=Constant(
            log(sum(labels == 1)
                / sum(labels == 0))))(hidden)

    return Model(inputs, outputs)


def build_standard_nn(
        input_shape,
        output_shape,
        labels,
        hyperparameters,
        squash_output):
    """
    Build the standard neural network architecture with the functional API.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the hyperparameter names and values for the \
        neural network outcome prediction model.

    squash_output : bool
        Indicator for the use of a sigmoid activation function in the \
        output layer.

    Returns
    -------
    object of class 'Functional'
        Instance of the class `Functional`, which provides a functional \
        standard neural network architecture.
    """
    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the input layer by normalization, dropout and dense layers
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['input_dropout_rate'])(hidden)
    hidden = Dense(
        units=hyperparameters['input_neuron_number'],
        activation=hyperparameters['input_activation'])(hidden)

    # Iterate over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layers by dropout and dense layers
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(
            hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer])(
                hidden)

    # Check if the network output should be squashed
    if squash_output:
        activation = hyperparameters['output_activation']
    else:
        activation = None

    # Define the output layer by a dense layer
    outputs = Dense(
        units=output_shape,
        activation=activation,
        bias_initializer=Constant(
            log(sum(labels == 1)
                / sum(labels == 0))))(hidden)

    return Model(inputs, outputs)

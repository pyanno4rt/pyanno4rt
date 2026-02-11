"""Feed-forward neural network architectures."""

# Author: Tim Ortkamp

# %% External package import

from tensorflow.keras import Input, Model
from tensorflow.keras.backend import clear_session
from tensorflow.keras.constraints import non_neg
from tensorflow.keras.layers import (
    Activation, BatchNormalization, Dense, Dropout)
from tensorflow.keras.regularizers import l2

# %% Function definitions


def build_fnn(
        input_shape,
        output_shape,
        hyperparameters):
    """
    Build a standard neural network.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    Returns
    -------
    object of class :class:`~tensorflow.keras.Model`
        The object used to represent the model architecture.
    """

    # Clear the session cache
    clear_session()

    # Initialize the input layer
    inputs = Input((input_shape,), name='input')

    # Set the "anchor" variable
    x = inputs

    # Loop over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layer
        x = Dense(
            hyperparameters['hidden_neuron_number'][layer],
            use_bias=False,
            kernel_regularizer=l2(0.01),
            name=f'dense_hidden_{layer}')(x)
        x = BatchNormalization(name=f'batch_norm_hidden_{layer}')(x)
        x = Activation(
            hyperparameters['hidden_activation'][layer],
            name=f'activation_hidden_{layer}')(x)
        x = Dropout(
            hyperparameters['hidden_dropout_rate'][layer],
            name=f'dropout_hidden_{layer}')(x)

    # Define the output layer
    outputs = Dense(
        output_shape,
        activation=hyperparameters['output_activation'],
        kernel_regularizer=l2(0.01),
        name='output')(x)

    return Model(inputs, outputs)


def build_icnn(
        input_shape,
        output_shape,
        hyperparameters):
    """
    Build an input-convex neural network.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    Returns
    -------
    object of class :class:`~tensorflow.keras.Model`
        The object used to represent the model architecture.
    """

    # Clear the session cache
    clear_session()

    # Initialize the input layer
    inputs = Input((input_shape,), name='input')

    # Set the "anchor" variable
    x = inputs

    # Loop over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']):

        # Define the hidden layer
        x = Dense(
            hyperparameters['hidden_neuron_number'][layer],
            use_bias=False,
            kernel_regularizer=l2(0.01),
            kernel_constraint=non_neg(),
            name=f'dense_hidden_{layer}')(x)
        x = BatchNormalization(name=f'batch_norm_hidden_{layer}')(x)
        x = Activation(
            hyperparameters['hidden_activation'][layer],
            name=f'activation_hidden_{layer}')(x)
        x = Dropout(
            hyperparameters['hidden_dropout_rate'][layer],
            name=f'dropout_hidden_{layer}')(x)

    # Define the output layer
    outputs = Dense(
        output_shape,
        activation=hyperparameters['output_activation'],
        kernel_constraint=non_neg(),
        name='output')(x)

    return Model(inputs, outputs)

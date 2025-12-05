"""Neural network architectures."""

# Author: Tim Ortkamp

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
        hyperparameters):
    """
    Build the vanilla input-output convex neural network architecture.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    bias : int or float
        Initial network bias.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    Returns
    -------
    object of class 'Functional'
        The object used to represent the prediction model.
    """

    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the first hidden layer
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['hidden_dropout_rate'][0])(hidden)
    hidden = Dense(
        units=hyperparameters['hidden_neuron_number'][0],
        activation=hyperparameters['hidden_activation'][0])(hidden)

    # Loop over the number of additional hidden layers
    for layer in range(hyperparameters['hidden_layer_number']-1):

        # Define the hidden layer
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer],
            kernel_constraint=non_neg())(hidden)

    # Define the output layer
    outputs = Dense(
        units=output_shape, activation=hyperparameters['output_activation'],
        kernel_constraint=non_neg(), bias_initializer=Constant(log(bias))
        )(hidden)

    return Model(inputs, outputs)


def build_vanilla_nn(
        input_shape,
        output_shape,
        bias,
        hyperparameters):
    """
    Build the vanilla neural network architecture.

    Parameters
    ----------
    input_shape : int
        Shape of the input features.

    output_shape : int
        Shape of the output labels.

    bias : int or float
        Initial network bias.

    hyperparameters : dict
        Dictionary with the values of the hyperparameters.

    Returns
    -------
    object of class 'Functional'
        The object used to represent the prediction model.
    """

    # Initialize the network input
    inputs = Input((input_shape,), name='input')

    # Define the input layer
    hidden = BatchNormalization()(inputs)
    hidden = Dropout(hyperparameters['hidden_dropout_rate'][0])(hidden)
    hidden = Dense(
        units=hyperparameters['hidden_neuron_number'][0],
        activation=hyperparameters['hidden_activation'][0])(hidden)

    # Loop over the number of hidden layers
    for layer in range(hyperparameters['hidden_layer_number']-1):

        # Define the hidden layer
        hidden = BatchNormalization()(hidden)
        hidden = Dropout(hyperparameters['hidden_dropout_rate'][layer])(hidden)
        hidden = Dense(
            units=hyperparameters['hidden_neuron_number'][layer],
            activation=hyperparameters['hidden_activation'][layer])(hidden)

    # Define the output layer
    outputs = Dense(
        units=output_shape, activation=hyperparameters['output_activation'],
        bias_initializer=Constant(log(bias)))(hidden)

    return Model(inputs, outputs)

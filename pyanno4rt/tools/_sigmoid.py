"""Sigmoid function calculation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import exp

# %% Function definition


def sigmoid(value, multiplier=1, summand=0):
    """
    Calculate the sigmoid function value(s).

    Parameters
    ----------
    value : int, float, tuple or list
        Value(s) at which to calculate the sigmoid function.

    multiplier : int or float, default=1
        Multiplicative coefficient in the linear term.

    summand : int or float, default=0
        Additive coefficient in the linear term.

    Returns
    -------
    float or tuple
        Value(s) of the sigmoid function.
    """

    # Check if the value is a tuple or a list
    if isinstance(value, (tuple, list)):

        # Return the sigmoid function values
        return tuple(1/(1 + exp(-(multiplier*val + summand))) for val in value)

    # Return the sigmoid function value
    return 1/(1 + exp(-(multiplier*value + summand)))

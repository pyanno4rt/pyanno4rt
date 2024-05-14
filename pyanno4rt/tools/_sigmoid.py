"""Sigmoid function calculation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import exp

# %% Function definition


def sigmoid(value, multiplier=1, summand=0):
    """
    Calculate the sigmoid function value.

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

    # Check if the passed value is tuple or a list
    if isinstance(value, (tuple, list)):

        return tuple(1/(1 + exp(-multiplier*val + summand)) for val in value)

    return 1/(1 + exp(-multiplier*value + summand))

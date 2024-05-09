"""Inverse sigmoid function calculation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import inf
from numpy import log

# %% Function definition


def inverse_sigmoid(value, multiplier=1, summand=0):
    """
    Calculate the inverse sigmoid function value.

    Parameters
    ----------
    value : int, float, tuple or list
        Value(s) at which to calculate the inverse sigmoid function.

    multiplier : int or float, default=1
        Multiplicative coefficient in the linear term.

    summand : int or float, default=0
        Additive coefficient in the linear term.

    Returns
    -------
    float or tuple
        Value(s) of the inverse sigmoid function.
    """

    # Check if the passed value is a tuple or a list
    if isinstance(value, (tuple, list)):

        return tuple((log(val/(1-val))-summand)/multiplier if val != 1 else inf
                     for val in value)

    return (log(value/(1-value))-summand)/multiplier if value != 1 else inf

"""Inverse sigmoid function calculation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import log

# %% Function definition


def inverse_sigmoid(value, multiplier, summand):
    """
    Calculate the inverse sigmoid function value.

    Parameters
    ----------
    value : int, float, tuple or list
        Value(s) at which to calculate the inverse sigmoid function.

    multiplier : int or float
        Multiplicative coefficient in the linear term.

    summand : int or float
        Additive coefficient in the linear term.

    Returns
    -------
    float or tuple
        Value(s) of the inverse sigmoid function.
    """

    # Check if the passed value is a tuple or a list
    if isinstance(value, (tuple, list)):

        return tuple((log(val/(1-val))-summand)/multiplier for val in value)

    return (log(value/(1-value))-summand)/multiplier

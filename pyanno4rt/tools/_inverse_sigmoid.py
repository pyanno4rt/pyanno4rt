"""Inverse sigmoid computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import log

# %% Function definition


def inverse_sigmoid(value, multiplier, summand):
    """
    Compute the inverse sigmoid function value.

    Parameters
    ----------
    value : int or float, or tuple/list of int or float
        Value(s) at which to compute the inverse sigmoid function.

    Returns
    -------
    float or tuple of floats
        Value(s) of the inverse sigmoid function.
    """

    # Check if the passed value is a tuple or a list
    if isinstance(value, (tuple, list)):

        return tuple((log(val/(1-val))-summand)/multiplier for val in value)

    return (log(value/(1-value))-summand)/multiplier

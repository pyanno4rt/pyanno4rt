"""Soft affine linear unit (SALU) function calculation."""

# Author: Tim Ortkamp

# %% External package import

from numpy import exp

# %% Function definition


def salu(value, multiplier=1, summand=0, sign=1):
    """
    Calculate the SALU function value(s).

    Parameters
    ----------
    value : int, float, tuple or list
        Value(s) at which to calculate the SALU function.

    multiplier : int or float, default=1
        Multiplicative coefficient in the linear term.

    summand : int or float, default=0
        Additive coefficient in the linear term.

    sign : {-1, 1}, default=1
        Indicator for the orientation at the break point. If -1, the affine \
        linear segment is below, if +1, it is above the break point.

    Returns
    -------
    float or tuple
        Value(s) of the SALU function.
    """

    # Check if the value is a tuple or a list
    if isinstance(value, (tuple, list)):

        # Return the SALU function values
        return tuple(
            0.25*(multiplier*val+summand)+0.5
            if sign*(multiplier*val + summand) > 0
            else 1/(1 + exp(-(multiplier*val + summand))) for val in value)

    # Return the SALU function value
    return (
        0.25*(multiplier*value + summand)+0.5
        if sign*(multiplier*value + summand) > 0
        else 1/(1 + exp(-(multiplier*value + summand))))

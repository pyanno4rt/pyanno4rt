"""Inverse soft affine linear unit (SALU) function calculation."""

# Author: Tim Ortkamp

# %% External package import

from math import inf
from numpy import log

# %% Function definition


def inverse_salu(value, multiplier=1, summand=0, sign=1):
    """
    Calculate the inverse SALU function value(s).

    Parameters
    ----------
    value : int, float, tuple or list
        Value(s) at which to calculate the inverse SALU function.

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
        Value(s) of the inverse SALU function.
    """

    # Check if the value is a tuple or a list
    if isinstance(value, (tuple, list)):

        # Return the inverse SALU function values
        return tuple(
            (4*val-2-summand)/multiplier if sign*val > sign*0.5
            else (log(val/(1-val))-summand)/multiplier if val != 1*(sign == -1)
            else -sign*inf for val in value)

    # Return the inverse SALU function value
    return (
        (4*value-2-summand)/multiplier if sign*value > sign*0.5
        else (log(value/(1-value))-summand)/multiplier
        if value != 1*(sign == -1) else -sign*inf)

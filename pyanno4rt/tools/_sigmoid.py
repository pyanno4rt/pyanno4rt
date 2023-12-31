"""Sigmoid computation."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import exp

# %% Function definition


def sigmoid(value, multiplier, summand):
    """
    Compute the sigmoid function value.

    Parameters
    ----------
    value : int or float, or tuple/list of int or float
        Value(s) at which to compute the sigmoid function.

    coeff_A : int or float
        Multiplicate coefficient for the value in the linear predictor term.

    coeff_B : int or float
        Additive coefficient for the value in the linear predictor term.

    Returns
    -------
    float or tuple of floats
        Value(s) of the sigmoid function.
    """

    # Check if the passed value is tuple or a list
    if isinstance(value, (tuple, list)):

        return tuple(1/(1 + exp(-multiplier*val + summand)) for val in value)

    return 1/(1 + exp(-multiplier*value + summand))

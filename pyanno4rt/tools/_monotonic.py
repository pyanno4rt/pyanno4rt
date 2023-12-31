"""Monotonicity check."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import non_increasing
from pyanno4rt.tools import non_decreasing

# %% Function definition


def monotonic(array):
    """
    Check if an array is monotonic.

    Parameters
    ----------
    array : ndarray
        One-dimensional array to be checked.

    Returns
    -------
    bool
        Indicator for the monotonicity of the array.
    """

    return non_increasing(array) or non_decreasing(array)

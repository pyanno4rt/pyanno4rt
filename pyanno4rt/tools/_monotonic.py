"""Monotonicity testing."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Internal package import

from pyanno4rt.tools import non_decreasing, non_increasing

# %% Function definition


def monotonic(array):
    """
    Test whether an array is monotonic.

    Parameters
    ----------
    array : ndarray
        One-dimensional input array.

    Returns
    -------
    bool
        Indicator for the monotonicity of the array.
    """

    return non_increasing(array) or non_decreasing(array)

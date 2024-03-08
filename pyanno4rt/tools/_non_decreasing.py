"""Non-decrease testing."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def non_decreasing(array):
    """
    Test whether an array is non-decreasing.

    Parameters
    ----------
    array : ndarray
        One-dimensional input array.

    Returns
    -------
    bool
        Indicator for the non-decrease of the array.
    """

    # Convert the array to a list
    lst = array.tolist()

    return all(pair[0] <= pair[1] for pair in zip(lst, lst[1:]))

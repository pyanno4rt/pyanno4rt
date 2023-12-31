"""Non-increase check."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def non_increasing(array):
    """
    Check if an array is non-increasing.

    Parameters
    ----------
    array : ndarray
        One-dimensional array to be checked.

    Returns
    -------
    bool
        Indicator for the non-increase of the array.
    """

    # Convert the array to a list
    lst = array.tolist()

    return all(elem_1 >= elem_2 for elem_1, elem_2 in zip(lst, lst[1:]))

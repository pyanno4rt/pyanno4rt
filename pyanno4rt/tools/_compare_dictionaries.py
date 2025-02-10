"""Dictionary comparison."""

# Author: Tim Ortkamp

# %% External package import

from numpy import array_equal, ndarray

# %% Internal package import

from pyanno4rt.tools import filter_dict

# %% Function definition


def compare_dictionaries(reference_dict, compare_dict, ignore_keys=None):
    """
    Compare two dictionaries by their keys and values (including numpy arrays).

    Parameters
    ----------
    reference_dict : dict
        Reference dictionary.

    compare_dict : dict
        Dictionary for the comparison.

    ignore_keys : list or tuple
        Names of the keys to be excluded from comparison.

    Returns
    -------
    bool
        Indicator for the equality of the dictionaries.
    """

    # Check if any keys should be ignored
    if ignore_keys is not None:

        # Filter the reference dictionary
        reference_dict = filter_dict(reference_dict, remove_keys=ignore_keys)

        # Filter the comparison dictionary
        compare_dict = filter_dict(compare_dict, remove_keys=ignore_keys)

    # Check if the dictionary keys are not equal
    if reference_dict.keys() != compare_dict.keys():

        # Return false
        return False

    # Loop over the pairwise dictionary values
    for reference, compare in zip(
            reference_dict.values(), compare_dict.values()):

        # Check if the types are not equal
        if not isinstance(reference, type(compare)):

            # Return false
            return False

        # Check if the value is a numpy array
        if isinstance(reference, ndarray):

            # Check if the arrays are not equal
            if not array_equal(reference, compare):

                # Return false
                return False

        # Else, check if the non-array values are not equal
        elif reference != compare:

            # Return false
            return False

    # Else, return true
    return True

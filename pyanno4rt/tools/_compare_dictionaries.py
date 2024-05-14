"""Dictionary comparison."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import array_equal, ndarray

# %% Function definition


def compare_dictionaries(reference_dict, compare_dict):
    """
    Compare two dictionaries by their keys and values (including numpy arrays).

    Parameters
    ----------
    reference_dict : dict
        Reference dictionary.

    compare_dict : dict
        Dictionary for the comparison.

    Returns
    -------
    bool
        Indicator for the equality of the dictionaries.
    """

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

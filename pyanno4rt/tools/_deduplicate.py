"""Deduplicate indexing via dictionary."""

# Author: Tim Ortkamp

# %% Function definition


def deduplicate(iterable):
    """
    Convert an iterable to a dictionary with the elements (keys) and their \
    indices (values).

    Parameters
    ----------
    elements : iterable
        Iterable over which to loop.

    Returns
    -------
    dict
        Dictionary with the element-index pairs.
    """

    # Initialize the mapping dictionary
    dictionary = {}

    # Loop over all elements in the iterable
    for index, element in enumerate(iterable):

        # Check if the element is already a key
        if element in dictionary:

            # Add the index
            dictionary[element] += (index,)

        else:

            # Create a new key and add the index
            dictionary[element] = [index]

    return dictionary

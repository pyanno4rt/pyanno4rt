"""Deduplicating indexing via dictionary."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def deduplicate(elements):
    """
    Convert an iterable to a dictionary with index tuple for each element.

    Parameters
    ----------
    elements : iterable
        Iterable over which to loop.

    Returns
    -------
    dict
        Dictionary with the element-indices pairs.
    """

    # Initialize the mapping dictionary
    mapping = {}

    # Set the starting index to zero
    index = 0

    # Loop over all elements in the iterable
    for element in elements:

        # Check if the element is already a dictionary key
        if element in mapping:

            # Add the index to the values
            mapping[element] += (index,)

        else:

            # Create a new key and initialize the index list
            mapping[element] = [index]

        # Increment the index
        index += 1

    return {key: list(value) for key, value in mapping.items()}

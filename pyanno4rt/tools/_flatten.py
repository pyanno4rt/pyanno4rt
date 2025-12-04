"""Iterable flattening."""

# Author: Tim Ortkamp

# %% External package import

from collections.abc import Iterable

# %% Function definition


def flatten(iterable):
    """
    Flatten a nested iterable.

    Parameters
    ----------
    iterable : iterable
        (Nested) iterable to be flattened.

    Returns
    -------
    generator
        Flattened iterable with all atomic elements.
    """

    # Loop over the elements of the iterable
    for element in iterable:

        # Check if the element is an iterable
        if (isinstance(element, Iterable)
                and not isinstance(element, (str, bytes))):

            # Recursively flatten the element
            yield from flatten(element)

        else:

            # Return the element
            yield element

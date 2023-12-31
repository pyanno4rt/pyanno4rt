"""Iterable flattening."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from collections.abc import Iterable

# %% Function definition


def flatten(iterable):
    """
    Convert a nested iterable to a flat one.

    Parameters
    ----------
    iterable : iterable
        (Nested) iterable to be flattened.

    Returns
    -------
    generator
        Generator object with the flattened iterable values.
    """

    # Loop over the elements of the iterable
    for elem in iterable:

        # Check if the element is an iterable
        if isinstance(elem, Iterable) and not isinstance(elem, (str, bytes)):

            # Recursively flatten the element
            yield from flatten(elem)

        else:

            # Return the element
            yield elem

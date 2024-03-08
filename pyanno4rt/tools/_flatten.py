"""Iterable flattening."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from collections.abc import Iterable

# %% Function definition


def flatten(elements):
    """
    Convert a nested iterable to a flat one.

    Parameters
    ----------
    elements : iterable
        (Nested) iterable to be flattened.

    Returns
    -------
    generator
        Generator object with the flattened iterable values.
    """

    # Loop over the elements of the iterable
    for element in elements:

        # Check if the element is an iterable again
        if (isinstance(element, Iterable)
                and not isinstance(element, (str, bytes))):

            # Recursively flatten the element
            yield from flatten(element)

        else:

            # Return the element
            yield element

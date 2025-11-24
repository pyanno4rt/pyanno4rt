"""Iterable object wrapper."""

# Author: Tim Ortkamp

# %% Function definition


def wrap(value, dtype='tuple'):
    """
    Wrap an object with an iterable.

    Parameters
    ----------
    value : arbitrary
        Value to be wrapped in an iterable.

    dtype : {'tuple', 'list'}, default='tuple'
        Iterable data type.

    Returns
    -------
    tuple or list
        Iterable with the object.
    """

    # Map the iterable types
    types = {'tuple': tuple, 'list': list}

    try:

        # Return the wrapped object
        return types[dtype](value)

    except TypeError:

        # Return the wrapped object
        return types[dtype]([value])

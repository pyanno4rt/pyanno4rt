"""NaN replacement."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from math import isnan

# %% Function definition


def replace_nan(elements, value):
    """
    Replace NaN in an iterable by a specific value.

    Parameters
    ----------
    elements : iterable
        Iterable over which to loop.

    value : arbitrary
        Value by which to replace NaNs.

    Returns
    -------
    generator
        Generator with the replaced elements.
    """

    return (value if isnan(element) else element for element in elements)

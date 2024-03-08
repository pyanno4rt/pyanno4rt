"""Numpy's arange with endpoint inclusion."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from numpy import arange, concatenate

# %% Function definition


def arange_with_endpoint(start, stop, step):
    """
    Return evenly spaced values within an interval, including the endpoint.

    Parameters
    ----------
    start : int or float
        Starting point of the interval.

    stop : int or float
        Stopping point of the interval.

    step : int or float
        Spacing between points in the interval.

    Returns
    -------
    ndarray
        Array of evenly spaced values.
    """

    # Run the common arange function
    array = arange(start, stop, step)

    # Check if the final step leads to the stop point
    if array[-1] + step == stop:

        # Add the endpoint to the array
        array = concatenate([array, [stop]])

    return array

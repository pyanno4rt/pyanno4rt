"""Elementwise function application."""

# Author: Tim Ortkamp

# %% Function definition


def apply(function, iterable, returns=False):
    """
    Apply a function to each element of an iterable.

    Parameters
    ----------
    function : function
        Function to be applied.

    iterable : iterable
        Iterable over which to loop.

    returns : bool, default=False
        Indicator for the output return.
    """

    # Check if an output should be returned
    if returns:

        # Return a list of output values
        return [function(element) for element in iterable]

    # Loop over the elements of the iterable
    for element in iterable:

        # Call the function
        function(element)

    return None

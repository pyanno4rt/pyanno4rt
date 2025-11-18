"""Elementwise function application."""

# Author: Tim Ortkamp

# %% Function definition


def apply(function, elements, returns=False):
    """
    Apply a function to each element of an iterable.

    Parameters
    ----------
    function : function
        Function to be applied.

    elements : iterable
        Iterable over which to loop.

    returns : bool
        Indicator for the output return.
    """

    # Check if an output should be returned
    if returns:

        # Return a list of output values
        return [function(element) for element in elements]

    # Loop over the elements of the iterable
    for element in elements:

        # Call the function
        function(element)

    return None

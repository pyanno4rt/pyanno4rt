"""Elementwise function application."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def apply(function, elements):
    """
    Apply a function to each element of an iterable.

    Parameters
    ----------
    function : function
        Function to be applied.

    elements : iterable
        Iterable over which to loop.
    """

    # Loop over the elements of the iterable
    for element in elements:

        # Call the function on the element
        function(element)

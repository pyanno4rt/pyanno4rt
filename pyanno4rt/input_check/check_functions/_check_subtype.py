"""Subtype checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_subtype(label, data, types):
    """
    Check if any element type in a list or tuple is invalid.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : list or tuple
        List or tuple with the element types to be checked.

    types : type or tuple
        Single type or tuple with the allowed element types.

    Raises
    ------
    TypeError
        If one or more elements of the data have an invalid type.
    """

    # Check if the data is a list or tuple with invalid element types
    if (isinstance(data, (list, tuple))
            and not all(isinstance(element, types) for element in data)):

        # Raise an error to indicate one or more elements with invalid type
        raise TypeError(
            f"One or more elements of the treatment plan parameter '{label}' "
            "have an invalid data type!")

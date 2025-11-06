"""Subtype validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_subtype(label, data, options):
    """
    Validate if all subtypes are supported.

    Parameters
    ----------
    label : str
        Label for the item to be validated.

    data : list or tuple
        Input value to be validated.

    options : type or tuple
        Type or tuple with the type options.

    Raises
    ------
    TypeError
        If any subtype is unsupported.
    """

    # Check if the value is a list or tuple with unsupported subtypes
    if (isinstance(data, (list, tuple))
            and not all(isinstance(element, options) for element in data)):

        # Raise an error
        raise TypeError(
            f"One or more elements of the treatment plan parameter '{label}' "
            "have an unsupported data type!")

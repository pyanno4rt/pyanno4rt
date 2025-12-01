"""Subtype validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_subtype(label, item, options):
    """
    Validation function for the item subtypes.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : list or tuple
        Validation item.

    options : type or tuple
        Type or tuple with the valid options.

    Raises
    ------
    TypeError
        If any item subtype is invalid.
    """

    # Check if the item is an iterable with invalid subtypes
    if (isinstance(item, (list, tuple))
            and not all(isinstance(element, options) for element in item)):

        # Raise an error
        raise TypeError(
            f"One or more elements of the treatment plan parameter '{label}' "
            "have an unsupported data type!")

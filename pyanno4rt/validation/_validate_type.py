"""Type validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_type(label, item, options, condition=None):
    """
    Validation function for the item type.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item
        Validation item.

    options : tuple or dict
        Tuple or dictionary with the valid options.

    condition : None or str, default=None
        Filter condition (key) on the options (dictionary).

    Raises
    ------
    TypeError
        If the item type is invalid.
    """

    # Check if no condition applies
    if condition is None:

        # Check if an invalid type has been passed
        if not isinstance(item, options):

            # Raise an error
            raise TypeError(
                f"The treatment plan parameter '{label}' has data type "
                f"{type(item)}, but should be from {options}!")

    else:

        # Check if an invalid type has been passed
        if not isinstance(item, options[condition]):

            # Raise an error
            raise TypeError(
                f"The treatment plan parameter '{label}' has data type "
                f"{type(item)}, but should be from {options[condition]}!")

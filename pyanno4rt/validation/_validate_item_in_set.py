"""Set membership validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_item_in_set(label, item, options, condition=None):
    """
    Validation function for the set membership of an item.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : str or list
        Validation item.

    options : tuple or dict
        Tuple or dictionary with the valid options.

    condition : None or str, default=None
        Filter condition (key) on the options (dictionary).

    Raises
    ------
    ValueError
        If the item has no set membership.
    """

    # Check if no condition applies
    if condition is None:

        # Check if an invalid string has been passed
        if isinstance(item, str) and item not in options:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' has the value "
                f"'{item}', but should be from the set {set(options)}!")

        # Check if a list with invalid elements has been passed
        if isinstance(item, list) and any(
                element not in options for element in item):

            # Raise an error
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set {set(options)}!")

    else:

        # Check if an invalid string has been passed
        if isinstance(item, str) and item not in options[condition]:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' has the value "
                f"'{item}', but should be from the set "
                f"{set(options[condition])} for {condition}!")

        # Check if a list with invalid elements has been passed
        if isinstance(item, list) and any(
                element not in options[condition] for element in item):

            # Raise an error
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set {set(options[condition])} "
                f"for {condition}!")

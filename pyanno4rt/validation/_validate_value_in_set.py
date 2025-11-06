"""Value-in-set validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_value_in_set(label, data, options, value_condition=None):
    """
    Validate if a value is included in a set.

    Parameters
    ----------
    label : str
        Label for the item to be validated.

    data : str or list
        Input value to be validated.

    options : tuple or dict
        Tuple or dictionary with the value options.

    value_condition : None or str, default=None
        Value of the conditional (only used if options is a dictionary).

    Raises
    ------
    ValueError
        If the value is not included in the set.
    """

    # Check if no value condition applies
    if value_condition is None:

        # Check if an unsupported string value has been passed
        if isinstance(data, str) and data not in options:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' has the value "
                f"'{data}', but should be from the set {set(options)}!")

        # Check if a list with unsupported elements has been passed
        if isinstance(data, list) and any(
                element not in options for element in data):

            # Raise an error
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set {set(options)}!")

    else:

        # Check if an unsupported string value has been passed
        if isinstance(data, str) and data not in options[value_condition]:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' has the value "
                f"'{data}', but should be from the set "
                f"{set(options[value_condition])} for {value_condition}!")

        # Check if a list with unsupported elements has been passed
        if isinstance(data, list) and any(
                element not in options[value_condition] for element in data):

            # Raise an error
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set "
                f"{set(options[value_condition])} for {value_condition}!")

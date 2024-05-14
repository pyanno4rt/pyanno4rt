"""Value set checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_value_in_set(label, data, options, value_condition=None):
    """
    Check if a value is not included in a set of options.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str or list
        Input value to be checked.

    options : tuple or dict
        Tuple or dictionary with the value options.

    value_condition : None or str, default=None
        Value of the conditional variable (used as a selector if options is a \
        dictionary).

    Raises
    ------
    ValueError
        If the data has a value not included in the set of options.
    """

    # Check if no condition applies
    if value_condition is None:

        # Check if no list is passed and the value is not supported
        if not isinstance(data, list) and data not in options:

            # Raise an error to indicate an invalid value
            raise ValueError(
                f"The treatment plan parameter '{label}' is '{data}', but "
                f"should be from the set {set(options)}!")

        # Check if a list is passed and one or more elements are not supported
        if (isinstance(data, list)
                and any(element not in options for element in data)):

            # Raise an error to indicate an invalid value
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set {set(options)}!")

    # Check if a condition applies
    if value_condition is not None:

        # Check if no list is passed and the value is not supported
        if not isinstance(data, list) and data not in options[value_condition]:

            # Raise an error to indicate an invalid value
            raise ValueError(
                f"The treatment plan parameter '{label}' is '{data}', but "
                f"should be from the set {set(options[value_condition])} for "
                f"{value_condition}!")

        # Check if a list is passed and one or more elements are not supported
        if (isinstance(data, list) and any(
                element not in options[value_condition] for element in data)):

            # Raise an error to indicate an invalid value
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                f"'{label}' are not in the set "
                "{set(options[value_condition])} for {value_condition}!")

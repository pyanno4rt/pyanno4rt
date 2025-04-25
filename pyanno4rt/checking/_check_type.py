"""Type checking."""

# Author: Tim Ortkamp

# %% Function definition


def check_type(label, data, options, type_condition=None):
    """
    Check if a value type is supported.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data
        Input value to be checked.

    options : tuple or dict
        Tuple or dictionary with the type options.

    type_condition : None or str, default=None
        Value of the conditional (only used if types is a dictionary).

    Raises
    ------
    TypeError
        If the value type is unsupported.
    """

    # Check if no type condition applies
    if type_condition is None:

        # Check if an unsupported type has been passed
        if not isinstance(data, options):

            # Raise an error
            raise TypeError(
                f"The treatment plan parameter '{label}' has data type "
                f"{type(data)}, but should be from {options}!")

    else:

        # Check if an unsupported type has been passed
        if not isinstance(data, options[type_condition]):

            # Raise an error
            raise TypeError(
                f"The treatment plan parameter '{label}' has data type "
                f"{type(data)}, but should be from {options[type_condition]}!")

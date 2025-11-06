"""String-to-integer validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_string_is_number(label, data):
    """
    Validate if a string value can be converted to an integer.

    Parameters
    ----------
    label : str
        Label for the item to be validated.

    data : str
        Input value to be validated.

    Raises
    ------
    ValueError
        If the string value has an unsupported literal for integer conversion.
    """

    # Check if the value is a string
    if isinstance(data, str):

        try:

            # Convert the value to integer
            int(data)

        except ValueError as error:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' is a string, but has "
                "an unsupported literal for int() with base 10!") from error

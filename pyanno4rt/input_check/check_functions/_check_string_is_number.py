"""String number checking."""

# Author: Tim Ortkamp

# %% Function definition


def check_string_is_number(label, data):
    """
    Check if a string can be converted to a number.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        String to be converted to integer.

    Raises
    ------
    ValueError
        If the string has an invalid literal for integer conversion.
    """

    # Check if the data is string
    if isinstance(data, str):

        try:

            # Convert the data to integer
            int(data)

        except ValueError:

            # Raise an error to indicate an invalid literal
            raise ValueError(
                f"The treatment plan parameter '{label}' is a string, but "
                "it has an invalid literal for int() with base 10!")

"""String-to-numeric conversion."""

# Author: Tim Ortkamp

# %% Function definition


def string_to_numeric(text):
    """
    Convert a string text to a numeric value.

    Parameters
    ----------
    text : str
        String value to be converted into numeric.

    Returns
    -------
    int or float
        Numeric value from the string.
    """

    # Convert the text to float
    string_to_float = float(text)

    # Convert the float to integer
    string_to_int = int(string_to_float)

    # Check if integer and float are equivalent
    if string_to_int == string_to_float:

        # Return the integer
        return string_to_int

    # Else, return the float
    return string_to_float

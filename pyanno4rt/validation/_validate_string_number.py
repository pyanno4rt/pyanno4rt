"""String-to-integer validation."""

# Author: Tim Ortkamp

# %% Function definition


def validate_string_number(label, item):
    """
    Validation function for the string-to-integer conversion.

    Parameters
    ----------
    label : str
        Label for the validation item.

    data : str
        Validation item.

    Raises
    ------
    ValueError
        If the item has an invalid literal for integer conversion.
    """

    # Check if the item is a string
    if isinstance(item, str):

        try:

            # Convert the item
            int(item)

        except ValueError as error:

            # Raise an error
            raise ValueError(
                f"The treatment plan parameter '{label}' is a string, but has "
                "an invalid literal for int() with base 10!") from error

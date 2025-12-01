"""File validation."""

# Author: Tim Ortkamp

# %% External package import

from os.path import isfile, splitext

# %% Function definition


def validate_file(label, item, options):
    """
    Validation function for the regularity of a file path.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : str
        Validation item.

    options : tuple
        Tuple with the valid file formats.

    Raises
    ------
    FileNotFoundError
        If the item references an irregular file.

    TypeError
        If the referenced file has an invalid format.
    """

    # Check if the item references an irregular file
    if not isfile(item) and splitext(item)[1] != '':

        # Raise an error
        raise FileNotFoundError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular file!")

    # Check if the referenced file has an invalid format
    if (isfile(item) and not any(item.endswith(option) for option in options)):

        # Check if the number of valid formats is one
        if len(options) == 1:

            # Get the output string from the single format
            options_string = options[0]

        else:

            # Get the output string by joining all formats
            options_string = " or ".join((
                ", ".join(options[:-1]), options[-1]))

        # Raise an error
        raise TypeError(
            f"The treatment plan parameter '{label}' references a regular "
            f"file, but it does not end with {options_string}!")

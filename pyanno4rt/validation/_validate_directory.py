"""Directory validation."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir
from os.path import isdir, splitext

# %% Function definition


def validate_directory(label, item, options, alt):
    """
    Validation function for the regularity of a directory path.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : str
        Validation item.

    options : tuple
        Tuple with the valid directory file formats.

    alt : tuple
        Tuple with the alternative valid single file formats.

    Raises
    ------
    NotADirectoryError
        If the item references an irregular directory.

    TypeError
        If any file in the directory has an invalid format.
    """

    # Check if the item is no directory and no valid single file
    if not isdir(item) and splitext(item)[1] not in alt:

        # Raise an error
        raise NotADirectoryError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular directory!")

    # Check if any file in a directory has an invalid format
    if isdir(item) and not all(
            any(file.endswith(option) for option in options)
            for file in listdir(item)):

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
            f"directory, but not all files end with {options_string}!")

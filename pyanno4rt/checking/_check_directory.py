"""Directory checking."""

# Author: Tim Ortkamp

# %% External package import

from os import listdir
from os.path import isdir, splitext

# %% Function definition


def check_directory(label, data, options, alt):
    """
    Check if a directory path is regular with supported file formats.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Input value to be checked.

    options : tuple
        Tuple with the supported directory file formats.

    alt : tuple
        Tuple with the alternative supported single file formats.

    Raises
    ------
    NotADirectoryError
        If the path references an irregular directory.

    TypeError
        If a file in the directory has an unsupported format.
    """

    # Check if the value is no directory and no supported single file
    if not isdir(data) and splitext(data)[1] not in alt:

        # Raise an error
        raise NotADirectoryError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular directory!")

    # Check if any file in a directory has an unsupported format
    if isdir(data) and not all(
            any(file.endswith(option) for option in options)
            for file in listdir(data)):

        # Check if the number of supported formats is one
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

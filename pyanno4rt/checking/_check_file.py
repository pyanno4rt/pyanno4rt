"""File checking."""

# Author: Tim Ortkamp

# %% External package import

from os.path import isfile, splitext

# %% Function definition


def check_file(label, data, options):
    """
    Check if a file path is regular with supported file format.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Input value to be checked.

    options : tuple
        Tuple with the supported file formats.

    Raises
    ------
    FileNotFoundError
        If the path references an irregular file.

    TypeError
        If the path has an unsupported format.
    """

    # Check if the path references an irregular file
    if not isfile(data) and splitext(data)[1] != '':

        # Raise an error
        raise FileNotFoundError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular file!")

    # Check if the file has an unsupported format
    if (isfile(data) and not any(data.endswith(option) for option in options)):

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
            f"file, but it does not end with {options_string}!")

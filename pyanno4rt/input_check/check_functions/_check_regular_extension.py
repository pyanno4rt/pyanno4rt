"""File regularity and extension checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import isfile, splitext

# %% Function definition


def check_regular_extension(label, data, extensions):
    """
    Check if a file path is irregular or has an invalid extension.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Path to the file.

    extensions : tuple
        Tuple with the allowed extensions for the file path.

    Raises
    ------
    FileNotFoundError
        If the path references an irregular file.

    TypeError
        If the path has an invalid extension.
    """

    # Check if the path references an irregular file
    if not isfile(data) and splitext(data)[1] != '':

        # Raise an error to indicate an irregular file
        raise FileNotFoundError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular file!")

    # Check if the file has an invalid extension
    if (isfile(data) and not any(
            data.endswith(extension) for extension in extensions)):

        # Check if the number of allowed extensions is one
        if len(extensions) == 1:

            # Get the extension output string from the single element
            extensions_string = extensions[0]

        else:

            # Get the extension output string by joining multiple elements
            extensions_string = " or ".join(
                (", ".join(extensions[:-1]), extensions[-1]))

        # Raise an error to indicate an invalid extension
        raise TypeError(
            f"The treatment plan parameter '{label}' references a regular "
            f"file, but it does not end with {extensions_string}!")

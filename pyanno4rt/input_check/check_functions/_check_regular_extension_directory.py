"""Directory regularity and extension checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os import listdir
from os.path import isdir, splitext

# %% Function definition


def check_regular_extension_directory(label, data, extensions):
    """
    Check if a directory path is irregular or has invalid file extensions.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Path to the file directory.

    extensions : tuple
        Tuple with the allowed extensions for the directory files.

    Raises
    ------
    NotADirectoryError
        If the path references an irregular directory.

    TypeError
        If a file in the directory has an invalid extension.
    """

    # Check if the path references an irregular directory and not a valid file
    if not isdir(data) and splitext(data)[1] not in ('.mat', '.p'):

        # Raise an error to indicate an irregular directory
        raise NotADirectoryError(
            f"The treatment plan parameter '{label}' does not reference a "
            "regular directory!")

    # Check if any file in the directory has an invalid extension
    if isdir(data) and not all(
            (any((file.endswith(extension) for extension in extensions))
             for file in listdir(data))):

        # Check if the number of allowed extensions is one
        if len(extensions) == 1:

            # Get the extension output string from the single element
            extensions_string = extensions[0]

        else:

            # Get the extension output string by joining multiple elements
            extensions_string = " or ".join(
                (", ".join(extensions[:-1]), extensions[-1]))

        # Raise an error to indicate an invalid file extension
        raise TypeError(
            f"The treatment plan parameter '{label}' references a regular "
            f"directory, but not all files end with {extensions_string}!")

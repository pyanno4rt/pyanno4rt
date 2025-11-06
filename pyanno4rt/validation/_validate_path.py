"""Path validation."""

# Author: Tim Ortkamp

# %% External package import

from os.path import isdir, isfile

# %% Function definition


def validate_path(label, data):
    """
    Validate a file or directory path.

    Parameters
    ----------
    label : str
        Label for the item to be validated.

    data : str
        Input value to be validated.

    Raises
    ------
    IOError
        If the path references an invalid file or directory.
    """

    # Check if the path references an invalid file or directory
    if data is not None and not (isdir(data) or isfile(data)):

        # Raise an error
        raise IOError(
            f"The treatment plan parameter '{label}' neither leads to a valid "
            "file nor a directory!")

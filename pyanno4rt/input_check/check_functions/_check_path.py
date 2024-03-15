"""Path checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import isdir, isfile

# %% Function definition


def check_path(label, data):
    """
    Check if a file or directory path is invalid.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Path to the file or directory.

    Raises
    ------
    IOError
        If the path references an invalid file or directory.
    """

    # Check if the path references an invalid file or directory
    if data and not (isdir(data) or isfile(data)):

        # Raise an error to indicate an invalid value
        raise IOError(
            f"The treatment plan parameter '{label}' neither leads to a valid "
            "file nor a directory!")

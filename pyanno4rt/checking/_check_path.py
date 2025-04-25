"""Path checking."""

# Author: Tim Ortkamp

# %% External package import

from os.path import isdir, isfile

# %% Function definition


def check_path(label, data):
    """
    Check if a file or directory path is valid.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : str
        Input value to be checked.

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

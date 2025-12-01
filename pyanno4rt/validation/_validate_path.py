"""Path validation."""

# Author: Tim Ortkamp

# %% External package import

from os.path import isdir, isfile

# %% Function definition


def validate_path(label, item):
    """
    Validation function for a file or directory path.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : str
        Validation item.

    Raises
    ------
    IOError
        If the item references an invalid file or directory.
    """

    # Check if the item references an invalid file or directory
    if item is not None and not (isdir(item) or isfile(item)):

        # Raise an error
        raise IOError(
            f"The treatment plan parameter '{label}' neither leads to a valid "
            "file nor a directory!")

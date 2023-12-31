"""Path checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from os.path import isdir, isfile

# %% Function definition


def check_path(key, value):
    """Check if the file path is not valid."""

    # Check if the value is neither a valid file nor a directory
    if not (isdir(value) or isfile(value)):

        # Raise an error to indicate an invalid value
        raise OSError(
            f"The treatment plan parameter '{key}' neither leads to a valid "
            "file nor a valid directory!")

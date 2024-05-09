"""Python data reading."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from pickle import load

# %% Function definition


def read_data_from_p(path):
    """
    Read the Python data from the path.

    Parameters
    ----------
    path : str
        Path to the Python file.

    Returns
    -------
    dict
        Dictionary with information on the CT slices.

    dict
        Dictionary with information on the segmented structures.
    """

    # Open a file stream
    with open(path, 'rb') as file:

        # Load the Python file
        data = load(file)

    return data[0], data[1]

"""External list loading."""

# Author: Tim Ortkamp

# %% External package import

from json import load as jload
from pickle import load

# %% Function definition


def load_list_from_file(path):
    """
    Load a list of values from a file path.

    Parameters
    ----------
    path : str
        Path to the list file.

    Returns
    -------
    list
        Loaded list of values.
    """

    # Check if a JSON file has been selected
    if path.endswith('.json'):

        # Open a file stream
        with open(path, 'rb') as file:

            # Get the list of values
            return jload(file)

    # Check if a python binary file has been selected
    if path.endswith('.p'):

        # Open a file stream
        with open(path, 'rb') as file:

            # Get the list of values
            return load(file)

    # Check if a text file has been selected
    if path.endswith('.txt'):

        # Open a file stream
        with open(path, 'r', encoding='utf-8') as file:

            # Get the list of values
            return [float(line.rstrip('\n')) for line in file]

    return []

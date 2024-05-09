"""External list loading."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

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
            value_list = jload(file)

    # Else, check if a python binary file has been selected
    elif path.endswith('.p'):

        # Open a file stream
        with open(path, 'rb') as file:

            # Get the list of values
            value_list = load(file)

    # Else, check if a text file has been selected
    elif path.endswith('.txt'):

        # Open a file stream
        with open(path, 'r', encoding='utf-8') as file:

            # Get the list of values
            value_list = [float(line.rstrip('\n')) for line in file]

    return value_list

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
        List of values.
    """

    # Check if a JSON file has been provided
    if path.endswith('.json'):

        # Open a file stream
        with open(path, 'rb') as file:

            # Return the list of values
            return jload(file)

    # Else, check if a python binary file has been provided
    elif path.endswith('.p'):

        # Open a file stream
        with open(path, 'rb') as file:

            # Return the list of values
            return load(file)

    # Else, check if a text file has been provided
    elif path.endswith('.txt'):

        # Open a file stream
        with open(path, 'r', encoding='utf-8') as file:

            # Return the list of values
            return [float(line.rstrip('\n')) for line in file]

    return []

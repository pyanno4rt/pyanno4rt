"""Dictionary filter by key."""

# Author: Tim Ortkamp

# %% Function definition


def filter_dict(dictionary, retain_keys=None, remove_keys=None):
    """
    Filter a dictionary by the keys.

    Parameters
    ----------
    dictionary : dict
        Dictionary with the keys to be filtered.

    retain_keys : list or tuple, default=None
        Names of the keys to be retained from the dictionary.

    remove_keys : list or tuple, default=None
        Names of the keys to be removed from the dictionary.

    Returns
    -------
    dict
        Dictionary with the filtered keys.
    """

    # Check if specific keys should be retained
    if retain_keys is not None:

        # Filter the dictionary
        dictionary = {
            key: value for key, value in dictionary.items()
            if key in retain_keys}

    # Check if specific keys should be removed
    if remove_keys is not None:

        # Filter the dictionary
        dictionary = {
            key: value for key, value in dictionary.items()
            if key not in remove_keys}

    return dictionary

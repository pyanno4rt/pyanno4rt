"""Dictionary key checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_key_in_dict(label, data, keys):
    """
    Check if a key is not featured in a dictionary.

    Parameters
    ----------
    key : str
        Label for the item to be checked.

    data : dict
        Dictionary with the reference keys.

    keys : tuple
        Tuple with the keys to search for in the dictionary.

    Raises
    ------
    KeyError
        If a key is not featured in the dictionary.
    """

    # Check if any key is missing
    if not all(key in data for key in keys):

        # Check if the number of searched keys is one
        if len(keys) == 1:

            # Get the label output string from the single element
            labels_string = keys[0]

        else:

            # Get the label output string by joining multiple elements
            labels_string = " or ".join((", ".join(keys[:-1]), keys[-1]))

        # Raise an error to indicate a missing key
        raise KeyError(
            f"The treatment plan parameter '{label}' is a dictionary, but it "
            f"seems that {labels_string} is missing as a key!")

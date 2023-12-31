"""Dictionary key checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_key_in_dict(key, value, key_choices):
    """Check if the key is included in the value."""
    # Check if any key is missing
    if not all(key in value for key in key_choices):

        # Raise an error to indicate a missing key
        raise AttributeError(
            "The treatment plan parameter '{}' is a dictionary, but it seems "
            "that {} is missing as a key!"
            .format(key,
                    key_choices[0]
                    if len(key_choices) == 1
                    else " or ".join((", ".join(key_choices[:-1]),
                                      key_choices[-1]))))

"""Subtype checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_subtype(key, value, key_type, type_group=(tuple, list)):
    """Check if the subvalue types are not valid."""
    if (isinstance(value, type_group)
            and not all(isinstance(val, key_type) for val in value)):

        # Raise an error to indicate some wrong subvalue type
        raise TypeError(
            "One or more elements of the treatment plan parameter '{}' have "
            "the wrong data type!"
            .format(key))

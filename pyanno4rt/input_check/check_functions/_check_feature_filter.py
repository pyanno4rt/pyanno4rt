"""Feature filter checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_feature_filter(key, value, subfunctions):
    """Check the optimization components."""
    # Check if the first element is not a tuple or a list
    subfunctions[0](key, value[0])

    # Check if any subvalue in the first element is not a string
    subfunctions[1](key, value[0])

    # Check if the second element is not a string
    subfunctions[2](key, value[1])

    # Check if the second element is not 'retain' or 'remove'
    subfunctions[3](key, value[1])

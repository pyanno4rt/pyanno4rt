"""Feature filter checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_feature_filter(key, value, subfunctions):
    """Check the feature filter."""

    # Check if 'features' and 'filter_mode' are no keys
    subfunctions[0](key, value)

    # Check if 'features' is not a list
    subfunctions[1](key, value['features'])

    # Check if any element in 'features' is not a string
    subfunctions[2](key, value['features'])

    # Check if 'retain' or 'remove' are no keys
    subfunctions[3](key, value['filter_mode'])

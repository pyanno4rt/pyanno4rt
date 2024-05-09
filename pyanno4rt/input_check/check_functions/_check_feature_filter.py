"""Feature filter checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_feature_filter(label, data, check_functions):
    """
    Check the feature filter.

    Parameters
    ----------
    label : str
        Label for the item to be checked ('feature_filter').

    data : dict
        Dictionary with the parameters of the feature filter.

    check_functions : tuple
        Tuple with the individual check functions for the dictionary items.
    """

    # Check if 'features' and 'filter_mode' are unavailable keys
    check_functions[0](label, data)

    # Check if 'features' is not a list
    check_functions[1](label, data['features'])

    # Check if any element in 'features' is not a string
    check_functions[2](label, data['features'])

    # Check if 'filter_mode' is not set to 'retain' or 'remove'
    check_functions[3](label, data['filter_mode'])

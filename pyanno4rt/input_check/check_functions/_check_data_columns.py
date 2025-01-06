"""Data columns checking."""

# Author: Tim Ortkamp

# %% Function definition


def check_data_columns(label, data, check_functions):
    """
    Check the data column information.

    Parameters
    ----------
    label : str
        Label for the item to be checked ('data_columns').

    data : dict
        Dictionary with the column information on features and label.

    check_functions : tuple
        Tuple with the individual check functions for the dictionary items.
    """

    # # Check if 'features' and 'filter_mode' are unavailable keys
    # check_functions[0](label, data)

    # # Check if 'features' is not a list
    # check_functions[1](label, data['features'])

    # # Check if any element in 'features' is not a string
    # check_functions[2](label, data['features'])

    # # Check if 'filter_mode' is not set to 'retain' or 'remove'
    # check_functions[3](label, data['filter_mode'])

    pass

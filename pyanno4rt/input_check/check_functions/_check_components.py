"""Optimization components checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_components(label, data, check_functions):
    """
    Check the optimization components.

    Parameters
    ----------
    label : str
        Label for the item to be checked ('components').

    data : dict
        Dictionary with the optimization components.

    check_functions : tuple
        Tuple with the individual check functions for the dictionary items.
    """

    def check_single_component(dict_paths, dict_value):
        """Check a single component."""

        # Check if 'type' and 'instance' are unavailable keys
        check_functions[0](dict_paths[0], dict_value)

        # Check if the component type is neither 'objective' nor 'constraint'
        check_functions[1](dict_paths[1], dict_value['type'])

        # Get the component instance
        instance = dict_value['instance']

        # Check if the instance is neither dictionary nor list
        check_functions[2](dict_paths[2], instance)

        # Check if the instance is a list
        if isinstance(instance, list):

            # Check if any of the instance elements is not a dictionary
            check_functions[6](dict_paths[2], instance)

            # Loop over the elements
            for index, element in enumerate(instance):

                # Get the dictionary paths to check
                paths = tuple(f'{dict_paths[0]}{string}' for string in (
                    f"['instance'][{index}]{extension}"
                    for extension in ('', "['class']", "['parameters']")))

                # Check if 'class' and 'parameters' are unavailable keys
                check_functions[3](paths[0], element)

                # Check if the 'class' key is invalid
                check_functions[4](paths[1], element['class'])

                # Check if the 'parameters' key is not a dictionary
                check_functions[5](paths[2], element['parameters'])

        else:

            # Get the dictionary paths to check
            paths = tuple(f'{dict_paths[0]}{string}' for string in (
                f"['instance']{extension}"
                for extension in ('', "['class']", "['parameters']")))

            # Check if 'class' and 'parameters' are unavailable keys
            check_functions[3](paths[0], instance)

            # Check if the 'class' key is invalid
            check_functions[4](paths[1], instance['class'])

            # Check if the 'parameters' key is not a dictionary
            check_functions[5](paths[2], instance['parameters'])

    # Loop over the dictionary keys
    for dict_key in data:

        # Get the value for the key
        dict_value = data[dict_key]

        # Check if the value is a list
        if isinstance(dict_value, list):

            # Loop over the value list
            for index, element in enumerate(dict_value):

                # Get the dictionary paths to check
                dict_paths = tuple(f'{label}{string}' for string in (
                    f"['{dict_key}'][{index}]{extension}"
                    for extension in ('', "['type']", "['instance']")))

                # Check the component
                check_single_component(dict_paths, element)

        else:

            # Get the dictionary paths to check
            dict_paths = tuple(f'{label}{string}' for string in (
                f"['{dict_key}']{extension}"
                for extension in ('', "['type']", "['instance']")))

            # Check the component
            check_single_component(dict_paths, dict_value)

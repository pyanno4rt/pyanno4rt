"""Optimization components checking."""

# Author: Tim Ortkamp

# %% Function definition


def check_components(label, data, segments, check_functions):
    """
    Check the optimization components.

    Parameters
    ----------
    label : str
        Label for the item to be checked ('components').

    data : dict
        Dictionary with the optimization components.

    segments : dict
        Dictionary with the segment names and types.

    check_functions : tuple
        Tuple with the individual check functions for the dictionary items.
    """

    def check_single_component(paths, value):
        """Check a single component."""

        # Check if 'type' and 'instance' are unavailable keys
        check_functions[1](paths[0], value)

        # Check if the component type is neither 'objective' nor 'constraint'
        check_functions[2](paths[1], value['type'])

        # Get the component instance
        instance = value['instance']

        # Check if the instance is neither dictionary nor list
        check_functions[3](paths[2], instance)

        # Check if the instance is a list
        if isinstance(instance, list):

            # Check if any of the instance elements is not a dictionary
            check_functions[7](paths[2], instance)

            # Loop over the elements
            for index, element in enumerate(instance):

                # Get the dictionary paths to check
                paths = tuple(f'{paths[0]}{string}' for string in (
                    f"['instance'][{index}]{extension}"
                    for extension in ('', "['class']", "['parameters']")))

                # Check if 'class' and 'parameters' are unavailable keys
                check_functions[4](paths[0], element)

                # Check if the 'class' key is invalid
                check_functions[5](paths[1], element['class'])

                # Check if the 'parameters' key is not a dictionary
                check_functions[6](paths[2], element['parameters'])

        else:

            # Get the dictionary paths to check
            paths = tuple(f'{paths[0]}{string}' for string in (
                f"['instance']{extension}"
                for extension in ('', "['class']", "['parameters']")))

            # Check if 'class' and 'parameters' are unavailable keys
            check_functions[4](paths[0], instance)

            # Check if the 'class' key is invalid
            check_functions[5](paths[1], instance['function'])

            # Check if the 'parameters' key is not a dictionary
            check_functions[6](paths[2], instance['parameters'])

    # Loop over the dictionary keys
    for key in data:

        # Check the dictionary key
        check_functions[0](label, key, segments)

        # Get the value for the key
        value = data[key]

        # Check if the value is a list
        if isinstance(value, list):

            # Loop over the value list
            for index, element in enumerate(value):

                # Get the dictionary paths to check
                paths = tuple(f'{label}{string}' for string in (
                    f"['{key}'][{index}]{extension}"
                    for extension in ('', "['type']", "['instance']")))

                # Check the component
                check_single_component(paths, element)

        else:

            # Get the dictionary paths to check
            paths = tuple(f'{label}{string}' for string in (
                f"['{key}']{extension}"
                for extension in ('', "['type']", "['instance']")))

            # Check the component
            check_single_component(paths, value)

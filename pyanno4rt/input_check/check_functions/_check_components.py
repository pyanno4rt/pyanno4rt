"""Optimization components checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_components(key, value, subfunctions):
    """Check the optimization components."""
    # Check if the dictionary is empty
    subfunctions[0](key, value.keys())

    # Loop over the dictionary keys
    for k in value:

        # Get the value for the key
        v = value[k]

        # Construct the component key string
        component_key = ''.join((key, "['{}']".format(k)))

        # Check if the value is not a tuple or a list
        subfunctions[1](component_key, v)

        # Check if the length of the value is not valid
        subfunctions[2](component_key, v)

        # Check if the first element is not supported
        subfunctions[3](''.join((component_key, '[{}]'.format(0))), v[0])

        # Check if the second element is not a dict, tuple or list
        subfunctions[4](''.join((component_key, '[{}]'.format(0))), v[1])

        # Check if the second element is a list
        if isinstance(v[1], list):

            # Check if the elements are dictionaries
            subfunctions[5](''.join((component_key, '[{}]'.format(1))), v[1])

            # Loop over the subdictionaries
            for i, _ in enumerate(v[1]):

                # Check if dictionary keys are missing
                subfunctions[6](
                    ''.join((component_key, '[{}]'.format(1),
                             '[{}]'.format(i))), v[1][i])

                # Check if the 'class' key is not valid
                subfunctions[7](
                    ''.join((component_key, '[{}]'.format(1),
                             '[{}]'.format(i), "['class']")), v[1][i]['class'])

                # Check if the 'parameters' key is not a dictionary
                subfunctions[8](
                    ''.join((component_key, '[{}]'.format(1),
                             '[{}]'.format(i), "['parameters']")),
                    v[1][i]['parameters'])

        else:

            # Check if dictionary keys are missing
            subfunctions[6](''.join((component_key, '[{}]'.format(1))), v[1])

            # Check if the 'class' key is not valid
            subfunctions[7](
                ''.join((component_key, '[{}]'.format(1), "['class']")),
                v[1]['class'])

            # Check if the 'parameters' key is not a dictionary
            subfunctions[8](
                ''.join((component_key, '[{}]'.format(1), "['parameters']")),
                v[1]['parameters'])

"""Optimization components checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_components(key, value, subfunctions):
    """Check the optimization components."""

    # Loop over the dictionary keys
    for k in value:

        # Get the value for the key
        v = value[k]

        # Construct the component key string
        component_key = ''.join((key, "['{}']".format(k)))

        # Check if 'type' and 'instance' are no keys
        subfunctions[0](component_key, v)

        # Check if the component type is not 'objective' or 'constraint'
        subfunctions[1](''.join((component_key, "['type']")), v['type'])

        # Check if the component instance is not a dict or list
        subfunctions[2](''.join((
            component_key, "['instance']")), v['instance'])

        # Check if the second element is a list
        if isinstance(v['instance'], list):

            # Check if the elements are not dictionaries
            subfunctions[6](''.join((component_key, "['instance']")),
                            v['instance'])

            # Loop over the subdictionaries
            for i, _ in enumerate(v['instance']):

                # Check if 'class' and 'parameters' are no keys
                subfunctions[3](''.join((
                    component_key, "['instance']", '[{}]'.format(i))),
                    v['instance'][i])

                # Check if the 'class' key is not valid
                subfunctions[4](''.join((
                    component_key, "['instance']", '[{}]'.format(i),
                    "['class']")), v['instance'][i]['class'])

                # Check if the 'parameters' key is not a dictionary
                subfunctions[5](''.join((
                    component_key, "['instance']", '[{}]'.format(i),
                    "['parameters']")), v['instance'][i]['parameters'])

        else:

            # Check if 'class' and 'parameters' are no keys
            subfunctions[3](''.join((
                component_key, "['instance']")), v['instance'])

            # Check if the 'class' key is not valid
            subfunctions[4](''.join((
                component_key, "['instance']", "['class']")),
                v['instance']['class'])

            # Check if the 'parameters' key is not a dictionary
            subfunctions[5](''.join((
                component_key, "['instance']", "['parameters']")),
                v['instance']['parameters'])

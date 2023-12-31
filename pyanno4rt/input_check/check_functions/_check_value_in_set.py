"""Value set checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_value_in_set(key, value, options, type_group=None, value_group=None):
    """Check if the value is not supported w.r.t the options."""

    # Check if no value group applies
    if value_group is None:

        # Check if no list is passed and the value is not supported
        if not isinstance(value, list) and value not in options:

            # Raise an error to indicate a wrong value
            raise ValueError(
                "The treatment plan parameter '{}' has to be {}, got {}!"
                .format(key,
                        options[0]
                        if len(options) == 1
                        else " or ".join((", ".join(options[:-1]),
                                          options[-1])),
                        value))

        elif (isinstance(value, list)
              and any(val not in options for val in value)):

            # Raise an error to indicate a wrong value
            raise ValueError(
                "One or more elements of the treatment plan parameter '{}' "
                "are not in {}!"
                .format(key,
                        options[0]
                        if len(options) == 1
                        else " or ".join((", ".join(options[:-1]),
                                          options[-1]))))

    # Else, check if a value group applies but the value is not supported
    elif value_group is not None and value not in options[value_group]:

        # Check if no list is passed and the value is not supported
        if not isinstance(value, list) and value not in options[value_group]:

            # Raise an error to indicate a wrong value
            raise ValueError(
                "The treatment plan parameter '{}' has to be {} for {}, got "
                "{}!"
                .format(key,
                        options[value_group][0]
                        if len(options[value_group]) == 1
                        else " or ".join((", ".join(options[value_group][:-1]),
                                          options[value_group][-1])),
                        value_group,
                        value))

        elif (isinstance(value, list)
              and any(val not in options[value_group] for val in value)):

            # Raise an error to indicate a wrong value
            raise ValueError(
                "One or more elements of the treatment plan parameter '{}' "
                "are not in {} for {}!"
                .format(key,
                        options[value_group][0]
                        if len(options[value_group]) == 1
                        else " or ".join((", ".join(options[value_group][:-1]),
                                          options[value_group][-1])),
                        value_group))

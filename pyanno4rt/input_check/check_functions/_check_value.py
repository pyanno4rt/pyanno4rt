"""Value checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_value(key, value, reference, sign, type_group=None,
                value_group=None):
    """Check if the value has an inappropriate value range."""

    # Check if some value is passed
    if value is not None:

        # Check if the 'equal to' sign is applied
        if sign == '==':
            if value_group == 'scalar':
                logical_value = value == reference
            elif value_group == 'vector':
                logical_value = all(val == reference for val in value
                                    if val is not None)

        # Check if the 'greater than' sign is applied
        if sign == '>':
            if value_group == 'scalar':
                logical_value = value > reference
            elif value_group == 'vector':
                logical_value = all(val > reference for val in value
                                    if val is not None)

        # Check if the 'greater equal' sign is applied
        if sign == '>=':
            if value_group == 'scalar':
                logical_value = value >= reference
            elif value_group == 'vector':
                logical_value = all(val >= reference for val in value
                                    if val is not None)

        # Check if the 'smaller than' sign is applied
        if sign == '<':
            if value_group == 'scalar':
                logical_value = value < reference
            elif value_group == 'vector':
                logical_value = all(val < reference for val in value
                                    if val is not None)

        # Check if the 'smaller equal' sign is applied
        if sign == '<=':
            if value_group == 'scalar':
                logical_value = value <= reference
            elif value_group == 'vector':
                logical_value = all(val <= reference for val in value
                                    if val is not None)

        # Check if the scalar group applies and the logical value is False
        if value_group == 'scalar' and not logical_value:

            # Raise an error to indicate an invalid value
            raise ValueError(
                "The treatment plan parameter '{}' must be {} {}, got {}!"
                .format(key,
                        sign,
                        reference,
                        value))

        # Check if the vector group applies and the logical value is False
        elif value_group == 'vector' and not logical_value:

            # Raise an error to indicate an invalid element
            raise ValueError(
                "One or more elements of the treatment plan parameter '{}' "
                "are not {} {}!"
                .format(key,
                        sign,
                        reference))

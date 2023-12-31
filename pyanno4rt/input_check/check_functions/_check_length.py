"""Vector length checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_length(key, value, reference, sign, type_group=(tuple, list)):
    """Check if the vector length is not valid."""

    # Check if some value is passed
    if value is not None:

        # Check if the 'equal to' sign is applied
        if sign == '==':
            logical_value = len(value) == reference

        # Check if the 'greater than' sign is applied
        if sign == '>':
            logical_value = len(value) > reference

        # Check if the 'greater equal' sign is applied
        if sign == '>=':
            logical_value = len(value) >= reference

        # Check if the 'smaller than' sign is applied
        if sign == '<':
            logical_value = len(value) < reference

        # Check if the 'smaller equal' sign is applied
        if sign == '<=':
            logical_value = len(value) <= reference

        # Check if the logical value is False
        if not logical_value:

            # Raise an error to indicate a wrong vector length
            raise ValueError(
                "The treatment plan parameter '{}' has length {}, but should "
                "be {} {}!"
                .format(key,
                        len(value),
                        sign,
                        reference))

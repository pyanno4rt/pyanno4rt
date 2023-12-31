"""Type checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_type(key, value, key_type, type_group=None, value_group=None):
    """Check if the input data type is not valid."""
    # Check if no type group applies and the data type is not valid
    if type_group is None and not isinstance(value, key_type):

        # Raise an error to indicate a wrong data type
        raise TypeError(
            "The treatment plan parameter '{}' has data type {}, but should "
            "be from {}!"
            .format(key,
                    type(value),
                    key_type))

    # Else, check if a type group applies but the data type is not valid
    elif (type_group is not None
          and not isinstance(value, key_type[type_group])):

        # Raise an error to indicate a wrong data type
        raise TypeError(
            "The treatment plan parameter '{}' has data type {}, but should "
            "be from {}!"
            .format(key,
                    type(value),
                    key_type[type_group]))

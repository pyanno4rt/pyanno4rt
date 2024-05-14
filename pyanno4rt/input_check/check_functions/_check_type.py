"""Type checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% Function definition


def check_type(label, data, types, type_condition=None):
    """
    Check if the input data type is invalid.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data
        Input data with arbitrary type to be checked.

    types : tuple or dict
        Tuple or dictionary with the allowed data types.

    type_condition : None or str, default=None
        Value of the conditional variable (used as a selector if types is a \
        dictionary).

    Raises
    ------
    TypeError
        If the input data has an invalid type.
    """

    # Check if no condition applies but the data type is invalid
    if type_condition is None and not isinstance(data, types):

        # Raise an error to indicate an invalid data type
        raise TypeError(
            f"The treatment plan parameter '{label}' has data type "
            f"{type(data)}, but should be from {types}!")

    # Check if a condition applies but the data type is invalid
    if (type_condition is not None
            and not isinstance(data, types[type_condition])):

        # Raise an error to indicate an invalid data type
        raise TypeError(
            f"The treatment plan parameter '{label}' has data type "
            f"{type(data)}, but should be from {types[type_condition]}!")

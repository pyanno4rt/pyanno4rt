"""Vector length checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from operator import eq, ge, gt, le, lt

# %% Function definition


def check_length(label, data, reference, sign):
    """
    Check if the length of a vector-type object is invalid.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : list, tuple or ndarray
        Vector-type object with length property.

    reference : int
        Reference value for the length comparison.

    sign : {'==', '>', '>=', '<', '<='}
        Sign for the length comparison.

    Raises
    ------
    ValueError
        If the vector-type object has an invalid length.
    """

    # Check if an input is passed
    if data:

        # Create the operator dictionary
        operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

        # Check if the length of the vector-type object is invalid
        if not operator_dict[sign](len(data), reference):

            # Raise an error to indicate an invalid vector length
            raise ValueError(
                f"The treatment plan parameter '{label}' has length "
                f"{len(data)}, but should be {sign} {reference}!")

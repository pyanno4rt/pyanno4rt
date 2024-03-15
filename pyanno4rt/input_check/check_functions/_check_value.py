"""Value checking."""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

# %% External package import

from operator import eq, ge, gt, le, lt

# %% Function definition


def check_value(label, data, reference, sign, is_vector=False):
    """
    Check if the data has an invalid value range.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : int, float, None, list or tuple
        Scalar or vector input to be checked.

    reference : int or float
        Reference for the value comparison.

    sign : {'==', '>', '>=', '<', '<='}
        Sign for the value comparison.

    is_vector : bool, default=False
        Indicator for the vector property of the data.

    Raises
    ------
    ValueError
        If the data has an invalid value range.
    """

    # Check if an input is passed
    if data:

        # Create the operator dictionary
        operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

        # Check if the data is a scalar with invalid value
        if not is_vector and not operator_dict[sign](data, reference):

            # Raise an error to indicate an invalid value
            raise ValueError(
                f"The treatment plan parameter '{label}' must be {sign} "
                f"{reference}, got {data}!")

        # Check if the data is a vector with one or more invalid values
        if is_vector and not all(operator_dict[sign](element, reference)
                                 for element in data if element):

            # Raise an error to indicate an invalid element
            raise ValueError(
                "One or more elements of the treatment plan parameter "
                "'{label}' are not {sign} {reference}!")

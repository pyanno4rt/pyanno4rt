"""Value checking."""

# Author: Tim Ortkamp

# %% External package import

from operator import eq, ge, gt, le, lt

# %% Function definition


def check_value(label, data, reference, sign):
    """
    Check if a (numeric) value is supported.

    Parameters
    ----------
    label : str
        Label for the item to be checked.

    data : None, int, float, list or tuple
        Scalar or vector input value to be checked.

    reference : int or float
        Reference for the value comparison.

    sign : {'==', '>', '>=', '<', '<='}
        Sign for the value comparison.

    Raises
    ------
    ValueError
        If the (numeric) value is unsupported.
    """

    # Map the operators
    operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

    # Check if a scalar with unsupported value has been passed
    if isinstance(data, (int, float)) and not operator_dict[sign](
            data, reference):

        # Raise an error
        raise ValueError(
            f"The treatment plan parameter '{label}' must be {sign} "
            f"{reference}, got {data}!")

    # Check if a vector with unsupported elements has been passed
    if isinstance(data, (list, tuple)) and not all(
            operator_dict[sign](element, reference)
            for element in data if isinstance(element, (int, float))):

        # Raise an error
        raise ValueError(
            "One or more elements of the treatment plan parameter "
            f"'{label}' are not {sign} {reference}!")

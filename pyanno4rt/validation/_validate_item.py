"""(Numeric) item validation."""

# Author: Tim Ortkamp

# %% External package import

from operator import eq, ge, gt, le, lt

# %% Function definition


def validate_item(label, item, reference, sign):
    """
    Validation function for a (numeric) item.

    Parameters
    ----------
    label : str
        Label for the validation item.

    item : None, int, float, list or tuple
        Validation item.

    reference : int or float
        Reference value.

    sign : {'==', '>', '>=', '<', '<='}
        Symbol for the comparison sign.

    Raises
    ------
    ValueError
        If the item is invalid.
    """

    # Map the operators
    operator_dict = {'==': eq, '>=': ge, '>': gt, '<=': le, '<': lt}

    # Check if an invalid scalar has been passed
    if isinstance(item, (int, float)) and not operator_dict[sign](
            item, reference):

        # Raise an error
        raise ValueError(
            f"The treatment plan parameter '{label}' must be {sign} "
            f"{reference}, got {item}!")

    # Check if an iterable with invalid elements has been passed
    if isinstance(item, (list, tuple)) and not all(
            operator_dict[sign](element, reference)
            for element in item if isinstance(element, (int, float))):

        # Raise an error
        raise ValueError(
            "One or more elements of the treatment plan parameter "
            f"'{label}' are not {sign} {reference}!")
